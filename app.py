import os
import gc
import csv
import time
import random
import uuid
import zipfile
import threading

import gradio as gr
from runtime import GPU
import torch
import numpy as np
from PIL import Image

# ── Local modules — single source of truth for each concern ─────────────────
from config import (
    MODEL_VARIANT,
    MODEL_REPO,
    MAX_SEED,
    MAX_LORA_SLOTS,
    ENABLE_LOGGING,
)
from ui_theme import orange_red_theme
from upscale import UPSCALE_MODELS, apply_realesrgan
from lora_registry import (
    LORA_STYLES,
    LOADED_ADAPTERS,
    get_selectable_styles,
    get_style_by_title,
    update_weight_sliders,
    add_custom_lora,
)
from logging_utils import log_inference
from image_utils import (
    fix_orientation,
    compute_canvas_dimensions,
    fit_to_canvas,
    on_base_image_change,
    on_reference_change,
    process_images,
    reencode_upload,
    send_editor_to_base,
    send_editor_to_reference,
    load_heic_to_editor,
    send_output_to_base,
    send_output_to_reference,
    send_base_to_editor,
    send_reference_to_editor,
    send_output_to_editor,
    save_with_metadata,
    build_pnginfo,
    push_pil_to_base,
    push_pil_to_reference,
    extend_editor_canvas,
    render_extend_schematic,
)
from control_tools import (
    generate_depthmap,
    detect_pose,
    render_pose_skeleton,
    render_pose_overlay,
    move_joint,
    hide_joint,
    clear_all_joints,
    default_pose_template,
    person_choices,
    parse_person_idx,
    joint_name_to_index,
    OPENPOSE_KEYPOINT_NAMES,
)
from compare_view import build_comparison_html, build_placeholder_html
from prompt_enhance import enhance_prompt, ENHANCE_STYLES

# ── Model load ──────────────────────────────────────────────────────────────
# Pipeline class depends on MODEL_VARIANT and is the only thing here that
# can't live in config.py (config must stay torch/diffusers-free).
from runtime import LazyPipeline, model_access_status

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = LazyPipeline()


# ── UI helper callbacks ──────────────────────────────────────────────────────

def on_canvas_mode_change(mode):
    """Custom W/H sliders only relevant when mode == Custom."""
    is_custom = (mode == "Custom")
    return gr.update(visible=is_custom), gr.update(visible=is_custom)


def on_fit_mode_change(fit_mode):
    """Pad colour swatch only relevant for Pad (color)."""
    return gr.update(visible=(fit_mode == "Pad (color)"))


def on_batch_vary_change(vary_mode):
    """Sweep range only relevant for the LoRA sweep mode."""
    is_sweep = (vary_mode == "Sweep first LoRA weight")
    return gr.update(visible=is_sweep), gr.update(visible=is_sweep)


def on_gallery_select(evt: gr.SelectData, gallery_value):
    """Remember which gallery item is selected so Send→* uses it."""
    if evt is None or gallery_value is None or evt.index is None:
        return None
    try:
        item = gallery_value[evt.index]
    except (IndexError, TypeError):
        return None
    return item[0] if isinstance(item, (list, tuple)) else item


# ── Logging ──────────────────────────────────────────────────────────────────

def _spawn_log(pil_images, result_image, prompt, seed, steps, guidance_scale,
               width, height, duration, success, error="",
               lora_titles=None, lora_weights=None, upscale_factor="None",
               lora_prompt_text=""):
    """Fire-and-forget logger. ENABLE_LOGGING is read once at import time in
    config.py — there's no per-call env lookup. Used by single, batch and
    bulk inference paths."""
    if not ENABLE_LOGGING:
        return
    threading.Thread(
        target=log_inference,
        args=(pil_images, result_image, prompt, seed, steps, guidance_scale,
              width, height, duration, success, error),
        kwargs={"lora_titles": lora_titles, "lora_weights": lora_weights,
                "upscale_factor": upscale_factor, "lora_prompt_text": lora_prompt_text},
        daemon=True,
    ).start()


# ── Prompt enhancement ──────────────────────────────────────────────────────

def _enhance_prompt_invoke(prompt_text, style, base_image, use_image_context):
    """Call the prompt enhancer. Returns original if enhancement fails."""
    if not prompt_text or not prompt_text.strip():
        gr.Warning("Type a prompt first, then click Enhance.")
        return prompt_text
    try:
        enhanced = enhance_prompt(
            prompt_text,
            base_image=base_image,
            style=style,
            use_image_context=use_image_context,
        )
        if enhanced and enhanced.strip():
            gr.Info("Prompt enhanced!")
            return enhanced.strip()
        gr.Warning("Enhancer returned empty — keeping original.")
        return prompt_text
    except Exception as e:
        gr.Warning(f"Enhancement failed: {e}")
        return prompt_text


# ── GPU step (shared by single, batch, and bulk) ─────────────────────────────

@GPU
def _infer_gpu(
    pil_images, prompt, lora_prompt_text, custom_prompt_text, selected_titles,
    seed, guidance_scale, steps, upscale_factor,
    canvas_mode, custom_width, custom_height,
    canvas_fit_mode, pad_color, dynamic_loras,
    *slider_values, progress=None,
):
    if "Best-Face-Swap" in selected_titles:
        if len(pil_images) < 2:
            raise gr.Error("Face Swap requires 2 images: a Base image and one Reference image.")
        if len(pil_images) > 2:
            gr.Warning("Face Swap uses only the Base image and the first Reference image.")
            pil_images = pil_images[:2]

    active_styles = [get_style_by_title(t, dynamic_loras) for t in selected_titles
                     if get_style_by_title(t, dynamic_loras)
                     and get_style_by_title(t, dynamic_loras)["adapter_name"] is not None]
    weights = list(slider_values[:len(active_styles)])

    if not active_styles:
        if LOADED_ADAPTERS:
            pipe.disable_lora()
    else:
        for style in active_styles:
            an = style["adapter_name"]
            if an not in LOADED_ADAPTERS:
                try:
                    pipe.load_lora_weights(style["repo"], weight_name=style["weights"], adapter_name=an)
                    LOADED_ADAPTERS.add(an)
                except Exception as e:
                    raise gr.Error(f"Failed to load {style['title']}: {e}")
        pipe.enable_lora()
        pipe.set_adapters([s["adapter_name"] for s in active_styles],
                          adapter_weights=[float(w) for w in weights])

    full_prompt = "\n".join(p for p in [
        (prompt or "").strip(),
        (lora_prompt_text or "").strip(),
        (custom_prompt_text or "").strip(),
    ] if p)

    width, height = compute_canvas_dimensions(pil_images[0], canvas_mode, custom_width, custom_height)
    print(f"Generating at: {width}×{height} (canvas={canvas_mode}, fit={canvas_fit_mode})")

    processed = [fit_to_canvas(img, width, height, canvas_fit_mode, pad_color) for img in pil_images]
    image_input = processed if len(processed) > 1 else processed[0]

    try:
        kwargs = dict(image=image_input, prompt=full_prompt,
                      width=width, height=height,
                      num_inference_steps=steps,
                      generator=torch.Generator(device=device).manual_seed(seed))
        if MODEL_VARIANT != "9B-KV":
            kwargs["guidance_scale"] = guidance_scale
        image = pipe(**kwargs).images[0]
    except Exception as e:
        raise gr.Error(f"Inference failed: {e}")

    if upscale_factor and upscale_factor != "None":
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        try:
            image = apply_realesrgan(image, upscale_factor, device)
        except Exception as e:
            gr.Warning(f"Upscaling failed, returning {width}×{height} result: {e}")

    gc.collect(); torch.cuda.empty_cache()
    return image, seed, width, height


def _meta_for(prompt, seed, steps, guidance_scale, width, height, upscale_factor,
              canvas_mode, canvas_fit_mode, lora_titles, lora_weights,
              lora_prompt, custom_prompt, extra=None):
    meta = {
        "prompt": (prompt or "").strip(),
        "seed": seed, "steps": steps,
        "guidance_scale": guidance_scale if MODEL_VARIANT != "9B-KV" else None,
        "width": width, "height": height,
        "model": f"FLUX.2-Klein-{MODEL_VARIANT}",
        "upscale_factor": upscale_factor,
        "canvas_mode": canvas_mode, "canvas_fit_mode": canvas_fit_mode,
        "loras": list(zip(lora_titles or [], lora_weights or [])),
        "lora_prompt": (lora_prompt or "").strip(),
        "custom_prompt": (custom_prompt or "").strip(),
    }
    if extra:
        meta.update(extra)
    return meta


# ── Single / batch infer (generator → streams into gr.Gallery) ───────────────

def infer(
    base_image, reference_images, prompt, lora_prompt_text, custom_prompt_text,
    selected_titles, seed, randomize_seed, guidance_scale, steps, upscale_factor,
    canvas_mode, custom_width, custom_height, canvas_fit_mode, pad_color,
    batch_count, batch_vary, sweep_min, sweep_max,
    dynamic_loras, *slider_values, progress=None,
):
    """Generator. Streams a list of PNG paths into the output gallery, one
    result at a time. Each iteration is its own @GPU call, so a
    ZeroGPU quota wall mid-batch only loses the in-progress item — earlier
    PNGs are already saved to disk and surfaced in the gallery."""
    gc.collect(); torch.cuda.empty_cache()
    if not isinstance(upscale_factor, str) or upscale_factor not in UPSCALE_MODELS:
        upscale_factor = "None"
    if base_image is None:
        raise gr.Error("Please upload a base image.")
    pil_images = process_images(base_image, reference_images)
    if not pil_images:
        raise gr.Error("Could not process uploaded images.")

    selected_titles = selected_titles or []
    batch_count = max(1, int(batch_count))

    base_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    seeds, weight_overrides = [], []
    for i in range(batch_count):
        if batch_vary == "Sequential seed (+1 each)":
            seeds.append((base_seed + i) % (MAX_SEED + 1)); weight_overrides.append(None)
        elif batch_vary == "Sweep first LoRA weight":
            seeds.append(base_seed)
            t = i / max(batch_count - 1, 1)
            weight_overrides.append((0, float(sweep_min) + t * (float(sweep_max) - float(sweep_min))))
        else:  # "Random seed each run" (default)
            seeds.append(random.randint(0, MAX_SEED)); weight_overrides.append(None)

    active_styles = [get_style_by_title(t, dynamic_loras) for t in selected_titles
                     if get_style_by_title(t, dynamic_loras)
                     and get_style_by_title(t, dynamic_loras)["adapter_name"] is not None]
    log_titles = [s["title"] for s in active_styles]

    results = []
    last_seed_text = ""
    for i in range(batch_count):
        sliders = list(slider_values)
        if weight_overrides[i] is not None:
            slot, val = weight_overrides[i]
            if slot < len(sliders):
                sliders[slot] = val

        cur_seed = seeds[i]
        log_weights = [float(w) for w in sliders[:len(active_styles)]]
        t0 = time.perf_counter()
        try:
            image, used_seed, w, h = _infer_gpu(
                pil_images, prompt, lora_prompt_text, custom_prompt_text, selected_titles,
                cur_seed, guidance_scale, steps, upscale_factor,
                canvas_mode, custom_width, custom_height,
                canvas_fit_mode, pad_color, dynamic_loras,
                *sliders, progress=progress,
            )
            meta = _meta_for(prompt, used_seed, steps, guidance_scale, w, h, upscale_factor,
                             canvas_mode, canvas_fit_mode, log_titles, log_weights,
                             lora_prompt_text, custom_prompt_text,
                             extra={"batch_index": i, "batch_total": batch_count,
                                    "batch_vary": batch_vary})
            results.append(save_with_metadata(image, meta))
            last_seed_text = str(used_seed)
            _spawn_log(pil_images, image, prompt, used_seed, steps, guidance_scale,
                       w, h, time.perf_counter() - t0, True,
                       lora_titles=log_titles, lora_weights=log_weights,
                       upscale_factor=upscale_factor, lora_prompt_text=lora_prompt_text or "")
            yield results, last_seed_text
        except Exception as e:
            _spawn_log(pil_images, None, prompt, cur_seed, steps, guidance_scale,
                       0, 0, time.perf_counter() - t0, False, str(e),
                       lora_titles=log_titles, lora_weights=log_weights,
                       upscale_factor=upscale_factor, lora_prompt_text=lora_prompt_text or "")
            yield results, f"Batch {i+1}/{batch_count} failed: {e}"
            # keep going — earlier results are preserved in the gallery


# ── Bulk processing (one input image per iteration) ─────────────────────────

def _new_bulk_workdir() -> str:
    sid = uuid.uuid4().hex[:8]
    path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"bulk_{sid}")
    os.makedirs(path, exist_ok=True)
    return path


def bulk_infer(
    input_files,
    prompt, lora_prompt_text, custom_prompt_text, selected_titles,
    seed, randomize_seed, guidance_scale, steps, upscale_factor,
    canvas_mode, custom_width, custom_height, canvas_fit_mode, pad_color,
    dynamic_loras, *slider_values, progress=None,
):
    """Process each uploaded image as its own GPU call, streaming results
    into the bulk output gallery as soon as they complete. Outputs and a CSV
    manifest are written to /tmp/bulk_<sid>/ with stable filenames, so a
    ZeroGPU quota wall mid-run still leaves earlier outputs grabbable from
    the gallery and from disk.

    Logging: each image is logged individually via _spawn_log on success or
    failure — same code path as single/batch, so bulk shows up in the dataset
    with the same schema, just with extra bulk_index / bulk_total fields
    embedded in the PNG metadata."""
    if not input_files:
        raise gr.Error("Upload at least one image first.")

    work_dir = _new_bulk_workdir()
    manifest_path = os.path.join(work_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "idx", "input_filename", "output_path", "seed",
            "width", "height", "success", "error", "duration_sec",
        ])

    active_styles = [get_style_by_title(t, dynamic_loras) for t in (selected_titles or [])
                     if get_style_by_title(t, dynamic_loras)
                     and get_style_by_title(t, dynamic_loras)["adapter_name"] is not None]
    log_titles = [s["title"] for s in active_styles]
    log_weights = [float(w) for w in slider_values[:len(active_styles)]]

    results, succeeded, failed = [], 0, 0
    total = len(input_files)

    for i, path in enumerate(input_files):
        progress(i / total, desc=f"Image {i+1}/{total}")
        fname = os.path.basename(path) if isinstance(path, str) else f"input_{i}"
        t0 = time.perf_counter()
        try:
            img = fix_orientation(Image.open(path)).convert("RGB")
            cur_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

            image, used_seed, w, h = _infer_gpu(
                [img], prompt, lora_prompt_text, custom_prompt_text, selected_titles or [],
                cur_seed, guidance_scale, steps, upscale_factor,
                canvas_mode, custom_width, custom_height,
                canvas_fit_mode, pad_color, dynamic_loras,
                *slider_values, progress=progress,
            )

            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(work_dir, f"{i:03d}_{stem}.png")
            meta = _meta_for(prompt, used_seed, steps, guidance_scale, w, h, upscale_factor,
                             canvas_mode, canvas_fit_mode, log_titles, log_weights,
                             lora_prompt_text, custom_prompt_text,
                             extra={"bulk_input": fname,
                                    "bulk_index": i, "bulk_total": total})
            image.save(out_path, format="PNG", pnginfo=build_pnginfo(meta))
            results.append(out_path)
            succeeded += 1
            duration = time.perf_counter() - t0

            with open(manifest_path, "a", newline="") as f:
                csv.writer(f).writerow([i, fname, out_path, used_seed, w, h, True, "", f"{duration:.2f}"])

            _spawn_log([img], image, prompt, used_seed, steps, guidance_scale,
                       w, h, duration, True,
                       lora_titles=log_titles, lora_weights=log_weights,
                       upscale_factor=upscale_factor, lora_prompt_text=lora_prompt_text or "")

            status = f"✅ {succeeded}/{total} done ({failed} failed)"
            yield results, status, gr.update()
        except Exception as e:
            failed += 1
            duration = time.perf_counter() - t0
            with open(manifest_path, "a", newline="") as f:
                csv.writer(f).writerow([i, fname, "", "", "", "", False, str(e)[:300], f"{duration:.2f}"])
            _spawn_log([], None, prompt, 0, steps, guidance_scale,
                       0, 0, duration, False, str(e),
                       lora_titles=log_titles, lora_weights=log_weights,
                       upscale_factor=upscale_factor, lora_prompt_text=lora_prompt_text or "")
            yield results, f"⚠️ Image {i+1} failed: {e} | {succeeded} ok, {failed} failed", gr.update()
            continue

    zip_path = os.path.join(work_dir, "outputs.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in results:
            zf.write(p, arcname=os.path.basename(p))
        zf.write(manifest_path, arcname="manifest.csv")

    yield results, f"🎉 Done — {succeeded}/{total} succeeded, {failed} failed", zip_path


# ── Custom prompt manager (session-local) ────────────────────────────────────

def add_custom_prompt(name, text, prompts_state, counter_state):
    prompts = dict(prompts_state); counter = int(counter_state)
    text = text.strip() if text else ""
    name = name.strip() if name else ""
    if not text:
        return "Please enter some prompt text.", prompts, counter, gr.update(), gr.update(), gr.update()
    if not name:
        counter += 1; name = f"Prompt {counter}"
    if name in prompts:
        return f"⚠️ '{name}' already exists.", prompts, counter, gr.update(), gr.update(), gr.update()
    prompts[name] = text
    choices = list(prompts.keys())
    return (f"✅ Saved: '{name}'", prompts, counter,
            gr.update(choices=choices), gr.update(choices=choices),
            gr.update(value="", interactive=True))


def delete_custom_prompt(name, currently_selected, prompts_state):
    prompts = dict(prompts_state)
    msg = f"🗑️ Deleted: '{name}'" if name and name in prompts else "Nothing to delete."
    if name and name in prompts:
        del prompts[name]
    choices = list(prompts.keys())
    new_sel = [n for n in (currently_selected or []) if n in prompts]
    return (msg, prompts,
            gr.update(choices=choices, value=new_sel),
            gr.update(choices=choices, value=None))


def update_custom_prompt_display(selected_names, prompts_state):
    if not selected_names:
        return gr.update(value="", visible=False)
    texts = [prompts_state[n] for n in selected_names if n in prompts_state]
    if texts:
        return gr.update(value="\n\n".join(texts), visible=True)
    return gr.update(value="", visible=False)


# ── UI ───────────────────────────────────────────────────────────────────────

head_html = """
<script>
function copySeedToClipboard() {
    const el = document.querySelector('#used_seed textarea, #used_seed input');
    if (!el) return;
    navigator.clipboard.writeText(el.value).then(() => {
        const btn = document.querySelector('#copy_seed_btn');
        if (btn) { btn.textContent = '✅'; setTimeout(() => { btn.textContent = '📋'; }, 1500); }
    });
}
</script>
"""

css = """
#col-container { margin: 0 auto; max-width: 980px; }
#main-title h1 { font-size: 2.4em !important; }
#reference_gallery .grid-wrap { min-height: 120px }
.lora-weight-row { background: var(--block-background-fill); border-radius: 8px; padding: 4px 12px; margin-bottom: 4px; }
#compare_display { min-height: 520px; }
"""

# Gradio 6.0: theme/css go on launch(), not Blocks()
with gr.Blocks() as demo:
    custom_prompts_state = gr.State({})
    custom_prompt_counter_state = gr.State(0)
    dynamic_loras_state = gr.State({})
    selected_output_state = gr.State(None)
    compare_base_state = gr.State(None)
    compare_gen_state = gr.State(None)

    with gr.Column(elem_id="col-container"):
        _logging_badge = "🟢 On" if ENABLE_LOGGING else "🔴 Off"

        gr.Markdown("# **FLUX.2-Klein-LoRA-Studio**", elem_id="main-title")
        gr.Markdown(
            f"Apply one or more [LoRA](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.2-klein-9B) "
            f"adapters using [FLUX.2-Klein-{MODEL_VARIANT}]({MODEL_REPO}). "
            f"**Model:** `{MODEL_VARIANT}` · **Logging:** {_logging_badge}"
        )

        access_status = gr.Textbox(label="Model readiness", value="Click Check access before the first generation.", interactive=False)
        check_access = gr.Button("Check access")
        check_access.click(model_access_status, outputs=access_status, api_name="model_access")

        with gr.Tabs() as main_tabs:
            # ── Generate tab ─────────────────────────────────────────────────
            with gr.Tab("🎨 Generate", id="tab_generate"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        base_image = gr.Image(
                            label="Base Image", type="pil",
                            sources=["upload", "clipboard"], height=260, elem_id="base_image",
                        )
                        size_info = gr.Markdown("*No image uploaded yet*")

                        reference_images = gr.Gallery(
                            label="Reference Image(s) — optional", type="filepath",
                            columns=2, rows=1, height=140, allow_preview=True,
                            elem_id="reference_gallery",
                        )
                        reference_info = gr.Markdown("📷 No reference images")
                        gr.Markdown("*For Face Swap, the first Reference image is used as the face source.*")

                        prompt = gr.Text(label="Prompt", max_lines=3,
                                         placeholder="Describe the edit, or leave blank for style-only LoRAs")
                        with gr.Row():
                            enhance_style = gr.Dropdown(
                                label="Enhance style", choices=list(ENHANCE_STYLES.keys()),
                                value="FLUX natural (general)", scale=2, interactive=True,
                            )
                            enhance_btn = gr.Button("Enhance", size="sm", scale=1)
                            use_image_context = gr.Checkbox(
                                label="Include image context",
                                value=False, scale=1,
                            )
                        lora_prompt_display = gr.Textbox(label="LoRA Default Prompts (auto-appended)",
                                                         interactive=False, visible=False, lines=3)
                        custom_prompt_display = gr.Textbox(label="Custom Prompts (auto-appended)",
                                                           interactive=False, visible=False, lines=3)
                        run_button = gr.Button("▶ Generate", variant="primary", size="lg")

                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                            guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=10.0,
                                                       step=0.1, value=1.0, visible=MODEL_VARIANT != "9B-KV")
                            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=4, step=1)
                            upscale_factor = gr.Dropdown(label="Upscale model",
                                                         choices=list(UPSCALE_MODELS.keys()), value="None")
                            gr.Markdown("#### 🖼️ Output canvas size")
                            canvas_mode = gr.Radio(
                                choices=["Auto (from base image)", "Custom"],
                                value="Auto (from base image)", label="Canvas mode",
                                info=("Auto matches the base image's aspect ratio (longest side 1024). "
                                      "Use Custom when base and references have very different proportions."),
                            )
                            custom_width = gr.Slider(label="Width", minimum=512, maximum=2048,
                                                     step=16, value=1024, visible=False)
                            custom_height = gr.Slider(label="Height", minimum=512, maximum=2048,
                                                      step=16, value=1024, visible=False)
                            canvas_fit_mode = gr.Radio(
                                choices=["Stretch", "Pad (color)", "Pad (blur)", "Crop (cover)"],
                                value="Stretch", label="Canvas fit mode",
                                info=("How input images are placed into the canvas. "
                                      "Stretch = current default (can squish). "
                                      "Pad keeps aspect; Crop fills by trimming edges."),
                            )
                            pad_color = gr.ColorPicker(label="Pad colour",
                                                       value="#000000", visible=False)

                            gr.Markdown("#### 🔁 Batch")
                            batch_count = gr.Slider(label="Number of runs", minimum=1, maximum=12,
                                                    step=1, value=1)
                            batch_vary = gr.Radio(
                                choices=["Random seed each run",
                                         "Sequential seed (+1 each)",
                                         "Sweep first LoRA weight"],
                                value="Random seed each run", label="Variation strategy",
                                info=("Sweep linearly varies the weight of whichever LoRA is in "
                                      "slot 1 (first ticked) across the runs."),
                            )
                            sweep_min = gr.Slider(label="Sweep min weight", minimum=0.0, maximum=2.0,
                                                  step=0.05, value=0.4, visible=False)
                            sweep_max = gr.Slider(label="Sweep max weight", minimum=0.0, maximum=2.0,
                                                  step=0.05, value=1.4, visible=False)

                    with gr.Column(scale=1):
                        output_gallery = gr.Gallery(
                            label="Output", type="filepath", columns=2, rows=2,
                            height=420, allow_preview=True, preview=True,
                            object_fit="contain", show_label=True,
                        )
                        with gr.Row():
                            used_seed = gr.Textbox(label="🌱 Seed used (last run)",
                                                   interactive=False, max_lines=1, scale=3)
                            copy_seed_btn = gr.Button("📋", size="sm", scale=0, min_width=36,
                                                       elem_id="copy_seed_btn")
                        with gr.Row():
                            send_out_to_base_btn   = gr.Button("Send image to base", size="sm")
                            send_out_to_ref_btn    = gr.Button("Send image to reference", size="sm")
                            send_out_to_editor_btn = gr.Button("Send image to editor", size="sm")
                        send_out_to_comp_btn = gr.Button("Compare generated image with base image", size="sm")
                        with gr.Row():
                            base_to_editor_btn = gr.Button("Send base image to editor", size="sm")
                            ref_to_editor_btn  = gr.Button("Send reference image to editor", size="sm")
                        gr.Markdown(
                            "*Click a gallery thumbnail before Send→ to pick that specific "
                            "image; otherwise the latest is used. PNGs contain seed, prompt, "
                            "LoRAs and settings as `parameters` tEXt chunks (sd-webui / "
                            "Civitai / ComfyUI readable).*"
                        )

                gr.Markdown("### 🎨 Select LoRA(s)")
                lora_selector = gr.CheckboxGroup(
                    choices=[s["title"] for s in get_selectable_styles({})],
                    value=[], label="Active LoRAs — tick one or more",
                )
                gr.Markdown("#### Weights for selected LoRAs")
                weight_sliders = []
                with gr.Group():
                    for i in range(MAX_LORA_SLOTS):
                        with gr.Row(elem_classes="lora-weight-row"):
                            weight_sliders.append(gr.Slider(
                                minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                                label=f"LoRA slot {i+1}", visible=False, interactive=True,
                            ))

                with gr.Accordion("➕ Load Custom LoRA from HuggingFace", open=False):
                    gr.Markdown("Add any FLUX.2-Klein-compatible LoRA. *Added LoRAs are only visible in your own session.*")
                    with gr.Row():
                        lora_repo_id = gr.Textbox(label="Repo ID", placeholder="username/repo-name")
                    with gr.Row():
                        lora_weight_name = gr.Textbox(label="Weight filename (optional)",
                                                      placeholder="pytorch_lora_weights.safetensors")
                        lora_adapter_name = gr.Textbox(label="Adapter name (optional)", placeholder="my-lora")
                    with gr.Row():
                        add_lora_btn = gr.Button("Add LoRA", variant="primary")
                        lora_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("📝 Custom Prompts", open=False):
                    gr.Markdown("Save reusable prompt snippets for this session.")
                    custom_prompt_selector = gr.CheckboxGroup(
                        choices=[], value=[],
                        label="Saved prompts — tick to append to generation",
                    )
                    with gr.Row():
                        prompt_name_input = gr.Textbox(label="Name",
                                                       placeholder="e.g. Skin detail enhancer", scale=1)
                    with gr.Row():
                        prompt_text_input = gr.Textbox(label="Prompt text", lines=4,
                                                       placeholder="Enter the prompt snippet you want to save…")
                    with gr.Row():
                        add_prompt_btn = gr.Button("💾 Save Prompt", variant="primary")
                        prompt_status = gr.Textbox(label="Status", interactive=False, scale=2)
                    with gr.Row():
                        delete_prompt_name = gr.Dropdown(label="Delete a saved prompt",
                                                         choices=[], value=None, interactive=True, scale=2)
                        delete_prompt_btn = gr.Button("🗑️ Delete", variant="secondary", scale=1)

            # ── Crop / Fix Image tab ─────────────────────────────────────────
            with gr.Tab("✂️ Crop / Fix Image", id="tab_editor"):
                gr.Markdown(
                    "Upload an image to crop / paint on it, then send the result to the Base "
                    "Image or add it as a Reference. EXIF orientation is corrected on export."
                )
                editor = gr.ImageEditor(
                    label="Editor", type="pil",
                    transforms=("crop",),
                    brush=gr.Brush(default_size=12,
                                   colors=["#FF4500", "#FFFFFF", "#000000",
                                           "#FF0000", "#00FF00", "#0000FF"],
                                   color_mode="fixed"),
                    eraser=gr.Eraser(default_size=20),
                    layers=False,
                    sources=["upload", "clipboard"],
                    height=420,
                )
                with gr.Row():
                    heic_uploader = gr.File(
                        label="📸 Load HEIC / HEIF (iPhone photos)",
                        file_types=[".heic", ".heif", ".HEIC", ".HEIF"],
                        file_count="single", type="filepath",
                    )
                with gr.Row():
                    send_to_base_btn = gr.Button("→ Send to Base Image", variant="primary")
                    send_to_ref_btn  = gr.Button("→ Add to Reference Images")

                # ── Extend canvas section ────────────────────────────────────────────────
                # Uses the editor's current composite as the source so cropping + painting
                # happen first, then we grow the canvas around the result. Output is
                # loaded back into the same editor — Send → Base / Reference from there.
                with gr.Accordion("📐 Extend canvas (add padding around image)", open=True):
                    gr.Markdown(
                        "Grow the editor image's canvas by a percentage in any combination "
                        "of directions. Percentages are relative to the *current* image "
                        "size — `Down = 100` doubles the height with the image on top. "
                        "The result replaces the editor contents so you can crop again or "
                        "send it to Base / Reference with the buttons above."
                    )
                    with gr.Row():
                        ext_up    = gr.Number(label="Up %",    value=0,   minimum=0, precision=2)
                        ext_down  = gr.Number(label="Down %",  value=0,   minimum=0, precision=2)
                        ext_left  = gr.Number(label="Left %",  value=0,   minimum=0, precision=2)
                        ext_right = gr.Number(label="Right %", value=0,   minimum=0, precision=2)
                    with gr.Row():
                        ext_fill = gr.ColorPicker(label="Fill colour", value="#000000")
                        extend_btn = gr.Button("📐 Extend canvas", variant="primary")
                    with gr.Row():
                        ext_schematic = gr.Image(
                            label="Layout preview (red outline = current image)",
                            type="pil", interactive=False, height=220,
                        )
                        ext_info = gr.Markdown("*Upload something into the editor first.*")

            # ── Bulk processing tab ──────────────────────────────────────────
            with gr.Tab("📦 Bulk Process", id="tab_bulk"):
                gr.Markdown(
                    "Upload many images and process them with the **same settings as the "
                    "Generate tab** (prompt, LoRAs, weights, canvas, upscaler, etc.). "
                    "Outputs stream in one-by-one — each image is its own GPU call, so a "
                    "ZeroGPU quota wall mid-run only loses the in-progress item. "
                    "Earlier outputs stay in the gallery and on disk under `/tmp/bulk_<id>/`."
                )
                bulk_files = gr.File(
                    label="Input images",
                    file_count="multiple", type="filepath",
                    file_types=["image", ".heic", ".heif"],
                )
                with gr.Row():
                    bulk_run_btn  = gr.Button("▶ Start bulk run", variant="primary")
                    bulk_stop_btn = gr.Button("⏹ Stop", variant="stop")
                bulk_status = gr.Markdown("*Ready.*")
                bulk_gallery = gr.Gallery(
                    label="Bulk outputs", type="filepath",
                    columns=4, rows=2, height=480, allow_preview=True, object_fit="contain",
                )
                bulk_zip = gr.File(label="📥 Download all (zip + manifest.csv)",
                                   interactive=False)

            # ── Depth / Pose tab ─────────────────────────────────────────────
            with gr.Tab("🦴 Depth / Pose", id="tab_control"):
                gr.Markdown(
                    "Generate ControlNet-style **depthmaps** and editable **OpenPose** "
                    "skeletons. The result feeds well into the **RefControl – Depth** / "
                    "**RefControl – Pose** LoRAs on the Generate tab when sent as a "
                    "Reference image."
                )

                pose_source_state = gr.State(None)
                pose_keypoints_state = gr.State([])

                with gr.Row():
                    with gr.Column(scale=1):
                        ctrl_source = gr.Image(
                            label="Source image", type="pil",
                            sources=["upload", "clipboard"], height=320,
                        )
                        with gr.Row():
                            detect_depth_btn = gr.Button("🌐 Generate depthmap", variant="primary")
                            detect_pose_btn  = gr.Button("🦴 Detect pose", variant="primary")
                        insert_blank_btn = gr.Button("➕ Insert blank skeleton template")

                    with gr.Column(scale=1):
                        depth_output = gr.Image(label="Depthmap", type="pil",
                                                interactive=False, height=320, format="png")
                        with gr.Row():
                            send_depth_ref_btn  = gr.Button("→ Send depth to Reference",
                                                            variant="primary")
                            send_depth_base_btn = gr.Button("→ Send depth to Base")

                gr.Markdown("### ✏️ Pose editor")
                gr.Markdown(
                    "Pick a person and a joint, then **click anywhere on the editor preview** "
                    "to move that joint. Hidden joints can be re-added the same way — select "
                    "them and click. Use the buttons below for delete / clear / re-detect."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        pose_overlay = gr.Image(
                            label="Editor — click to place active joint",
                            type="pil", interactive=False, height=420, format="png",
                        )
                    with gr.Column(scale=1):
                        pose_clean = gr.Image(
                            label="Skeleton (sent to Reference / Base)",
                            type="pil", interactive=False, height=420, format="png",
                        )

                with gr.Row():
                    active_person_dd = gr.Dropdown(
                        label="Active person", choices=[], value=None, interactive=True,
                    )
                    active_joint_dd = gr.Dropdown(
                        label="Active joint",
                        choices=list(OPENPOSE_KEYPOINT_NAMES),
                        value=None, interactive=True,
                    )

                with gr.Row():
                    delete_joint_btn = gr.Button("🗑️ Hide active joint")
                    reset_pose_btn   = gr.Button("🔄 Re-detect from source")
                    clear_pose_btn   = gr.Button("🧹 Clear all joints")

                with gr.Row():
                    send_pose_ref_btn  = gr.Button("→ Send pose to Reference",
                                                   variant="primary")
                    send_pose_base_btn = gr.Button("→ Send pose to Base")

            # ── Compare tab ───────────────────────────────────────────────────
            with gr.Tab("Compare", id="tab_compare"):
                gr.Markdown(
                    "Drag the **slider** (click and drag anywhere on the image) to compare "
                    "the **base image** (left) with the **generated output** (right). "
                    "Use the **Compare generated image with base image** button on the Generate tab after generating an image."
                )
                compare_display = gr.HTML(
                    value=build_placeholder_html(512, 512),
                    elem_id="compare_display",
                )

    # ── Event wiring ─────────────────────────────────────────────────────────

    base_image.upload(fn=reencode_upload, inputs=[base_image], outputs=[base_image])
    base_image.change(fn=on_base_image_change, inputs=[base_image], outputs=[size_info])
    reference_images.change(fn=on_reference_change, inputs=[reference_images], outputs=[reference_info])

    # update_weight_sliders is the one imported from lora_registry now.
    lora_selector.change(
        fn=update_weight_sliders,
        inputs=[lora_selector, dynamic_loras_state],
        outputs=weight_sliders + [lora_prompt_display],
    )

    # add_custom_lora is also imported from lora_registry.
    add_lora_btn.click(
        fn=add_custom_lora,
        inputs=[lora_repo_id, lora_weight_name, lora_adapter_name, dynamic_loras_state],
        outputs=[lora_status, lora_selector, dynamic_loras_state],
    )

    add_prompt_btn.click(
        fn=add_custom_prompt,
        inputs=[prompt_name_input, prompt_text_input, custom_prompts_state, custom_prompt_counter_state],
        outputs=[prompt_status, custom_prompts_state, custom_prompt_counter_state,
                 custom_prompt_selector, delete_prompt_name, prompt_name_input],
    )
    delete_prompt_btn.click(
        fn=delete_custom_prompt,
        inputs=[delete_prompt_name, custom_prompt_selector, custom_prompts_state],
        outputs=[prompt_status, custom_prompts_state, custom_prompt_selector, delete_prompt_name],
    )
    custom_prompt_selector.change(
        fn=update_custom_prompt_display,
        inputs=[custom_prompt_selector, custom_prompts_state],
        outputs=[custom_prompt_display],
    )

    canvas_mode.change(fn=on_canvas_mode_change, inputs=[canvas_mode],
                       outputs=[custom_width, custom_height])
    canvas_fit_mode.change(fn=on_fit_mode_change, inputs=[canvas_fit_mode],
                           outputs=[pad_color])
    batch_vary.change(fn=on_batch_vary_change, inputs=[batch_vary],
                      outputs=[sweep_min, sweep_max])

    output_gallery.select(fn=on_gallery_select, inputs=[output_gallery],
                          outputs=[selected_output_state])
    # Reset any stale gallery selection before a new run starts, so Send→Base
    # / Send→Ref after this run can't accidentally reuse a path from the
    # previous run's gallery contents.
    run_button.click(fn=lambda: None, outputs=[selected_output_state])

    copy_seed_btn.click(fn=lambda: gr.skip(), js="copySeedToClipboard")
    
    run_event = run_button.click(
        fn=infer,
        inputs=[base_image, reference_images, prompt, lora_prompt_display, custom_prompt_display,
                lora_selector, seed, randomize_seed, guidance_scale, steps, upscale_factor,
                canvas_mode, custom_width, custom_height, canvas_fit_mode, pad_color,
                batch_count, batch_vary, sweep_min, sweep_max,
                dynamic_loras_state] + weight_sliders,
        outputs=[output_gallery, used_seed],
    )

    # ── Enhance prompt ──────────────────────────────────────────────────────
    enhance_btn.click(
        fn=_enhance_prompt_invoke,
        inputs=[prompt, enhance_style, base_image, use_image_context],
        outputs=[prompt],
    )

    # ── Compare tab wiring ───────────────────────────────────────────────────
    def _update_compare_display(path_base, path_gen):
        if not path_base and not path_gen:
            return build_placeholder_html(512, 512)
        return build_comparison_html(path_base, path_gen, 512, 512)

    def _send_to_compare(base_image_path, selected_path, gallery_value, current_base, current_gen):
        # base_image_path comes from the base_image component
        base = base_image_path if base_image_path else current_base
        gen = selected_path
        if not gen and gallery_value:
            try:
                last = gallery_value[-1]
                gen = last[0] if isinstance(last, (list, tuple)) else last
            except (IndexError, TypeError):
                pass
        if not gen:
            raise gr.Error("Generate an image first, then click Send → Compare.")
        return base, gen

    compare_base_state.change(
        fn=_update_compare_display,
        inputs=[compare_base_state, compare_gen_state],
        outputs=[compare_display],
    )
    compare_gen_state.change(
        fn=_update_compare_display,
        inputs=[compare_base_state, compare_gen_state],
        outputs=[compare_display],
    )

    send_out_to_comp_btn.click(
        fn=_send_to_compare,
        inputs=[base_image, selected_output_state, output_gallery, compare_base_state, compare_gen_state],
        outputs=[compare_base_state, compare_gen_state],
    ).then(fn=_update_compare_display,
           inputs=[compare_base_state, compare_gen_state],
           outputs=[compare_display]
    ).then(fn=lambda: gr.Tabs(selected="tab_compare"), outputs=[main_tabs])

        # ── Editor tab wiring ────────────────────────────────────────────────────
    heic_uploader.upload(fn=load_heic_to_editor, inputs=[heic_uploader], outputs=[editor])

    send_to_base_btn.click(fn=send_editor_to_base, inputs=[editor], outputs=[base_image]) \
        .then(fn=on_base_image_change, inputs=[base_image], outputs=[size_info]) \
        .success(fn=lambda: gr.Tabs(selected="tab_generate"), outputs=[main_tabs])

    send_to_ref_btn.click(fn=send_editor_to_reference,
                          inputs=[editor, reference_images], outputs=[reference_images]) \
        .then(fn=on_reference_change, inputs=[reference_images], outputs=[reference_info]) \
        .success(fn=lambda: gr.Tabs(selected="tab_generate"), outputs=[main_tabs])

    send_out_to_base_btn.click(
        fn=send_output_to_base,
        inputs=[selected_output_state, output_gallery],
        outputs=[base_image],
    ).then(fn=on_base_image_change, inputs=[base_image], outputs=[size_info])

    send_out_to_ref_btn.click(
        fn=send_output_to_reference,
        inputs=[selected_output_state, output_gallery, reference_images],
        outputs=[reference_images],
    ).then(fn=on_reference_change, inputs=[reference_images], outputs=[reference_info])

    # ── Send → Editor (reverse direction) ─────────────────────────────────

    send_out_to_editor_btn.click(
        fn=send_output_to_editor,
        inputs=[selected_output_state, output_gallery],
        outputs=[editor],
    ).success(
        fn=lambda: gr.Tabs(selected="tab_editor"), outputs=[main_tabs]
    )

    base_to_editor_btn.click(
        fn=send_base_to_editor,
        inputs=[base_image],
        outputs=[editor],
    ).success(
        fn=lambda: gr.Tabs(selected="tab_editor"), outputs=[main_tabs]
    )

    ref_to_editor_btn.click(
        fn=send_reference_to_editor,
        inputs=[reference_images],
        outputs=[editor],
    ).success(
        fn=lambda: gr.Tabs(selected="tab_editor"), outputs=[main_tabs]
    )
    
    # Extend-canvas wiring
    # The schematic previews the *current editor composite* so users see live
    # feedback as they nudge the percentages / fill colour.
    def _editor_source_for_preview(editor_value):
        if not editor_value or editor_value.get("composite") is None:
            return None
        comp = editor_value["composite"]
        if isinstance(comp, np.ndarray):
            comp = Image.fromarray(comp)
        return comp

    def _update_extend_preview(editor_value, up, down, left, right, fill):
        return render_extend_schematic(
            _editor_source_for_preview(editor_value), up, down, left, right, fill,
        )

    _extend_preview_inputs = [editor, ext_up, ext_down, ext_left, ext_right, ext_fill]
    _extend_preview_outputs = [ext_schematic, ext_info]
    for _c in (ext_up, ext_down, ext_left, ext_right, ext_fill):
        _c.change(fn=_update_extend_preview,
                  inputs=_extend_preview_inputs, outputs=_extend_preview_outputs)
    editor.upload(fn=_update_extend_preview,
                  inputs=_extend_preview_inputs, outputs=_extend_preview_outputs)
    heic_uploader.upload(fn=_update_extend_preview,
                         inputs=_extend_preview_inputs, outputs=_extend_preview_outputs)

# Run: extend, then hand the new PIL back to the editor and refresh the 
# schematic preview so it reflects the new canvas dimensions
    extend_btn.click(
        fn=extend_editor_canvas,
        inputs=[editor, ext_up, ext_down, ext_left, ext_right, ext_fill],
        outputs=[editor],
    ).then(fn=_update_extend_preview,
           inputs=_extend_preview_inputs,
           outputs=_extend_preview_outputs)

    # ── Bulk tab wiring ──────────────────────────────────────────────────────
    bulk_event = bulk_run_btn.click(
        fn=bulk_infer,
        inputs=[bulk_files,
                prompt, lora_prompt_display, custom_prompt_display, lora_selector,
                seed, randomize_seed, guidance_scale, steps, upscale_factor,
                canvas_mode, custom_width, custom_height, canvas_fit_mode, pad_color,
                dynamic_loras_state] + weight_sliders,
        outputs=[bulk_gallery, bulk_status, bulk_zip],
    )

    bulk_stop_btn.click(fn=lambda: gr.Info("Stop requested — finishing current image."),
                        cancels=[bulk_event, run_event])

    # ── Depth / Pose tab wiring ──────────────────────────────────────────────

    ctrl_source.change(
        fn=lambda img: img, inputs=[ctrl_source], outputs=[pose_source_state],
    )

    detect_depth_btn.click(
        fn=generate_depthmap, inputs=[ctrl_source], outputs=[depth_output],
    )

    def _on_detect_pose(source):
        if source is None:
            raise gr.Error("Upload a source image first.")
        poses, w, h = detect_pose(source)
        if not poses:
            gr.Warning("No people detected — try 'Insert blank skeleton template' "
                       "or a different image.")
            return ([], gr.update(choices=[], value=None),
                    gr.update(value=None), None, None)
        return (
            poses,
            gr.update(choices=person_choices(poses), value="Person 1"),
            gr.update(value=OPENPOSE_KEYPOINT_NAMES[0]),
            render_pose_overlay(source, poses, 0, 0),
            render_pose_skeleton(poses, w, h),
        )

    detect_pose_btn.click(
        fn=_on_detect_pose, inputs=[ctrl_source],
        outputs=[pose_keypoints_state, active_person_dd, active_joint_dd,
                 pose_overlay, pose_clean],
    )
    reset_pose_btn.click(
        fn=_on_detect_pose, inputs=[ctrl_source],
        outputs=[pose_keypoints_state, active_person_dd, active_joint_dd,
                 pose_overlay, pose_clean],
    )

    def _on_insert_blank(source):
        if source is None:
            raise gr.Error("Upload a source image first.")
        w, h = source.size
        poses = [default_pose_template(w, h)]
        return (
            poses,
            gr.update(choices=["Person 1"], value="Person 1"),
            gr.update(value=OPENPOSE_KEYPOINT_NAMES[0]),
            render_pose_overlay(source, poses, 0, 0),
            render_pose_skeleton(poses, w, h),
        )

    insert_blank_btn.click(
        fn=_on_insert_blank, inputs=[ctrl_source],
        outputs=[pose_keypoints_state, active_person_dd, active_joint_dd,
                 pose_overlay, pose_clean],
    )

    def _on_overlay_click(evt: gr.SelectData, poses, source, person_label, joint_name):
        if not poses or source is None or evt is None or evt.index is None:
            return gr.update(), gr.update(), gr.update()
        person_idx = parse_person_idx(person_label)
        joint_idx  = joint_name_to_index(joint_name)
        if person_idx is None or joint_idx < 0:
            return gr.update(), gr.update(), gr.update()
        x, y = evt.index
        w, h = source.size
        new_poses = move_joint(poses, person_idx, joint_idx, x, y, w, h)
        return (
            new_poses,
            render_pose_overlay(source, new_poses, person_idx, joint_idx),
            render_pose_skeleton(new_poses, w, h),
        )

    pose_overlay.select(
        fn=_on_overlay_click,
        inputs=[pose_keypoints_state, pose_source_state,
                active_person_dd, active_joint_dd],
        outputs=[pose_keypoints_state, pose_overlay, pose_clean],
    )

    def _on_active_change(poses, source, person_label, joint_name):
        if not poses or source is None:
            return gr.update()
        person_idx = parse_person_idx(person_label) or 0
        joint_idx  = max(joint_name_to_index(joint_name), 0)
        return render_pose_overlay(source, poses, person_idx, joint_idx)

    active_person_dd.change(
        fn=_on_active_change,
        inputs=[pose_keypoints_state, pose_source_state,
                active_person_dd, active_joint_dd],
        outputs=[pose_overlay],
    )
    active_joint_dd.change(
        fn=_on_active_change,
        inputs=[pose_keypoints_state, pose_source_state,
                active_person_dd, active_joint_dd],
        outputs=[pose_overlay],
    )

    def _on_hide_active(poses, source, person_label, joint_name):
        person_idx = parse_person_idx(person_label)
        joint_idx  = joint_name_to_index(joint_name)
        new_poses  = hide_joint(poses, person_idx, joint_idx)
        if source is None:
            return new_poses, gr.update(), gr.update()
        w, h = source.size
        return (new_poses,
                render_pose_overlay(source, new_poses, person_idx, joint_idx),
                render_pose_skeleton(new_poses, w, h))

    delete_joint_btn.click(
        fn=_on_hide_active,
        inputs=[pose_keypoints_state, pose_source_state,
                active_person_dd, active_joint_dd],
        outputs=[pose_keypoints_state, pose_overlay, pose_clean],
    )

    def _on_clear_all(poses, source):
        new_poses = clear_all_joints(poses)
        if source is None:
            return new_poses, gr.update(), gr.update()
        w, h = source.size
        return (new_poses,
                render_pose_overlay(source, new_poses, None, None),
                render_pose_skeleton(new_poses, w, h))

    clear_pose_btn.click(
        fn=_on_clear_all,
        inputs=[pose_keypoints_state, pose_source_state],
        outputs=[pose_keypoints_state, pose_overlay, pose_clean],
    )

    send_depth_ref_btn.click(
        fn=push_pil_to_reference, inputs=[depth_output, reference_images],
        outputs=[reference_images],
    ).then(fn=on_reference_change, inputs=[reference_images], outputs=[reference_info]
    ).then(fn=lambda: gr.Tabs(selected="tab_generate"), outputs=[main_tabs])

    send_depth_base_btn.click(
        fn=push_pil_to_base, inputs=[depth_output], outputs=[base_image],
    ).then(fn=on_base_image_change, inputs=[base_image], outputs=[size_info]
    ).then(fn=lambda: gr.Tabs(selected="tab_generate"), outputs=[main_tabs])

    send_pose_ref_btn.click(
        fn=push_pil_to_reference, inputs=[pose_clean, reference_images],
        outputs=[reference_images],
    ).then(fn=on_reference_change, inputs=[reference_images], outputs=[reference_info]
    ).then(fn=lambda: gr.Tabs(selected="tab_generate"), outputs=[main_tabs])

    send_pose_base_btn.click(
        fn=push_pil_to_base, inputs=[pose_clean], outputs=[base_image],
    ).then(fn=on_base_image_change, inputs=[base_image], outputs=[size_info]
    ).then(fn=lambda: gr.Tabs(selected="tab_generate"), outputs=[main_tabs])


if __name__ == "__main__":
    # Gradio 6.0: theme and css go on launch(), not Blocks()
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7861, css=css, head=head_html, theme=orange_red_theme,
                        mcp_server=False, ssr_mode=False, show_error=True)