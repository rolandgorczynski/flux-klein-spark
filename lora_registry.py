"""LoRA catalog + per-session custom LoRA loader.

Two layers of state:

* LORA_STYLES (module-level constant) — the curated catalog, identical for
  every visitor.
* dynamic_loras (gr.State, per session) — a session-private dict of any
  custom HuggingFace LoRAs the user has added via the UI. Other sessions
  never see them.

Adapter loading itself is tracked in LOADED_ADAPTERS, which is a *legitimate*
module-level set: it mirrors what's been attached to the shared pipe object
in generation.py so we don't re-download/re-attach the same weights twice.
"""
from __future__ import annotations

import uuid

import gradio as gr

from config import MAX_LORA_SLOTS


FACE_SWAP_PROMPT = """head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. Remove the head from Picture 1 completely and replace it with the head from Picture 2.
FROM PICTURE 1 (strictly preserve):
- Scene: lighting conditions, shadows, highlights, color temperature, environment, background
- Head positioning: exact rotation angle, tilt, direction the head is facing
- Expression: facial expression, micro-expressions, eye gaze direction, mouth position, emotion
FROM PICTURE 2 (strictly preserve identity):
- Facial structure: face shape, bone structure, jawline, chin
- All facial features: eye color, eye shape, nose structure, lip shape and fullness, eyebrows
- Hair: color, style, texture, hairline
- Skin: texture, tone, complexion
The replaced head must seamlessly match Picture 1's lighting and expression while maintaining the complete identity from Picture 2. High quality, photorealistic, sharp details, 4k."""

# NOTE: MAX_LORA_SLOTS is imported from config.py — do NOT re-declare it here.

LORA_STYLES = [
    {
        "title": "None",
        "adapter_name": None,
        "repo": None,
        "weights": None,
        "default_prompt": None,
        "default_weight": 1.0,
    },
    {
        "title": "Klein-Delight-Style",
        "adapter_name": "klein-delight",
        "repo": "linoyts/Flux2-Klein-Delight-LoRA",
        "weights": "pytorch_lora_weights.safetensors",
        "default_prompt": "Relight the image to remove all existing lighting conditions and replace them with neutral, uniform illumination. Apply soft, evenly distributed lighting with no directional shadows, no harsh highlights, and no dramatic contrast. Maintain the original identity of all subjects exactly—preserve facial structure, skin tone, proportions, expressions, hair, clothing, and textures. Do not alter pose, camera angle, background geometry, or image composition. Lighting should appear balanced, and studio-neutral, similar to diffuse overcast or a soft lightbox setup. Ensure consistent exposure across the entire image with realistic depth and subtle shading only where necessary for form.",
        "default_weight": 1.0,
    },
    {
        "title": "Klein-Consistency",
        "adapter_name": "klein-consistency",
        "repo": "dx8152/Flux2-Klein-9B-Consistency",
        "weights": "Klein-consistency.safetensors",
        "default_prompt": None,
        "default_weight": 0.3,
    },
    {
        "title": "Instapic",
        "adapter_name": "Instapic",
        "repo": "Alex995647/loras-flux2-klein-9b",
        "weights": "instapic-ultrareal-the-authentic-social-media-lora-realism/V3_flux_klein.safetensors",
        "default_prompt": "instapic, hard flash, camera flash, smartphone photo, candid, shot on iphone, high detail skin",
        "default_weight": 1.0,
    },
    {
        "title": "Upscaler",
        "adapter_name": "Upscaler",
        "repo": "Alex995647/loras-flux2-klein-9b",
        "weights": "detail-reconstruction-upscaler-workflow-klein-9b-turbo/(FK)r12_s700.safetensors",
        "default_prompt": "zqvx, sharpen the image, enrich the original textures with more detail",
        "default_weight": 1.0,
    },
    {
        "title": "Best-Face-Swap",
        "adapter_name": "face-swap",
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "weights": "bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors",
        "default_prompt": FACE_SWAP_PROMPT,
        "default_weight": 1.0,
    },
    {
        "title": "NSFW v2",
        "adapter_name": "nsfw-v2",
        "repo": "diroverflo/FLux_Klein_9B_NSFW",
        "weights": "Flux Klein - NSFW v2.safetensors",
        "default_prompt": None,
        "default_weight": 1.0,
    },
    {
        "title": "KLEIN Unchained",
        "adapter_name": "Unchained",
        "repo": "Keltezaa/KLEIN_Unchained_V2",
        "weights": "KLEIN-Unchained-V2.safetensors",
        "default_prompt": None,
        "default_weight": 1.0,
    },
    {
        "title": "Flatchested",
        "adapter_name": "Flatchested",
        "repo": "Alex995647/loras-flux2-klein-9b",
        "weights": "flat-chested/Flux2-Klein-9b-FlatChested-v1.safetensors",
        "default_prompt": None,
        "default_weight": 1.0,
    },
    {
        "title": "Controllight",
        "adapter_name": "Controllight",
        "repo": "ControlLight/ControlLight",
        "weights": "controllight.safetensors",
        "default_prompt": None,
        "default_weight": 1.0,
    },
    {
        "title": "RefControl - Depth",
        "adapter_name": "RefConDep",
        "repo": "thedeoxen/refcontrol-FLUX.2-klein-9B-reference-depth-lora",
        "weights": "flux2_klein_9b_refcontrol_depth.safetensors",
        "default_prompt": "refcontrol",
        "default_weight": 1.0,
    },
    {
        "title": "RefControl - Pose",
        "adapter_name": "RefConPos",
        "repo": "thedeoxen/refcontrol-FLUX.2-klein-9B-reference-pose-lora",
        "weights": "refcontrol_v2_poses.safetensors",
        "default_prompt": "apply pose from image 1 with reference from image 2",
        "default_weight": 1.0,
    },
    {
        "title": "Character Sheet",
        "adapter_name": "CharacterSheet",
        "repo": "Alissonerdx/CharacterSheet",
        "weights": "QuadView_klein9b_v1.safetensors",
        "default_prompt": "Convert the character in the image to a Character Sheet showing a face close-up, front, side and back full body views",
        "default_weight": 1.0,
    },
    {
        "title": "Coloring book",
        "adapter_name": "Coloring book",
        "repo": "artificialguybr/COLORINGBOOK-REDMOND-FLUXKLEIN9B",
        "weights": "[FLUX.2.Klein]ColoringBook_Redmond.safetensors",
        "default_prompt": "coloring book and ColoringBookAF",
        "default_weight": 1.0,
    },
    {
        "title": "Shitty watercolour",
        "adapter_name": "Shitty watercolour",
        "repo": "artificialguybr/COLORINGBOOK-REDMOND-FLUXKLEIN9B",
        "weights": "shitty-watercolor-style/shitty_watercolor_klein9b.safetensors",
        "default_prompt": "Watercolor image",
        "default_weight": 1.0,
    },
]


# LOADED_ADAPTERS is the only piece of LoRA state that's legitimately global:
# it just tracks which adapter names have been loaded onto the shared `pipe`
# at least once, so we don't re-download/re-attach weights on every call.
# It is NOT used to decide what a user can select — that comes from each
# session's own `dynamic_loras_state` (a gr.State dict), so one user's custom
# HuggingFace LoRA never shows up in another user's checklist.
LOADED_ADAPTERS: set[str] = set()


# ── Catalog lookup helpers ────────────────────────────────────────────────────

def get_all_styles(dynamic_loras):
    all_styles = list(LORA_STYLES)
    for dynamic_lora in (dynamic_loras or {}).values():
        all_styles.append(dynamic_lora)
    return all_styles


def get_selectable_styles(dynamic_loras):
    """Return all styles except 'None' for the checkbox selector."""
    return [s for s in get_all_styles(dynamic_loras) if s["adapter_name"] is not None]


def get_style_by_title(title, dynamic_loras):
    for style in get_all_styles(dynamic_loras):
        if style["title"] == title:
            return style
    return None


def get_style_by_adapter_name(adapter_name, dynamic_loras):
    for style in get_all_styles(dynamic_loras):
        if style["adapter_name"] == adapter_name:
            return style
    return None


# ── LoRA selection → update weight sliders + LoRA prompt display ──────────────

def update_weight_sliders(selected_titles, dynamic_loras):
    """Given a list of selected LoRA titles, return updated visibility/label/value
    for each of the MAX_LORA_SLOTS pre-rendered sliders, plus an update for the
    LoRA prompt display box (visible when any selected LoRA has a default prompt).

    The user's own prompt is never touched here — LoRA prompts are shown
    separately and appended at inference time.
    """
    selected_styles = [
        get_style_by_title(t, dynamic_loras)
        for t in (selected_titles or [])
        if get_style_by_title(t, dynamic_loras)
    ]
    slider_updates = []
    for i in range(MAX_LORA_SLOTS):
        if i < len(selected_styles):
            style = selected_styles[i]
            slider_updates.append(gr.update(
                visible=True,
                label=f"{style['title']} — weight",
                value=style.get("default_weight", 1.0),
            ))
        else:
            slider_updates.append(gr.update(visible=False, value=1.0))

    lora_prompts = [s.get("default_prompt") for s in selected_styles if s.get("default_prompt")]
    if lora_prompts:
        combined = "\n\n".join(lora_prompts)
        lora_prompt_update = gr.update(value=combined, visible=True)
    else:
        lora_prompt_update = gr.update(value="", visible=False)

    return slider_updates + [lora_prompt_update]


# ── Dynamic HF LoRA loader (per-session) ─────────────────────────────────────

def add_custom_lora(repo_id, weight_name, adapter_name, dynamic_loras_state):
    """Adds a custom LoRA to *this session's* state only — other users never see
    it in their selector. The adapter_name attached to the shared pipe is
    still made globally unique (via a short random suffix) since the pipe
    object itself is one shared resource across all sessions.
    """
    dynamic_loras = dict(dynamic_loras_state or {})
    if not repo_id or not repo_id.strip():
        return "Please enter a valid HuggingFace repo ID.", gr.update(), dynamic_loras
    repo_id = repo_id.strip()
    requested_name = adapter_name.strip() if adapter_name and adapter_name.strip() else None
    try:
        from huggingface_hub import model_info
        info = model_info(repo_id)
        actual_weight = weight_name.strip() if weight_name and weight_name.strip() else None
        if not actual_weight:
            for name in ["pytorch_lora_weights.safetensors", "lora.safetensors", "adapter_model.safetensors"]:
                if any(f.filename == name for f in info.siblings):
                    actual_weight = name
                    break
        if not actual_weight:
            available = [f.filename for f in info.siblings if f.filename.endswith(('.safetensors', '.bin'))]
            return (
                f"No weight found. Available: {', '.join(available) or 'None'}",
                gr.update(),
                dynamic_loras,
            )

        base_name = (
            "".join(c if c.isalnum() or c in "-_" else "_" for c in requested_name)
            if requested_name else "custom"
        )

        static_names = {s["adapter_name"] for s in LORA_STYLES if s["adapter_name"]}
        final_adapter_name = f"{base_name}_{uuid.uuid4().hex[:6]}"
        while final_adapter_name in static_names or final_adapter_name in LOADED_ADAPTERS:
            final_adapter_name = f"{base_name}_{uuid.uuid4().hex[:6]}"

        custom_style = {
            "image": "https://huggingface.co/spaces/prithivMLmods/FLUX.2-Klein-LoRA-Studio/resolve/main/examples/image.webp",
            "title": f"Custom: {base_name}",
            "adapter_name": final_adapter_name,
            "repo": repo_id,
            "weights": actual_weight,
            "default_prompt": None,
            "default_weight": 1.0,
        }
        dynamic_loras[final_adapter_name] = custom_style
        new_choices = [s["title"] for s in get_selectable_styles(dynamic_loras)]
        return f"✅ Added: {base_name} from {repo_id}", gr.update(choices=new_choices), dynamic_loras
    except Exception as e:
        return f"❌ Failed: {str(e)}", gr.update(), dynamic_loras
