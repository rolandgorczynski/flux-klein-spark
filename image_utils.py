"""Pure image helpers — no torch, no diffusers, no gradio state.

Owns: EXIF handling, dimension snapping, canvas fitting, editor-composite
extraction, HEIC decoding, PNG metadata embedding.
"""
from __future__ import annotations

import json
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageColor
from PIL.PngImagePlugin import PngInfo
import gradio as gr

# ── HEIC / HEIF support ──────────────────────────────────────────────────────
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("pillow-heif not installed — HEIC/HEIF uploads will not work. "
          "Add `pillow-heif` to requirements.txt.")


# ── EXIF / dimension helpers ─────────────────────────────────────────────────

def fix_orientation(img: Image.Image | None) -> Image.Image | None:
    if img is None:
        return None
    return ImageOps.exif_transpose(img)


def _snap16(v: float) -> int:
    """Snap to a multiple of 16 — required by FLUX's VAE."""
    return max(16, (int(v) // 16) * 16)


def compute_base_dimensions(image: Image.Image | None) -> tuple[int, int]:
    if image is None:
        return 1024, 1024
    w, h = image.size
    scale = min(1024 / w, 1024 / h)
    return _snap16(w * scale), _snap16(h * scale)


def compute_canvas_dimensions(
    base_image: Image.Image | None,
    canvas_mode: str,
    custom_width: int,
    custom_height: int,
) -> tuple[int, int]:
    if canvas_mode == "Custom":
        return _snap16(custom_width), _snap16(custom_height)
    return compute_base_dimensions(base_image)


# ── Canvas fitting ──────────────────────────────────────────────────────────

def fit_to_canvas(
    img: Image.Image,
    width: int,
    height: int,
    mode: str = "Stretch",
    pad_color: str = "#000000",
) -> Image.Image:
    """Return `img` resized to exactly width×height using the given strategy.

    Modes:
      - "Stretch"        : resize ignoring aspect (current default, may distort)
      - "Pad (color)"    : scale to fit, pad with `pad_color`
      - "Pad (blur)"     : scale to fit, pad with a blurred cover of the image
      - "Crop (cover)"   : scale to cover, center-crop to canvas
    """
    img = img.convert("RGB")

    if mode == "Stretch":
        return img.resize((width, height), Image.LANCZOS)

    iw, ih = img.size

    if mode == "Pad (color)":
        scale = min(width / iw, height / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = img.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (width, height), pad_color)
        canvas.paste(resized, ((width - nw) // 2, (height - nh) // 2))
        return canvas

    if mode == "Pad (blur)":
        # Foreground: scale-to-fit
        scale = min(width / iw, height / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        fg = img.resize((nw, nh), Image.LANCZOS)
        # Background: scale-to-cover, center-crop, then blur heavily
        cscale = max(width / iw, height / ih)
        cw, ch = max(1, int(iw * cscale)), max(1, int(ih * cscale))
        bg = img.resize((cw, ch), Image.LANCZOS)
        bg = bg.crop(((cw - width) // 2, (ch - height) // 2,
                     (cw - width) // 2 + width, (ch - height) // 2 + height))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=32))
        bg.paste(fg, ((width - nw) // 2, (height - nh) // 2))
        return bg

    if mode == "Crop (cover)":
        cscale = max(width / iw, height / ih)
        nw, nh = max(1, int(iw * cscale)), max(1, int(ih * cscale))
        resized = img.resize((nw, nh), Image.LANCZOS)
        left = (nw - width) // 2
        top = (nh - height) // 2
        return resized.crop((left, top, left + width, top + height))

    # Unknown mode → fall back to stretch rather than erroring during inference
    print(f"[fit_to_canvas] unknown mode {mode!r} — falling back to Stretch.")
    return img.resize((width, height), Image.LANCZOS)


# ── UI label updates ────────────────────────────────────────────────────────

def on_base_image_change(img) -> str:
    if img is None:
        return "*No base image uploaded yet*"
    try:
        pil_img = img if isinstance(img, Image.Image) else Image.open(img)
        ow, oh = pil_img.size
        bw, bh = compute_base_dimensions(pil_img)
        return (
            f"Input: **{ow} × {oh}** px → "
            f"Auto canvas (pre-upscale): **{bw} × {bh}** px"
        )
    except Exception as e:
        return f"*Could not read dimensions: {e}*"


def on_reference_change(images) -> str:
    if not images:
        return "📷 No reference images"
    count = len(images)
    return f"📷 {count} reference image{'s' if count != 1 else ''} uploaded"


# ── Upload round-trip (fixes HEIC preview in main tab) ──────────────────────

def reencode_upload(img):
    if img is None:
        return None
    if not isinstance(img, Image.Image):
        try:
            img = Image.open(img)
        except Exception:
            return img
    return fix_orientation(img).convert("RGB")


# ── Inference input assembly ────────────────────────────────────────────────

def process_images(base_image, reference_images) -> list[Image.Image]:
    pil_images: list[Image.Image] = []
    if base_image is not None:
        try:
            img = base_image if isinstance(base_image, Image.Image) else Image.open(base_image)
            pil_images.append(fix_orientation(img).convert("RGB"))
        except Exception as e:
            print(f"Skipping invalid base image: {e}")
    for item in (reference_images or []):
        try:
            path_or_img = item[0] if isinstance(item, (tuple, list)) else item
            if isinstance(path_or_img, Image.Image):
                img = path_or_img
            elif isinstance(path_or_img, str):
                img = Image.open(path_or_img)
            else:
                img = Image.open(path_or_img.name)
            pil_images.append(fix_orientation(img).convert("RGB"))
        except Exception as e:
            print(f"Skipping invalid reference image: {e}")
    return pil_images


# ── ImageEditor helpers ─────────────────────────────────────────────────────

def _editor_value_like_upload(img: Image.Image) -> Image.Image:
    """Return an ImageEditor value that behaves exactly like a manual upload.

    DIAGNOSED root cause of the crop-ignored-after-Send-to-Editor bug:
    a manual upload (and the working HEIC path, load_heic_to_editor) hands the
    editor a BARE PIL image and lets Gradio build its own internal state,
    including the drawing layer that makes a crop-only edit commit to
    composite. Our programmatic sends instead hand-crafted the value dict
    (background/composite/layers). With an empty layers list the crop had
    nothing to composite against; a later attempt to seed a layer manually
    produced a value the editor could not ingest (the image sat behind the
    canvas and the crop tool was dead). The robust fix is to stop hand-crafting
    the dict entirely and do exactly what the working paths do: return the bare
    image and let Gradio initialize the editor itself.
    """
    return img.convert("RGB")


def _editor_composite(editor_value) -> Image.Image:
    """Extract the final edited image from the Gradio ImageEditor value.

    Gradio's gr.ImageEditor (with layers=False) returns a dict with:
      - "background": the original loaded image (never cropped)
      - "composite":  background + all transforms baked in (crop, brush, eraser)
      - "layers":     drawing layer(s)

    We trust "composite" when present. If it is None we refuse to silently
    substitute the uncropped "background" (that was the old bug's symptom).
    """
    if not editor_value:
        raise gr.Error("Upload an image in the editor first.")

    composite = editor_value.get("composite")
    if composite is not None:
        if isinstance(composite, np.ndarray):
            composite = Image.fromarray(composite)
        elif isinstance(composite, str):
            composite = Image.open(composite)
        return composite.convert("RGB")

    # composite is None despite a background present - crop not committed.
    # Fail loudly rather than returning the uncropped background.
    if editor_value.get("background") is not None:
        raise gr.Error(
            "Crop not applied yet. Make any small brush mark (or erase) to "
            "commit the crop, then Send again."
        )

    raise gr.Error("Upload an image in the editor first.")


def send_editor_to_base(editor_value) -> Image.Image:
    composite = fix_orientation(_editor_composite(editor_value))
    gr.Info("Sent to Base Image")
    return composite


def send_editor_to_reference(editor_value, current_gallery) -> list:
    composite = fix_orientation(_editor_composite(editor_value))
    current = list(current_gallery or [])
    current.append(composite)
    gr.Info("Added to Reference Images")
    return current


def load_heic_to_editor(path):
    if not path:
        return gr.update()
    try:
        img = fix_orientation(Image.open(path)).convert("RGB")
    except Exception as e:
        raise gr.Error(f"Could not decode HEIC/HEIF: {e}")
    gr.Info("HEIC loaded into editor.")
    return img


# ── Send output → base / reference (gallery-aware) ──────────────────────────

def _resolve_gallery_path(selected_path, gallery_value):
    """Pick the path the Send-to-* buttons should use.

    Bug we're guarding against: `selected_path` comes from a gr.State that
    persists across generation runs, so after a new batch has replaced the
    gallery it can still hold a stale path from a previous run — or even a
    path from the *first* image of the current batch, because some Gradio
    builds auto-fire .select(index=0) right after the gallery repopulates.

    Rule: only honour the selection if it's actually still one of the paths
    currently in the gallery. Otherwise use the *last* (most recent) item.
    """
    if not gallery_value:
        return None

    current_paths = []
    for item in gallery_value:
        p = item[0] if isinstance(item, (list, tuple)) else item
        current_paths.append(p)

    if selected_path and selected_path in current_paths:
        return selected_path
    return current_paths[-1]


def send_output_to_base(selected_path, gallery_value):
    path = _resolve_gallery_path(selected_path, gallery_value)
    if not path:
        raise gr.Error("Nothing to send — generate an image first.")
    img = Image.open(path).convert("RGB")
    gr.Info("Output sent to Base Image.")
    return img


def send_output_to_reference(selected_path, gallery_value, current_gallery):
    path = _resolve_gallery_path(selected_path, gallery_value)
    if not path:
        raise gr.Error("Nothing to send — generate an image first.")
    img = Image.open(path).convert("RGB")
    current = list(current_gallery or [])
    current.append(img)
    gr.Info("Output added to Reference Images.")
    return current


# ── PNG metadata embedding ──────────────────────────────────────────────────

def _format_parameters_string(meta: dict[str, Any]) -> str:
    prompt = meta.get("prompt", "") or ""
    fields = [
        ("Seed", meta.get("seed")),
        ("Steps", meta.get("steps")),
        ("CFG scale", meta.get("guidance_scale")),
        ("Size", f"{meta.get('width')}x{meta.get('height')}"),
        ("Model", meta.get("model")),
        ("Upscaler", meta.get("upscale_factor")),
        ("Canvas mode", meta.get("canvas_mode")),
        ("Fit mode", meta.get("canvas_fit_mode")),
    ]
    loras = meta.get("loras") or []
    if loras:
        fields.append(("LoRAs", ", ".join(f"{n}:{w:.2f}" for n, w in loras)))
    kv = ", ".join(f"{k}: {v}" for k, v in fields if v not in (None, "", "None"))
    return f"{prompt}\n{kv}".strip()


def build_pnginfo(meta: dict[str, Any]) -> PngInfo:
    """Public so bulk processing can reuse it for in-place saves."""
    info = PngInfo()
    info.add_text("parameters", _format_parameters_string(meta))
    for k in ("prompt", "seed", "steps", "guidance_scale", "width", "height",
              "model", "upscale_factor", "canvas_mode", "canvas_fit_mode",
              "lora_prompt", "custom_prompt"):
        info.add_text(k, str(meta.get(k, "")))
    info.add_text("loras", json.dumps(meta.get("loras") or []))
    return info


def save_with_metadata(image: Image.Image, meta: dict[str, Any],
                       path: str | None = None) -> str:
    """Save `image` as PNG with embedded generation metadata.

    If `path` is given, write there (used by bulk-process to keep predictable
    filenames inside its work directory). Otherwise allocate a temp PNG.
    """
    if path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="flux2_klein_")
        tmp.close()
        path = tmp.name
    image.save(path, format="PNG", pnginfo=build_pnginfo(meta))
    return path

# ── Send arbitrary PIL → base / reference (used by the Depth/Pose tab) ──────

def push_pil_to_base(img):
    if img is None:
        raise gr.Error("Nothing to send — generate it first.")
    gr.Info("Sent to Base Image.")
    return img


def push_pil_to_reference(img, current_gallery):
    if img is None:
        raise gr.Error("Nothing to send — generate it first.")
    current = list(current_gallery or [])
    current.append(img)
    gr.Info("Added to Reference Images.")
    return current

# ── Send base / reference / output → editor (reverse direction) ─────────────

def send_base_to_editor(base_image):
    """Take the current base image and load it into the ImageEditor."""
    if base_image is None:
        raise gr.Error("Nothing to send — upload a base image first.")
    img = base_image if isinstance(base_image, Image.Image) else Image.open(base_image)
    img = fix_orientation(img).convert("RGB")
    gr.Info("Base image sent to Editor.")
    return _editor_value_like_upload(img)


def send_reference_to_editor(reference_images):
    """Take the FIRST reference image and load it into the ImageEditor."""
    if not reference_images:
        raise gr.Error("Nothing to send — add at least one reference image first.")
    item = reference_images[0]
    path = item[0] if isinstance(item, (list, tuple)) else item
    img = Image.open(path).convert("RGB")
    img = fix_orientation(img)
    gr.Info("First reference image sent to Editor.")
    return _editor_value_like_upload(img)


def send_output_to_editor(selected_path, gallery_value):
    """Take the selected (or latest) generated image and load it into the ImageEditor."""
    path = _resolve_gallery_path(selected_path, gallery_value)
    if not path:
        raise gr.Error("Nothing to send — generate an image first.")
    img = Image.open(path).convert("RGB")
    gr.Info("Output sent to Editor.")
    return _editor_value_like_upload(img)

# ── Canvas extension (used by the Crop / Fix tab) ───────────────────────────
# Percentages are relative to the *original* dimensions. Padding is filled
# with a flat colour. Returns a new PIL image; does NOT save to disk — the
# result is loaded straight back into the ImageEditor so the existing
# Send → Base / Send → Reference buttons handle export.

def _parse_fill(color_str: str, mode: str):
    rgba = ImageColor.getcolor(color_str or "#000000", "RGBA")
    if mode == "RGB":
        return rgba[:3]
    if mode == "L":
        r, g, b, _ = rgba
        return int(0.299 * r + 0.587 * g + 0.114 * b)
    return rgba  # RGBA


def compute_extend_padding(w: int, h: int, up, down, left, right) -> tuple[int, int, int, int]:
    """Return (pad_left, pad_top, pad_right, pad_bottom) in pixels for the
    given percentage inputs. Percentages are relative to original W/H so
    Down=100 on 960×960 → 960×1920 with the source at the top."""
    pl = int(round(w * (float(left  or 0) / 100.0)))
    pr = int(round(w * (float(right or 0) / 100.0)))
    pt = int(round(h * (float(up    or 0) / 100.0)))
    pb = int(round(h * (float(down  or 0) / 100.0)))
    return pl, pt, pr, pb


def extend_canvas(img: Image.Image, up, down, left, right, fill: str) -> Image.Image:
    """Extend `img`'s canvas by the given per-side percentages, filling the
    new area with `fill`. Original image mode is preserved so RGBA stays
    RGBA (no transparency loss)."""
    if img is None:
        raise gr.Error("Nothing to extend — upload an image into the editor first.")
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA")
    for name, v in (("Up", up), ("Down", down), ("Left", left), ("Right", right)):
        if v is None or float(v) < 0:
            raise gr.Error(f"'{name} %' must be ≥ 0.")
    w, h = img.size
    pl, pt, pr, pb = compute_extend_padding(w, h, up, down, left, right)
    if pl == pt == pr == pb == 0:
        gr.Info("All percentages are 0 — image unchanged.")
        return img
    new_size = (w + pl + pr, h + pt + pb)
    canvas = Image.new(img.mode, new_size, _parse_fill(fill, img.mode))
    canvas.paste(img, (pl, pt))
    return canvas


def render_extend_schematic(
    img: Image.Image | None, up, down, left, right, fill: str,
    max_dim: int = 320,
) -> tuple[Image.Image | None, str]:
    """Live, to-scale preview of what extend_canvas() will produce, without
    doing the full render. Returns (schematic PIL, info markdown)."""
    if img is None:
        return None, "*Upload something into the editor first.*"
    w, h = img.size
    pl, pt, pr, pb = compute_extend_padding(w, h, up, down, left, right)
    nw, nh = w + pl + pr, h + pt + pb
    scale = min(max_dim / nw, max_dim / nh, 1.0)
    sw, sh = max(1, int(nw * scale)), max(1, int(nh * scale))
    spl, spt = int(pl * scale), int(pt * scale)
    sow, soh = max(1, int(w * scale)), max(1, int(h * scale))

    schem = Image.new("RGB", (sw, sh), _parse_fill(fill, "RGB"))
    d = ImageDraw.Draw(schem)  # ImageDraw already imported at top of module
    d.rectangle([spl, spt, spl + sow - 1, spt + soh - 1],
                fill=(200, 200, 200), outline=(255, 0, 0), width=2)
    info = (f"**Original:** {w} × {h}  \n"
            f"**Padding (L, T, R, B):** {pl}, {pt}, {pr}, {pb}  \n"
            f"**Final canvas:** {nw} × {nh}")
    return schem, info


def extend_editor_canvas(editor_value, up, down, left, right, fill) -> Image.Image:
    """Take the current editor composite, extend it, and return the result so
    it can be loaded straight back into the same ImageEditor. Any crop marks
    the user had placed are baked in before extending (that's what
    `_editor_composite` already does)."""
    composite = fix_orientation(_editor_composite(editor_value))
    out = extend_canvas(composite, up, down, left, right, fill)
    gr.Info(f"Canvas extended to {out.size[0]} × {out.size[1]}.")
    return out