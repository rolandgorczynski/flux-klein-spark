"""ESRGAN-family upscaler registry + tiled inference.

Spandrel handles any ESRGAN-family .pth file — to add a model, just append an
entry to UPSCALE_MODELS with its download URL and integer scale factor.
"""
from __future__ import annotations

import gc
import os
import urllib.request

import numpy as np
import torch
import gradio as gr
from PIL import Image


# ── Upscaler model registry ───────────────────────────────────────────────────
# Add new entries here to extend the dropdown — spandrel handles any ESRGAN-
# family .pth file automatically. scale= is the integer output multiplier.
UPSCALE_MODELS = {
    "None": {
        "scale": None, "file": None, "url": None,
    },
    "2× — RealESRGAN (balanced)": {
        "scale": 2,
        "file": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
    "4× — RealESRGAN (balanced)": {
        "scale": 4,
        "file": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "4× — UltraSharp (crisp)": {
        "scale": 4,
        "file": "4x-UltraSharpV2.pth",
        "url": "https://huggingface.co/Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2.pth",
    },
    "4× — Remacri (natural)": {
        "scale": 4,
        "file": "4x_foolhardy_Remacri.pth",
        "url": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    },
    "4× — Nomos2 HQ DAT2 (Photography)": {
        "scale": 4,
        "file": "4xNomos2_hq_dat2.pth",
        "url": "https://github.com/Phhofm/models/releases/download/4xNomos2_hq_dat2/4xNomos2_hq_dat2.pth",
    },
}


def _upscale_tiled(
    model_fn,
    img_t: torch.Tensor,
    tile: int = 512,
    overlap: int = 32,
) -> torch.Tensor:
    """Run the upscaler model patch-by-patch to cap peak VRAM usage.

    Overlapping tiles are averaged so there are no seam artifacts. Output is
    assembled on CPU so only one tile lives on GPU at a time.
    """
    _, c, h, w = img_t.shape

    # Probe output scale with a tiny crop rather than hard-coding it
    with torch.no_grad():
        probe = model_fn(img_t[:, :, :min(4, h), :min(4, w)])
    scale = probe.shape[-1] // min(4, w)
    del probe
    torch.cuda.empty_cache()

    out_h, out_w = h * scale, w * scale
    canvas = torch.zeros(1, c, out_h, out_w, dtype=torch.float32)
    weights = torch.zeros(1, 1, out_h, out_w, dtype=torch.float32)

    step = max(tile - overlap, 1)
    # Ensure the last tile always reaches the edge
    ys = list(range(0, max(h - tile, 0), step)) + [max(h - tile, 0)]
    xs = list(range(0, max(w - tile, 0), step)) + [max(w - tile, 0)]
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            patch = img_t[:, :, y0:y1, x0:x1]
            with torch.no_grad():
                out = model_fn(patch).cpu().float()
            oy0, ox0 = y0 * scale, x0 * scale
            oy1, ox1 = y1 * scale, x1 * scale
            canvas[:, :, oy0:oy1, ox0:ox1] += out
            weights[:, :, oy0:oy1, ox0:ox1] += 1.0

    return (canvas / weights.clamp(min=1)).clamp(0, 1)


def apply_realesrgan(image: Image.Image, model_key: str, device: torch.device) -> Image.Image:
    """Upscale a PIL image using the model selected in UPSCALE_MODELS.

    Weights are downloaded once to /tmp/realesrgan_weights/ and reused.
    """
    cfg = UPSCALE_MODELS[model_key]

    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError:
        raise gr.Error("spandrel is not installed. Add 'spandrel' to requirements.txt.")

    cache_dir = "/tmp/realesrgan_weights"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cfg["file"])

    if not os.path.exists(cache_path):
        print(f"Downloading {cfg['file']}…")
        urllib.request.urlretrieve(cfg["url"], cache_path)

    model = ModelLoader().load_from_file(cache_path)
    if not isinstance(model, ImageModelDescriptor):
        raise gr.Error(f"Loaded model is not a single-image descriptor: {type(model)}")

    sr_model = model.model.to(device).eval()

    img_np = np.array(image).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    out_t = _upscale_tiled(sr_model, img_t, tile=512, overlap=32)

    # Free upscaler weights immediately — FLUX pipeline stays resident
    del sr_model, img_t
    gc.collect()
    torch.cuda.empty_cache()

    out_np = out_t.squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((out_np * 255).astype(np.uint8))
