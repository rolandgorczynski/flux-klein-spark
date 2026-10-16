"""Side-by-side image comparison with a draggable divider.

Uses a hidden <input type="range"> slider + inline event handlers
(oninput, onmousedown, ontouchstart, onmousemove, ontouchmove) to
drive the CSS clip-path split.  This survives Gradio's HTML sanitizer
because it avoids <script> tags entirely.
"""
from __future__ import annotations

import base64
import os
import random
from io import BytesIO
from PIL import Image


def _pil_to_data_uri(img, fmt="jpeg"):
    """Convert a PIL image to a base64 data URI for inline HTML display."""
    buf = BytesIO()
    # Preserve RGBA → RGBA for transparency support
    if img.mode == "RGBA":
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{encoded}"
    img.convert("RGB").save(buf, format=fmt.upper(), quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt};base64,{encoded}"


def _path_to_data_uri(path):
    """Load an image from a file path or PIL Image and convert to base64 data URI."""
    # Gradio gr.Image components pass PIL Image objects, not file paths
    if isinstance(path, Image.Image):
        return _pil_to_data_uri(path)
    # Otherwise treat as file path (from gallery selection, etc.)
    img = Image.open(path).convert("RGB")
    return _pil_to_data_uri(img)


def build_comparison_html(path_base, path_generated, width=512, height=512):
    """Return HTML for a draggable before/after slider comparison.

    Left = base image, Right = generated image.
    Click and drag anywhere to move the vertical divider.
    path_base / path_generated can be file paths (str) or PIL Image objects.
    """
    img_base = _path_to_data_uri(path_base) if path_base else ""
    img_gen  = _path_to_data_uri(path_generated) if path_generated else ""
    label_base = os.path.basename(path_base) if isinstance(path_base, str) else "(base)"
    label_gen  = os.path.basename(path_generated) if isinstance(path_generated, str) else "(generated)"
    w, h = str(width), str(height)
    uid = random.randint(10000, 99999)

    return (
        # Hidden range slider (Gradio keeps <input> elements)
        '<input type="range" id="rng' + str(uid) + '" min="2" max="98" value="50"'
        ' style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;"'
        ' oninput="'
        "var p=this.value;"
        "var g=document.getElementById('g" + str(uid) + "');"
        "var l=document.getElementById('l" + str(uid) + "');"
        "var h=document.getElementById('h" + str(uid) + "');"
        "if(g)g.style.clipPath='inset(0 0 0 '+p+'%)';"
        "if(l)l.style.left=p+'%';"
        "if(h)h.style.left=p+'%';"
        '">'

        # Main container — mouse/touch events update the hidden range
        '<div id="c' + str(uid) + '" style="margin:0 auto;max-width:' + w + 'px;">'
        '<div style="position:relative;width:' + w + 'px;height:' + h
        + 'px;cursor:ew-resize;user-select:none;border-radius:8px;overflow:hidden;background:#222;"'
        ' onmousedown="'
        "var r=document.getElementById('rng" + str(uid) + "');if(!r)return;"
        "r._d=1;var b=this.getBoundingClientRect();"
        "r.value=Math.max(2,Math.min(98,(event.clientX-b.left)/b.width*100));"
        "r.dispatchEvent(new Event('input',{bubbles:true}));"
        '" ontouchstart="'
        "var r=document.getElementById('rng" + str(uid) + "');if(!r)return;"
        "r._d=1;var b=this.getBoundingClientRect();"
        "r.value=Math.max(2,Math.min(98,(event.touches[0].clientX-b.left)/b.width*100));"
        "r.dispatchEvent(new Event('input',{bubbles:true}));"
        '" onmousemove="'
        "var r=document.getElementById('rng" + str(uid) + "');if(!r||!r._d)return;"
        "var b=this.getBoundingClientRect();"
        "r.value=Math.max(2,Math.min(98,(event.clientX-b.left)/b.width*100));"
        "r.dispatchEvent(new Event('input',{bubbles:true}));"
        '" onmouseup="'
        "var r=document.getElementById('rng" + str(uid) + "');if(r)r._d=0;"
        '" onmouseleave="'
        "var r=document.getElementById('rng" + str(uid) + "');if(r)r._d=0;"
        '" ontouchmove="'
        "var r=document.getElementById('rng" + str(uid) + "');if(!r||!r._d)return;"
        "var b=this.getBoundingClientRect();"
        "r.value=Math.max(2,Math.min(98,(event.touches[0].clientX-b.left)/b.width*100));"
        "r.dispatchEvent(new Event('input',{bubbles:true}));"
        '" ontouchend="'
        "var r=document.getElementById('rng" + str(uid) + "');if(r)r._d=0;"
        '">'

        # Base image (bottom layer, full width)
        '<img src="' + img_base + '" style="position:absolute;top:0;left:0;width:' + w
        + 'px;height:' + h + 'px;object-fit:contain;pointer-events:none;" alt="Base">'

        # Generated image (top layer, clipped from left by clip-path)
        '<img id="g' + str(uid) + '" src="' + img_gen + '" style="position:absolute;top:0;left:0;width:'
        + w + 'px;height:' + h + 'px;object-fit:contain;clip-path:inset(0 0 0 50%);pointer-events:none;" alt="Generated">'

        # Divider line
        '<div id="l' + str(uid) + '" style="position:absolute;top:0;left:50%;width:3px;height:100%;'
        'background:white;box-shadow:0 0 6px rgba(0,0,0,0.5);pointer-events:none;transform:translateX(-1.5px);"></div>'

        # Handle knob
        '<div id="h' + str(uid) + '" style="position:absolute;top:50%;left:50%;'
        'transform:translate(-50%,-50%);width:32px;height:32px;background:white;border-radius:50%;'
        'box-shadow:0 0 10px rgba(0,0,0,.5);pointer-events:none;display:flex;align-items:center;'
        'justify-content:center;font-size:14px;color:#333;">&#8644;</div>'

        '</div>'
        # Labels
        '<div style="display:flex;justify-content:space-between;margin-top:6px;font-size:13px;color:#bbb;max-width:'
        + w + 'px;"><span>&#128309; Base: ' + label_base
        + '</span><span>&#128310; Generated: ' + label_gen + '</span></div>'
        '</div>'
    )


def build_placeholder_html(width=512, height=512):
    """Return HTML for the empty state (no images loaded yet)."""
    w, h = str(width), str(height)
    return (
        '<div style="margin:0 auto;max-width:' + w + 'px;width:' + w + 'px;height:' + h
        + 'px;display:flex;align-items:center;justify-content:center;'
        'background:var(--block-background-fill);border-radius:8px;'
        'color:var(--body-text-color-subdued);font-size:15px;text-align:center;line-height:1.6;">'
        '<div>Generate an image, then click<br>'
        '<b>Send &#8594; Compare</b> to compare with the base image</div>'
        '</div>'
    )