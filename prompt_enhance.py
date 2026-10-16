"""Prompt enhancement via a single vision-language model (Qwen3-VL-2B).

Single-stage pipeline (lazily loaded):
  Qwen3-VL-2B-Instruct -> enhanced prompt, optionally conditioned on the
  base image passed directly into the chat message.

This replaces the previous two-stage design (Florence-2 caption -> Qwen2.5
text rewrite). A unified VLM removes the lossy caption intermediary and the
Florence-specific `forced_bos_token_id` crash, and lets the enhancer reason
over the actual pixels rather than a text description of them.

The model is loaded on first use inside @GPU and cached for the
worker's lifetime. Memory impact: ~4-5 GB in bfloat16 - fits comfortably
alongside FLUX.2 Klein on typical ZeroGPU quotas.
"""
from __future__ import annotations

import os
import re
import gradio as gr
from runtime import GPU
import torch
from PIL import Image

# -- Style definitions --------------------------------------------------------
# Aligned with Black Forest Labs FLUX prompting guidance:
# use natural descriptive language, not comma-separated keyword lists.
# Domain-specific styles differ by *what details to notice*, not output format.

ENHANCE_STYLES = {
    "FLUX natural (general)": (
        "You are a prompt rewriter for an AI image editor. "
        "Rewrite the user's edit instruction into a clear, descriptive, "
        "natural-language prompt suitable for FLUX image editing. "
        "CRITICAL RULES:\n"
        "- Do NOT change the edit the user requested.\n"
        "- Do NOT add new subjects, objects, scene elements, or modifications.\n"
        "- Do NOT suggest alternative edits.\n"
        "- Use descriptive language (not comma-separated tags).\n"
        "- Output ONLY the rewritten prompt, no commentary.\n\n"
        "Instruction: "
    ),
    "Portrait / human refinement": (
        "You are a prompt rewriter for an AI portrait/image editor. "
        "Rewrite the user's edit instruction into a clear, descriptive, "
        "natural-language prompt suitable for FLUX image editing. "
        "CRITICAL RULES:\n"
        "- Do NOT change the edit the user requested.\n"
        "- Do NOT add new subjects, objects, scene elements, or modifications.\n"
        "- Do NOT alter clothing, hair colour, lighting, or background "
        "unless the user explicitly asked you to.\n"
        "- Do NOT suggest alternative edits or safer versions.\n"
        "- Use descriptive language (not comma-separated tags).\n"
        "- Output ONLY the rewritten prompt, no commentary.\n\n"
        "Instruction: "
    ),
    "Landscape / object / scene": (
        "You are a prompt rewriter for an AI image editor specialising in "
        "landscapes, interiors, objects, and outdoor scenes. "
        "Rewrite the user's edit instruction into a clear, descriptive, "
        "natural-language prompt suitable for FLUX image editing. "
        "CRITICAL RULES:\n"
        "- Do NOT change the edit the user requested.\n"
        "- Do NOT add new objects, buildings, weather, or scene elements.\n"
        "- Do NOT suggest alternative edits or safer versions.\n"
        "- Use descriptive language (not comma-separated tags).\n"
        "- Output ONLY the rewritten prompt, no commentary.\n\n"
        "Instruction: "
    ),
    "Fix grammar only": (
        "Fix any spelling, grammar, or awkward phrasing in the following "
        "prompt. Do NOT add details, expand, or change meaning. "
        "Output ONLY the corrected text, no commentary:\n\n"
    ),
}

# -- Cached model -------------------------------------------------------------

_VLM = None            # (processor, model)


def _load_vlm():
    """Lazy-load the vision-language model used for prompt enhancement."""
    global _VLM
    if _VLM is not None:
        return _VLM

    model_id = os.environ.get("PROMPT_ENHANCE_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    )

    _VLM = (processor, model)
    print(f"[prompt_enhance] loaded VLM enhancer ({model_id})")
    return _VLM


# -- Faithfulness guard -------------------------------------------------------
# Cheap keyword-overlap check that catches the worst failure mode: the enhancer
# quietly swapping in a completely different edit. If the rewrite drops too many
# of the user's content words we reject it and keep the original instruction.

_STOPWORDS = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "with", "for", "at",
    "by", "from", "into", "make", "turn", "change", "set", "this", "that", "it",
    "its", "is", "are", "be", "as", "image", "photo", "picture", "please",
    "edit", "instruction",
}


def _content_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _is_faithful(original: str, enhanced: str, min_overlap: float = 0.5) -> bool:
    """Return True if `enhanced` keeps enough of `original`'s content words."""
    orig_kw = _content_keywords(original)
    if not orig_kw:
        return True
    kept = orig_kw & _content_keywords(enhanced)
    return (len(kept) / len(orig_kw)) >= min_overlap


def _enhance(prompt: str, style: str, image: Image.Image | None) -> str:
    """Rephrase the prompt with the VLM, passing the image directly when given."""
    processor, model = _load_vlm()

    system_prompt = ENHANCE_STYLES.get(style, ENHANCE_STYLES["FLUX natural (general)"])

    # Multimodal user turn: attach the image only when context is requested.
    user_content: list[dict] = []
    if image is not None:
        user_content.append({"type": "image", "image": image.convert("RGB")})
    user_content.append({"type": "text", "text": f"Instruction: {prompt}"})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            # Greedy (deterministic) decoding. Sampling was enough randomness
            # for the model to occasionally drift into an alternative edit.
            # Greedy keeps the rewrite tightly anchored to the user's
            # instruction and makes the Enhance button reproducible.
            do_sample=False,
            repetition_penalty=1.05,
        )

    trimmed = outputs[:, inputs["input_ids"].shape[1]:]
    result = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return result.strip()


@GPU
def enhance_prompt(
    original: str,
    base_image=None,
    style: str | None = None,
    use_image_context: bool = False,
) -> str:
    """Enhance a user's prompt, optionally using the base image for context.

    Parameters
    ----------
    original : str
        The raw prompt from the user.
    base_image : PIL.Image or None
        The current base image. Passed directly to the VLM when
        use_image_context is True.
    style : str or None
        Key into ENHANCE_STYLES. Defaults to "FLUX natural (general)".
    use_image_context : bool
        Whether to condition the rewrite on the base image.

    Returns
    -------
    str
        The enhanced prompt, or the original if enhancement fails or drifts.
    """
    if not original or not original.strip():
        return ""

    style = style or "FLUX natural (general)"

    image = base_image if (use_image_context and base_image is not None) else None

    try:
        enhanced = _enhance(original, style, image)
        if not enhanced or not enhanced.strip():
            gr.Warning("Enhancer returned empty - keeping original.")
            return original.strip()

        # Reject rewrites that wandered off into a different edit. The grammar
        # style is exempt because it may legitimately rewrite most words.
        if style != "Fix grammar only" and not _is_faithful(original, enhanced):
            gr.Warning(
                "Enhanced prompt drifted from your instruction - keeping "
                "original. Try again or pick a different enhance style."
            )
            return original.strip()

        return enhanced.strip()
    except Exception as e:
        gr.Warning(f"Enhancement failed: {e}")
        return original.strip()