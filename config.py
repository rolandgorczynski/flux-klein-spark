"""Environment-derived configuration and global constants.

Importing this module has no side effects beyond reading env vars and computing
a handful of constants. It must not import torch, diffusers, or gradio so it can
be imported anywhere without dragging in heavy dependencies.
"""
import os
import numpy as np

# ── Model variant selection ──────────────────────────────────────────────────
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "9B")
if MODEL_VARIANT == "9B-KV":
    _MODEL_REPO = "black-forest-labs/FLUX.2-klein-9b-kv"
else:
    MODEL_VARIANT = "9B"  # normalise any typos back to default
    _MODEL_REPO = "black-forest-labs/FLUX.2-klein-9B"

MODEL_REPO = _MODEL_REPO

# ── Misc constants ───────────────────────────────────────────────────────────
MAX_SEED = int(np.iinfo(np.int32).max)

# Maximum number of simultaneous LoRA weight sliders to pre-render in the UI.
MAX_LORA_SLOTS = 6

# Logging toggle — read once, exposed as a plain bool so other modules don't
# repeat the env-string-parsing dance.
ENABLE_LOGGING = False
