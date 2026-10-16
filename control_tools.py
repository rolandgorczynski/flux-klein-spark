"""Depth + pose estimation helpers for the Depth/Pose tab.

Two heavy models are lazy-loaded on first use, both inside @GPU
functions so ZeroGPU schedules them as normal short-lived workloads:
  - Depth-Anything-V2-Small via transformers' depth-estimation pipeline
  - OpenposeDetector (lllyasviel/Annotators) via controlnet_aux

Pose keypoints round-trip through a plain JSON-friendly list-of-dicts so they
serialise cleanly into gr.State — that's what makes click-to-edit feasible.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import gradio as gr
import torch
from runtime import GPU


# ── Standard OpenPose body-18 layout (COCO-18 ordering) ─────────────────────
OPENPOSE_KEYPOINT_NAMES = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
]

# Bones as 0-indexed (a, b) pairs into OPENPOSE_KEYPOINT_NAMES. Mirrors the
# OpenPose `limbSeq` constant so ControlNet/RefControl-Pose readers see the
# exact skeleton topology they expect.
SKELETON_EDGES = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# OpenPose-canonical 18-joint colour table (used for both bones and dots).
JOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]


# ── Lazy model loaders ───────────────────────────────────────────────────────

_depth_pipeline = None
_pose_detector = None


def _load_depth_pipeline():
    """Loaded inside the first @GPU call, then cached for the rest of
    the worker's lifetime. Avoids paying the cold-start cost for users who
    only ever use depth or only ever use pose."""
    global _depth_pipeline
    if _depth_pipeline is None:
        from transformers import pipeline
        _depth_pipeline = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if torch.cuda.is_available() else -1,
        )
    return _depth_pipeline


def _load_pose_detector():
    global _pose_detector
    if _pose_detector is None:
        try:
            from controlnet_aux import OpenposeDetector
        except ImportError:
            raise gr.Error(
                "controlnet_aux is not installed. Add `controlnet_aux` to requirements.txt."
            )
        _pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        if torch.cuda.is_available():
            _pose_detector = _pose_detector.to("cuda")
    return _pose_detector


# ── Depth ────────────────────────────────────────────────────────────────────

@GPU
def generate_depthmap(image: Image.Image) -> Image.Image:
    if image is None:
        raise gr.Error("Upload a source image first.")
    pipe = _load_depth_pipeline()
    depth = pipe(image.convert("RGB"))["depth"]  # PIL L
    return depth.convert("RGB")


# ── Pose detection → editable keypoint list ─────────────────────────────────

@GPU
def detect_pose(image: Image.Image) -> tuple[list[list[dict]], int, int]:
    """Run pose detection and return:
      - poses : list of dicts, one per detected person. Each pose is a list of
                18 entries {"name", "x", "y", "visible"} — in *pixel* coords
                (not normalised), so the editor can pass click positions in
                directly without a coordinate transform.
      - w, h  : source image dimensions.
    """
    if image is None:
        raise gr.Error("Upload a source image first.")
    img = image.convert("RGB")
    w, h = img.size
    detector = _load_pose_detector()
    try:
        poses_raw = detector.detect_poses(np.array(img))
    except Exception as e:
        raise gr.Error(f"Pose detection failed: {e}")

    out = []
    for pose in poses_raw:
        body_kps = pose.body.keypoints if pose.body else [None] * 18
        person = []
        for i in range(18):
            kp = body_kps[i] if i < len(body_kps) else None
            name = OPENPOSE_KEYPOINT_NAMES[i]
            # controlnet_aux returns normalised (x, y) ∈ [0, 1] with a score.
            # Drop low-confidence detections so the editor starts clean.
            if kp is None or (getattr(kp, "score", 1.0) or 0) < 0.3:
                person.append({"name": name, "x": 0.0, "y": 0.0, "visible": False})
            else:
                person.append({
                    "name": name,
                    "x": float(kp.x) * w,
                    "y": float(kp.y) * h,
                    "visible": True,
                })
        out.append(person)
    return out, w, h


# ── Rendering ───────────────────────────────────────────────────────────────

def render_pose_skeleton(poses: list[list[dict]], width: int, height: int) -> Image.Image:
    """Skeleton on a black canvas — this is what ControlNet/RefControl-Pose wants."""
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for pose in poses:
        for a, b in SKELETON_EDGES:
            ka, kb = pose[a], pose[b]
            if not (ka["visible"] and kb["visible"]):
                continue
            draw.line([ka["x"], ka["y"], kb["x"], kb["y"]],
                      fill=JOINT_COLORS[a], width=4)
        for i, kp in enumerate(pose):
            if not kp["visible"]:
                continue
            r = 4
            draw.ellipse([kp["x"] - r, kp["y"] - r, kp["x"] + r, kp["y"] + r],
                         fill=JOINT_COLORS[i])
    return canvas


def render_pose_overlay(
    source: Image.Image,
    poses: list[list[dict]],
    active_person: int | None = None,
    active_joint: int | None = None,
) -> Image.Image | None:
    """Skeleton on top of a dimmed source — the editing view. The active joint
    gets a white-ringed highlight so the user can see what their next click
    will move."""
    if source is None:
        return None
    w, h = source.size
    base = ImageEnhance.Brightness(source.convert("RGB")).enhance(0.45)
    skel = render_pose_skeleton(poses, w, h)
    np_base, np_skel = np.array(base), np.array(skel)
    mask = np_skel.any(axis=2)
    np_base[mask] = np_skel[mask]
    out = Image.fromarray(np_base)

    if (active_person is not None and 0 <= active_person < len(poses)
            and active_joint is not None and 0 <= active_joint < 18):
        kp = poses[active_person][active_joint]
        if kp["visible"]:
            d = ImageDraw.Draw(out)
            x, y = kp["x"], kp["y"]
            # Double ring so it stays visible on any background colour
            for r, col in [(10, (255, 255, 255)), (7, (0, 0, 0))]:
                d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=2)
    return out


# ── Edit operations ─────────────────────────────────────────────────────────

def move_joint(poses, person_idx, joint_idx, x, y, width, height):
    """Move (and make visible) a joint. Clamped to image bounds so clicks just
    outside the canvas don't fly off."""
    if (not poses or person_idx is None or joint_idx is None
            or not (0 <= person_idx < len(poses) and 0 <= joint_idx < 18)):
        return poses
    x = float(max(0, min(x, width - 1)))
    y = float(max(0, min(y, height - 1)))
    new = [list(p) for p in poses]
    new[person_idx][joint_idx] = {
        **new[person_idx][joint_idx], "x": x, "y": y, "visible": True,
    }
    return new


def hide_joint(poses, person_idx, joint_idx):
    if (not poses or person_idx is None or joint_idx is None
            or not (0 <= person_idx < len(poses) and 0 <= joint_idx < 18)):
        return poses
    new = [list(p) for p in poses]
    new[person_idx][joint_idx] = {**new[person_idx][joint_idx], "visible": False}
    return new


def clear_all_joints(poses):
    return [[{**kp, "visible": False} for kp in pose] for pose in (poses or [])]


def default_pose_template(width: int, height: int) -> list[dict]:
    """Standing-figure skeleton centred in the canvas — handy when detection
    misses everyone, or when you want to draw a pose from scratch."""
    cx = width / 2
    s = min(width, height) * 0.40
    top = height / 2 - s
    def y(f): return top + f * (2 * s)
    return [
        {"name": "nose",           "x": cx,             "y": y(0.05), "visible": True},
        {"name": "neck",           "x": cx,             "y": y(0.15), "visible": True},
        {"name": "right_shoulder", "x": cx + s * 0.18,  "y": y(0.18), "visible": True},
        {"name": "right_elbow",    "x": cx + s * 0.25,  "y": y(0.35), "visible": True},
        {"name": "right_wrist",    "x": cx + s * 0.30,  "y": y(0.50), "visible": True},
        {"name": "left_shoulder",  "x": cx - s * 0.18,  "y": y(0.18), "visible": True},
        {"name": "left_elbow",     "x": cx - s * 0.25,  "y": y(0.35), "visible": True},
        {"name": "left_wrist",     "x": cx - s * 0.30,  "y": y(0.50), "visible": True},
        {"name": "right_hip",      "x": cx + s * 0.12,  "y": y(0.55), "visible": True},
        {"name": "right_knee",     "x": cx + s * 0.13,  "y": y(0.75), "visible": True},
        {"name": "right_ankle",    "x": cx + s * 0.14,  "y": y(0.95), "visible": True},
        {"name": "left_hip",       "x": cx - s * 0.12,  "y": y(0.55), "visible": True},
        {"name": "left_knee",      "x": cx - s * 0.13,  "y": y(0.75), "visible": True},
        {"name": "left_ankle",     "x": cx - s * 0.14,  "y": y(0.95), "visible": True},
        {"name": "right_eye",      "x": cx + s * 0.025, "y": y(0.03), "visible": True},
        {"name": "left_eye",       "x": cx - s * 0.025, "y": y(0.03), "visible": True},
        {"name": "right_ear",      "x": cx + s * 0.05,  "y": y(0.05), "visible": True},
        {"name": "left_ear",       "x": cx - s * 0.05,  "y": y(0.05), "visible": True},
    ]


# ── Dropdown helpers ─────────────────────────────────────────────────────────

def person_choices(poses):
    return [f"Person {i+1}" for i in range(len(poses or []))]

def parse_person_idx(label: str | None) -> int | None:
    if not label:
        return None
    try:
        return int(label.replace("Person ", "")) - 1
    except ValueError:
        return None

def joint_name_to_index(name: str | None) -> int:
    if not name:
        return -1
    try:
        return OPENPOSE_KEYPOINT_NAMES.index(name)
    except ValueError:
        return -1
