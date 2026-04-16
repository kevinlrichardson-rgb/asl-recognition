"""
app.py -- Gradio web app for ASL Recognition.

Fingerspell: real-time letter-by-letter recognition (A-Z)

Runs on Hugging Face Spaces or locally with: python app.py
"""

import urllib.request
from collections import deque
from pathlib import Path

import cv2
import gradio as gr
import mediapipe as mp_lib
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    RunningMode,
)
from spellchecker import SpellChecker


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent

FS_MODEL_PATH   = ROOT / "models" / "asl_model.pt"
FS_CLASSES_PATH = ROOT / "models" / "label_classes.npy"
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


# ── Constants ──────────────────────────────────────────────────────────────────

NUM_LANDMARKS, COORDS_PER_LM = 21, 3

WINDOW_SIZE = 15
STABLE_COUNT = 10
COOLDOWN_FRAMES = 8
FS_CONFIDENCE_THRESHOLD = 0.50

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]


# ── Model definition ───────────────────────────────────────────────────────────

class ASLClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Landmark helpers ───────────────────────────────────────────────────────────

def normalise_landmarks(landmarks_list):
    pts = np.array(landmarks_list, dtype=np.float32).reshape(NUM_LANDMARKS, COORDS_PER_LM)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_hand(frame_bgr: np.ndarray, landmarks, color=(0, 255, 255)):
    """Draw hand skeleton on a BGR frame."""
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        radius = 10 if i == 0 else 7
        cv2.circle(frame_bgr, pt, radius + 2, (255, 255, 255), 2, cv2.LINE_AA)  # white halo
        cv2.circle(frame_bgr, pt, radius, color, -1, cv2.LINE_AA)


def _annotate(frame_rgb: np.ndarray, hand_landmarks_list=None) -> np.ndarray:
    """Convert RGB frame to BGR, draw hand landmarks, return RGB result."""
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if hand_landmarks_list:
        colors = [(0, 255, 255), (255, 0, 255)]  # bright yellow, bright magenta (BGR)
        for i, lms in enumerate(hand_landmarks_list):
            draw_hand(out, lms, colors[i % 2])
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _make_placeholder_image(line1: str, line2: str = "",
                             w: int = 640, h: int = 480) -> np.ndarray:
    """Dark placeholder image shown before the camera feed is active."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    tw, _ = cv2.getTextSize(line1, font, scale, thick)[0]
    cv2.putText(img, line1, ((w - tw) // 2, h // 2 - (18 if line2 else 0)),
                font, scale, (200, 200, 200), thick, cv2.LINE_AA)
    if line2:
        tw2, _ = cv2.getTextSize(line2, font, 0.6, 1)[0]
        cv2.putText(img, line2, ((w - tw2) // 2, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
    return img


# ── Multi-strategy hand detection (mirrors demo.py) ───────────────────────────

def _detect_hand_live(frame_rgb: np.ndarray, detector):
    """Try several preprocessing strategies to maximise hand detection rate."""
    mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_img)
    if result.hand_landmarks:
        return result

    h, w = frame_rgb.shape[:2]
    pad = int(max(h, w) * 0.15)
    padded = cv2.copyMakeBorder(frame_rgb, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=padded)
    result = detector.detect(mp_img)
    if result.hand_landmarks:
        return result

    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=enhanced)
    return detector.detect(mp_img)


# ── Caption HTML helpers ───────────────────────────────────────────────────────

_CAPTION_BASE = (
    "background:#000;border-radius:10px;padding:28px 36px;"
    "min-height:180px;display:flex;flex-direction:column;"
    "align-items:center;justify-content:center;gap:14px;width:100%;"
)


def _fs_caption_html(current_letter: str | None, confidence: float,
                     word_buffer: list, suggestion: str) -> str:
    spelled     = "".join(word_buffer)
    letter_html = (current_letter or "").strip() or "&nbsp;"
    conf_html   = f"{confidence*100:.0f}%" if (current_letter or "").strip() else ""
    sugg_html   = (f'<span style="color:#fff;">Suggestion:&nbsp;</span>'
                   f'<b style="color:#fff;">{suggestion}</b>') if suggestion else "&nbsp;"
    return (
        f'<div style="{_CAPTION_BASE}font-family:\'Courier New\',monospace;">'
        f'<div style="color:#fff;font-size:240px;font-weight:bold;line-height:1;'
        f'min-height:264px;text-align:center;">{letter_html}</div>'
        f'<div style="color:#888;font-size:15px;min-height:20px;">{conf_html}</div>'
        f'<div style="color:#fff;font-size:30px;letter-spacing:8px;'
        f'min-height:44px;text-align:center;">{spelled}_</div>'
        f'<div style="color:#fff;font-size:22px;font-style:italic;">{sugg_html}</div>'
        f'</div>'
    )


# ── LetterSmoother ─────────────────────────────────────────────────────────────

class LetterSmoother:
    def __init__(self):
        self.window: deque[str | None] = deque(maxlen=WINDOW_SIZE)
        self._prev_smoothed: str | None = None
        self._same_count = 0
        self._cooldown_remaining = 0

    def update(self, letter: str | None) -> str | None:
        self.window.append(letter)
        counts: dict[str, int] = {}
        for ltr in self.window:
            if ltr is not None:
                counts[ltr] = counts.get(ltr, 0) + 1
        if not counts:
            self._prev_smoothed = None
            self._same_count = 0
            return None
        smoothed = max(counts, key=lambda k: counts[k])
        if smoothed == self._prev_smoothed:
            self._same_count += 1
        else:
            self._prev_smoothed = smoothed
            self._same_count = 1
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return None
        if self._same_count >= STABLE_COUNT:
            self._same_count = 0
            self._cooldown_remaining = COOLDOWN_FRAMES
            return smoothed
        return None


# ── Model loading (cached) ─────────────────────────────────────────────────────

def _ensure_model(path: Path, url: str, name: str):
    if not path.exists():
        print(f"Downloading {name} model ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)


def load_fingerspell():
    if not FS_MODEL_PATH.exists() or not FS_CLASSES_PATH.exists():
        return None
    _ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "hand")

    ckpt    = torch.load(FS_MODEL_PATH, map_location="cpu", weights_only=False)
    classes = np.load(FS_CLASSES_PATH, allow_pickle=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = ASLClassifier(ckpt["input_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    ))
    print(f"Fingerspell model loaded ({len(classes)} classes, device={device})")
    return model, classes, device, detector


def _warmup_fingerspell(assets):
    """Run one dummy inference to pre-compile kernels and avoid first-frame lag."""
    if assets is None:
        return
    model, classes, device, detector = assets
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=dummy)
    detector.detect(mp_img)
    feat_dim = model.net[0].in_features
    with torch.no_grad():
        model(torch.zeros((1, feat_dim), device=device))
    print("Fingerspell warmup complete")


# Load model at import time (cached by the process)
fs_assets = load_fingerspell()
spell     = SpellChecker()

# Warm up model so the first real frame has no compilation delay
_warmup_fingerspell(fs_assets)


# ── Fingerspell processing ─────────────────────────────────────────────────────

def _suggest_word(word_buffer: list[str]) -> str:
    raw = "".join(word_buffer).strip().lower()
    if not raw:
        return ""
    if not spell.unknown([raw]):
        return raw.upper()
    correction = spell.correction(raw)
    return correction.upper() if correction else raw.upper()


_FS_PLACEHOLDER = _make_placeholder_image(
    "Enable camera to begin", "Show ASL letters to the camera")


def process_fingerspell(frame, state):
    if frame is None or fs_assets is None:
        return _FS_PLACEHOLDER, _fs_caption_html(None, 0.0, [], ""), state

    model, classes, device, detector = fs_assets
    smoother    = state["smoother"]
    word_buffer = state["word_buffer"]

    result = _detect_hand_live(frame, detector)

    current_letter = None
    confidence     = 0.0

    if result.hand_landmarks:
        raw = []
        for lm in result.hand_landmarks[0]:
            raw.extend([lm.x, lm.y, lm.z])
        features = normalise_landmarks(raw)

        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        probs          = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx       = int(np.argmax(probs))
        current_letter = str(classes[pred_idx])
        confidence     = float(probs[pred_idx])

        if confidence < FS_CONFIDENCE_THRESHOLD:
            current_letter = None

    if current_letter in ("del", "nothing"):
        current_letter = None
    elif current_letter == "space":
        current_letter = " "

    accepted = smoother.update(current_letter)

    if accepted is not None:
        if accepted == " ":
            if word_buffer and word_buffer[-1] != " ":
                word_buffer.append(" ")
        elif not word_buffer or word_buffer[-1] != accepted:
            word_buffer.append(accepted)

    state["smoother"]    = smoother
    state["word_buffer"] = word_buffer

    annotated = _annotate(
        frame,
        hand_landmarks_list=result.hand_landmarks if result.hand_landmarks else None,
    )
    suggestion = _suggest_word(word_buffer)
    caption    = _fs_caption_html(current_letter, confidence, word_buffer, suggestion)
    return annotated, caption, state


def clear_fingerspell(state):
    state["word_buffer"] = []
    state["smoother"]    = LetterSmoother()
    return _fs_caption_html(None, 0.0, [], ""), state


# ── Gradio state factory ───────────────────────────────────────────────────────

def make_fs_state():
    return {"smoother": LetterSmoother(), "word_buffer": []}


# ── Gradio UI ──────────────────────────────────────────────────────────────────

# When the webcam is closed and re-opened, Gradio's internal recording state (I)
# is not reset, so the Record button shows "Stop" even though nothing is streaming.
# This JS auto-clicks any stale "Stop" button when a video element starts playing
# (i.e., when the webcam is re-enabled), driving Gradio's he() handler to reset I.
_WEBCAM_SYNC_JS = """
function() {
    document.addEventListener('play', function(e) {
        if (e.target.tagName !== 'VIDEO') return;
        setTimeout(function() {
            document.querySelectorAll('[title="stop recording"]').forEach(function(icon) {
                var btn = icon.closest('button');
                if (btn) btn.click();
            });
        }, 150);
    }, true);
}
"""

with gr.Blocks(title="ASL Recognition", js=_WEBCAM_SYNC_JS) as demo:
    gr.Markdown("# ASL Recognition\nReal-time American Sign Language recognition using your camera.")
    gr.Markdown(
        "**Fingerspelling mode**: Show ASL letters to the camera one at a time. "
        "Hold each letter steady — the model spells out words in the caption below."
    )

    fs_state = gr.State(value=make_fs_state)

    with gr.Row():
        with gr.Column(scale=1, min_width=180):
            fs_webcam = gr.Image(sources=["webcam"], streaming=True,
                                 label="Enable Camera", mirror_webcam=False,
                                 height=180)
        with gr.Column(scale=3):
            fs_output = gr.Image(label="Live Feed", mirror_webcam=False)

    fs_caption   = gr.HTML(value=_fs_caption_html(None, 0.0, [], ""))
    fs_clear_btn = gr.Button("Clear")

    fs_webcam.stream(
        fn=process_fingerspell,
        inputs=[fs_webcam, fs_state],
        outputs=[fs_output, fs_caption, fs_state],
        stream_every=0.1,
    )
    fs_clear_btn.click(
        fn=clear_fingerspell,
        inputs=[fs_state],
        outputs=[fs_caption, fs_state],
    )


if __name__ == "__main__":
    demo.launch()
