"""
app.py -- Gradio web app for ASL Recognition.

Two tabs:
  1. Fingerspell: real-time letter-by-letter recognition (A-Z)
  2. Word Sign:   sliding-window word-sign recognition (WLASL)

Runs on Hugging Face Spaces or locally with: python app.py
"""

import os
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
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode,
)
from spellchecker import SpellChecker


# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent

FS_MODEL_PATH   = ROOT / "models" / "asl_model.pt"
FS_CLASSES_PATH = ROOT / "models" / "label_classes.npy"
WS_MODEL_PATH   = ROOT / "models" / "wlasl_word_model.pt"
WS_CLASSES_PATH = ROOT / "models" / "wlasl_classes.npy"
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
POSE_MODEL_PATH = ROOT / "models" / "pose_landmarker.task"

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


# ── Constants ────────────────────────────────────────────────────────────────

POSE_N, HAND_N, COORDS = 33, 21, 3
NUM_LANDMARKS, COORDS_PER_LM = 21, 3

WINDOW_SIZE = 15
STABLE_COUNT = 10
COOLDOWN_FRAMES = 8
FS_CONFIDENCE_THRESHOLD = 0.55
WS_STRIDE = 8

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
_POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
]


# ── Model definitions ────────────────────────────────────────────────────────

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


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class WLASLModel(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = AttentionPool(hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.attn(x, lengths)
        return self.classifier(x)


# ── Landmark helpers ─────────────────────────────────────────────────────────

def _lm_to_array(landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def _normalise_hand(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def _normalise_pose(pts: np.ndarray) -> np.ndarray:
    centre = (pts[23] + pts[24]) / 2.0
    pts = pts - centre
    shoulder_dist = np.linalg.norm(pts[11] - pts[12])
    if shoulder_dist > 1e-6:
        pts /= shoulder_dist
    return pts.flatten()


def normalise_landmarks(landmarks_list):
    pts = np.array(landmarks_list, dtype=np.float32).reshape(NUM_LANDMARKS, COORDS_PER_LM)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def extract_frame_features(pose_result, hand_result) -> np.ndarray:
    pose_vec = _normalise_pose(_lm_to_array(pose_result.pose_landmarks[0])) \
        if pose_result.pose_landmarks else np.zeros(POSE_N * COORDS, dtype=np.float32)

    left_hand = right_hand = None
    if hand_result.handedness:
        for i, hand in enumerate(hand_result.handedness):
            if hand[0].category_name == "Left":
                left_hand = hand_result.hand_landmarks[i]
            elif hand[0].category_name == "Right":
                right_hand = hand_result.hand_landmarks[i]

    lh_vec = _normalise_hand(_lm_to_array(left_hand)) \
        if left_hand is not None else np.zeros(HAND_N * COORDS, dtype=np.float32)
    rh_vec = _normalise_hand(_lm_to_array(right_hand)) \
        if right_hand is not None else np.zeros(HAND_N * COORDS, dtype=np.float32)
    return np.concatenate([pose_vec, lh_vec, rh_vec]).astype(np.float32)


# ── Drawing helpers ──────────────────────────────────────────────────────────

def draw_hand(frame, landmarks, color=(0, 255, 255)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 0), 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        c = (0, 0, 255) if i == 0 else color
        cv2.circle(frame, pt, 5, c, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_pose(frame, pose_landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
    for a, b in _POSE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2, cv2.LINE_AA)
    for idx in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        cv2.circle(frame, pts[idx], 4, (0, 255, 0), -1, cv2.LINE_AA)


# ── LetterSmoother ───────────────────────────────────────────────────────────

class LetterSmoother:
    def __init__(self):
        self.window: deque[str | None] = deque(maxlen=WINDOW_SIZE)
        self._prev_smoothed: str | None = None
        self._same_count = 0
        self._cooldown_remaining = 0

    def update(self, letter: str | None) -> str | None:
        self.window.append(letter)
        counts: dict[str, int] = {}
        for l in self.window:
            if l is not None:
                counts[l] = counts.get(l, 0) + 1
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


# ── Model loading (cached) ───────────────────────────────────────────────────

def _ensure_model(path: Path, url: str, name: str):
    if not path.exists():
        print(f"Downloading {name} model ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)


def load_fingerspell():
    """Load fingerspell model + hand detector. Returns (model, classes, device, detector) or None."""
    if not FS_MODEL_PATH.exists() or not FS_CLASSES_PATH.exists():
        return None
    _ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "hand")

    ckpt = torch.load(FS_MODEL_PATH, map_location="cpu")
    classes = np.load(FS_CLASSES_PATH, allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLClassifier(ckpt["input_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    print(f"Fingerspell model loaded ({len(classes)} classes, device={device})")
    return model, classes, device, detector


def load_wordsign():
    """Load wordsign model + pose/hand detectors. Returns (model, classes, device, seq_len, pose_det, hand_det) or None."""
    if not WS_MODEL_PATH.exists() or not WS_CLASSES_PATH.exists():
        return None
    _ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL, "pose")
    _ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "hand")

    ckpt = torch.load(WS_MODEL_PATH, map_location="cpu")
    classes = np.load(WS_CLASSES_PATH, allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WLASLModel(
        ckpt["feat_dim"], ckpt["num_classes"],
        ckpt["hidden"], ckpt["num_layers"], ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    seq_len = ckpt["seq_len"]

    pose_det = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    ))
    hand_det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    ))
    print(f"Word-sign model loaded ({len(classes)} classes, seq_len={seq_len}, device={device})")
    return model, classes, device, seq_len, pose_det, hand_det


# Load models at import time (cached by the process)
fs_assets = load_fingerspell()
ws_assets = load_wordsign()
spell = SpellChecker()


# ── Fingerspell processing ───────────────────────────────────────────────────

def _suggest_word(word_buffer: list[str]) -> str:
    raw = "".join(word_buffer).strip().lower()
    if not raw:
        return ""
    if not spell.unknown([raw]):
        return raw.upper()
    correction = spell.correction(raw)
    return correction.upper() if correction else raw.upper()


def process_fingerspell(frame, state):
    """Gradio streaming callback for fingerspell mode."""
    if frame is None or fs_assets is None:
        msg = "Fingerspell model not found." if fs_assets is None else ""
        return None, msg, state

    model, classes, device, detector = fs_assets
    smoother = state["smoother"]
    word_buffer = state["word_buffer"]

    # Gradio sends RGB frames; copy so we can draw on it
    frame = frame.copy()
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)

    current_letter = None
    confidence = 0.0

    if result.hand_landmarks:
        raw = []
        for lm in result.hand_landmarks[0]:
            raw.extend([lm.x, lm.y, lm.z])
        features = normalise_landmarks(raw)

        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        current_letter = str(classes[pred_idx])
        confidence = float(probs[pred_idx])

        if confidence < FS_CONFIDENCE_THRESHOLD:
            current_letter = None

        # Draw hand skeleton
        draw_hand(frame, result.hand_landmarks[0])

    # Filter special classes
    if current_letter in ("del", "nothing"):
        current_letter = None
    elif current_letter == "space":
        current_letter = " "

    # Temporal smoothing
    accepted = smoother.update(current_letter)

    if accepted is not None:
        if accepted == " ":
            if word_buffer and word_buffer[-1] != " ":
                word_buffer.append(" ")
        elif not word_buffer or word_buffer[-1] != accepted:
            word_buffer.append(accepted)

    # Draw big letter on frame
    if current_letter and current_letter != " ":
        h, w = frame.shape[:2]
        font_scale, thickness = 4.0, 6
        (tw, th), _ = cv2.getTextSize(current_letter, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        cx = (w - tw) // 2
        cv2.putText(frame, current_letter, (cx, th + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    # Build status text
    spelled = "".join(word_buffer)
    suggestion = _suggest_word(word_buffer)
    lines = [
        f"Raw: {current_letter or '---'} ({confidence*100:.0f}%)",
        f"Accepted: {accepted or '---'}",
        f"Spelled: {spelled}_",
    ]
    if suggestion:
        lines.append(f"Suggested word: {suggestion}")

    state["smoother"] = smoother
    state["word_buffer"] = word_buffer
    return frame, "\n".join(lines), state


def clear_fingerspell(state):
    """Reset the fingerspell word buffer."""
    state["word_buffer"] = []
    state["smoother"] = LetterSmoother()
    return "", state


# ── Word-sign processing ────────────────────────────────────────────────────

def process_wordsign(frame, state):
    """Gradio streaming callback for word-sign mode."""
    if frame is None or ws_assets is None:
        msg = "Word-sign model not found." if ws_assets is None else ""
        return None, msg, state

    model, classes, device, seq_len, pose_det, hand_det = ws_assets
    frame_buf = state["frame_buf"]
    frame_count = state["frame_count"]
    recent_words = state["recent_words"]
    last_word = state["last_word"]
    last_conf = state["last_conf"]
    conf_thresh = state.get("conf_thresh", 0.4)

    # Gradio sends RGB frames; copy so we can draw on it
    frame = frame.copy()
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=frame)
    pose_result = pose_det.detect(mp_image)
    hand_result = hand_det.detect(mp_image)

    # Draw landmarks
    if pose_result.pose_landmarks:
        draw_pose(frame, pose_result.pose_landmarks[0])
    if hand_result.hand_landmarks:
        for i, hand_lms in enumerate(hand_result.hand_landmarks):
            color = (255, 200, 0) if i == 0 else (0, 200, 255)
            draw_hand(frame, hand_lms, color)

    # Extract features and buffer
    feat = extract_frame_features(pose_result, hand_result)
    frame_buf.append(feat)
    frame_count += 1

    # Run inference when buffer is full and on stride
    if len(frame_buf) >= seq_len and frame_count % WS_STRIDE == 0:
        seq = np.stack(list(frame_buf)[-seq_len:], axis=0)
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
        length_t = torch.tensor([seq_len], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(seq_t, length_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if conf >= conf_thresh:
            word = str(classes[idx])
            if word != last_word:
                last_word = word
                last_conf = conf
                if not recent_words or recent_words[-1] != word:
                    recent_words.append(word)
        else:
            last_word = None
            last_conf = conf

    # Draw caption
    if recent_words:
        caption = "  ".join(list(recent_words)[-3:])
        h, w = frame.shape[:2]
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        (tw, th), baseline = cv2.getTextSize(caption, font, scale, thickness)
        cx = (w - tw) // 2
        cy = h - 20 - baseline
        cv2.rectangle(frame, (cx - 20, cy - th - 12),
                      (cx + tw + 20, cy + baseline + 12), (0, 0, 0), -1)
        cv2.putText(frame, caption, (cx, cy), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # Status text
    lines = [
        f"Current: {last_word or '---'} ({(last_conf or 0)*100:.0f}%)",
        f"Buffer: {len(frame_buf)}/{seq_len} frames",
        f"Recent: {', '.join(list(recent_words)[-5:]) if recent_words else '---'}",
    ]

    state["frame_buf"] = frame_buf
    state["frame_count"] = frame_count
    state["recent_words"] = recent_words
    state["last_word"] = last_word
    state["last_conf"] = last_conf
    return frame, "\n".join(lines), state


def clear_wordsign(state):
    """Reset word-sign state."""
    seq_len = ws_assets[3] if ws_assets else 64
    state["frame_buf"] = deque(maxlen=seq_len)
    state["frame_count"] = 0
    state["recent_words"] = deque(maxlen=10)
    state["last_word"] = None
    state["last_conf"] = 0.0
    return "", state


# ── Gradio UI ────────────────────────────────────────────────────────────────

def make_fs_state():
    return {"smoother": LetterSmoother(), "word_buffer": []}


def make_ws_state():
    seq_len = ws_assets[3] if ws_assets else 64
    return {
        "frame_buf": deque(maxlen=seq_len),
        "frame_count": 0,
        "recent_words": deque(maxlen=10),
        "last_word": None,
        "last_conf": 0.0,
    }


with gr.Blocks(title="ASL Recognition") as demo:
    gr.Markdown("# ASL Recognition\nReal-time American Sign Language recognition using your camera.")

    with gr.Tabs():
        # ── Fingerspell Tab ──
        with gr.TabItem("Fingerspell (A-Z)"):
            gr.Markdown(
                "**Fingerspelling mode**: Show ASL letters to the camera one at a time. "
                "The model recognises individual hand shapes and spells out words."
            )
            fs_state = gr.State(value=make_fs_state)

            with gr.Row():
                fs_webcam = gr.Image(sources=["webcam"], streaming=True, label="Camera")
                fs_output = gr.Image(label="Detection")

            fs_text = gr.Textbox(label="Predictions", lines=4, interactive=False)
            fs_clear_btn = gr.Button("Clear Word Buffer")

            fs_webcam.stream(
                fn=process_fingerspell,
                inputs=[fs_webcam, fs_state],
                outputs=[fs_output, fs_text, fs_state],
                stream_every=0.1,
            )
            fs_clear_btn.click(
                fn=clear_fingerspell,
                inputs=[fs_state],
                outputs=[fs_text, fs_state],
            )

        # ── Word Sign Tab ──
        with gr.TabItem("Word Sign"):
            if ws_assets is None:
                gr.Markdown(
                    "**Word-sign model not found.** Train the model first "
                    "with `python src/wlasl_train.py`."
                )
            else:
                gr.Markdown(
                    f"**Word-sign mode**: Perform ASL signs in front of the camera. "
                    f"The model analyses sequences of {ws_assets[3]} frames to recognise words."
                )
            ws_state = gr.State(value=make_ws_state)

            with gr.Row():
                ws_webcam = gr.Image(sources=["webcam"], streaming=True, label="Camera")
                ws_output = gr.Image(label="Detection")

            ws_text = gr.Textbox(label="Predictions", lines=3, interactive=False)
            ws_clear_btn = gr.Button("Clear History")

            ws_webcam.stream(
                fn=process_wordsign,
                inputs=[ws_webcam, ws_state],
                outputs=[ws_output, ws_text, ws_state],
                stream_every=0.1,
            )
            ws_clear_btn.click(
                fn=clear_wordsign,
                inputs=[ws_state],
                outputs=[ws_text, ws_state],
            )


if __name__ == "__main__":
    demo.launch()
