"""
app.py -- Gradio web app for ASL Recognition.

Two tabs:
  1. Fingerspell: real-time letter-by-letter recognition (A-Z)
  2. Word Sign:   sliding-window word-sign recognition (WLASL)

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
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode,
)
from spellchecker import SpellChecker


# ── Paths ─────────────────────────────────────────────────────────────────────

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


# ── Constants ──────────────────────────────────────────────────────────────────

POSE_N, HAND_N, COORDS = 33, 21, 3
NUM_LANDMARKS, COORDS_PER_LM = 21, 3

WINDOW_SIZE = 15
STABLE_COUNT = 10
COOLDOWN_FRAMES = 8
FS_CONFIDENCE_THRESHOLD = 0.50
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


# ── Model definitions ──────────────────────────────────────────────────────────

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


# ── Landmark helpers ───────────────────────────────────────────────────────────

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


def draw_pose(frame_bgr: np.ndarray, pose_landmarks):
    """Draw upper-body pose skeleton on a BGR frame."""
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
    pose_color = (0, 255, 80)  # bright lime green in BGR
    for a, b in _POSE_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], pose_color, 2, cv2.LINE_AA)
    for idx in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        cv2.circle(frame_bgr, pts[idx], 9, (255, 255, 255), 2, cv2.LINE_AA)  # white halo
        cv2.circle(frame_bgr, pts[idx], 7, pose_color, -1, cv2.LINE_AA)


def _annotate(frame_rgb: np.ndarray, hand_landmarks_list=None, pose_landmarks=None) -> np.ndarray:
    """Convert RGB frame to BGR, draw landmarks, return RGB result."""
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if pose_landmarks:
        draw_pose(out, pose_landmarks)
    if hand_landmarks_list:
        colors = [(0, 255, 255), (255, 0, 255)]  # bright yellow, bright magenta (BGR)
        for i, lms in enumerate(hand_landmarks_list):
            draw_hand(out, lms, colors[i % 2])
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _overlay_text(frame_rgb: np.ndarray, line1: str, line2: str = "") -> np.ndarray:
    """Draw a semi-transparent banner with status text on the bottom of a frame."""
    out = frame_rgb.copy()
    h, w = out.shape[:2]
    banner_h = 70 if line2 else 46
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    tw1, th1 = cv2.getTextSize(line1, font, scale, thick)[0]
    cv2.putText(out, line1, ((w - tw1) // 2, h - banner_h + 28),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    if line2:
        tw2, _ = cv2.getTextSize(line2, font, 0.55, 1)[0]
        cv2.putText(out, line2, ((w - tw2) // 2, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return out


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
    "align-items:center;justify-content:center;gap:14px;"
)


def _fs_caption_html(current_letter: str | None, confidence: float,
                     word_buffer: list, suggestion: str) -> str:
    spelled     = "".join(word_buffer)
    letter_html = (current_letter or "").strip() or "&nbsp;"
    conf_html   = f"{confidence*100:.0f}%" if (current_letter or "").strip() else ""
    sugg_html   = f"Suggestion:&nbsp;<b>{suggestion}</b>" if suggestion else "&nbsp;"
    return (
        f'<div style="{_CAPTION_BASE}font-family:\'Courier New\',monospace;">'
        f'<div style="color:#fff;font-size:80px;font-weight:bold;line-height:1;'
        f'min-height:88px;text-align:center;">{letter_html}</div>'
        f'<div style="color:#888;font-size:15px;min-height:20px;">{conf_html}</div>'
        f'<div style="color:#fff;font-size:30px;letter-spacing:8px;'
        f'min-height:44px;text-align:center;">{spelled}_</div>'
        f'<div style="color:#6af;font-size:17px;font-style:italic;">{sugg_html}</div>'
        f'</div>'
    )


def _ws_caption_html(recent_words, last_word: str | None, last_conf: float,
                     buf_len: int, seq_len: int) -> str:
    words_html = "&nbsp;&nbsp;".join(str(w) for w in list(recent_words)[-3:]) \
        if recent_words else "&nbsp;"

    pct = min(buf_len / seq_len, 1.0)

    if last_word:
        status_html = (
            f'<div style="color:#aaa;font-size:15px;">'
            f'{last_word}&nbsp;&nbsp;·&nbsp;&nbsp;{last_conf*100:.0f}%</div>'
        )
    elif buf_len >= seq_len:
        status_html = '<div style="color:#555;font-size:15px;">Detecting signs…</div>'
    elif buf_len > 0:
        status_html = '<div style="color:#555;font-size:15px;">Starting up…</div>'
    else:
        status_html = '<div style="color:#333;font-size:15px;">Enable camera to begin</div>'

    # Thin progress bar — only shown while the buffer is filling
    if 0 < buf_len < seq_len:
        bar_html = (
            f'<div style="width:220px;height:3px;background:#1a1a1a;border-radius:2px;">'
            f'<div style="width:{pct*100:.0f}%;height:100%;'
            f'background:#444;border-radius:2px;transition:width 0.2s;"></div>'
            f'</div>'
        )
    else:
        bar_html = ""

    return (
        f'<div style="{_CAPTION_BASE}font-family:Arial,sans-serif;">'
        f'<div style="color:#fff;font-size:42px;font-weight:bold;line-height:1.3;'
        f'text-align:center;min-height:56px;">{words_html}</div>'
        f'{status_html}'
        f'{bar_html}'
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


def load_wordsign():
    if not WS_MODEL_PATH.exists() or not WS_CLASSES_PATH.exists():
        return None
    _ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL, "pose")
    _ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "hand")

    ckpt    = torch.load(WS_MODEL_PATH, map_location="cpu", weights_only=False)
    classes = np.load(WS_CLASSES_PATH, allow_pickle=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = WLASLModel(
        ckpt["feat_dim"], ckpt["num_classes"],
        ckpt["hidden"], ckpt["num_layers"], ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    seq_len = ckpt["seq_len"]

    pose_det = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    ))
    hand_det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    ))
    print(f"Word-sign model loaded ({len(classes)} classes, seq_len={seq_len}, device={device})")
    return model, classes, device, seq_len, pose_det, hand_det


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


def _warmup_wordsign(assets):
    """Run one dummy inference to pre-compile kernels and avoid first-frame lag."""
    if assets is None:
        return
    model, classes, device, seq_len, pose_det, hand_det = assets
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    mp_img = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=dummy)
    pose_det.detect(mp_img)
    hand_det.detect(mp_img)
    feat_dim = model.input_proj[0].in_features
    dummy_seq = torch.zeros((1, seq_len, feat_dim), device=device)
    dummy_len = torch.tensor([seq_len], dtype=torch.long, device=device)
    with torch.no_grad():
        model(dummy_seq, dummy_len)
    print("Word-sign warmup complete")


# Load models at import time (cached by the process)
fs_assets = load_fingerspell()
ws_assets = load_wordsign()
spell     = SpellChecker()

# Warm up all models so the first real frame has no compilation delay
_warmup_fingerspell(fs_assets)
_warmup_wordsign(ws_assets)


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
_WS_PLACEHOLDER = _make_placeholder_image(
    "Enable camera to begin", "Perform ASL signs in front of the camera")


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


# ── Word-sign processing ───────────────────────────────────────────────────────

def process_wordsign(frame, state):
    if frame is None or ws_assets is None:
        return _WS_PLACEHOLDER, _ws_caption_html([], None, 0.0, 0, 64), state

    model, classes, device, seq_len, pose_det, hand_det = ws_assets
    frame_buf    = state["frame_buf"]
    frame_count  = state["frame_count"]
    recent_words = state["recent_words"]
    last_word    = state["last_word"]
    last_conf    = state["last_conf"]
    conf_thresh  = state.get("conf_thresh", 0.5)

    mp_image    = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=frame)
    pose_result = pose_det.detect(mp_image)
    hand_result = hand_det.detect(mp_image)

    feat = extract_frame_features(pose_result, hand_result)
    frame_buf.append(feat)
    frame_count += 1

    if len(frame_buf) >= seq_len and frame_count % WS_STRIDE == 0:
        seq      = np.stack(list(frame_buf)[-seq_len:], axis=0)
        seq_t    = torch.from_numpy(seq).unsqueeze(0).to(device)
        length_t = torch.tensor([seq_len], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(seq_t, length_t)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx  = int(np.argmax(probs))
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

    state["frame_buf"]    = frame_buf
    state["frame_count"]  = frame_count
    state["recent_words"] = recent_words
    state["last_word"]    = last_word
    state["last_conf"]    = last_conf

    annotated = _annotate(
        frame,
        hand_landmarks_list=hand_result.hand_landmarks if hand_result.hand_landmarks else None,
        pose_landmarks=pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None,
    )

    buf_now = len(frame_buf)
    if buf_now < seq_len:
        pct = int(buf_now / seq_len * 100)
        annotated = _overlay_text(
            annotated,
            f"Buffering: {buf_now} / {seq_len} frames  ({pct}%)",
            "Please wait — detection starts once buffer is full",
        )

    caption = _ws_caption_html(recent_words, last_word, last_conf, buf_now, seq_len)
    return annotated, caption, state


def clear_wordsign(state):
    seq_len = ws_assets[3] if ws_assets else 64
    state["frame_buf"]    = deque(maxlen=seq_len)
    state["frame_count"]  = 0
    state["recent_words"] = deque(maxlen=10)
    state["last_word"]    = None
    state["last_conf"]    = 0.0
    return _ws_caption_html([], None, 0.0, 0, seq_len), state


# ── Gradio state factories ─────────────────────────────────────────────────────

def make_fs_state():
    return {"smoother": LetterSmoother(), "word_buffer": []}


def make_ws_state():
    seq_len = ws_assets[3] if ws_assets else 64
    return {
        "frame_buf":    deque(maxlen=seq_len),
        "frame_count":  0,
        "recent_words": deque(maxlen=10),
        "last_word":    None,
        "last_conf":    0.0,
    }


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="ASL Recognition") as demo:
    gr.Markdown("# ASL Recognition\nReal-time American Sign Language recognition using your camera.")

    with gr.Tabs():

        # ── Fingerspell Tab ────────────────────────────────────────────────────
        with gr.TabItem("Fingerspell (A-Z)"):
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

        # ── Word Sign Tab ──────────────────────────────────────────────────────
        with gr.TabItem("Word Sign"):
            if ws_assets is None:
                gr.Markdown(
                    "**Word-sign model not found.** "
                    "Train the model first with `python src/wlasl_train.py`."
                )
            else:
                gr.Markdown(
                    f"**Word-sign mode**: Perform ASL signs in front of the camera. "
                    f"The model analyses {ws_assets[3]}-frame sequences to recognise words."
                )
            ws_state = gr.State(value=make_ws_state)

            with gr.Row():
                with gr.Column(scale=1, min_width=180):
                    ws_webcam = gr.Image(sources=["webcam"], streaming=True,
                                         label="Enable Camera", mirror_webcam=False,
                                         height=180)
                with gr.Column(scale=3):
                    ws_output = gr.Image(label="Live Feed", mirror_webcam=False)

            ws_caption   = gr.HTML(value=_ws_caption_html([], None, 0.0, 0,
                                                          ws_assets[3] if ws_assets else 64))
            ws_clear_btn = gr.Button("Clear")

            ws_webcam.stream(
                fn=process_wordsign,
                inputs=[ws_webcam, ws_state],
                outputs=[ws_output, ws_caption, ws_state],
                stream_every=0.1,
            )
            ws_clear_btn.click(
                fn=clear_wordsign,
                inputs=[ws_state],
                outputs=[ws_caption, ws_state],
            )


if __name__ == "__main__":
    demo.launch()
