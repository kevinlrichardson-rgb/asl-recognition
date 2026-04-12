"""
infer.py — Unified ASL inference script.

Supports two modes via --mode:
  fingerspell : Real-time ASL fingerspelling (letter-by-letter) recognition
  wordsign    : Sliding-window word-sign recognition (WLASL LSTM model)

Usage:
    # Fingerspelling from webcam
    python src/infer.py --mode fingerspell --webcam

    # Fingerspelling from video file
    python src/infer.py --mode fingerspell --video path/to/clip.mp4

    # Word-sign from video file (headless)
    python src/infer.py --mode wordsign --video path/to/clip.mp4 --headless

    # Save annotated output
    python src/infer.py --mode wordsign --video clip.mp4 --output out.mp4

Controls (GUI mode):
    q / ESC   quit
    BACKSPACE delete last character  (fingerspell only)
    c         clear word buffer      (fingerspell only)
"""

import argparse
import collections
import csv
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode,
)


# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

# Fingerspell model
FS_MODEL_PATH   = ROOT / "models" / "asl_model.pt"
FS_CLASSES_PATH = ROOT / "models" / "label_classes.npy"

# Word-sign model
WS_MODEL_PATH   = ROOT / "models" / "wlasl_word_model.pt"
WS_CLASSES_PATH = ROOT / "models" / "wlasl_classes.npy"

# MediaPipe task files
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

HEADLESS_OUT_DIR = ROOT / "data" / "asl_alphabet_Processed_Folder"


# ── Constants ────────────────────────────────────────────────────────────────

POSE_N = 33
HAND_N = 21
COORDS = 3
FEAT_DIM = (POSE_N + HAND_N + HAND_N) * COORDS   # 225

NUM_LANDMARKS = 21
COORDS_PER_LM = 3

# Fingerspell smoothing / stabilisation
WINDOW_SIZE = 15
STABLE_COUNT = 10
COOLDOWN_FRAMES = 8
CONFIDENCE_THRESHOLD = 0.55


# ── Model definitions ────────────────────────────────────────────────────────

class ASLClassifier(nn.Module):
    """Feedforward classifier for single-frame fingerspelling (A-Z)."""
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
    """BiLSTM + attention model for sequence-based word-sign recognition."""
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = AttentionPool(hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
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
    """Normalise a flat list of 63 hand-landmark floats (wrist-centred, unit-scaled)."""
    pts = np.array(landmarks_list, dtype=np.float32).reshape(NUM_LANDMARKS, COORDS_PER_LM)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def extract_frame_features(pose_result, hand_result) -> np.ndarray:
    """Build 225-float feature vector from separate pose and hand results."""
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


# ── Drawing constants & helpers ──────────────────────────────────────────────

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

_POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
]


def _draw_hand_landmarks_simple(frame, landmarks):
    """Draw hand skeleton overlay (for fingerspell mode)."""
    if not landmarks:
        return
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 0), 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        color = (0, 0, 255) if i == 0 else (0, 255, 255)
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_hand_landmarks(frame, hand_landmarks, color=(0, 255, 255)):
    """Draw hand skeleton (for wordsign mode, with configurable colour)."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 0), 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        c = (0, 0, 255) if i == 0 else color
        cv2.circle(frame, pt, 4, c, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_pose_landmarks(frame, pose_landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
    for a, b in _POSE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2, cv2.LINE_AA)
    for idx in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        cv2.circle(frame, pts[idx], 4, (0, 255, 0), -1, cv2.LINE_AA)


def _draw_ws_landmarks(frame, pose_result, hand_result):
    """Draw all detected pose and hand landmarks (wordsign mode)."""
    if pose_result.pose_landmarks:
        _draw_pose_landmarks(frame, pose_result.pose_landmarks[0])
    if hand_result.hand_landmarks:
        for i, hand_lms in enumerate(hand_result.hand_landmarks):
            color = (255, 200, 0) if i == 0 else (0, 200, 255)
            _draw_hand_landmarks(frame, hand_lms, color)


def _draw_caption(frame, recent_words):
    """Draw rolling caption: up to 3 recent words, centred at the bottom."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2

    label = "  ".join(recent_words) if recent_words else "---"
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad_x, pad_y = 20, 12
    cx = (w - tw) // 2
    cy = h - 20 - baseline
    cv2.rectangle(frame,
                  (cx - pad_x, cy - th - pad_y),
                  (cx + tw + pad_x, cy + baseline + pad_y),
                  (0, 0, 0), -1)
    cv2.putText(frame, label, (cx, cy),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _has_display() -> bool:
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _ensure_model(path: Path, url: str, name: str):
    if not path.exists():
        print(f"Downloading {name} model ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)


# ══════════════════════════════════════════════════════════════════════════════
#  FINGERSPELL MODE
# ══════════════════════════════════════════════════════════════════════════════

class LetterSmoother:
    """Majority-vote smoothing + stability gate to reduce jitter."""

    def __init__(self, window_size=WINDOW_SIZE, stable_count=STABLE_COUNT,
                 cooldown=COOLDOWN_FRAMES):
        self.window: deque[str | None] = deque(maxlen=window_size)
        self.stable_count = stable_count
        self.cooldown = cooldown

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

        if self._same_count >= self.stable_count:
            self._same_count = 0
            self._cooldown_remaining = self.cooldown
            return smoothed

        return None


def _fs_extract_features(img_rgb_np, detector, timestamp_ms):
    """Extract normalised landmark vector using VIDEO running mode.
    Returns (feature_vector, landmark_list) or (None, None)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_np)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None, None
    raw = []
    for lm in result.hand_landmarks[0]:
        raw.extend([lm.x, lm.y, lm.z])
    return normalise_landmarks(raw), result.hand_landmarks[0]


def _fs_predict_frame(features, model, classes, device):
    """Return (predicted_class, confidence) from a feature vector."""
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], float(probs[pred_idx])


def _fs_draw_hud(frame, current_letter, confidence, word_buffer, smoothed_letter,
                 suggested_word=None):
    """Overlay fingerspell HUD on the video frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 150), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    raw_text = f"Raw: {current_letter or '---'}  ({confidence * 100:.0f}%)"
    cv2.putText(frame, raw_text, (15, h - 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    smooth_text = f"Accepted: {smoothed_letter or '---'}"
    cv2.putText(frame, smooth_text, (15, h - 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)

    word_text = f"Spelled: {''.join(word_buffer)}_"
    cv2.putText(frame, word_text, (15, h - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if suggested_word:
        sugg_text = f"Word: {suggested_word}"
        cv2.putText(frame, sugg_text, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    if current_letter:
        font_scale = 4.0
        thickness = 6
        (tw, th), _ = cv2.getTextSize(current_letter, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        cx = (w - tw) // 2
        cv2.putText(frame, current_letter, (cx, th + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    cv2.putText(frame, "q:quit  c:clear  bksp:del", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def _fs_process_frame(frame, detector, model, classes, device, smoother,
                      word_buffer, timestamp_ms):
    """Run detection + prediction + smoothing on one frame.
    Returns (current_letter, confidence, accepted_letter)."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    features, landmarks = _fs_extract_features(img_rgb, detector, timestamp_ms)
    current_letter = None
    confidence = 0.0

    if features is not None:
        current_letter, confidence = _fs_predict_frame(features, model, classes, device)
        if confidence < CONFIDENCE_THRESHOLD:
            current_letter = None

    if landmarks:
        _draw_hand_landmarks_simple(frame, landmarks)

    if current_letter in ("del", "nothing"):
        current_letter = None
    elif current_letter == "space":
        current_letter = " "

    accepted = smoother.update(current_letter)

    if accepted is not None:
        if accepted == " ":
            if word_buffer and word_buffer[-1] != " ":
                word_buffer.append(" ")
                print(f"  [SPACE]  ->  {''.join(word_buffer).strip()}")
        elif not word_buffer or word_buffer[-1] != accepted:
            word_buffer.append(accepted)
            print(f"  + {accepted}  ->  {''.join(word_buffer)}")
        else:
            accepted = None

    return current_letter, confidence, accepted


def _fs_suggest_word(spell, word_buffer: list[str]) -> str | None:
    raw = "".join(word_buffer).strip().lower()
    if not raw:
        return None
    if not spell.unknown([raw]):
        return raw.upper()
    correction = spell.correction(raw)
    return correction.upper() if correction else raw.upper()


def _run_fingerspell(source, headless=False, output_path=None):
    """Full fingerspell inference loop."""
    from spellchecker import SpellChecker

    # Validate model files
    for path, label in [(FS_MODEL_PATH, "fingerspell model"),
                        (FS_CLASSES_PATH, "fingerspell classes"),
                        (HAND_MODEL_PATH, "hand landmarker")]:
        if not path.exists():
            sys.exit(f"[ERROR] {label} not found: {path}\n"
                     "Run extract_landmarks and train first.")

    # Load model
    checkpoint = torch.load(FS_MODEL_PATH, map_location="cpu")
    classes = np.load(FS_CLASSES_PATH, allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ASLClassifier(checkpoint["input_dim"], checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Fingerspell model loaded ({len(classes)} classes, device={device})")

    # MediaPipe hand detector (VIDEO mode, 1 hand)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    # Video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        detector.close()
        sys.exit(f"[ERROR] Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            sys.exit(f"[ERROR] Cannot create output video: {output_path}")
        print(f"Writing annotated video to: {output_path}")

    smoother = LetterSmoother()
    word_buffer: list[str] = []
    spell = SpellChecker()
    current_suggestion: str | None = None

    frame_idx = 0
    mode_label = "headless" if headless else "GUI"
    if output_path:
        mode_label += " + saving"
    print(f"\n=== Fingerspelling session started ({mode_label}) ===")
    if headless:
        print(f"Processing {'webcam' if isinstance(source, int) else source}  "
              f"({total_frames} frames @ {fps:.0f} fps)" if total_frames > 0
              else f"Processing webcam stream @ {fps:.0f} fps")
    else:
        print("Show ASL letters to the camera. Press 'q' to quit.")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)
            current_letter, confidence, accepted = _fs_process_frame(
                frame, detector, model, classes, device, smoother,
                word_buffer, timestamp_ms,
            )

            if accepted is not None:
                current_suggestion = _fs_suggest_word(spell, word_buffer)
                if current_suggestion:
                    print(f"  >> Suggested word: {current_suggestion}")

            if not headless or writer:
                _fs_draw_hud(frame, current_letter, confidence, word_buffer, accepted,
                             suggested_word=current_suggestion)

            if writer:
                writer.write(frame)

            if not headless:
                cv2.imshow("ASL Fingerspelling", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == 8 and word_buffer:
                    removed = word_buffer.pop()
                    print(f"  [BACKSPACE] removed '{removed}'  ->  {''.join(word_buffer)}")
                elif key == ord("c"):
                    word_buffer.clear()
                    current_suggestion = None
                    print("  [CLEAR]")

            if headless and total_frames > 0 and frame_idx % 100 == 0:
                pct = frame_idx / total_frames * 100
                print(f"  ... frame {frame_idx}/{total_frames} ({pct:.0f}%)")

            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\nAnnotated video saved to: {os.path.abspath(output_path)}")
        if not headless:
            cv2.destroyAllWindows()
        detector.close()

    final_word = "".join(word_buffer).strip()
    final_suggestion = _fs_suggest_word(spell, word_buffer) if final_word else None
    print(f"\n=== Session ended ({frame_idx} frames processed) ===")
    print(f"Spelled letters: {final_word if final_word else '(empty)'}")
    if final_suggestion and final_suggestion != final_word.upper():
        print(f"Recognised word:  {final_suggestion}")
    elif final_suggestion:
        print(f"Recognised word:  {final_suggestion}")
    return final_word


# ══════════════════════════════════════════════════════════════════════════════
#  WORDSIGN MODE
# ══════════════════════════════════════════════════════════════════════════════

class WordSignRecogniser:
    """Stateful sliding-window word-sign recogniser for live video."""

    def __init__(self, confidence_threshold: float = 0.4, stride: int = 8):
        self._ready = self._load_model()
        if not self._ready:
            return

        self._pose_landmarker, self._hand_landmarker = self._init_landmarkers()
        self._frame_buf: deque[np.ndarray] = deque(maxlen=self._seq_len)
        self._frame_count = 0
        self._stride = stride
        self._conf_thresh = confidence_threshold
        self._last_word: str | None = None
        self._last_conf: float = 0.0

    def _load_model(self) -> bool:
        if not WS_MODEL_PATH.exists() or not WS_CLASSES_PATH.exists():
            return False
        try:
            ckpt = torch.load(WS_MODEL_PATH, map_location="cpu")
            self._classes = np.load(WS_CLASSES_PATH, allow_pickle=True)
            self._device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model   = WLASLModel(
                ckpt["feat_dim"], ckpt["num_classes"],
                ckpt["hidden"], ckpt["num_layers"], ckpt["dropout"],
            ).to(self._device)
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._model.eval()
            self._seq_len = ckpt["seq_len"]
            return True
        except Exception as exc:
            print(f"[WordSignRecogniser] Failed to load model: {exc}")
            return False

    def _init_landmarkers(self):
        _ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL, "pose")
        _ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "hand")

        pose_opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        hand_opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return (
            PoseLandmarker.create_from_options(pose_opts),
            HandLandmarker.create_from_options(hand_opts),
        )

    def is_ready(self) -> bool:
        return self._ready

    def update(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple[str | None, float]:
        """Feed one BGR frame. Returns (word, confidence) or (None, 0.0)."""
        if not self._ready:
            return None, 0.0

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = self._pose_landmarker.detect_for_video(mp_img, timestamp_ms)
        hand_result = self._hand_landmarker.detect_for_video(mp_img, timestamp_ms)
        self.last_pose_result = pose_result
        self.last_hand_result = hand_result
        feat = extract_frame_features(pose_result, hand_result)
        self._frame_buf.append(feat)
        self._frame_count += 1

        _MIN_FRAMES = 16  # start predicting early; zero-pad to seq_len
        avail = len(self._frame_buf)
        if avail < _MIN_FRAMES or self._frame_count % self._stride != 0:
            return self._last_word, self._last_conf

        seq_list = list(self._frame_buf)
        if avail < self._seq_len:
            pad = [np.zeros(225, dtype=np.float32)] * (self._seq_len - avail)
            seq_list = pad + seq_list  # prepend zeros; AttentionPool masks them out
        seq = np.stack(seq_list, axis=0)
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(self._device)
        length_t = torch.tensor([min(avail, self._seq_len)], dtype=torch.long,
                                 device=self._device)
        # Higher threshold during warmup to suppress noisy early predictions
        eff_thresh = 0.75 if avail < self._seq_len else self._conf_thresh

        with torch.no_grad():
            logits = self._model(seq_t, length_t)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        word = str(self._classes[idx]) if conf >= eff_thresh else None

        self._last_word = word
        self._last_conf = conf
        return word, conf

    def close(self):
        if self._ready:
            self._pose_landmarker.close()
            self._hand_landmarker.close()


def _run_wordsign(source, headless=False, output_path=None, conf_threshold=0.4):
    """Full word-sign inference loop."""
    rec = WordSignRecogniser(confidence_threshold=conf_threshold)
    if not rec.is_ready():
        sys.exit("Word-sign model not ready. Train the model first (wlasl_train.py).")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        rec.close()
        sys.exit(f"[ERROR] Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    rows = []
    recent_words: deque[str] = deque(maxlen=3)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            cap.release()
            rec.close()
            sys.exit(f"[ERROR] Cannot create output video: {output_path}")
        print(f"Writing annotated video to: {output_path}")

    MIN_CONFIRM = 5
    pending_pred = None
    pending_count = 0

    mode_label = "headless" if headless else "GUI"
    if output_path:
        mode_label += " + saving"
    print(f"\n=== Word-sign session started ({mode_label}) ===")
    if headless:
        print(f"Processing {'webcam' if isinstance(source, int) else source}  "
              f"({total_frames} frames @ {fps:.0f} fps)" if total_frames > 0
              else f"Processing webcam stream @ {fps:.0f} fps")
    else:
        print("Show ASL signs to the camera. Press 'q' to quit.")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms = int(frame_idx * 1000 / fps)
            word, word_conf = rec.update(frame, ts_ms)

            # Confirmation gate
            accepted = None
            if word is not None:
                if pending_pred and pending_pred[0] == word:
                    pending_count += 1
                else:
                    pending_pred = (word, word_conf)
                    pending_count = 1

                if pending_count >= MIN_CONFIRM:
                    accepted = word
                    accepted_conf = word_conf
                    pending_count = 0

            if accepted is not None:
                rows.append((frame_idx, ts_ms, accepted, f"{accepted_conf:.4f}"))
                if not recent_words or recent_words[-1] != accepted:
                    recent_words.append(accepted)
                if headless:
                    print(f"  frame {frame_idx:5d}  {ts_ms:7d} ms  "
                          f"{accepted}  ({accepted_conf:.2f})")

            if writer or not headless:
                _draw_ws_landmarks(frame, rec.last_pose_result, rec.last_hand_result)
                _draw_caption(frame, recent_words)

            if writer:
                writer.write(frame)

            if not headless:
                cv2.imshow("WLASL Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if headless and total_frames > 0 and frame_idx % 200 == 0:
                pct = frame_idx / total_frames * 100
                print(f"  ... frame {frame_idx}/{total_frames} ({pct:.0f}%)")

            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
        rec.close()

    # Save CSV
    if headless or output_path:
        HEADLESS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem if isinstance(source, str) else "webcam"
        csv_path = HEADLESS_OUT_DIR / f"{stem}_predictions.csv"
        with open(csv_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["frame", "timestamp_ms", "word", "confidence"])
            csv_writer.writerows(rows)
        print(f"\nTotal frames processed : {frame_idx}")
        print(f"Predictions recorded   : {len(rows)}")
        print(f"CSV saved -> {csv_path}")
        if output_path:
            print(f"Video saved -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified ASL Inference — fingerspell or word-sign mode",
    )
    parser.add_argument("--mode", required=True, choices=["fingerspell", "wordsign"],
                        help="Inference mode: 'fingerspell' for letter recognition, "
                             "'wordsign' for WLASL word-sign recognition")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--webcam", action="store_true",
                       help="Use live webcam feed")
    group.add_argument("--video", type=str, metavar="PATH",
                       help="Path to a video file")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0, used with --webcam)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI window (terminal output only)")
    parser.add_argument("--output", "-o", type=str, metavar="PATH",
                        help="Save annotated video to this file (e.g. output.mp4)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4, wordsign only)")
    args = parser.parse_args()

    if args.video and not os.path.isfile(args.video):
        sys.exit(f"[ERROR] Video file not found: {args.video}")

    headless = args.headless or not _has_display()
    if headless and not args.headless:
        print("[INFO] No display detected - running in headless mode.")

    source = args.camera if args.webcam else args.video

    if args.mode == "fingerspell":
        _run_fingerspell(source, headless=headless, output_path=args.output)
    else:
        _run_wordsign(source, headless=headless, output_path=args.output,
                      conf_threshold=args.conf)


if __name__ == "__main__":
    main()
