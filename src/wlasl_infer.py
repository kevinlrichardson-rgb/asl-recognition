"""
wlasl_infer.py — Sliding-window word-sign inference using the trained WLASL LSTM model.

Provides WordSignRecogniser: a stateful class that accumulates frames from a
live video stream and returns a (word, confidence) prediction whenever a full
window of frames is ready.

Usage (standalone test):
    python src/wlasl_infer.py --video path/to/video.mp4
"""

import argparse
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = ROOT / "models" / "wlasl_word_model.pt"
CLASSES_PATH = ROOT / "models" / "wlasl_classes.npy"
HOLISTIC_MODEL = ROOT / "models" / "holistic_landmarker.task"

HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)

POSE_N = 33
HAND_N = 21
COORDS = 3
FEAT_DIM = (POSE_N + HAND_N + HAND_N) * COORDS   # 225


# ── Landmark helpers (mirrored from wlasl_extract.py) ─────────────────────────

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


def extract_frame_features(result) -> np.ndarray:
    pose_vec = _normalise_pose(_lm_to_array(result.pose_landmarks)) \
        if result.pose_landmarks else np.zeros(POSE_N * COORDS, dtype=np.float32)
    lh_vec = _normalise_hand(_lm_to_array(result.left_hand_landmarks)) \
        if result.left_hand_landmarks else np.zeros(HAND_N * COORDS, dtype=np.float32)
    rh_vec = _normalise_hand(_lm_to_array(result.right_hand_landmarks)) \
        if result.right_hand_landmarks else np.zeros(HAND_N * COORDS, dtype=np.float32)
    return np.concatenate([pose_vec, lh_vec, rh_vec]).astype(np.float32)


# ── Model (must match wlasl_train.py) ─────────────────────────────────────────

class AttentionPool(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class WLASLModel(torch.nn.Module):
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.lstm = torch.nn.LSTM(
            hidden, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = AttentionPool(hidden * 2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.attn(x, lengths)
        return self.classifier(x)


# ── Recogniser ────────────────────────────────────────────────────────────────

class WordSignRecogniser:
    """
    Stateful sliding-window word-sign recogniser for live video.

    Usage:
        rec = WordSignRecogniser()
        if rec.is_ready():
            for frame_bgr in video:
                word, conf = rec.update(frame_bgr, timestamp_ms)
    """

    def __init__(self, confidence_threshold: float = 0.4, stride: int = 8):
        """
        Args:
            confidence_threshold: Minimum softmax probability to report a word.
            stride: Run inference every `stride` new frames (reduces CPU load).
        """
        self._ready = self._load_model()
        if not self._ready:
            return

        self._holistic = self._init_holistic()
        self._frame_buf: deque[np.ndarray] = deque(maxlen=self._seq_len)
        self._frame_count = 0
        self._stride = stride
        self._conf_thresh = confidence_threshold
        self._last_word: str | None = None
        self._last_conf: float = 0.0
        self._ts_ms = 0

    def _load_model(self) -> bool:
        if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
            return False
        try:
            ckpt = torch.load(MODEL_PATH, map_location="cpu")
            self._classes = np.load(CLASSES_PATH, allow_pickle=True)
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

    def _init_holistic(self):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HolisticLandmarker, HolisticLandmarkerOptions, RunningMode,
        )
        if not HOLISTIC_MODEL.exists():
            print("[WordSignRecogniser] Downloading holistic model …")
            HOLISTIC_MODEL.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HOLISTIC_MODEL_URL, HOLISTIC_MODEL)

        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HOLISTIC_MODEL)),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
            min_pose_landmarks_confidence=0.3,
            min_hand_landmarks_confidence=0.3,
        )
        return HolisticLandmarker.create_from_options(options)

    def is_ready(self) -> bool:
        """Returns True if the model loaded successfully and inference is available."""
        return self._ready

    def update(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple[str | None, float]:
        """
        Feed one BGR video frame. Returns (word, confidence) or (None, 0.0).

        A non-None word is returned only when confidence >= threshold and
        inference ran on this frame (every `stride` frames once buffer is full).
        """
        if not self._ready:
            return None, 0.0

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._holistic.detect_for_video(mp_img, timestamp_ms)
        feat = extract_frame_features(result)
        self._frame_buf.append(feat)
        self._frame_count += 1

        # Only run inference when buffer is full and on stride
        if (len(self._frame_buf) < self._seq_len or
                self._frame_count % self._stride != 0):
            return self._last_word, self._last_conf

        seq = np.stack(self._frame_buf, axis=0)   # (T, F)
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(self._device)   # (1, T, F)
        length_t = torch.tensor([self._seq_len], dtype=torch.long, device=self._device)

        with torch.no_grad():
            logits = self._model(seq_t, length_t)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        word = str(self._classes[idx]) if conf >= self._conf_thresh else None

        self._last_word = word
        self._last_conf = conf
        return word, conf

    def close(self):
        if self._ready and self._holistic:
            self._holistic.close()


# ── Standalone test ───────────────────────────────────────────────────────────

def _test_video(video_path: str):
    rec = WordSignRecogniser()
    if not rec.is_ready():
        print("Model not ready. Train the model first (wlasl_train.py).")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms = int(frame_idx * 1000 / fps)
            word, conf = rec.update(frame, ts_ms)
            label = f"{word}  ({conf:.2f})" if word else f"--- ({conf:.2f})"
            cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 2)
            cv2.imshow("WLASL Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rec.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test WLASL word-sign inference on a video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    args = parser.parse_args()
    _test_video(args.video)
