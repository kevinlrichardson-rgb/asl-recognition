#!/usr/local/bin/python3.11
"""
fingerspell.py – Phase 2: Fingerspelling Word Recognition
---------------------------------------------------------
Classifies ASL fingerspelled letters frame-by-frame from a webcam or video
file, applies temporal smoothing to reduce jitter, and concatenates stable
letter predictions into spelled words.

Usage:
    # Webcam (default camera index 0)
    python src/fingerspell.py --webcam

    # Webcam with a specific camera index
    python src/fingerspell.py --webcam --camera 1

    # Video file
    python src/fingerspell.py --video path/to/clip.mp4

    # Force headless mode (no GUI window, terminal output only)
    python src/fingerspell.py --video clip.mp4 --headless

    # Save annotated video with HUD overlay to a file
    python src/fingerspell.py --video clip.mp4 --output output.mp4

Controls (GUI mode, while the window is open):
    q / ESC   – quit
    BACKSPACE – delete last character
    c         – clear the current word buffer
"""

import argparse
import collections
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_model.pt")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "label_classes.npy")
LANDMARK_MODEL = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

NUM_LANDMARKS = 21
COORDS_PER_LM = 3

# ---------------------------------------------------------------------------
# Smoothing / stabilisation parameters
# ---------------------------------------------------------------------------
WINDOW_SIZE = 15          # frames for majority-vote smoothing
STABLE_COUNT = 10         # consecutive identical smoothed predictions to accept
COOLDOWN_FRAMES = 8       # minimum frames after accepting a letter before next
CONFIDENCE_THRESHOLD = 0.55  # ignore predictions below this confidence


# ---------------------------------------------------------------------------
# Model definition (must match train.ipynb / demo.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Landmark helpers (mirror of extract_landmarks / demo.py)
# ---------------------------------------------------------------------------
def normalise_landmarks(landmarks_list):
    pts = np.array(landmarks_list, dtype=np.float32).reshape(NUM_LANDMARKS, COORDS_PER_LM)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def extract_features_from_frame(img_rgb_np, detector, timestamp_ms):
    """Extract normalised landmark vector using VIDEO running mode."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_np)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    if not result.hand_landmarks:
        return None
    raw = []
    for lm in result.hand_landmarks[0]:
        raw.extend([lm.x, lm.y, lm.z])
    return normalise_landmarks(raw)


def predict_frame(features, model, classes, device):
    """Return (predicted_class, confidence) from a feature vector."""
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], float(probs[pred_idx])


# ---------------------------------------------------------------------------
# Temporal smoothing helper
# ---------------------------------------------------------------------------
class LetterSmoother:
    """Majority-vote smoothing + stability gate to reduce jitter."""

    def __init__(self, window_size=WINDOW_SIZE, stable_count=STABLE_COUNT,
                 cooldown=COOLDOWN_FRAMES):
        self.window: collections.deque[str | None] = collections.deque(maxlen=window_size)
        self.stable_count = stable_count
        self.cooldown = cooldown

        self._prev_smoothed: str | None = None
        self._same_count = 0
        self._cooldown_remaining = 0

    def update(self, letter: str | None) -> str | None:
        """Feed a raw per-frame prediction (or None). Returns accepted letter
        when stability threshold is met, else None."""
        self.window.append(letter)

        # Majority vote over the window (ignoring None)
        counts: dict[str, int] = {}
        for l in self.window:
            if l is not None:
                counts[l] = counts.get(l, 0) + 1
        if not counts:
            self._prev_smoothed = None
            self._same_count = 0
            return None

        smoothed = max(counts, key=lambda k: counts[k])

        # Track how many consecutive frames the smoothed result is the same
        if smoothed == self._prev_smoothed:
            self._same_count += 1
        else:
            self._prev_smoothed = smoothed
            self._same_count = 1

        # Cooldown management
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return None

        # Accept if stable long enough
        if self._same_count >= self.stable_count:
            self._same_count = 0
            self._cooldown_remaining = self.cooldown
            return smoothed

        return None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_hud(frame, current_letter, confidence, word_buffer, smoothed_letter):
    """Overlay information on the video frame."""
    h, w = frame.shape[:2]

    # Semi-transparent bar at the bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 120), (w, h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Current raw prediction
    raw_text = f"Raw: {current_letter or '---'}  ({confidence * 100:.0f}%)"
    cv2.putText(frame, raw_text, (15, h - 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Smoothed (accepted) letter
    smooth_text = f"Accepted: {smoothed_letter or '---'}"
    cv2.putText(frame, smooth_text, (15, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)

    # Word buffer
    word_text = f"Word: {''.join(word_buffer)}_"
    cv2.putText(frame, word_text, (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Big letter in the top-right corner
    if current_letter:
        cv2.putText(frame, current_letter, (w - 80, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    # Instructions
    cv2.putText(frame, "q:quit  c:clear  bksp:del", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
# Display capability detection
# ---------------------------------------------------------------------------
def _has_display() -> bool:
    """Return True if a GUI display is likely available."""
    if sys.platform == "win32":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def _process_frame(frame, detector, model, classes, device, smoother,
                   word_buffer, timestamp_ms):
    """Run detection + prediction + smoothing on one frame.
    Returns (current_letter, confidence, accepted_letter)."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    features = extract_features_from_frame(img_rgb, detector, timestamp_ms)
    current_letter = None
    confidence = 0.0

    if features is not None:
        current_letter, confidence = predict_frame(features, model, classes, device)
        if confidence < CONFIDENCE_THRESHOLD:
            current_letter = None

    # Handle special classes
    if current_letter in ("del", "nothing"):
        current_letter = None
    elif current_letter == "space":
        current_letter = " "

    accepted = smoother.update(current_letter)

    if accepted is not None:
        if accepted == " ":
            if word_buffer and word_buffer[-1] != " ":
                word_buffer.append(" ")
                print(f"  [SPACE]  →  {''.join(word_buffer).strip()}")
        else:
            word_buffer.append(accepted)
            print(f"  + {accepted}  →  {''.join(word_buffer)}")

    return current_letter, confidence, accepted


def run(source, model, classes, detector, device, headless=False,
       output_path=None):
    """Process video frames from *source* (int for webcam, str for file)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer if --output was requested
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            sys.exit(f"[ERROR] Cannot create output video: {output_path}")
        print(f"Writing annotated video to: {output_path}")

    smoother = LetterSmoother()
    word_buffer: list[str] = []

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / fps)
        current_letter, confidence, accepted = _process_frame(
            frame, detector, model, classes, device, smoother,
            word_buffer, timestamp_ms,
        )

        # Draw HUD if we need to display or save
        if not headless or writer:
            draw_hud(frame, current_letter, confidence, word_buffer, accepted)

        # Write annotated frame to output file
        if writer:
            writer.write(frame)

        if not headless:
            cv2.imshow("ASL Fingerspelling", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == 8 and word_buffer:  # Backspace
                removed = word_buffer.pop()
                print(f"  [BACKSPACE] removed '{removed}'  →  {''.join(word_buffer)}")
            elif key == ord("c"):
                word_buffer.clear()
                print("  [CLEAR]")

        # Progress indicator for headless video mode
        if headless and total_frames > 0 and frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  ... frame {frame_idx}/{total_frames} ({pct:.0f}%)")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"\nAnnotated video saved to: {os.path.abspath(output_path)}")
    if not headless:
        cv2.destroyAllWindows()

    final_word = "".join(word_buffer).strip()
    print(f"\n=== Session ended ({frame_idx} frames processed) ===")
    print(f"Spelled word: {final_word if final_word else '(empty)'}")
    return final_word


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ASL Fingerspelling Word Recognition (Phase 2)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--webcam", action="store_true",
                       help="Use live webcam feed")
    group.add_argument("--video", type=str, metavar="PATH",
                       help="Path to a video file containing fingerspelling")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0, used with --webcam)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI window (terminal output only). "
                             "Auto-enabled when no display is detected.")
    parser.add_argument("--output", "-o", type=str, metavar="PATH",
                        help="Save annotated video (with HUD overlay) to this "
                             "file (e.g. output.mp4)")
    args = parser.parse_args()

    # -- Validate model files --
    for path, label in [(MODEL_PATH, "model"), (CLASSES_PATH, "classes"),
                        (LANDMARK_MODEL, "hand landmarker")]:
        if not os.path.exists(path):
            sys.exit(
                f"[ERROR] {label} file not found: {path}\n"
                "Run the extract_landmarks and train notebooks first."
            )

    if args.video and not os.path.isfile(args.video):
        sys.exit(f"[ERROR] Video file not found: {args.video}")

    # -- Load model --
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ASLClassifier(checkpoint["input_dim"], checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded  ({len(classes)} classes, device={device})")

    # -- Create MediaPipe detector in VIDEO mode --
    base_opts = mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL)
    options = HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    # -- Determine headless mode --
    headless = args.headless or not _has_display()
    if headless and not args.headless:
        print("[INFO] No display detected – running in headless mode.")

    # -- Run --
    source = args.camera if args.webcam else args.video
    try:
        run(source, model, classes, detector, device, headless=headless,
            output_path=args.output)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
