#!/usr/local/bin/python3.11
"""
demo.py
-------
Demonstration script: run the trained ASL classifier on one or more image
files and print the predicted letter with its confidence score.

Also processes the official test images in data/asl_alphabet_test/ and prints
a quick report.

Usage:
    # Predict on the provided test images
    python src/demo.py

    # Predict on one or more custom images
    python src/demo.py path/to/hand1.jpg path/to/hand2.jpg
"""

import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np
import torch
import torch.nn as nn


# Paths

BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH      = os.path.join(BASE_DIR, "models", "asl_model.pt")
CLASSES_PATH    = os.path.join(BASE_DIR, "models", "label_classes.npy")
LANDMARK_MODEL  = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
TEST_DIR        = os.path.join(BASE_DIR, "data",   "asl_alphabet_test")

NUM_LANDMARKS    = 21
COORDS_PER_LM    = 3



# Model definition (must match train.py)

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



# Landmark helpers (mirror of extract_landmarks.py)

def normalise_landmarks(landmarks_list):
    pts = np.array(landmarks_list, dtype=np.float32).reshape(NUM_LANDMARKS, COORDS_PER_LM)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def extract_features(img_bgr, detector):
    """Extract normalised landmark vector from a BGR image. Returns None if no hand found."""
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    raw = []
    for lm in result.hand_landmarks[0]:
        raw.extend([lm.x, lm.y, lm.z])
    return normalise_landmarks(raw)



# Prediction

def _try_extract(img_bgr, detector):
    """Try multiple orientation-preserving pre-processing strategies to find a hand.

    NOTE: We intentionally do NOT flip the image — ASL signs are hand-specific
    and a mirror image changes which letter the pose represents.
    """
    h, w = img_bgr.shape[:2]

    # 1. Original image
    feat = extract_features(img_bgr, detector)
    if feat is not None:
        return feat

    # 2. Bilateral filter (edge-preserving denoise — helps with fist poses)
    bilat = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    feat = extract_features(bilat, detector)
    if feat is not None:
        return feat

    # 3. Pad with black border (tight crops confuse the detector)
    pad = int(max(h, w) * 0.15)
    padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    feat = extract_features(padded, detector)
    if feat is not None:
        return feat

    # 4. Pad with white border
    padded_w = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    feat = extract_features(padded_w, detector)
    if feat is not None:
        return feat

    # 5. CLAHE contrast enhancement
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    feat = extract_features(enhanced, detector)
    if feat is not None:
        return feat

    # 6. Enhanced + padded (larger border to give detector more context)
    pad2 = int(max(h, w) * 0.20)
    enhanced_padded = cv2.copyMakeBorder(enhanced, pad2, pad2, pad2, pad2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    feat = extract_features(enhanced_padded, detector)
    if feat is not None:
        return feat

    # 7. Sharpened image (accentuates finger edges for fist-like poses)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(img_bgr, -1, kernel)
    feat = extract_features(sharpened, detector)
    return feat


def predict_image(img_path: str, model, classes, detector, device) -> tuple[str | None, float]:
    """Return (predicted_class, confidence) or (None, 0.0) on failure."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] Cannot read image: {img_path}")
        return None, 0.0

    features = _try_extract(img, detector)
    if features is None:
        return None, 0.0

    x = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs      = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return classes[pred_idx], confidence



# Main

def main():
    # -- Load model --
    for path, label in [(MODEL_PATH, "model"), (CLASSES_PATH, "classes")]:
        if not os.path.exists(path):
            sys.exit(
                f"[ERROR] {label} file not found: {path}\n"
                "Run  python src/extract_landmarks.py  then  python src/train.py  first."
            )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes    = np.load(CLASSES_PATH, allow_pickle=True)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ASLClassifier(checkpoint["input_dim"], checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded. Classes: {list(classes)}\n")

    base_opts = mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL)
    options   = HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.05,
        min_hand_presence_confidence=0.05,
        min_tracking_confidence=0.05,
    )
    detector = HandLandmarker.create_from_options(options)

    # -- Determine images to process --
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
        print("=== Custom image predictions ===")
        for path in image_paths:
            pred, conf = predict_image(path, model, classes, detector, device)
            if pred is None:
                print(f"  {os.path.basename(path):30s} → [no hand detected]")
            else:
                print(f"  {os.path.basename(path):30s} → {pred}  ({conf*100:.1f}%)")
    else:
        # Run on the test set and compute accuracy for the 26 alpha letters
        test_images = sorted(
            f for f in os.listdir(TEST_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        print("=== Official test-set predictions ===")
        correct = 0
        total   = 0

        for fname in test_images:
            # Filename format: <LETTER>_test.jpg  or  nothing_test.jpg etc.
            label_str = fname.split("_")[0].upper()
            if label_str not in classes:
                continue   # skip non-alpha classes (nothing, space)

            path       = os.path.join(TEST_DIR, fname)
            pred, conf = predict_image(path, model, classes, detector, device)
            total     += 1

            if pred is None:
                status = "[no hand]"
            else:
                match   = pred == label_str
                correct += int(match)
                status  = f"→ {pred}  ({conf*100:.1f}%)  {'OK' if match else f'WRONG (true={label_str})'}"

            print(f"  {fname:30s} {status}")

        if total > 0:
            print(f"\nTest accuracy on {total} alpha images: {correct}/{total} = {correct/total*100:.1f}%")

    detector.close()


if __name__ == "__main__":
    main()
