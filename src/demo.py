#!/usr/bin/env python3
"""
demo.py — ASL model evaluation suite.

Tests both the fingerspell (A-Z) and word-sign (WLASL) models against all
available data and prints accuracy / confidence reports.

Usage:
    python src/demo.py                   # run both test suites
    python src/demo.py --letters         # fingerspell only
    python src/demo.py --words           # word-sign only
    python src/demo.py --words --verbose # word-sign with full per-word table
    python src/demo.py img1.jpg ...      # predict on custom images (fingerspell)
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)
import numpy as np
import torch
import torch.nn as nn


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT            = Path(__file__).resolve().parent.parent
FS_MODEL_PATH   = ROOT / "models" / "asl_model.pt"
FS_CLASSES_PATH = ROOT / "models" / "label_classes.npy"
WS_MODEL_PATH   = ROOT / "models" / "wlasl_word_model.pt"
WS_CLASSES_PATH = ROOT / "models" / "wlasl_classes.npy"
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
TEST_DIR        = ROOT / "data"   / "asl_alphabet_test"
WLASL_LM_DIR    = ROOT / "data"   / "wlasl_landmarks"

NUM_LANDMARKS = 21
COORDS_PER_LM = 3


# ── Model definitions ─────────────────────────────────────────────────────────

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
        return (x * torch.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)


class WLASLModel(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
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


# ── Landmark / image helpers ──────────────────────────────────────────────────

def normalise_landmarks(landmarks_list):
    pts = np.array(landmarks_list, dtype=np.float32).reshape(NUM_LANDMARKS, COORDS_PER_LM)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def extract_features(img_bgr, detector):
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    raw = []
    for lm in result.hand_landmarks[0]:
        raw.extend([lm.x, lm.y, lm.z])
    return normalise_landmarks(raw)


def _try_extract(img_bgr, detector):
    """Try several pre-processing strategies to get a hand detection."""
    h, w = img_bgr.shape[:2]

    feat = extract_features(img_bgr, detector)
    if feat is not None:
        return feat

    bilat = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    feat = extract_features(bilat, detector)
    if feat is not None:
        return feat

    pad = int(max(h, w) * 0.15)
    for color in [(0, 0, 0), (255, 255, 255)]:
        padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=color)
        feat = extract_features(padded, detector)
        if feat is not None:
            return feat

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    feat = extract_features(enhanced, detector)
    if feat is not None:
        return feat

    pad2 = int(max(h, w) * 0.20)
    enhanced_padded = cv2.copyMakeBorder(enhanced, pad2, pad2, pad2, pad2,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return extract_features(enhanced_padded, detector)


def predict_image(img_path: str, model, classes, detector, device):
    """Return (predicted_class, confidence) or (None, 0.0) on failure."""
    img = cv2.imread(img_path)
    if img is None:
        return None, 0.0
    features = _try_extract(img, detector)
    if features is None:
        return None, 0.0
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs    = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return str(classes[pred_idx]), float(probs[pred_idx])


# ── Fingerspell test ──────────────────────────────────────────────────────────

def run_fingerspell_test():
    print("\n" + "=" * 62)
    print("  FINGERSPELL MODEL  (A-Z)")
    print("=" * 62)

    missing = [p for p in (FS_MODEL_PATH, FS_CLASSES_PATH, HAND_MODEL_PATH)
               if not p.exists()]
    if missing:
        for p in missing:
            print(f"  [SKIP] File not found: {p}")
        return

    ckpt    = torch.load(FS_MODEL_PATH, map_location="cpu", weights_only=False)
    classes = np.load(FS_CLASSES_PATH, allow_pickle=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = ASLClassifier(ckpt["input_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model: {ckpt['num_classes']} classes  |  device: {device}")

    detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.05,
        min_hand_presence_confidence=0.05,
        min_tracking_confidence=0.05,
    ))

    test_images = sorted(TEST_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.png"))
    if not test_images:
        print(f"  [SKIP] No test images found in {TEST_DIR}")
        detector.close()
        return

    print(f"\n  {'Letter':<8} {'Predicted':<12} {'Confidence':>10}  Result")
    print("  " + "-" * 45)

    correct, no_hand = 0, 0
    results = []

    for img_path in test_images:
        true_label = img_path.stem.split("_")[0].upper()
        if true_label not in classes:
            continue

        img = cv2.imread(str(img_path))
        features = _try_extract(img, detector) if img is not None else None

        if features is None:
            no_hand += 1
            results.append((true_label, None, 0.0, False))
            print(f"  {true_label:<8} {'---':<12} {'---':>10}  ✗  [no hand detected]")
            continue

        x = torch.from_numpy(features).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        probs    = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred     = str(classes[pred_idx])
        conf     = float(probs[pred_idx])
        match    = pred == true_label
        correct += int(match)
        results.append((true_label, pred, conf, match))

        marker = "✓" if match else "✗"
        detail = "" if match else f"  ← predicted {pred}"
        print(f"  {true_label:<8} {pred:<12} {conf*100:>9.1f}%  {marker}{detail}")

    alpha_tested = len(results) - no_hand
    print("  " + "-" * 45)
    if alpha_tested:
        print(f"  Accuracy : {correct}/{alpha_tested} = {correct/alpha_tested*100:.1f}%")
        avg_conf = np.mean([c for _, _, c, _ in results if c > 0])
        print(f"  Avg conf : {avg_conf*100:.1f}%")
    if no_hand:
        print(f"  No hand  : {no_hand} image(s) skipped")

    detector.close()


# ── Word-sign test ────────────────────────────────────────────────────────────

def run_wordsign_test(verbose: bool = False, batch_size: int = 256):
    print("\n" + "=" * 62)
    print("  WORD-SIGN MODEL  (WLASL)")
    print("=" * 62)

    missing = [p for p in (WS_MODEL_PATH, WS_CLASSES_PATH) if not p.exists()]
    if missing:
        for p in missing:
            print(f"  [SKIP] File not found: {p}")
        return

    ckpt    = torch.load(WS_MODEL_PATH, map_location="cpu", weights_only=False)
    classes = np.load(WS_CLASSES_PATH, allow_pickle=True)
    device  = torch.device("cpu")   # large batch LSTM inference; GPU OOMs on Jetson
    model   = WLASLModel(
        ckpt["feat_dim"], ckpt["num_classes"],
        ckpt["hidden"], ckpt["num_layers"], ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    seq_len = ckpt["seq_len"]
    print(f"  Model: {ckpt['num_classes']} classes  |  seq_len: {seq_len}  |  device: {device}")

    class_set    = set(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    word_dirs    = sorted(p for p in WLASL_LM_DIR.iterdir()
                          if p.is_dir() and p.name in class_set)

    if not word_dirs:
        print(f"  [SKIP] No landmark data in {WLASL_LM_DIR}")
        return

    print(f"  Words with local data: {len(word_dirs)} / {len(classes)}")
    print(f"\n  Loading sequences ...", end="", flush=True)

    all_feats, all_lengths, all_true_idx, all_words = [], [], [], []

    for wd in word_dirs:
        true_idx = class_to_idx[wd.name]
        for npz_path in sorted(wd.glob("*.npz")):
            data = np.load(npz_path)
            all_feats.append(data["features"].astype(np.float32))
            all_lengths.append(int(min(data["original_length"], seq_len)))
            all_true_idx.append(true_idx)
            all_words.append(wd.name)

    n_total = len(all_feats)
    print(f" {n_total} sequences loaded.")
    print(f"  Running inference in batches of {batch_size} ...", end="", flush=True)

    feats_t = torch.from_numpy(np.stack(all_feats))
    lens_t  = torch.tensor(all_lengths, dtype=torch.long)
    preds_all, confs_all = [], []

    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            fb = feats_t[start : start + batch_size].to(device)
            lb = lens_t [start : start + batch_size].to(device)
            logits = model(fb, lb)
            probs  = torch.softmax(logits, dim=1)
            top_conf, top_idx = probs.max(dim=1)
            preds_all.append(top_idx.cpu())
            confs_all.append(top_conf.cpu())

    preds = torch.cat(preds_all).numpy()
    confs = torch.cat(confs_all).numpy()
    trues = np.array(all_true_idx)
    print(" done.\n")

    # Per-word stats
    word_stats: dict[str, dict] = {}
    for i, word in enumerate(all_words):
        s = word_stats.setdefault(word, {"correct": 0, "total": 0, "conf_sum": 0.0})
        s["total"]    += 1
        s["conf_sum"] += float(confs[i])
        if preds[i] == trues[i]:
            s["correct"] += 1

    def sort_key(item):
        s = item[1]
        acc      = s["correct"] / s["total"]
        avg_conf = s["conf_sum"] / s["total"]
        return (acc, avg_conf)   # ascending → worst first

    sorted_stats = sorted(word_stats.items(), key=sort_key)

    # ── Overall summary ────────────────────────────────────────────────────
    total_correct = sum(s["correct"] for s in word_stats.values())
    total_seqs    = sum(s["total"]   for s in word_stats.values())
    perfect       = sum(1 for s in word_stats.values() if s["correct"] == s["total"])
    zero_acc      = sum(1 for s in word_stats.values() if s["correct"] == 0)
    partial       = len(word_stats) - perfect - zero_acc
    avg_conf_all  = float(confs.mean()) * 100

    print(f"  {'OVERALL SUMMARY':}")
    print(f"  {'─'*50}")
    print(f"  Sequence accuracy : {total_correct}/{total_seqs}  "
          f"({total_correct/total_seqs*100:.1f}%)")
    print(f"  Word accuracy     : {perfect}/{len(word_stats)}  "
          f"({perfect/len(word_stats)*100:.1f}%) perfect")
    print(f"  Partial (some ok) : {partial}")
    print(f"  Words at 0%       : {zero_acc}")
    print(f"  Avg confidence    : {avg_conf_all:.1f}%")

    # ── Per-word table ─────────────────────────────────────────────────────
    col_w  = 24
    header = f"  {'Word':<{col_w}} {'Seqs':>5}  {'Correct':>7}  {'Acc':>6}  {'Avg Conf':>8}"
    divider = "  " + "─" * (len(header) - 2)

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  FULL PER-WORD RESULTS  (sorted worst → best accuracy)")
        print(f"  {'─'*50}")
        print(header)
        print(divider)
        for word, s in sorted_stats:
            acc      = s["correct"] / s["total"]
            avg_conf = s["conf_sum"] / s["total"]
            marker   = "✓" if s["correct"] == s["total"] else ("~" if s["correct"] > 0 else "✗")
            print(f"  {marker} {word:<{col_w-2}} {s['total']:>5}  "
                  f"{s['correct']:>7}  {acc*100:>5.1f}%  {avg_conf*100:>7.1f}%")
        print(divider)
    else:
        # Show worst 30 and best 30
        worst = sorted_stats[:30]
        best  = sorted_stats[-30:]

        print(f"\n  {'─'*50}")
        print(f"  WORST 30 WORDS  (lowest accuracy)")
        print(f"  {'─'*50}")
        print(header)
        print(divider)
        for word, s in worst:
            acc      = s["correct"] / s["total"]
            avg_conf = s["conf_sum"] / s["total"]
            marker   = "~" if s["correct"] > 0 else "✗"
            print(f"  {marker} {word:<{col_w-2}} {s['total']:>5}  "
                  f"{s['correct']:>7}  {acc*100:>5.1f}%  {avg_conf*100:>7.1f}%")
        print(divider)

        print(f"\n  {'─'*50}")
        print(f"  BEST 30 WORDS  (highest accuracy)")
        print(f"  {'─'*50}")
        print(header)
        print(divider)
        for word, s in reversed(best):
            acc      = s["correct"] / s["total"]
            avg_conf = s["conf_sum"] / s["total"]
            marker   = "✓" if s["correct"] == s["total"] else "~"
            print(f"  {marker} {word:<{col_w-2}} {s['total']:>5}  "
                  f"{s['correct']:>7}  {acc*100:>5.1f}%  {avg_conf*100:>7.1f}%")
        print(divider)
        print(f"\n  (use --verbose to see all {len(word_stats)} words)")


# ── Custom image prediction ───────────────────────────────────────────────────

def run_custom_images(image_paths: list[str]):
    print("\n" + "=" * 62)
    print("  CUSTOM IMAGE PREDICTION  (fingerspell)")
    print("=" * 62)

    missing = [p for p in (FS_MODEL_PATH, FS_CLASSES_PATH, HAND_MODEL_PATH)
               if not p.exists()]
    if missing:
        for p in missing:
            print(f"  [ERROR] File not found: {p}")
        return

    ckpt    = torch.load(FS_MODEL_PATH, map_location="cpu", weights_only=False)
    classes = np.load(FS_CLASSES_PATH, allow_pickle=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = ASLClassifier(ckpt["input_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.05,
        min_hand_presence_confidence=0.05,
        min_tracking_confidence=0.05,
    ))

    print(f"\n  {'File':<35} {'Prediction':<12} {'Confidence':>10}")
    print("  " + "-" * 60)
    for path in image_paths:
        pred, conf = predict_image(path, model, classes, detector, device)
        name = Path(path).name
        if pred is None:
            print(f"  {name:<35} {'[no hand]':<12} {'---':>10}")
        else:
            print(f"  {name:<35} {pred:<12} {conf*100:>9.1f}%")

    detector.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASL fingerspell and word-sign models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--letters", action="store_true",
                      help="Run fingerspell test only")
    mode.add_argument("--words",   action="store_true",
                      help="Run word-sign test only")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every word in the word-sign table")
    parser.add_argument("images", nargs="*",
                        help="Custom image paths (runs fingerspell prediction)")
    args = parser.parse_args()

    if args.images:
        run_custom_images(args.images)
    elif args.letters:
        run_fingerspell_test()
    elif args.words:
        run_wordsign_test(verbose=args.verbose)
    else:
        run_fingerspell_test()
        run_wordsign_test(verbose=args.verbose)

    print()


if __name__ == "__main__":
    main()
