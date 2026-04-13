"""
train_fingerspell.py — Train the ASL fingerspelling letter classifier.

Loads data/landmarks.npz (produced by extract_landmarks.ipynb), trains a
fully-connected classifier, and saves:
    models/asl_model.pt       — PyTorch checkpoint
    models/label_classes.npy  — class label array (A-Z)

Usage:
    python src/train_fingerspell.py
"""

import csv
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT         = Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / "data" / "landmarks.npz"
MODEL_DIR    = ROOT / "models"
LOG_DIR      = ROOT / "logs"
MODEL_PATH   = MODEL_DIR / "asl_model.pt"
CLASSES_PATH = MODEL_DIR / "label_classes.npy"
HISTORY_PATH = MODEL_DIR / "training_history.npz"
LOG_PATH     = LOG_DIR / "training.log"

VALIDATION_SPLIT = 0.20
RANDOM_SEED      = 42
BATCH_SIZE       = 128
MAX_EPOCHS       = 100
PATIENCE         = 10
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4


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


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_acc"], label="train")
    axes[0].plot(history["val_acc"],   label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["train_loss"], label="train")
    axes[1].plot(history["val_loss"],   label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)
    fig.tight_layout()
    out = MODEL_DIR / "training_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved training curves → {out}")


def plot_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                ax=ax, annot_kws={"size": 7})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalised)")
    fig.tight_layout()
    out = MODEL_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix → {out}")


def main():
    if not DATA_PATH.exists():
        sys.exit(f"[ERROR] Landmark file not found: {DATA_PATH}\n"
                 "Run extract_landmarks.ipynb first.")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    archive = np.load(DATA_PATH, allow_pickle=True)
    X = archive["X"].astype(np.float32)
    y = archive["y"].astype(np.int64)
    classes = archive["classes"]
    num_classes = len(classes)
    print(f"Loaded {X.shape[0]:,} samples | feature dim={X.shape[1]} | classes={num_classes}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y)
    print(f"Train: {len(X_train):,}   Val: {len(X_val):,}")

    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = ASLClassifier(X.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc, best_state, patience_count = -1.0, None, 0
    log_rows = []
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = correct = total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb + torch.randn_like(xb) * 0.005
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)
        train_loss = running_loss / total
        train_acc  = correct / total

        model.eval()
        vl_sum = vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vl_sum += criterion(logits, yb).item() * len(xb)
                vc += (logits.argmax(1) == yb).sum().item()
                vt += len(xb)
        val_loss = vl_sum / vt
        val_acc  = vc / vt
        scheduler.step(val_loss)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        log_rows.append([epoch, train_loss, train_acc, val_loss, val_acc])

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  "
              f"loss={train_loss:.4f}  acc={train_acc*100:.2f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (best val acc: {best_val_acc:.3f})")
                break

    print(f"\nTraining finished in {time.time() - t0:.1f}s")

    model.load_state_dict(best_state)
    model.eval()

    def evaluate(dl):
        correct = total = 0
        preds = []
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += len(xb)
                preds.extend(pred.cpu().numpy())
        return correct / total, np.array(preds)

    train_acc, _      = evaluate(train_dl)
    val_acc,   y_pred = evaluate(val_dl)
    print(f"\nFinal train accuracy : {train_acc*100:.2f}%")
    print(f"Final val   accuracy : {val_acc*100:.2f}%")
    print("\n" + classification_report(y_val, y_pred, target_names=classes))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    plot_history(history)
    plot_confusion(y_val, y_pred, classes)

    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerows(
            [["epoch", "loss", "accuracy", "val_loss", "val_accuracy"]] + log_rows)
    print(f"Log saved → {LOG_PATH}")

    np.savez_compressed(HISTORY_PATH,
        train_acc=np.array(history["train_acc"]),
        val_acc=np.array(history["val_acc"]),
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"]))

    torch.save({"model_state_dict": model.state_dict(),
                "input_dim": X.shape[1],
                "num_classes": num_classes}, MODEL_PATH)
    np.save(CLASSES_PATH, classes)

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Classes saved → {CLASSES_PATH}")

    if val_acc >= 0.85:
        print(f"\n[PASS] Validation accuracy {val_acc*100:.2f}% >= 85% target.")
    else:
        print(f"\n[NOTE] Validation accuracy {val_acc*100:.2f}% < 85%")


if __name__ == "__main__":
    main()
