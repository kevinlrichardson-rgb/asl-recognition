"""
wlasl_train.py — Train a BiLSTM + attention classifier on WLASL landmark sequences.

Reads all .npz files from data/wlasl_landmarks/, trains a sequence model,
and saves:
    models/wlasl_word_model.pt   — PyTorch checkpoint
    models/wlasl_classes.npy     — class label array

Usage:
    python src/wlasl_train.py [--epochs N] [--batch-size B] [--min-samples K]
                               [--hidden H] [--layers L] [--dropout D]

Options:
    --epochs       Max training epochs (default: 80)
    --batch-size   Batch size (default: 64)
    --min-samples  Minimum instances per class to include (default: 5)
    --hidden       LSTM hidden size (default: 256)
    --layers       LSTM layers (default: 2)
    --dropout      Dropout rate (default: 0.4)
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data" / "wlasl_landmarks"
MODEL_OUT = ROOT / "models" / "wlasl_word_model.pt"
CLASS_OUT = ROOT / "models" / "wlasl_classes.npy"
LOG_OUT   = ROOT / "logs"   / "wlasl_training.log"

FEAT_DIM  = 225   # matches wlasl_extract.py
VAL_FRAC  = 0.15
TEST_FRAC = 0.10
SEED      = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

class LazyLandmarkDataset(Dataset):
    """Loads .npz files on demand instead of holding all data in RAM."""
    def __init__(self, file_paths: list[Path], labels: list[int],
                 augment: bool = False):
        self.file_paths = file_paths
        self.labels     = labels
        self.augment    = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        d = np.load(self.file_paths[idx])
        seq    = torch.from_numpy(d["features"].astype(np.float32))
        length = int(d["original_length"])

        if self.augment:
            T = seq.size(0)

            # 1. Gaussian noise on valid frames
            noise = torch.randn_like(seq) * 0.012
            noise[length:] = 0.0
            seq = seq + noise

            # 2. Temporal shift: slide the valid segment by ±3 frames
            if length > 4:
                shift = torch.randint(-3, 4, (1,)).item()
                new_start = max(0, min(shift, T - length))
                shifted = torch.zeros_like(seq)
                shifted[new_start:new_start + length] = seq[:length]
                seq = shifted

            # 3. Frame dropout: zero out 1–2 random valid frames
            n_drop = torch.randint(0, 3, (1,)).item()
            if n_drop > 0 and length > n_drop:
                drop_idx = torch.randperm(length)[:n_drop]
                seq[drop_idx] = 0.0

            # 4. Hand mirror: swap left ↔ right hand features with p=0.5
            #    Pose: dims 0–98  |  Left hand: 99–161  |  Right hand: 162–224
            if torch.rand(1).item() < 0.5:
                left  = seq[:, 99:162].clone()
                right = seq[:, 162:225].clone()
                seq[:, 99:162]  = right
                seq[:, 162:225] = left

            # 5. Scale jitter: multiply valid frames by a factor in [0.85, 1.15]
            scale = 0.85 + torch.rand(1).item() * 0.30
            seq[:length] = seq[:length] * scale

        return seq, torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(length)


# ── Model ─────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """Soft-attention over time dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        scores = self.attn(x).squeeze(-1)   # (B, T)
        # Mask padding
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)   # (B, T, 1)
        return (x * weights).sum(dim=1)   # (B, H)


class WLASLModel(nn.Module):
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
        # x: (B, T, F)
        x = self.input_proj(x)                    # (B, T, H)
        x, _ = self.lstm(x)                       # (B, T, H*2)
        x = self.attn(x, lengths)                 # (B, H*2)
        return self.classifier(x)                 # (B, num_classes)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(data_dirs: list[Path], min_samples: int) -> tuple:
    """Scan landmark dirs, collect file paths and labels (no bulk loading)."""
    counts: dict[str, int] = {}
    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"  Warning: {data_dir} not found, skipping.")
            continue
        for gd in sorted(data_dir.iterdir()):
            if gd.is_dir():
                counts[gd.name] = counts.get(gd.name, 0) + len(list(gd.glob("*.npz")))

    classes = sorted(g for g, c in counts.items() if c >= min_samples)
    if not classes:
        raise RuntimeError(
            f"No classes with >= {min_samples} samples found in {data_dirs}.\n"
            "Run the extract scripts first."
        )

    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Data dirs       : {[str(d) for d in data_dirs]}")
    print(f"Classes (>={min_samples} samples): {len(classes)}")
    print(f"  Range: {classes[0]} … {classes[-1]}")

    file_paths: list[Path] = []
    labels: list[int] = []
    for gloss in classes:
        for data_dir in data_dirs:
            gd = data_dir / gloss
            if not gd.exists():
                continue
            for npz_path in sorted(gd.glob("*.npz")):
                file_paths.append(npz_path)
                labels.append(class_to_idx[gloss])

    # Read seq_len and feat_dim from first file
    d = np.load(file_paths[0])
    seq_len  = d["features"].shape[0]
    feat_dim = d["features"].shape[1]

    print(f"Total instances : {len(labels)}  | Seq len: {seq_len}  | Feature dim: {feat_dim}")
    return file_paths, labels, np.array(classes), seq_len, feat_dim


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimiser, device) -> tuple[float, float]:
    model.train()
    total_loss = total_correct = total = 0
    for seqs, lbls, lens in loader:
        seqs, lbls, lens = seqs.to(device), lbls.to(device), lens.to(device)
        optimiser.zero_grad()
        logits = model(seqs, lens)
        loss = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item() * len(lbls)
        total_correct += (logits.argmax(1) == lbls).sum().item()
        total += len(lbls)
    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total = 0
    for seqs, lbls, lens in loader:
        seqs, lbls, lens = seqs.to(device), lbls.to(device), lens.to(device)
        logits = model(seqs, lens)
        loss = criterion(logits, lbls)
        total_loss += loss.item() * len(lbls)
        total_correct += (logits.argmax(1) == lbls).sum().item()
        total += len(lbls)
    return total_loss / total, total_correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train WLASL word-sign LSTM classifier")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Min instances per class (default: 5)")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dirs", type=str,
                        default=str(DATA_DIR),
                        help="Comma-separated landmark dirs (default: wlasl_landmarks). "
                             "Example: data/wlasl_landmarks,data/asl_citizen_landmarks")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load data
    data_dirs = [ROOT / p.strip() if not Path(p.strip()).is_absolute()
                 else Path(p.strip()) for p in args.data_dirs.split(",")]
    file_paths, labels, classes, seq_len, feat_dim = load_dataset(data_dirs, args.min_samples)
    num_classes = len(classes)

    # ── Split indices
    n = len(labels)
    n_val  = max(1, int(n * VAL_FRAC))
    n_test = max(1, int(n * TEST_FRAC))
    n_train = n - n_val - n_test

    rng = np.random.RandomState(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_ds = LazyLandmarkDataset(
        [file_paths[i] for i in train_idx],
        [labels[i] for i in train_idx], augment=True)
    val_ds = LazyLandmarkDataset(
        [file_paths[i] for i in val_idx],
        [labels[i] for i in val_idx], augment=False)
    test_ds = LazyLandmarkDataset(
        [file_paths[i] for i in test_idx],
        [labels[i] for i in test_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"\nSplit: train={n_train}  val={n_val}  test={n_test}")

    # ── Model
    model = WLASLModel(feat_dim, num_classes, args.hidden, args.layers, args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}\n")

    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-5
    )

    # ── Training
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15

    log_rows = [["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimiser, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()

        log_rows.append([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                         f"{vl_loss:.4f}", f"{vl_acc:.4f}", f"{current_lr:.2e}"])

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={tr_acc:.3f} ({tr_loss:.3f})  "
              f"val={vl_acc:.3f} ({vl_loss:.3f})  "
              f"lr={current_lr:.1e}  {elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            # Save best checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "feat_dim": feat_dim,
                "num_classes": num_classes,
                "hidden": args.hidden,
                "num_layers": args.layers,
                "dropout": args.dropout,
                "seq_len": seq_len,
                "val_acc": vl_acc,
            }, MODEL_OUT)
            np.save(CLASS_OUT, classes)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (best val acc: {best_val_acc:.3f})")
                break

    # ── Write log
    with open(LOG_OUT, "w", newline="") as f:
        csv.writer(f).writerows(log_rows)

    # ── Test evaluation
    print(f"\nBest val acc: {best_val_acc:.4f}")
    checkpoint = torch.load(MODEL_OUT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"Test  acc: {test_acc:.4f}  (loss: {test_loss:.4f})")
    print(f"\nModel saved → {MODEL_OUT}")
    print(f"Classes saved → {CLASS_OUT}")
    print(f"Log saved → {LOG_OUT}")


if __name__ == "__main__":
    main()
