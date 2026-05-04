"""
Two-phase fine-tuning script for EfficientNet-B0 on Celeb-DF v2.
Uses the same face extraction pipeline as inference to keep crops consistent.
"""

import os
import sys
import csv
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.processing import extract_frames, extract_face_regions, preprocess_faces

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {DEVICE}")

# update this path to point to your local Celeb-DF root
CELEB_DF_ROOT   = Path("c:/Users/krist/Desktop/Celeb-DF")
REAL_DIRS       = [CELEB_DF_ROOT / "Celeb-real", CELEB_DF_ROOT / "YouTube-real"]
FAKE_DIRS       = [CELEB_DF_ROOT / "Celeb-synthesis"]

# ── Training knobs ───────────────────────────────────────────────────────────
MAX_REAL_VIDEOS  = 150   # more videos now that MediaPipe detects faces reliably
MAX_FAKE_VIDEOS  = 150
FACES_PER_VIDEO  = 10    # cap per video to keep class balance
SAMPLE_RATE      = 5     # same as inference default
TRAIN_VAL_SPLIT  = 0.85
BATCH_SIZE       = 16
HEAD_EPOCHS      = 5
FULL_EPOCHS      = 12
HEAD_LR          = 1e-3
FULL_LR          = 5e-5
SAVE_PATH        = "data/models/deepfake_detector.pth"
RANDOM_SEED      = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def extract_faces_from_video(video_path: str, max_faces: int, sample_rate: int):
    """Extract face crops using the same pipeline as inference (MediaPipe → Haar fallback)."""
    frames = extract_frames(video_path, sample_rate=sample_rate)
    if not frames:
        return []
    regions, _ = extract_face_regions(frames, confidence_threshold=0.5)
    if not regions:
        return []
    # preprocess_faces returns float32 RGB [0,1]; convert back to uint8 for the Dataset
    processed = preprocess_faces(regions[:max_faces], normalize=True)
    return [(p * 255).astype(np.uint8) for p in processed]


# ── Dataset class ────────────────────────────────────────────────────────────
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(8),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FaceDataset(Dataset):
    def __init__(self, faces, labels, transform):
        self.faces = faces        # list of uint8 RGB ndarrays (224,224,3)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        img = Image.fromarray(self.faces[idx])
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Model builder ────────────────────────────────────────────────────────────
def build_model():
    """EfficientNet-B0 with custom classifier head, matching colab_training.py architecture."""
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2),
    )
    return m


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith('classifier'):
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ── Training helpers ─────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, training: bool):
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return total_loss / total, 100.0 * correct / total


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("Collecting video paths")
    logger.info("=" * 60)

    real_videos = []
    for d in REAL_DIRS:
        real_videos += [str(p) for p in d.glob("*.mp4")]
    fake_videos = []
    for d in FAKE_DIRS:
        fake_videos += [str(p) for p in d.glob("*.mp4")]

    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    real_videos = real_videos[:MAX_REAL_VIDEOS]
    fake_videos = fake_videos[:MAX_FAKE_VIDEOS]
    logger.info(f"Real videos: {len(real_videos)}, Fake videos: {len(fake_videos)}")

    logger.info("=" * 60)
    logger.info("Extracting faces")
    logger.info("=" * 60)

    all_faces, all_labels = [], []

    for i, vp in enumerate(real_videos):
        faces = extract_faces_from_video(vp, FACES_PER_VIDEO, SAMPLE_RATE)
        all_faces.extend(faces)
        all_labels.extend([0] * len(faces))  # 0 = REAL
        if (i + 1) % 20 == 0:
            logger.info(f"  Real: {i+1}/{len(real_videos)} processed, {len(all_faces)} faces so far")

    real_count = len(all_faces)
    logger.info(f"Real faces extracted: {real_count}")

    for i, vp in enumerate(fake_videos):
        faces = extract_faces_from_video(vp, FACES_PER_VIDEO, SAMPLE_RATE)
        all_faces.extend(faces)
        all_labels.extend([1] * len(faces))  # 1 = FAKE
        if (i + 1) % 20 == 0:
            logger.info(f"  Fake: {i+1}/{len(fake_videos)} processed, {len(all_faces)} faces so far")

    fake_count = len(all_faces) - real_count
    logger.info(f"Fake faces extracted: {fake_count}")
    logger.info(f"Total faces: {len(all_faces)}")

    if real_count == 0 or fake_count == 0:
        logger.error("Need both real and fake faces for training. Aborting.")
        return

    combined = list(zip(all_faces, all_labels))
    random.shuffle(combined)
    all_faces, all_labels = zip(*combined)
    all_faces, all_labels = list(all_faces), list(all_labels)

    split = int(len(all_faces) * TRAIN_VAL_SPLIT)
    train_faces, train_labels = all_faces[:split], all_labels[:split]
    val_faces,   val_labels   = all_faces[split:], all_labels[split:]

    n_real_train = train_labels.count(0)
    n_fake_train = train_labels.count(1)
    logger.info(f"Train: {len(train_labels)} ({n_real_train} real, {n_fake_train} fake)")
    logger.info(f"Val:   {len(val_labels)}")

    train_ds = FaceDataset(train_faces, train_labels, TRAIN_TRANSFORM)
    val_ds   = FaceDataset(val_faces,   val_labels,   EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Class weights to handle any residual imbalance
    total = n_real_train + n_fake_train
    w_real = total / (2.0 * max(n_real_train, 1))
    w_fake = total / (2.0 * max(n_fake_train, 1))
    class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(DEVICE)
    logger.info(f"Class weights: real={w_real:.3f}, fake={w_fake:.3f}")

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    best_state   = None
    history      = []   # list of dicts, one per epoch

    # ── Phase 1: train only classifier head ──────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Phase 1: Classifier-head only ({HEAD_EPOCHS} epochs, lr={HEAD_LR})")
    logger.info("=" * 60)
    freeze_backbone(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=HEAD_LR)

    global_epoch = 0
    for epoch in range(1, HEAD_EPOCHS + 1):
        global_epoch += 1
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, training=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, training=False)
        is_best = va_acc > best_val_acc
        marker = " ← best" if is_best else ""
        logger.info(
            f"  Epoch {epoch}/{HEAD_EPOCHS}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.1f}%  "
            f"val_loss={va_loss:.4f} acc={va_acc:.1f}%{marker}"
        )
        history.append(dict(epoch=global_epoch, phase="head_only",
                            train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 1),
                            val_loss=round(va_loss, 4),   val_acc=round(va_acc, 1),
                            best_checkpoint=is_best))
        if is_best:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Phase 2: fine-tune entire network ────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Phase 2: Full fine-tune ({FULL_EPOCHS} epochs, lr={FULL_LR})")
    logger.info("=" * 60)
    unfreeze_all(model)
    optimizer = optim.Adam(model.parameters(), lr=FULL_LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FULL_EPOCHS)

    for epoch in range(1, FULL_EPOCHS + 1):
        global_epoch += 1
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, training=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, training=False)
        scheduler.step()
        is_best = va_acc > best_val_acc
        marker = " ← best" if is_best else ""
        logger.info(
            f"  Epoch {epoch}/{FULL_EPOCHS}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.1f}%  "
            f"val_loss={va_loss:.4f} acc={va_acc:.1f}%{marker}"
        )
        history.append(dict(epoch=global_epoch, phase="full_finetune",
                            train_loss=round(tr_loss, 4), train_acc=round(tr_acc, 1),
                            val_loss=round(va_loss, 4),   val_acc=round(va_acc, 1),
                            best_checkpoint=is_best))
        if is_best:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Save model ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Best val accuracy: {best_val_acc:.1f}%")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(best_state, SAVE_PATH)
    logger.info(f"Model saved → {SAVE_PATH}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = "data/training_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "phase", "train_loss", "train_acc",
                                               "val_loss", "val_acc", "best_checkpoint"])
        writer.writeheader()
        writer.writerows(history)
    logger.info(f"Metrics saved → {csv_path}")

    # ── Save plot ─────────────────────────────────────────────────────────────
    _save_training_plot(history)


def _save_training_plot(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs     = [r["epoch"]     for r in history]
        train_accs = [r["train_acc"] for r in history]
        val_accs   = [r["val_acc"]   for r in history]
        train_loss = [r["train_loss"] for r in history]
        val_loss   = [r["val_loss"]   for r in history]
        phase_end  = max(r["epoch"] for r in history if r["phase"] == "head_only")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0f1117")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="#aaa")
            ax.spines[:].set_color("#444")
            ax.grid(axis="y", color="#333", linewidth=0.6, linestyle="--")
            ax.axvline(x=phase_end + 0.5, color="#888", linewidth=1, linestyle="--", alpha=0.7)

        ax1.plot(epochs, train_accs, "o-", color="#4c8bf5", linewidth=2, markersize=5, label="Train")
        ax1.plot(epochs, val_accs,   "o-", color="#34d399", linewidth=2, markersize=5, label="Val")
        best_eps = [r["epoch"] for r in history if r["best_checkpoint"]]
        best_vas = [r["val_acc"] for r in history if r["best_checkpoint"]]
        ax1.scatter(best_eps, best_vas, s=120, color="#fbbf24", zorder=5, marker="*", label="Best ckpt")
        best_val = max(val_accs)
        best_ep  = epochs[val_accs.index(best_val)]
        ax1.annotate(f"Best: {best_val}%",
                     xy=(best_ep, best_val), xytext=(best_ep - 2.5, best_val - 4),
                     color="#fbbf24", fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.2))
        ax1.set_title("Accuracy", color="white", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch", color="#ccc"); ax1.set_ylabel("Accuracy (%)", color="#ccc")
        ax1.set_ylim(50, 100); ax1.legend(facecolor="#252836", edgecolor="#444", labelcolor="white")

        ax2.plot(epochs, train_loss, "o-", color="#4c8bf5", linewidth=2, markersize=5, label="Train")
        ax2.plot(epochs, val_loss,   "o-", color="#34d399", linewidth=2, markersize=5, label="Val")
        ax2.set_title("Loss", color="white", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", color="#ccc"); ax2.set_ylabel("Cross-entropy loss", color="#ccc")
        ax2.legend(facecolor="#252836", edgecolor="#444", labelcolor="white")

        fig.suptitle("EfficientNet-B0 Training — Celeb-DF v2", color="white",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        out = "data/training_history.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Plot saved: {out}")
    except Exception as e:
        logger.warning(f"Could not save plot: {e}")

    # ── Quick sanity check on test video ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Sanity check on test video")
    logger.info("=" * 60)
    from ml.model import DeepfakeDetector
    from ml.processing import extract_frames, extract_face_regions, preprocess_faces

    det = DeepfakeDetector(model_path=SAVE_PATH)

    # Sample diverse random videos rather than hardcoded IDs to avoid misleading results
    import random as _rnd
    _rnd.seed(7)
    _fake_dir = CELEB_DF_ROOT / "Celeb-synthesis"
    _real_dir = CELEB_DF_ROOT / "Celeb-real"
    sanity_videos = (
        [(_rnd.choice(list(_fake_dir.glob("*.mp4"))), "FAKE") for _ in range(3)] +
        [(_rnd.choice(list(_real_dir.glob("*.mp4"))),  "REAL") for _ in range(3)]
    )
    correct = 0
    for path, true_label in sanity_videos:
        frames = extract_frames(str(path), sample_rate=5)
        regions, _ = extract_face_regions(frames, confidence_threshold=0.5)
        if not regions:
            logger.warning(f"  {true_label}: no faces detected in {path.name}")
            continue
        processed = preprocess_faces(regions, normalize=True)
        preds, _ = det.predict_batch(processed, batch_size=16)
        mean_fp = np.mean(preds)
        predicted = "FAKE" if mean_fp > 0.5 else "REAL"
        ok = predicted == true_label
        correct += int(ok)
        status = "✓" if ok else "✗"
        logger.info(f"  {status} {true_label} → {predicted} (fake_prob={mean_fp:.3f}) [{path.name}]")
    logger.info(f"Sanity result: {correct}/6")


if __name__ == "__main__":
    main()
