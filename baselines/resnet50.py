"""Fine-tuned ResNet-50 supervised comparator for the 3-class water-level task.

Mirrors baselines/kshot_probe.py's leave-one-season-out protocol and output schema, but
the learner is a fine-tuned ImageNet ResNet-50 (whole network, new 3-way head) rather
than a frozen-embedding probe. Included because the source paper (Ranieri et al. 2024)
used CNN architectures, and this task is retrained for the reframed 3-class taxonomy.

For each held-out season: train on the training-pool rows (knowledge_base/image_bank_backup)
from the other three seasons, holding out ~15% as a validation split for early stopping;
predict on the evaluation rows (data/processed/test_examples.jsonl) for the held-out season.
The four folds' predictions are pooled into one file covering all 1592 evaluation rows.

Usage (run from repo root with the CUDA-enabled venv):
    .venv-gpu/bin/python -m baselines.resnet50
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from data.enoe_images import SEASONS
from data.taxonomy import labels_for, map_label

BANK_DIR = Path("knowledge_base/image_bank_backup")
BANK_IMAGES = Path("knowledge_base/image_bank_images")
TEST_EXAMPLES = Path("data/processed/test_examples.jsonl")
RESULTS_DIR = Path("experiments/results")
TAXONOMY = "3class"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ImageLabelDataset(Dataset):
    """Loads an image by path and returns (transformed tensor, label index or -1)."""

    def __init__(self, paths: list[str], label_idx: np.ndarray | None, transform):
        self.paths = paths
        self.label_idx = label_idx
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self.transform(img)
        y = int(self.label_idx[i]) if self.label_idx is not None else -1
        return img, y


def bank_image_path(path: str) -> str:
    return str(BANK_IMAGES / path.replace("/", "_"))


def build_model(num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_fold(
    train_paths, train_y, val_paths, val_y, num_classes, device,
    batch_size, max_epochs, patience, lr, weight_decay, num_workers, seed,
):
    """Fine-tune a fresh ResNet-50 on this fold's train split; select the best epoch by
    validation accuracy. Returns (best_model_state, history, epochs_run, best_val_acc)."""
    torch.manual_seed(seed)

    class_counts = np.bincount(train_y, minlength=num_classes).astype(np.float64)
    class_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_ds = ImageLabelDataset(train_paths, train_y, TRAIN_TRANSFORM)
    val_ds = ImageLabelDataset(val_paths, val_y, EVAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = build_model(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    # bf16 autocast rather than fp16: fp16 forward passes intermittently overflowed
    # (BatchNorm activation spikes -> inf/nan) on this GPU/model combo even with a
    # GradScaler, independent of the optimizer step. bf16's wider dynamic range avoids
    # that and needs no loss scaling; grad clipping is kept as a second safety net.
    autocast_dtype = torch.bfloat16

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                out = model(imgs)
                loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    out = model(imgs)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_acc": round(val_acc, 4)})
        print(f"    epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    early stop at epoch {epoch} (best epoch {best_epoch}, best val_acc {best_val_acc:.4f})")
                break

    return best_state, history, best_epoch, best_val_acc


@torch.no_grad()
def predict(model, paths, device, batch_size, num_workers):
    ds = ImageLabelDataset(paths, None, EVAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    all_proba = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(imgs)
        proba = torch.softmax(out.float(), dim=1).cpu().numpy()
        all_proba.append(proba)
    return np.concatenate(all_proba, axis=0)


def write_cell(path: Path, examples: list[dict], preds, proba, labels, model_name: str, meta: dict):
    with open(path, "w") as f:
        for ex, pred, row in zip(examples, preds, proba):
            f.write(json.dumps({
                "observations": "",
                "rationale": f"{model_name} (leave-one-season-out, fine-tuned)",
                "classification": str(pred),
                "confidence": float(row.max()),
                "probabilities": {label: float(p) for label, p in zip(labels, row)},
                "cited_evidence": [],
                "example": ex,
                "parse_failed": False,
                "config": {"model": model_name, "taxonomy": TAXONOMY, **meta},
                "model": model_name,
                "usage": {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(RESULTS_DIR / "resnet50_3class.jsonl"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available -- refusing to silently fall back to CPU.")
    device = torch.device("cuda")
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")

    labels = labels_for(TAXONOMY)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    bank_meta = pd.read_parquet(BANK_DIR / "metadata.parquet")
    bank_meta["tax_label"] = bank_meta["label"].map(lambda l: map_label(l, TAXONOMY))
    bank_meta["local_path"] = bank_meta["path"].map(bank_image_path)

    examples = [json.loads(line) for line in open(TEST_EXAMPLES)]
    test_season = np.array([ex["season"] for ex in examples])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n = len(examples)
    preds = np.empty(n, dtype=object)
    proba = np.zeros((n, len(labels)))

    fold_meta = []
    t_start = time.time()

    for season in SEASONS:
        print(f"\n=== fold: held-out season {season} ===")
        fold_pool = bank_meta[bank_meta["season"] != season]
        train_rows, val_rows = train_test_split(
            fold_pool, test_size=args.val_frac, random_state=args.seed,
            stratify=fold_pool["tax_label"],
        )
        train_paths = train_rows["local_path"].tolist()
        train_y = train_rows["tax_label"].map(label_to_idx).to_numpy()
        val_paths = val_rows["local_path"].tolist()
        val_y = val_rows["tax_label"].map(label_to_idx).to_numpy()
        print(f"  train={len(train_paths)} val={len(val_paths)} "
              f"train_class_counts={np.bincount(train_y, minlength=len(labels)).tolist()}")

        t_fold = time.time()
        best_state, history, best_epoch, best_val_acc = train_one_fold(
            train_paths, train_y, val_paths, val_y, len(labels), device,
            args.batch_size, args.max_epochs, args.patience, args.lr, args.weight_decay,
            args.num_workers, args.seed,
        )
        fold_wall = time.time() - t_fold

        model = build_model(len(labels)).to(device)
        model.load_state_dict(best_state)

        mask = test_season == season
        test_paths = [ex["image_path"] for ex, m in zip(examples, mask) if m]
        fold_proba = predict(model, test_paths, device, args.batch_size, args.num_workers)
        fold_preds = np.array(labels)[fold_proba.argmax(axis=1)]

        preds[mask] = fold_preds
        proba[mask] = fold_proba

        fold_meta.append({
            "held_out_season": season,
            "n_train": len(train_paths),
            "n_val": len(val_paths),
            "n_test": int(mask.sum()),
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_val_acc": round(best_val_acc, 4),
            "history": history,
            "wall_clock_seconds": round(fold_wall, 1),
        })
        print(f"  fold done in {fold_wall:.1f}s, best_epoch={best_epoch}, best_val_acc={best_val_acc:.4f}")

        del model, best_state
        torch.cuda.empty_cache()

    total_wall = time.time() - t_start

    out_path = Path(args.out)
    write_cell(out_path, examples, preds, proba, labels, "resnet50", {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_frac": args.val_frac,
        "image_size": IMAGE_SIZE,
        "class_weighting": "inverse-frequency cross-entropy (balanced)",
        "augmentation": "RandomResizedCrop(scale=0.8-1.0)+RandomHorizontalFlip",
        "pretrained": "ImageNet (ResNet50_Weights.IMAGENET1K_V2)",
        "protocol": "leave-one-season-out",
    })
    print(f"\nWrote {n} rows to {out_path}")

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent
                                              ).decode().strip()
    except Exception:
        git_commit = None

    meta = {
        "hyperparameters": {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_frac": args.val_frac,
            "image_size": IMAGE_SIZE,
            "seed": args.seed,
            "class_weighting": "inverse-frequency cross-entropy (balanced)",
            "augmentation": "RandomResizedCrop(scale=0.8-1.0)+RandomHorizontalFlip, no vertical flip, no colour jitter",
            "optimizer": "AdamW",
            "precision": "torch.amp mixed precision (bf16 autocast; fp16+GradScaler produced intermittent NaN activations)",
            "grad_clip_max_norm": 5.0,
            "pretrained": "ImageNet (ResNet50_Weights.IMAGENET1K_V2)",
        },
        "folds": fold_meta,
        "total_training_wall_clock_seconds": round(total_wall, 1),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "git_commit": git_commit,
    }
    meta_path = Path(str(out_path) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote meta to {meta_path}")
    print(f"Total training wall clock: {total_wall:.1f}s")


if __name__ == "__main__":
    main()
