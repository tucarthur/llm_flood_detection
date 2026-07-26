"""Two extensions to baselines/resnet50.py that close methodological gaps in the
fine-tuned-CNN comparator, without touching that file.

Job 1 -- learning-rate sensitivity band. baselines/resnet50.py used lr=1e-4, chosen a
priori with no search. Grid-searching is indefensible on ~50 flood examples per fold, so
instead we quantify how much the a-priori choice mattered: the full-pool (K=all)
configuration re-run at lr in {3e-5, 1e-4, 3e-4}, everything else (early stopping,
batch size, augmentation, ...) identical. This reuses baselines.resnet50.train_one_fold
unmodified, so its bf16-autocast choice carries over as-is here too.

Job 2 -- multi-budget curve. Puts ResNet-50 on the same budget axis as the frozen-feature
probes (baselines/kshot_probe.py, baselines/supervised_budget_curve.py) so it is more than
a single K=all endpoint. Reuses those modules' exact sampling (episode manifests for
K-shot, sample_balanced/sample_natural with the same RNG seeding for balanced/natural) so
the only thing that differs between ResNet-50 and the probes at a given budget point is the
learner, not the data.

Early stopping on a 15% holdout is impossible at K=1 (3 training images total). To keep
the protocol uniform across the whole curve -- and to avoid handing ResNet-50 an adaptive
stopping-time advantage at high budgets that it structurally cannot have at low ones --
Job 2 trains every budget point for a fixed epoch count with NO early stopping and NO
validation split. The count is chosen a priori from the completed K=all run's per-fold
best epochs (see FIXED_EPOCHS below) and is not tuned per budget point.

Usage (run from repo root with the CUDA venv):
    .venv-gpu/bin/python -m baselines.resnet50_budget --job lr
    .venv-gpu/bin/python -m baselines.resnet50_budget --job kshot --k 1 2 4 6
    .venv-gpu/bin/python -m baselines.resnet50_budget --job balanced --budgets 12 40
    .venv-gpu/bin/python -m baselines.resnet50_budget --job natural --budgets 100 400
    .venv-gpu/bin/python -m baselines.resnet50_budget --job time-precision
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.resnet50 import (
    BANK_DIR,
    EVAL_TRANSFORM,
    IMAGE_SIZE,
    RESULTS_DIR,
    TAXONOMY,
    TEST_EXAMPLES,
    TRAIN_TRANSFORM,
    ImageLabelDataset,
    bank_image_path,
    build_model,
    train_one_fold,
    write_cell,
)
from baselines.supervised_budget_curve import sample_balanced, sample_natural
from data.enoe_images import SEASONS
from data.episodes import load_manifest
from data.taxonomy import labels_for, map_label

# --- Job 2's fixed epoch count -------------------------------------------------------
# Median of the 4 per-fold `best_epoch` values (0-indexed) in
# experiments/results/resnet50_3class.jsonl.meta.json, from the already-completed K=all
# run, +1 to convert "best epoch index" to "epochs to run". Chosen a priori from that one
# run and NOT tuned against Job 2's own results.
FIXED_EPOCHS = None  # filled in by inspecting the .meta.json; see report.

BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
DEFAULT_LR = 1e-4
NUM_WORKERS = 0
SEED = 42


def build_bank(taxonomy: str = TAXONOMY) -> pd.DataFrame:
    """Same bank frame supervised_budget_curve.main() builds -- tax_label + event, so
    sample_balanced/sample_natural see identical inputs and RNG draws to the probe runs."""
    bank = pd.read_parquet(BANK_DIR / "metadata.parquet").reset_index(drop=True)
    bank["tax_label"] = bank["label"].map(lambda l: map_label(l, taxonomy))
    bank["event"] = bank["path"].str.slice(0, 10).str.replace("/", "-") + "|" + bank["place"]
    return bank


def load_examples():
    examples = [json.loads(line) for line in open(TEST_EXAMPLES)]
    test_season = np.array([ex["season"] for ex in examples])
    return examples, test_season


# --- Job 2 training: fixed epochs, no early stopping, no val split -------------------

def train_fixed_epochs(train_paths, train_y, num_classes, device, epochs, lr,
                        autocast_dtype, seed=SEED):
    torch.manual_seed(seed)

    class_counts = np.bincount(train_y, minlength=num_classes).astype(np.float64)
    class_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_ds = ImageLabelDataset(train_paths, train_y, TRAIN_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    model = build_model(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if autocast_dtype is not None:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    out = model(imgs)
                    loss = criterion(out, labels)
            else:
                out = model(imgs)
                loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4)})
        print(f"      epoch {epoch}: train_loss={train_loss:.4f}")
    return model, history


@torch.no_grad()
def predict_fixed(model, paths, device, autocast_dtype):
    ds = ImageLabelDataset(paths, None, EVAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    model.eval()
    all_proba = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        if autocast_dtype is not None:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                out = model(imgs)
        else:
            out = model(imgs)
        proba = torch.softmax(out.float(), dim=1).cpu().numpy()
        all_proba.append(proba)
    return np.concatenate(all_proba, axis=0)


def run_cell(mode, budget_label, draw_label, per_fold_paths, per_fold_y, labels, device,
             epochs, lr, autocast_dtype, examples, test_season, out_path, extra_meta):
    n = len(examples)
    preds = np.empty(n, dtype=object)
    proba = np.zeros((n, len(labels)))
    train_sizes = {}

    for season in SEASONS:
        train_paths = per_fold_paths[season]
        train_y = per_fold_y[season]
        train_sizes[season] = len(train_paths)
        print(f"    fold {season}: n_train={len(train_paths)}")
        t0 = time.time()
        model, history = train_fixed_epochs(train_paths, train_y, len(labels), device,
                                             epochs, lr, autocast_dtype)
        mask = test_season == season
        test_paths = [ex["image_path"] for ex, m in zip(examples, mask) if m]
        fold_proba = predict_fixed(model, test_paths, device, autocast_dtype)
        fold_preds = np.array(labels)[fold_proba.argmax(axis=1)]
        preds[mask] = fold_preds
        proba[mask] = fold_proba
        print(f"    fold {season} done in {time.time() - t0:.1f}s")
        del model
        torch.cuda.empty_cache()

    write_cell(out_path, examples, preds, proba, labels, "resnet50", {
        "mode": mode,
        "budget": budget_label,
        "draw": draw_label,
        "lr": lr,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "val_frac": 0.0,
        "early_stopping": False,
        "precision": "bf16" if autocast_dtype is torch.bfloat16 else "fp32",
        "image_size": IMAGE_SIZE,
        "class_weighting": "inverse-frequency cross-entropy (balanced)",
        "pretrained": "ImageNet (ResNet50_Weights.IMAGENET1K_V2)",
        "protocol": "leave-one-season-out, fixed-epoch (no early stopping)",
        "train_size_per_fold": train_sizes,
        **extra_meta,
    })
    print(f"  wrote {out_path}")


# --- Job 1: lr sensitivity, full pool, early stopping (mirrors resnet50.main()) ------

def run_lr_job(lrs, device, max_epochs=15, patience=4, val_frac=0.15, seed=SEED):
    labels = labels_for(TAXONOMY)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    bank_meta = pd.read_parquet(BANK_DIR / "metadata.parquet")
    bank_meta["tax_label"] = bank_meta["label"].map(lambda l: map_label(l, TAXONOMY))
    bank_meta["local_path"] = bank_meta["path"].map(bank_image_path)

    examples, test_season = load_examples()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import train_test_split

    for lr in lrs:
        print(f"\n=== Job 1: lr={lr} ===")
        n = len(examples)
        preds = np.empty(n, dtype=object)
        proba = np.zeros((n, len(labels)))
        fold_epochs = {}

        for season in SEASONS:
            print(f"  fold: held-out season {season}")
            fold_pool = bank_meta[bank_meta["season"] != season]
            train_rows, val_rows = train_test_split(
                fold_pool, test_size=val_frac, random_state=seed,
                stratify=fold_pool["tax_label"],
            )
            train_paths = train_rows["local_path"].tolist()
            train_y = train_rows["tax_label"].map(label_to_idx).to_numpy()
            val_paths = val_rows["local_path"].tolist()
            val_y = val_rows["tax_label"].map(label_to_idx).to_numpy()

            best_state, history, best_epoch, best_val_acc = train_one_fold(
                train_paths, train_y, val_paths, val_y, len(labels), device,
                BATCH_SIZE, max_epochs, patience, lr, WEIGHT_DECAY, NUM_WORKERS, seed,
            )
            model = build_model(len(labels)).to(device)
            model.load_state_dict(best_state)

            mask = test_season == season
            test_paths = [ex["image_path"] for ex, m in zip(examples, mask) if m]
            fold_proba = predict_fixed(model, test_paths, device, torch.bfloat16)
            fold_preds = np.array(labels)[fold_proba.argmax(axis=1)]
            preds[mask] = fold_preds
            proba[mask] = fold_proba
            fold_epochs[season] = {"best_epoch": best_epoch, "best_val_acc": round(best_val_acc, 4)}
            print(f"  fold {season} done, best_epoch={best_epoch}, best_val_acc={best_val_acc:.4f}")
            del model, best_state
            torch.cuda.empty_cache()

        out_path = RESULTS_DIR / f"resnet50_3class_lr{lr:.0e}.jsonl"
        write_cell(out_path, examples, preds, proba, labels, "resnet50", {
            "mode": "full_pool_lr_sweep",
            "budget": "all",
            "draw": None,
            "lr": lr,
            "epochs": "early-stopped (see fold_epochs)",
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "val_frac": val_frac,
            "early_stopping": True,
            "patience": patience,
            "precision": "bf16",
            "image_size": IMAGE_SIZE,
            "class_weighting": "inverse-frequency cross-entropy (balanced)",
            "pretrained": "ImageNet (ResNet50_Weights.IMAGENET1K_V2)",
            "protocol": "leave-one-season-out",
            "fold_epochs": fold_epochs,
        })
        print(f"Wrote {out_path}")


# --- Job 2: kshot / balanced / natural ------------------------------------------------

def run_kshot_job(ks, episodes, device, epochs, lr, autocast_dtype):
    labels = labels_for(TAXONOMY)
    examples, test_season = load_examples()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for k in ks:
        manifest = load_manifest(f"experiments/episodes_3class_k{k}_e10.json")
        for ep_idx in episodes:
            episode = manifest["episodes"][ep_idx]
            assert episode["episode"] == ep_idx
            print(f"\n=== Job 2: kshot k={k} episode={ep_idx} ===")
            per_fold_paths, per_fold_y = {}, {}
            for season in SEASONS:
                support = episode["supports"][season]
                per_fold_paths[season] = [bank_image_path(ex["path"]) for ex in support]
                per_fold_y[season] = np.array([labels.index(ex["tax_label"]) for ex in support])
            out_path = RESULTS_DIR / f"resnet50_3class_kshot{k}_e{ep_idx}.jsonl"
            run_cell("kshot", k, ep_idx, per_fold_paths, per_fold_y, labels, device,
                     epochs, lr, autocast_dtype, examples, test_season, out_path,
                     {"episodes_manifest": f"experiments/episodes_3class_k{k}_e10.json"})


def run_sampled_job(mode, budgets, draws, device, epochs, lr, autocast_dtype, seed=SEED):
    assert mode in ("balanced", "natural")
    labels = labels_for(TAXONOMY)
    examples, test_season = load_examples()
    bank = build_bank()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sampler = sample_balanced if mode == "balanced" else sample_natural

    for budget in budgets:
        for draw in draws:
            print(f"\n=== Job 2: {mode}={budget} draw={draw} ===")
            # Identical RNG seeding to baselines.supervised_budget_curve.main(), and the
            # rng object is threaded across seasons in SEASONS order within this draw --
            # matching that module exactly is what makes the training sets identical to
            # the probes', not just same-sized.
            rng = np.random.default_rng(seed + 1000 * draw + budget)
            per_fold_paths, per_fold_y = {}, {}
            for season in SEASONS:
                pool = bank[bank["season"] != season]
                rows = sampler(pool, budget, rng)
                if len(np.unique(bank.loc[rows, "tax_label"])) < len(labels):
                    raise SystemExit(f"{mode} budget {budget} fold {season} missing a class")
                per_fold_paths[season] = bank.loc[rows, "path"].map(bank_image_path).tolist()
                per_fold_y[season] = bank.loc[rows, "tax_label"].map(labels.index).to_numpy()
            out_path = RESULTS_DIR / f"resnet50_3class_{mode}{budget}_e{draw}.jsonl"
            run_cell(mode, budget, draw, per_fold_paths, per_fold_y, labels, device,
                     epochs, lr, autocast_dtype, examples, test_season, out_path, {})


# --- fp32 vs bf16 timing probe --------------------------------------------------------

def time_precision(device, n_images=400, epochs=1):
    """One epoch of Job 2's fixed-epoch loop, bf16 vs fp32, on a representative
    (natural, budget=400, season=2018-2019) training set. Turing (compute 7.5) has no
    native bf16 tensor cores, so torch emulates it -- worth checking it isn't slower
    than plain fp32 before committing Job 2's whole sweep to one or the other."""
    labels = labels_for(TAXONOMY)
    bank = build_bank()
    pool = bank[bank["season"] != SEASONS[0]]
    rng = np.random.default_rng(SEED)
    rows = sample_natural(pool, n_images, rng)
    paths = bank.loc[rows, "path"].map(bank_image_path).tolist()
    y = bank.loc[rows, "tax_label"].map(labels.index).to_numpy()

    for name, dtype in [("bf16", torch.bfloat16), ("fp32", None)]:
        torch.cuda.synchronize()
        t0 = time.time()
        train_fixed_epochs(paths, y, len(labels), device, epochs, DEFAULT_LR, dtype)
        torch.cuda.synchronize()
        print(f"{name}: {time.time() - t0:.1f}s for {epochs} epoch(s) on n={len(paths)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True,
                         choices=["lr", "kshot", "balanced", "natural", "time-precision"])
    parser.add_argument("--lrs", type=float, nargs="*", default=[3e-5, 1e-4, 3e-4])
    parser.add_argument("--k", type=int, nargs="*", default=[1, 2, 4, 6])
    parser.add_argument("--episodes", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--budgets", type=int, nargs="*")
    parser.add_argument("--draws", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available -- refusing to silently fall back to CPU.")
    device = torch.device("cuda")
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else None

    if args.job == "time-precision":
        time_precision(device)
        return

    if args.job == "lr":
        run_lr_job(args.lrs, device)
        return

    if args.epochs is None:
        raise SystemExit("--epochs is required for job2 (FIXED_EPOCHS not set; pass --epochs explicitly)")

    if args.job == "kshot":
        run_kshot_job(args.k, args.episodes, device, args.epochs, args.lr, autocast_dtype)
    elif args.job == "balanced":
        budgets = args.budgets or [12, 40]
        run_sampled_job("balanced", budgets, args.draws, device, args.epochs, args.lr, autocast_dtype)
    elif args.job == "natural":
        budgets = args.budgets or [100, 400]
        run_sampled_job("natural", budgets, args.draws, device, args.epochs, args.lr, autocast_dtype)


if __name__ == "__main__":
    main()
