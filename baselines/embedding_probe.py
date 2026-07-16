"""Supervised baselines on frozen DINOv2 embeddings: logistic-regression and small-MLP
probes, trained on the image-exemplar bank and evaluated on the LLM test set under the
same leave-one-season-out protocol as everything else.

Training data is the bank itself (knowledge_base/image_bank_backup): every rare
(medium/high/flood) image plus the FPS-selected diverse lows, per-season balanced --
i.e. the probes see exactly the same curated image pool the image-RAG arm retrieves
from, which makes them the natural supervised reference for it. For each fold, train
on the bank rows from the 3 train seasons, predict the test-set rows of the held-out
season; every test example is therefore scored by a model that never saw its season.

Test images are embedded locally (CPU is fine, ~30-45 min for 1,592; cached to
--test-embeddings so it only happens once). Predictions are written in the standard
results-JSONL schema so eval/ and the cross-model comparison work unchanged.

Usage:
    python -m baselines.embedding_probe                       # both probes
    python -m baselines.embedding_probe --probe logreg        # just one
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.enoe_images import LABELS, SEASONS

BANK_DIR = Path("knowledge_base/image_bank_backup")
TEST_EXAMPLES = Path("data/processed/test_examples.jsonl")
TEST_EMB_CACHE = Path("data/processed/test_embeddings.npy")


def embed_test_images(examples: list[dict], cache_path: Path) -> np.ndarray:
    """DINOv2-embed the test images with the exact preprocessing the bank used,
    caching to disk (row order matches `examples`)."""
    if cache_path.exists():
        emb = np.load(cache_path)
        if len(emb) == len(examples):
            return emb
    import torch

    from agent.image_retrieval import _build_preprocess

    preprocess = _build_preprocess()
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dino.eval()
    from PIL import Image

    from tqdm import tqdm

    out = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(examples), 16), desc="embedding test images"):
            batch = []
            for ex in examples[i : i + 16]:
                with Image.open(ex["image_path"]) as img:
                    batch.append(preprocess(img.convert("RGB")))
            out.append(dino(torch.stack(batch)).numpy())
    emb = np.concatenate(out, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    return emb


def make_probe(kind: str, seed: int):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        # 2-layer MLP: 768 -> 256 -> 4. No class_weight support, so training rows are
        # oversampled to balance in fit_probe instead.
        return MLPClassifier(
            hidden_layer_sizes=(256,),
            early_stopping=True,
            max_iter=500,
            random_state=seed,
        )
    raise ValueError(f"unknown probe kind: {kind}")


def balance_by_oversampling(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = pd.Series(y).value_counts()
    target = counts.max()
    idx = []
    for label, n in counts.items():
        label_idx = np.flatnonzero(y == label)
        idx.append(label_idx)
        if n < target:
            idx.append(rng.choice(label_idx, size=target - n, replace=True))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=["logreg", "mlp", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-embeddings", default=str(TEST_EMB_CACHE))
    parser.add_argument("--out-dir", default="experiments/results")
    args = parser.parse_args()

    bank_emb = np.load(BANK_DIR / "embeddings.npy")
    bank_meta = pd.read_parquet(BANK_DIR / "metadata.parquet")
    examples = [json.loads(line) for line in open(TEST_EXAMPLES)]
    test_emb = embed_test_images(examples, Path(args.test_embeddings))
    test_season = np.array([ex["season"] for ex in examples])

    kinds = ["logreg", "mlp"] if args.probe == "both" else [args.probe]
    for kind in kinds:
        preds = np.empty(len(examples), dtype=object)
        confs = np.zeros(len(examples))
        for season in SEASONS:
            train_mask = (bank_meta["season"] != season).to_numpy()
            X, y = bank_emb[train_mask], bank_meta["label"].to_numpy()[train_mask]
            if kind == "mlp":
                X, y = balance_by_oversampling(X, y, args.seed)
            probe = make_probe(kind, args.seed)
            probe.fit(X, y)
            fold_mask = test_season == season
            proba = probe.predict_proba(test_emb[fold_mask])
            fold_preds = probe.classes_[proba.argmax(axis=1)]
            preds[fold_mask] = fold_preds
            confs[fold_mask] = proba.max(axis=1)
            n_correct = sum(
                p == ex["label"] for p, ex in zip(fold_preds, np.array(examples, dtype=object)[fold_mask])
            )
            print(f"{kind} | test season {season}: {fold_mask.sum()} examples, exact {n_correct}")

        out_path = Path(args.out_dir) / f"probe_dinov2_{kind}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for ex, p, c in zip(examples, preds, confs):
                f.write(json.dumps({
                    "classification": str(p),
                    "confidence": float(c),
                    "rationale": f"supervised dinov2-{kind} probe (leave-one-season-out)",
                    "cited_evidence": [],
                    "example": ex,
                    "tool_call_log": [],
                    "model": f"dinov2-vitb14+{kind}",
                    "usage": {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }) + "\n")
        exact = sum(p == ex["label"] for p, ex in zip(preds, examples))
        print(f"{kind}: overall exact {exact}/{len(examples)} ({exact/len(examples):.1%}) -> {out_path}")


if __name__ == "__main__":
    main()
