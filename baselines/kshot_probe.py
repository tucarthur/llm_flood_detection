"""Matched-K supervised comparators on frozen DINOv2 embeddings.

This is the supervised side of the crossover: at each K it trains on *exactly the same
support set* the LLM saw for that episode and fold, so the two arms differ only in the
learner, not in the data budget or the draw. Episode identity comes from the manifests in
data/episodes.py, and episodes 0-2 of the E=10 manifests are byte-identical to the E=3
manifests used for the LLM cells.

Two learners, both standard low-data references:
  logreg    -- multinomial logistic regression on the frozen embeddings, class-balanced.
  prototype -- nearest class centroid (the prototypical-network reduction for a frozen
               backbone). This is the learner that makes "few-shot" technically accurate
               on the supervised side: an explicit K-shot support set, no episodic
               meta-training required.

Plus `K=all`: trained on every bank row from the three training seasons (~900 images),
which is the "specialist with the site's whole labeled history" reference point.

Full per-class probabilities are written into every row so a recall-matched operating
point can be computed post hoc. Without that, comparing a cost-asymmetric prompt (the LLM
arms are instructed to break ties toward severity) against an untuned argmax would
attribute a decision-rule difference to the model class.

Usage:
    python -m baselines.kshot_probe                       # all K, both learners
    python -m baselines.kshot_probe --k 1 2 --probe logreg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.enoe_images import SEASONS
from data.episodes import load_manifest
from data.taxonomy import labels_for, map_label

BANK_DIR = Path("knowledge_base/image_bank_backup")
TEST_EXAMPLES = Path("data/processed/test_examples.jsonl")
TEST_EMB = Path("data/processed/test_embeddings.npy")
RESULTS_DIR = Path("experiments/results")
TAXONOMY = "3class"


def fit_predict(kind: str, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, labels: list[str], seed: int):
    """-> (predicted labels, probability matrix aligned to `labels`)."""
    if kind == "prototype":
        # Nearest class centroid on L2-normalised embeddings; "probability" is a softmax
        # over negative distances, which is monotone in the decision function and so
        # supports threshold sweeps without claiming calibration.
        centroids = np.stack([X[y == label].mean(axis=0) for label in labels])
        Xn = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-9)
        Cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        sims = Xn @ Cn.T
        exp = np.exp((sims - sims.max(axis=1, keepdims=True)) * 10.0)
        proba = exp / exp.sum(axis=1, keepdims=True)
        return np.array(labels)[proba.argmax(axis=1)], proba

    from sklearn.linear_model import LogisticRegression

    probe = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)
    probe.fit(X, y)
    raw = probe.predict_proba(X_test)
    # Re-index columns onto the canonical label order; a support set can lack a class only
    # if K=0, but being explicit keeps the output schema stable.
    proba = np.zeros((len(X_test), len(labels)))
    for j, label in enumerate(probe.classes_):
        proba[:, labels.index(label)] = raw[:, j]
    return np.array(labels)[proba.argmax(axis=1)], proba


def write_cell(path: Path, examples: list[dict], preds, proba, labels, model_name: str, meta: dict):
    with open(path, "w") as f:
        for ex, pred, row in zip(examples, preds, proba):
            f.write(json.dumps({
                "observations": "",
                "rationale": f"{model_name} (leave-one-season-out, matched support set)",
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
    parser.add_argument("--k", type=int, nargs="*", default=[1, 2, 4, 6])
    parser.add_argument("--episodes-glob", default="experiments/episodes_3class_k{k}_e10.json")
    parser.add_argument("--probe", nargs="*", default=["logreg", "prototype"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-full", action="store_true", help="Skip the K=all reference point")
    args = parser.parse_args()

    labels = labels_for(TAXONOMY)
    bank_emb = np.load(BANK_DIR / "embeddings.npy")
    bank_meta = pd.read_parquet(BANK_DIR / "metadata.parquet")
    row_of_path = {p: i for i, p in enumerate(bank_meta["path"])}
    bank_tax = bank_meta["label"].map(lambda l: map_label(l, TAXONOMY)).to_numpy()

    examples = [json.loads(line) for line in open(TEST_EXAMPLES)]
    test_emb = np.load(TEST_EMB)
    if len(test_emb) != len(examples):
        raise SystemExit(f"embedding cache is stale: {len(test_emb)} rows vs {len(examples)} examples")
    test_season = np.array([ex["season"] for ex in examples])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for kind in args.probe:
        for k in args.k:
            manifest = load_manifest(args.episodes_glob.format(k=k))
            for episode in manifest["episodes"]:
                idx = episode["episode"]
                preds = np.empty(len(examples), dtype=object)
                proba = np.zeros((len(examples), len(labels)))
                for season in SEASONS:
                    support = episode["supports"][season]
                    rows = [row_of_path[ex["path"]] for ex in support]
                    X = bank_emb[rows]
                    y = np.array([map_label(ex["label"], TAXONOMY) for ex in support])
                    mask = test_season == season
                    fold_preds, fold_proba = fit_predict(kind, X, y, test_emb[mask], labels, args.seed)
                    preds[mask] = fold_preds
                    proba[mask] = fold_proba
                out = RESULTS_DIR / f"probe_{kind}_3class_k{k}_e{idx}.jsonl"
                write_cell(out, examples, preds, proba, labels, f"dinov2-vitb14+{kind}",
                           {"k": k, "episode": idx, "episodes_manifest": args.episodes_glob.format(k=k)})
                written += 1
            print(f"{kind} k={k}: wrote {len(manifest['episodes'])} episode cells")

        if not args.skip_full:
            # K=all: the specialist trained on the site's entire labeled history available
            # for the fold -- the upper bound on what supervision can buy here.
            preds = np.empty(len(examples), dtype=object)
            proba = np.zeros((len(examples), len(labels)))
            for season in SEASONS:
                mask_train = (bank_meta["season"] != season).to_numpy()
                mask = test_season == season
                fold_preds, fold_proba = fit_predict(
                    kind, bank_emb[mask_train], bank_tax[mask_train], test_emb[mask], labels, args.seed
                )
                preds[mask] = fold_preds
                proba[mask] = fold_proba
            out = RESULTS_DIR / f"probe_{kind}_3class_kall.jsonl"
            write_cell(out, examples, preds, proba, labels, f"dinov2-vitb14+{kind}",
                       {"k": "all", "episode": None})
            written += 1
            print(f"{kind} k=all: wrote {out.name}")

    print(f"\n{written} cells written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
