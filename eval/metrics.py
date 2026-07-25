"""Classification metrics for the Mineirinho Creek water-level benchmark.

Three things distinguish this from a stock scoring script, all forced by the data:

**Gold labels are mapped into the run's taxonomy before scoring.** Results files come
from three prompted tasks (see data/taxonomy.py) but the gold labels are always the
4-class scale. Scoring a binary run against 4-class labels is what made the previous
version raise `KeyError: 'accuracy'`.

**Frames are not independent, so events are the resampling unit.** The 75 flood frames
in the evaluation set come from 10 days, one of which supplies 22 of them. Confidence
intervals therefore come from a *cluster* bootstrap over `event_id`; resampling frames
would understate the width by roughly 3x on flood, and single-frame differences would
read as real effects.

**Detection is reported per event as well as per frame.** An alarm system that flags one
of the 22 frames on 2020-01-12 has caught that flood. Per-frame recall answers a
different (and, over correlated frames, inflated) question, so both are reported, along
with how far into an event the first correct flag arrives.

Accuracy is near-meaningless on the raw distribution (low = 98.84%, so always-predict-low
scores ~98.8%) and is not comparable across the resampled evaluation set either -- the
test set is ~50% low by construction. Per-class recall and the two conditional rates
(`missed_flood_rate`, `false_alarm_rate`) are unbiased under that resampling; accuracy,
precision and macro-F1 are NOT, because they depend on the class prior.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix

from data.taxonomy import (
    TAXONOMIES,
    event_id,
    labels_for,
    map_label,
    ordinal_index,
    top_label,
)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


def load_predictions(path: str) -> pd.DataFrame:
    """Read a results JSONL into a flat frame, adding the event grouping key.

    Handles the agent schema (example nested under "example"), the probe schema (same),
    and older flattened rows.
    """
    rows = []
    for line in open(path):
        r = json.loads(line)
        ex = r.get("example", r)
        rows.append({
            "label": ex["label"],
            "classification": r["classification"],
            "confidence": r.get("confidence"),
            "is_night": ex.get("is_night"),
            "season": ex.get("season"),
            "place": ex.get("place"),
            "datetime": ex.get("datetime"),
            "event_id": event_id(ex) if {"datetime", "place"} <= ex.keys() else None,
            # Older files marked failures only by a rationale prefix; newer ones carry a flag.
            "parse_failed": bool(
                r.get("parse_failed", str(r.get("rationale", "")).startswith("unparseable"))
            ),
            "taxonomy": (r.get("config") or {}).get("taxonomy"),
        })
    return pd.DataFrame(rows)


def detect_taxonomy(df: pd.DataFrame) -> str:
    """Determine which taxonomy a results frame was produced under.

    Prefers the recorded config; otherwise picks the smallest label set that covers every
    prediction actually present.
    """
    recorded = df["taxonomy"].dropna().unique() if "taxonomy" in df else []
    if len(recorded) == 1:
        return str(recorded[0])
    if len(recorded) > 1:
        raise ValueError(f"results file mixes taxonomies: {sorted(recorded)}")
    predicted = set(df["classification"].dropna())
    candidates = [t for t in TAXONOMIES if predicted <= set(labels_for(t))]
    if not candidates:
        raise ValueError(f"predictions {sorted(predicted)} match no known taxonomy")
    return min(candidates, key=lambda t: len(labels_for(t)))


def prepare(df: pd.DataFrame, taxonomy: str | None = None, drop_parse_failures: bool = False):
    """-> (frame with gold mapped into the taxonomy, taxonomy name).

    `drop_parse_failures=True` removes rows that fell back to the least-severe label.
    Those fallbacks are not missing-at-random -- they concentrate on ambiguous frames --
    so every headline number should be reported both ways.
    """
    taxonomy = taxonomy or detect_taxonomy(df)
    out = df.copy()
    if drop_parse_failures:
        out = out[~out["parse_failed"]]
    out["gold"] = out["label"].map(lambda label: map_label(label, taxonomy))
    return out, taxonomy


# --------------------------------------------------------------------------------------
# Frame-level scores
# --------------------------------------------------------------------------------------
def frame_scores(df: pd.DataFrame, taxonomy: str) -> dict:
    """Scalar frame-level metrics. Kept cheap and side-effect free so the bootstrap can
    call it thousands of times."""
    labels = labels_for(taxonomy)
    flood, lowest = top_label(taxonomy), labels[0]
    ordinals = ordinal_index(taxonomy)
    gold, pred = df["gold"], df["classification"]

    report = classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0)
    flood_rows, low_rows = df[gold == flood], df[gold == lowest]
    valid = df[pred.isin(ordinals)]
    ordinal_errors = (
        valid["gold"].map(ordinals) - valid["classification"].map(ordinals)
    ).abs()

    return {
        "n": len(df),
        # `accuracy` is absent from the report when the label set does not cover every
        # observed value; micro avg is the same quantity in that case.
        "accuracy": report.get("accuracy", report.get("micro avg", {}).get("f1-score")),
        "macro_recall": report["macro avg"]["recall"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_f1": report["macro avg"]["f1-score"],
        "flood_recall": float((flood_rows["classification"] == flood).mean()) if len(flood_rows) else None,
        "missed_flood_rate": float((flood_rows["classification"] != flood).mean()) if len(flood_rows) else None,
        "flood_precision": report[flood]["precision"],
        # Against the taxonomy's own least-severe class. NOT comparable across taxonomies:
        # under 'binary' that class absorbs medium/high, where over-calling is common, so a
        # binary run's rate looks far worse than a 4-class run's for identical behaviour.
        "false_alarm_rate": float((low_rows["classification"] == flood).mean()) if len(low_rows) else None,
        # Against gold `low` always, i.e. flood alarms raised on dry-weather frames. This
        # is the taxonomy-invariant false-alarm number -- use it for cross-arm comparison.
        "dry_false_alarm_rate": (
            float((df.loc[df["label"] == "low", "classification"] == flood).mean())
            if (df["label"] == "low").any()
            else None
        ),
        "mean_ordinal_error": float(ordinal_errors.mean()) if len(ordinal_errors) else None,
        "parse_failure_rate": float(df["parse_failed"].mean()) if len(df) else None,
    }


def classification_metrics(df: pd.DataFrame, taxonomy: str | None = None) -> dict:
    """Full frame-level report, including per-class breakdown and confusion matrix."""
    prepared, taxonomy = prepare(df, taxonomy)
    labels = labels_for(taxonomy)
    report = classification_report(
        prepared["gold"], prepared["classification"], labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(prepared["gold"], prepared["classification"], labels=labels)
    return {
        "taxonomy": taxonomy,
        **frame_scores(prepared, taxonomy),
        "per_class": {label: report[label] for label in labels if label in report},
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
    }


# --------------------------------------------------------------------------------------
# Event-level scores
# --------------------------------------------------------------------------------------
def event_scores(df: pd.DataFrame, taxonomy: str) -> dict:
    """Detection metrics with the event as the unit.

    A flood event counts as detected if ANY of its gold-flood frames was predicted flood
    -- the operationally relevant question for an alarm. `frames_to_detection` counts how
    many gold-flood frames elapsed before the first correct flag (0 = caught on the first
    frame of the event); `minutes_to_detection` is the same delay in wall-clock time.
    """
    flood = top_label(taxonomy)
    if df["event_id"].isna().any():
        return {"n_flood_events": None, "note": "results file lacks datetime/place; no event grouping"}

    detected, delays_frames, delays_minutes, n_flood_events = 0, [], [], 0
    for _, event in df[df["gold"] == flood].groupby("event_id"):
        n_flood_events += 1
        event = event.sort_values("datetime")
        hits = np.flatnonzero((event["classification"] == flood).to_numpy())
        if len(hits) == 0:
            continue
        detected += 1
        delays_frames.append(int(hits[0]))
        times = pd.to_datetime(event["datetime"].to_numpy())
        delays_minutes.append((times[hits[0]] - times[0]).total_seconds() / 60.0)

    # An event with no gold flood frame that nonetheless drew a flood prediction is an
    # event-level false alarm -- the quantity an operator experiences as a nuisance call.
    non_flood_events = df[df["gold"] != flood].groupby("event_id")
    n_non_flood_events = non_flood_events.ngroups
    alarmed = sum(1 for _, e in non_flood_events if (e["classification"] == flood).any())

    return {
        "n_flood_events": n_flood_events,
        "n_flood_events_detected": detected,
        "event_detection_rate": detected / n_flood_events if n_flood_events else None,
        "mean_frames_to_detection": float(np.mean(delays_frames)) if delays_frames else None,
        "median_minutes_to_detection": float(np.median(delays_minutes)) if delays_minutes else None,
        "n_non_flood_events": n_non_flood_events,
        "event_false_alarm_rate": alarmed / n_non_flood_events if n_non_flood_events else None,
    }


# --------------------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------------------
def bootstrap_ci(
    df: pd.DataFrame,
    taxonomy: str | None = None,
    metrics: tuple[str, ...] = ("flood_recall", "macro_f1", "false_alarm_rate", "event_detection_rate"),
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 95.0,
) -> dict:
    """Percentile CIs from a cluster bootstrap over events.

    Events are resampled with replacement (not frames), which is the only valid unit
    here: 64 of 74 consecutive flood frames are within 15 minutes of the previous one, so
    frame-level resampling treats near-duplicates as independent evidence.

    Note the ceiling this exposes: with 10 flood events, a flood-recall CI cannot be
    narrower than roughly +/-15 points however many frames are scored. That is a property
    of the data, not of the estimator.
    """
    prepared, taxonomy = prepare(df, taxonomy)
    groups = {eid: sub for eid, sub in prepared.groupby("event_id")}
    event_ids = np.array(list(groups))
    rng = np.random.default_rng(seed)

    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_resamples):
        sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
        resampled = pd.concat([groups[eid] for eid in sampled], ignore_index=True)
        scores = {**frame_scores(resampled, taxonomy), **event_scores(resampled, taxonomy)}
        for m in metrics:
            value = scores.get(m)
            if value is not None:
                draws[m].append(float(value))

    point = {**frame_scores(prepared, taxonomy), **event_scores(prepared, taxonomy)}
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "n_events": len(event_ids),
        "n_resamples": n_resamples,
        "ci": ci,
        "metrics": {
            m: {
                "point": point.get(m),
                "lo": float(np.percentile(draws[m], lo_q)) if draws[m] else None,
                "hi": float(np.percentile(draws[m], hi_q)) if draws[m] else None,
            }
            for m in metrics
        },
    }


def paired_bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str = "flood_recall",
    taxonomy: str | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 95.0,
) -> dict:
    """CI on (A - B) for two arms scored on the same examples, resampling events jointly.

    Every arm runs the identical evaluation set, so a paired test is both valid and much
    tighter than comparing two independent CIs -- which is the comparison to use for any
    "X beats Y" claim, including the K-curve crossover.
    """
    a, taxonomy = prepare(df_a, taxonomy)
    b, _ = prepare(df_b, taxonomy)
    shared = sorted(set(a["event_id"]) & set(b["event_id"]))
    groups_a = {eid: sub for eid, sub in a[a["event_id"].isin(shared)].groupby("event_id")}
    groups_b = {eid: sub for eid, sub in b[b["event_id"].isin(shared)].groupby("event_id")}
    rng = np.random.default_rng(seed)

    def score(frame):
        return {**frame_scores(frame, taxonomy), **event_scores(frame, taxonomy)}.get(metric)

    deltas = []
    for _ in range(n_resamples):
        sampled = rng.choice(np.array(shared), size=len(shared), replace=True)
        sa = score(pd.concat([groups_a[e] for e in sampled], ignore_index=True))
        sb = score(pd.concat([groups_b[e] for e in sampled], ignore_index=True))
        if sa is not None and sb is not None:
            deltas.append(sa - sb)

    point_a, point_b = score(a), score(b)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    lo = float(np.percentile(deltas, lo_q)) if deltas else None
    hi = float(np.percentile(deltas, hi_q)) if deltas else None
    return {
        "metric": metric,
        "n_shared_events": len(shared),
        "a": point_a,
        "b": point_b,
        "delta": (point_a - point_b) if (point_a is not None and point_b is not None) else None,
        "lo": lo,
        "hi": hi,
        # A CI straddling zero means the arms are not distinguishable on this metric at
        # this event count, regardless of how far apart the point estimates look.
        "significant": (lo is not None and hi is not None and (lo > 0 or hi < 0)),
    }


# --------------------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------------------
def stratified_scores(df: pd.DataFrame, by: str = "is_night", taxonomy: str | None = None) -> dict:
    """Frame + event scores per stratum.

    Day/night is a first-class axis here, not a footnote: ~47% of all frames are night
    shots but the severe classes skew further that way (flood is 64% night), and night
    frames often carry streetlight glare heavy enough to obscure the waterline. Pooled
    numbers hide that.
    """
    prepared, taxonomy = prepare(df, taxonomy)
    out = {}
    for value, group in prepared.groupby(by, dropna=False):
        out[str(value)] = {
            **frame_scores(group, taxonomy),
            **event_scores(group, taxonomy),
        }
    return out


def calibration_metrics(df: pd.DataFrame, taxonomy: str | None = None) -> dict | None:
    """One-vs-rest Brier score for the flood class.

    Only computed for binary runs. For 3- and 4-class runs the stored `confidence` is the
    confidence in whichever class was predicted, and there is no defensible way to turn
    that into P(flood): the previous `1 - confidence` mapping assigned all remaining mass
    to flood, systematically overstating it on non-flood predictions. A real calibration
    curve needs per-class probabilities, which means asking the model for them explicitly.
    """
    prepared, taxonomy = prepare(df, taxonomy)
    if taxonomy != "binary" or prepared["confidence"].isna().all():
        return None
    prepared = prepared.dropna(subset=["confidence"])
    flood = top_label(taxonomy)
    is_flood_pred = prepared["classification"] == flood
    prob_flood = np.where(is_flood_pred, prepared["confidence"], 1 - prepared["confidence"])
    return {
        "brier_score_flood": float(
            brier_score_loss((prepared["gold"] == flood).astype(int), prob_flood)
        )
    }
