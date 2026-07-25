"""Label taxonomies for the water-level benchmark, and the event grouping that is the
unit of statistical analysis.

The dataset's gold labels are the source paper's 4-class ordinal scale
(low < medium < high < flood). Two coarser framings are evaluated as separate
*prompted* tasks rather than post-hoc relabelings, because asking a model for fewer
categories measurably moves its operating point: on the same model and the same
physical threshold, natively binary-prompted runs differ from 4-class runs collapsed
to binary by up to 0.11 macro-F1, in a model-dependent direction.

  4class : low < medium < high < flood   -- the published taxonomy, retained for
           comparability with Ranieri et al. 2024.
  3class : no_risk < risky < flood       -- the PRIMARY task. Collapses medium/high,
           which (a) dominates the error mass for every arm measured so far,
           including a supervised probe trained on those very labels, and (b) maps
           to no operational decision. no_risk/risky/flood map to ignore/watch/alarm.
  binary : not_flood < flood             -- diagnostic. Isolates the single
           physically unambiguous threshold (water touching the grass bank), at the
           cost of destroying the ordinal structure: every over-call becomes a hard
           false alarm rather than an ordinal error.

Event grouping
--------------
The camera fires every few minutes, so frames from one rising-water event are near
duplicates. In the 1,592-example evaluation set the 75 flood frames come from only
10 distinct days, and one day (2020-01-12) supplies 22 of them; 64 of 74 consecutive
flood frames are within 15 minutes of the previous one. Frames are therefore not
independent observations. Events -- not frames -- are the unit for

  * bootstrap resampling (cluster bootstrap over events; resampling frames gives
    confidence intervals roughly 3x too narrow on the flood class), and
  * few-shot support-set sampling (K exemplars drawn frame-wise can easily be K
    frames of one event, i.e. visually K=1).

`event_id` is (calendar date, camera), which is the finest grouping the annotation
CSV supports and is conservative: a single rising/receding event never spans two
dates at this site in the labeled subset.
"""
from __future__ import annotations

GOLD_LABELS = ["low", "medium", "high", "flood"]

# Ordinal, ascending severity. Order matters: it defines the ordinal-distance metric,
# the canonical ordering of few-shot exemplars in a prompt, and the column order of
# every confusion matrix.
TAXONOMY_LABELS: dict[str, list[str]] = {
    "4class": ["low", "medium", "high", "flood"],
    "3class": ["no_risk", "risky", "flood"],
    "binary": ["not_flood", "flood"],
}

# gold 4-class label -> label in the target taxonomy
TAXONOMY_MAPS: dict[str, dict[str, str]] = {
    "4class": {label: label for label in GOLD_LABELS},
    "3class": {"low": "no_risk", "medium": "risky", "high": "risky", "flood": "flood"},
    "binary": {"low": "not_flood", "medium": "not_flood", "high": "not_flood", "flood": "flood"},
}

TAXONOMIES = list(TAXONOMY_LABELS)

# Label a run falls back to when a completion cannot be parsed at all. This is the
# majority/least-severe class, so a fallback is a *conservative* prediction that
# deflates flood recall -- runs must report their parse-failure rate alongside any
# metric, and analyses should be repeated with fallback rows excluded.
FALLBACK_LABEL = {taxonomy: labels[0] for taxonomy, labels in TAXONOMY_LABELS.items()}


def labels_for(taxonomy: str) -> list[str]:
    try:
        return TAXONOMY_LABELS[taxonomy]
    except KeyError:
        raise ValueError(f"unknown taxonomy {taxonomy!r} (expected one of {TAXONOMIES})") from None


def ordinal_index(taxonomy: str) -> dict[str, int]:
    """label -> position on the ascending-severity scale."""
    return {label: i for i, label in enumerate(labels_for(taxonomy))}


def map_label(gold_label: str, taxonomy: str) -> str:
    """Map a gold 4-class label into `taxonomy`'s label space."""
    try:
        return TAXONOMY_MAPS[taxonomy][gold_label]
    except KeyError:
        if taxonomy not in TAXONOMY_MAPS:
            raise ValueError(f"unknown taxonomy {taxonomy!r} (expected one of {TAXONOMIES})") from None
        raise ValueError(f"unknown gold label {gold_label!r} (expected one of {GOLD_LABELS})") from None


def top_label(taxonomy: str) -> str:
    """The most-severe label -- always `flood`; named for readability at call sites."""
    return labels_for(taxonomy)[-1]


def event_id(example: dict) -> str:
    """Group key for near-duplicate frames: (calendar date, camera).

    Takes an example/result-row dict carrying `datetime` (ISO 8601) and `place`.
    """
    return f"{example['datetime'][:10]}|{example['place']}"
