"""Loader for the CrisisNLP consolidated event-aware English tweet dataset.

Source: https://crisisnlp.qcri.org/data/crisis_datasets_benchmarks/crisis_datasets_benchmarks_v1.0.tar.gz
(the "CrisisBench" benchmark, Alam et al., ICWSM 2021). Extracted under data/raw/data/event_aware_en/.

The consolidated file does not retain per-tweet timestamps (verified by inspecting the
schema: id, event, source, text, lang, lang_confidence, class_label -- no date column,
and tweet IDs are not consistently Twitter snowflake IDs across sources/events). Text
reports are therefore treated as event-level context, not date-specific evidence -- see
data/align.py and the README limitations section.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw" / "data" / "event_aware_en"
INFORMATIVENESS_TRAIN = RAW_DIR / "crisis_consolidated_informativeness_filtered_lang_en_w_event_info_train.tsv"

# Real event tags present in the dataset (verified 2026-07-09 by inspecting class_label
# value_counts on the informativeness train split).
EVENT_TAGS = {
    "sandy": "2012_sandy_hurricane-ontopic",
    "harvey": "hurricane_harvey",
    "irma": "hurricane_irma",
    "alberta": "2013_alberta_floods-ontopic",
    "queensland": "2013_queensland_floods-ontopic",
}


def load_informativeness(path: Path = INFORMATIVENESS_TRAIN) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Extract data/raw/crisisnlp_benchmarks.tar.gz first:\n"
            f"  tar xzf data/raw/crisisnlp_benchmarks.tar.gz -C data/raw --exclude='._*'"
        )
    return pd.read_csv(path, sep="\t")


def sample_texts_for_event(df: pd.DataFrame, event_key: str, n: int = 6, informative_only: bool = True, seed: int = 0) -> list[str]:
    tag = EVENT_TAGS[event_key]
    sub = df[df["event"] == tag]
    if informative_only:
        sub = sub[sub["class_label"] == "informative"]
    if sub.empty:
        return []
    sample = sub.sample(n=min(n, len(sub)), random_state=seed)
    return sample["text"].tolist()


if __name__ == "__main__":
    df = load_informativeness()
    for key in EVENT_TAGS:
        texts = sample_texts_for_event(df, key, n=3)
        print(f"\n=== {key} ({EVENT_TAGS[key]}) ===")
        for t in texts:
            print(" -", t[:120])
