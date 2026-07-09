"""Join structured USGS gauge data + CrisisNLP text into (event, site, target_date) ->
label classification examples.

Ground-truth labels are anchored to each hurricane's independently-documented public
landfall/impact timeline (not derived from the gauge readings themselves) specifically
to avoid circularity: the threshold/rule-based classical baseline (baselines/threshold.py)
operates on gauge discharge deviation, so if labels were *also* derived from a gauge
discharge threshold, that baseline would trivially score ~100% by construction. Anchoring
labels to the external event timeline makes gauge deviation a genuine predictive *feature*
that any approach -- classical or LLM agent -- must learn to use, rather than the label
definition itself.

Caveat: the landfall/impact dates below are the author's own recollection of public-record
event timelines and should be cross-checked against NOAA's official tropical cyclone
reports before this dataset is used in any published benchmark result.

Caveat 2: CrisisNLP text reports are event-level, not date-specific (see data/crisis_text.py
docstring) -- every target_date within an event gets the same sampled text reports. This
is a real limitation: it means the text branch cannot help discriminate the pre-storm
no_flood dates from the active_flood dates within the same event using its own signal
alone; it currently serves as general event context and as the informativeness-filtering
signal (real reports vs. off-topic noise), not as date-level evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.crisis_text import EVENT_TAGS, load_informativeness, sample_texts_for_event
from data.usgs_nwis import CASE_STUDY_SITES

USGS_CSV = Path(__file__).parent / "raw" / "usgs_daily_values.csv"
OUT_PATH = Path(__file__).parent / "processed" / "examples.jsonl"

# (no_flood_dates, flood_watch_dates, active_flood_dates) anchored to each event's
# public landfall/peak-impact timeline, independent of any gauge reading.
EVENT_LABEL_WINDOWS = {
    "sandy": {  # NJ landfall Oct 29, 2012
        "no_flood": ["2012-10-24", "2012-10-25", "2012-10-26", "2012-10-27"],
        "flood_watch": ["2012-10-28", "2012-10-29"],
        "active_flood": ["2012-10-30", "2012-10-31", "2012-11-01", "2012-11-02"],
    },
    "harvey": {  # TX landfall Aug 25, 2017; stalled over Houston through ~Aug 30
        "no_flood": ["2017-08-23", "2017-08-24"],
        "flood_watch": ["2017-08-25", "2017-08-26"],
        "active_flood": ["2017-08-27", "2017-08-28", "2017-08-29", "2017-08-30", "2017-08-31"],
    },
    "irma": {  # FL Keys landfall Sep 10, 2017; central FL impact days later
        "no_flood": ["2017-09-05", "2017-09-06", "2017-09-07"],
        "flood_watch": ["2017-09-08", "2017-09-09", "2017-09-10"],
        "active_flood": ["2017-09-11", "2017-09-12", "2017-09-13", "2017-09-14"],
    },
}


def build_examples() -> list[dict]:
    if not USGS_CSV.exists():
        raise FileNotFoundError(f"{USGS_CSV} not found -- run `python -m data.usgs_nwis` first.")
    usgs = pd.read_csv(USGS_CSV, dtype={"site_id": str})
    usgs["site_id"] = usgs["site_id"].str.zfill(8)

    text_df = load_informativeness()

    examples = []
    for event, windows in EVENT_LABEL_WINDOWS.items():
        text_reports = sample_texts_for_event(text_df, event, n=6)
        sites = [s for s in CASE_STUDY_SITES if s.event == event]
        available_dates = set(usgs[usgs["event"] == event]["date"])

        for label, dates in windows.items():
            for date in dates:
                if date not in available_dates:
                    continue  # site didn't report a value that day; skip rather than fabricate
                for site in sites:
                    site_dates = set(usgs[(usgs.event == event) & (usgs.site_id == site.site_id)]["date"])
                    if date not in site_dates:
                        continue
                    examples.append(
                        {
                            "event": event,
                            "site_id": site.site_id,
                            "site_name": site.name,
                            "start_date": min(available_dates),
                            "end_date": max(available_dates),
                            "target_date": date,
                            "label": label,
                            "text_reports": text_reports,
                        }
                    )
    return examples


if __name__ == "__main__":
    examples = build_examples()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples to {OUT_PATH}")
    label_counts = pd.Series([e["label"] for e in examples]).value_counts()
    print(label_counts)
