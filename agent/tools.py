"""Structured-data tool functions exposed to the classifier agent via tool-calling.

Deliberately NOT dumped into the prompt as raw text -- the agent must call these
tools to pull gauge readings, which is what distinguishes the "agent" architecture
from a single-shot prompt (see the no-tools ablation in baselines/).
"""
from __future__ import annotations

import pandas as pd

TOOL_SCHEMAS = [
    {
        "name": "get_gauge_reading",
        "description": "Get discharge (cfs) and gauge height (ft) for one USGS site on one date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {"type": "string", "description": "USGS site number, e.g. '01463500'"},
                "date": {"type": "string", "description": "ISO date, e.g. '2012-10-30'"},
            },
            "required": ["site_id", "date"],
        },
    },
    {
        "name": "get_gauge_time_series",
        "description": "Get the full daily discharge/gauge-height time series for one USGS site over a date range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "site_id": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["site_id", "start_date", "end_date"],
        },
    },
    {
        "name": "get_site_baseline",
        "description": (
            "Get the pre-event baseline (mean discharge/gauge-height over the first 3 "
            "available days in this site's fetched window) to compare a current reading against."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"site_id": {"type": "string"}},
            "required": ["site_id"],
        },
    },
]


class StructuredDataTools:
    """Binds the tool functions to a loaded USGS daily-values dataframe."""

    def __init__(self, usgs_df: pd.DataFrame):
        self.df = usgs_df.copy()
        self.df["site_id"] = self.df["site_id"].astype(str).str.zfill(8)
        self.df["date"] = pd.to_datetime(self.df["date"])

    def _site_slice(self, site_id: str) -> pd.DataFrame:
        return self.df[self.df["site_id"] == str(site_id).zfill(8)]

    def get_gauge_reading(self, site_id: str, date: str) -> dict:
        sub = self._site_slice(site_id)
        sub = sub[sub["date"] == pd.to_datetime(date)]
        if sub.empty:
            return {"error": f"no reading for site {site_id} on {date}"}
        out = {"site_id": site_id, "date": date}
        for _, row in sub.iterrows():
            out[row["parameter_name"]] = row["value"]
        return out

    def get_gauge_time_series(self, site_id: str, start_date: str, end_date: str) -> dict:
        sub = self._site_slice(site_id)
        sub = sub[(sub["date"] >= pd.to_datetime(start_date)) & (sub["date"] <= pd.to_datetime(end_date))]
        if sub.empty:
            return {"error": f"no data for site {site_id} in [{start_date}, {end_date}]"}
        series = (
            sub.pivot_table(index="date", columns="parameter_name", values="value")
            .reset_index()
        )
        series["date"] = series["date"].dt.strftime("%Y-%m-%d")
        return {"site_id": site_id, "readings": series.to_dict(orient="records")}

    def get_site_baseline(self, site_id: str) -> dict:
        sub = self._site_slice(site_id).sort_values("date")
        if sub.empty:
            return {"error": f"no data for site {site_id}"}
        first_dates = sub["date"].drop_duplicates().sort_values().head(3)
        baseline_slice = sub[sub["date"].isin(first_dates)]
        baseline = baseline_slice.groupby("parameter_name")["value"].mean().to_dict()
        return {"site_id": site_id, "baseline_window_days": len(first_dates), **baseline}

    def dispatch(self, tool_name: str, tool_input: dict) -> dict:
        fn = getattr(self, tool_name, None)
        if fn is None:
            return {"error": f"unknown tool {tool_name}"}
        return fn(**tool_input)
