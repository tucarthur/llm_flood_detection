"""Client for the USGS NWIS daily-values web service (waterservices.usgs.gov).

Used to pull historical gauge height / discharge for the structured half of the
joint (structured + text) classification examples. The instantaneous-values (iv)
service only retains ~120 days of history, so historical event windows (Sandy 2012,
Harvey/Irma 2017) must use the daily-values (dv) service instead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests

NWIS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"

PARAMETER_NAMES = {
    "00060": "discharge_cfs",
    "00065": "gauge_height_ft",
    "00045": "precipitation_in",
}


@dataclass(frozen=True)
class GaugeSite:
    site_id: str
    name: str
    event: str  # which case-study event this site is curated for


# Curated, verified (via USGS site-inventory service) gauges for the three
# joint-track case-study events. Each is a real NWIS site_no confirmed reachable
# via the site service as of 2026-07-09.
CASE_STUDY_SITES: list[GaugeSite] = [
    # Hurricane Sandy, Oct 2012 -- NJ riverine flooding from Sandy's rainfall
    GaugeSite("01463500", "Delaware River at Trenton NJ", "sandy"),
    GaugeSite("01389500", "Passaic River at Little Falls NJ", "sandy"),
    GaugeSite("01464000", "Assunpink Creek at Trenton NJ", "sandy"),
    # Hurricane Harvey, Aug 2017 -- Houston-area bayous/rivers
    GaugeSite("08072600", "Buffalo Bayou at State Hwy 6 nr Addicks TX", "harvey"),
    GaugeSite("08073500", "Buffalo Bayou nr Addicks TX", "harvey"),
    GaugeSite("08071500", "San Jacinto River nr Huffman TX", "harvey"),
    # Hurricane Irma, Sep 2017 -- St Johns River basin, FL
    GaugeSite("02234000", "St Johns River above Lake Harney nr Geneva FL", "irma"),
    GaugeSite("02232500", "St Johns River near Christmas FL", "irma"),
]

EVENT_WINDOWS = {
    # (start, end) inclusive, padded a few days either side of landfall/peak impact
    # to capture rising and falling limb for negative ("no flood yet") examples too.
    "sandy": ("2012-10-24", "2012-11-05"),
    "harvey": ("2017-08-23", "2017-09-05"),
    "irma": ("2017-09-05", "2017-09-18"),
}


def fetch_daily_values(
    site_ids: list[str],
    start: str,
    end: str,
    parameter_codes: tuple[str, ...] = ("00060", "00065"),
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily-value time series for the given sites/date range.

    Returns a tidy dataframe: site_id, site_name, date, parameter_code,
    parameter_name, value, unit.
    """
    params = {
        "format": "json",
        "sites": ",".join(site_ids),
        "startDT": start,
        "endDT": end,
        "parameterCd": ",".join(parameter_codes),
        "siteStatus": "all",
    }
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(NWIS_DV_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            time.sleep(2**attempt)
    else:
        raise RuntimeError(f"NWIS dv request failed after {max_retries} attempts") from last_exc

    rows = []
    for series in payload["value"]["timeSeries"]:
        site_id = series["sourceInfo"]["siteCode"][0]["value"]
        site_name = series["sourceInfo"]["siteName"]
        var_code = series["variable"]["variableCode"][0]["value"]
        unit = series["variable"]["unit"]["unitCode"]
        param_name = PARAMETER_NAMES.get(var_code, var_code)
        for value_block in series["values"]:
            for point in value_block["value"]:
                rows.append(
                    {
                        "site_id": site_id,
                        "site_name": site_name,
                        "date": point["dateTime"][:10],
                        "parameter_code": var_code,
                        "parameter_name": param_name,
                        "value": float(point["value"]) if point["value"] not in ("", None) else None,
                        "unit": unit,
                    }
                )
    return pd.DataFrame(rows)


def fetch_all_case_study_sites() -> pd.DataFrame:
    """Fetch daily values for every curated site over its event's window."""
    frames = []
    for event, (start, end) in EVENT_WINDOWS.items():
        sites = [s.site_id for s in CASE_STUDY_SITES if s.event == event]
        df = fetch_daily_values(sites, start, end)
        df["event"] = event
        frames.append(df)
        time.sleep(1)  # be polite to the API
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    out = fetch_all_case_study_sites()
    out.to_csv("data/raw/usgs_daily_values.csv", index=False)
    print(f"Wrote {len(out)} rows to data/raw/usgs_daily_values.csv")
    print(out.groupby(["event", "parameter_name"]).size())
