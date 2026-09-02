from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PairChoice:
    s2_time: pd.Timestamp
    s1_time: pd.Timestamp
    s1_product_id: str
    gap_days: float
    orbit_direction: str
    relative_orbit: int | None


def millis_to_timestamp(value: int) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromtimestamp(value / 1000, tz=timezone.utc))


def nearest_s1(s2_time: pd.Timestamp, s1_rows: pd.DataFrame, max_gap_days: int) -> PairChoice | None:
    if s1_rows.empty:
        return None
    candidates = s1_rows.copy()
    candidates["time"] = pd.to_datetime(candidates["system:time_start"], unit="ms", utc=True)
    candidates["gap_days"] = (candidates["time"] - s2_time).abs().dt.total_seconds() / 86400.0
    candidates = candidates[candidates["gap_days"] <= max_gap_days]
    if candidates.empty:
        return None
    row = candidates.sort_values(["gap_days", "system:index"]).iloc[0]
    rel = row.get("relativeOrbitNumber_start")
    return PairChoice(
        s2_time=s2_time,
        s1_time=row["time"],
        s1_product_id=str(row["system:index"]),
        gap_days=float(row["gap_days"]),
        orbit_direction=str(row.get("orbitProperties_pass", "")),
        relative_orbit=int(rel) if pd.notna(rel) else None,
    )


def season_distance_days(a: pd.Timestamp, b: pd.Timestamp) -> int:
    base = abs(a.dayofyear - b.dayofyear)
    return int(min(base, 366 - base))


def choose_t1_t2(candidates: pd.DataFrame, minimum_gap_days: int) -> pd.DataFrame:
    rows = []
    candidates = candidates.sort_values("time")
    for left_index, left in candidates.iterrows():
        later = candidates[candidates["time"] >= left["time"] + pd.Timedelta(days=minimum_gap_days)]
        if later.empty:
            continue
        ranked = later.assign(season_gap=later["time"].apply(lambda x: season_distance_days(left["time"], x)))
        right = ranked.sort_values(["season_gap", "time"]).iloc[0]
        rows.append({"t1_index": left_index, "t2_index": right.name, "t1": left["time"], "t2": right["time"], "season_gap_days": int(right["season_gap"])})
    return pd.DataFrame(rows)
