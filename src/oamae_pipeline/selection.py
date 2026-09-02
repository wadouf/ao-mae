from __future__ import annotations

import numpy as np
import pandas as pd


def closest_row(frame: pd.DataFrame, column: str, target: float) -> pd.Series:
    ranked = frame.assign(distance=(frame[column] - target).abs()).sort_values(["distance", "sample_id"])
    return ranked.iloc[0]


def select_qf1(metrics: pd.DataFrame, cloud_minimum: float, coverage_minimum: float, coverage_maximum: float) -> pd.DataFrame:
    eligible = metrics[(metrics["mean_cloud"] >= cloud_minimum) &
                       (metrics["coverage"] >= coverage_minimum) &
                       (metrics["coverage"] <= coverage_maximum) &
                       (metrics["reference_positive_pixels"] > 0)].copy()
    if eligible.empty:
        raise RuntimeError("No eligible QF1 sample")
    eligible["gain"] = eligible["iou_OA_MAE"] - eligible["iou_CROMA"]
    positive = eligible[eligible["gain"] > 0]
    if positive.empty:
        raise RuntimeError("No positive-gain QF1 sample")
    median = closest_row(positive, "gain", float(positive["gain"].median()))
    upper = closest_row(positive, "gain", float(positive["gain"].quantile(0.75)))
    regression = eligible.sort_values(["gain", "sample_id"]).iloc[0]
    selected = pd.DataFrame([median, upper, regression]).drop_duplicates("sample_id")
    return selected


def select_qf7(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, frame in metrics.groupby("city", sort=True):
        frame = frame.copy()
        frame["gain"] = frame["iou_OA_MAE"] - frame["iou_CROMA"]
        choices = [
            frame.sort_values(["mean_cloud", "sample_id"]).iloc[0],
            closest_row(frame, "mean_cloud", float(frame["mean_cloud"].median())),
            frame.sort_values(["mean_cloud", "sample_id"], ascending=[False, True]).iloc[0],
            frame.sort_values(["gain", "sample_id"]).iloc[0],
        ]
        used = set()
        for rank, choice in enumerate(choices):
            if choice["sample_id"] in used:
                alternatives = frame[~frame["sample_id"].isin(used)].sort_values(["sample_id"])
                if alternatives.empty:
                    continue
                choice = alternatives.iloc[0]
            used.add(choice["sample_id"])
            row = choice.to_dict()
            row["selection_rank"] = rank
            rows.append(row)
    return pd.DataFrame(rows)
