"""
Build model-ready datasets from the provided historical Seoul datasets.

This script converts the raw historical CSVs into the files expected by
`seoul_grid_vitality_pipeline.py`.

Important assumptions:
1. The KT 50m cell data is the canonical grid unit.
2. Card datasets do not include a direct mapping to the KT grid, so card demand
   is distributed to grids using flow-based shares as a proxy.
3. Rainfall is only available at station/gu level here, so a month-level Seoul
   summary is used and localized with grid flow volatility.
4. Transit access and rent level are not present in the provided files, so
   stable proxies are derived from spatial centrality and demand intensity.
5. Correction targets are pseudo-labels derived from historical deviations.

These assumptions keep the project runnable end-to-end while preserving the
proposal's structure.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class Paths:
    source_dir: str = "./source_data"
    data_dir: str = "./data"

    kt_hourly: str = "./source_data/kt_hourly_flow.csv"
    kt_resident: str = "./source_data/kt_resident_flow.csv"
    rainfall: str = "./source_data/rainfall.csv"
    card_domestic_block: str = "./source_data/card_domestic_block.csv"
    card_domestic_tract_age: str = "./source_data/card_domestic_tract_age.csv"
    card_domestic_inflow: str = "./source_data/card_domestic_inflow.csv"
    card_foreign_block: str = "./source_data/card_foreign_block.csv"
    card_foreign_country: str = "./source_data/card_foreign_country.csv"

    base_train: str = "./data/base_train.csv"
    base_infer: str = "./data/base_infer.csv"
    correction_train: str = "./data/correction_train.csv"
    correction_infer: str = "./data/correction_infer.csv"
    final_actual: str = "./data/final_actual.csv"
    mapping_csv: str = "./data/grid_place_mapping.csv"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="cp949")


def minmax(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - series.min()) / span


def zscore(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - series.mean()) / std


def normalize_month_col(value: object) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".")[0]
    return text[:6]


def build_base_features(paths: Paths) -> pd.DataFrame:
    hourly = load_csv(paths.kt_hourly)
    resident = load_csv(paths.kt_resident)
    rainfall = load_csv(paths.rainfall)
    card_dom_block = load_csv(paths.card_domestic_block)
    card_dom_age = load_csv(paths.card_domestic_tract_age)
    card_dom_inflow = load_csv(paths.card_domestic_inflow)
    card_for_block = load_csv(paths.card_foreign_block)
    card_for_country = load_csv(paths.card_foreign_country)

    grid_col = "셀id(ID)"
    x_col = "x좌표(X_COORD)"
    y_col = "y좌표(Y_COORD)"
    yoil_col = "요일(YOIL)"
    hour_col = "시간대(TIMEZN_CD)"
    total_col = "합계(TOTAL)"
    month_col = "기준년월(ETL_YM)"
    admi_col = "행정동코드(ADMI_CD)"

    hourly = hourly.rename(
        columns={
            grid_col: "grid_id",
            x_col: "x",
            y_col: "y",
            yoil_col: "yoil",
            hour_col: "hour",
            total_col: "total_flow",
            month_col: "ym",
            admi_col: "admi_cd",
        }
    )
    hourly["ym"] = hourly["ym"].map(normalize_month_col)
    hourly["weekend_flag"] = hourly["yoil"].astype(int).isin([6, 7]).astype(int)

    grid_summary = (
        hourly.groupby("grid_id")
        .agg(
            x=("x", "mean"),
            y=("y", "mean"),
            admi_cd=("admi_cd", "first"),
            avg_flow=("total_flow", "mean"),
            flow_std=("total_flow", "std"),
            monthly_count=("ym", "nunique"),
        )
        .reset_index()
    )
    grid_summary["flow_std"] = grid_summary["flow_std"].fillna(0.0)

    weekday_weekend = (
        hourly.groupby(["grid_id", "weekend_flag"])["total_flow"]
        .mean()
        .unstack(fill_value=0.0)
        .rename(columns={0: "weekday_flow", 1: "weekend_flow"})
        .reset_index()
    )
    weekday_weekend["weekday_weekend_gap"] = (
        weekday_weekend["weekend_flow"] - weekday_weekend["weekday_flow"]
    ).abs()

    hour_profile = (
        hourly.groupby(["grid_id", "hour"])["total_flow"]
        .mean()
        .reset_index()
    )
    hour_stats = hour_profile.groupby("grid_id")["total_flow"].agg(["max", "mean"]).reset_index()
    hour_stats["hourly_concentration"] = hour_stats["max"] / hour_stats["mean"].replace(0, np.nan)
    hour_stats["hourly_concentration"] = hour_stats["hourly_concentration"].fillna(0.0)

    resident = resident.rename(
        columns={
            "셀id(ID)": "grid_id",
            "주중보행인구수(WKDY_FLPOP_CNT)": "resident_weekday_flow",
            "주말보행인구수(WKND_FLPOP_CNT)": "resident_weekend_flow",
        }
    )
    resident_summary = (
        resident.groupby("grid_id")[["resident_weekday_flow", "resident_weekend_flow"]]
        .mean()
        .reset_index()
    )

    rain = rainfall.rename(
        columns={
            "시우량(RAINFALLHOUR)": "rain_hour",
            "최대우량(RAINFALLMAX)": "rain_max",
            "송신지_자료수집_시각(RECEIVE_TIME)": "rain_time",
        }
    )
    rain["rain_time"] = pd.to_datetime(rain["rain_time"], errors="coerce")
    rain["ym"] = rain["rain_time"].dt.strftime("%Y%m")
    rain = rain.dropna(subset=["ym"])
    rain_summary = rain.groupby("ym").agg(
        rainfall_mean=("rain_hour", "mean"),
        rainfall_peak=("rain_max", "mean"),
    )

    flow_month = hourly.groupby(["grid_id", "ym"])["total_flow"].mean().reset_index()
    flow_month = flow_month.merge(rain_summary.reset_index(), on="ym", how="left")
    flow_month["rainfall_mean"] = flow_month["rainfall_mean"].fillna(flow_month["rainfall_mean"].mean())
    flow_month["rainfall_peak"] = flow_month["rainfall_peak"].fillna(flow_month["rainfall_peak"].mean())

    def rain_impact_fn(group: pd.DataFrame) -> float:
        if group["rainfall_mean"].nunique() <= 1:
            return 0.0
        corr = group["total_flow"].corr(group["rainfall_mean"])
        return float(corr) if pd.notna(corr) else 0.0

    rain_impact = pd.DataFrame(
        [
            {"grid_id": grid_id, "rainfall_impact": rain_impact_fn(group[["total_flow", "rainfall_mean"]])}
            for grid_id, group in flow_month.groupby("grid_id")
        ]
    )
    rain_grid_mean = (
        flow_month.groupby("grid_id")["rainfall_mean"]
        .mean()
        .reset_index()
    )

    card_total_amount = (
        card_dom_block["카드이용금액계(AMT_CORR)"].sum()
        + card_dom_age["카드이용금액계(AMT_CORR)"].sum()
        + card_dom_inflow["카드이용금액계(AMT_CORR)"].sum()
        + card_for_block["카드이용금액계(AMT_CORR)"].sum()
        + card_for_country["카드이용금액계(AMT_CORR)"].sum()
    )
    card_total_count = (
        card_dom_block["카드이용건수(USECT_CORR)"].sum()
        + card_dom_age["카드이용건수(USECT_CORR)"].sum()
        + card_dom_inflow["카드이용건수(USECT_CORR)"].sum()
        + card_for_block["카드이용건수(USECT_CORR)"].sum()
        + card_for_country["카드이용건수(USECT_CORR)"].sum()
    )

    grid_summary["flow_share"] = grid_summary["avg_flow"] / grid_summary["avg_flow"].sum()
    grid_summary["card_sales_amount"] = card_total_amount * grid_summary["flow_share"]
    grid_summary["card_sales_count"] = card_total_count * grid_summary["flow_share"]

    center_x = grid_summary["x"].mean()
    center_y = grid_summary["y"].mean()
    grid_summary["distance_to_center"] = np.sqrt((grid_summary["x"] - center_x) ** 2 + (grid_summary["y"] - center_y) ** 2)
    centrality = 1.0 - minmax(grid_summary["distance_to_center"])
    flow_strength = minmax(grid_summary["avg_flow"])
    concentration = minmax(grid_summary["flow_std"])
    grid_summary["bus_subway_access"] = 0.55 * centrality + 0.45 * flow_strength
    grid_summary["rent_level"] = 0.6 * flow_strength + 0.4 * minmax(grid_summary["card_sales_amount"])

    base = grid_summary.merge(
        weekday_weekend[["grid_id", "weekday_weekend_gap"]],
        on="grid_id",
        how="left",
    ).merge(
        hour_stats[["grid_id", "hourly_concentration"]],
        on="grid_id",
        how="left",
    ).merge(
        resident_summary,
        on="grid_id",
        how="left",
    ).merge(
        rain_grid_mean,
        on="grid_id",
        how="left",
    ).merge(
        rain_impact,
        on="grid_id",
        how="left",
    )

    base["weekday_weekend_gap"] = base["weekday_weekend_gap"].fillna(0.0)
    base["hourly_concentration"] = base["hourly_concentration"].fillna(0.0)
    base["resident_weekday_flow"] = base["resident_weekday_flow"].fillna(base["avg_flow"])
    base["resident_weekend_flow"] = base["resident_weekend_flow"].fillna(base["avg_flow"])
    base["rainfall_mean"] = base["rainfall_mean"].fillna(rain["rain_hour"].mean())
    base["rainfall_impact"] = base["rainfall_impact"].fillna(0.0)

    base["base_target"] = (
        35 * minmax(base["avg_flow"])
        + 15 * minmax(base["weekday_weekend_gap"])
        + 12 * minmax(base["hourly_concentration"])
        + 18 * minmax(base["card_sales_amount"])
        + 8 * minmax(base["card_sales_count"])
        + 5 * (1 - minmax(base["rainfall_mean"]))
        + 4 * (1 - minmax(base["distance_to_center"]))
        + 3 * minmax(base["bus_subway_access"])
    )

    base["lon"] = base["x"]
    base["lat"] = base["y"]
    return base


def build_correction_sets(base_df: pd.DataFrame, paths: Paths) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly = load_csv(paths.kt_hourly).rename(
        columns={
            "셀id(ID)": "grid_id",
            "시간대(TIMEZN_CD)": "hour",
            "요일(YOIL)": "yoil",
            "합계(TOTAL)": "total_flow",
            "기준년월(ETL_YM)": "ym",
        }
    )
    hourly["ym"] = hourly["ym"].map(normalize_month_col)
    hourly["hour"] = pd.to_numeric(hourly["hour"], errors="coerce").fillna(12).astype(int)
    hourly["yoil"] = pd.to_numeric(hourly["yoil"], errors="coerce").fillna(1).astype(int)
    hourly["month"] = hourly["ym"].str[4:6].astype(int)

    rain = load_csv(paths.rainfall).rename(
        columns={
            "시우량(RAINFALLHOUR)": "rain_hour",
            "최대우량(RAINFALLMAX)": "rain_max",
            "송신지_자료수집_시각(RECEIVE_TIME)": "rain_time",
        }
    )
    rain["rain_time"] = pd.to_datetime(rain["rain_time"], errors="coerce")
    rain["ym"] = rain["rain_time"].dt.strftime("%Y%m")
    rain_month = rain.groupby("ym").agg(real_time_rain=("rain_hour", "mean")).reset_index()

    correction = (
        hourly.groupby(["grid_id", "ym", "yoil", "hour"], as_index=False)["total_flow"]
        .mean()
    )
    correction["month"] = correction["ym"].str[4:6].astype(int)
    correction = correction.merge(
        base_df[["grid_id", "avg_flow", "hourly_concentration", "bus_subway_access"]],
        on="grid_id",
        how="left",
    ).merge(
        rain_month,
        on="ym",
        how="left",
    )

    correction["real_time_rain"] = correction["real_time_rain"].fillna(rain["rain_hour"].mean())
    correction["real_time_population"] = correction["total_flow"]
    correction = correction.sort_values(["grid_id", "ym", "yoil", "hour"]).reset_index(drop=True)
    correction["real_time_population_growth"] = (
        correction.groupby("grid_id")["real_time_population"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    correction["traffic_congestion"] = 0.7 * minmax(correction["real_time_population"]) + 0.3 * minmax(correction["hourly_concentration"])
    correction["transit_change"] = (
        correction.groupby("grid_id")["real_time_population"].diff().fillna(0.0)
    )
    correction["transit_change"] = zscore(correction["transit_change"]).fillna(0.0)
    correction["real_time_temp"] = (
        14
        + 10 * np.sin((correction["month"] - 1) / 12 * 2 * math.pi)
        - 0.8 * correction["real_time_rain"]
    )
    correction["event_flag"] = (
        correction["real_time_population"] > correction.groupby("grid_id")["real_time_population"].transform("quantile", 0.85)
    ).astype(int)
    correction["holiday_flag"] = correction["yoil"].isin([6, 7]).astype(int)

    local_baseline = correction.groupby("grid_id")["real_time_population"].transform("mean")
    correction["correction_target"] = (
        1.5 * (correction["real_time_population"] - local_baseline) / local_baseline.replace(0, np.nan)
        + 0.9 * correction["traffic_congestion"]
        + 0.4 * correction["transit_change"]
        - 0.5 * minmax(correction["real_time_rain"])
        + 0.6 * correction["event_flag"]
        - 0.2 * correction["holiday_flag"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    correction["timestamp"] = pd.to_datetime(
        correction["ym"] + "01",
        format="%Y%m%d",
        errors="coerce",
    ) + pd.to_timedelta((correction["yoil"] - 1) * 24 + correction["hour"], unit="h")

    keep_cols = [
        "grid_id",
        "timestamp",
        "real_time_population",
        "real_time_population_growth",
        "traffic_congestion",
        "transit_change",
        "real_time_temp",
        "real_time_rain",
        "event_flag",
        "holiday_flag",
        "correction_target",
    ]
    correction = correction[keep_cols].sort_values(["grid_id", "timestamp"]).reset_index(drop=True)

    infer = correction.groupby("grid_id").tail(8).copy()
    infer["timestamp"] = infer["timestamp"] + pd.DateOffset(months=1)
    infer["real_time_population"] *= 1.02
    infer["traffic_congestion"] = np.clip(infer["traffic_congestion"] * 1.03, 0, None)
    infer["real_time_temp"] += 0.5
    correction_infer = infer.drop(columns=["correction_target"]).reset_index(drop=True)

    final_actual = infer[["grid_id", "timestamp"]].copy()
    base_score_proxy = base_df.set_index("grid_id")["base_target"]
    correction_proxy = infer["correction_target"].reset_index(drop=True)
    final_actual["final_actual"] = (
        final_actual["grid_id"].map(base_score_proxy).to_numpy()
        + correction_proxy.to_numpy()
    )

    return correction, correction_infer, final_actual


def build_mapping(base_df: pd.DataFrame) -> pd.DataFrame:
    mapping = base_df[["grid_id"]].copy()
    mapping["place_id"] = [f"PLACE_{idx:03d}" for idx in range(1, len(mapping) + 1)]
    mapping["place_code"] = ""
    mapping["place_name"] = ""
    mapping["weight"] = 1.0
    return mapping[["place_id", "place_code", "place_name", "grid_id", "weight"]]


def main() -> None:
    paths = Paths()
    ensure_dir(paths.data_dir)

    base = build_base_features(paths)
    correction_train, correction_infer, final_actual = build_correction_sets(base, paths)
    mapping = build_mapping(base)

    base_train = base[
        [
            "grid_id",
            "avg_flow",
            "weekday_weekend_gap",
            "hourly_concentration",
            "card_sales_amount",
            "card_sales_count",
            "rainfall_mean",
            "rainfall_impact",
            "bus_subway_access",
            "rent_level",
            "base_target",
        ]
    ].copy()

    base_infer = base[
        [
            "grid_id",
            "lon",
            "lat",
            "avg_flow",
            "weekday_weekend_gap",
            "hourly_concentration",
            "card_sales_amount",
            "card_sales_count",
            "rainfall_mean",
            "rainfall_impact",
            "bus_subway_access",
            "rent_level",
        ]
    ].copy()

    base_train.to_csv(paths.base_train, index=False)
    base_infer.to_csv(paths.base_infer, index=False)
    correction_train.to_csv(paths.correction_train, index=False)
    correction_infer.to_csv(paths.correction_infer, index=False)
    final_actual.to_csv(paths.final_actual, index=False)
    mapping.to_csv(paths.mapping_csv, index=False)

    print("Saved preprocessed datasets:")
    print(f"- {paths.base_train}")
    print(f"- {paths.base_infer}")
    print(f"- {paths.correction_train}")
    print(f"- {paths.correction_infer}")
    print(f"- {paths.final_actual}")
    print(f"- {paths.mapping_csv}")
    print()
    print(f"base_train rows: {len(base_train)}")
    print(f"correction_train rows: {len(correction_train)}")
    print(f"correction_infer rows: {len(correction_infer)}")


if __name__ == "__main__":
    main()
