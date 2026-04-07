"""
Fetch Seoul real-time city data OpenAPI responses and convert them into
`correction_infer`-style rows using the local grid-place mapping table.

Expected mapping columns:
- place_id: local unique identifier
- place_code: official AREA_CD or empty
- place_name: official AREA_NM or empty
- grid_id
- weight: optional weight for one place -> many grids
- api_query: optional direct query string for AREA_NM

Required environment variables:
- SEOUL_RT_API_URL
- SEOUL_RT_API_KEY

Optional environment variables:
- SEOUL_RT_API_TYPE (default: json)
- SEOUL_RT_OUTPUT (default: ./data/correction_infer_live.csv)
"""

from __future__ import annotations

import os
from urllib.parse import quote
from typing import Dict, Iterable, List

os.environ.pop("SSLKEYLOGFILE", None)

import numpy as np
import pandas as pd
import requests


def ensure_tmp() -> None:
    tmp = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp, exist_ok=True)
    os.environ["TMP"] = tmp
    os.environ["TEMP"] = tmp
    os.environ["TMPDIR"] = tmp
    os.environ.pop("SSLKEYLOGFILE", None)


def _flatten_records(payload: object) -> List[dict]:
    if isinstance(payload, dict):
        if all(not isinstance(v, (dict, list)) for v in payload.values()):
            return [payload]
        rows: List[dict] = []
        for value in payload.values():
            rows.extend(_flatten_records(value))
        return rows
    if isinstance(payload, list):
        rows: List[dict] = []
        for item in payload:
            rows.extend(_flatten_records(item))
        return rows
    return []


def find_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for col in candidates:
        if col in df.columns:
            return col
        if str(col).strip().lower() in lowered:
            return lowered[str(col).strip().lower()]
    return None


def numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def normalize_realtime_payload(payload: dict) -> pd.DataFrame:
    rows = _flatten_records(payload)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename_candidates: Dict[str, tuple[str, ...]] = {
        "place_code": ("AREA_CD", "area_cd", "place_code"),
        "place_name": ("AREA_NM", "area_nm", "place_name"),
        "timestamp": ("PPLTN_TIME", "timestamp", "ts", "datetime", "time"),
        "real_time_population": (
            "AREA_PPLTN_MIN",
            "AREA_PPLTN_MAX",
            "PPLTN_MIN",
            "population",
            "real_time_population",
        ),
        "traffic_congestion": (
            "AREA_CONGEST_LVL",
            "ROAD_TRAFFIC_IDX",
            "traffic_congestion",
            "congestion",
        ),
        "real_time_temp": ("TEMP", "temperature", "temp", "real_time_temp"),
        "real_time_rain": ("RAIN_CHANCE", "RAIN", "rainfall", "real_time_rain"),
    }

    rename_map = {}
    for target, candidates in rename_candidates.items():
        matched = find_existing_column(df, candidates)
        if matched is not None:
            rename_map[matched] = target
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(pd.Timestamp.now())

    if "real_time_population" not in df.columns:
        # Fall back to congestion-derived pseudo population if the response does not
        # expose a direct population count in the flattened rows.
        base = pd.to_numeric(df.get("traffic_congestion", 0), errors="coerce").fillna(0)
        df["real_time_population"] = 1000 + 300 * base

    if "traffic_congestion" not in df.columns:
        df["traffic_congestion"] = 0.0

    df["real_time_population"] = numeric_series(df, "real_time_population", default=0.0)
    df["traffic_congestion"] = numeric_series(df, "traffic_congestion", default=0.0)
    df["real_time_temp"] = numeric_series(df, "real_time_temp", default=0.0)
    df["real_time_rain"] = numeric_series(df, "real_time_rain", default=0.0)

    if "place_code" not in df.columns:
        df["place_code"] = ""
    if "place_name" not in df.columns:
        df["place_name"] = ""

    out = df[["place_code", "place_name", "timestamp", "real_time_population", "traffic_congestion", "real_time_temp", "real_time_rain"]].copy()
    out["real_time_population_growth"] = 0.0
    out["transit_change"] = 0.0
    out["event_flag"] = 0
    out["holiday_flag"] = 0
    return out


def fetch_one_place(
    api_url: str,
    api_key: str,
    *,
    place_query: str = "",
    place_code: str = "",
    response_type: str = "json",
) -> dict:
    session = requests.Session()
    session.trust_env = False
    if isinstance(place_code, str) and place_code.strip():
        query_value = place_code.strip()
        query_field = "AREA_CD"
    elif isinstance(place_query, str) and place_query.strip():
        query_value = place_query.strip()
        query_field = "AREA_NM"
    else:
        raise ValueError("Either place_code or place_query must be provided.")

    api_url = api_url.strip()
    if "{KEY}" in api_url or "{TYPE}" in api_url or "{QUERY}" in api_url:
        request_url = (
            api_url.replace("{KEY}", api_key)
            .replace("{TYPE}", response_type)
            .replace("{QUERY}", quote(query_value))
        )
        response = session.get(request_url, timeout=30)
    elif "openapi.seoul.go.kr:8088" in api_url and "citydata" not in api_url:
        request_url = f"{api_url.rstrip('/')}/{api_key}/{response_type}/citydata/1/5/{quote(query_value)}"
        response = session.get(request_url, timeout=30)
    else:
        params = {
            "ServiceKey": api_key,
            "type": response_type,
            query_field: query_value,
        }
        response = session.get(
            api_url,
            params=params,
            timeout=30,
        )
    response.raise_for_status()
    return response.json()


def build_live_correction(mapping_path: str, output_path: str) -> None:
    api_url = os.environ.get("SEOUL_RT_API_URL", "http://openapi.seoul.go.kr:8088").strip()
    api_key = os.environ.get("SEOUL_RT_API_KEY", "").strip()
    response_type = os.environ.get("SEOUL_RT_API_TYPE", "json").strip().lower()

    if not api_url or not api_key:
        raise ValueError("SEOUL_RT_API_URL and SEOUL_RT_API_KEY must be set.")

    mapping = pd.read_csv(mapping_path)
    required_cols = {"place_id", "grid_id"}
    if not required_cols.issubset(mapping.columns):
        raise ValueError(f"mapping file must include columns: {sorted(required_cols)}")

    if "place_code" not in mapping.columns:
        mapping["place_code"] = ""
    if "place_name" not in mapping.columns:
        mapping["place_name"] = ""
    if "weight" not in mapping.columns:
        mapping["weight"] = 1.0

    place_cols = ["place_id", "place_code", "place_name"]
    if "api_query" in mapping.columns:
        place_cols.append("api_query")
    place_rows = mapping[place_cols].drop_duplicates()
    fetched_frames: List[pd.DataFrame] = []

    for row in place_rows.itertuples(index=False):
        api_query = getattr(row, "api_query", None) if hasattr(row, "api_query") else None
        place_query = api_query if isinstance(api_query, str) and api_query.strip() else row.place_code
        if not isinstance(place_query, str) or not place_query.strip():
            place_query = row.place_name
        if not isinstance(place_query, str) or not place_query.strip():
            continue
        payload = fetch_one_place(
            api_url,
            api_key,
            place_query=place_query.strip() if isinstance(place_query, str) else "",
            place_code=row.place_code if isinstance(row.place_code, str) else "",
            response_type=response_type,
        )
        normalized = normalize_realtime_payload(payload)
        if normalized.empty:
            continue
        normalized["place_id"] = row.place_id
        fetched_frames.append(normalized)

    if not fetched_frames:
        raise ValueError("No live rows were fetched. Fill place_code/place_name in grid_place_mapping.csv first.")

    realtime = pd.concat(fetched_frames, ignore_index=True)
    merged = mapping.merge(realtime, on="place_id", how="inner")

    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(1.0)
    merged["real_time_population"] = merged["real_time_population"] * merged["weight"]
    merged["traffic_congestion"] = merged["traffic_congestion"] * merged["weight"]
    merged["real_time_temp"] = merged["real_time_temp"] * merged["weight"]
    merged["real_time_rain"] = merged["real_time_rain"] * merged["weight"]

    out = (
        merged.groupby(["grid_id", "timestamp"], as_index=False)
        .agg(
            real_time_population=("real_time_population", "sum"),
            real_time_population_growth=("real_time_population_growth", "mean"),
            traffic_congestion=("traffic_congestion", "mean"),
            transit_change=("transit_change", "mean"),
            real_time_temp=("real_time_temp", "mean"),
            real_time_rain=("real_time_rain", "mean"),
            event_flag=("event_flag", "max"),
            holiday_flag=("holiday_flag", "max"),
        )
        .sort_values(["grid_id", "timestamp"])
        .reset_index(drop=True)
    )

    out.to_csv(output_path, index=False)
    print(f"Saved live correction rows -> {output_path}")
    print(f"rows: {len(out)}")


def main() -> None:
    ensure_tmp()
    mapping_path = os.environ.get("SEOUL_RT_MAPPING_PATH", "./data/grid_place_mapping_api_ready.csv").strip()
    output_path = os.environ.get("SEOUL_RT_OUTPUT", "./data/correction_infer_live.csv").strip()
    build_live_correction(mapping_path, output_path)


if __name__ == "__main__":
    main()
