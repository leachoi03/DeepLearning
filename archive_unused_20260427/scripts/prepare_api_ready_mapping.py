"""
Prepare an API-ready mapping subset from the reviewed priority mapping rows.

This script takes the P1 rows from `grid_place_mapping_priority_review.csv`,
normalizes the place names for direct AREA_NM querying, and writes a compact
mapping file for live API fetching.
"""

from __future__ import annotations

import os
import pandas as pd


def normalize_place_name(name: str) -> str:
    text = str(name).strip()
    replacements = {
        "홍대입구역(2호선)": "홍대입구역(2호선)",
        "광화문·덕수궁": "광화문·덕수궁",
        "신논현역·논현역": "신논현역·논현역",
        "신촌·이대역": "신촌·이대역",
        "오목교역·목동운동장": "오목교역·목동운동장",
        "국립중앙박물관·용산가족공원": "국립중앙박물관·용산가족공원",
        "DDP(동대문디자인플라자)": "DDP(동대문디자인플라자)",
    }
    return replacements.get(text, text)


def main() -> None:
    data_dir = "./data"
    mapping_path = os.path.join(data_dir, "grid_place_mapping.csv")
    review_path = os.path.join(data_dir, "grid_place_mapping_priority_review.csv")
    api_ready_path = os.path.join(data_dir, "grid_place_mapping_api_ready.csv")
    p1_path = os.path.join(data_dir, "grid_place_mapping_p1.csv")

    mapping = pd.read_csv(mapping_path)
    review = pd.read_csv(review_path)

    p1 = review[review["priority"] == "P1"].copy()
    p1["api_query"] = p1["place_name"].map(normalize_place_name)
    p1["api_type"] = "AREA_NM"
    p1["manual_check"] = "CHECK_REQUIRED"
    p1.to_csv(p1_path, index=False)

    keep_cols = ["place_id", "grid_id", "weight"]
    api_ready = mapping.merge(
        p1[["place_id", "place_name", "api_query", "priority"]],
        on=["place_id", "place_name"],
        how="inner",
    )
    api_ready["place_code"] = api_ready.get("place_code", "")
    api_ready["place_code"] = api_ready["place_code"].fillna("")
    api_ready["api_query"] = api_ready["api_query"].fillna(api_ready["place_name"])
    api_ready["weight"] = pd.to_numeric(api_ready["weight"], errors="coerce").fillna(1.0)
    api_ready["review_status"] = "API_READY_P1"

    ordered_cols = [
        "place_id",
        "place_code",
        "place_name",
        "api_query",
        "grid_id",
        "weight",
        "priority",
        "review_status",
    ]
    api_ready = api_ready[ordered_cols].sort_values(["priority", "place_name", "grid_id"])
    api_ready.to_csv(api_ready_path, index=False)

    print(f"Saved P1 review subset -> {p1_path}")
    print(f"Saved API-ready mapping -> {api_ready_path}")
    print(f"P1 rows: {len(p1)}")
    print(f"API-ready rows: {len(api_ready)}")


if __name__ == "__main__":
    main()
