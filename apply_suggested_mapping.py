"""
Apply the suggested Seoul place mapping into the main grid_place_mapping.csv.

This is a heuristic bootstrap step. The result should still be manually reviewed
before production OpenAPI use.
"""

from __future__ import annotations

import os
import pandas as pd


def main() -> None:
    data_dir = "./data"
    mapping_path = os.path.join(data_dir, "grid_place_mapping.csv")
    suggested_path = os.path.join(data_dir, "grid_place_mapping_suggested.csv")
    priority_path = os.path.join(data_dir, "grid_place_mapping_priority.csv")

    mapping = pd.read_csv(mapping_path)
    suggested = pd.read_csv(suggested_path)

    if "review_status" not in mapping.columns:
        mapping["review_status"] = "UNMAPPED"
    if "mapping_source" not in mapping.columns:
        mapping["mapping_source"] = ""

    fill_cols = ["place_code", "place_name", "weight"]
    suggested_keep = ["grid_id"] + [col for col in ["place_code", "place_name", "weight"] if col in suggested.columns]
    merged = mapping.drop(columns=[col for col in fill_cols if col in mapping.columns], errors="ignore").merge(
        suggested[suggested_keep], on="grid_id", how="left"
    )

    for col in fill_cols:
        if col not in merged.columns:
            if col == "weight":
                merged[col] = 1.0
            else:
                merged[col] = ""
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(1.0)

    has_name = merged["place_name"].fillna("").astype(str).str.strip() != ""
    merged["review_status"] = merged["review_status"].where(~has_name, "SUGGESTED")
    merged["mapping_source"] = merged["mapping_source"].where(~has_name, "rank_based_suggestion")

    ordered_cols = ["place_id", "place_code", "place_name", "grid_id", "weight", "review_status", "mapping_source"]
    other_cols = [col for col in merged.columns if col not in ordered_cols]
    merged = merged[ordered_cols + other_cols]
    merged.to_csv(mapping_path, index=False)

    priority = merged[merged["review_status"] == "SUGGESTED"].head(30).copy()
    priority.to_csv(priority_path, index=False)

    print(f"Updated main mapping -> {mapping_path}")
    print(f"Saved priority review subset -> {priority_path}")
    print(f"suggested rows: {int((merged['review_status'] == 'SUGGESTED').sum())}")


if __name__ == "__main__":
    main()
