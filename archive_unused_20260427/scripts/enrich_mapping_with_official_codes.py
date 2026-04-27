"""
Fill place_code values in mapping files by exact place_name match against the
official Seoul 121-place catalog.
"""

from __future__ import annotations

import os
import pandas as pd


def enrich_one(path: str, official: pd.DataFrame) -> None:
    df = pd.read_csv(path)
    if "place_name" not in df.columns:
        return

    lookup = official[["place_name", "place_code", "category"]].drop_duplicates()
    if "place_code" in df.columns:
        df = df.rename(columns={"place_code": "place_code_existing"})
    if "category" in df.columns:
        df = df.rename(columns={"category": "category_existing"})

    merged = df.merge(lookup, on="place_name", how="left")

    merged["place_code_existing"] = merged.get("place_code_existing", pd.Series(index=merged.index, dtype=object))
    merged["place_code"] = merged["place_code_existing"].fillna(merged["place_code"])
    merged["category_existing"] = merged.get("category_existing", pd.Series(index=merged.index, dtype=object))
    merged["category"] = merged["category_existing"].fillna(merged["category"])

    if "review_status" in merged.columns:
        matched = merged["place_code"].fillna("").astype(str).str.strip() != ""
        merged["review_status"] = merged["review_status"].where(~matched, "CODE_MATCHED")

    drop_cols = [col for col in ["place_code_existing", "category_existing"] if col in merged.columns]
    merged = merged.drop(columns=drop_cols)
    merged.to_csv(path, index=False)
    print(f"Updated -> {path}")


def main() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    official_path = os.path.join(root_dir, "data", "seoul_live_place_catalog_official.csv")
    official = pd.read_csv(official_path)

    targets = [
        os.path.join(root_dir, "data", "grid_place_mapping.csv"),
        os.path.join(root_dir, "data", "grid_place_mapping_priority.csv"),
        os.path.join(root_dir, "data", "grid_place_mapping_priority_review.csv"),
        os.path.join(root_dir, "data", "grid_place_mapping_p1.csv"),
        os.path.join(root_dir, "data", "grid_place_mapping_api_ready.csv"),
    ]
    for path in targets:
        if os.path.exists(path):
            enrich_one(path, official)


if __name__ == "__main__":
    main()
