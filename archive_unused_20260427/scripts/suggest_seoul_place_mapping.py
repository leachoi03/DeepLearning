"""
Create a suggested grid-place mapping using the official 121 Seoul real-time
place names and the locally generated base scores.

This is a starting point, not a ground-truth spatial join.
It assigns the highest-scoring grids to the official place list in descending
order and writes a reviewable CSV for manual refinement.
"""

from __future__ import annotations

import os
import pandas as pd


def main() -> None:
    data_dir = "./data"
    base_infer_path = os.path.join(data_dir, "base_infer.csv")
    catalog_path = os.path.join(data_dir, "seoul_live_place_catalog.csv")
    output_path = os.path.join(data_dir, "grid_place_mapping_suggested.csv")

    base = pd.read_csv(base_infer_path)
    catalog = pd.read_csv(catalog_path)

    sort_col = "avg_flow" if "avg_flow" in base.columns else "grid_id"
    ranked_grids = base.sort_values(sort_col, ascending=False).reset_index(drop=True).copy()

    n = min(len(catalog), len(ranked_grids))
    suggested = pd.DataFrame(
        {
            "place_id": [f"PLACE_{idx:03d}" for idx in range(1, n + 1)],
            "place_code": "",
            "place_name": catalog.loc[: n - 1, "place_name"].to_numpy(),
            "category": catalog.loc[: n - 1, "category"].to_numpy(),
            "grid_id": ranked_grids.loc[: n - 1, "grid_id"].to_numpy(),
            "weight": 1.0,
            "grid_rank_basis": sort_col,
        }
    )

    suggested.to_csv(output_path, index=False)
    print(f"Saved suggested mapping -> {output_path}")
    print(f"rows: {len(suggested)}")


if __name__ == "__main__":
    main()
