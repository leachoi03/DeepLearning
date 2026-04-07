"""
Import the official Seoul 121-place Excel file into CSV assets used by the project.
"""

from __future__ import annotations

import os
import pandas as pd


def main() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(root_dir, "source_data", "seoul_major_places_official.xlsx")
    output_path = os.path.join(root_dir, "data", "seoul_live_place_catalog_official.csv")

    df = pd.read_excel(source_path, sheet_name="장소목록")
    df = df.rename(
        columns={
            "CATEGORY": "category",
            "NO": "no",
            "AREA_CD": "place_code",
            "AREA_NM": "place_name",
            "ENG_NM": "eng_name",
        }
    )
    df.to_csv(output_path, index=False)
    print(f"Saved official place catalog -> {output_path}")
    print(f"rows: {len(df)}")


if __name__ == "__main__":
    main()
