"""
Build a place-to-grid mapping by spatially joining KT grid centroids to the
official Seoul 121-place polygons.

Assumption:
- KT grid coordinates are in EPSG:5186, based on coordinate range validation.
- Official 121-place polygons are in WGS84, as declared in the `.prj` file.
"""

from __future__ import annotations

import math
import os
from typing import Iterable, List

import pandas as pd
import shapefile
from pyproj import Transformer


BASE_INFER_CSV = os.environ.get("SEOUL_GRID_BASE_INFER_CSV", "./data/base_infer.csv")
PLACE_SHP_PATH = os.environ.get(
    "SEOUL_GRID_PLACE_SHP_PATH",
    "./source_data/seoul_major_places_area_ascii/seoul_121_places_area.shp",
)
OUTPUT_PATH = os.environ.get(
    "SEOUL_GRID_SPATIAL_MAPPING_CSV",
    "./data/grid_place_mapping_spatial.csv",
)
GRID_SOURCE_CRS = os.environ.get("SEOUL_GRID_SOURCE_CRS", "EPSG:5186")
TARGET_CRS = "EPSG:4326"


def point_in_ring(x: float, y: float, ring: List[tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_in_shape(x: float, y: float, shp: shapefile.Shape) -> bool:
    points = shp.points
    part_starts = list(shp.parts) + [len(points)]
    inside_any = False
    for idx in range(len(part_starts) - 1):
        ring = points[part_starts[idx] : part_starts[idx + 1]]
        if len(ring) >= 3 and point_in_ring(x, y, ring):
            inside_any = not inside_any
    return inside_any


def polygon_centroid(shp: shapefile.Shape) -> tuple[float, float]:
    minx, miny, maxx, maxy = shp.bbox
    return (minx + maxx) / 2.0, (miny + maxy) / 2.0


def main() -> None:
    base_df = pd.read_csv(BASE_INFER_CSV)
    required = {"grid_id", "avg_flow"}
    if not required.issubset(base_df.columns):
        raise ValueError(f"base infer file must include: {sorted(required)}")

    x_col = "lon" if "lon" in base_df.columns else "x"
    y_col = "lat" if "lat" in base_df.columns else "y"
    if x_col not in base_df.columns or y_col not in base_df.columns:
        raise ValueError("base infer file must include grid coordinates.")

    transformer = Transformer.from_crs(GRID_SOURCE_CRS, TARGET_CRS, always_xy=True)
    base_df[["grid_lon_wgs84", "grid_lat_wgs84"]] = base_df.apply(
        lambda row: pd.Series(transformer.transform(float(row[x_col]), float(row[y_col]))),
        axis=1,
    )

    reader = shapefile.Reader(PLACE_SHP_PATH, encoding="utf-8")
    rows: List[dict] = []

    for shape_record in reader.shapeRecords():
        record = shape_record.record.as_dict()
        shp = shape_record.shape
        centroid_lon, centroid_lat = polygon_centroid(shp)

        matched = base_df[
            base_df.apply(
                lambda row: point_in_shape(
                    float(row["grid_lon_wgs84"]),
                    float(row["grid_lat_wgs84"]),
                    shp,
                ),
                axis=1,
            )
        ].copy()

        if matched.empty:
            continue

        flow_sum = matched["avg_flow"].clip(lower=0).sum()
        if flow_sum <= 0:
            matched["weight"] = 1.0 / len(matched)
        else:
            matched["weight"] = matched["avg_flow"].clip(lower=0) / flow_sum

        for row in matched.itertuples(index=False):
            rows.append(
                {
                    "place_id": record["AREA_CD"],
                    "place_code": record["AREA_CD"],
                    "place_name": record["AREA_NM"],
                    "api_query": record["AREA_CD"],
                    "category": record["CATEGORY"],
                    "grid_id": row.grid_id,
                    "weight": float(row.weight),
                    "mapping_source": "SPATIAL_JOIN",
                    "review_status": "SPATIAL_MATCHED",
                    "grid_lon_wgs84": float(row.grid_lon_wgs84),
                    "grid_lat_wgs84": float(row.grid_lat_wgs84),
                    "place_centroid_lon": centroid_lon,
                    "place_centroid_lat": centroid_lat,
                }
            )

    if not rows:
        raise ValueError("No spatial matches were found. Re-check source CRS.")

    out_df = pd.DataFrame(rows).sort_values(["place_code", "grid_id"]).reset_index(drop=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved spatial mapping -> {OUTPUT_PATH}")
    print(f"rows: {len(out_df)}")
    print(f"places: {out_df['place_code'].nunique()}")
    print(f"grids: {out_df['grid_id'].nunique()}")


if __name__ == "__main__":
    main()
