"""
Render the current base-grid coverage footprint against real Seoul district
boundaries so covered and uncovered areas are visually separated.
"""

from __future__ import annotations

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

BASE_INFER_CSV = os.environ.get(
    "SEOUL_GRID_BASE_INFER_CSV",
    "./data/base_infer.csv",
)
SEOUL_GU_BOUNDARY_GEOJSON = os.environ.get(
    "SEOUL_GRID_GU_BOUNDARY_GEOJSON",
    "./data/seoul_municipalities_geo.json",
)
OUTPUT_DIR = os.environ.get(
    "SEOUL_GRID_COVERAGE_OUTPUT_DIR",
    "./outputs/coverage",
)
GRID_SOURCE_CRS = os.environ.get("SEOUL_GRID_SOURCE_CRS", "EPSG:5186")
TARGET_CRS = "EPSG:4326"
GRID_SIZE_METERS = float(os.environ.get("SEOUL_GRID_SIZE_METERS", "50"))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    base_df = pd.read_csv(BASE_INFER_CSV)
    if not {"grid_id", "lon", "lat"}.issubset(base_df.columns):
        raise ValueError("base_infer.csv must include grid_id, lon, lat columns.")

    gu_gdf = gpd.read_file(SEOUL_GU_BOUNDARY_GEOJSON)
    gu_gdf = gu_gdf.rename(columns={"name": "gu_name", "name_eng": "gu_name_eng"})
    if gu_gdf.crs is None:
        gu_gdf = gu_gdf.set_crs(TARGET_CRS)
    else:
        gu_gdf = gu_gdf.to_crs(TARGET_CRS)

    half = GRID_SIZE_METERS / 2.0
    grid_gdf = gpd.GeoDataFrame(
        base_df[["grid_id"]].copy(),
        geometry=[box(x - half, y - half, x + half, y + half) for x, y in base_df[["lon", "lat"]].to_numpy(dtype=float)],
        crs=GRID_SOURCE_CRS,
    ).to_crs(TARGET_CRS)

    covered_geom = grid_gdf.geometry.union_all()
    seoul_geom = gu_gdf.geometry.union_all()
    uncovered_geom = seoul_geom.difference(covered_geom)

    covered_gdf = gpd.GeoDataFrame([{"layer": "covered", "geometry": covered_geom}], geometry="geometry", crs=TARGET_CRS)
    uncovered_gdf = gpd.GeoDataFrame([{"layer": "uncovered", "geometry": uncovered_geom}], geometry="geometry", crs=TARGET_CRS)

    fig, ax = plt.subplots(figsize=(14, 14))
    uncovered_gdf.plot(ax=ax, color="#e6e6e6", edgecolor="none", alpha=1.0)
    gu_gdf.boundary.plot(ax=ax, color="#7d8597", linewidth=0.9, alpha=0.8)
    covered_gdf.plot(ax=ax, color="#d1495b", edgecolor="none", alpha=0.9)

    label_points = gu_gdf.geometry.representative_point()
    for row, pt in zip(gu_gdf.itertuples(index=False), label_points):
        gu_label = getattr(row, "gu_name_eng", None) or getattr(row, "gu_name", "")
        ax.text(
            pt.x,
            pt.y,
            gu_label,
            fontsize=8,
            color="#1d3557",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.2},
        )

    ax.set_title("Current Base Grid Coverage vs Uncovered Seoul Areas")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(color="#c8d1da", alpha=0.12, linewidth=0.4)

    coverage_note = (
        f"Covered grids: {len(grid_gdf)}\n"
        f"Covered layer: current base grid footprint\n"
        f"Gray area: no base grid coverage"
    )
    ax.text(
        0.02,
        0.98,
        coverage_note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#adb5bd", "pad": 4},
    )

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "base_grid_coverage_map.png")
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    grid_points = grid_gdf.copy()
    grid_points["lon_wgs84"] = grid_points.geometry.centroid.x
    grid_points["lat_wgs84"] = grid_points.geometry.centroid.y
    joined = gpd.sjoin(
        grid_points[["grid_id", "lon_wgs84", "lat_wgs84", "geometry"]],
        gu_gdf[["gu_name", "geometry"]],
        how="left",
        predicate="intersects",
    )
    summary = joined.groupby("gu_name").size().rename("covered_grid_count").reset_index().sort_values("covered_grid_count", ascending=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "base_grid_coverage_by_gu.csv"), index=False)

    print(f"Saved coverage map -> {png_path}")
    print(f"Saved coverage summary -> {os.path.join(OUTPUT_DIR, 'base_grid_coverage_by_gu.csv')}")


if __name__ == "__main__":
    main()
