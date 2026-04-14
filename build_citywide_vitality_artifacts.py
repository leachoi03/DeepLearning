"""
Build whole-Seoul vitality artifacts from live pipeline outputs.

This version uses a more realistic propagation strategy:
1. Keep correction scores for grids directly covered by the live spatial join.
2. Build place-level influence zones from official place polygons joined to KT grids.
3. Propagate correction to uncovered grids using both:
   - place influence decay from nearby live-covered places
   - local neighbor decay from directly covered live grids
   - base-score similarity as a stabilizer
4. Save all-grid scores plus a geographically annotated heatmap.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely import concave_hull
from shapely.geometry import MultiPoint, Point, box

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


BASE_SCORES_CSV = os.environ.get(
    "SEOUL_GRID_BASE_SCORES_CSV",
    "./outputs/live_api_spatial_run/base_scores.csv",
)
LIVE_FINAL_SCORES_CSV = os.environ.get(
    "SEOUL_GRID_LIVE_FINAL_SCORES_CSV",
    "./outputs/live_api_spatial_run/final_scores.csv",
)
SPATIAL_MAPPING_CSV = os.environ.get(
    "SEOUL_GRID_SPATIAL_MAPPING_CSV",
    "./data/grid_place_mapping_spatial.csv",
)
OFFICIAL_PLACE_CSV = os.environ.get(
    "SEOUL_GRID_OFFICIAL_PLACE_CSV",
    "./data/seoul_live_place_catalog_official.csv",
)
SEOUL_GU_BOUNDARY_GEOJSON = os.environ.get(
    "SEOUL_GRID_GU_BOUNDARY_GEOJSON",
    "./data/seoul_municipalities_geo.json",
)
KT_HOURLY_CSV = os.environ.get(
    "SEOUL_GRID_KT_HOURLY_CSV",
    "./source_data/kt_hourly_flow.csv",
)
OUTPUT_DIR = os.environ.get(
    "SEOUL_GRID_CITYWIDE_OUTPUT_DIR",
    "./outputs/citywide_vitality",
)
GRID_SOURCE_CRS = os.environ.get("SEOUL_GRID_SOURCE_CRS", "EPSG:5186")
TARGET_CRS = "EPSG:4326"
MAX_NEIGHBOR_DISTANCE = float(os.environ.get("SEOUL_GRID_PROPAGATION_MAX_DISTANCE", "2200"))
MAX_PLACE_DISTANCE = float(os.environ.get("SEOUL_GRID_PLACE_MAX_DISTANCE", "3000"))
K_NEIGHBORS = int(os.environ.get("SEOUL_GRID_PROPAGATION_K", "6"))
GRID_STEP_DEG = float(os.environ.get("SEOUL_GRID_RENDER_STEP_DEG", "0.0012"))
MIN_PLACE_RADIUS = float(os.environ.get("SEOUL_GRID_MIN_PLACE_RADIUS", "350"))
LABEL_TOP_PLACES = int(os.environ.get("SEOUL_GRID_LABEL_TOP_PLACES", "12"))

SEOUL_GU_CODE_MAP: Dict[str, str] = {
    "11110": "종로구",
    "11140": "중구",
    "11170": "용산구",
    "11200": "성동구",
    "11215": "광진구",
    "11230": "동대문구",
    "11260": "중랑구",
    "11290": "성북구",
    "11305": "강북구",
    "11320": "도봉구",
    "11350": "노원구",
    "11380": "은평구",
    "11410": "서대문구",
    "11440": "마포구",
    "11470": "양천구",
    "11500": "강서구",
    "11530": "구로구",
    "11545": "금천구",
    "11560": "영등포구",
    "11590": "동작구",
    "11620": "관악구",
    "11650": "서초구",
    "11680": "강남구",
    "11710": "송파구",
    "11740": "강동구",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def quantize(series: pd.Series, step: float) -> pd.Series:
    return (np.round(series.astype(float) / step) * step).astype(float)


def exp_decay(distance: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    safe_scale = np.maximum(scale, 1.0)
    return np.exp(-distance / safe_scale)


def to_wgs84(df: pd.DataFrame, x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    transformer = Transformer.from_crs(GRID_SOURCE_CRS, TARGET_CRS, always_xy=True)
    out = df.copy()
    coords = out[[x_col, y_col]].to_numpy(dtype=float)
    lon_lat = np.array([transformer.transform(x, y) for x, y in coords], dtype=float)
    out["lon_wgs84"] = lon_lat[:, 0]
    out["lat_wgs84"] = lon_lat[:, 1]
    return out


def build_place_profiles(
    spatial_df: pd.DataFrame,
    live_df: pd.DataFrame,
    official_df: pd.DataFrame,
) -> pd.DataFrame:
    direct_live = live_df[["grid_id", "correction_score", "base_score"]].drop_duplicates("grid_id")
    merged = spatial_df.merge(direct_live, on="grid_id", how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "place_code",
                "place_name",
                "eng_name",
                "category",
                "place_cx",
                "place_cy",
                "place_radius",
                "place_correction",
                "place_base_score",
                "live_grid_count",
            ]
        )

    if "base_score" not in merged.columns:
        if "base_score_y" in merged.columns:
            merged["base_score"] = merged["base_score_y"]
        elif "base_score_x" in merged.columns:
            merged["base_score"] = merged["base_score_x"]

    merged["dist_to_center"] = np.sqrt(
        (merged["x"] - merged["place_cx"]) ** 2 + (merged["y"] - merged["place_cy"]) ** 2
    )
    profiles = (
        merged.groupby(["place_code", "place_name", "category", "place_cx", "place_cy"], as_index=False)
        .agg(
            place_radius=("dist_to_center", lambda s: float(max(np.percentile(s, 85), MIN_PLACE_RADIUS))),
            place_correction=("correction_score", "mean"),
            place_base_score=("base_score", "mean"),
            live_grid_count=("grid_id", "nunique"),
        )
    )
    profiles = profiles.merge(
        official_df[["place_code", "eng_name"]],
        on="place_code",
        how="left",
    )
    profiles["eng_name"] = profiles["eng_name"].fillna(profiles["place_name"])
    return profiles


def build_gu_profiles(base_df: pd.DataFrame) -> gpd.GeoDataFrame:
    if os.path.exists(SEOUL_GU_BOUNDARY_GEOJSON):
        gu_gdf = gpd.read_file(SEOUL_GU_BOUNDARY_GEOJSON)
        rename_map = {}
        if "name" in gu_gdf.columns:
            rename_map["name"] = "gu_name"
        if "name_eng" in gu_gdf.columns:
            rename_map["name_eng"] = "gu_name_eng"
        gu_gdf = gu_gdf.rename(columns=rename_map)
        if "gu_name" not in gu_gdf.columns:
            raise ValueError("District boundary GeoJSON must include a district name column.")
        if gu_gdf.crs is None:
            gu_gdf = gu_gdf.set_crs(TARGET_CRS)
        else:
            gu_gdf = gu_gdf.to_crs(TARGET_CRS)

        base_points = gpd.GeoDataFrame(
            base_df[["grid_id"]].copy(),
            geometry=gpd.points_from_xy(base_df["lon_wgs84"], base_df["lat_wgs84"]),
            crs=TARGET_CRS,
        )
        joined = gpd.sjoin(base_points, gu_gdf[["gu_name", "geometry"]], how="left", predicate="within")
        counts = joined.groupby("gu_name").size().rename("grid_count").reset_index()
        gu_gdf = gu_gdf.merge(counts, on="gu_name", how="left")
        gu_gdf["grid_count"] = gu_gdf["grid_count"].fillna(0).astype(int)
        centroids = gu_gdf.geometry.representative_point()
        gu_gdf["gu_lon"] = centroids.x
        gu_gdf["gu_lat"] = centroids.y
        return gu_gdf

    hourly = pd.read_csv(KT_HOURLY_CSV, encoding="cp949", usecols=["셀id(ID)", "행정동코드(ADMI_CD)"])
    hourly = hourly.rename(columns={"셀id(ID)": "grid_id", "행정동코드(ADMI_CD)": "admi_cd"})
    hourly["gu_code"] = hourly["admi_cd"].astype(str).str[:5]
    hourly["gu_name"] = hourly["gu_code"].map(SEOUL_GU_CODE_MAP)
    hourly = hourly.dropna(subset=["gu_name"])

    grid_gu = (
        hourly.groupby(["grid_id", "gu_name"], as_index=False)
        .size()
        .sort_values(["grid_id", "size"], ascending=[True, False])
        .drop_duplicates(subset=["grid_id"], keep="first")
        [["grid_id", "gu_name"]]
    )

    gu_df = base_df.merge(grid_gu, on="grid_id", how="left").dropna(subset=["gu_name"]).copy()
    if gu_df.empty:
        return gpd.GeoDataFrame(columns=["gu_name", "grid_count", "geometry"], geometry="geometry", crs=TARGET_CRS)

    records = []
    for gu_name, group in gu_df.groupby("gu_name"):
        points = [Point(lon, lat) for lon, lat in group[["lon_wgs84", "lat_wgs84"]].to_numpy(dtype=float)]
        if len(points) >= 4:
            hull = concave_hull(MultiPoint(points), ratio=0.35)
        elif len(points) >= 2:
            hull = MultiPoint(points).convex_hull.buffer(0.003)
        else:
            hull = points[0].buffer(0.004)
        if hull.is_empty:
            hull = MultiPoint(points).convex_hull.buffer(0.003)
        records.append(
            {
                "gu_name": gu_name,
                "grid_count": int(group["grid_id"].nunique()),
                "gu_lon": float(group["lon_wgs84"].mean()),
                "gu_lat": float(group["lat_wgs84"].mean()),
                "geometry": hull,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=TARGET_CRS)


def compute_neighbor_signal(
    target_xy: np.ndarray,
    target_base: float,
    live_xy: np.ndarray,
    live_corr: np.ndarray,
    live_base: np.ndarray,
) -> tuple[float, float]:
    if len(live_xy) == 0:
        return 0.0, 0.0

    distances = np.sqrt(((live_xy - target_xy) ** 2).sum(axis=1))
    exact_idx = np.where(distances < 1e-9)[0]
    if len(exact_idx):
        idx = exact_idx[0]
        return float(live_corr[idx]), 1.0

    order = np.argsort(distances)
    selected = order[:K_NEIGHBORS]
    selected = selected[distances[selected] <= MAX_NEIGHBOR_DISTANCE]
    if len(selected) == 0:
        return 0.0, 0.0

    selected_dist = distances[selected]
    base_gap = np.abs(live_base[selected] - target_base)
    base_scale = max(float(np.std(live_base)), 0.5)
    dist_w = exp_decay(selected_dist, MAX_NEIGHBOR_DISTANCE / 2.5)
    base_w = np.exp(-base_gap / base_scale)
    weights = dist_w * base_w
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return 0.0, 0.0
    value = float(np.sum(weights * live_corr[selected]) / weight_sum)
    confidence = float(np.clip(weight_sum / len(selected), 0.0, 1.0))
    return value, confidence


def compute_place_signal(
    target_xy: np.ndarray,
    target_base: float,
    place_profiles: pd.DataFrame,
) -> tuple[float, float, str]:
    if place_profiles.empty:
        return 0.0, 0.0, ""

    centers = place_profiles[["place_cx", "place_cy"]].to_numpy(dtype=float)
    distances = np.sqrt(((centers - target_xy) ** 2).sum(axis=1))
    active = distances <= MAX_PLACE_DISTANCE
    if not np.any(active):
        return 0.0, 0.0, ""

    sub = place_profiles.loc[active].copy()
    sub["distance"] = distances[active]
    base_scale = max(float(place_profiles["place_base_score"].std()), 0.5)
    dist_w = exp_decay(sub["distance"].to_numpy(dtype=float), sub["place_radius"].to_numpy(dtype=float) * 1.2)
    base_w = np.exp(-np.abs(sub["place_base_score"].to_numpy(dtype=float) - target_base) / base_scale)
    strength_w = np.sqrt(np.maximum(sub["live_grid_count"].to_numpy(dtype=float), 1.0))
    weights = dist_w * base_w * strength_w
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return 0.0, 0.0, ""

    value = float(np.sum(weights * sub["place_correction"].to_numpy(dtype=float)) / weight_sum)
    confidence = float(np.clip(weight_sum / max(len(sub), 1), 0.0, 1.0))
    best_place = str(sub.iloc[int(np.argmax(weights))]["eng_name"])
    return value, confidence, best_place


def render_geographic_heatmap(
    citywide_df: pd.DataFrame,
    place_profiles: pd.DataFrame,
    gu_profiles: gpd.GeoDataFrame,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 12))
    cell_half = 25.0
    grid_gdf = gpd.GeoDataFrame(
        citywide_df.copy(),
        geometry=[box(x - cell_half, y - cell_half, x + cell_half, y + cell_half) for x, y in citywide_df[["x", "y"]].to_numpy(dtype=float)],
        crs=GRID_SOURCE_CRS,
    ).to_crs(TARGET_CRS)

    city_outline = concave_hull(MultiPoint([geom.centroid for geom in grid_gdf.geometry]), ratio=0.28)
    boundary_gdf = gpd.GeoDataFrame([{"geometry": city_outline}], geometry="geometry", crs=TARGET_CRS)

    boundary_gdf.boundary.plot(ax=ax, color="#6c757d", linewidth=1.0, alpha=0.7)
    if not gu_profiles.empty:
        gu_profiles.boundary.plot(ax=ax, color="#184e77", linewidth=0.9, alpha=0.75)

    grid_plot = grid_gdf.plot(
        ax=ax,
        column="final_score_citywide",
        cmap="YlOrRd",
        linewidth=0.0,
        edgecolor="none",
        legend=True,
        legend_kwds={"label": "Vitality score", "shrink": 0.78},
    )
    ax.set_title("Seoul Citywide Grid Vitality")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(color="#c8d1da", alpha=0.15, linewidth=0.4)

    label_df = place_profiles.sort_values(["live_grid_count", "place_correction"], ascending=False).head(LABEL_TOP_PLACES)
    for row in label_df.itertuples(index=False):
        ax.scatter(row.place_centroid_lon, row.place_centroid_lat, s=18, c="#163a5f", alpha=0.85)
        ax.text(
            row.place_centroid_lon + 0.0025,
            row.place_centroid_lat + 0.0018,
            row.eng_name,
            fontsize=8,
            color="#163a5f",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.8},
        )

    for row in gu_profiles.itertuples(index=False):
        label_point = row.geometry.representative_point()
        ax.text(
            label_point.x,
            label_point.y,
            row.gu_name,
            fontsize=8,
            color="#0b2545",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "#184e77", "linewidth": 0.4, "pad": 1.2},
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    grid_gdf[["grid_id", "final_score_citywide", "score_source", "driver_place", "geometry"]].to_csv(
        os.path.join(os.path.dirname(output_path), "citywide_vitality_matrix.csv"),
        index=False,
    )


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    base_df = pd.read_csv(BASE_SCORES_CSV).rename(columns={"lon": "x", "lat": "y"})
    live_df = pd.read_csv(LIVE_FINAL_SCORES_CSV).rename(columns={"lon": "x", "lat": "y"})
    spatial_df = pd.read_csv(SPATIAL_MAPPING_CSV)
    official_df = pd.read_csv(OFFICIAL_PLACE_CSV)

    required_base = {"grid_id", "base_score", "x", "y"}
    required_live = {"grid_id", "correction_score", "base_score", "x", "y"}
    if not required_base.issubset(base_df.columns):
        raise ValueError(f"Base scores CSV must include: {sorted(required_base)}")
    if not required_live.issubset(live_df.columns):
        raise ValueError(f"Live final scores CSV must include: {sorted(required_live)}")

    base_df = base_df.drop_duplicates(subset=["grid_id"]).copy()
    live_df = live_df.sort_values("timestamp").drop_duplicates(subset=["grid_id"], keep="last").copy()

    spatial_df = spatial_df.rename(
        columns={
            "grid_lon_wgs84": "lon_wgs84",
            "grid_lat_wgs84": "lat_wgs84",
            "place_centroid_lon": "place_centroid_lon",
            "place_centroid_lat": "place_centroid_lat",
        }
    )
    if "eng_name" not in official_df.columns:
        official_df["eng_name"] = official_df["place_name"]

    transformer = Transformer.from_crs(TARGET_CRS, GRID_SOURCE_CRS, always_xy=True)
    to_geo_transformer = Transformer.from_crs(GRID_SOURCE_CRS, TARGET_CRS, always_xy=True)
    spatial_df[["place_cx", "place_cy"]] = spatial_df.apply(
        lambda row: pd.Series(
            transformer.transform(float(row["place_centroid_lon"]), float(row["place_centroid_lat"]))
        ),
        axis=1,
    )
    spatial_df = spatial_df.merge(
        base_df[["grid_id", "x", "y", "base_score"]],
        on="grid_id",
        how="left",
    )

    base_df = to_wgs84(base_df, x_col="x", y_col="y")
    live_df = to_wgs84(live_df, x_col="x", y_col="y")
    gu_profiles = build_gu_profiles(base_df)

    place_profiles = build_place_profiles(spatial_df, live_df, official_df)
    if not place_profiles.empty:
        place_profiles[["place_centroid_lon", "place_centroid_lat"]] = place_profiles.apply(
            lambda row: pd.Series(to_geo_transformer.transform(float(row["place_cx"]), float(row["place_cy"]))),
            axis=1,
        )

    live_lookup = live_df.set_index("grid_id")["correction_score"].to_dict()
    live_xy = live_df[["x", "y"]].to_numpy(dtype=float)
    live_corr = live_df["correction_score"].to_numpy(dtype=float)
    live_base = live_df["base_score"].to_numpy(dtype=float)

    propagated_values: List[float] = []
    source_types: List[str] = []
    driver_places: List[str] = []
    neighbor_confs: List[float] = []
    place_confs: List[float] = []

    for row in base_df.itertuples(index=False):
        grid_id = str(row.grid_id)
        if grid_id in live_lookup:
            propagated_values.append(float(live_lookup[grid_id]))
            source_types.append("DIRECT_LIVE")
            driver_places.append("")
            neighbor_confs.append(1.0)
            place_confs.append(1.0)
            continue

        target_xy = np.array([float(row.x), float(row.y)])
        target_base = float(row.base_score)
        neighbor_signal, neighbor_conf = compute_neighbor_signal(
            target_xy=target_xy,
            target_base=target_base,
            live_xy=live_xy,
            live_corr=live_corr,
            live_base=live_base,
        )
        place_signal, place_conf, best_place = compute_place_signal(
            target_xy=target_xy,
            target_base=target_base,
            place_profiles=place_profiles,
        )

        if neighbor_conf > 0 and place_conf > 0:
            total = neighbor_conf + place_conf
            value = (neighbor_signal * neighbor_conf + place_signal * place_conf) / total
            source_type = "HYBRID_PROPAGATED"
        elif place_conf > 0:
            value = place_signal
            source_type = "PLACE_PROPAGATED"
        elif neighbor_conf > 0:
            value = neighbor_signal
            source_type = "NEIGHBOR_PROPAGATED"
        else:
            value = 0.0
            source_type = "BASE_ONLY"

        propagated_values.append(float(value))
        source_types.append(source_type)
        driver_places.append(best_place)
        neighbor_confs.append(neighbor_conf)
        place_confs.append(place_conf)

    citywide_df = base_df.copy()
    citywide_df["correction_score_citywide"] = propagated_values
    citywide_df["score_source"] = source_types
    citywide_df["driver_place"] = driver_places
    citywide_df["neighbor_confidence"] = neighbor_confs
    citywide_df["place_confidence"] = place_confs
    citywide_df["final_score_citywide"] = citywide_df["base_score"] + citywide_df["correction_score_citywide"]
    citywide_df = citywide_df.sort_values("final_score_citywide", ascending=False).reset_index(drop=True)

    citywide_csv = os.path.join(OUTPUT_DIR, "citywide_final_scores.csv")
    citywide_df.to_csv(citywide_csv, index=False)

    heatmap_path = os.path.join(OUTPUT_DIR, "citywide_vitality_heatmap.png")
    render_geographic_heatmap(citywide_df, place_profiles, gu_profiles, heatmap_path)

    summary = (
        citywide_df.groupby("score_source")
        .size()
        .rename("grid_count")
        .reset_index()
        .sort_values("grid_count", ascending=False)
    )
    summary.to_csv(os.path.join(OUTPUT_DIR, "citywide_score_source_summary.csv"), index=False)
    place_profiles.to_csv(os.path.join(OUTPUT_DIR, "citywide_place_profiles.csv"), index=False)
    gu_profiles.to_csv(os.path.join(OUTPUT_DIR, "citywide_gu_profiles.csv"), index=False)

    print(f"Saved citywide scores -> {citywide_csv}")
    print(f"Saved citywide heatmap -> {heatmap_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
