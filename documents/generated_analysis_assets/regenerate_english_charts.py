from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "documents" / "generated_analysis_assets"
DATA = ROOT / "outputs"


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    source = pd.read_csv(DATA / "citywide_vitality" / "citywide_score_source_summary.csv")
    city = pd.read_csv(DATA / "citywide_vitality" / "citywide_final_scores.csv")
    place = pd.read_csv(DATA / "citywide_vitality" / "citywide_place_profiles.csv")

    source_labels = {
        "HYBRID_PROPAGATED": "Hybrid propagated",
        "BASE_ONLY": "Base only",
        "DIRECT_LIVE": "Direct live",
        "PLACE_PROPAGATED": "Place propagated",
    }
    source_plot = source.copy()
    source_plot["label"] = source_plot["score_source"].map(source_labels).fillna(source_plot["score_source"])
    source_plot = source_plot.sort_values("grid_count", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=180)
    colors = ["#A7C957", "#2A9D8F", "#E9C46A", "#E76F51"]
    ax.barh(source_plot["label"], source_plot["grid_count"], color=colors[: len(source_plot)])
    ax.set_title("Grid Count by Score Source", fontsize=14, pad=12)
    ax.set_xlabel("Number of grids")
    for i, value in enumerate(source_plot["grid_count"]):
        ax.text(value + 2, i, str(value), va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "score_source_counts.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=180)
    ax.hist(city["final_score_citywide"], bins=24, color="#2A9D8F", edgecolor="white")
    mean = city["final_score_citywide"].mean()
    ax.axvline(mean, color="#E76F51", lw=2, label=f"Mean {mean:.2f}")
    ax.set_title("Distribution of Final Vitality Scores", fontsize=14, pad=12)
    ax.set_xlabel("Final vitality score")
    ax.set_ylabel("Number of grids")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "score_distribution.png", bbox_inches="tight")
    plt.close(fig)

    place_labels = {
        "경복궁": "Gyeongbokgung",
        "광화문·덕수궁": "Gwanghwamun & Deoksugung",
        "창덕궁·종묘": "Changdeokgung & Jongmyo",
        "북촌한옥마을": "Bukchon Hanok Village",
        "서촌": "Seochon",
        "인사동": "Insa-dong",
        "익선동": "Ikseon-dong",
        "송현녹지광장": "Songhyeon Green Plaza",
    }
    place_plot = place.copy()
    place_plot["label"] = place_plot["place_name"].map(place_labels).fillna(place_plot["eng_name"])
    place_plot = place_plot.sort_values("place_base_score")

    fig, (ax_base, ax_corr) = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(9.8, 5.0),
        dpi=180,
        gridspec_kw={"width_ratios": [3.3, 1.8], "wspace": 0.08},
    )
    y = range(len(place_plot))

    ax_base.barh(y, place_plot["place_base_score"], color="#264653")
    ax_base.set_yticks(list(y), place_plot["label"])
    ax_base.set_title("Base vitality", fontsize=12.5, pad=8)
    ax_base.set_xlabel("Base score")
    ax_base.set_xlim(0, max(place_plot["place_base_score"]) * 1.12)
    for i, value in enumerate(place_plot["place_base_score"]):
        ax_base.text(value + 0.04, i, f"{value:.2f}", va="center", fontsize=8.5, color="#264653")

    ax_corr.barh(y, place_plot["place_correction"], color="#E76F51")
    ax_corr.set_title("Live correction", fontsize=12.5, pad=8)
    ax_corr.set_xlabel("Correction score")
    ax_corr.set_xlim(0, max(place_plot["place_correction"]) * 1.22)
    ax_corr.tick_params(axis="y", left=False, labelleft=False)
    for i, value in enumerate(place_plot["place_correction"]):
        ax_corr.text(value + 0.0012, i, f"{value:.3f}", va="center", fontsize=8.5, color="#9A3412")

    fig.suptitle("Place Profile: Base Vitality vs. Live Correction", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.025,
        "The two metrics use separate x-axes; correction scores are shown at their actual scale.",
        ha="center",
        fontsize=8.5,
        color="#5C6B73",
    )
    for ax in (ax_base, ax_corr):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.82, bottom=0.17, wspace=0.08)
    fig.savefig(OUT / "place_scores.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
