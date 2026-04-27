"""
Build a human-review guide for the priority Seoul grid-place mapping rows.
"""

from __future__ import annotations

import os
import pandas as pd


def classify_priority(row: pd.Series) -> str:
    avg_flow = float(row.get("avg_flow", 0.0))
    name = str(row.get("place_name", "") or "")

    if avg_flow > 100:
        return "P1"
    if any(keyword in name for keyword in ["역", "관광특구", "경복궁", "DDP", "성수", "홍대", "강남"]):
        return "P1"
    if avg_flow > 20:
        return "P2"
    return "P3"


def review_note(row: pd.Series) -> str:
    name = str(row.get("place_name", "") or "")
    avg_flow = float(row.get("avg_flow", 0.0))
    concentration = float(row.get("hourly_concentration", 0.0))

    notes = []
    if "역" in name:
        notes.append("지하철역 중심 상권 여부 확인")
    if "관광특구" in name:
        notes.append("관광특구 경계와 실제 grid 중심 일치 여부 확인")
    if any(keyword in name for keyword in ["공원", "궁", "폭포"]):
        notes.append("관광/공원 POI와 좌표 중심이 맞는지 확인")
    if avg_flow > 100:
        notes.append("유동인구가 매우 커서 대표 hotspot 후보")
    if concentration > 1.5:
        notes.append("시간대 집중도가 높아 이벤트성/거점성 가능")
    if not notes:
        notes.append("주변 대표 장소와 비교해 수동 검토 권장")
    return "; ".join(notes)


def main() -> None:
    data_dir = "./data"
    priority_path = os.path.join(data_dir, "grid_place_mapping_priority.csv")
    base_path = os.path.join(data_dir, "base_infer.csv")
    output_csv = os.path.join(data_dir, "grid_place_mapping_priority_review.csv")
    output_md = os.path.join(data_dir, "grid_place_mapping_priority_review.md")

    priority = pd.read_csv(priority_path)
    base = pd.read_csv(base_path)

    merged = priority.merge(
        base[["grid_id", "lon", "lat", "avg_flow", "card_sales_amount", "hourly_concentration"]],
        on="grid_id",
        how="left",
    )
    merged["priority"] = merged.apply(classify_priority, axis=1)
    merged["review_note"] = merged.apply(review_note, axis=1)
    merged["recommended_action"] = "place_name 확인 후 place_code 입력"
    merged["verified_by_human"] = ""

    merged = merged[
        [
            "priority",
            "place_id",
            "place_name",
            "grid_id",
            "lon",
            "lat",
            "avg_flow",
            "card_sales_amount",
            "hourly_concentration",
            "review_note",
            "recommended_action",
            "verified_by_human",
        ]
    ].sort_values(["priority", "avg_flow"], ascending=[True, False])

    merged.to_csv(output_csv, index=False)

    lines = ["# Grid Place Mapping Priority Review", ""]
    for priority_level in ["P1", "P2", "P3"]:
        subset = merged[merged["priority"] == priority_level]
        if subset.empty:
            continue
        lines.append(f"## {priority_level}")
        lines.append("")
        for row in subset.itertuples(index=False):
            lines.append(
                f"- `{row.place_name}` | grid `{row.grid_id}` | avg_flow `{row.avg_flow:.2f}` | "
                f"점검: {row.review_note}"
            )
        lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved review csv -> {output_csv}")
    print(f"Saved review md -> {output_md}")
    print(f"rows: {len(merged)}")


if __name__ == "__main__":
    main()
