"""
Build an LSTM-ready live correction input by attaching the latest live snapshot
to the most recent historical rows per grid.

Inputs:
- ./data/correction_infer.csv
- ./data/correction_infer_live.csv

Output:
- ./data/correction_infer_live_sequence.csv
"""

from __future__ import annotations

import os

import pandas as pd


SEQ_LEN = int(os.environ.get("SEOUL_GRID_SEQ_LEN", "4"))
HISTORY_PATH = os.environ.get("SEOUL_GRID_HISTORY_CORRECTION_CSV", "./data/correction_infer.csv")
TRAIN_HISTORY_PATH = os.environ.get("SEOUL_GRID_TRAIN_CORRECTION_CSV", "./data/correction_train.csv")
LIVE_PATH = os.environ.get("SEOUL_GRID_LIVE_CORRECTION_CSV", "./data/correction_infer_live.csv")
OUTPUT_PATH = os.environ.get("SEOUL_GRID_LIVE_SEQUENCE_CSV", "./data/correction_infer_live_sequence.csv")
GRID_ID_COL = "grid_id"
TIME_COL = "timestamp"


def main() -> None:
    history_frames = []
    for path in [TRAIN_HISTORY_PATH, HISTORY_PATH]:
        if os.path.exists(path):
            history_frames.append(pd.read_csv(path))
    if not history_frames:
        raise ValueError("No historical correction CSV was found.")
    history_df = pd.concat(history_frames, ignore_index=True)
    live_df = pd.read_csv(LIVE_PATH)

    history_df[TIME_COL] = pd.to_datetime(history_df[TIME_COL], errors="coerce")
    live_df[TIME_COL] = pd.to_datetime(live_df[TIME_COL], errors="coerce")

    feature_cols = [col for col in live_df.columns if col not in {GRID_ID_COL, TIME_COL}]
    if not feature_cols:
        raise ValueError("Live correction CSV does not contain feature columns.")

    if "correction_target" in history_df.columns:
        history_df = history_df.drop(columns=["correction_target"])

    history_df = history_df[[GRID_ID_COL, TIME_COL, *feature_cols]].copy()
    history_df = history_df.dropna(subset=[GRID_ID_COL, TIME_COL]).sort_values([GRID_ID_COL, TIME_COL])
    live_df = live_df[[GRID_ID_COL, TIME_COL, *feature_cols]].copy()
    live_df = live_df.dropna(subset=[GRID_ID_COL, TIME_COL]).sort_values([GRID_ID_COL, TIME_COL])
    live_df = live_df.groupby(GRID_ID_COL, as_index=False).tail(1).sort_values([GRID_ID_COL, TIME_COL]).reset_index(drop=True)

    combined_frames = []
    required_history = max(0, SEQ_LEN - 1)

    for row in live_df.itertuples(index=False):
        grid_id = row.grid_id
        history_tail = history_df[history_df[GRID_ID_COL] == grid_id].drop_duplicates(subset=[TIME_COL]).tail(required_history)
        if len(history_tail) < required_history:
            continue
        live_row = pd.DataFrame([row._asdict()])
        combined_frames.append(pd.concat([history_tail, live_row], ignore_index=True))

    if not combined_frames:
        raise ValueError("Could not build any live LSTM sequences. Check grid overlap and seq_len.")

    out_df = pd.concat(combined_frames, ignore_index=True)
    out_df = out_df.sort_values([GRID_ID_COL, TIME_COL]).reset_index(drop=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved live sequence correction rows -> {OUTPUT_PATH}")
    print(f"rows: {len(out_df)}")
    print(f"grids: {out_df[GRID_ID_COL].nunique()}")


if __name__ == "__main__":
    main()
