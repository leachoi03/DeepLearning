# DL Quickstart

## Current Scope
- This project is currently a `covered-area analysis`, not a full Seoul-wide analysis.
- The current KT grid sample in [base_infer.csv](D:/subUser/cgh/CodexProj/project/DL/data/base_infer.csv) covers only part of Seoul, mainly the north-central area.
- Because of that, the current heatmap should be interpreted as `regional vitality analysis for the covered grid area`.

## Expansion Path
- The code is already structured so it can expand to full Seoul coverage later.
- To expand, replace only these inputs:
  - [base_train.csv](D:/subUser/cgh/CodexProj/project/DL/data/base_train.csv)
  - [base_infer.csv](D:/subUser/cgh/CodexProj/project/DL/data/base_infer.csv)
  - [correction_train.csv](D:/subUser/cgh/CodexProj/project/DL/data/correction_train.csv)
  - [grid_place_mapping_spatial.csv](D:/subUser/cgh/CodexProj/project/DL/data/grid_place_mapping_spatial.csv)
- The main pipelines already read paths from environment variables, so the same scripts can be reused.

## Main Scripts
- [preprocess_seoul_grid_data.py](D:/subUser/cgh/CodexProj/project/DL/preprocess_seoul_grid_data.py)
  - Builds model-ready datasets from historical source CSVs.
- [seoul_grid_vitality_pipeline.py](D:/subUser/cgh/CodexProj/project/DL/seoul_grid_vitality_pipeline.py)
  - Trains the base MLP and correction LSTM, then writes score outputs.
- [build_spatial_grid_place_mapping.py](D:/subUser/cgh/CodexProj/project/DL/build_spatial_grid_place_mapping.py)
  - Builds `place -> grid` mapping by spatial join.
- [fetch_seoul_realtime_api_to_csv.py](D:/subUser/cgh/CodexProj/project/DL/fetch_seoul_realtime_api_to_csv.py)
  - Pulls Seoul real-time city data into correction input format.
- [build_live_correction_sequence.py](D:/subUser/cgh/CodexProj/project/DL/build_live_correction_sequence.py)
  - Builds LSTM-ready live correction sequences.
- [build_citywide_vitality_artifacts.py](D:/subUser/cgh/CodexProj/project/DL/build_citywide_vitality_artifacts.py)
  - Produces the final covered-area vitality heatmap and summaries.
- [build_base_coverage_map.py](D:/subUser/cgh/CodexProj/project/DL/build_base_coverage_map.py)
  - Shows which part of Seoul is actually covered by the current base grid.

## Run Order
```bash
python preprocess_seoul_grid_data.py
python build_spatial_grid_place_mapping.py
python fetch_seoul_realtime_api_to_csv.py
python build_live_correction_sequence.py
python seoul_grid_vitality_pipeline.py
python build_citywide_vitality_artifacts.py
python build_base_coverage_map.py
```

## Important Outputs
- Covered-area vitality results:
  - [citywide_final_scores.csv](D:/subUser/cgh/CodexProj/project/DL/outputs/citywide_vitality/citywide_final_scores.csv)
  - [citywide_vitality_heatmap.png](D:/subUser/cgh/CodexProj/project/DL/outputs/citywide_vitality/citywide_vitality_heatmap.png)
  - [analysis_scope_manifest.csv](D:/subUser/cgh/CodexProj/project/DL/outputs/citywide_vitality/analysis_scope_manifest.csv)
- Coverage diagnostics:
  - [base_grid_coverage_map.png](D:/subUser/cgh/CodexProj/project/DL/outputs/coverage/base_grid_coverage_map.png)
  - [base_grid_coverage_by_gu.csv](D:/subUser/cgh/CodexProj/project/DL/outputs/coverage/base_grid_coverage_by_gu.csv)

## Notes
- If full Seoul grid data is added later, do not change the overall code structure first.
- Start by swapping in wider `base` and `correction` inputs.
- After that, rerun the same pipeline and only then tune propagation or visualization if needed.
