# Seoul Grid Vitality Pipeline Quickstart

## 폴더 구조
```text
DL/
├─ data/
│  ├─ base_train.csv
│  ├─ base_infer.csv
│  ├─ correction_train.csv
│  ├─ correction_infer.csv
│  ├─ final_actual.csv
│  └─ grid_place_mapping.csv
├─ outputs/
├─ seoul_grid_vitality_pipeline.py
└─ README_quickstart.md
```

## 설치
```bash
pip install pandas numpy scikit-learn torch requests matplotlib
```

## 실행
```bash
python seoul_grid_vitality_pipeline.py
```

## 실제 과거 데이터에서 입력셋 생성
원천 CSV를 `source_data/`에 두었다면 아래 전처리 스크립트로
`data/base_train.csv`, `data/correction_train.csv` 등을 다시 만들 수 있습니다.

```bash
python preprocess_seoul_grid_data.py
python seoul_grid_vitality_pipeline.py
```

## 서울 실시간 도시데이터 OpenAPI 연결
공식 문서 기준 실시간 도시데이터 API는 `ServiceKey`, `AREA_NM`, `type`
요청변수를 사용하고, 한 번에 1개 장소씩 호출합니다.

1. `data/grid_place_mapping.csv`의 `place_code` 또는 `place_name`을 채웁니다.
2. 환경변수를 설정합니다.
3. 실시간 보정 입력 CSV를 생성합니다.

```bash
$env:SEOUL_RT_API_URL="YOUR_API_ENDPOINT"
$env:SEOUL_RT_API_KEY="YOUR_SERVICE_KEY"
D:\subUser\cgh\python.exe fetch_seoul_realtime_api_to_csv.py
```

생성 결과:
- `data/correction_infer_live.csv`

장소 매핑 초안이 필요하면 공식 121장소 목록 템플릿과 추천 매핑 스크립트를 사용할 수 있습니다.

```bash
D:\subUser\cgh\python.exe suggest_seoul_place_mapping.py
```

생성 결과:
- `data/seoul_live_place_catalog.csv`
- `data/grid_place_mapping_suggested.csv`

P1 우선 후보만 실제 API 호출 직전 형태로 정리하려면:

```bash
D:\subUser\cgh\python.exe prepare_api_ready_mapping.py
```

생성 결과:
- `data/grid_place_mapping_p1.csv`
- `data/grid_place_mapping_api_ready.csv`

공식 `서울시 주요 121장소 목록.xlsx`가 있으면 장소 코드까지 자동 보강할 수 있습니다.

```bash
D:\subUser\cgh\python.exe import_official_seoul_places.py
D:\subUser\cgh\python.exe enrich_mapping_with_official_codes.py
```

생성/갱신 결과:
- `data/seoul_live_place_catalog_official.csv`
- `data/grid_place_mapping.csv`
- `data/grid_place_mapping_priority.csv`
- `data/grid_place_mapping_p1.csv`
- `data/grid_place_mapping_api_ready.csv`

참고:
- 데이터셋 페이지: [서울시 실시간 도시데이터](https://data.seoul.go.kr/dataList/OA-21285/A/1/datasetView.do)
- 안내 페이지: [서울 실시간 도시데이터 가이드](https://data.seoul.go.kr/dataVisual/seoul/guide.do)

## 샘플 데이터 설명
- `base_train.csv`: grid별 정적 특성과 `base_target`
- `base_infer.csv`: 추론용 정적 특성 + 좌표
- `correction_train.csv`: 시계열 보정 학습용 데이터 + `correction_target`
- `correction_infer.csv`: 시계열 보정 추론용 데이터
- `final_actual.csv`: `timestamp`, `grid_id`, `final_actual`
- `grid_place_mapping.csv`: `place_id`와 `grid_id` 연결용 예시

## 생성 결과
- `outputs/base_mlp_best.pt`
- `outputs/correction_lstm_best.pt`
- `outputs/base_valid_predictions.csv`
- `outputs/correction_valid_predictions.csv`
- `outputs/final_scores.csv`
- `outputs/scenario_summary.csv`
- `outputs/metrics_summary.json`
- 좌표가 있으면 `outputs/final_score_heatmap.png`

## 샘플 데이터 특징
- 8개 grid
- correction 데이터는 grid당 8개 시점
- `seq_len=6` 기준으로 grid당 3개 최종 예측 시점을 만들 수 있게 구성
