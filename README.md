# 사용 방법

## 1) 필요한 파일 구조
```bash
project/
 ┣ data/
 ┃ ┣ base_train.csv
 ┃ ┣ base_infer.csv
 ┃ ┣ correction_train.csv
 ┃ ┣ correction_infer.csv
 ┃ ┣ final_actual.csv            # 선택
 ┃ ┗ grid_place_mapping.csv      # 선택
 ┣ outputs/
 ┗ seoul_grid_vitality_pipeline.py
```

## 2) 설치
```bash
pip install pandas numpy scikit-learn scipy torch requests
```

## 3) 실행
```bash
python seoul_grid_vitality_pipeline.py
```

## 4) 컬럼 예시

### base_train.csv
- grid_id
- avg_flow
- weekday_weekend_gap
- hourly_concentration
- card_sales_amount
- card_sales_count
- rainfall_mean
- rainfall_impact
- bus_subway_access
- rent_level
- base_target

### correction_train.csv
- grid_id
- timestamp
- real_time_population
- real_time_population_growth
- traffic_congestion
- transit_change
- real_time_temp
- real_time_rain
- event_flag
- holiday_flag
- correction_target

### final_actual.csv
- grid_id
- final_actual

## 5) 성능 고도화 포인트
- Base MLP: BatchNorm + Dropout + AdamW
- Correction LSTM: sequence learning + Dropout
- Loss: HuberLoss
- Learning-rate scheduler: ReduceLROnPlateau
- Gradient clipping
- Early stopping
- Precision@K로 hotspot 일치도 평가
```
