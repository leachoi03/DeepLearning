$ErrorActionPreference = "Stop"

$root = (Resolve-Path ".").Path
$docDir = Join-Path $root "documents"
$outDir = Join-Path $docDir "작성본"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$summarySrc = Join-Path $docDir "[2026 서울시 빅데이터 통합 경진대회 분석부문]_[팀명]_분석결과서(요약)양식.doc"
$applicationSrc = Join-Path $docDir "[2026 서울시 빅데이터 통합 경진대회 분석부문]_[팀명]_참가신청서 및 참가각서 양식.doc"
$summaryOut = Join-Path $outDir "[팀명]_분석요약서_양식작성.doc"
$applicationOut = Join-Path $outDir "[팀명]_참가신청서_양식작성.doc"

Copy-Item -LiteralPath $summarySrc -Destination $summaryOut -Force
Copy-Item -LiteralPath $applicationSrc -Destination $applicationOut -Force

function Clean-CellText([string]$text) {
    return ($text -replace "`r`n", "`r" -replace "`n", "`r")
}

function Set-CellText($table, [int]$row, [int]$col, [string]$text, [double]$fontSize = 9.0) {
    $cell = $table.Cell($row, $col)
    $range = $cell.Range
    $range.Text = (Clean-CellText $text)
    $range.Font.Name = "맑은 고딕"
    $range.Font.NameFarEast = "맑은 고딕"
    $range.Font.Size = $fontSize
}

$summaryContent = @{
    Team = "[팀명]"
    Title = "실시간 도시데이터 기반 서울 상권 활력 예측 및 의사결정 지원"
    Background = @"
○ 추진배경
 - 서울 상권 활력은 유동인구, 소비, 날씨, 교통, 행사 여부에 따라 시간대별로 크게 변동함.
 - 정적 매출·인구 지표만으로는 현재 시점의 혼잡, 방문 수요, 상권 반응을 즉시 판단하기 어려움.
▸ 필요성
 - 50m 격자 단위 평시 상권 체력과 서울 실시간 도시데이터 기반 보정 신호를 결합해 현장 의사결정에 활용 가능한 최종 활력 점수 산출 필요.
"@
    Data = @"
○ 빅데이터캠퍼스 및 서울시 데이터 활용
 - 격자별 유동인구: avg_flow, weekday_weekend_gap, hourly_concentration
 - 카드 소비: card_sales_amount, card_sales_count
 - 환경·입지: rainfall_mean, rainfall_impact, bus_subway_access, rent_level proxy
 - 실시간 도시데이터: real_time_population, population_growth, traffic_congestion, transit_change, temp/rain, event/holiday flag
 - 공간자료: grid_id, place_code/name, weight, 서울 행정구 경계, 장소-grid 매핑
"@
    Method = @"
○ 전처리
 - 원천 데이터를 50m grid_id 기준으로 정렬하고 결측·proxy 특성을 보강한 뒤 base_train/base_infer, correction_train/correction_infer 생성.
○ 모델링
 - Base MLP: 평시 상권 체력을 base_score로 예측.
 - Correction LSTM: 최근 4개 시점 실시간 특성으로 현재 보정값 correction_score 예측.
○ 결합 및 확장
 - final_score = base_score + correction_score.
 - 직접 관측(DIRECT_LIVE), 장소 전파(PLACE_PROPAGATED), 혼합 전파(HYBRID_PROPAGATED), 기본값(BASE_ONLY)을 구분해 해석 가능성 확보.
"@
    Result = @"
○ 최종 산출
 - 분석 가능 격자 337개, 직접 실시간 관측 격자 20개, 주요 장소 8개 기준 최종 활력 점수 산출.
 - final_score_citywide 평균 4.288, 표준편차 1.210, 최댓값 11.641.
 - 상위 격자는 북촌한옥마을, 창덕궁·종묘, 서촌 등 관광·문화·상업 기능이 겹치는 도심부에서 주로 확인.
 - 산출물: final_scores.csv, citywide_final_scores.csv, metrics_summary.json, vitality heatmap.
"@
    Impact = @"
○ 활용방안
 - 시간대별 방문 추천, 혼잡 분산 안내, 행사·관광지 운영 인력 배치, 소상공인 프로모션 타이밍 결정에 활용.
 - score_source를 함께 제공해 직접 관측 구역과 전파 추정 구역을 구분하여 정책 담당자가 신뢰 수준별로 해석 가능.
○ 기대효과
 - 정적 상권 분석을 넘어 실시간 도시 변화에 반응하는 격자 단위 상권 모니터링 체계 구축.
 - 향후 실제 방문·매출·혼잡 관측값 누적 시 서울 전역 실시간 상권 활력 대시보드로 확장 가능.
"@
    Tools = @"
○ 분석툴
 - Python, pandas, NumPy, scikit-learn, PyTorch, SciPy, Matplotlib
 - Base MLP, Correction LSTM, HuberLoss, AdamW, ReduceLROnPlateau, Early stopping
○ 참고자료
 - 서울시 빅데이터캠퍼스 제공 데이터, 서울 실시간 도시데이터 OpenAPI
 - preprocess_seoul_grid_data.py, seoul_grid_vitality_pipeline.py, build_citywide_vitality_artifacts.py
"@
}

$applicationContent = @{
    Team = "[팀명]"
    Title = "실시간 도시데이터 기반 서울 상권 활력 예측 및 의사결정 지원"
    Background = @"
○ 서울 상권 활력은 유동인구, 소비, 날씨, 교통, 행사 요인에 따라 시간대별로 빠르게 변동함.
○ 정적 상권 지표와 실시간 도시데이터를 결합해 격자 단위로 현재 활력 수준을 판단할 수 있는 분석 체계가 필요함.
"@
    Analysis = @"
○ 투입데이터
 - 50m 격자별 유동인구, 카드 소비, 강우, 교통 접근성, 임대료 proxy
 - 서울 실시간 도시데이터 API 기반 인구·교통·날씨·행사 신호
 - 장소-grid 매핑, 행정구 경계, 모델 검증용 final_actual 및 metrics_summary
○ 분석내용
 - Base MLP로 평시 상권 체력(base_score) 예측
 - Correction LSTM으로 실시간 보정값(correction_score) 예측
 - final_score = base_score + correction_score로 최종 활력 점수 산출
 - 분석 범위: 337개 격자, 직접 실시간 관측 20개 격자, 주요 장소 8개
"@
    Impact = @"
○ 상권 활력 상위·변동 지역을 파악해 관광 동선 안내, 혼잡 완화, 현장 운영 인력 배치에 활용.
○ 실시간 보정 체계를 통해 날씨·행사·교통 변화에 따른 상권 반응을 빠르게 재계산하고, 소상공인 지원 및 공공정책 타기팅에 활용.
"@
}

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $doc = $word.Documents.Open($summaryOut, $false, $false)
    $table = $doc.Tables.Item(1)
    Set-CellText $table 1 2 $summaryContent.Team 10.0
    Set-CellText $table 2 2 $summaryContent.Title 10.0
    Set-CellText $table 3 2 $summaryContent.Background 8.0
    Set-CellText $table 4 2 $summaryContent.Data 8.0
    Set-CellText $table 5 2 $summaryContent.Method 8.0
    Set-CellText $table 6 2 $summaryContent.Result 8.0
    Set-CellText $table 7 2 $summaryContent.Impact 8.0
    Set-CellText $table 8 2 $summaryContent.Tools 8.0
    $doc.Save()
    $doc.Close($false)

    $doc = $word.Documents.Open($applicationOut, $false, $false)
    $table = $doc.Tables.Item(1)
    Set-CellText $table 1 1 $applicationContent.Team 10.0
    Set-CellText $table 11 3 $applicationContent.Title 9.0
    Set-CellText $table 12 3 $applicationContent.Background 8.0
    Set-CellText $table 13 3 $applicationContent.Analysis 8.0
    Set-CellText $table 14 3 $applicationContent.Impact 8.0
    $doc.Save()
    $doc.Close($false)
}
finally {
    $word.Quit()
}

Write-Output $summaryOut
Write-Output $applicationOut

