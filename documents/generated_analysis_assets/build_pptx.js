const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const DOCS = path.join(ROOT, "documents");
const ASSET = path.join(DOCS, "generated_analysis_assets");
const OUT = path.join(DOCS, "작성본");
fs.mkdirSync(OUT, { recursive: true });

const summary = JSON.parse(fs.readFileSync(path.join(ASSET, "analysis_summary.json"), "utf8"));

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Codex";
pptx.company = "";
pptx.subject = "2026 서울시 빅데이터 활용 경진대회 분석부문 분석결과서";
pptx.title = "실시간 도시데이터 기반 서울 상권 활력 예측 및 의사결정 지원";
pptx.lang = "ko-KR";
pptx.theme = {
  headFontFace: "Malgun Gothic",
  bodyFontFace: "Malgun Gothic",
  lang: "ko-KR"
};
pptx.defineLayout({ name: "CUSTOM_LAYOUT", width: 13.333, height: 7.5 });
pptx.layout = "CUSTOM_LAYOUT";

const C = {
  ink: "17324D",
  muted: "5C6B73",
  teal: "2A9D8F",
  deep: "264653",
  green: "A7C957",
  yellow: "E9C46A",
  coral: "E76F51",
  bg: "F7FAFC",
  line: "D8E2EA",
  white: "FFFFFF"
};

const SLIDE_W = 13.333;
const SLIDE_H = 7.5;

function addBg(slide, accent = C.teal) {
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: SLIDE_W, h: 0.12, fill: { color: accent }, line: { color: accent } });
  slide.addText("2026 빅데이터 활용 경진대회 분석부문", {
    x: 0.55, y: 7.1, w: 5.2, h: 0.18,
    margin: 0, fontFace: "Malgun Gothic", fontSize: 7.5, color: "8796A1"
  });
}

function title(slide, text, subtitle = "") {
  slide.addText(text, {
    x: 0.62, y: 0.42, w: 10.5, h: 0.58,
    margin: 0, fontFace: "Malgun Gothic", fontSize: 24, bold: true, color: C.ink,
    breakLine: false, fit: "shrink"
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.64, y: 1.04, w: 10.1, h: 0.3,
      margin: 0, fontFace: "Malgun Gothic", fontSize: 10.5, color: C.muted,
      fit: "shrink"
    });
  }
}

function addFooter(slide, n) {
  slide.addText(String(n).padStart(2, "0"), {
    x: 12.42, y: 7.06, w: 0.35, h: 0.18,
    fontSize: 7.5, color: "8796A1", margin: 0, align: "right"
  });
}

function pill(slide, text, x, y, w, color = C.teal) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.34, rectRadius: 0.05, fill: { color, transparency: 8 }, line: { color, transparency: 100 } });
  slide.addText(text, { x: x + 0.12, y: y + 0.075, w: w - 0.24, h: 0.12, margin: 0, fontSize: 8.2, bold: true, color: C.white, fit: "shrink", align: "center" });
}

function metric(slide, label, value, x, y, w, color = C.deep) {
  slide.addText(value, { x, y, w, h: 0.5, margin: 0, fontSize: 26, bold: true, color, fit: "shrink" });
  slide.addText(label, { x, y: y + 0.52, w, h: 0.23, margin: 0, fontSize: 8.8, color: C.muted, fit: "shrink" });
}

function bullets(slide, items, x, y, w, opts = {}) {
  const h = opts.h || Math.max(0.48 * items.length, 1.0);
  const text = items.map(t => ({ text: t, options: { bullet: { indent: 12 }, hanging: 4, breakLine: true } }));
  slide.addText(text, {
    x, y, w, h,
    fontFace: "Malgun Gothic", fontSize: opts.size || 12.2,
    color: opts.color || C.ink,
    breakLine: false, fit: "shrink",
    margin: 0.02,
    paraSpaceAfterPt: 7
  });
}

function card(slide, x, y, w, h, header, body, color = C.teal) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.03, fill: { color: C.white }, line: { color: C.line, transparency: 10 } });
  slide.addShape(pptx.ShapeType.rect, { x, y, w: 0.08, h, fill: { color }, line: { color } });
  slide.addText(header, { x: x + 0.25, y: y + 0.18, w: w - 0.42, h: 0.24, margin: 0, fontSize: 11, bold: true, color: C.ink, fit: "shrink" });
  slide.addText(body, { x: x + 0.25, y: y + 0.52, w: w - 0.42, h: h - 0.68, margin: 0, fontSize: 9.1, color: C.muted, fit: "shrink", breakLine: false });
}

function image(slide, rel, x, y, w, h) {
  const p = path.join(ROOT, rel);
  if (fs.existsSync(p)) slide.addImage({ path: p, x, y, w, h });
}

function addSlide(n, header, subtitle, accent, fn) {
  const slide = pptx.addSlide();
  addBg(slide, accent);
  title(slide, header, subtitle);
  fn(slide);
  addFooter(slide, n);
}

// 1
{
  const s = pptx.addSlide();
  s.background = { color: "F2F7F5" };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 4.1, h: 7.5, fill: { color: C.deep }, line: { color: C.deep } });
  s.addShape(pptx.ShapeType.arc, { x: 7.6, y: -1.0, w: 5.9, h: 5.9, adjustPoint: 0.25, line: { color: C.teal, transparency: 100 }, fill: { color: C.teal, transparency: 82 } });
  s.addText("실시간 도시데이터 기반", { x: 0.74, y: 1.0, w: 2.8, h: 0.32, margin: 0, fontSize: 12, color: C.green, bold: true });
  s.addText("서울 상권 활력 예측 및\n의사결정 지원", { x: 0.72, y: 1.54, w: 6.6, h: 1.45, margin: 0, fontSize: 28, bold: true, color: C.white, fit: "shrink", breakLine: false });
  s.addText("50m 격자 단위 평시 상권 체력과 실시간 인구·교통·날씨·행사 신호를 결합한 최종 활력 점수 산출", { x: 4.65, y: 1.62, w: 7.25, h: 0.7, margin: 0, fontSize: 16, color: C.ink, fit: "shrink" });
  metric(s, "분석 격자", `${summary.covered_grid_count}개`, 4.66, 3.2, 1.8, C.teal);
  metric(s, "직접 실시간 격자", `${summary.live_grid_count}개`, 6.9, 3.2, 2.0, C.coral);
  metric(s, "최고 활력 점수", `${summary.city_max}`, 9.35, 3.2, 2.0, C.deep);
  image(s, "outputs/citywide_vitality/citywide_vitality_heatmap_covered_area_detail.png", 4.65, 4.38, 7.2, 2.25);
  addFooter(s, 1);
}

addSlide(2, "개요: 정적 상권 지표만으로는 현재 활력을 설명하기 어렵다", "목표는 평시 잠재력과 지금의 도시 신호를 같은 격자 위에서 결합하는 것", C.teal, s => {
  card(s, 0.74, 1.7, 3.65, 1.38, "문제", "유동·소비·교통·날씨·행사가 시간대별로 엇갈리며 상권 활력이 빠르게 변한다.", C.coral);
  card(s, 4.85, 1.7, 3.65, 1.38, "접근", "50m 격자 단위의 기본 상권 체력과 실시간 보정 신호를 분리해 학습한다.", C.teal);
  card(s, 8.96, 1.7, 3.65, 1.38, "결과", "격자별 final_score와 score_source를 함께 제공해 해석 가능한 의사결정을 지원한다.", C.green);
  bullets(s, [
    "상권 추천, 혼잡 완화, 행사·관광 운영 등 즉시성이 필요한 현장 의사결정에 초점",
    "직접 관측이 없는 격자도 장소·이웃 기반 전파로 추정하되 산출 방식을 명시",
    "현재 구현 범위는 커버 지역 분석이며, 서울 전역 확장은 동일 파이프라인으로 가능"
  ], 0.95, 3.82, 11.3, { h: 1.7, size: 14 });
});

addSlide(3, "분석 목적: ‘어디가 원래 강한가’와 ‘지금 뜨는가’를 분리해 본다", "최종 점수는 장기 잠재력과 현재 변화 신호의 합성 지표", C.deep, s => {
  s.addText("base_score", { x: 1.1, y: 2.15, w: 2.2, h: 0.38, fontSize: 20, bold: true, color: C.deep, margin: 0 });
  s.addText("격자의 평시 상권 체력\n유동·소비·접근성·임대 proxy", { x: 1.1, y: 2.68, w: 2.75, h: 0.75, fontSize: 12, color: C.muted, margin: 0, fit: "shrink" });
  s.addShape(pptx.ShapeType.plus, { x: 4.26, y: 2.55, w: 0.55, h: 0.55, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("correction_score", { x: 5.35, y: 2.15, w: 3.0, h: 0.38, fontSize: 20, bold: true, color: C.coral, margin: 0 });
  s.addText("현재 시점 보정 신호\n실시간 인구·교통·날씨·행사", { x: 5.35, y: 2.68, w: 3.05, h: 0.75, fontSize: 12, color: C.muted, margin: 0, fit: "shrink" });
  s.addShape(pptx.ShapeType.rightArrow, { x: 8.7, y: 2.47, w: 0.8, h: 0.55, fill: { color: C.green }, line: { color: C.green } });
  s.addText("final_score", { x: 10.05, y: 2.15, w: 2.1, h: 0.38, fontSize: 20, bold: true, color: C.teal, margin: 0 });
  s.addText("방문·운영·정책 판단에 쓰는\n최종 활력 점수", { x: 10.05, y: 2.68, w: 2.15, h: 0.75, fontSize: 12, color: C.muted, margin: 0, fit: "shrink" });
  s.addShape(pptx.ShapeType.line, { x: 1.08, y: 4.25, w: 10.9, h: 0, line: { color: C.line, width: 1 } });
  bullets(s, [
    "기본 점수가 높고 보정 점수도 높으면 안정적으로 활력 높은 격자",
    "기본 점수가 낮아도 보정 점수가 높으면 특정 시점에 반짝이는 후보지",
    "보정 점수가 낮으면 기존 강상권이라도 현재 상황에서는 체감 활력이 약할 수 있음"
  ], 1.18, 4.68, 10.6, { h: 1.25, size: 12.6 });
});

addSlide(4, "투입 데이터: 50m 격자 공통키로 상권·도시 신호를 연결", "빅데이터캠퍼스 핵심 데이터와 실시간 도시데이터를 분석 가능한 테이블로 표준화", C.teal, s => {
  const rows = [
    ["기본 상권", "유동인구, 카드 매출·건수, 시간 집중도, 평일/주말 차이", "base_score"],
    ["환경·접근성", "강우, 교통 접근성 proxy, 임대료/상권강도 proxy", "base_score 보강"],
    ["실시간 신호", "인구 규모·증감, 교통 혼잡, 대중교통 변화, 기온·강우, 행사·휴일", "correction_score"],
    ["공간 매핑", "place_code, grid_id, weight, 좌표, 행정구 경계", "장소 신호의 격자화"],
    ["검증·산출", "final_actual, validation predictions, metrics_summary", "성능 점검"]
  ];
  s.addTable([["데이터 구분", "주요 변수", "역할"], ...rows], {
    x: 0.75, y: 1.62, w: 11.8, h: 4.8,
    border: { type: "solid", color: "D8E2EA", pt: 0.8 },
    fill: { color: C.white },
    color: C.ink,
    fontFace: "Malgun Gothic",
    fontSize: 9.5,
    valign: "mid",
    margin: 0.08,
    autoFit: false,
    colW: [1.65, 7.15, 2.6],
    rowH: [0.45, 0.68, 0.68, 0.82, 0.68, 0.68],
    bold: false,
    fit: "shrink",
    options: { }
  });
  s.addShape(pptx.ShapeType.rect, { x: 0.75, y: 1.62, w: 11.8, h: 0.45, fill: { color: C.deep }, line: { color: C.deep } });
  ["데이터 구분", "주요 변수", "역할"].forEach((t, i) => {
    const xs = [0.86, 2.48, 9.67];
    const ws = [1.2, 6.7, 2.3];
    s.addText(t, { x: xs[i], y: 1.75, w: ws[i], h: 0.14, margin: 0, fontSize: 9, bold: true, color: C.white, align: "center" });
  });
});

addSlide(5, "전처리: 원천 데이터를 모델 입력 CSV로 변환", "공통 격자, 결측 처리, 표준화, 시계열 샘플링을 같은 파이프라인에서 수행", C.deep, s => {
  const steps = [
    ["1", "원천 데이터 적재", "유동·소비·날씨·교통·장소"],
    ["2", "50m 격자 정렬", "grid_id 중심 병합"],
    ["3", "특성 생성", "평시/실시간 feature 구성"],
    ["4", "모델 입력 생성", "base_train, correction_train"],
    ["5", "산출물 저장", "scores, metrics, heatmap"]
  ];
  steps.forEach((st, i) => {
    const x = 0.78 + i * 2.48;
    s.addShape(pptx.ShapeType.roundRect, { x, y: 2.05, w: 1.95, h: 1.45, rectRadius: 0.03, fill: { color: i % 2 ? "FFFFFF" : "E8F5F2" }, line: { color: C.line } });
    s.addText(st[0], { x: x + 0.15, y: 2.22, w: 0.36, h: 0.28, margin: 0, fontSize: 14, bold: true, color: C.teal, align: "center" });
    s.addText(st[1], { x: x + 0.2, y: 2.62, w: 1.55, h: 0.25, margin: 0, fontSize: 10.5, bold: true, color: C.ink, fit: "shrink", align: "center" });
    s.addText(st[2], { x: x + 0.2, y: 3.03, w: 1.55, h: 0.25, margin: 0, fontSize: 8.7, color: C.muted, fit: "shrink", align: "center" });
    if (i < steps.length - 1) s.addShape(pptx.ShapeType.chevron, { x: x + 2.03, y: 2.52, w: 0.33, h: 0.52, fill: { color: C.teal }, line: { color: C.teal } });
  });
  bullets(s, [
    "Base MLP 입력은 2차원 tabular feature, Correction LSTM 입력은 grid별 최근 4개 시점 sequence",
    "결측 변수는 0 또는 proxy로 보강하고, StandardScaler로 학습 안정성 확보",
    "Windows 임시 폴더 이슈를 줄이기 위해 프로젝트 내부 tmp 경로와 seed 고정 적용"
  ], 1.0, 4.48, 10.9, { size: 12.2, h: 1.2 });
});

addSlide(6, "모델 구조: Base MLP와 Correction LSTM의 이중 예측", "정적 잠재력과 시간 흐름을 서로 다른 모델이 담당", C.teal, s => {
  card(s, 0.85, 1.72, 3.55, 3.95, "Base MLP", "입력: avg_flow, card_sales, rainfall, access, rent proxy 등 9개 내외 정적 feature\n\n구조: Linear-BatchNorm-ReLU-Dropout 반복\n\n출력: 격자의 평시 상권 체력", C.deep);
  card(s, 4.9, 1.72, 3.55, 3.95, "Correction LSTM", "입력: 최근 4개 시점의 실시간 인구·교통·날씨·행사 feature\n\n구조: 2-layer LSTM + FC head\n\n출력: 현재 시점 보정값", C.coral);
  card(s, 8.95, 1.72, 3.55, 3.95, "Fusion", "final_score = base_score + correction_score\n\n해석: 장기 상권 잠재력과 지금의 변화를 동시에 반영\n\n산출: CSV, JSON, heatmap", C.teal);
});

addSlide(7, "Base MLP: 평시 상권 체력을 학습", "유동·소비·접근성의 구조적 강도를 격자별 기준 점수로 변환", C.deep, s => {
  metric(s, "Base 검증 RMSE", summary.metrics.base_metrics.rmse.toFixed(2), 0.95, 1.75, 2.2, C.deep);
  metric(s, "Base 검증 MAE", summary.metrics.base_metrics.mae.toFixed(2), 3.3, 1.75, 2.2, C.teal);
  metric(s, "학습 반복", "최대 100", 5.65, 1.75, 2.1, C.coral);
  metric(s, "Early stopping", "patience 12", 7.9, 1.75, 2.55, C.green);
  bullets(s, [
    "카드 소비와 유동인구는 상권의 기본 수요를, 시간 집중도와 평일/주말 차이는 이용 패턴을 설명",
    "HuberLoss를 사용해 큰 이상치가 많은 도시·소비 데이터에서 손실 안정성 확보",
    "BatchNorm, Dropout, AdamW, Gradient clipping으로 소규모 격자 데이터의 과적합 리스크를 완화"
  ], 0.98, 3.45, 6.2, { size: 12.3, h: 1.55 });
  image(s, "documents/generated_analysis_assets/score_distribution.png", 7.55, 3.18, 4.65, 2.45);
});

addSlide(8, "Correction LSTM: 실시간 변화 신호를 시간 흐름으로 반영", "최근 4개 시점의 변화를 보고 현재 활력 보정값을 예측", C.coral, s => {
  metric(s, "Correction RMSE", summary.metrics.correction_metrics.rmse.toFixed(2), 0.92, 1.78, 2.25, C.coral);
  metric(s, "Correction MAE", summary.metrics.correction_metrics.mae.toFixed(2), 3.32, 1.78, 2.05, C.teal);
  metric(s, "Spearman", summary.metrics.correction_metrics.spearman.toFixed(2), 5.58, 1.78, 1.85, C.deep);
  s.addShape(pptx.ShapeType.line, { x: 1.0, y: 3.55, w: 10.8, h: 0, line: { color: C.line, width: 1.2 } });
  ["t-3", "t-2", "t-1", "t"].forEach((t, i) => {
    const x = 1.0 + i * 2.1;
    pill(s, t, x, 3.28, 0.72, i === 3 ? C.coral : C.teal);
    s.addShape(pptx.ShapeType.line, { x: x + 0.36, y: 3.62, w: 0, h: 0.7, line: { color: C.line, width: 1 } });
  });
  card(s, 9.6, 3.0, 2.15, 1.38, "보정값", "현재 시점의 활력 상승·하락 방향", C.coral);
  bullets(s, [
    "실시간 인구 증가율, 교통 혼잡도, 대중교통 변화, 기온·강우, 행사·휴일 여부를 사용",
    "정적인 강상권 여부와 별개로 ‘지금 더 붐비는가’를 반영",
    "Spearman 0.50은 순위형 활용 가능성을 보여주며, 실제 운영에서는 누적 데이터로 재학습 필요"
  ], 0.98, 5.0, 10.4, { size: 11.8, h: 1.1 });
});

addSlide(9, "최종 점수: 해석 가능한 단순 결합으로 운영 의사결정에 연결", "base_score와 correction_score를 분리 보관해 사후 설명 가능성을 높임", C.teal, s => {
  const combos = [
    ["높은 base + 높은 correction", "원래 강하고 지금도 좋은 후보지", C.teal],
    ["높은 base + 낮은 correction", "잠재력은 높지만 현재 상황은 약함", C.yellow],
    ["낮은 base + 높은 correction", "특정 시점에 반짝이는 이벤트형 후보지", C.coral],
    ["낮은 base + 낮은 correction", "추천 우선순위 낮음", C.muted]
  ];
  combos.forEach((c, i) => {
    const x = 0.86 + (i % 2) * 5.9;
    const y = 1.72 + Math.floor(i / 2) * 1.75;
    card(s, x, y, 5.35, 1.18, c[0], c[1], c[2]);
  });
  s.addText("final_score = base_score + correction_score", { x: 2.2, y: 5.55, w: 8.8, h: 0.52, margin: 0, fontSize: 23, bold: true, color: C.deep, align: "center" });
  s.addText("단순한 결합식을 채택해 점수 변화의 원인을 base와 correction으로 나눠 확인할 수 있게 설계", { x: 2.35, y: 6.13, w: 8.5, h: 0.25, margin: 0, fontSize: 10.4, color: C.muted, align: "center", fit: "shrink" });
});

addSlide(10, "실시간 API-격자 매핑: 장소 단위 신호를 50m 격자로 변환", "서울 실시간 도시데이터의 장소 정보를 grid_id와 weight로 재집계", C.deep, s => {
  card(s, 0.85, 1.72, 2.7, 1.35, "장소 카탈로그", "서울 주요 장소의 place_code, 중심 좌표, 반경", C.deep);
  s.addShape(pptx.ShapeType.rightArrow, { x: 3.75, y: 2.08, w: 0.75, h: 0.45, fill: { color: C.teal }, line: { color: C.teal } });
  card(s, 4.7, 1.72, 2.7, 1.35, "공간 조인", "장소 polygon 내부 또는 주변 grid 탐색", C.teal);
  s.addShape(pptx.ShapeType.rightArrow, { x: 7.6, y: 2.08, w: 0.75, h: 0.45, fill: { color: C.teal }, line: { color: C.teal } });
  card(s, 8.55, 1.72, 2.7, 1.35, "가중 전파", "유동량 비중과 거리 기반 confidence 반영", C.green);
  bullets(s, [
    "직접 관측된 장소 신호는 grid 단위 correction_infer_live_spatial로 변환",
    "직접 관측 격자 주변에는 장소·이웃 기반 confidence를 계산해 HYBRID_PROPAGATED로 확장",
    "산출 방식은 DIRECT_LIVE, PLACE_PROPAGATED, HYBRID_PROPAGATED, BASE_ONLY로 구분"
  ], 1.02, 4.15, 10.8, { size: 12.4, h: 1.3 });
});

addSlide(11, "분석 범위: 현재는 커버 지역 중심, 전역 확장 가능 구조", "직접 실시간 관측 20개를 시작점으로 총 337개 격자에 점수 산출", C.teal, s => {
  image(s, "outputs/coverage/base_grid_coverage_map.png", 0.76, 1.6, 5.45, 4.65);
  metric(s, "총 산출 격자", `${summary.covered_grid_count}`, 7.0, 1.8, 1.7, C.teal);
  metric(s, "직접 관측", `${summary.live_grid_count}`, 9.15, 1.8, 1.7, C.coral);
  metric(s, "주요 장소", `${summary.place_count}`, 11.0, 1.8, 1.4, C.deep);
  bullets(s, [
    "대회 제출용 해석에서는 ‘현재 커버 지역 분석’으로 명시",
    "full_seoul_coverage=False이므로 서울 전역 절대 순위로 과대 해석하지 않음",
    "행정구 경계와 격자 좌표가 준비되어 있어 데이터 보강 시 전역 확장 가능"
  ], 7.0, 3.45, 5.1, { size: 11.4, h: 1.55 });
});

addSlide(12, "결과 개요: 평균 4.29, 최고 11.64의 격자별 활력 점수", "상위 격자는 관광·문화·상업 기능이 겹친 도심부에 집중", C.deep, s => {
  metric(s, "평균", `${summary.city_mean}`, 0.92, 1.8, 1.3, C.deep);
  metric(s, "표준편차", `${summary.city_std}`, 2.65, 1.8, 1.5, C.teal);
  metric(s, "최소", `${summary.city_min}`, 4.68, 1.8, 1.25, C.muted);
  metric(s, "최대", `${summary.city_max}`, 6.35, 1.8, 1.45, C.coral);
  image(s, "documents/generated_analysis_assets/score_distribution.png", 8.05, 1.65, 4.2, 2.35);
  bullets(s, [
    "최고 격자는 북촌한옥마을 영향권으로 나타났으며 final_score_citywide 11.641",
    "중앙값은 4.03으로 상위 일부 격자가 평균을 끌어올리는 분포",
    "실시간 보정값은 규모가 작지만 순위 미세 조정과 이벤트성 상승 포착에 기여"
  ], 0.98, 4.55, 10.6, { size: 12.3, h: 1.25 });
});

addSlide(13, "활력 히트맵: 도심 관광·상권 축에서 높은 점수가 관찰", "최종 점수를 공간적으로 시각화해 우선 관리·추천 후보지를 식별", C.teal, s => {
  image(s, "outputs/citywide_vitality/citywide_vitality_heatmap_covered_area_detail.png", 0.72, 1.48, 7.1, 5.28);
  card(s, 8.18, 1.65, 3.85, 1.35, "읽는 방법", "색이 진할수록 final_score_citywide가 높고, 격자 단위 우선순위가 높다.", C.teal);
  card(s, 8.18, 3.28, 3.85, 1.35, "해석 유의", "현재 산출은 커버 지역 중심 분석이며 서울 전역 완전 커버리지가 아니다.", C.coral);
  card(s, 8.18, 4.92, 3.85, 1.35, "활용", "현장 안내, 행사 운영, 프로모션 타이밍, 방문 분산 정책의 공간 후보지 도출", C.deep);
});

addSlide(14, "상위 격자: 북촌·창덕궁·서촌 영향권이 상위권 형성", "장소 기반 신호와 평시 상권 체력이 함께 높게 나타남", C.deep, s => {
  const rows = [["순위", "grid_id", "최종점수", "score_source", "driver_place"]];
  summary.top5.forEach((d, i) => rows.push([String(i + 1), d.grid_id, d.final_score_citywide.toFixed(3), d.score_source, d.driver_place]));
  s.addTable(rows, {
    x: 0.72, y: 1.68, w: 11.9, h: 3.35,
    colW: [0.65, 1.45, 1.2, 2.45, 5.45],
    rowH: [0.42, 0.48, 0.48, 0.48, 0.48, 0.48],
    border: { type: "solid", color: "D8E2EA", pt: 0.7 },
    fill: { color: C.white }, color: C.ink,
    margin: 0.06, fontSize: 9.2, fontFace: "Malgun Gothic", fit: "shrink", valign: "mid"
  });
  s.addShape(pptx.ShapeType.rect, { x: 0.72, y: 1.68, w: 11.9, h: 0.42, fill: { color: C.deep }, line: { color: C.deep } });
  s.addText("상위 격자는 단일 장소가 아니라 관광·보행·상업 활동이 겹친 미세 공간으로 해석해야 한다.", { x: 1.0, y: 5.65, w: 10.8, h: 0.35, margin: 0, fontSize: 13, bold: true, color: C.teal, align: "center", fit: "shrink" });
});

addSlide(15, "점수 산출 방식: 직접 관측과 전파 추정의 범위를 함께 표시", "의사결정자는 score_source를 보고 관측 신뢰도를 구분할 수 있음", C.teal, s => {
  image(s, "documents/generated_analysis_assets/score_source_counts.png", 0.9, 1.65, 5.75, 3.15);
  const items = [
    ["DIRECT_LIVE", summary.source_counts.DIRECT_LIVE, "서울 실시간 장소 신호가 직접 연결된 격자"],
    ["PLACE_PROPAGATED", summary.source_counts.PLACE_PROPAGATED, "장소 영향권에서 전파된 격자"],
    ["HYBRID_PROPAGATED", summary.source_counts.HYBRID_PROPAGATED, "장소·이웃 confidence를 함께 반영"],
    ["BASE_ONLY", summary.source_counts.BASE_ONLY, "실시간 보정 없이 기본 점수만 사용"]
  ];
  items.forEach((it, i) => {
    const y = 1.62 + i * 1.0;
    s.addText(it[0], { x: 7.2, y, w: 2.15, h: 0.26, margin: 0, fontSize: 10.6, bold: true, color: i === 0 ? C.coral : C.deep });
    s.addText(`${it[1]}개`, { x: 9.55, y, w: 0.85, h: 0.24, margin: 0, fontSize: 10.6, bold: true, color: C.teal, align: "right" });
    s.addText(it[2], { x: 7.2, y: y + 0.32, w: 4.55, h: 0.22, margin: 0, fontSize: 8.9, color: C.muted, fit: "shrink" });
  });
});

addSlide(16, "장소 프로파일: 관광·문화유산과 발달상권의 실시간 보정 비교", "장소별 base_score와 correction_score를 나눠 운영 포인트를 확인", C.deep, s => {
  image(s, "documents/generated_analysis_assets/place_scores.png", 0.72, 1.42, 11.9, 4.92);
  s.addText("왼쪽은 평시 상권 체력, 오른쪽은 실제 실시간 보정값이다. 두 지표는 단위와 범위가 달라 별도 x축으로 분리했다.", {
    x: 1.05, y: 6.48, w: 11.0, h: 0.28,
    margin: 0, fontSize: 10.4, color: C.muted, align: "center", fit: "shrink"
  });
});

addSlide(17, "검증 결과: 순위 활용 가능성과 데이터 보강 과제가 함께 확인", "현재 평가지표는 pseudo-label 및 소규모 실시간 관측의 한계를 전제로 해석", C.coral, s => {
  card(s, 0.9, 1.75, 2.55, 1.45, "Base MLP", `RMSE ${summary.metrics.base_metrics.rmse.toFixed(2)}\nMAE ${summary.metrics.base_metrics.mae.toFixed(2)}\nSpearman ${summary.metrics.base_metrics.spearman.toFixed(2)}`, C.deep);
  card(s, 3.85, 1.75, 2.55, 1.45, "Correction LSTM", `RMSE ${summary.metrics.correction_metrics.rmse.toFixed(2)}\nMAE ${summary.metrics.correction_metrics.mae.toFixed(2)}\nSpearman ${summary.metrics.correction_metrics.spearman.toFixed(2)}`, C.coral);
  card(s, 6.8, 1.75, 4.8, 1.45, "해석", "Correction 순위상관은 양의 방향을 보였지만, Base는 참조 label 보강과 feature 품질 개선이 필요", C.teal);
  bullets(s, [
    "최종 목적은 절대 수요 예측보다 격자별 우선순위와 변화 신호 탐지",
    "final_actual은 실제 관측 정답이라기보다 성능 점검용 참조값이므로 실제 운영 데이터 누적 후 재학습 필요",
    "향후 실제 방문·매출·혼잡 관측값이 들어오면 모델 검증력이 크게 개선될 수 있음"
  ], 0.95, 4.15, 10.8, { size: 12.2, h: 1.35 });
});

addSlide(18, "활용방안: 도시 운영과 소상공인 지원을 같은 지도 위에서 연결", "격자별 활력 점수는 현장 대응, 방문 분산, 정책 타기팅의 공통 지표가 될 수 있음", C.teal, s => {
  card(s, 0.82, 1.72, 3.55, 1.5, "관광·방문 안내", "혼잡 예상 지역 주변 대체 동선, 시간대별 추천 지점 안내", C.teal);
  card(s, 4.87, 1.72, 3.55, 1.5, "현장 운영", "행사·날씨·교통 변화에 따른 인력 배치와 안전관리 우선순위 설정", C.coral);
  card(s, 8.92, 1.72, 3.55, 1.5, "상권 정책", "상권 회복·활성화 대상지 탐색과 홍보·쿠폰·이벤트 타이밍 설계", C.deep);
  bullets(s, [
    "점수와 산출 방식을 함께 제공해 공공 담당자와 현장 운영자가 같은 기준으로 논의 가능",
    "CSV·지도·히트맵 산출물이 분리되어 대시보드, 보고서, API 서비스로 확장하기 쉬움",
    "실시간 보정값을 별도 관리하므로 단기 이벤트 효과와 장기 상권 잠재력을 구분 가능"
  ], 0.95, 4.35, 11.0, { size: 12.2, h: 1.3 });
});

addSlide(19, "한계와 고도화: 관측 확대, 실제 정답 축적, 서비스화가 다음 단계", "현재 구현은 작동 가능한 파이프라인이며 운영 데이터가 누적될수록 정밀도가 개선됨", C.deep, s => {
  const rows = [
    ["한계", "개선 방향"],
    ["직접 실시간 관측 격자 수가 20개로 제한", "서울 전역 장소 매핑 확대 및 API 호출 주기화"],
    ["final_actual이 실제 완전 정답이 아닌 참조값", "실제 방문·매출·혼잡 관측값과 사후 검증 데이터 축적"],
    ["전파 추정 구역의 불확실성 존재", "confidence 기반 표기와 임계값별 정책 시나리오 분리"],
    ["모델 성능이 데이터 품질에 민감", "특성 중요도, 시계열 길이, 공간 인접 특성 추가 실험"]
  ];
  s.addTable(rows, {
    x: 0.85, y: 1.65, w: 11.6, h: 3.25,
    colW: [5.2, 6.4], rowH: [0.5, 0.6, 0.6, 0.6, 0.6],
    border: { type: "solid", color: "D8E2EA", pt: 0.8 },
    fill: { color: C.white }, color: C.ink,
    margin: 0.08, fontSize: 10.3, fontFace: "Malgun Gothic", fit: "shrink", valign: "mid"
  });
  s.addShape(pptx.ShapeType.rect, { x: 0.85, y: 1.65, w: 11.6, h: 0.5, fill: { color: C.deep }, line: { color: C.deep } });
  s.addText("다음 단계는 분석 결과를 배치 산출물에서 실시간 대시보드·알림형 서비스로 전환하는 것이다.", { x: 1.08, y: 5.65, w: 10.8, h: 0.35, margin: 0, fontSize: 12.6, bold: true, color: C.teal, align: "center", fit: "shrink" });
});

addSlide(20, "분석 데이터·도구·참고", "제출 형식의 마지막 페이지에 활용 데이터, 분석툴, 참고 산출물을 정리", C.teal, s => {
  card(s, 0.85, 1.55, 3.65, 3.9, "데이터", "서울시 빅데이터캠퍼스 제공 데이터\n서울 실시간 도시데이터 OpenAPI\n50m 격자·장소 매핑 산출물\n행정구 경계 geojson", C.deep);
  card(s, 4.85, 1.55, 3.65, 3.9, "분석툴", "Python, pandas, NumPy\nscikit-learn, PyTorch, SciPy\nMatplotlib\nPptxGenJS / OOXML 산출", C.teal);
  card(s, 8.85, 1.55, 3.65, 3.9, "주요 코드·산출", "preprocess_seoul_grid_data.py\nseoul_grid_vitality_pipeline.py\nbuild_citywide_vitality_artifacts.py\nmetrics_summary.json, final_scores.csv, heatmap.png", C.coral);
  s.addText("PPT 본문은 공정 심사를 위해 분석명과 분석결과 내용만으로 구성했다.", { x: 0.95, y: 6.15, w: 11.2, h: 0.25, margin: 0, fontSize: 10.2, color: C.muted, align: "center" });
});

const outPath = path.join(OUT, "[팀명]_분석결과서.pptx");
pptx.writeFile({ fileName: outPath }).then(() => {
  console.log(outPath);
});
