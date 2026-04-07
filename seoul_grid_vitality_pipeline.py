"""
Seoul grid vitality pipeline based on the team proposal.

This script implements:
1. Base suitability model with an MLP on historical grid-level features.
2. Real-time correction model with an LSTM on time-series features.
3. Optional Seoul OpenAPI ingestion helper.
4. Final score generation in the proposal format:
   timestamp, grid_id, base_score, correction_score, final_score
5. Optional heatmap-style scatter visualization when coordinate columns exist.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Some Windows environments set SSLKEYLOGFILE to a protected location,
# which can break requests import before the script even starts running.
os.environ.pop("SSLKEYLOGFILE", None)
_TMP_DIR = os.path.join(os.getcwd(), "tmp")
os.makedirs(_TMP_DIR, exist_ok=True)
os.environ["TMP"] = _TMP_DIR
os.environ["TEMP"] = _TMP_DIR
os.environ["TMPDIR"] = _TMP_DIR
os.environ["MPLCONFIGDIR"] = os.path.join(_TMP_DIR, "mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def precision_at_k(pred_scores: np.ndarray, true_scores: np.ndarray, ids: np.ndarray, k: int = 100) -> float:
    pred_top_ids = set(ids[np.argsort(-pred_scores)[:k]])
    true_top_ids = set(ids[np.argsort(-true_scores)[:k]])
    return len(pred_top_ids & true_top_ids) / max(1, k)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_rank = pd.Series(y_true).rank(method="average")
    y_pred_rank = pd.Series(y_pred).rank(method="average")
    corr = y_true_rank.corr(y_pred_rank, method="pearson")
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(corr if not np.isnan(corr) else 0.0),
    }


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {path} | shape={df.shape}")
    return df


def find_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for col in candidates:
        matched = lowered.get(str(col).strip().lower())
        if matched is not None:
            return matched
    return None


def require_columns(df: pd.DataFrame, columns: Sequence[str], df_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def fill_missing(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].ffill().bfill()
    return df


def add_missing_feature_columns(df: pd.DataFrame, feature_cols: Sequence[str], fill_value: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


def cast_columns_to_float(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    return df


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir: str = os.environ.get("SEOUL_GRID_DATA_DIR", "./data")
    output_dir: str = os.environ.get("SEOUL_GRID_OUTPUT_DIR", "./outputs")

    base_train_csv: str = os.environ.get("SEOUL_GRID_BASE_TRAIN_CSV", "./data/base_train.csv")
    base_infer_csv: str = os.environ.get("SEOUL_GRID_BASE_INFER_CSV", "./data/base_infer.csv")
    correction_train_csv: str = os.environ.get("SEOUL_GRID_CORRECTION_TRAIN_CSV", "./data/correction_train.csv")
    correction_infer_csv: str = os.environ.get("SEOUL_GRID_CORRECTION_INFER_CSV", "./data/correction_infer.csv")
    final_actual_csv: str = os.environ.get("SEOUL_GRID_FINAL_ACTUAL_CSV", "./data/final_actual.csv")
    mapping_csv: str = os.environ.get("SEOUL_GRID_MAPPING_CSV", "./data/grid_place_mapping.csv")

    seoul_api_url: str = os.environ.get("SEOUL_RT_API_URL", "")
    seoul_api_key: str = os.environ.get("SEOUL_RT_API_KEY", "")
    seoul_api_save_csv: str = os.environ.get("SEOUL_RT_SNAPSHOT_CSV", "./outputs/realtime_api_snapshot.csv")

    grid_id_col: str = "grid_id"
    place_id_col: str = "place_id"
    time_col: str = "timestamp"
    base_target_col: str = "base_target"
    correction_target_col: str = "correction_target"
    final_actual_col: str = "final_actual"

    x_col_candidates: Tuple[str, ...] = ("x", "coord_x", "utm_x")
    y_col_candidates: Tuple[str, ...] = ("y", "coord_y", "utm_y")
    lon_col_candidates: Tuple[str, ...] = ("lon", "lng", "longitude")
    lat_col_candidates: Tuple[str, ...] = ("lat", "latitude")

    base_features: Tuple[str, ...] = (
        "avg_flow",
        "weekday_weekend_gap",
        "hourly_concentration",
        "card_sales_amount",
        "card_sales_count",
        "rainfall_mean",
        "rainfall_impact",
        "bus_subway_access",
        "rent_level",
    )
    correction_features: Tuple[str, ...] = (
        "real_time_population",
        "real_time_population_growth",
        "traffic_congestion",
        "transit_change",
        "real_time_temp",
        "real_time_rain",
        "event_flag",
        "holiday_flag",
    )

    seq_len: int = 4
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    num_workers: int = 0

    base_hidden_dims: Tuple[int, ...] = (128, 64, 32)
    correction_hidden_dim: int = 64
    correction_num_layers: int = 2
    dropout: float = 0.2

    alpha: float = 1.0
    beta: float = 1.0
    topk: int = 100


class TabularRegressionDataset(Dataset):
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class SequenceRegressionDataset(Dataset):
    def __init__(self, x_seq: np.ndarray, y: Optional[np.ndarray] = None):
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.x_seq)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x_seq[idx]
        return self.x_seq[idx], self.y[idx]


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CorrectionLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_seq)
        return self.head(out[:, -1, :])


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return True

        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class TorchRegressor:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model.to(config.device)
        self.config = config
        self.criterion = nn.HuberLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, model_path: str) -> Dict[str, List[float]]:
        history = {"train_loss": [], "valid_loss": []}
        early_stopper = EarlyStopping(patience=self.config.patience, mode="min")
        best_state = None

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_losses: List[float] = []
            for xb, yb in train_loader:
                xb = xb.to(self.config.device)
                yb = yb.to(self.config.device)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(float(loss.item()))

            train_loss = float(np.mean(train_losses))

            self.model.eval()
            valid_losses: List[float] = []
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(self.config.device)
                    yb = yb.to(self.config.device)
                    pred = self.model(xb)
                    loss = self.criterion(pred, yb)
                    valid_losses.append(float(loss.item()))

            valid_loss = float(np.mean(valid_losses))
            self.scheduler.step(valid_loss)
            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)

            if early_stopper.step(valid_loss):
                best_state = {key: value.cpu().clone() for key, value in self.model.state_dict().items()}

            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.5f} "
                f"valid_loss={valid_loss:.5f} lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if early_stopper.should_stop:
                print("Early stopping triggered.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(best_state, model_path)
            print(f"Best model saved -> {model_path}")

        return history

    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                xb = xb.to(self.config.device)
                preds.append(self.model(xb).squeeze(-1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def build_base_data(
    config: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    df = load_csv(config.base_train_csv)
    df = add_missing_feature_columns(df, config.base_features)
    require_columns(df, [config.grid_id_col, config.base_target_col], "base_train.csv")
    df = cast_columns_to_float(df, list(config.base_features) + [config.base_target_col])
    df = fill_missing(df, list(config.base_features) + [config.base_target_col])

    x = df[list(config.base_features)].to_numpy(dtype=np.float32)
    y = df[config.base_target_col].to_numpy(dtype=np.float32)
    grid_ids = df[config.grid_id_col].to_numpy()

    x_train, x_valid, y_train, y_valid, _, gid_valid = train_test_split(
        x, y, grid_ids, test_size=0.2, random_state=config.seed
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    valid_df = pd.DataFrame({config.grid_id_col: gid_valid, "y_true": y_valid})
    return x_train, x_valid, y_train, y_valid, scaler, valid_df


def build_sequence_samples(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    grid_id_col: str,
    time_col: str,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = df.sort_values([grid_id_col, time_col]).copy()

    x_seqs: List[np.ndarray] = []
    y_list: List[float] = []
    grid_list: List[object] = []
    time_list: List[object] = []

    for grid_id, gdf in df.groupby(grid_id_col):
        gdf = gdf.sort_values(time_col).reset_index(drop=True)
        if len(gdf) < seq_len:
            continue

        feat = gdf[list(feature_cols)].to_numpy(dtype=np.float32)
        targ = gdf[target_col].to_numpy(dtype=np.float32)
        timestamps = gdf[time_col].to_numpy()

        for idx in range(seq_len - 1, len(gdf)):
            x_seqs.append(feat[idx - seq_len + 1 : idx + 1])
            y_list.append(float(targ[idx]))
            grid_list.append(grid_id)
            time_list.append(timestamps[idx])

    return (
        np.asarray(x_seqs, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(grid_list),
        np.asarray(time_list),
    )


def build_correction_data(
    config: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    df = load_csv(config.correction_train_csv)
    df = add_missing_feature_columns(df, config.correction_features)
    require_columns(df, [config.grid_id_col, config.time_col, config.correction_target_col], "correction_train.csv")
    df = cast_columns_to_float(df, list(config.correction_features) + [config.correction_target_col])
    df = fill_missing(df, list(config.correction_features) + [config.correction_target_col])
    df[config.time_col] = pd.to_datetime(df[config.time_col])

    scaler = StandardScaler()
    df.loc[:, list(config.correction_features)] = scaler.fit_transform(df[list(config.correction_features)])

    x_seq, y, gids, timestamps = build_sequence_samples(
        df=df,
        feature_cols=config.correction_features,
        target_col=config.correction_target_col,
        grid_id_col=config.grid_id_col,
        time_col=config.time_col,
        seq_len=config.seq_len,
    )
    if len(x_seq) == 0:
        raise ValueError("correction_train.csv does not contain enough rows to build LSTM sequences.")

    idx = np.arange(len(x_seq))
    train_idx, valid_idx = train_test_split(idx, test_size=0.2, random_state=config.seed)

    valid_df = pd.DataFrame(
        {
            config.grid_id_col: gids[valid_idx],
            config.time_col: pd.to_datetime(timestamps[valid_idx]),
            "y_true": y[valid_idx],
        }
    )
    return x_seq[train_idx], x_seq[valid_idx], y[train_idx], y[valid_idx], scaler, valid_df


def fetch_seoul_realtime_data(api_url: str, api_key: str = "", params: Optional[dict] = None) -> dict:
    if not api_url:
        raise ValueError("Config.seoul_api_url is empty.")

    req_params = dict(params or {})
    if api_key:
        req_params.setdefault("serviceKey", api_key)

    response = requests.get(api_url, params=req_params, timeout=20)
    response.raise_for_status()
    return response.json()


def _flatten_records(payload: object) -> List[dict]:
    if isinstance(payload, dict):
        if all(not isinstance(value, (dict, list)) for value in payload.values()):
            return [payload]
        rows: List[dict] = []
        for value in payload.values():
            rows.extend(_flatten_records(value))
        return rows
    if isinstance(payload, list):
        rows: List[dict] = []
        for item in payload:
            rows.extend(_flatten_records(item))
        return rows
    return []


def normalize_realtime_api_payload(payload: dict, config: Config) -> pd.DataFrame:
    rows = _flatten_records(payload)
    if not rows:
        return pd.DataFrame(columns=[config.place_id_col, config.time_col, *config.correction_features])

    df = pd.DataFrame(rows)
    rename_candidates = {
        config.place_id_col: ("place_id", "AREA_CD", "area_cd", "id"),
        config.time_col: ("timestamp", "ts", "datetime", "PPLTN_TIME", "time"),
        "real_time_population": ("real_time_population", "population", "PPLTN_MIN", "people"),
        "real_time_population_growth": ("real_time_population_growth", "population_growth", "growth"),
        "traffic_congestion": ("traffic_congestion", "congestion", "road_congestion"),
        "transit_change": ("transit_change", "transit_delta", "transport_change"),
        "real_time_temp": ("real_time_temp", "temp", "temperature"),
        "real_time_rain": ("real_time_rain", "rain", "rainfall"),
        "event_flag": ("event_flag", "event", "festival_flag"),
        "holiday_flag": ("holiday_flag", "holiday", "holiday_flag"),
    }

    rename_map = {}
    for target, candidates in rename_candidates.items():
        matched = find_existing_column(df, candidates)
        if matched is not None:
            rename_map[matched] = target
    df = df.rename(columns=rename_map)

    if config.time_col not in df.columns:
        df[config.time_col] = pd.Timestamp.now()
    df[config.time_col] = pd.to_datetime(df[config.time_col], errors="coerce").fillna(pd.Timestamp.now())

    for feature in config.correction_features:
        if feature not in df.columns:
            df[feature] = 0.0

    if config.place_id_col not in df.columns:
        raise ValueError("Could not infer place_id column from Seoul API payload.")

    cols = [config.place_id_col, config.time_col, *config.correction_features]
    return fill_missing(df[cols], config.correction_features)


def map_place_to_grid(
    realtime_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    place_id_col: str = "place_id",
    grid_id_col: str = "grid_id",
) -> pd.DataFrame:
    require_columns(realtime_df, [place_id_col], "realtime_df")
    require_columns(mapping_df, [place_id_col, grid_id_col], "mapping_df")
    return mapping_df.merge(realtime_df, on=place_id_col, how="left")


def aggregate_realtime_to_grid(realtime_grid_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    numeric_cols = [col for col in config.correction_features if col in realtime_grid_df.columns]
    agg_df = (
        realtime_grid_df.groupby([config.grid_id_col, config.time_col], as_index=False)[numeric_cols]
        .mean()
    )
    return fill_missing(agg_df, numeric_cols)


def train_base_model(config: Config) -> Tuple[BaseMLP, StandardScaler, Dict[str, float], pd.DataFrame]:
    print("\n=== Train Base MLP ===")
    x_train, x_valid, y_train, y_valid, scaler, valid_df = build_base_data(config)

    train_loader = DataLoader(
        TabularRegressionDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    valid_loader = DataLoader(
        TabularRegressionDataset(x_valid, y_valid),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = BaseMLP(input_dim=x_train.shape[1], hidden_dims=config.base_hidden_dims, dropout=config.dropout)
    trainer = TorchRegressor(model, config)
    trainer.fit(train_loader, valid_loader, model_path=os.path.join(config.output_dir, "base_mlp_best.pt"))

    pred_valid = trainer.predict(valid_loader)
    metrics = regression_metrics(y_valid, pred_valid)
    valid_df["pred_base"] = pred_valid
    print("Base Model Metrics:", metrics)
    return model, scaler, metrics, valid_df


def train_correction_model(
    config: Config,
) -> Tuple[CorrectionLSTM, StandardScaler, Dict[str, float], pd.DataFrame]:
    print("\n=== Train Correction LSTM ===")
    x_train, x_valid, y_train, y_valid, scaler, valid_df = build_correction_data(config)

    train_loader = DataLoader(
        SequenceRegressionDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    valid_loader = DataLoader(
        SequenceRegressionDataset(x_valid, y_valid),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = CorrectionLSTM(
        input_dim=x_train.shape[2],
        hidden_dim=config.correction_hidden_dim,
        num_layers=config.correction_num_layers,
        dropout=config.dropout,
    )
    trainer = TorchRegressor(model, config)
    trainer.fit(train_loader, valid_loader, model_path=os.path.join(config.output_dir, "correction_lstm_best.pt"))

    pred_valid = trainer.predict(valid_loader)
    metrics = regression_metrics(y_valid, pred_valid)
    valid_df["pred_correction"] = pred_valid
    print("Correction Model Metrics:", metrics)
    return model, scaler, metrics, valid_df


def predict_base_scores(
    config: Config,
    model: BaseMLP,
    scaler: StandardScaler,
    input_csv: Optional[str] = None,
) -> pd.DataFrame:
    df = load_csv(input_csv or config.base_infer_csv)
    df = add_missing_feature_columns(df, config.base_features)
    require_columns(df, [config.grid_id_col], "base_infer.csv")
    df = cast_columns_to_float(df, list(config.base_features))
    df = fill_missing(df, config.base_features)

    x = scaler.transform(df[list(config.base_features)].to_numpy(dtype=np.float32))
    loader = DataLoader(TabularRegressionDataset(x), batch_size=config.batch_size, shuffle=False)
    preds = TorchRegressor(model, config).predict(loader)

    keep_cols = [config.grid_id_col]
    optional_cols = list(config.lon_col_candidates) + list(config.lat_col_candidates) + list(config.x_col_candidates) + list(config.y_col_candidates)
    for optional_col in optional_cols:
        if optional_col in df.columns and optional_col not in keep_cols:
            keep_cols.append(optional_col)

    out = df[keep_cols].copy()
    out["base_score"] = preds
    return out


def predict_correction_scores(
    config: Config,
    model: CorrectionLSTM,
    scaler: StandardScaler,
    input_csv: Optional[str] = None,
) -> pd.DataFrame:
    df = load_csv(input_csv or config.correction_infer_csv)
    df = add_missing_feature_columns(df, config.correction_features)
    require_columns(df, [config.grid_id_col, config.time_col], "correction_infer.csv")
    df = cast_columns_to_float(df, list(config.correction_features))
    df = fill_missing(df, config.correction_features)
    df[config.time_col] = pd.to_datetime(df[config.time_col])
    df.loc[:, list(config.correction_features)] = scaler.transform(df[list(config.correction_features)])

    dummy_target = "__dummy_target__"
    df[dummy_target] = 0.0
    x_seq, _, gids, timestamps = build_sequence_samples(
        df=df,
        feature_cols=config.correction_features,
        target_col=dummy_target,
        grid_id_col=config.grid_id_col,
        time_col=config.time_col,
        seq_len=config.seq_len,
    )
    if len(x_seq) == 0:
        raise ValueError("correction_infer.csv does not contain enough rows to build LSTM sequences.")

    loader = DataLoader(SequenceRegressionDataset(x_seq), batch_size=config.batch_size, shuffle=False)
    preds = TorchRegressor(model, config).predict(loader)
    return pd.DataFrame(
        {
            config.grid_id_col: gids,
            config.time_col: pd.to_datetime(timestamps),
            "correction_score": preds,
        }
    )


def combine_scores(base_score_df: pd.DataFrame, correction_score_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    merged = correction_score_df.merge(base_score_df, on=config.grid_id_col, how="left")
    merged["base_score"] = merged["base_score"].fillna(0.0)
    merged["correction_score"] = merged["correction_score"].fillna(0.0)
    merged["final_score"] = config.alpha * merged["base_score"] + config.beta * merged["correction_score"]

    ordered_cols = [config.time_col, config.grid_id_col, "base_score", "correction_score", "final_score"]
    other_cols = [col for col in merged.columns if col not in ordered_cols]
    return merged[ordered_cols + other_cols]


def evaluate_final_scores(pred_final_df: pd.DataFrame, actual_df: pd.DataFrame, config: Config) -> Dict[str, float]:
    actual_df = actual_df.copy()
    require_columns(actual_df, [config.grid_id_col, config.final_actual_col], "final_actual.csv")

    merge_cols = [config.grid_id_col]
    if config.time_col in pred_final_df.columns and config.time_col in actual_df.columns:
        actual_df[config.time_col] = pd.to_datetime(actual_df[config.time_col])
        merge_cols.append(config.time_col)

    merged = pred_final_df.merge(actual_df[merge_cols + [config.final_actual_col]], on=merge_cols, how="inner")
    if merged.empty:
        raise ValueError("No overlapping rows were found between predictions and final_actual.csv.")

    y_true = merged[config.final_actual_col].to_numpy()
    y_pred = merged["final_score"].to_numpy()
    ids = merged[config.grid_id_col].to_numpy()
    metrics = regression_metrics(y_true, y_pred)
    metrics[f"precision_at_{config.topk}"] = precision_at_k(y_pred, y_true, ids, k=min(config.topk, len(merged)))
    return metrics


def scenario_check(pred_final_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    score_summary = pred_final_df[["base_score", "correction_score", "final_score"]].describe().T.reset_index()
    score_summary = score_summary.rename(columns={"index": "metric"})
    score_summary[config.time_col] = pd.Series([pd.NaT] * len(score_summary), dtype="datetime64[ns]")
    score_summary["time_mean"] = np.nan
    score_summary["time_max"] = np.nan
    score_summary["time_min"] = np.nan
    time_summary = (
        pred_final_df.groupby(config.time_col)["final_score"]
        .agg(time_mean="mean", time_max="max", time_min="min")
        .reset_index()
    )
    ordered_cols = list(score_summary.columns)
    time_summary["metric"] = np.nan
    for col in ordered_cols:
        if col not in time_summary.columns:
            time_summary[col] = np.nan
    time_summary = time_summary[ordered_cols]
    return pd.concat([score_summary, time_summary], ignore_index=True)


def save_heatmap_like_plot(pred_final_df: pd.DataFrame, config: Config) -> Optional[str]:
    if plt is None or pred_final_df.empty:
        return None

    lon_col = find_existing_column(pred_final_df, config.lon_col_candidates)
    lat_col = find_existing_column(pred_final_df, config.lat_col_candidates)
    x_col = find_existing_column(pred_final_df, config.x_col_candidates)
    y_col = find_existing_column(pred_final_df, config.y_col_candidates)

    if lon_col and lat_col:
        x_name, y_name = lon_col, lat_col
    elif x_col and y_col:
        x_name, y_name = x_col, y_col
    else:
        print("[INFO] No coordinate columns found. Skipping heatmap output.")
        return None

    plot_df = pred_final_df.dropna(subset=[x_name, y_name, "final_score"]).copy()
    if plot_df.empty:
        return None

    latest_ts = plot_df[config.time_col].max()
    latest_df = plot_df[plot_df[config.time_col] == latest_ts].copy()

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        latest_df[x_name],
        latest_df[y_name],
        c=latest_df["final_score"],
        cmap="YlOrRd",
        s=18,
        alpha=0.85,
    )
    ax.set_title(f"Seoul Grid Final Score Heatmap Snapshot ({pd.Timestamp(latest_ts)})")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    fig.colorbar(scatter, ax=ax, label="final_score")

    output_path = os.path.join(config.output_dir, "final_score_heatmap.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def maybe_collect_realtime_snapshot(config: Config) -> Optional[pd.DataFrame]:
    if not config.seoul_api_url:
        return None
    try:
        payload = fetch_seoul_realtime_data(config.seoul_api_url, config.seoul_api_key)
        realtime_df = normalize_realtime_api_payload(payload, config)
        realtime_df.to_csv(config.seoul_api_save_csv, index=False)
        print(f"Saved real-time API snapshot -> {config.seoul_api_save_csv}")
        return realtime_df
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to collect Seoul OpenAPI snapshot: {exc}")
        return None


def main() -> None:
    config = Config()
    seed_everything(config.seed)
    ensure_dir(config.output_dir)

    print("Using config:")
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))

    maybe_collect_realtime_snapshot(config)

    base_model, base_scaler, base_metrics, base_valid_df = train_base_model(config)
    base_valid_df.to_csv(os.path.join(config.output_dir, "base_valid_predictions.csv"), index=False)

    correction_model, corr_scaler, corr_metrics, corr_valid_df = train_correction_model(config)
    corr_valid_df.to_csv(os.path.join(config.output_dir, "correction_valid_predictions.csv"), index=False)

    base_score_df = predict_base_scores(config, base_model, base_scaler)
    correction_score_df = predict_correction_scores(config, correction_model, corr_scaler)

    final_score_df = combine_scores(base_score_df, correction_score_df, config)
    final_score_path = os.path.join(config.output_dir, "final_scores.csv")
    final_score_df.to_csv(final_score_path, index=False)

    scenario_df = scenario_check(final_score_df, config)
    scenario_df.to_csv(os.path.join(config.output_dir, "scenario_summary.csv"), index=False)

    heatmap_path = save_heatmap_like_plot(final_score_df, config)

    summary: Dict[str, object] = {
        "base_metrics": base_metrics,
        "correction_metrics": corr_metrics,
        "artifacts": {
            "final_scores_csv": final_score_path,
            "heatmap_png": heatmap_path,
        },
    }

    if os.path.exists(config.final_actual_csv):
        actual_df = pd.read_csv(config.final_actual_csv)
        final_metrics = evaluate_final_scores(final_score_df, actual_df, config)
        summary["final_metrics"] = final_metrics
        print("Final Metrics:", final_metrics)
    else:
        print(f"[INFO] {config.final_actual_csv} not found. Skipping final evaluation.")

    save_json(os.path.join(config.output_dir, "metrics_summary.json"), summary)

    print("\n=== Saved Outputs ===")
    print(f"- {os.path.join(config.output_dir, 'base_mlp_best.pt')}")
    print(f"- {os.path.join(config.output_dir, 'correction_lstm_best.pt')}")
    print(f"- {os.path.join(config.output_dir, 'base_valid_predictions.csv')}")
    print(f"- {os.path.join(config.output_dir, 'correction_valid_predictions.csv')}")
    print(f"- {final_score_path}")
    print(f"- {os.path.join(config.output_dir, 'scenario_summary.csv')}")
    print(f"- {os.path.join(config.output_dir, 'metrics_summary.json')}")
    if heatmap_path:
        print(f"- {heatmap_path}")


if __name__ == "__main__":
    main()
