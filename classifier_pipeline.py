"""
Classifier Pipeline v3 — T+N Forward Prediction
================================================
1. Load data via FeatureEngine, compute benchmark from anomaly models
2. Shift target forward by N=5 trading days (close_pct_change_T+5)
3. Assign binary labels: close_pct_change_T+5 >= benchmark → 1
4. Compute temporal features per ticker (lags, rolling, diff, calendar) for tree models
5. Create overlapping sequences (stride=1, seq_len=20) for LSTM/DNN
6. Time-based 70/15/15 split (no shuffle)
7. Optuna (50 trials each): LSTM & DNN (regression→threshold), RF & HistGB (direct classification)
8. Evaluate on test set
9. Walk-forward backtesting (tree models only, expanding windows)
10. Ensemble majority vote (4 models for test, 2 for walk-forward)
11. Save artifacts to models/classifier/
"""

import json
import logging
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.features import FeatureEngine

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
_log = logging.getLogger(__name__)

TARGET = "close_pct_change"
TARGET_FWD = "close_pct_change_T+5"
N_FWD = 5
SEQ_LEN = 20

FEATURES = [
    "hl_spread_pct_change",
    "volume_pct_change",
    "open_next_close_prev_diff_pct",
    "sentiment_ticker_count",
    "blockchain",
    "earnings",
    "economy_fiscal",
    "economy_macro",
    "economy_monetary",
    "energy_transportation",
    "finance",
    "financial_markets",
    "ipo",
    "life_sciences",
    "manufacturing",
    "mergers_and_acquisitions",
    "real_estate",
    "retail_wholesale",
    "technology",
    "article_count",
    "final_sentiment_score",
]
ALL_COLS = FEATURES + [TARGET]

TEMP_LAG = ["close_pct_change_lag_1", "close_pct_change_lag_2", "close_pct_change_lag_3"]
TEMP_ROLL = ["close_pct_change_roll_5_mean", "close_pct_change_roll_10_mean"]
TEMP_DIFF = ["close_pct_change_prev_diff"]
TEMP_CAL = ["day_of_week", "month", "is_month_start", "is_month_end"]
TEMPORAL_FEATURES = TEMP_LAG + TEMP_ROLL + TEMP_DIFF + TEMP_CAL

WHOLE_MODEL_DIR = "models/anomaly_detection/whole"
CLASSIFIER_DIR = "models/classifier"
OPTUNA_DIR = os.path.join(CLASSIFIER_DIR, "optuna_studies")
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 50
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 8
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ALL_MODEL_NAMES = ["lstm", "dnn", "randomforest", "histgb"]

_log.info("Device: %s", DEVICE)


# ── Data Loading ──────────────────────────────────────────────────────────


def load_data():
    fe = FeatureEngine()
    df_daily = fe.load_daily_prices()
    df_news = fe.load_news()
    df = fe.join_prices_with_sentiment(df_daily, df_news)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def load_whole_models():
    dbscan = joblib.load(os.path.join(WHOLE_MODEL_DIR, "dbscan.joblib"))
    dbscan_scaler = joblib.load(os.path.join(WHOLE_MODEL_DIR, "dbscan_scaler.joblib"))
    iforest = joblib.load(os.path.join(WHOLE_MODEL_DIR, "isolation_forest.joblib"))
    iforest_scaler = joblib.load(os.path.join(WHOLE_MODEL_DIR, "isolation_forest_scaler.joblib"))
    return dbscan, dbscan_scaler, iforest, iforest_scaler


def compute_benchmark(df):
    dbscan, dbscan_scaler, iforest, iforest_scaler = load_whole_models()
    X = df[ALL_COLS].values
    X_dbscan = dbscan_scaler.transform(X)
    dbscan_labels = dbscan.fit_predict(X_dbscan)
    dbscan_anom = dbscan_labels == -1
    X_iforest = iforest_scaler.transform(X)
    iforest_preds = iforest.predict(X_iforest)
    iforest_anom = iforest_preds == -1
    union_anom = dbscan_anom | iforest_anom
    n_anom = int(union_anom.sum())
    if n_anom == 0:
        _log.warning("No anomalies found; using median as fallback benchmark")
        benchmark = float(df[TARGET].median())
    else:
        benchmark = float(df.loc[union_anom, TARGET].mean())
    _log.info("DBSCAN: %d | IForest: %d | Union: %d anomalies",
              int(dbscan_anom.sum()), int(iforest_anom.sum()), n_anom)
    _log.info("Benchmark (mean close_pct_change of union): %.6f", benchmark)
    info = {
        "benchmark": benchmark,
        "n_dbscan_anomalies": int(dbscan_anom.sum()),
        "n_iforest_anomalies": int(iforest_anom.sum()),
        "n_union_anomalies": n_anom,
        "n_total": len(df),
        "anomaly_rate": round(n_anom / len(df), 4),
        "method": "mean close_pct_change of union(DBSCAN, IsolationForest) anomalies",
    }
    return benchmark, info


# ── Target Shift (T+N) ────────────────────────────────────────────────────


def shift_target(df, n=N_FWD):
    df = df.copy().sort_values("date").reset_index(drop=True)
    df[TARGET_FWD] = df[TARGET].shift(-n)
    n_before = len(df)
    df = df.dropna(subset=[TARGET_FWD]).reset_index(drop=True)
    _log.info("Target shift T+%d: %d rows → %d rows (dropped %d)",
              n, n_before, len(df), n_before - len(df))
    return df


def assign_labels(df, benchmark):
    df = df.copy()
    df["label"] = (df[TARGET_FWD] >= benchmark).astype(int)
    n_pos = int(df["label"].sum())
    _log.info("Labels: %d pos / %d (%.1f%%)", n_pos, len(df), 100 * n_pos / len(df))
    return df


# ── Temporal Feature Engineering (per ticker, for tree models) ────────────


def compute_temporal_features(df):
    df = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker")[TARGET]

    df["close_pct_change_lag_1"] = g.shift(1)
    df["close_pct_change_lag_2"] = g.shift(2)
    df["close_pct_change_lag_3"] = g.shift(3)

    df["close_pct_change_roll_5_mean"] = g.transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["close_pct_change_roll_10_mean"] = g.transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    df["close_pct_change_prev_diff"] = (
        df["close_pct_change_lag_1"] - df["close_pct_change_lag_2"]
    )

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_month_start"] = (df["date"].dt.day <= 5).astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    for col in TEMPORAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


def tree_features():
    return FEATURES + TEMPORAL_FEATURES


# ── Sequence Creation (for LSTM/DNN) ──────────────────────────────────────


def create_sequences(df, seq_len=SEQ_LEN, stride=1):
    data = df[ALL_COLS].values
    targets = df[TARGET_FWD].values
    n = len(data)
    sequences, seq_targets, seq_indices = [], [], []
    for s in range(0, n - seq_len + 1, stride):
        end = s + seq_len - 1
        if end < n:
            sequences.append(data[s:s + seq_len])
            seq_targets.append(targets[end])
            seq_indices.append(end)
    _log.info("Created %d sequences (seq_len=%d, stride=%d) from %d rows",
              len(sequences), seq_len, stride, n)
    return np.array(sequences), np.array(seq_targets), np.array(seq_indices)


def split_sequences(sequences, targets, indices, train_end, val_end):
    train_mask = indices < train_end
    val_mask = (indices >= train_end) & (indices < val_end)
    test_mask = indices >= val_end

    return (
        (sequences[train_mask], targets[train_mask], indices[train_mask]),
        (sequences[val_mask], targets[val_mask], indices[val_mask]),
        (sequences[test_mask], targets[test_mask], indices[test_mask]),
    )


# ── PyTorch Dataset ───────────────────────────────────────────────────────


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1)).squeeze(-1)


# ── Training Utilities ────────────────────────────────────────────────────


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs=MAX_EPOCHS,
                patience=EARLY_STOP_PATIENCE, device=DEVICE):
    model = model.to(device)
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += criterion(model(Xb), yb).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _log.debug("Early stop at epoch %d (val_loss=%.6f)", epoch + 1, val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def predict_model(model, loader, device=DEVICE):
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            preds.append(model(Xb).cpu().numpy())
    return np.concatenate(preds)


def create_loader(sequences, targets, batch_size=BATCH_SIZE, shuffle=False):
    dataset = SequenceDataset(sequences, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ── Build Scaler for Sequences ────────────────────────────────────────────


def fit_sequence_scaler(train_sequences):
    B, S, F = train_sequences.shape
    flat = train_sequences.reshape(-1, F)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def scale_sequences(sequences, scaler):
    B, S, F = sequences.shape
    flat = scaler.transform(sequences.reshape(-1, F))
    return flat.reshape(B, S, F)


# ── Split ─────────────────────────────────────────────────────────────────


def temporal_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    _log.info("Time split: %d train / %d val / %d test",
              train_end, val_end - train_end, n - val_end)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def time_based_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    return temporal_split(df, train_ratio, val_ratio)


# ── Optuna ────────────────────────────────────────────────────────────────


def _suggest_params(trial, space):
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            log = spec[3] if len(spec) > 3 else False
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def _tscv_f1(model_cls, params, fixed, X, y):
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        m = model_cls(**{**params, **fixed})
        m.fit(X[tr_idx], y[tr_idx])
        yp = m.predict(X[va_idx])
        scores.append(f1_score(y[va_idx], yp, zero_division=0))
    return float(np.mean(scores))


TREE_SPACE = {
    "randomforest": {
        "class": RandomForestClassifier,
        "fixed": {"random_state": RANDOM_STATE},
        "space": {
            "n_estimators": ("int", 100, 500),
            "max_depth": ("int", 3, 15),
            "min_samples_split": ("int", 2, 10),
            "max_features": ("float", 0.3, 1.0),
        },
    },
    "histgb": {
        "class": HistGradientBoostingClassifier,
        "fixed": {"random_state": RANDOM_STATE},
        "space": {
            "learning_rate": ("float", 0.01, 0.3, True),
            "max_iter": ("int", 100, 500),
            "max_leaf_nodes": ("int", 10, 50),
            "min_samples_leaf": ("int", 5, 50),
        },
    },
}

LSTM_SPACE = {
    "hidden_dim": ("int", 16, 64),
    "num_layers": ("int", 1, 2),
    "dropout": ("float", 0.0, 0.5),
    "lr": ("float", 1e-4, 1e-2, True),
    "weight_decay": ("float", 1e-5, 1e-3, True),
}

DNN_SPACE = {
    "hidden_1": ("int", 32, 256),
    "hidden_2": ("int", 16, 128),
    "dropout": ("float", 0.0, 0.5),
    "lr": ("float", 1e-4, 1e-2, True),
    "weight_decay": ("float", 1e-5, 1e-3, True),
}


def _f1_from_threshold(y_true_reg, y_pred_reg, benchmark):
    y_true_bin = (y_true_reg >= benchmark).astype(int)
    y_pred_bin = (y_pred_reg >= benchmark).astype(int)
    return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))


def objective_lstm(trial, train_seq, train_tgt, val_seq, val_tgt,
                   train_idx, val_idx, benchmark, scaler):
    set_seed()
    params = _suggest_params(trial, LSTM_SPACE)
    input_dim = train_seq.shape[2]

    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )
    train_loader = create_loader(train_seq, train_tgt, shuffle=True)
    val_loader = create_loader(val_seq, val_tgt, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    model, _ = train_model(model, train_loader, val_loader, criterion, optimizer)
    val_preds = predict_model(model, val_loader)
    return _f1_from_threshold(val_tgt, val_preds, benchmark)


def objective_dnn(trial, train_seq, train_tgt, val_seq, val_tgt,
                  train_idx, val_idx, benchmark, scaler):
    set_seed()
    params = _suggest_params(trial, DNN_SPACE)
    input_dim = train_seq.shape[1] * train_seq.shape[2]

    hidden_dims = tuple(d for d in [params["hidden_1"], params["hidden_2"]] if d > 0)
    model = DNNModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=params["dropout"],
    )
    train_loader = create_loader(train_seq, train_tgt, shuffle=True)
    val_loader = create_loader(val_seq, val_tgt, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    model, _ = train_model(model, train_loader, val_loader, criterion, optimizer)
    val_preds = predict_model(model, val_loader)
    return _f1_from_threshold(val_tgt, val_preds, benchmark)


def optimize_tree(name, X_train, y_train):
    cfg = TREE_SPACE[name]
    _log.info("── Optuna: %s (%d trials, TimeSeriesSplit) ──", name, N_OPTUNA_TRIALS)

    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    def objective(trial):
        params = _suggest_params(trial, cfg["space"])
        return _tscv_f1(cfg["class"], params, cfg["fixed"], X_train, y_train)

    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    _log.info("Best %s F1=%.4f | %s", name, study.best_value, study.best_params)

    os.makedirs(OPTUNA_DIR, exist_ok=True)
    joblib.dump(study, os.path.join(OPTUNA_DIR, f"study_{name}.pkl"))

    bp = study.best_params.copy()
    model = cfg["class"](**{**bp, **cfg["fixed"]})
    model.fit(X_train, y_train)
    _log.info("%s retrained on full train (%d samples)", name, len(X_train))
    return model, study.best_value


def optimize_lstm_or_dnn(name, train_seq, train_tgt, val_seq, val_tgt,
                         train_idx, val_idx, benchmark, scaler):
    _log.info("── Optuna: %s (%d trials) ──", name, N_OPTUNA_TRIALS)
    objective_fn = objective_lstm if name == "lstm" else objective_dnn

    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    study.optimize(
        lambda t: objective_fn(t, train_seq, train_tgt, val_seq, val_tgt,
                               train_idx, val_idx, benchmark, scaler),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    _log.info("Best %s F1=%.4f | %s", name, study.best_value, study.best_params)

    os.makedirs(OPTUNA_DIR, exist_ok=True)
    joblib.dump(study, os.path.join(OPTUNA_DIR, f"study_{name}.pkl"))
    return study


# ── Evaluation ────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "n_samples": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except Exception:
            metrics["roc_auc"] = None
    return metrics


def evaluate_trees(models, X_test, y_test):
    results = {}
    for name in ["randomforest", "histgb"]:
        model = models[name]
        y_pred = model.predict(X_test)
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        results[name] = compute_metrics(y_test, y_pred, y_score)
    return results


def evaluate_nn(name, model, test_seq, test_tgt, benchmark, scaler, device=DEVICE):
    test_loader = create_loader(test_seq, test_tgt, shuffle=False)
    y_pred_reg = predict_model(model, test_loader)
    y_true_bin = (test_tgt >= benchmark).astype(int)
    y_pred_bin = (y_pred_reg >= benchmark).astype(int)
    y_score = (y_pred_reg - benchmark) / (np.std(y_pred_reg) + 1e-8)
    metrics = compute_metrics(y_true_bin, y_pred_bin, y_score)
    metrics["algorithm"] = name
    return metrics


def ensemble_predict(models, benchmark, test_seq=None, test_tgt=None,
                     X_test_tree=None, y_test_tree=None, scaler_nn=None):
    preds_list = []

    for name in ALL_MODEL_NAMES:
        model = models[name]
        if name in ("lstm", "dnn"):
            loader = create_loader(test_seq, test_tgt, shuffle=False)
            y_pred_reg = predict_model(model, loader)
            y_pred_bin = (y_pred_reg >= benchmark).astype(int)
            preds_list.append(y_pred_bin)
        else:
            preds_list.append(model.predict(X_test_tree))

    all_preds = np.array(preds_list)
    majority = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds
    )
    y_true = (test_tgt >= benchmark).astype(int) if test_tgt is not None else y_test_tree
    return majority, all_preds, y_true


def predict_full_dataset(models, benchmark, df, scaler_nn):
    """Predict on all rows. Tree models predict directly; NN models map seq predictions to rows."""
    feat_tree = tree_features()
    X_tree = df[feat_tree].fillna(0).values
    n = len(df)
    row_preds = {}

    for name in ["randomforest", "histgb"]:
        row_preds[name] = models[name].predict(X_tree)

    seq, tgt, idx = create_sequences(df, seq_len=SEQ_LEN, stride=1)
    seq_scaled = scale_sequences(seq, scaler_nn)
    for name in ["lstm", "dnn"]:
        loader = create_loader(seq_scaled, tgt, shuffle=False)
        reg = predict_model(models[name], loader)
        binary = (reg >= benchmark).astype(int)
        mapped = np.full(n, np.nan)
        for j, ei in enumerate(idx):
            mapped[ei] = binary[j]
        row_preds[name] = mapped

    ensemble = np.full(n, np.nan)
    for i in range(n):
        votes = []
        for name in ALL_MODEL_NAMES:
            v = row_preds[name][i]
            if not (isinstance(v, float) and np.isnan(v)):
                votes.append(int(v))
        if votes:
            ensemble[i] = np.bincount(votes).argmax()
    return ensemble, row_preds


# ── Retrain NN with best params on full train ─────────────────────────────


def retrain_lstm(train_seq, train_tgt, val_seq, val_tgt, best_params, scaler, input_dim):
    set_seed()
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
    )
    train_loader = create_loader(train_seq, train_tgt, shuffle=True)
    val_loader = create_loader(val_seq, val_tgt, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )
    model, loss = train_model(model, train_loader, val_loader, criterion, optimizer)
    _log.info("LSTM retrained on full train, final val_loss=%.6f", loss)
    return model


def retrain_dnn(train_seq, train_tgt, val_seq, val_tgt, best_params, scaler, input_dim):
    set_seed()
    hidden_dims = tuple(d for d in [best_params.get("hidden_1", 128), best_params.get("hidden_2", 64)] if d > 0)
    model = DNNModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=best_params["dropout"],
    )
    train_loader = create_loader(train_seq, train_tgt, shuffle=True)
    val_loader = create_loader(val_seq, val_tgt, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )
    model, loss = train_model(model, train_loader, val_loader, criterion, optimizer)
    _log.info("DNN retrained on full train, final val_loss=%.6f", loss)
    return model


# ── Walk-Forward Backtesting (tree models only) ───────────────────────────


def walk_forward_backtest(models, df_sorted, step_size=5, init_ratio=0.4):
    n = len(df_sorted)
    init_end = int(n * init_ratio)
    _log.info("Walk-forward: initial=%d rows, step=%d, total=%d",
              init_end, step_size, n)
    feat_list = tree_features()
    tree_names = ["randomforest", "histgb"]
    cfg_map = TREE_SPACE

    all_preds = []
    all_true = []
    window_results = []

    for start in range(init_end, n, step_size):
        end = min(start + step_size, n)
        train_df = df_sorted.iloc[:start]
        pred_df = df_sorted.iloc[start:end]
        if len(pred_df) == 0:
            break

        window_preds = {}
        for name in tree_names:
            cfg = cfg_map[name]
            X_tr = train_df[feat_list].fillna(0).values
            y_tr = train_df["label"].values
            X_te = pred_df[feat_list].fillna(0).values
            m = cfg["class"](**{**models.get(f"best_params_{name}", {}), **cfg["fixed"]})
            m.fit(X_tr, y_tr)
            window_preds[name] = m.predict(X_te)

        y_true = pred_df["label"].values
        ensemble = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0,
            arr=np.array([window_preds[n] for n in tree_names])
        )

        all_true.extend(y_true)
        all_preds.extend(ensemble)
        wm = compute_metrics(np.array(y_true), np.array(ensemble))
        wm["window_start"] = int(start)
        wm["window_end"] = int(end)
        wm["window_size"] = int(len(y_true))
        window_results.append(wm)
        _log.info("  Window [%d:%d] F1=%.4f Acc=%.4f (%d samples)",
                  start, end, wm["f1"], wm["accuracy"], len(y_true))

    agg = compute_metrics(np.array(all_true), np.array(all_preds))
    f1s = [w["f1"] for w in window_results]
    accs = [w["accuracy"] for w in window_results]
    stability = {
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std": round(float(np.std(f1s)), 4),
        "f1_min": round(float(np.min(f1s)), 4),
        "f1_max": round(float(np.max(f1s)), 4),
        "acc_mean": round(float(np.mean(accs)), 4),
        "n_windows": len(window_results),
        "total_predictions": len(all_preds),
    }
    _log.info("Backtest aggregate: F1=%.4f Acc=%.4f (stability σ=%.4f) across %d windows",
              agg["f1"], agg["accuracy"], stability["f1_std"], len(window_results))
    return {
        "aggregate": agg,
        "stability": stability,
        "per_window": window_results,
        "n_predictions": len(all_preds),
    }


# ── Saving ────────────────────────────────────────────────────────────────


def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_artifacts(models, benchmark_info, eval_results, ensemble_metrics,
                   backtest_results, df_labeled_preds, train, val, test,
                   nn_best_params, nn_scaler, lstm_input_dim, dnn_input_dim):
    os.makedirs(CLASSIFIER_DIR, exist_ok=True)

    # Benchmark
    with open(os.path.join(CLASSIFIER_DIR, "benchmark.json"), "w") as f:
        json.dump(benchmark_info, f, indent=2)

    # Labeled + predictions
    df_labeled_preds.to_csv(os.path.join(CLASSIFIER_DIR, "labeled_dataset.csv"), index=False)
    train.to_csv(os.path.join(CLASSIFIER_DIR, "train_data.csv"), index=False)
    val.to_csv(os.path.join(CLASSIFIER_DIR, "val_data.csv"), index=False)
    test.to_csv(os.path.join(CLASSIFIER_DIR, "test_data.csv"), index=False)

    # NN scaler
    if nn_scaler is not None:
        joblib.dump(nn_scaler, os.path.join(CLASSIFIER_DIR, "nn_scaler.joblib"))

    # Models + metadata
    trained_at = datetime.now(timezone.utc).isoformat()
    for name in ALL_MODEL_NAMES:
        model = models[name]
        if name == "lstm":
            torch.save(model.state_dict(), os.path.join(CLASSIFIER_DIR, "classifier_lstm.pth"))
            meta = {
                "algorithm": "LSTM",
                "framework": "PyTorch",
                "type": "regression (thresholded)",
                "input_dim": lstm_input_dim,
                "seq_len": SEQ_LEN,
                "n_fwd": N_FWD,
                "best_params": nn_best_params.get("lstm", {}),
                "features": ALL_COLS,
                "optimization_metric": "f1 (val set)",
                "optimization_library": "optuna",
                "n_trials": N_OPTUNA_TRIALS,
                "trained_at": trained_at,
                "eval_on_test": eval_results.get(name, {}),
            }
            with open(os.path.join(CLASSIFIER_DIR, "classifier_lstm_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
        elif name == "dnn":
            torch.save(model.state_dict(), os.path.join(CLASSIFIER_DIR, "classifier_dnn.pth"))
            meta = {
                "algorithm": "DNN",
                "framework": "PyTorch",
                "type": "regression (thresholded)",
                "input_dim": dnn_input_dim,
                "seq_len": SEQ_LEN,
                "n_fwd": N_FWD,
                "best_params": nn_best_params.get("dnn", {}),
                "features": ALL_COLS,
                "optimization_metric": "f1 (val set)",
                "optimization_library": "optuna",
                "n_trials": N_OPTUNA_TRIALS,
                "trained_at": trained_at,
                "eval_on_test": eval_results.get(name, {}),
            }
            with open(os.path.join(CLASSIFIER_DIR, "classifier_dnn_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
        else:
            cls_name = "RandomForest" if name == "randomforest" else "HistGradientBoosting"
            joblib.dump(model, os.path.join(CLASSIFIER_DIR, f"classifier_{name}.joblib"))
            meta = {
                "algorithm": cls_name,
                "family": "tree",
                "type": "binary_classifier",
                "features": tree_features(),
                "feature_count": len(tree_features()),
                "temporal_features_included": True,
                "class_weight": "balanced",
                "n_fwd": N_FWD,
                "optimization_metric": "f1 (TimeSeriesSplit, 3 folds)",
                "optimization_library": "optuna",
                "n_trials": N_OPTUNA_TRIALS,
                "best_params": model.get_params() if hasattr(model, "get_params") else {},
                "trained_at": trained_at,
                "eval_on_test": eval_results.get(name, {}),
            }
            with open(os.path.join(CLASSIFIER_DIR, f"classifier_{name}_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

    # Predictions CSV subset
    pred_cols = [c for c in df_labeled_preds.columns
                 if c.startswith("pred_") or c in ["date", "ticker", TARGET, TARGET_FWD, "label", "ensemble_pred"]]
    df_labeled_preds[pred_cols].to_csv(os.path.join(CLASSIFIER_DIR, "predictions.csv"), index=False)

    # Temporal feature config
    with open(os.path.join(CLASSIFIER_DIR, "temporal_feature_config.json"), "w") as f:
        json.dump({
            "temporal_features": TEMPORAL_FEATURES,
            "n_fwd": N_FWD,
            "seq_len": SEQ_LEN,
            "models_using_sequences": ["lstm", "dnn"],
            "models_using_temporal_features": ["randomforest", "histgb"],
        }, f, indent=2)

    # Backtest results
    bt_clean = {
        "aggregate": {k: convert(v) for k, v in backtest_results["aggregate"].items()},
        "stability": {k: convert(v) for k, v in backtest_results["stability"].items()},
        "per_window": [{k: convert(v) for k, v in w.items()} for w in backtest_results["per_window"]],
        "n_predictions": backtest_results["n_predictions"],
    }
    with open(os.path.join(CLASSIFIER_DIR, "backtest_results.json"), "w") as f:
        json.dump(bt_clean, f, indent=2)

    # Summary
    summary = {
        "pipeline": "Classifier Pipeline v3 — T+N Forward Prediction",
        "n_fwd": N_FWD,
        "seq_len": SEQ_LEN,
        "model_summary": {
            "lstm": "LSTM regression → thresholded (PyTorch)",
            "dnn": "DNN regression → thresholded (PyTorch)",
            "randomforest": "RandomForest direct classification (balanced)",
            "histgb": "HistGradientBoosting direct classification (balanced)",
        },
        "features": {"sequence_models": ALL_COLS, "tree_models": tree_features()},
        "benchmark": benchmark_info,
        "individual_models_test": eval_results,
        "ensemble_majority_vote_test": ensemble_metrics,
        "walk_forward_backtest": {
            "aggregate": {k: convert(v) for k, v in bt_clean["aggregate"].items()},
            "stability": bt_clean["stability"],
        },
        "walk_forward_models": ["randomforest", "histgb"],
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": round(1 - TRAIN_RATIO - VAL_RATIO, 2)},
        "validation_strategy": "Tree: TimeSeriesSplit (3 folds) | NN: fixed val split",
        "created_at": trained_at,
    }
    with open(os.path.join(CLASSIFIER_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _log.info("All artifacts saved to %s/", CLASSIFIER_DIR)


# ── Main Pipeline ─────────────────────────────────────────────────────────


def main():
    set_seed()
    _log.info("=" * 60)
    _log.info("  Classifier Pipeline v3 — T+N Forward Prediction (N=%d)", N_FWD)
    _log.info("=" * 60)

    # Step 1: Load data
    _log.info("\n[1/7] Loading data via FeatureEngine...")
    df = load_data()
    df_clean = df.dropna(subset=ALL_COLS).reset_index(drop=True)
    _log.info("%d rows, %d tickers", len(df_clean), df_clean["ticker"].nunique())

    # Step 2: Compute benchmark from anomaly models
    _log.info("\n[2/7] Computing benchmark from anomaly models...")
    benchmark, benchmark_info = compute_benchmark(df_clean)
    _log.info("Benchmark: %.6f", benchmark)

    # Step 3: Shift target T+N and assign labels
    _log.info("\n[3/7] Shifting target T+%d and assigning labels...", N_FWD)
    df_shifted = shift_target(df_clean, n=N_FWD)
    df_labeled = assign_labels(df_shifted, benchmark)
    df_labeled = compute_temporal_features(df_labeled)
    _log.info("Temporal features added: %s", TEMPORAL_FEATURES)

    # Step 4: Temporal split
    _log.info("\n[4/7] Temporal split (%.0f/%.0f/%.0f, NO shuffle)...",
              100 * TRAIN_RATIO, 100 * VAL_RATIO, 100 * (1 - TRAIN_RATIO - VAL_RATIO))
    train, val, test = temporal_split(df_labeled)
    _log.info("Train: %d | Val: %d | Test: %d", len(train), len(val), len(test))

    # Step 5: Create sequences for LSTM/DNN
    _log.info("\n[5/7] Creating sequences (seq_len=%d, stride=1)...", SEQ_LEN)
    all_seq, all_tgt, all_idx = create_sequences(df_labeled, seq_len=SEQ_LEN, stride=1)
    train_end = int(len(df_labeled) * TRAIN_RATIO)
    val_end = int(len(df_labeled) * (TRAIN_RATIO + VAL_RATIO))

    (train_seq, train_tgt, train_idx), (val_seq, val_tgt, val_idx), (test_seq, test_tgt, test_idx) = \
        split_sequences(all_seq, all_tgt, all_idx, train_end, val_end)

    _log.info("Sequences: %d train / %d val / %d test",
              len(train_seq), len(val_seq), len(test_seq))

    # Fit scaler on training sequences
    nn_scaler = fit_sequence_scaler(train_seq)
    train_seq_scaled = scale_sequences(train_seq, nn_scaler)
    val_seq_scaled = scale_sequences(val_seq, nn_scaler)
    test_seq_scaled = scale_sequences(test_seq, nn_scaler)

    # Tree model feature matrices (row-level, not sequences)
    feat_tree = tree_features()
    X_train_tree = train[feat_tree].fillna(0).values
    y_train_tree = train["label"].values
    X_val_tree = val[feat_tree].fillna(0).values
    y_val_tree = val["label"].values
    X_test_tree = test[feat_tree].fillna(0).values
    y_test_tree = test["label"].values

    # ── Step 6: Optuna ──
    _log.info("\n[6/7] Optuna optimization (%d trials each)...", N_OPTUNA_TRIALS)

    models = {}
    eval_results = {}
    nn_best_params = {}

    # 6a: LSTM
    _log.info("\n── LSTM Optuna ──")
    lstm_study = optimize_lstm_or_dnn(
        "lstm", train_seq_scaled, train_tgt, val_seq_scaled, val_tgt,
        train_idx, val_idx, benchmark, nn_scaler,
    )
    nn_best_params["lstm"] = lstm_study.best_params
    lstm_input_dim = train_seq_scaled.shape[2]
    models["lstm"] = retrain_lstm(
        np.concatenate([train_seq_scaled, val_seq_scaled], axis=0),
        np.concatenate([train_tgt, val_tgt], axis=0),
        val_seq_scaled, val_tgt,
        lstm_study.best_params, nn_scaler, lstm_input_dim,
    )
    eval_results["lstm"] = evaluate_nn(
        "lstm", models["lstm"], test_seq_scaled, test_tgt, benchmark, nn_scaler
    )

    # 6b: DNN
    _log.info("\n── DNN Optuna ──")
    dnn_study = optimize_lstm_or_dnn(
        "dnn", train_seq_scaled, train_tgt, val_seq_scaled, val_tgt,
        train_idx, val_idx, benchmark, nn_scaler,
    )
    nn_best_params["dnn"] = dnn_study.best_params
    dnn_input_dim = train_seq_scaled.shape[1] * train_seq_scaled.shape[2]
    models["dnn"] = retrain_dnn(
        np.concatenate([train_seq_scaled, val_seq_scaled], axis=0),
        np.concatenate([train_tgt, val_tgt], axis=0),
        val_seq_scaled, val_tgt,
        dnn_study.best_params, nn_scaler, dnn_input_dim,
    )
    eval_results["dnn"] = evaluate_nn(
        "dnn", models["dnn"], test_seq_scaled, test_tgt, benchmark, nn_scaler
    )

    # 6c: RandomForest
    _log.info("\n── RandomForest Optuna ──")
    models["randomforest"], rf_best = optimize_tree("randomforest", X_train_tree, y_train_tree)
    eval_results["randomforest"] = compute_metrics(
        y_test_tree, models["randomforest"].predict(X_test_tree)
    )

    # 6d: HistGB
    _log.info("\n── HistGradientBoosting Optuna ──")
    models["histgb"], hgb_best = optimize_tree("histgb", X_train_tree, y_train_tree)
    eval_results["histgb"] = compute_metrics(
        y_test_tree, models["histgb"].predict(X_test_tree)
    )

    # Print individual results
    print("\n── Individual Model Performance (Test Set) ──")
    print(f"{'Model':<18} {'Method':<22} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8}")
    print("-" * 72)
    methods = {
        "lstm": "LSTM regression → threshold",
        "dnn": "DNN regression → threshold",
        "randomforest": "RF direct (balanced)",
        "histgb": "HistGB direct (balanced)",
    }
    for name in ALL_MODEL_NAMES:
        m = eval_results[name]
        print(f"{name:<18} {methods[name]:<22} {m['f1']:<8.4f} {m['accuracy']:<8.4f} "
              f"{m['precision']:<8.4f} {m['recall']:<8.4f}")

    # ── Step 7: Ensemble & Backtest ──
    _log.info("\n[7/7] Ensemble + Walk-forward...")

    # Ensemble on test set
    ensemble_preds, all_preds, y_true_ens = ensemble_predict(
        models, benchmark,
        test_seq=test_seq_scaled, test_tgt=test_tgt,
        X_test_tree=X_test_tree, y_test_tree=y_test_tree,
    )
    ensemble_metrics = compute_metrics(y_true_ens, ensemble_preds)

    print(f"\nEnsemble (Majority Vote — All 4 Models):")
    print(f"  Test set: F1={ensemble_metrics['f1']:.4f}  "
          f"Acc={ensemble_metrics['accuracy']:.4f}  "
          f"Prec={ensemble_metrics['precision']:.4f}  "
          f"Recall={ensemble_metrics['recall']:.4f}")

    # Walk-forward (tree models only)
    _log.info("\nWalk-forward backtesting (tree models only, expanding windows)...")
    backtest_results = walk_forward_backtest(models, df_labeled)

    bt = backtest_results
    print(f"\nWalk-Forward Backtest Summary (RF + HistGB):")
    print(f"  Windows:          {bt['stability']['n_windows']}")
    print(f"  Predictions:      {bt['n_predictions']}")
    print(f"  Aggregate F1:     {bt['aggregate']['f1']:.4f}")
    print(f"  Aggregate Acc:    {bt['aggregate']['accuracy']:.4f}")
    print(f"  F1 stability (σ): {bt['stability']['f1_std']:.4f}")
    print(f"  F1 range:         [{bt['stability']['f1_min']:.4f}, {bt['stability']['f1_max']:.4f}]")

    # Attach predictions to full dataframe
    df_labeled_preds = df_labeled.copy()

    full_ensemble_preds, full_row_preds = predict_full_dataset(
        models, benchmark, df_labeled, nn_scaler,
    )

    df_labeled_preds["ensemble_pred"] = full_ensemble_preds
    for name in ALL_MODEL_NAMES:
        df_labeled_preds[f"pred_{name}"] = full_row_preds[name]

    # Save
    save_artifacts(
        models, benchmark_info, eval_results, ensemble_metrics,
        backtest_results, df_labeled_preds, train, val, test,
        nn_best_params, nn_scaler, lstm_input_dim, dnn_input_dim,
    )

    # Final table
    print("\n" + "=" * 72)
    print("  Final Comparison — Test Set")
    print("=" * 72)
    print(f"{'Model':<18} {'Method':<22} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8}")
    print("-" * 72)
    for name in ALL_MODEL_NAMES:
        m = eval_results[name]
        print(f"{name:<18} {methods[name]:<22} {m['f1']:<8.4f} {m['accuracy']:<8.4f} "
              f"{m['precision']:<8.4f} {m['recall']:<8.4f}")
    em = ensemble_metrics
    print(f"{'Ensemble (4)':<18} {'Majority Vote':<22} {em['f1']:<8.4f} {em['accuracy']:<8.4f} "
          f"{em['precision']:<8.4f} {em['recall']:<8.4f}")
    print("-" * 72)
    print(f"\nBacktest Aggregate F1: {bt['aggregate']['f1']:.4f}  |  "
          f"Stability σ: {bt['stability']['f1_std']:.4f}")
    print(f"\nAll artifacts → {CLASSIFIER_DIR}/")

    return models, eval_results, ensemble_metrics, backtest_results


if __name__ == "__main__":
    main()
