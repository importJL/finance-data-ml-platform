"""
Backtesting Visualization for Trained ML Classifier Models
==========================================================
Loads all 4 trained classifier models (LSTM, DNN, RandomForest, HistGB)
from models/classifier/, runs inference on test data, and generates an
interactive Plotly dashboard with:

  - Dual y-axis: close price (left) + daily close % change (right)
  - Prediction markers colored by correct/incorrect
  - Hover tooltip with news articles + sentiment at each date
  - Ticker dropdown selector
  - Range slider for date navigation
  - Per-ticker performance metrics table

Usage:
    python backtest_visualize.py                          # pre-computed predictions
    python backtest_visualize.py --rerun                   # re-run inference from saved models
    python backtest_visualize.py --ticker NVDA             # single ticker
    python backtest_visualize.py --output chart.html       # custom output path
    python backtest_visualize.py --no-display              # metrics only, no plot

Output:
    models/classifier/backtest_charts/backtest_dashboard.html
    Console: per-ticker + aggregate metrics table
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_log = logging.getLogger("backtest_visualize")

CLASSIFIER_DIR = "models/classifier"
DATA_DIR = "data"
OUTPUT_DIR = os.path.join(CLASSIFIER_DIR, "backtest_charts")

SEQ_LEN = 20
N_FWD = 5
TARGET = "close_pct_change"
ALL_MODEL_NAMES = ["lstm", "dnn", "randomforest", "histgb"]
NN_NAMES = ["lstm", "dnn"]
TREE_NAMES = ["randomforest", "histgb"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# ── Model Architecture Definitions (for loading saved .pth files) ──────────

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1)).squeeze(-1)


# ── Data Loading ──────────────────────────────────────────────────────────

def load_precomputed():
    path = os.path.join(CLASSIFIER_DIR, "labeled_dataset.csv")
    if not os.path.exists(path):
        _log.error("Pre-computed predictions not found at %s", path)
        _log.error("Run `python classifier_pipeline.py` first, or use --rerun")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["date"])
    _log.info("Loaded %s: %d rows x %d cols", path, len(df), len(df.columns))
    return df


def load_raw_data():
    from src.features import FeatureEngine
    fe = FeatureEngine()
    df_daily = fe.load_daily_prices()
    df_news = fe.load_news()
    df = fe.join_prices_with_sentiment(df_daily, df_news)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    from classifier_pipeline import ALL_COLS as PIPELINE_COLS
    for col in PIPELINE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    _log.info("Loaded raw data: %d rows x %d cols", len(df), len(df.columns))
    return df


def load_benchmark():
    path = os.path.join(CLASSIFIER_DIR, "benchmark.json")
    with open(path) as f:
        info = json.load(f)
    _log.info("Benchmark: %.6f", info["benchmark"])
    return info["benchmark"], info


def load_news_grouped():
    records = []
    for fpath in sorted(Path(DATA_DIR).glob("*_news.json")):
        ticker = fpath.stem.replace("_news", "")
        with open(fpath) as f:
            articles = json.load(f)
        for art in articles:
            ts = art.get("time_published", "")
            try:
                dt = pd.to_datetime(ts, format="%Y%m%dT%H%M%S", errors="coerce")
            except Exception:
                dt = pd.NaT
            records.append({
                "ticker": ticker,
                "date": dt.normalize() if pd.notna(dt) else pd.NaT,
                "title": art.get("title", ""),
                "sentiment_score": art.get("overall_sentiment_score"),
                "sentiment_label": art.get("overall_sentiment_label", ""),
            })

    news_all = pd.DataFrame(records).dropna(subset=["date"])
    grouped = {}
    for (ticker, date), group in news_all.groupby(["ticker", "date"], sort=False):
        articles = []
        for _, row in group.iterrows():
            score = row["sentiment_score"]
            lbl = row["sentiment_label"]
            sentiment_str = f"{lbl} ({score:+.3f})" if pd.notna(score) else lbl
            articles.append({"title": row["title"], "sentiment": sentiment_str})
        key = (ticker, date.date() if hasattr(date, "date") else date)
        grouped[key] = articles
    _log.info("Grouped news: %d ticker-date entries", len(grouped))
    return grouped


def load_nn_scaler():
    path = os.path.join(CLASSIFIER_DIR, "nn_scaler.joblib")
    return joblib.load(path)


def load_trained_models():
    models = {}

    models["randomforest"] = joblib.load(os.path.join(CLASSIFIER_DIR, "classifier_randomforest.joblib"))
    models["histgb"] = joblib.load(os.path.join(CLASSIFIER_DIR, "classifier_histgb.joblib"))
    _log.info("Loaded tree models: randomforest, histgb")

    with open(os.path.join(CLASSIFIER_DIR, "classifier_lstm_metadata.json")) as f:
        lstm_meta = json.load(f)
    with open(os.path.join(CLASSIFIER_DIR, "classifier_dnn_metadata.json")) as f:
        dnn_meta = json.load(f)

    lstm = LSTMModel(
        input_dim=lstm_meta["input_dim"],
        hidden_dim=lstm_meta["best_params"]["hidden_dim"],
        num_layers=lstm_meta["best_params"]["num_layers"],
        dropout=lstm_meta["best_params"]["dropout"],
    )
    lstm.load_state_dict(
        torch.load(os.path.join(CLASSIFIER_DIR, "classifier_lstm.pth"), map_location="cpu", weights_only=True)
    )
    lstm.eval()
    models["lstm"] = lstm
    _log.info("Loaded LSTM (input_dim=%d, hidden=%d, layers=%d)",
              lstm_meta["input_dim"], lstm_meta["best_params"]["hidden_dim"],
              lstm_meta["best_params"]["num_layers"])

    dnn_hidden = (dnn_meta["best_params"]["hidden_1"], dnn_meta["best_params"]["hidden_2"])
    dnn = DNNModel(
        input_dim=dnn_meta["input_dim"],
        hidden_dims=dnn_hidden,
        dropout=dnn_meta["best_params"]["dropout"],
    )
    dnn.load_state_dict(
        torch.load(os.path.join(CLASSIFIER_DIR, "classifier_dnn.pth"), map_location="cpu", weights_only=True)
    )
    dnn.eval()
    models["dnn"] = dnn
    _log.info("Loaded DNN (input_dim=%d, hidden=%s)", dnn_meta["input_dim"], dnn_hidden)

    return models


# ── Inference (--rerun path) ─────────────────────────────────────────────

def create_sequences(df, seq_len=SEQ_LEN, stride=1):
    from classifier_pipeline import ALL_COLS
    data = df[ALL_COLS].fillna(0).values
    n = len(data)
    sequences, seq_indices = [], []
    for s in range(0, n - seq_len + 1, stride):
        end = s + seq_len - 1
        if end < n:
            sequences.append(data[s:s + seq_len])
            seq_indices.append(end)
    return np.array(sequences), np.array(seq_indices)


def scale_sequences(sequences, scaler):
    B, S, F = sequences.shape
    flat = scaler.transform(sequences.reshape(-1, F))
    return flat.reshape(B, S, F)


@torch.no_grad()
def predict_nn(model, sequences, device=DEVICE):
    model = model.to(device)
    seq_t = torch.tensor(sequences, dtype=torch.float32, device=device)
    preds = []
    batch_size = 32
    for i in range(0, len(seq_t), batch_size):
        preds.append(model(seq_t[i:i + batch_size]).cpu().numpy())
    return np.concatenate(preds)


def predict_nn_mapped(model, sequences, indices, benchmark, n_total):
    reg = predict_nn(model, sequences)
    binary = (reg >= benchmark).astype(int)
    mapped = np.full(n_total, np.nan)
    for j, ei in enumerate(indices):
        mapped[ei] = binary[j]
    return mapped


def run_inference(df, models, scaler, benchmark):
    from classifier_pipeline import tree_features as get_tree_features, ALL_COLS

    feat_tree = get_tree_features()
    X_tree = df[feat_tree].fillna(0).values
    n = len(df)
    row_preds = {}

    for name in TREE_NAMES:
        row_preds[name] = models[name].predict(X_tree)

    seq, idx = create_sequences(df, seq_len=SEQ_LEN, stride=1)
    if len(seq) > 0:
        seq_scaled = scale_sequences(seq, scaler)
        for name in NN_NAMES:
            row_preds[name] = predict_nn_mapped(models[name], seq_scaled, idx, benchmark, n)
    else:
        for name in NN_NAMES:
            row_preds[name] = np.full(n, np.nan)

    ensemble = np.full(n, np.nan)
    for i in range(n):
        votes = []
        for name in ALL_MODEL_NAMES:
            v = row_preds[name][i]
            if not (isinstance(v, float) and np.isnan(v)):
                votes.append(int(v))
        if votes:
            ensemble[i] = np.bincount(votes).argmax()

    df = df.copy()
    df["ensemble_pred"] = ensemble
    for name in ALL_MODEL_NAMES:
        df[f"pred_{name}"] = row_preds[name]
    return df


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, name=""):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    if len(yt) == 0:
        return {"name": name, "n_samples": 0}
    return {
        "name": name,
        "n_samples": int(len(yt)),
        "n_pos": int(yt.sum()),
        "n_neg": int((1 - yt).sum()),
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
        "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }


def compute_all_metrics(df):
    results = {}
    for name in ALL_MODEL_NAMES:
        col = f"pred_{name}"
        if col in df.columns:
            results[name] = compute_metrics(df["label"].values, df[col].values, name=name)
    if "ensemble_pred" in df.columns:
        results["ensemble"] = compute_metrics(df["label"].values, df["ensemble_pred"].values, name="ensemble")
    return results


def compute_ticker_metrics(df):
    ticker_results = {}
    for ticker in sorted(df["ticker"].unique()):
        sub = df[df["ticker"] == ticker]
        m = compute_metrics(sub["label"].values, sub["ensemble_pred"].values, name=ticker)
        if m["n_samples"] > 0:
            ticker_results[ticker] = m
    return ticker_results


# ── News Hover Text ──────────────────────────────────────────────────────

def make_news_hover(news_grouped, ticker, date):
    key = (ticker, date)
    articles = news_grouped.get(key, [])
    if not articles:
        return "<i>No news articles on this date</i>"
    lines = ["<b>News & Sentiment:</b><br>"]
    for a in articles:
        title = a["title"]
        if len(title) > 120:
            title = title[:117] + "..."
        lines.append(f"&nbsp;&bull; {title}<br>")
        lines.append(f"&nbsp;&nbsp;&nbsp;{a['sentiment']}<br>")
    return "<br>".join(lines)


# ── Plotting ─────────────────────────────────────────────────────────────

COLOR_CORRECT = "#2ECC71"
COLOR_WRONG = "#E74C3C"
COLOR_CLOSE = "#1A5276"
COLOR_PCT = "#E67E22"

TICKER_DISPLAY = {
    "NVDA": "NVIDIA", "AMD": "AMD", "ASML": "ASML", "AVGO": "Broadcom",
    "MU": "Micron", "ENTG": "Entegris", "VRT": "Vertiv", "PWR": "Quanta Services",
    "CEG": "Constellation Energy", "IREN": "Iris Energy", "ALAB": "Alab",
    "CRWV": "Crossover",
}


def create_backtest_dashboard(df, news_grouped):
    tickers = sorted(df["ticker"].unique())
    _log.info("Building dashboard for %d tickers: %s", len(tickers), tickers)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for ticker in tickers:
        sub = df[df["ticker"] == ticker].sort_values("date").copy()
        if sub.empty:
            continue

        d = sub["date"].dt.date if hasattr(sub["date"].iloc[0], "date") else sub["date"]
        close = sub["close"].values
        pct = sub["close_pct_change"].values * 100
        label = sub["label"].values
        pred = sub["ensemble_pred"].values

        correct_mask = (label == pred) & ~np.isnan(pred)
        wrong_mask = (label != pred) & ~np.isnan(pred)

        label_dates = d.values
        label_close = close
        label_pct = pct

        hover_close = [f"<b>{t}</b><br>Close: ${c:.2f}" for t, c in zip(label_dates, label_close)]
        hover_pct = [f"<b>{t}</b><br>% Change: {p:+.2f}%" for t, p in zip(label_dates, label_pct)]

        # ── Close price line (left y-axis) ──
        fig.add_trace(
            go.Scatter(
                x=label_dates, y=label_close, mode="lines",
                name=f"{ticker} Close", legendgroup=ticker,
                line=dict(color=COLOR_CLOSE, width=1.5),
                hovertemplate="%{text}<extra></extra>",
                text=hover_close,
                visible=True if ticker == tickers[0] else False,
            ),
            secondary_y=False,
        )

        # ── % Change line (right y-axis) ──
        fig.add_trace(
            go.Scatter(
                x=label_dates, y=label_pct, mode="lines",
                name=f"{ticker} %Chg", legendgroup=ticker,
                line=dict(color=COLOR_PCT, width=1.2, dash="dot"),
                hovertemplate="%{text}<extra></extra>",
                text=hover_pct,
                visible=True if ticker == tickers[0] else False,
            ),
            secondary_y=True,
        )

        # ── Prediction markers ──
        for dates_sub, values_sub, mask, color, desc in [
            (label_dates, pred, correct_mask, COLOR_CORRECT, "Correct"),
            (label_dates, pred, wrong_mask, COLOR_WRONG, "Wrong"),
        ]:
            if mask.sum() == 0:
                continue
            x_pts = dates_sub[mask]
            y_pts = close[mask]
            y_pct_pts = pct[mask]
            pred_vals = values_sub[mask].astype(int)
            labels_sub = label[mask].astype(int)
            directions = np.where(pred_vals == 1, "Up ▲", "Down ▼")

            news_hover = []
            for i in range(len(x_pts)):
                dt = x_pts[i]
                dt_key = dt if isinstance(dt, str) else dt
                nh = make_news_hover(news_grouped, ticker, dt_key)
                news_hover.append(
                    f"<b>{dt}</b><br>"
                    f"Close: ${y_pts[i]:.2f}<br>"
                    f"% Chg: {y_pct_pts[i]:+.2f}%<br>"
                    f"Prediction: {directions[i]}<br>"
                    f"Actual: {'Up ▲' if labels_sub[i] == 1 else 'Down ▼'}<br>"
                    f"Result: {desc}<br><br>{nh}"
                )

            marker_symbol = "circle" if desc == "Correct" else "x"
            marker_size = 14 if desc == "Correct" else 12
            fig.add_trace(
                go.Scatter(
                    x=x_pts, y=y_pts, mode="markers",
                    connectgaps=False,
                    name=f"{ticker} {desc}", legendgroup=ticker,
                    marker=dict(
                        size=marker_size, color=color, symbol=marker_symbol,
                        line=dict(width=2, color="black"),
                    ),
                    hovertemplate="%{text}<extra></extra>",
                    text=news_hover,
                    visible=True if ticker == tickers[0] else False,
                ),
                secondary_y=False,
            )

    # ── Dropdown for ticker selection ──
    buttons = []
    for i, ticker in enumerate(tickers):
        vis = [False] * len(fig.data)
        for j in range(len(fig.data)):
            if fig.data[j].legendgroup == ticker:
                vis[j] = True
        display_name = TICKER_DISPLAY.get(ticker, ticker)
        buttons.append(dict(label=f"{ticker} ({display_name})", method="update", args=[{"visible": vis}]))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.02, y=1.0,
            xanchor="left", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            font=dict(size=12),
        )],
        title=dict(
            text="<b>Backtesting Dashboard — T+N Classifier Predictions</b><br>"
                 "<sup>Ensemble (LSTM+DNN+RF+HistGB) | T+5 forward return prediction</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Date", rangeslider=dict(visible=True, thickness=0.08),
            type="date", tickformat="%b %d, %Y",
        ),
        yaxis=dict(title="Close Price ($)", tickprefix="$", side="left"),
        yaxis2=dict(
            title="Daily Close % Change", ticksuffix="%", side="right",
            overlaying="y", anchor="x",
        ),
        hovermode="closest",
        template="plotly_white",
        height=650,
        margin=dict(l=80, r=80, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )

    # Start with first ticker visible
    for j in range(len(fig.data)):
        fig.data[j].visible = (fig.data[j].legendgroup == tickers[0])

    return fig


# ── CLI ──────────────────────────────────────────────────────────────────

def print_metrics_table(all_metrics, ticker_metrics):
    print("\n" + "=" * 90)
    print("  BACKTEST PERFORMANCE METRICS")
    print("=" * 90)

    print(f"\n{'Model':<16} {'n':<6} {'Pos':<6} {'Neg':<6} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 90)
    for name in ALL_MODEL_NAMES + ["ensemble"]:
        m = all_metrics.get(name)
        if m and m["n_samples"] > 0:
            print(f"{name:<16} {m['n_samples']:<6} {m['n_pos']:<6} {m['n_neg']:<6} "
                  f"{m['accuracy']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['f1']:<8.4f}")

    print(f"\n{'Ticker':<8} {'n':<6} {'Pos':<6} {'Neg':<6} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("-" * 90)
    for ticker, m in sorted(ticker_metrics.items()):
        print(f"{ticker:<8} {m['n_samples']:<6} {m['n_pos']:<6} {m['n_neg']:<6} "
              f"{m['accuracy']:<8.4f} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['f1']:<8.4f}")

    # Aggregate (ensemble across all tickers)
    all_ens = [m for m in ticker_metrics.values()]
    if all_ens:
        avg_f1 = np.mean([m["f1"] for m in all_ens])
        avg_acc = np.mean([m["accuracy"] for m in all_ens])
        total_n = sum(m["n_samples"] for m in all_ens)
        print("-" * 90)
        print(f"{'AVERAGE':<8} {total_n:<6} {'':<6} {'':<6} "
              f"{avg_acc:<8.4f} {'':<8} {'':<8} {avg_f1:<8.4f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Backtesting Visualization for ML Classifier Models")
    parser.add_argument("--rerun", action="store_true", help="Re-run inference from saved models instead of using pre-computed predictions")
    parser.add_argument("--ticker", type=str, default=None, help="Filter to specific ticker (default: all)")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path (default: models/classifier/backtest_charts/backtest_dashboard.html)")
    parser.add_argument("--no-display", action="store_true", help="Metrics only, skip plot generation")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ──
    if args.rerun:
        _log.info("── Re-running inference from saved models ──")
        benchmark, benchmark_info = load_benchmark()
        df = load_raw_data()

        _log.info("Computing temporal features and target shift...")
        from classifier_pipeline import compute_temporal_features, shift_target, assign_labels
        df = compute_temporal_features(df)
        df = shift_target(df, n=N_FWD)
        df = assign_labels(df, benchmark)

        _log.info("Loading trained models...")
        models = load_trained_models()
        scaler = load_nn_scaler()

        _log.info("Running inference on all rows...")
        df = run_inference(df, models, scaler, benchmark)
    else:
        _log.info("── Using pre-computed predictions ──")
        df = load_precomputed()

    # Ensure columns
    for col in ["label", "ensemble_pred"]:
        if col not in df.columns:
            _log.error("Required column '%s' not found in data. Run classifier_pipeline.py first or use --rerun.", col)
            sys.exit(1)

    if args.ticker:
        if args.ticker not in df["ticker"].unique():
            _log.error("Ticker '%s' not found in data. Available: %s", args.ticker, sorted(df["ticker"].unique()))
            sys.exit(1)
        df = df[df["ticker"] == args.ticker].copy()
        _log.info("Filtered to ticker: %s (%d rows)", args.ticker, len(df))

    # ── News grouping ──
    _log.info("Grouping news articles by (ticker, date)...")
    news_grouped = load_news_grouped()

    # ── Metrics ──
    _log.info("Computing metrics...")
    all_metrics = compute_all_metrics(df)
    ticker_metrics = compute_ticker_metrics(df)
    print_metrics_table(all_metrics, ticker_metrics)

    if args.no_display:
        _log.info("--no-display set; skipping plot generation")
        return

    # ── Plot ──
    _log.info("Generating interactive Plotly dashboard...")
    fig = create_backtest_dashboard(df, news_grouped)

    output_path = args.output or os.path.join(OUTPUT_DIR, "backtest_dashboard.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    _log.info("Dashboard saved to: %s", output_path)

    print(f"\nInteractive dashboard: file://{os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
