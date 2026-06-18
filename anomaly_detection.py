"""
Anomaly Detection Pipeline
3 algorithms: Z-Score (statistical), Isolation Forest (ML), DBSCAN (clustering)
Target: close_pct_change
Features: 21 price + sentiment + topic features
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from src.features import FeatureEngine

TARGET = 'close_pct_change'
FEATURES = [
    'hl_spread_pct_change', 'volume_pct_change', 'open_next_close_prev_diff_pct',
    'sentiment_ticker_count', 'blockchain', 'earnings', 'economy_fiscal',
    'economy_macro', 'economy_monetary', 'energy_transportation', 'finance',
    'financial_markets', 'ipo', 'life_sciences', 'manufacturing',
    'mergers_and_acquisitions', 'real_estate', 'retail_wholesale',
    'technology', 'article_count', 'final_sentiment_score'
]

MODEL_DIR = './models/anomaly_detection'
IMG_DIR = os.path.join(MODEL_DIR, 'img')
MIN_ZSCORE = 5
MIN_IFOREST = 10
MIN_DBSCAN = 20
ZSCORE_THRESHOLD = 2.5
DBSCAN_MIN_SAMPLES = 5
CONTAMINATION = 'auto'
N_ESTIMATORS = 100
RANDOM_STATE = 42
ALL_COLS = FEATURES + [TARGET]


def load_data():
    fe = FeatureEngine()
    df_daily_prices = fe.load_daily_prices()
    df_news_dict = fe.load_news()
    df = fe.join_prices_with_sentiment(df_daily_prices, df_news_dict)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df


def prepare_timeseries_plot(df):
    groups = []
    for _, grp in df.groupby('ticker', sort=False):
        groups.append(grp)
        sep = pd.DataFrame({c: [np.nan] for c in df.columns})
        groups.append(sep)
    return pd.concat(groups, ignore_index=True)


# ── Algorithm Training Functions ─────────────────────────────────────────────

def run_zscore(df_scope, scope_name):
    values = df_scope[TARGET].values
    mean = float(np.mean(values))
    std = float(np.std(values))
    z_scores = np.abs((values - mean) / std)
    is_anomaly = z_scores > ZSCORE_THRESHOLD
    results = df_scope[['date', 'ticker', TARGET] + FEATURES].copy()
    results['z_score'] = z_scores
    results['is_anomaly'] = is_anomaly
    params = {'threshold': ZSCORE_THRESHOLD, 'mean': round(mean, 6), 'std': round(std, 6),
              'anomaly_rule': f'|z-score| > {ZSCORE_THRESHOLD}'}
    save_path = os.path.join(MODEL_DIR, scope_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'zscore_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    return results, params


def run_isolation_forest(df_scope, scope_name):
    X = df_scope[ALL_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(contamination=CONTAMINATION, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    preds = model.fit_predict(X_scaled)
    is_anomaly = preds == -1
    results = df_scope[['date', 'ticker', TARGET] + FEATURES].copy()
    results['is_anomaly'] = is_anomaly
    results['anomaly_score'] = model.decision_function(X_scaled)
    params = {'contamination': CONTAMINATION, 'n_estimators': N_ESTIMATORS, 'random_state': RANDOM_STATE,
              'anomaly_rule': 'prediction == -1 (below decision_function threshold)'}
    save_path = os.path.join(MODEL_DIR, scope_name)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model, os.path.join(save_path, 'isolation_forest.joblib'))
    joblib.dump(scaler, os.path.join(save_path, 'isolation_forest_scaler.joblib'))
    with open(os.path.join(save_path, 'isolation_forest_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    return results, params


def compute_dbscan_eps(X, min_samples, percentile=90):
    from sklearn.neighbors import NearestNeighbors
    k = min(min_samples, len(X))
    if k < 2:
        return 1.0
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    kth_dist = np.sort(distances[:, -1])
    return max(float(np.percentile(kth_dist, percentile)), 0.1)


def run_dbscan(df_scope, scope_name):
    X = df_scope[ALL_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    eps = compute_dbscan_eps(X_scaled, DBSCAN_MIN_SAMPLES)
    model = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
    labels = model.fit_predict(X_scaled)
    is_anomaly = labels == -1
    results = df_scope[['date', 'ticker', TARGET] + FEATURES].copy()
    results['cluster_label'] = labels
    results['is_anomaly'] = is_anomaly
    num_clusters = len(set(labels) - {-1})
    params = {'eps': round(eps, 4), 'eps_method': 'k-distance 90th percentile',
              'min_samples': DBSCAN_MIN_SAMPLES, 'metric': 'euclidean',
              'num_clusters_found': int(num_clusters), 'anomaly_rule': 'cluster_label == -1'}
    save_path = os.path.join(MODEL_DIR, scope_name)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model, os.path.join(save_path, 'dbscan.joblib'))
    joblib.dump(scaler, os.path.join(save_path, 'dbscan_scaler.joblib'))
    with open(os.path.join(save_path, 'dbscan_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    return results, params


# ── Whole-Model Inference Functions ──────────────────────────────────────────

def _load_whole_params():
    whole_dir = os.path.join(MODEL_DIR, 'whole')
    with open(os.path.join(whole_dir, 'zscore_params.json')) as f:
        z = json.load(f)
    with open(os.path.join(whole_dir, 'isolation_forest_params.json')) as f:
        i = json.load(f)
    return z, i


def infer_zscore_from_whole(df_scope, scope_name):
    z_params, _ = _load_whole_params()
    values = df_scope[TARGET].values
    z_scores = np.abs((values - z_params['mean']) / z_params['std'])
    is_anomaly = z_scores > ZSCORE_THRESHOLD
    results = df_scope[['date', 'ticker', TARGET] + FEATURES].copy()
    results['z_score'] = z_scores
    results['is_anomaly'] = is_anomaly
    params = {**z_params, 'model_source': 'whole-dataset inference', 'threshold': ZSCORE_THRESHOLD}
    save_path = os.path.join(MODEL_DIR, scope_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'zscore_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    return results, params


def infer_iforest_from_whole(df_scope, scope_name):
    whole_dir = os.path.join(MODEL_DIR, 'whole')
    model = joblib.load(os.path.join(whole_dir, 'isolation_forest.joblib'))
    scaler = joblib.load(os.path.join(whole_dir, 'isolation_forest_scaler.joblib'))
    X = df_scope[ALL_COLS].values
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    is_anomaly = preds == -1
    results = df_scope[['date', 'ticker', TARGET] + FEATURES].copy()
    results['is_anomaly'] = is_anomaly
    results['anomaly_score'] = model.decision_function(X_scaled)
    params = {'contamination': CONTAMINATION, 'n_estimators': N_ESTIMATORS,
              'model_source': 'whole-dataset inference'}
    save_path = os.path.join(MODEL_DIR, scope_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'isolation_forest_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    return results, params


# ── Plotting ─────────────────────────────────────────────────────────────────

def _algo_label(scope_name, algo_name, suffix=''):
    safe_scope = scope_name.lower().replace(' ', '_')
    safe_algo = algo_name.lower().replace(' ', '_')
    return f'{safe_scope}_{safe_algo}{suffix}'


def plot_timeseries(results, algo_name, scope_name, show=False):
    fig = go.Figure()
    normal = results[~results['is_anomaly']].copy()
    anomaly = results[results['is_anomaly']].copy()
    n_anom = len(anomaly)
    n_total = len(results)

    if 'ticker' in normal.columns and normal['ticker'].nunique() > 1:
        plot_df = prepare_timeseries_plot(normal)
    else:
        plot_df = normal

    fig.add_trace(go.Scatter(
        x=plot_df['date'], y=plot_df[TARGET],
        mode='lines', name='Normal',
        line=dict(color='#1f77b4', width=1.5), connectgaps=False))
    if n_anom > 0:
        fig.add_trace(go.Scatter(
            x=anomaly['date'], y=anomaly[TARGET],
            mode='markers', name='Anomaly',
            marker=dict(color='red', size=6, symbol='circle'),
            hovertemplate='Date: %{x}<br>Value: %{y:.4f}<extra></extra>'))

    pct = n_anom / n_total * 100 if n_total else 0
    fig.update_layout(
        title=f'{algo_name} &mdash; {scope_name}<br><sup>{n_anom} anomalies out of {n_total} samples ({pct:.1f}%)</sup>',
        xaxis_title='Date', yaxis_title=TARGET, hovermode='x unified',
        template='plotly_white', height=450,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01))
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.write_html(os.path.join(IMG_DIR, _algo_label(scope_name, algo_name, '_timeseries.html')))
    if show:
        fig.show()
    return fig


def plot_dbscan_2d(results, scope_name, x_col='final_sentiment_score', y_col=TARGET, show=False):
    fig = go.Figure()
    normal = results[~results['is_anomaly']]
    anomaly = results[results['is_anomaly']]
    fig.add_trace(go.Scatter(
        x=normal[x_col], y=normal[y_col], mode='markers', name='Normal',
        marker=dict(color='#1f77b4', size=5, opacity=0.6),
        hovertemplate=f'{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<extra></extra>'))
    if len(anomaly) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly[x_col], y=anomaly[y_col], mode='markers', name='Anomaly',
            marker=dict(color='red', size=8, symbol='x'),
            hovertemplate=f'{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<extra></extra>'))
    fig.update_layout(
        title=f'DBSCAN: {x_col} vs {y_col} &mdash; {scope_name}',
        xaxis_title=x_col, yaxis_title=y_col, template='plotly_white', height=500,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01))
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.write_html(os.path.join(IMG_DIR, _algo_label(scope_name, 'dbscan', '_2d_scatter.html')))
    if show:
        fig.show()
    return fig


# ── Summary Helpers ──────────────────────────────────────────────────────────

def _make_row(algorithm, scope_name, n, results, threshold, param_str):
    n_anom = int(results['is_anomaly'].sum()) if results is not None else 0
    anom_mean = None
    if results is not None and n_anom > 0:
        anom_mean = round(float(results.loc[results['is_anomaly'], TARGET].mean()), 6)
    return {
        'algorithm': algorithm,
        'scope': scope_name,
        'n_samples': n,
        'n_anomalies': n_anom,
        'anomaly_rate': f'{n_anom / n * 100:.1f}%' if n else '0.0%',
        'anomaly_mean': anom_mean,
        'threshold': threshold,
        'params': param_str
    }


def _algo_display_name(key):
    return {'zscore': 'Z-Score', 'iforest': 'Isolation Forest', 'dbscan': 'DBSCAN'}.get(key, key)


# ── Core Analysis ────────────────────────────────────────────────────────────

def analyze_scope(df, scope_name, ticker=None, show_charts=False, can_infer=False):
    if ticker:
        scope_df = df[df['ticker'] == ticker].copy()
        label = f'ticker={ticker}'
    else:
        scope_df = df.copy()
        label = 'whole dataset'

    scope_df = scope_df.dropna(subset=ALL_COLS).reset_index(drop=True)
    n = len(scope_df)

    if n == 0:
        print(f'  \u23ed  {label}: 0 samples, skipping')
        return [], {}

    can_train_z = n >= MIN_ZSCORE
    can_train_i = n >= MIN_IFOREST
    can_train_d = n >= MIN_DBSCAN

    if not can_train_z and not can_infer:
        print(f'  \u23ed  {label}: only {n} samples (need {MIN_ZSCORE}), skipping')
        return [], {}

    print(f'  \u2713 {label}: {n} samples')
    rows = []
    results = {}

    # ── Z-Score ──────────────────────────────────────────────────────────
    if can_train_z:
        print(f'    Training Z-Score...', end=' ')
        r, p = run_zscore(scope_df, scope_name)
        print(f'{int(r["is_anomaly"].sum())} anomalies')
        results['zscore'] = r
        rows.append(_make_row('Z-Score', scope_name, n, r, f'|z| > {ZSCORE_THRESHOLD}',
                              f'mean={p["mean"]:.4f}, std={p["std"]:.4f} (trained)'))
    elif can_infer and n > 0:
        print(f'    Inferring Z-Score from whole model...', end=' ')
        r, p = infer_zscore_from_whole(scope_df, scope_name)
        print(f'{int(r["is_anomaly"].sum())} anomalies')
        results['zscore'] = r
        rows.append(_make_row('Z-Score', scope_name, n, r, f'|z| > {ZSCORE_THRESHOLD}',
                              f'mean={p["mean"]:.4f}, std={p["std"]:.4f} (inferred from whole)'))

    # ── Isolation Forest ─────────────────────────────────────────────────
    if can_train_i:
        print(f'    Training Isolation Forest...', end=' ')
        r, p = run_isolation_forest(scope_df, scope_name)
        print(f'{int(r["is_anomaly"].sum())} anomalies')
        results['iforest'] = r
        rows.append(_make_row('Isolation Forest', scope_name, n, r, 'pred == -1',
                              f'contamination={CONTAMINATION} (trained)'))
    elif can_infer and n > 0:
        print(f'    Inferring Isolation Forest from whole model...', end=' ')
        r, p = infer_iforest_from_whole(scope_df, scope_name)
        print(f'{int(r["is_anomaly"].sum())} anomalies')
        results['iforest'] = r
        rows.append(_make_row('Isolation Forest', scope_name, n, r, 'pred == -1',
                              f'contamination={CONTAMINATION} (inferred from whole)'))

    # ── DBSCAN ───────────────────────────────────────────────────────────
    if can_train_d:
        print(f'    Training DBSCAN...', end=' ')
        r, p = run_dbscan(scope_df, scope_name)
        n_anom = int(r['is_anomaly'].sum())
        n_clust = p['num_clusters_found']
        print(f'{n_anom} anomalies, {n_clust} clusters (eps={p["eps"]})')
        results['dbscan'] = r
        rows.append(_make_row('DBSCAN', scope_name, n, r, 'cluster_label == -1',
                              f'eps={p["eps"]}, min_samples={DBSCAN_MIN_SAMPLES}, clusters={n_clust} (trained)'))

    # ── Charts ───────────────────────────────────────────────────────────
    for algo_key in ['zscore', 'iforest']:
        if results.get(algo_key) is not None:
            plot_timeseries(results[algo_key], _algo_display_name(algo_key), scope_name, show=show_charts)
    if results.get('dbscan') is not None:
        plot_timeseries(results['dbscan'], 'DBSCAN', scope_name, show=show_charts)
        plot_dbscan_2d(results['dbscan'], scope_name, show=show_charts)

    return rows, results


# ── Labelled Dataset ─────────────────────────────────────────────────────────

def build_labelled_dataset(base_df, all_labels):
    print(f'\nBuilding labelled dataset...')
    scope_algos = []
    for scope_name, algo_dict in all_labels.items():
        for algo_key, result_df in algo_dict.items():
            if result_df is not None:
                col = f'{scope_name}_{algo_key}'
                result_df = result_df[['date', 'ticker']].copy()
                result_df[col] = (algo_dict[algo_key]['is_anomaly']).astype(int)
                scope_algos.append(result_df[['date', 'ticker', col]])

    if not scope_algos:
        print('  No labelled data to compile.')
        return None

    labels_df = base_df[['date', 'ticker', TARGET] + FEATURES].copy()
    for sa in scope_algos:
        col = sa.columns[-1]
        labels_df = labels_df.merge(sa, on=['date', 'ticker'], how='left')

    # Fill missing scope columns (algorithms not run for a given scope)
    for col in labels_df.columns:
        if col not in ['date', 'ticker', TARGET] + FEATURES:
            labels_df[col] = labels_df[col].fillna(-1).astype(int)

    path = os.path.join(MODEL_DIR, 'labelled_dataset.csv')
    labels_df.to_csv(path, index=False)
    print(f'  Saved {len(labels_df)} rows to {path}')
    print(f'  Columns: {[c for c in labels_df.columns if c not in FEATURES + [TARGET, "date", "ticker"]]}')
    return labels_df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading data...')
    df = load_data()
    print(f'  Loaded {len(df)} rows, {df["ticker"].nunique()} tickers\n')
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    df_clean = df.dropna(subset=ALL_COLS).reset_index(drop=True)
    all_summary_rows = []
    all_labels = {}

    # Phase 1: Whole dataset (always trains all algorithms)
    print('\u2550' * 60)
    print('Phase 1: Whole dataset')
    print('\u2550' * 60)
    rows, results = analyze_scope(df_clean, 'whole', show_charts=True, can_infer=False)
    all_summary_rows.extend(rows)
    all_labels['whole'] = results

    # Phase 2: Per ticker (tiered training + whole-model inference)
    tickers = sorted(df_clean['ticker'].unique())
    print(f'\n\u2550' * 60)
    print(f'Phase 2: {len(tickers)} tickers individually')
    print('\u2550' * 60)
    for ticker in tickers:
        rows, results = analyze_scope(df_clean, ticker, ticker=ticker, show_charts=False, can_infer=True)
        all_summary_rows.extend(rows)
        if results:
            all_labels[ticker] = results

    # Phase 3: Summary table
    print(f'\n\u2550' * 60)
    print('Phase 3: Summary Table')
    print('\u2550' * 60)
    summary = pd.DataFrame(all_summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(MODEL_DIR, 'summary.csv'), index=False)
    print(f'\nSummary saved to: {os.path.join(MODEL_DIR, "summary.csv")}')

    # Phase 4: Labelled dataset
    print(f'\n\u2550' * 60)
    print('Phase 4: Labelled Dataset')
    print('\u2550' * 60)
    build_labelled_dataset(df_clean, all_labels)

    # Phase 5: Done
    print(f'\n\u2550' * 60)
    print('Done.')
    print(f'  Models:    {MODEL_DIR}/<scope>/')
    print(f'  Images:    {IMG_DIR}/')
    print(f'  Summary:   {os.path.join(MODEL_DIR, "summary.csv")}')
    print(f'  Labels:    {os.path.join(MODEL_DIR, "labelled_dataset.csv")}')


if __name__ == '__main__':
    main()
