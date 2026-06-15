import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


class FeatureEngine:
    def sentiment_features(self, sentiment_df):
        if sentiment_df is None or sentiment_df.empty:
            return pd.DataFrame()

        df = sentiment_df.copy().sort_values("Date")
        df["sentiment_3d_avg"] = df["sentiment_score"].rolling(3, min_periods=1).mean()
        df["sentiment_7d_avg"] = df["sentiment_score"].rolling(7, min_periods=1).mean()
        df["sentiment_shock"] = df["sentiment_score"] - df["sentiment_7d_avg"]
        df["article_count_3d_avg"] = df["article_count"].rolling(3, min_periods=1).mean()
        return df

    def earnings_features(self, earnings_df):
        if earnings_df is None or earnings_df.empty:
            return pd.DataFrame()

        df = earnings_df.copy().sort_values("Date")
        eps_col = "eps_estimate_average" if "eps_estimate_average" in df.columns else "estimatedEPS"
        df["estimatedEPS"] = df[eps_col].ffill()
        df["revision_velocity_30d"] = df["estimatedEPS"].pct_change(periods=30)
        df["revision_velocity_7d"] = df["estimatedEPS"].pct_change(periods=7)
        return df

    def macro_features(self, yield_3m_df=None, yield_10y_df=None, gold_df=None):
        result = pd.DataFrame()

        if yield_3m_df is not None and yield_10y_df is not None:
            y3 = yield_3m_df.copy().sort_values("Date")[["Date", "Value"]].rename(columns={"Value": "yield_3m"})
            y10 = yield_10y_df.copy().sort_values("Date")[["Date", "Value"]].rename(columns={"Value": "yield_10y"})
            merged = pd.merge_asof(y10.sort_values("Date"), y3.sort_values("Date"), on="Date", direction="backward")
            merged["yield_spread_10y_3m"] = merged["yield_10y"] - merged["yield_3m"]
            result = merged

        if gold_df is not None and not gold_df.empty:
            g = gold_df.copy().sort_values("Date")[["Date", "Close"]].rename(columns={"Close": "gold_price"})
            if result.empty:
                result = g
            else:
                result = pd.merge_asof(result.sort_values("Date"), g.sort_values("Date"), on="Date", direction="backward")
            result["gold_momentum_20d"] = result["gold_price"].pct_change(periods=20)
            result["gold_momentum_5d"] = result["gold_price"].pct_change(periods=5)

        return result

    def build_all(self, price_df, sentiment_df=None, earnings_df=None,
                  yield_3m_df=None, yield_10y_df=None, gold_df=None):
        if price_df is None or price_df.empty:
            return None, []

        base = price_df[["Date"]].copy()
        warnings = []

        if sentiment_df is not None and not sentiment_df.empty:
            sf = self.sentiment_features(sentiment_df)
            sf = sf[["Date", "sentiment_3d_avg", "sentiment_7d_avg", "sentiment_shock", "article_count_3d_avg"]]
            base = pd.merge_asof(base.sort_values("Date"), sf.sort_values("Date"), on="Date", direction="backward")

        if earnings_df is not None and not earnings_df.empty:
            ef = self.earnings_features(earnings_df)
            ef = ef[["Date", "estimatedEPS", "revision_velocity_30d", "revision_velocity_7d"]]
            base = pd.merge_asof(base.sort_values("Date"), ef.sort_values("Date"), on="Date", direction="backward")

        mf = self.macro_features(yield_3m_df, yield_10y_df, gold_df)
        if mf is not None and not mf.empty:
            merge_cols = [c for c in mf.columns if c != "Date"]
            if merge_cols:
                base = pd.merge_asof(base.sort_values("Date"), mf.sort_values("Date")[["Date"] + merge_cols],
                                     on="Date", direction="backward")

        feature_cols = [c for c in base.columns if c not in ("Date", "Return", "Volatility", "Close")]
        base = base.dropna(subset=feature_cols, how="all")
        base = base.ffill().fillna(0)

        _log.info(
            "Feature matrix: %d rows × %d columns (%d features)",
            base.shape[0], base.shape[1], len(feature_cols),
        )
        return base, warnings

    def run_leakage_check(self, feature_df, target_df=None, target_col="Return"):
        warnings = []
        if feature_df is None or feature_df.empty:
            return warnings
        if target_df is None or target_col not in target_df.columns:
            _log.info("Leakage check skipped — target column '%s' not available", target_col)
            return warnings

        combined = feature_df.copy()
        combined[target_col] = target_df[target_col].values[: len(combined)]

        for col in combined.columns:
            if col in ("Date", target_col, "Close", "Volatility", "Volume", "Open", "High", "Low"):
                continue
            valid = combined[[col, target_col]].dropna()
            if len(valid) < 20:
                continue
            corr = valid[col].corr(valid[target_col])
            if abs(corr) > 0.05:
                warnings.append(
                    f"Leakage suspect: '{col}' has contemporaneous correlation "
                    f"{corr:.4f} with '{target_col}'"
                )
                _log.warning("Leakage suspect: %s (corr=%.4f)", col, corr)

        if not warnings:
            _log.info("Leakage check passed — no contemporaneous correlation detected")
        else:
            _log.warning("Leakage check: %d warning(s)", len(warnings))
        return warnings
