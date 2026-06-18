import json
import logging
from pathlib import Path

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

    def load_daily_prices(self, data_dir="data"):
        data_path = Path(data_dir)
        frames = []
        for fpath in sorted(data_path.glob("*_daily.csv")):
            ticker = fpath.stem.replace("_daily", "")
            df = pd.read_csv(str(fpath))
            required_cols = {"date", "open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns):
                _log.warning("Skipping %s: missing required columns", fpath)
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df["ticker"] = ticker
            df["close_pct_change"] = df["close"].pct_change()
            df["hl_spread"] = df["high"] - df["low"]
            df["hl_spread_pct_change"] = df["hl_spread"].pct_change()
            df["volume_pct_change"] = df["volume"].pct_change()
            next_open = df["open"].shift(-1)
            df["open_next_close_prev_diff"] = next_open - df["close"]
            df["open_next_close_prev_diff_pct"] = df["open_next_close_prev_diff"] / df["close"]
            df["open_next_close_prev_diff_sign"] = np.sign(df["open_next_close_prev_diff"])
            frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def load_news(self, data_dir="data"):
        data_path = Path(data_dir)
        main_records = []
        topic_records = []
        ts_records = []

        for fpath in sorted(data_path.glob("*_news.json")):
            ticker = fpath.stem.replace("_news", "")
            with open(fpath, "r") as f:
                articles = json.load(f)
            for article in articles:
                title = article.get("title", "")
                time_published = article.get("time_published", "")
                summary = article.get("summary", "")
                overall_sentiment_score = article.get("overall_sentiment_score")
                overall_sentiment_label = article.get("overall_sentiment_label", "")
                topics = article.get("topics", [])

                main_records.append({
                    "ticker": ticker,
                    "title": title,
                    "time_published": time_published,
                    "summary": summary,
                    "topics": str([t["topic"] for t in topics]) if topics else "",
                    "overall_sentiment_score": overall_sentiment_score,
                    "overall_sentiment_label": overall_sentiment_label,
                })

                for t in topics:
                    topic_records.append({
                        "ticker": ticker,
                        "title": title,
                        "time_published": time_published,
                        "topic": t.get("topic", ""),
                        "relevance_score": t.get("relevance_score", ""),
                    })

                for ts in article.get("ticker_sentiment", []):
                    ts_records.append({
                        "ticker": ticker,
                        "title": title,
                        "time_published": time_published,
                        "sentiment_ticker": ts.get("ticker", ""),
                        "relevance_score": ts.get("relevance_score", ""),
                        "ticker_sentiment_score": ts.get("ticker_sentiment_score", ""),
                        "ticker_sentiment_label": ts.get("ticker_sentiment_label", ""),
                    })

        news_main = pd.DataFrame(main_records)
        news_topics = pd.DataFrame(topic_records)
        news_ticker_sentiment = pd.DataFrame(ts_records)
        for _df in (news_main, news_topics, news_ticker_sentiment):
            if not _df.empty:
                _df["time_published"] = pd.to_datetime(
                    _df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce"
                )
        return {
            "news_main": news_main,
            "news_topics": news_topics,
            "news_ticker_sentiment": news_ticker_sentiment,
        }

    def join_prices_with_sentiment(self, price_df, news_dict):
        if price_df is None or price_df.empty:
            return pd.DataFrame()

        news_main = news_dict.get("news_main")
        news_topics = news_dict.get("news_topics")
        news_ticker_sentiment = news_dict.get("news_ticker_sentiment")

        sentiment_frames = []

        if news_ticker_sentiment is not None and not news_ticker_sentiment.empty:
            ts = news_ticker_sentiment.copy()
            ts["relevance_score"] = pd.to_numeric(ts["relevance_score"], errors="coerce")
            ts["ticker_sentiment_score"] = pd.to_numeric(ts["ticker_sentiment_score"], errors="coerce")
            ts["weighted_score"] = ts["relevance_score"] * ts["ticker_sentiment_score"]
            ts["date"] = ts["time_published"].dt.normalize()
            ts = ts.groupby(["ticker", "date"], as_index=False).agg(
                sentiment_ticker_count=("sentiment_ticker", "count"),
                avg_weighted_sentiment=("weighted_score", "mean"),
            )
            sentiment_frames.append(ts)

        if news_topics is not None and not news_topics.empty:
            tp = news_topics.copy()
            tp["relevance_score"] = pd.to_numeric(tp["relevance_score"], errors="coerce").fillna(0.0)
            tp["date"] = tp["time_published"].dt.normalize()
            tp = tp.pivot_table(
                index=["ticker", "date"],
                columns="topic",
                values="relevance_score",
                fill_value=0.0,
            ).reset_index()
            tp.columns.name = None
            sentiment_frames.append(tp)

        if news_main is not None and not news_main.empty:
            nm = news_main.copy()
            nm["overall_sentiment_score"] = pd.to_numeric(nm["overall_sentiment_score"], errors="coerce")
            nm["date"] = nm["time_published"].dt.normalize()
            nm = nm.groupby(["ticker", "date"], as_index=False).agg(
                avg_overall_sentiment_score=("overall_sentiment_score", "mean"),
                article_count=("date", "count"),
            )
            sentiment_frames.append(nm)

        if not sentiment_frames:
            _log.warning("No sentiment data available — returning price_df copy")
            return price_df.copy()

        sentiment_df = sentiment_frames[0]
        for sf in sentiment_frames[1:]:
            sentiment_df = pd.merge(sentiment_df, sf, on=["ticker", "date"], how="outer")

        price_key = price_df.copy()
        price_key["date"] = price_key["date"].dt.normalize()

        result = pd.merge(price_key, sentiment_df, on=["ticker", "date"], how="inner")
        _log.info(
            "Joined prices with sentiment: %d rows × %d columns",
            result.shape[0], result.shape[1],
        )
        
        result["final_sentiment_score"] = result[["avg_weighted_sentiment", "avg_overall_sentiment_score"]].mean(axis=1, skipna=True)
        result.drop(["avg_weighted_sentiment", "avg_overall_sentiment_score"], axis=1, inplace=True)
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
