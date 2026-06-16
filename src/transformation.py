import re

import pandas as pd


def _is_error_response(data):
    return isinstance(data, dict) and "error" in data


def _make_single_row_df(data):
    df = pd.DataFrame([data])
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError):
            pass
    return df


class TransformationAgent:
    def transform(self, endpoint, data):
        if _is_error_response(data):
            return {"error": data["error"]}

        if isinstance(data, dict) and "Information" in data:
            return {"error": data["Information"]}

        ep = endpoint.upper()

        if ep == "TIME_SERIES_DAILY":
            return self._transform_time_series_daily(data)
        elif ep == "TOP_GAINERS_LOSERS":
            return self._transform_gainers_losers(data)
        elif ep == "OVERVIEW":
            return self._transform_overview(data)
        elif ep == "ETF_PROFILE":
            return self._transform_etf_profile(data)
        elif ep == "NEWS_SENTIMENT":
            return self._transform_news_sentiment(data)
        elif ep == "EARNINGS_ESTIMATES":
            return self._transform_earnings_estimates(data)
        elif ep == "TREASURY_YIELDS":
            return self._transform_polygon_treasury_yields(data)
        elif ep.startswith("TREASURY_YIELD"):
            return self._transform_treasury_yield(data, ep)
        elif ep == "GOLD_SILVER_HISTORY":
            return self._transform_gold_silver_history(data)
        elif ep == "GOLD_SILVER_SPOT":
            return self._transform_gold_silver_spot(data)
        elif ep == "CURRENCY_EXCHANGE_RATE":
            return self._transform_currency_exchange(data)
        elif ep == "OPEN_CLOSE":
            return self._transform_polygon_open_close(data)
        elif ep == "NEWS":
            return self._transform_polygon_news(data)
        elif ep == "FX_GROUPED":
            return self._transform_polygon_fx(data)
        else:
            return {"error": f"Unknown endpoint: {endpoint}"}

    # ── Alpha Vantage transformers (existing) ──────────────────

    def _transform_time_series_daily(self, data):
        series = data.get("Time Series (Daily)")
        if not series:
            note = data.get("Note") or data.get("Information", "")
            return {"error": f"No time series data found. {note}"}

        records = []
        for date_str, values in series.items():
            records.append({
                "Date": date_str,
                "Open": float(values["1. open"]),
                "High": float(values["2. high"]),
                "Low": float(values["3. low"]),
                "Close": float(values["4. close"]),
                "Volume": int(values["5. volume"]),
            })

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date", ascending=True).reset_index(drop=True)

        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(20).std()
        return {"time_series_daily": df}

    def _transform_gainers_losers(self, data):
        result = {}
        for key in ["top_gainers", "top_losers", "most_actively_traded"]:
            entries = data.get(key, [])
            if entries:
                parsed = []
                for item in entries:
                    row = {
                        "ticker": item.get("ticker", ""),
                        "price": _safe_float(item.get("price")),
                        "change_amount": _safe_float(item.get("change_amount")),
                        "change_percentage": item.get("change_percentage", ""),
                        "volume": _safe_int(item.get("volume")),
                    }
                    parsed.append(row)
                result[key] = pd.DataFrame(parsed)
            else:
                result[key] = pd.DataFrame()
        return result

    def _transform_overview(self, data):
        if not data or data == {}:
            return {"error": "Empty response from API"}
        df = _make_single_row_df(data)
        return {"overview": df}

    def _transform_etf_profile(self, data):
        if not data or data == {}:
            return {"error": "Empty response from API"}

        profile_data = {k: v for k, v in data.items() if k not in ("sectors", "holdings")}
        result = {"etf_profile": _make_single_row_df(profile_data)}

        sectors = data.get("sectors", [])
        if sectors:
            sectors_df = pd.DataFrame(sectors)
            for col in sectors_df.columns:
                try:
                    sectors_df[col] = pd.to_numeric(sectors_df[col], errors="raise")
                except (ValueError, TypeError):
                    pass
            result["sectors"] = sectors_df

        holdings = data.get("holdings", [])
        if holdings:
            holdings_df = pd.DataFrame(holdings)
            for col in holdings_df.columns:
                try:
                    holdings_df[col] = pd.to_numeric(holdings_df[col], errors="raise")
                except (ValueError, TypeError):
                    pass
            result["holdings"] = holdings_df

        return result

    def _transform_news_sentiment(self, data):
        feed = data.get("feed", [])
        if not feed:
            return {"error": "No news feed data found."}

        records = []
        for article in feed:
            time_published = article.get("time_published", "")
            date_str = time_published[:8] if len(time_published) >= 8 else ""

            overall_score = _safe_float(article.get("overall_sentiment_score"))

            ticker_sentiments = article.get("ticker_sentiment", [])
            ticker_score = None
            for ts in ticker_sentiments:
                ticker_score = _safe_float(ts.get("ticker_sentiment_score"))
                break

            records.append({
                "Date": date_str,
                "overall_sentiment_score": overall_score,
                "ticker_sentiment_score": ticker_score,
                "source": article.get("source", ""),
            })

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["Date"])
        daily = df.groupby("Date").agg(
            sentiment_score=("ticker_sentiment_score", "mean"),
            sentiment_std=("ticker_sentiment_score", "std"),
            article_count=("Date", "count"),
        ).reset_index()
        daily = daily.sort_values("Date").reset_index(drop=True)
        return {"news_sentiment": daily}

    def _transform_earnings_estimates(self, data):
        estimates = data.get("estimates", [])
        if not estimates:
            estimates = data.get("annualEarningsEstimate", []) + data.get("quarterlyEarningsEstimate", [])

        if not estimates:
            return {"error": "No earnings estimates found."}

        records = []
        for est in estimates:
            date_val = est.get("date") or est.get("fiscalDateEnding", "")
            horizon = est.get("horizon", "fiscal year" if est.get("fiscalDateEnding") else "")
            eps_avg = _safe_float(est.get("eps_estimate_average") or est.get("estimatedEPS"))
            eps_high = _safe_float(est.get("eps_estimate_high"))
            eps_low = _safe_float(est.get("eps_estimate_low"))
            records.append({
                "Date": date_val,
                "horizon": horizon,
                "eps_estimate_average": eps_avg,
                "eps_estimate_high": eps_high,
                "eps_estimate_low": eps_low,
                "eps_estimate_analyst_count": _safe_int(est.get("eps_estimate_analyst_count") or est.get("numberOfEstimates")),
                "eps_estimate_average_7_days_ago": _safe_float(est.get("eps_estimate_average_7_days_ago")),
                "eps_estimate_average_30_days_ago": _safe_float(est.get("eps_estimate_average_30_days_ago")),
                "eps_estimate_revision_up_7d": _safe_int(est.get("eps_estimate_revision_up_trailing_7_days")),
                "eps_estimate_revision_down_7d": _safe_int(est.get("eps_estimate_revision_down_trailing_7_days")),
                "revenue_estimate_average": _safe_float(est.get("revenue_estimate_average")),
                "revenue_estimate_high": _safe_float(est.get("revenue_estimate_high")),
                "revenue_estimate_low": _safe_float(est.get("revenue_estimate_low")),
                "revenue_estimate_analyst_count": _safe_int(est.get("revenue_estimate_analyst_count")),
            })

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return {"earnings_estimates": df}

    def _transform_treasury_yield(self, data, endpoint_key):
        if _is_error_response(data):
            return {"error": data["error"]}
        if isinstance(data, dict) and "data" in data:
            records = data["data"]
        elif isinstance(data, list):
            records = data
        else:
            return {"error": "Unexpected treasury yield data format."}

        parsed = []
        for item in records:
            if isinstance(item, dict):
                parsed.append({
                    "Date": item.get("date", item.get("Date", "")),
                    "Value": _safe_float(item.get("value", item.get("Value"))),
                })
            elif isinstance(item, list) and len(item) >= 2:
                parsed.append({"Date": item[0], "Value": _safe_float(item[1])})

        if not parsed:
            return {"error": "No treasury yield data found."}

        df = pd.DataFrame(parsed)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        if "3month" in endpoint_key.lower():
            key = "treasury_yield_3month"
        elif isinstance(data, dict):
            maturity = data.get("maturity", "10year")
            key = "treasury_yield_3month" if maturity == "3month" else "treasury_yield_10year"
        else:
            key = "treasury_yield_10year"
        return {key: df}

    def _transform_gold_silver_history(self, data):
        if isinstance(data, dict) and "data" in data:
            records = data["data"]
        elif isinstance(data, list):
            records = data
        else:
            return {"error": "Unexpected gold/silver history data format."}

        parsed = []
        for item in records:
            if isinstance(item, dict):
                date_str = item.get("date", item.get("Date", ""))
                value = _safe_float(item.get("value", item.get("Value")))
                close = _safe_float(item.get("close", item.get("Close")))
                parsed.append({
                    "Date": date_str,
                    "Open": _safe_float(item.get("open", item.get("Open"))) or value,
                    "High": _safe_float(item.get("high", item.get("High"))) or value,
                    "Low": _safe_float(item.get("low", item.get("Low"))) or value,
                    "Close": close if close is not None else value,
                    "Volume": _safe_float(item.get("volume", item.get("Volume"))),
                })

        if not parsed:
            return {"error": "No gold/silver history data found."}

        df = pd.DataFrame(parsed)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return {"gold_silver_history": df}

    def _transform_gold_silver_spot(self, data):
        if not data or data == {}:
            return {"error": "Empty response from API"}
        record = {
            "nominal": data.get("nominal", ""),
            "timestamp": data.get("timestamp", ""),
            "price": _safe_float(data.get("price")),
        }
        df = pd.DataFrame([record])
        return {"gold_silver_spot": df}

    def _transform_currency_exchange(self, data):
        if not data or data == {}:
            return {"error": "Empty response from API"}

        inner = data.get("Realtime Currency Exchange Rate", data)
        if not isinstance(inner, dict):
            return {"error": "Unexpected currency exchange rate data format."}

        cleaned = {}
        for k, v in inner.items():
            clean_key = re.sub(r"^\d+\.\s*", "", k).strip().lower().replace(" ", "_")
            cleaned[clean_key] = _safe_float(v) if v is not None and re.search(r"[\d.]", str(v)) else v

        df = pd.DataFrame([cleaned])
        return {"currency_exchange": df}

    # ── Polygon transformers ───────────────────────────────────

    def _transform_polygon_open_close(self, data):
        records = data.get("polygon_open_close")
        if not records:
            return {"error": "No open-close data found in Polygon response"}

        df = pd.DataFrame(records)
        df.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }, inplace=True)

        numeric_cols = ["Open", "High", "Low", "Close"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype(int)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date", ascending=True).reset_index(drop=True)

        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(20).std()

        meta = {}
        if "_total_dates" in data:
            meta["total_dates"] = data["_total_dates"]
        if "_fetched" in data:
            meta["fetched"] = data["_fetched"]

        result = {"time_series_daily": df}
        if meta:
            result["_meta"] = meta
        return result

    def _transform_polygon_news(self, data):
        raw = data.get("polygon_news")
        if raw is None:
            return {"error": "No news data found in Polygon response"}

        articles = raw if isinstance(raw, list) else raw.get("results", [raw] if isinstance(raw, dict) else [])
        if not articles:
            return {"error": "No news articles found."}

        SENTIMENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        records = []
        for article in articles:
            published = article.get("published_utc", "")
            date_str = published[:10] if published else ""

            insights = article.get("insights", []) or []
            sentiment_score = None
            for insight in (insights if isinstance(insights, list) else []):
                numeric = SENTIMENT_MAP.get(insight.get("sentiment", "").lower())
                if numeric is not None:
                    sentiment_score = numeric
                    break

            publisher_raw = article.get("publisher")
            source = ""
            if isinstance(publisher_raw, dict):
                source = publisher_raw.get("name", "")
            elif isinstance(publisher_raw, str):
                source = publisher_raw

            records.append({
                "Date": date_str,
                "ticker_sentiment_score": sentiment_score,
                "source": source,
            })

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        daily = df.groupby("Date").agg(
            sentiment_score=("ticker_sentiment_score", "mean"),
            sentiment_std=("ticker_sentiment_score", "std"),
            article_count=("Date", "count"),
        ).reset_index()
        daily = daily.sort_values("Date").reset_index(drop=True)
        return {"news_sentiment": daily}

    def _transform_polygon_treasury_yields(self, data):
        raw = data.get("polygon_treasury_yields")
        if not raw:
            return {"error": "No treasury yield data found in Polygon response"}

        records = raw if isinstance(raw, list) else raw.get("results", [raw] if isinstance(raw, dict) else [])
        if not records:
            return {"error": "No treasury yield records found."}

        def _get_first(record, *keys):
            for key in keys:
                val = record.get(key)
                if val is not None and val != "" and val != "N/A":
                    return val
            return ""

        parsed = []
        for item in records:
            if not isinstance(item, dict):
                continue
            date_val = _get_first(item, "date", "Date", "timestamp", "observation_date")
            maturity_val = _get_first(item, "maturity", "Maturity", "term", "duration", "maturity_duration")
            value_val = _safe_float(_get_first(item, "value", "Value", "yield", "Yield", "close", "rate", "Rate", "percent"))
            if not date_val:
                continue
            parsed.append({"Date": date_val, "maturity": maturity_val, "Value": value_val})

        if not parsed:
            sample_keys = str(list(records[0].keys())) if records else "no records"
            return {"error": f"Could not parse treasury yield records. Sample record keys: {sample_keys}"}

        df = pd.DataFrame(parsed)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        result = {}
        result["all_yields"] = df

        df["_mat_clean"] = df["maturity"].str.lower().str.replace(r"[-\s]", "", regex=True)
        df_3m = df[df["_mat_clean"].str.contains("3month|3m|3mo|^3$", na=False)].copy()
        if not df_3m.empty:
            result["treasury_yield_3month"] = df_3m[["Date", "Value"]].reset_index(drop=True)

        df_10y = df[df["_mat_clean"].str.contains("10year|10y|10yr|^10$", na=False)].copy()
        if not df_10y.empty:
            result["treasury_yield_10year"] = df_10y[["Date", "Value"]].reset_index(drop=True)

        if "treasury_yield_3month" not in result and "treasury_yield_10year" not in result:
            if "maturity" in df.columns:
                maturities = df["maturity"].unique()
                if len(maturities) >= 2:
                    sorted_mats = sorted(maturities)
                    result["treasury_yield_3month"] = df[df["maturity"] == sorted_mats[0]][["Date", "Value"]].reset_index(drop=True)
                    result["treasury_yield_10year"] = df[df["maturity"] == sorted_mats[-1]][["Date", "Value"]].reset_index(drop=True)
                elif len(maturities) == 1:
                    result["treasury_yield_10year"] = df[["Date", "Value"]].reset_index(drop=True)
                    result["treasury_yield_3month"] = pd.DataFrame(columns=["Date", "Value"])

        return result

    def _transform_polygon_fx(self, data):
        raw = data.get("polygon_fx")
        if raw is None:
            return {"error": "No FX data found in Polygon response"}

        records = raw if isinstance(raw, list) else raw.get("results", [raw] if isinstance(raw, dict) else [])
        if not records:
            return {"error": "No FX data available for the selected date."}

        parsed = []
        for item in records:
            if not isinstance(item, dict):
                continue
            parsed.append({
                "Ticker": item.get("T", ""),
                "Open": _safe_float(item.get("o")),
                "High": _safe_float(item.get("h")),
                "Low": _safe_float(item.get("l")),
                "Close": _safe_float(item.get("c")),
                "Volume": _safe_float(item.get("v")),
                "Transactions": _safe_int(item.get("n")),
            })

        if not parsed:
            return {"error": "Could not parse FX records."}

        df = pd.DataFrame(parsed)
        return {"currency_exchange": df}


def _safe_float(val):
    if val is None:
        return None
    try:
        cleaned = str(val).replace("$", "").replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    if val is None:
        return None
    try:
        cleaned = str(val).replace(",", "").strip()
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None
