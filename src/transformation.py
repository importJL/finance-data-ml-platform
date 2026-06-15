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
        elif ep.startswith("TREASURY_YIELD"):
            return self._transform_treasury_yield(data, ep)
        elif ep == "GOLD_SILVER_HISTORY":
            return self._transform_gold_silver_history(data)
        elif ep == "GOLD_SILVER_SPOT":
            return self._transform_gold_silver_spot(data)
        elif ep == "CURRENCY_EXCHANGE_RATE":
            return self._transform_currency_exchange(data)
        else:
            return {"error": f"Unknown endpoint: {endpoint}"}

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
            return {"error": "No earnings estimates found."}

        records = []
        for est in estimates:
            records.append({
                "Date": est.get("date", ""),
                "horizon": est.get("horizon", ""),
                "eps_estimate_average": _safe_float(est.get("eps_estimate_average")),
                "eps_estimate_high": _safe_float(est.get("eps_estimate_high")),
                "eps_estimate_low": _safe_float(est.get("eps_estimate_low")),
                "eps_estimate_analyst_count": _safe_int(est.get("eps_estimate_analyst_count")),
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
                parsed.append({
                    "Date": item.get("date", item.get("Date", "")),
                    "Open": _safe_float(item.get("open", item.get("Open"))),
                    "High": _safe_float(item.get("high", item.get("High"))),
                    "Low": _safe_float(item.get("low", item.get("Low"))),
                    "Close": _safe_float(item.get("close", item.get("Close"))),
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
        df = _make_single_row_df(data)
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
