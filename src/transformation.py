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

        endpoint = endpoint.upper()

        if endpoint == "TIME_SERIES_DAILY":
            return self._transform_time_series_daily(data)
        elif endpoint == "TOP_GAINERS_LOSERS":
            return self._transform_gainers_losers(data)
        elif endpoint == "OVERVIEW":
            return self._transform_overview(data)
        elif endpoint == "ETF_PROFILE":
            return self._transform_etf_profile(data)
        else:
            return {"error": f"Unknown endpoint: {endpoint}"}

    def _transform_time_series_daily(self, data):
        series = data.get("Time Series (Daily)")
        if not series:
            note = data.get("Note") or data.get("Information", "")
            return {"error": f"No time series data found. {note}"}

        records = []
        for date_str, values in series.items():
            records.append(
                {
                    "Date": date_str,
                    "Open": float(values["1. open"]),
                    "High": float(values["2. high"]),
                    "Low": float(values["3. low"]),
                    "Close": float(values["4. close"]),
                    "Volume": int(values["5. volume"]),
                }
            )

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date", ascending=True).reset_index(drop=True)
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
        df = _make_single_row_df(data)
        return {"etf_profile": df}


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
