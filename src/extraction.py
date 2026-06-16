import time
from collections import deque
from datetime import date as _date, datetime, timedelta, timezone

import requests

from src.config import API_KEY, POLYGON_API_KEY, POLYGON_BASE_URL

BASE_URL = "https://www.alphavantage.co/query"

ENDPOINT_MAP = {
    "TIME_SERIES_DAILY": {
        "function": "TIME_SERIES_DAILY",
        "required_params": ["symbol"],
        "optional_params": ["outputsize"],
    },
    "TOP_GAINERS_LOSERS": {
        "function": "TOP_GAINERS_LOSERS",
        "required_params": [],
        "optional_params": [],
    },
    "OVERVIEW": {
        "function": "OVERVIEW",
        "required_params": ["symbol"],
        "optional_params": [],
    },
    "ETF_PROFILE": {
        "function": "ETF_PROFILE",
        "required_params": ["symbol"],
        "optional_params": [],
    },
    "NEWS_SENTIMENT": {
        "function": "NEWS_SENTIMENT",
        "required_params": [],
        "optional_params": ["tickers", "topics", "time_from", "time_to", "sort", "limit"],
    },
    "EARNINGS_ESTIMATES": {
        "function": "EARNINGS_ESTIMATES",
        "required_params": ["symbol"],
        "optional_params": [],
    },
    "TREASURY_YIELD": {
        "function": "TREASURY_YIELD",
        "required_params": [],
        "optional_params": ["interval", "maturity"],
    },
    "TREASURY_YIELD_3MONTH": {
        "function": "TREASURY_YIELD",
        "required_params": [],
        "optional_params": ["interval", "maturity"],
    },
    "TREASURY_YIELD_10YEAR": {
        "function": "TREASURY_YIELD",
        "required_params": [],
        "optional_params": ["interval", "maturity"],
    },
    "GOLD_SILVER_HISTORY": {
        "function": "GOLD_SILVER_HISTORY",
        "required_params": ["symbol", "interval"],
        "optional_params": [],
    },
    "GOLD_SILVER_SPOT": {
        "function": "GOLD_SILVER_SPOT",
        "required_params": ["symbol"],
        "optional_params": [],
    },
    "CURRENCY_EXCHANGE_RATE": {
        "function": "CURRENCY_EXCHANGE_RATE",
        "required_params": ["from_currency", "to_currency"],
        "optional_params": [],
    },
}

POLYGON_ENDPOINT_MAP = {
    "OPEN_CLOSE": {
        "path": "/v1/open-close/{stocksTicker}/{date}",
        "required_params": ["stocksTicker"],
        "optional_params": [],
        "replaces": "TIME_SERIES_DAILY",
    },
    "NEWS": {
        "path": "/v2/reference/news",
        "required_params": ["ticker"],
        "optional_params": ["published_utc.gte", "published_utc.lte", "limit"],
        "replaces": "NEWS_SENTIMENT",
    },
    "TREASURY_YIELDS": {
        "path": "/fed/v1/treasury-yields",
        "required_params": [],
        "optional_params": ["date", "date.gte", "date.lte", "limit"],
        "replaces": "TREASURY_YIELD",
    },
    "FX_GROUPED": {
        "path": "/v2/aggs/grouped/locale/global/market/fx/{date}",
        "required_params": ["date"],
        "optional_params": [],
        "replaces": "CURRENCY_EXCHANGE_RATE",
    },
}

MAX_CALLS = 5
WINDOW_SECONDS = 60
DAILY_LIMIT = 25

POLYGON_MAX_CALLS = 5
POLYGON_WINDOW_SECONDS = 60


class RateLimiter:
    def __init__(self, max_calls=MAX_CALLS, window=WINDOW_SECONDS, daily_limit=DAILY_LIMIT):
        self.max_calls = max_calls
        self.window = window
        self.daily_limit = daily_limit
        self._timestamps: deque[float] = deque()
        self._daily_count = 0
        self._daily_reset = datetime.now(timezone.utc).date()

    def _check_daily_reset(self):
        today = datetime.now(timezone.utc).date()
        if today != self._daily_reset:
            self._daily_count = 0
            self._daily_reset = today

    @property
    def remaining_calls(self):
        self._check_daily_reset()
        return max(0, self.daily_limit - self._daily_count)

    @property
    def daily_count(self):
        self._check_daily_reset()
        return self._daily_count

    def wait_if_needed(self):
        self._check_daily_reset()

        if self._daily_count >= self.daily_limit:
            raise RuntimeError(
                f"Daily API limit of {self.daily_limit} calls reached. "
                f"Try again tomorrow."
            )

        now = time.time()
        while self._timestamps and now - self._timestamps[0] > self.window:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_calls:
            sleep_time = self.window - (now - self._timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._timestamps.popleft()

        self._timestamps.append(time.time())
        self._daily_count += 1


_rate_limiter = RateLimiter()
_polygon_rate_limiter = RateLimiter(
    max_calls=POLYGON_MAX_CALLS,
    window=POLYGON_WINDOW_SECONDS,
    daily_limit=10_000,
)


class ExtractionAgent:
    def __init__(self, rate_limiter=None, polygon_rate_limiter=None):
        self.rate_limiter = rate_limiter or _rate_limiter
        self.polygon_rate_limiter = polygon_rate_limiter or _polygon_rate_limiter

    @property
    def remaining_calls(self):
        return self.rate_limiter.remaining_calls

    def fetch(self, source, endpoint, **kwargs):
        if source == "polygon":
            return self._fetch_polygon(endpoint, **kwargs)
        return self._fetch_alphavantage(endpoint, **kwargs)

    def fetch_multi(self, endpoint_configs, source="alphavantage"):
        results = {}
        for cfg in endpoint_configs:
            ep = cfg["endpoint"]
            kwargs = {k: v for k, v in cfg.items() if k != "endpoint"}
            results[ep] = self.fetch(source, ep, **kwargs)
        return results

    # ── Alpha Vantage ──────────────────────────────────────────

    def _fetch_alphavantage(self, endpoint, **kwargs):
        if endpoint not in ENDPOINT_MAP:
            return {"error": f"Unknown Alpha Vantage endpoint: {endpoint}"}

        config = ENDPOINT_MAP[endpoint]

        if endpoint == "NEWS_SENTIMENT":
            if "time_from" not in kwargs:
                kwargs["time_from"] = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y%m%dT%H%M")

        if endpoint == "TREASURY_YIELD":
            kwargs.setdefault("interval", "daily")

        if endpoint == "GOLD_SILVER_HISTORY":
            kwargs.setdefault("symbol", "GOLD")
            kwargs.setdefault("interval", "daily")

        if endpoint == "GOLD_SILVER_SPOT":
            kwargs.setdefault("symbol", "GOLD")

        params = {"function": config["function"], "apikey": API_KEY}

        missing = [p for p in config["required_params"] if p not in kwargs or not kwargs.get(p)]
        if missing:
            return {"error": f"Missing required parameters for {endpoint}: {', '.join(missing)}"}

        for p in config["required_params"]:
            params[p] = kwargs[p]

        for p in config.get("optional_params", []):
            if p in kwargs and kwargs[p] is not None:
                params[p] = kwargs[p]

        return self._make_request(params)

    def _make_request(self, params):
        self.rate_limiter.wait_if_needed()

        for attempt in range(3):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if "Error Message" in data:
                    return {"error": data["Error Message"]}
                if "Note" in data:
                    return {"error": data["Note"]}
                if "Information" in data and ("demo" in data["Information"].lower() or "premium" in data["Information"].lower() or "api key" in data["Information"].lower() or "rate" in data["Information"].lower()):
                    return {"error": data["Information"]}
                return data
            except requests.exceptions.Timeout:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": "Request timed out after 3 attempts"}
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": "Connection failed after 3 attempts"}
            except requests.exceptions.HTTPError as e:
                return {"error": f"HTTP {resp.status_code}: {str(e)}"}
            except ValueError:
                return {"error": "Invalid JSON response from API"}

    # ── Polygon ────────────────────────────────────────────────

    def _fetch_polygon(self, endpoint, **kwargs):
        if not POLYGON_API_KEY:
            return {"error": "POLYGON_API_KEY not configured in .env"}

        if endpoint not in POLYGON_ENDPOINT_MAP:
            return {"error": f"Unknown Polygon endpoint: {endpoint}"}

        config = POLYGON_ENDPOINT_MAP[endpoint]

        if endpoint == "OPEN_CLOSE":
            return self._fetch_polygon_open_close_range(**kwargs)
        elif endpoint == "NEWS":
            return self._fetch_polygon_news(**kwargs)
        elif endpoint == "TREASURY_YIELDS":
            return self._fetch_polygon_treasury_yields(**kwargs)
        elif endpoint == "FX_GROUPED":
            return self._fetch_polygon_fx(**kwargs)

        return {"error": f"Unimplemented Polygon endpoint: {endpoint}"}

    def _polygon_make_request(self, method, path, params=None):
        self.polygon_rate_limiter.wait_if_needed()

        url = f"{POLYGON_BASE_URL}{path}"
        if params is None:
            params = {}
        params["apiKey"] = POLYGON_API_KEY

        for attempt in range(3):
            try:
                resp = requests.request(method, url, params=params, timeout=15)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": "Polygon request timed out after 3 attempts"}
            except requests.exceptions.ConnectionError:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return {"error": "Polygon connection failed after 3 attempts"}
            except requests.exceptions.HTTPError as e:
                return {"error": f"Polygon HTTP {resp.status_code}: {str(e)}"}
            except ValueError:
                return {"error": "Invalid JSON response from Polygon API"}

    def _fetch_polygon_open_close_range(self, stocksTicker=None, start_date=None, end_date=None, days_prior=90, progress_callback=None, **kwargs):
        if not stocksTicker:
            return {"error": "Missing required parameter: stocksTicker"}

        ticker = stocksTicker.upper()
        end = _date.today()
        if end_date:
            try:
                end = datetime.strptime(str(end_date), "%Y-%m-%d").date()
            except ValueError:
                pass
        if start_date:
            try:
                start = datetime.strptime(str(start_date), "%Y-%m-%d").date()
            except ValueError:
                start = end - timedelta(days=int(days_prior))
        else:
            start = end - timedelta(days=int(days_prior))

        date_list = []
        cur = start
        while cur <= end:
            if cur.weekday() < 5:
                date_list.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)

        if not date_list:
            return {"error": "No trading days in the specified date range"}

        total = len(date_list)
        results = []
        first_error = None

        for i, date_str in enumerate(date_list):
            if progress_callback:
                progress_callback(i, total, date_str)

            path = f"/v1/open-close/{ticker}/{date_str}"
            resp_data = self._polygon_make_request("GET", path)

            if resp_data is None:
                continue
            if isinstance(resp_data, dict) and "error" in resp_data:
                if first_error is None:
                    first_error = resp_data["error"]
                continue
            if isinstance(resp_data, dict):
                record = {
                    "date": resp_data.get("from", date_str),
                    "open": resp_data.get("open"),
                    "high": resp_data.get("high"),
                    "low": resp_data.get("low"),
                    "close": resp_data.get("close"),
                    "volume": resp_data.get("volume"),
                }
                if record.get("open") is not None:
                    results.append(record)

        if not results:
            msg = f"No open-close data found for {ticker} in range {start} to {end}"
            if first_error:
                msg += f" (first API error: {first_error})"
            return {"error": msg}

        return {"polygon_open_close": results, "_total_dates": total, "_fetched": len(results)}

    def _fetch_polygon_news(self, ticker=None, **kwargs):
        if not ticker:
            return {"error": "Missing required parameter: ticker"}

        params = {"ticker": ticker.upper(), "limit": int(kwargs.get("limit", 10))}

        gte = kwargs.get("published_utc.gte")
        if gte:
            params["published_utc.gte"] = gte

        lte = kwargs.get("published_utc.lte")
        if lte:
            params["published_utc.lte"] = lte

        data = self._polygon_make_request("GET", "/v2/reference/news", params=params)
        if data is None:
            return {"error": "Polygon news API returned no data (HTTP 404)"}
        if isinstance(data, dict) and "error" in data:
            return data

        results = data.get("results", data if isinstance(data, list) else [])
        return {"polygon_news": results}

    def _fetch_polygon_treasury_yields(self, **kwargs):
        params = {"limit": int(kwargs.get("limit", 100))}

        for key in ("date", "date.gte", "date.lte"):
            val = kwargs.get(key)
            if val:
                params[key] = val

        data = self._polygon_make_request("GET", "/fed/v1/treasury-yields", params=params)
        if data is None:
            return {"error": "Polygon treasury yields API returned no data (HTTP 404)"}
        if isinstance(data, dict) and "error" in data:
            return data

        if isinstance(data, list):
            results = data
        else:
            results = data.get("results") or data.get("data") or []
        return {"polygon_treasury_yields": results}

    def _fetch_polygon_fx(self, date=None, **kwargs):
        if not date:
            return {"error": "Missing required parameter: date (YYYY-MM-DD)"}

        path = f"/v2/aggs/grouped/locale/global/market/fx/{date}"
        data = self._polygon_make_request("GET", path)
        if data is None:
            return {"error": "Polygon FX API returned no data (HTTP 404)"}
        if isinstance(data, list):
            return {"polygon_fx": data}
        if isinstance(data, dict) and "error" in data:
            return data

        results = data.get("results", [])
        return {"polygon_fx": results}
