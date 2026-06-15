import time
from collections import deque
from datetime import datetime, timedelta, timezone

import requests

from src.config import API_KEY

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

MAX_CALLS = 5
WINDOW_SECONDS = 60
DAILY_LIMIT = 25


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


class ExtractionAgent:
    def __init__(self, rate_limiter=None):
        self.rate_limiter = rate_limiter or _rate_limiter

    @property
    def remaining_calls(self):
        return self.rate_limiter.remaining_calls

    @property
    def daily_count(self):
        return self.rate_limiter.daily_count

    def fetch(self, endpoint, **kwargs):
        if endpoint not in ENDPOINT_MAP:
            return {"error": f"Unknown endpoint: {endpoint}"}

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

    def fetch_multi(self, endpoint_configs):
        results = {}
        for cfg in endpoint_configs:
            ep = cfg["endpoint"]
            kwargs = {k: v for k, v in cfg.items() if k != "endpoint"}
            results[ep] = self.fetch(ep, **kwargs)
        return results

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
