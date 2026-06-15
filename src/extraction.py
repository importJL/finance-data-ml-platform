import time
from collections import deque
from datetime import datetime, timezone

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
}

MAX_CALLS = 5
WINDOW_SECONDS = 60


class RateLimiter:
    def __init__(self, max_calls=MAX_CALLS, window=WINDOW_SECONDS):
        self.max_calls = max_calls
        self.window = window
        self._timestamps: deque[float] = deque()

    def wait_if_needed(self):
        now = time.time()
        while self._timestamps and now - self._timestamps[0] > self.window:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_calls:
            sleep_time = self.window - (now - self._timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._timestamps.popleft()

        self._timestamps.append(time.time())


_rate_limiter = RateLimiter()


class ExtractionAgent:
    def __init__(self, rate_limiter=None):
        self.rate_limiter = rate_limiter or _rate_limiter

    def fetch(self, endpoint, symbol=None, outputsize="compact"):
        if endpoint not in ENDPOINT_MAP:
            return {"error": f"Unknown endpoint: {endpoint}"}

        config = ENDPOINT_MAP[endpoint]
        params = {"function": config["function"], "apikey": API_KEY}

        if "symbol" in config["required_params"]:
            if not symbol:
                return {"error": f"Symbol is required for endpoint {endpoint}"}
            params["symbol"] = symbol

        if outputsize and "outputsize" in config["optional_params"]:
            params["outputsize"] = outputsize

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
