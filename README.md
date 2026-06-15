# Alpha Vantage ETL Pipeline

A Streamlit-based GUI that extracts financial market data via the [Alpha Vantage API](https://www.alphavantage.co/), transforms it into structured formats, and serves it for viewing and export.

## Requirements

- **Python 3.10+**
- **Alpha Vantage API key** — free tier available at [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)

## Setup

Clone the repo and set up a virtual environment:

```bash
git clone <repo-url> p28
cd p28
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **macOS/Linux**: activate with `source .venv/bin/activate`
> **Windows**: activate with `.venv\Scripts\activate`

## Configuration

Create a `.env` file in the project root with your Alpha Vantage API key:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder:

```
ALPHAVANTAGE_API_KEY=your_actual_api_key_here
```

The app will raise a clear error at startup if the key is missing.

## Running the App

```bash
source .venv/bin/activate
streamlit run app.py
```

Open the URL printed in your terminal (default: `http://localhost:8501`).

## Usage

1. **Select an endpoint** from the sidebar dropdown.

| Endpoint | Symbol Required | Description |
|---|---|---|
| `TIME_SERIES_DAILY` | Yes | Daily OHLCV price data with chart |
| `TOP_GAINERS_LOSERS` | No | Top gainers, losers, and most actively traded |
| `OVERVIEW` | Yes | Company financial overview |
| `ETF_PROFILE` | Yes | ETF profile data |

2. **Enter a symbol** (hidden when `TOP_GAINERS_LOSERS` is selected).
3. **Choose Output Size** (`compact` or `full`) — only shown for `TIME_SERIES_DAILY`.
4. Click **Fetch Data**.
5. View results — tables for all endpoints, plus a line chart for time-series data.
6. **Download** the data as CSV or JSON using the buttons below the results.

## Rate Limits

Alpha Vantage's free tier allows **5 API calls per minute** and **500 per day**. The app enforces this with a built-in rate limiter — it will automatically pause before making a call if the 5-call-per-minute threshold has been reached. Every submission triggers a live API call; there is no caching in Phase 1.

## Project Structure

```
├── app.py                 # Streamlit entry point (GUI)
├── src/
│   ├── config.py          # Loads API key from .env
│   ├── extraction.py      # HTTP client, rate limiter, retry logic
│   ├── transformation.py  # JSON parsing, type casting, error detection
│   └── delivery.py        # CSV/JSON serialization (in-memory)
├── .env.example           # Template for API key config
├── .gitignore
├── requirements.txt
└── AGENTS.md              # Architecture specification
```

## License

MIT
