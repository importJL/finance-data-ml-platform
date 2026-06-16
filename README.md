# Alpha Vantage ETL Pipeline + ML Analysis

A Streamlit-based GUI that extracts financial market data via the [Alpha Vantage API](https://www.alphavantage.co/), transforms it into structured formats, and runs on-demand machine learning models for volatility prediction, earnings momentum analysis, and macro regime classification.

**Phase 1:** Single-endpoint ETL (fetch, transform, display, download).
**Phase 2:** Multi-endpoint ML pipeline with Point-in-Time Feature Store, sentiment analysis, earnings estimates, macro indicators, and three trained models.

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

## Usage — Single Endpoint Mode

Select an endpoint from the sidebar dropdown (when **Analysis Module** is set to `None`).

| Endpoint | Symbol Required | Description |
|---|---|---|
| `TIME_SERIES_DAILY` | Yes | Daily OHLCV price data with chart |
| `TOP_GAINERS_LOSERS` | No | Top gainers, losers, and most actively traded |
| `OVERVIEW` | Yes | Company financial overview |
| `ETF_PROFILE` | Yes | ETF profile data |

1. Enter a symbol (hidden when `TOP_GAINERS_LOSERS` is selected).
2. Choose Output Size (`compact` or `full`) — only shown for `TIME_SERIES_DAILY`.
3. Click **Fetch Data**.
4. View results and download as CSV/JSON.

## Usage — ML Analysis Mode

Select an **Analysis Module** from the sidebar dropdown for multi-endpoint ML analysis.

| Module | API Calls | Endpoints Fetched | Model |
|---|---|---|---|
| **Event Risk** | 4 | TIME_SERIES_DAILY, NEWS_SENTIMENT, TREASURY_YIELD (3m+10y) | **XGBoost** Volatility Shock Classifier |
| **Earnings Momentum** | 3 | TIME_SERIES_DAILY, EARNINGS_ESTIMATES, NEWS_SENTIMENT | **LightGBM** PEAD Regressor |
| **Macro Regime** | 4 | TIME_SERIES_DAILY, TREASURY_YIELD (3m+10y), GOLD_SILVER_HISTORY | **Random Forest** Macro Regime Classifier |

1. Select an analysis module.
2. Enter a stock symbol/ticker.
3. Check the API call budget — the app shows how many calls are needed and how many remain today.
4. Click **Run ML Analysis**.
5. The app fetches all required endpoints, engineers features (sentiment rolling averages, yield curve spread, gold momentum, earnings revision velocity), trains the model on-demand using historical data, and displays results.

### Feature Engineering Details

- **Sentiment Features:** 3-day and 7-day rolling averages of news sentiment, Sentiment Shock (today minus 7-day avg)
- **Earnings Features:** Estimate Revision Velocity (30-day and 7-day % change in estimated EPS)
- **Macro Features:** Yield curve spread (10Y−3M), Gold momentum (20-day and 5-day rolling returns)
- **Leakage Check:** Automated unit test verifies zero contemporaneous correlation between features and target

## New Endpoints (Phase 2)

| Endpoint | Required Params | Description |
|---|---|---|
| `NEWS_SENTIMENT` | `tickers` | News articles with sentiment scores, aggregated daily |
| `EARNINGS_ESTIMATES` | `symbol` | Annual and quarterly EPS estimates |
| `TREASURY_YIELD` | `interval` (default: `daily`) | Yield curve data; dual call for 3-month and 10-year |
| `GOLD_SILVER_HISTORY` | `symbol` (default: `GOLD`), `interval` | Historical gold/silver prices |
| `GOLD_SILVER_SPOT` | `symbol` (default: `GOLD`) | Current spot price |
| `CURRENCY_EXCHANGE_RATE` | `from_currency` (default: `USD`), `to_currency` | Forex exchange rate |

## Rate Limits

Alpha Vantage's free tier allows **5 API calls per minute** and **500 per day**. The app enforces this with:
- A sliding-window rate limiter (auto-pauses if 5 calls in 60 seconds is reached)
- A daily call counter (tracked in session state, resets daily)
- A budget warning in the GUI when an analysis module requires more calls than remaining

## Models

All models are trained **on-demand** from the historical data fetched during the session.

| Model | Algorithm | Target | Training |
|---|---|---|---|
| Volatility Shock Classifier | **XGBoost** | Next-day volatility tertile (low/medium/high) | ~1,000 samples from `outputsize=full` price history |
| PEAD Regressor | **LightGBM** | 5-day forward return after earnings | Same feature matrix; predicts drift return % |
| Macro Regime Classifier | **Random Forest** | Regime (expansion/neutral/contraction) | Same feature matrix with 20-day rolling return labels |

Models are cached in memory during the session and can be saved to `models/` for reuse.

## Project Structure

```
├── app.py                    # Streamlit entry point (GUI + ML workflow)
├── src/
│   ├── config.py             # Loads API key from .env
│   ├── extraction.py         # HTTP client, rate limiter, 10 endpoints
│   ├── transformation.py     # JSON parsing, type casting, 10 parsers
│   ├── delivery.py           # CSV/JSON serialization + ML display components
│   ├── feature_store.py      # DuckDB PIT Feature Store (as-of joins)
│   ├── features.py           # Feature Engineering Agent (sentiment, earnings, macro)
│   └── models.py             # Model Training + Inference (XGBoost, LightGBM, RF)
├── models/                   # Saved model files (created on first training)
├── .env.example              # Template for API key config
├── .gitignore
└── requirements.txt
```

## License

MIT
