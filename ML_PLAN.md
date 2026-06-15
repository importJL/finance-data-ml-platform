## 1. System Overview & Phase 1 Integration
Phase 2 expands the Alpha Vantage ETL pipeline to include alternative data (News, Earnings, Macro) and deploys Machine Learning models. Because these new endpoints operate on vastly different frequencies (continuous news vs. daily prices vs. quarterly earnings), this phase introduces a **Point-in-Time (PIT) Feature Store**. The system will no longer just cache data; it will strictly align all features to the trading day close to prevent look-ahead bias.

## 2. Architectural Addition: The PIT Feature Store (DuckDB/PostgreSQL)
The Phase 1 SQLite cache is upgraded to a PIT Feature Store. 
*   **Temporal Alignment:** When daily prices are joined with macro data (Yields, Gold) or News, the system must use "as-of" joins. A news article published at 10:00 AM on Tuesday can only be used as a feature to predict Tuesday's close or Wednesday's open; it cannot be used to predict Monday's close.
*   **Forward-Filling Rules:** Macro data (like Treasury Yields) that updates daily will be forward-filled. Earnings estimates (which update sporadically) will be forward-filled from their last known revision date, strictly frozen at the time of the daily market close.

## 3. Updated Phase 1 Agents (Handling New Endpoints)

### Agent 2: Extraction Agent (Upgraded)
**New Responsibilities:**
*   Implement dynamic parameter handling for the new endpoints:
    *   `NEWS_SENTIMENT`: Automatically calculate `time_from` as exactly 90 days prior to the current date. Pass the user's `ticker`.
    *   `EARNINGS_ESTIMATES`: Pass the user's `symbol`.
    *   `TREASURY_YIELD`: Default `interval` to `daily`. Execute two separate calls per run: one for `maturity=3month` and one for `maturity=10year` to calculate the yield curve spread.
    *   `GOLD_SILVER_HISTORY` & `GOLD_SILVER_SPOT`: Default `symbol` to `GOLD` and `interval` to `daily`.
    *   `CURRENCY_EXCHANGE_RATE`: Require `from_currency` (default USD) and `to_currency` (e.g., EUR, JPY).
*   Implement strict rate-limit queuing, as the addition of these endpoints will easily exceed the 500 daily calls if multiple symbols are queried.

### Agent 3: Transformation Agent (Upgraded)
**New Responsibilities:**
*   **NLP Parsing:** Parse the `NEWS_SENTIMENT` JSON. Extract the `ticker_sentiment` array, isolate the `ticker_sentiment_score` (relevance-weighted), and aggregate it to a daily level.
*   **Earnings Structuring:** Flatten the `EARNINGS_ESTIMATES` JSON to isolate the `annualEarningsEstimate` and `quarterlyEarningsEstimate` arrays, specifically tracking the `estimatedEPS` and the date of the estimate.
*   **Macro Resampling:** Ensure `TREASURY_YIELD` and `GOLD_SILVER_HISTORY` data are indexed by date and formatted for "as-of" merging with the daily stock prices.

## 4. New Agent Definitions (Phase 2 ML Pipeline)

### Agent 5: Feature Engineering Agent (Upgraded)
**Role:** Calculates advanced, cross-asset, and alternative features while strictly enforcing temporal boundaries.
**Responsibilities:**
*   **Sentiment Features:** Calculate 3-day and 7-day rolling averages of news sentiment scores. Calculate "Sentiment Shock" (today's sentiment minus the 7-day average).
*   **Earnings Features:** Calculate "Estimate Revision Velocity" (the rate of change in Estimated EPS over the last 30 days).
*   **Macro Features:** Calculate the `10year_minus_3month` yield spread. Calculate the 20-day rolling momentum of Gold prices.
*   **Leakage Check:** Run an automated unit test on every feature to ensure the correlation between the feature at time *t* and the target at time *t* is zero (preventing accidental same-day leakage).

### Agent 6 & 7: Model Training & Inference Agents (Unchanged in role, updated in scope)
**Responsibilities:**
*   **Training (Agent 6):** Train the Volatility Shock Classifier (XGBoost), the PEAD Regressor (LightGBM), and the Macro Regime Classifier (Random Forest). Save models with metadata detailing the exact feature set and temporal alignment rules used.
*   **Inference (Agent 7):** Load the models. When a user queries a stock, Agent 7 must fetch not just the stock's daily data, but also the latest Treasury Yields, Gold prices, and News Sentiment to build the complete feature vector for the live prediction.

## 5. Updated Execution Workflow (Phase 2)
1.  **User Input:** User selects a symbol and chooses an analysis module: "Event Risk", "Earnings Momentum", or "Macro Regime".
2.  **Multi-Endpoint Extraction:** The Extraction Agent fires parallel requests to Alpha Vantage for the stock's `TIME_SERIES_DAILY`, `NEWS_SENTIMENT`, `EARNINGS_ESTIMATES`, and the required macro endpoints (`TREASURY_YIELD`, `GOLD_SILVER`).
3.  **Transformation & PIT Alignment:** The Transformation Agent cleans the data. The Feature Store aligns the intraday news and sporadic earnings estimates to the daily price timeline using strict "as-of" logic.
4.  **Feature Engineering:** The Feature Engine calculates sentiment shocks, estimate revision velocity, and yield curve spreads.
5.  **ML Inference:** The Inference Agent passes the aligned feature vector to the specific model requested by the user.
6.  **Delivery:** The Delivery Agent renders the prediction on the GUI. 
    *   *For Event Risk:* Displays a volatility probability gauge.
    *   *For Earnings:* Displays a projected PEAD return chart.
    *   *For Macro:* Displays a dashboard of current macro indicators and the classified regime.

## 6. Phase 2 Strict Constraints
*   **No Intraday Leakage:** Under no circumstances can a news article timestamped at 14:00 EST be used as a feature to predict the stock's closing price at 16:00 EST on the same day. It must be shifted to predict the *next* day's open or close.
*   **Macro Data Staleness:** If the `TREASURY_YIELD` or `GOLD_SILVER_SPOT` API fails or returns stale data (older than 3 trading days), the ML Inference Agent must abort the Macro Regime prediction and display a "Stale Macro Data" warning, rather than using outdated yields to classify the current regime.
*   **API Call Budgeting:** The Interface Agent must track the user's daily API quota. If a user requests a full multi-endpoint ML analysis, the GUI must warn them: "This analysis requires 6 API calls. You have 10 remaining for today."