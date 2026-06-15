## 1. System Overview
This document defines the architecture and responsibilities of the specialized agents (modules) responsible for Phase 1 of the Alpha Vantage ETL pipeline. The system extracts financial market data via the Alpha Vantage API, transforms it into structured formats, and loads it into a simple GUI for user interaction, visualization, and local export.

## 2. Environment & Configuration
*   **API Key Management:** All agents must retrieve the `ALPHAVANTAGE_API_KEY` exclusively from the `.env` file using a centralized configuration loader (e.g., `python-dotenv`). Hardcoding keys is strictly prohibited.
*   **Rate Limiting:** Alpha Vantage enforces strict rate limits (typically 5 API calls per minute, 500 per day for the free tier). The Extraction Agent must implement a global rate-limiting queue to prevent HTTP 429 errors.

## 3. Agent Definitions

### Agent 1: Interface Agent (GUI & Input Routing)
**Role:** Manages user interaction, validates inputs, and dynamically routes requests to the Extraction Agent.
**Responsibilities:**
*   Render a simple web GUI (e.g., using Streamlit, Gradio, or Dash).
*   Provide a dropdown for endpoint selection: `TIME_SERIES_DAILY`, `TOP_GAINERS_LOSERS`, `OVERVIEW`, `ETF_PROFILE`.
*   **Dynamic Parameter Handling:** 
    *   If `TOP_GAINERS_LOSERS` is selected, hide the symbol input (it requires no symbol).
    *   If any other endpoint is selected, enforce a mandatory text input for `symbol`.
*   Provide an `outputsize` selector for `TIME_SERIES_DAILY` (defaulting to `compact` as requested, but allowing `full` if needed).
*   Trigger the pipeline execution upon user submission.
*   Render the transformed data in tables/charts on the site.
*   Generate and serve download links (CSV/JSON) for the extracted data.

### Agent 2: Extraction Agent (API Fetcher)
**Role:** Handles all HTTP communication with the Alpha Vantage API.
**Responsibilities:**
*   Construct the correct API URLs based on the selected endpoint and parameters.
*   **Endpoint Mapping:**
    *   `TIME_SERIES_DAILY`: `?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={key}`
    *   `TOP_GAINERS_LOSERS`: `?function=TOP_GAINERS_LOSERS&apikey={key}`
    *   `OVERVIEW`: `?function=OVERVIEW&symbol={symbol}&apikey={key}`
    *   `ETF_PROFILE`: `?function=ETF_PROFILE&symbol={symbol}&apikey={key}`
*   Inject the API key from the `.env` configuration.
*   Implement retry logic with exponential backoff for transient network errors.
*   Enforce the global rate limit (sleep between calls if the 5-calls/minute threshold is approached).
*   Pass the raw JSON response to the Transformation Agent.

### Agent 3: Transformation Agent (Data Cleaner)
**Role:** Parses, cleans, and structures the raw JSON payload for the GUI and export.
**Responsibilities:**
*   **Error Checking:** Intercept Alpha Vantage error messages (e.g., "Invalid API call", "Thank you for using Alpha Vantage! Our standard API rate limit is 5...") and convert them into user-friendly GUI alerts.
*   **Data Flattening:** 
    *   For `TIME_SERIES_DAILY`: Extract the `"Time Series (Daily)"` dictionary and convert it into a tabular format (Date, Open, High, Low, Close, Volume).
    *   For `TOP_GAINERS_LOSERS`: Extract the `"top_gainers"`, `"top_losers"`, and `"most_actively_traded"` lists into separate, clean dataframes.
    *   For `OVERVIEW` & `ETF_PROFILE`: Convert the flat JSON key-value pairs into a structured dictionary or single-row dataframe for easy display.
*   Ensure all numeric strings (prices, volumes) are cast to appropriate float/integer types for proper sorting and charting in the GUI.

### Agent 4: Delivery Agent (Display & Export)
**Role:** Formats the transformed data for final user consumption and handles local downloads.
**Responsibilities:**
*   Format the cleaned data into UI components (data tables, basic line charts for time-series).
*   Serialize the transformed data into downloadable formats (`.csv` and `.json`).
*   Manage temporary file storage or in-memory byte streams for the download buttons in the GUI.
*   Clear temporary export files after the session ends to prevent local storage bloat.

## 4. Execution Workflow (Phase 1)
1.  **User Input:** User selects an endpoint via the Interface Agent. If required, they input a `symbol`.
2.  **Validation:** Interface Agent checks if required parameters are present.
3.  **Extraction:** Interface Agent passes the request to the Extraction Agent, which fetches the JSON from Alpha Vantage.
4.  **Transformation:** Raw JSON is passed to the Transformation Agent, which handles parsing, type-casting, and error interception.
5.  **Delivery:** The Delivery Agent receives the structured data, renders it on the GUI, and prepares the download payloads.

## 5. Known Constraints & Phase 1 Limitations
*   **No Caching:** Phase 1 does not include a database or local caching layer. Every GUI submission triggers a live API call. Users must be warned about rate limits.
*   **No Historical Batching:** The GUI only supports single-symbol extraction per request.
*   **Synchronous Execution:** The pipeline runs synchronously. If the API is slow, the GUI will block until the response is received. (Phase 2 should introduce asynchronous fetching or Celery workers).