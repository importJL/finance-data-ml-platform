from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import POLYGON_API_KEY
from src.delivery import DeliveryAgent
from src.extraction import ENDPOINT_MAP, POLYGON_ENDPOINT_MAP, ExtractionAgent
from src.feature_store import PITFeatureStore
from src.features import FeatureEngine
from src.models import MacroRegimeModel, PEADModel, VolatilityShockModel
from src.pipeline_logger import PipelineLogger
from src.transformation import TransformationAgent

st.set_page_config(page_title="Financial Data ETL + ML Analysis", layout="wide")

st.title("Financial Data ETL Pipeline + ML Analysis")
st.caption("Extracts data via Alpha Vantage or Polygon, transforms it, runs ML models, and displays results.")

ANALYSIS_MODULES = {
    "None": {
        "endpoints": [],
        "calls": 0,
        "description": "",
    },
    "Event Risk": {
        "endpoints": [
            {"endpoint": "TIME_SERIES_DAILY", "symbol": None, "outputsize": "compact"},
            {"endpoint": "NEWS_SENTIMENT", "tickers": None},
            {"endpoint": "TREASURY_YIELD_3MONTH", "interval": "daily", "maturity": "3month"},
            {"endpoint": "TREASURY_YIELD_10YEAR", "interval": "daily", "maturity": "10year"},
        ],
        "calls": 4,
        "description": "TIME_SERIES_DAILY + NEWS_SENTIMENT + TREASURY_YIELD (3m + 10y)",
    },
    "Earnings Momentum": {
        "endpoints": [
            {"endpoint": "TIME_SERIES_DAILY", "symbol": None, "outputsize": "compact"},
            {"endpoint": "EARNINGS_ESTIMATES", "symbol": None},
            {"endpoint": "NEWS_SENTIMENT", "tickers": None},
        ],
        "calls": 3,
        "description": "TIME_SERIES_DAILY + EARNINGS_ESTIMATES + NEWS_SENTIMENT",
    },
    "Macro Regime": {
        "endpoints": [
            {"endpoint": "TIME_SERIES_DAILY", "symbol": None, "outputsize": "compact"},
            {"endpoint": "TREASURY_YIELD_3MONTH", "interval": "daily", "maturity": "3month"},
            {"endpoint": "TREASURY_YIELD_10YEAR", "interval": "daily", "maturity": "10year"},
            {"endpoint": "GOLD_SILVER_HISTORY", "symbol": "GOLD", "interval": "daily"},
        ],
        "calls": 4,
        "description": "TIME_SERIES_DAILY + TREASURY_YIELD (3m + 10y) + GOLD_SILVER_HISTORY",
    },
}

POLYGON_ANALYSIS_MODULES = {
    "None": {
        "endpoints": [],
        "calls": 0,
        "description": "",
    },
    "Event Risk": {
        "endpoints": [
            {"endpoint": "OPEN_CLOSE", "stocksTicker": None, "days_prior": None, "end_date": None},
            {"endpoint": "NEWS", "ticker": None},
            {"endpoint": "TREASURY_YIELDS"},
        ],
        "calls": 1 + 1 + 1,
        "description": "OPEN_CLOSE (Polygon) + NEWS (Polygon) + TREASURY_YIELDS (Polygon)",
    },
    "Earnings Momentum": {
        "endpoints": [
            {"endpoint": "OPEN_CLOSE", "stocksTicker": None, "days_prior": None, "end_date": None},
            {"endpoint": "EARNINGS_ESTIMATES", "symbol": None},
            {"endpoint": "NEWS", "ticker": None},
        ],
        "calls": "90+1+1",
        "description": "OPEN_CLOSE (Polygon, ~90 calls) + EARNINGS_ESTIMATES (AV) + NEWS (Polygon)",
    },
    "Macro Regime": {
        "endpoints": [
            {"endpoint": "OPEN_CLOSE", "stocksTicker": None, "days_prior": None, "end_date": None},
            {"endpoint": "TREASURY_YIELDS"},
            {"endpoint": "GOLD_SILVER_HISTORY", "symbol": "GOLD", "interval": "daily"},
        ],
        "calls": "90+1+1",
        "description": "OPEN_CLOSE (Polygon, ~90 calls) + TREASURY_YIELDS (Polygon) + GOLD_SILVER_HISTORY (AV)",
    },
}

if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0
if "api_call_date" not in st.session_state:
    st.session_state.api_call_date = date.today()

DAILY_LIMIT = 25


def get_remaining_calls():
    today = date.today()
    if st.session_state.api_call_date != today:
        st.session_state.api_call_count = 0
        st.session_state.api_call_date = today
    return max(0, DAILY_LIMIT - st.session_state.api_call_count)


def track_api_calls(n):
    today = date.today()
    if st.session_state.api_call_date != today:
        st.session_state.api_call_count = 0
        st.session_state.api_call_date = today
    st.session_state.api_call_count += n


POLYGON_SINGLE_ENDPOINTS = {
    "OPEN_CLOSE": {"source": "polygon"},
    "NEWS": {"source": "polygon"},
    "TREASURY_YIELDS": {"source": "polygon"},
    "FX_GROUPED": {"source": "polygon"},
}

AV_SINGLE_ENDPOINTS = {
    "TOP_GAINERS_LOSERS": {"source": "alphavantage"},
    "OVERVIEW": {"source": "alphavantage"},
    "ETF_PROFILE": {"source": "alphavantage"},
    "EARNINGS_ESTIMATES": {"source": "alphavantage"},
    "GOLD_SILVER_HISTORY": {"source": "alphavantage"},
    "GOLD_SILVER_SPOT": {"source": "alphavantage"},
}

ENDPOINT_DISPLAY_NAMES = {
    "TIME_SERIES_DAILY": "Daily Time Series (Prices)",
    "TOP_GAINERS_LOSERS": "Top Gainers & Losers",
    "OVERVIEW": "Company Overview",
    "ETF_PROFILE": "ETF Profile",
    "NEWS_SENTIMENT": "News & Sentiment",
    "EARNINGS_ESTIMATES": "Earnings Estimates",
    "TREASURY_YIELD": "Treasury Yield",
    "TREASURY_YIELD_3MONTH": "3-Month Treasury Yield",
    "TREASURY_YIELD_10YEAR": "10-Year Treasury Yield",
    "GOLD_SILVER_HISTORY": "Gold / Silver Historical",
    "GOLD_SILVER_SPOT": "Gold / Silver Spot",
    "CURRENCY_EXCHANGE_RATE": "Currency Exchange Rate",
    "OPEN_CLOSE": "Open / Close (Daily Snapshot)",
    "NEWS": "News",
    "TREASURY_YIELDS": "Treasury Yields",
    "FX_GROUPED": "FX Rates (Grouped)",
}

polygon_available = bool(POLYGON_API_KEY)

with st.sidebar:
    st.header("Configuration")

    default_av_endpoints = [e for e in ENDPOINT_MAP.keys() if e not in ("TREASURY_YIELD_3MONTH", "TREASURY_YIELD_10YEAR")]

    mode = st.selectbox(
        "Analysis Module",
        options=list(ANALYSIS_MODULES.keys()),
        index=0,
    )
    is_ml_mode = mode != "None"

    st.divider()

    if is_ml_mode:
        data_source = st.radio(
            "Data Source",
            options=["Alpha Vantage (Fallback)", "Polygon (Primary)"],
            index=0 if not polygon_available else 0,
            disabled=not polygon_available,
            help="Polygon is the primary source for replaced endpoints. Alpha Vantage is used for remaining endpoints."
            if polygon_available
            else "Set POLYGON_API_KEY in .env to enable Polygon source.",
        )
        is_polygon_source = data_source == "Polygon (Primary)"

        symbol = st.text_input("Symbol / Ticker", placeholder="e.g. IBM").strip().upper()

        if is_polygon_source:
            ml_days_prior = st.number_input("Days Prior", min_value=1, max_value=365 * 3, value=90)
            ml_end_date = st.date_input("End Date", value=date.today())

        module_info = POLYGON_ANALYSIS_MODULES[mode] if is_polygon_source else ANALYSIS_MODULES[mode]
        required_calls = module_info["calls"]
        remaining = get_remaining_calls()

        st.caption(f"**Required calls:** {required_calls}")
        if not is_polygon_source:
            st.caption(f"**Remaining (AV):** {remaining}")

        can_run = bool(symbol)
        if not is_polygon_source and isinstance(required_calls, int) and remaining < required_calls:
            st.error(
                f"This analysis requires {required_calls} Alpha Vantage calls, "
                f"but you only have {remaining} remaining today."
            )
            can_run = False
        else:
            st.info(f"Fetches: {module_info['description']}")

        run_analysis = st.button(
            "Run ML Analysis",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )
        fetch = False

    else:
        polygon_endpoint_display = {
            "OPEN_CLOSE": "Open / Close (Daily Snapshot)",
            "NEWS": "News",
            "TREASURY_YIELDS": "Treasury Yields",
            "FX_GROUPED": "FX Rates (Grouped)",
        }
        polygon_endpoint_reverse = {v: k for k, v in polygon_endpoint_display.items()}

        av_endpoint_display = {k: ENDPOINT_DISPLAY_NAMES[k] for k in default_av_endpoints}
        av_endpoint_reverse = {v: k for k, v in av_endpoint_display.items()}

        poly_options = ["— None —"] + list(polygon_endpoint_display.values())
        av_options = ["— None —"] + list(av_endpoint_display.values())

        def clear_av():
            if st.session_state.get("poly_ep", "— None —") != "— None —":
                st.session_state.av_ep = "— None —"

        def clear_poly():
            if st.session_state.get("av_ep", "— None —") != "— None —":
                st.session_state.poly_ep = "— None —"

        poly_selected = st.session_state.get("poly_ep", "— None —")
        av_selected = st.session_state.get("av_ep", "— None —")

        st.subheader("Polygon Endpoints")
        st.selectbox(
            "Select Polygon endpoint",
            options=poly_options,
            key="poly_ep",
            on_change=clear_av,
            label_visibility="collapsed",
        )

        st.subheader("Alpha Vantage Endpoints")
        st.selectbox(
            "Select Alpha Vantage endpoint",
            options=av_options,
            key="av_ep",
            on_change=clear_poly,
            label_visibility="collapsed",
        )

        symbol = ""
        endpoint_params = {}
        source_key = "alphavantage"
        req = []
        opt = []

        if poly_selected != "— None —":
            endpoint = polygon_endpoint_reverse[poly_selected]
            source_key = "polygon"
        elif av_selected != "— None —":
            endpoint = av_endpoint_reverse[av_selected]
            source_key = "alphavantage"
        else:
            endpoint = None

        if endpoint is not None:
            if source_key == "polygon":
                if endpoint == "OPEN_CLOSE":
                    symbol = st.text_input("Ticker Symbol", placeholder="e.g. IBM").strip().upper()
                    endpoint_params["stocksTicker"] = symbol
                    days_prior = st.number_input("Days Prior", min_value=1, max_value=365 * 3, value=90)
                    end_d = st.date_input("End Date", value=date.today())
                    endpoint_params["days_prior"] = int(days_prior)
                    endpoint_params["end_date"] = end_d.strftime("%Y-%m-%d")

                elif endpoint == "NEWS":
                    ticker_val = st.text_input("Ticker Symbol", placeholder="e.g. AAPL").strip().upper()
                    endpoint_params["ticker"] = ticker_val
                    col1, col2 = st.columns(2)
                    with col1:
                        default_gte = date.today() - timedelta(days=90)
                        gte_val = st.date_input("Published After", value=default_gte)
                        if gte_val:
                            endpoint_params["published_utc.gte"] = gte_val.strftime("%Y-%m-%d")
                    with col2:
                        lte_val = st.date_input("Published Before", value=date.today())
                        if lte_val:
                            endpoint_params["published_utc.lte"] = lte_val.strftime("%Y-%m-%d")
                    endpoint_params["limit"] = st.slider("Limit", min_value=1, max_value=1000, value=100)

                elif endpoint == "TREASURY_YIELDS":
                    col1, col2 = st.columns(2)
                    with col1:
                        default_gte = date.today() - timedelta(days=90)
                        gte_val = st.date_input("From Date", value=default_gte)
                        if gte_val:
                            endpoint_params["date.gte"] = gte_val.strftime("%Y-%m-%d")
                    with col2:
                        lte_val = st.date_input("To Date", value=date.today())
                        if lte_val:
                            endpoint_params["date.lte"] = lte_val.strftime("%Y-%m-%d")
                    endpoint_params["limit"] = st.slider("Limit", min_value=1, max_value=1000, value=100)

                elif endpoint == "FX_GROUPED":
                    fx_date = st.date_input("Date", value=date.today())
                    endpoint_params["date"] = fx_date.strftime("%Y-%m-%d")

                fetch = st.button("Fetch Data", type="primary", use_container_width=True)
                run_analysis = False

            else:
                source_key = "alphavantage"
                ep_config = ENDPOINT_MAP.get(endpoint, {})
                req = ep_config.get("required_params", [])
                opt = ep_config.get("optional_params", [])

                if endpoint in ("TIME_SERIES_DAILY", "OVERVIEW", "EARNINGS_ESTIMATES"):
                    symbol = st.text_input("Stock Symbol", placeholder="e.g. IBM").strip().upper()
                    endpoint_params["symbol"] = symbol
                    if endpoint == "TIME_SERIES_DAILY":
                        endpoint_params["outputsize"] = st.selectbox("Output Size", options=["compact", "full"], index=0)

                elif endpoint == "TOP_GAINERS_LOSERS":
                    pass

                elif endpoint == "ETF_PROFILE":
                    symbol = st.text_input("ETF Symbol", placeholder="e.g. QQQ").strip().upper()
                    endpoint_params["symbol"] = symbol

                elif endpoint == "NEWS_SENTIMENT":
                    tickers_val = st.text_input("Tickers", placeholder="e.g. AAPL,MSFT,IBM").strip()
                    if tickers_val:
                        endpoint_params["tickers"] = tickers_val
                    topics_val = st.selectbox("Topic", options=["", "blockchain", "earnings", "ipo", "mergers_and_acquisitions", "financial_markets", "economy_fiscal", "economy_monetary", "economy_macro", "energy_transportation", "finance", "life_sciences", "manufacturing", "real_estate", "retail_wholesale", "technology"], index=0)
                    if topics_val:
                        endpoint_params["topics"] = topics_val
                    sort_val = st.selectbox("Sort Order", options=["LATEST", "EARLIEST", "RELEVANCE"], index=0)
                    if sort_val:
                        endpoint_params["sort"] = sort_val
                    limit_val = st.slider("Limit", min_value=50, max_value=1000, value=50, step=50)
                    if limit_val:
                        endpoint_params["limit"] = limit_val

                elif endpoint == "TREASURY_YIELD":
                    endpoint_params["interval"] = st.selectbox("Interval", options=["daily", "weekly", "monthly"], index=2)
                    selected_maturity = st.selectbox("Maturity", options=["3month", "2year", "5year", "7year", "10year", "30year"], index=4)
                    endpoint_params["maturity"] = selected_maturity
                    if selected_maturity == "3month":
                        endpoint = "TREASURY_YIELD_3MONTH"
                    elif selected_maturity == "10year":
                        endpoint = "TREASURY_YIELD_10YEAR"

                elif endpoint in ("GOLD_SILVER_HISTORY",):
                    endpoint_params["symbol"] = st.selectbox("Metal", options=["GOLD", "XAU", "SILVER", "XAG"], index=0)
                    endpoint_params["interval"] = st.selectbox("Interval", options=["daily", "weekly", "monthly"], index=0)

                elif endpoint == "GOLD_SILVER_SPOT":
                    endpoint_params["symbol"] = st.selectbox("Metal", options=["GOLD", "XAU", "SILVER", "XAG"], index=0)

                elif endpoint == "CURRENCY_EXCHANGE_RATE":
                    endpoint_params["from_currency"] = st.text_input("From Currency", placeholder="e.g. USD", value="USD").strip().upper()
                    endpoint_params["to_currency"] = st.text_input("To Currency", placeholder="e.g. JPY", value="EUR").strip().upper()

                elif endpoint in ("TREASURY_YIELD_3MONTH", "TREASURY_YIELD_10YEAR"):
                    endpoint_params["interval"] = st.selectbox("Interval", options=["daily", "weekly", "monthly"], index=2)
                    endpoint_params["maturity"] = "3month" if "3MONTH" in endpoint else "10year"

                fetch = st.button("Fetch Data", type="primary", use_container_width=True)
                run_analysis = False
        else:
            fetch = False
            run_analysis = False
            st.info("Select an endpoint above to configure parameters.")

    st.divider()
    if is_ml_mode:
        if is_polygon_source:
            st.sidebar.info(
                "**Polygon (Primary)** — 5 calls/min. "
                "Open-close batch fetches each trading day individually. "
                "Non-replaced endpoints use Alpha Vantage."
            )
        else:
            remaining = get_remaining_calls()
            st.sidebar.info(
                f"Alpha Vantage free tier: **5 calls/min, {remaining}/{DAILY_LIMIT} calls/day**. "
                "Each submission triggers a live API call."
            )
    else:
        if source_key == "polygon":
            st.sidebar.info(
                "**Polygon** — 5 calls/min. "
                "Open-close batch fetches each trading day individually."
            )
        else:
            remaining = get_remaining_calls()
            st.sidebar.info(
                f"Alpha Vantage free tier: **5 calls/min, {remaining}/{DAILY_LIMIT} calls/day**. "
                "Each submission triggers a live API call."
            )

# ── Display helpers ────────────────────────────────────────────


def display_error(msg):
    st.error(msg, icon="🚨")


def display_time_series(df):
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.subheader("Close Price Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Close Price (USD)",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Latest Data Points")
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)


def display_gainers_losers(data):
    labels = {
        "top_gainers": "Top Gainers",
        "top_losers": "Top Losers",
        "most_actively_traded": "Most Actively Traded",
    }
    for key, label in labels.items():
        df = data.get(key, None)
        if df is not None and not df.empty:
            with st.expander(f"{label} ({len(df)})", expanded=True):
                st.dataframe(df, use_container_width=True, hide_index=True)


def display_overview(df):
    display_df = df.T.reset_index().rename(columns={"index": "Field", 0: "Value"})
    display_df["Value"] = display_df["Value"].astype(str)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def display_etf_profile(df):
    display_df = df.T.reset_index().rename(columns={"index": "Field", 0: "Value"})
    display_df["Value"] = display_df["Value"].astype(str)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def display_news_sentiment(df):
    st.subheader("Daily News Sentiment")
    if df is not None and not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["sentiment_score"], mode="lines+markers", name="Sentiment Score"
        ))
        fig.add_trace(go.Bar(
            x=df["Date"], y=df["article_count"], name="Article Count", yaxis="y2", opacity=0.3
        ))
        fig.update_layout(
            template="plotly_white",
            height=350,
            yaxis_title="Sentiment Score",
            yaxis2=dict(title="Article Count", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)


def display_earnings(df):
    st.subheader("Earnings Estimates")
    if df is not None and not df.empty:
        fig = go.Figure()
        if "eps_estimate_average" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["eps_estimate_average"], mode="lines+markers",
                name="Avg Estimate"
            ))
            if "eps_estimate_high" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["eps_estimate_high"], mode="lines",
                    name="High Estimate", line=dict(dash="dot", width=1)
                ))
            if "eps_estimate_low" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["Date"], y=df["eps_estimate_low"], mode="lines",
                    name="Low Estimate", line=dict(dash="dot", width=1)
                ))
        elif "estimatedEPS" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["estimatedEPS"], mode="lines+markers", name="Estimated EPS"
            ))
        fig.update_layout(
            template="plotly_white",
            height=350,
            yaxis_title="Estimated EPS",
            legend=dict(x=0, y=1.1, orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)


def display_treasury_yields(df_all, df_3m=None, df_10y=None):
    if df_all is not None:
        if df_all.empty:
            st.warning("Treasury yield data was returned but could not be parsed into the expected format.")
            return
        st.subheader("All Treasury Yields")
        if "maturity" in df_all.columns:
            fig = go.Figure()
            for maturity in df_all["maturity"].unique():
                subset = df_all[df_all["maturity"] == maturity]
                if subset.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=subset["Date"], y=subset["Value"], mode="lines",
                    name=str(maturity), line=dict(width=1)
                ))
            if fig.data:
                fig.update_layout(template="plotly_white", height=350,
                                  xaxis_title="Date", yaxis_title="Yield (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No maturity series could be plotted from the data.")
        st.dataframe(df_all.tail(20), use_container_width=True, hide_index=True)

    if df_3m is not None and not df_3m.empty:
        st.subheader("3-Month Treasury Yield")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_3m["Date"], y=df_3m["Value"], mode="lines", name="3M Yield"))
        fig.update_layout(template="plotly_white", height=250,
                          xaxis_title="Date", yaxis_title="Yield (%)")
        st.plotly_chart(fig, use_container_width=True)

    if df_10y is not None and not df_10y.empty:
        st.subheader("10-Year Treasury Yield")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_10y["Date"], y=df_10y["Value"], mode="lines", name="10Y Yield"))
        fig.update_layout(template="plotly_white", height=250,
                          xaxis_title="Date", yaxis_title="Yield (%)")
        st.plotly_chart(fig, use_container_width=True)


def display_fx_rates(df):
    st.subheader("FX Rates (Grouped)")
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Ticker"], y=df["Volume"], name="Transaction Volume",
            marker_color="#3498DB",
        ))
        fig.update_layout(template="plotly_white", height=400,
                          xaxis_title="Currency Pair", yaxis_title="Volume")
        st.plotly_chart(fig, use_container_width=True)


def display_open_close_meta(meta):
    if meta:
        st.caption(f"Fetched {meta.get('fetched', '?')} of {meta.get('total_dates', '?')} trading days")


def serve_downloads(dataframes, prefix):
    delivery = DeliveryAgent()
    csv_data = delivery.to_csv(dataframes)
    json_data = delivery.to_json(dataframes)

    if not csv_data and not json_data:
        return

    tab_csv, tab_json = st.tabs(["CSV", "JSON"])
    with tab_csv:
        for name, content in csv_data.items():
            st.download_button(
                label=f"Download {name}.csv",
                data=content,
                file_name=f"{prefix}_{name}.csv",
                mime="text/csv",
                key=f"csv_{prefix}_{name}",
            )
    with tab_json:
        for name, content in json_data.items():
            st.download_button(
                label=f"Download {name}.json",
                data=content,
                file_name=f"{prefix}_{name}.json",
                mime="application/json",
                key=f"json_{prefix}_{name}",
            )


# ── Fetch & Transform ──────────────────────────────────────────


def fetch_and_transform(endpoint_configs, transformer, source="alphavantage"):
    extractor = ExtractionAgent()
    raw_results = {}

    for cfg in endpoint_configs:
        ep = cfg["endpoint"]
        kwargs = {k: v for k, v in cfg.items() if k != "endpoint"}

        if source == "polygon" and ep == "OPEN_CLOSE":
            total_dates = None
            progress_bar = st.progress(0, text="Preparing open-close fetch...")
            status_text = st.empty()

            def make_callback(pbar, status):
                def cb(current, total, date_str):
                    nonlocal total_dates
                    if total_dates is None:
                        total_dates = total
                    pbar.progress(min((current + 1) / total, 1.0))
                    status.text(f"Fetching day {current + 1}/{total}: {date_str}")
                return cb

            kwargs["progress_callback"] = make_callback(progress_bar, status_text)
            raw_results[ep] = extractor.fetch(source, ep, **kwargs)
            if total_dates and total_dates > 0:
                progress_bar.progress(1.0)
                status_text.text(f"Completed — fetched open-close data")
        else:
            raw_results[ep] = extractor.fetch(source, ep, **kwargs)

    if any(isinstance(v, dict) and "error" in v for v in raw_results.values()):
        errors = {k: v["error"] for k, v in raw_results.items() if isinstance(v, dict) and "error" in v}
        return raw_results, errors

    transformed = {}
    errors = {}
    for ep, raw in raw_results.items():
        result = transformer.transform(ep, raw)
        if isinstance(result, dict) and "error" in result:
            errors[ep] = result["error"]
        transformed[ep] = result
    return transformed, errors


# ── MAIN: ML Analysis mode ─────────────────────────────────────

if is_ml_mode and run_analysis:
    if not symbol:
        display_error("Please enter a valid stock symbol.")
        st.stop()

    if is_polygon_source:
        module_info = POLYGON_ANALYSIS_MODULES[mode]
    else:
        module_info = ANALYSIS_MODULES[mode]

    endpoint_configs = []
    for cfg in module_info["endpoints"]:
        ep_cfg = dict(cfg)

        if is_polygon_source:
            if ep_cfg["endpoint"] in ("OPEN_CLOSE",):
                ep_cfg["stocksTicker"] = symbol
                ep_cfg["days_prior"] = ml_days_prior
                ep_cfg["end_date"] = ml_end_date.strftime("%Y-%m-%d")
            if ep_cfg["endpoint"] == "NEWS":
                ep_cfg["ticker"] = symbol
                ep_cfg["published_utc.gte"] = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")
                ep_cfg["published_utc.lte"] = date.today().strftime("%Y-%m-%d")
                ep_cfg["limit"] = 100
            if ep_cfg["endpoint"] == "TREASURY_YIELDS":
                ep_cfg["date.gte"] = (date.today() - timedelta(days=90)).strftime("%Y-%m-%d")
                ep_cfg["date.lte"] = date.today().strftime("%Y-%m-%d")
                ep_cfg["limit"] = 100
            if ep_cfg["endpoint"] == "EARNINGS_ESTIMATES":
                ep_cfg["symbol"] = symbol
        else:
            if ep_cfg["endpoint"] in ("TIME_SERIES_DAILY", "EARNINGS_ESTIMATES"):
                ep_cfg["symbol"] = symbol
            if ep_cfg["endpoint"] == "NEWS_SENTIMENT":
                ep_cfg["tickers"] = symbol

        endpoint_configs.append(ep_cfg)

    transformer = TransformationAgent()
    store = PITFeatureStore()
    engine = FeatureEngine()
    delivery = DeliveryAgent()

    ml_source_label = "Polygon" if is_polygon_source else "Alpha Vantage"
    logger = PipelineLogger(
        total_steps=6,
        title=f"{mode} Analysis  —  {symbol}  ({ml_source_label})",
    )

    # ── Step 1: Fetch ──
    logger.start_step(1, f"Fetching data from {ml_source_label}", f"{module_info['calls']} API call(s)")
    raw_results, errors = fetch_and_transform(
        endpoint_configs, transformer,
        source="polygon" if is_polygon_source else "alphavantage",
    )

    if errors:
        for ep, err in errors.items():
            logger.fail_step(1, f"{ep}: {err}")
        st.stop()
    logger.complete_step(1, f"Retrieved {len(raw_results)} endpoint(s)")

    if not is_polygon_source:
        track_api_calls(module_info["calls"] if isinstance(module_info["calls"], int) else 0)

    # ── Step 2: Transform ──
    logger.start_step(2, "Transforming raw data", "Parsing JSON → structured DataFrames")

    if is_polygon_source:
        price_df = raw_results.get("OPEN_CLOSE", {}).get("time_series_daily")
        sentiment_df = raw_results.get("NEWS", {}).get("news_sentiment")
        earnings_df = raw_results.get("EARNINGS_ESTIMATES", {}).get("earnings_estimates")
        yield_data = raw_results.get("TREASURY_YIELDS", {})
        yield_3m = yield_data.get("treasury_yield_3month")
        yield_10y = yield_data.get("treasury_yield_10year")
        gold_df = raw_results.get("GOLD_SILVER_HISTORY", {}).get("gold_silver_history")
    else:
        price_df = raw_results.get("TIME_SERIES_DAILY", {}).get("time_series_daily")
        sentiment_df = raw_results.get("NEWS_SENTIMENT", {}).get("news_sentiment")
        earnings_df = raw_results.get("EARNINGS_ESTIMATES", {}).get("earnings_estimates")
        yield_3m = raw_results.get("TREASURY_YIELD_3MONTH", {}).get("treasury_yield_3month")
        yield_10y = raw_results.get("TREASURY_YIELD_10YEAR", {}).get("treasury_yield_10year")
        gold_df = raw_results.get("GOLD_SILVER_HISTORY", {}).get("gold_silver_history")

    if price_df is None or price_df.empty:
        logger.fail_step(2, "No price data returned")
        st.stop()
    n_sources = sum([
        price_df is not None,
        sentiment_df is not None and not sentiment_df.empty,
        earnings_df is not None and not earnings_df.empty,
        yield_3m is not None and not yield_3m.empty,
        yield_10y is not None and not yield_10y.empty,
        gold_df is not None and not gold_df.empty,
    ])
    logger.complete_step(2, f"{n_sources} dataset(s) parsed")

    # ── Step 3: Feature Engineering ──
    logger.start_step(3, "Building feature matrix", "Sentiment, earnings & macro features")
    feature_df, _ = engine.build_all(
        price_df=price_df,
        sentiment_df=sentiment_df,
        earnings_df=earnings_df,
        yield_3m_df=yield_3m,
        yield_10y_df=yield_10y,
        gold_df=gold_df,
    )

    if feature_df is None or feature_df.empty:
        logger.fail_step(3, "No valid features produced")
        st.stop()
    n_feat = len([c for c in feature_df.columns if c != "Date"])
    logger.complete_step(3, f"{n_feat} features × {feature_df.shape[0]} samples")
    logger.log_table("Feature Snapshot (last 5 rows)", feature_df.tail(5))

    # ── Step 4: Leakage Check ──
    logger.start_step(4, "Running leakage check", "Validating temporal alignment")
    leakage_warnings = engine.run_leakage_check(feature_df, target_df=price_df)
    if leakage_warnings:
        logger.log_info(f"⚠️  {len(leakage_warnings)} warning(s): {'; '.join(leakage_warnings)}")
        logger.complete_step(4, f"{len(leakage_warnings)} warning(s)")
    else:
        logger.complete_step(4, "No leakage detected  ✓")

    # ── Step 5: Training ──
    if mode == "Event Risk":
        model = VolatilityShockModel()
        algo = "XGBoost"
    elif mode == "Earnings Momentum":
        model = PEADModel()
        algo = "LightGBM"
    else:
        model = MacroRegimeModel()
        algo = "Random Forest"

    logger.start_step(5, f"Training {algo} model", f"{mode}")
    train_result = model.train(feature_df, price_df)

    if "error" in train_result:
        logger.fail_step(5, train_result["error"])
        st.stop()

    if "accuracy" in train_result:
        logger.log_metric("Test Accuracy", f"{train_result['accuracy']:.2%}")
    elif "mae" in train_result:
        logger.log_metric("Test MAE", f"{train_result['mae']:.4f}")
        logger.log_metric("Test R²", f"{train_result['r2']:.4f}")
    logger.complete_step(5, "Model trained successfully")

    # ── Step 6: Inference ──
    logger.start_step(6, "Running inference", "Latest feature vector → prediction")
    latest = feature_df.iloc[-1:].iloc[0].to_dict()
    model_metadata = model.metadata

    if mode == "Event Risk":
        probs = model.predict(latest)
        dominant = max(probs, key=probs.get)
        logger.log_metric(
            "Predicted Regime",
            dominant.capitalize(),
            f"{probs[dominant] * 100:.1f}% confidence",
        )
    elif mode == "Earnings Momentum":
        pred = model.predict(latest)
        logger.log_metric(
            "Projected Drift",
            f"{pred['expected_drift_pct']:.2f}%",
        )
    else:
        regime_result = model.predict(latest)
        logger.log_metric(
            "Predicted Regime",
            regime_result["regime"].upper(),
            f"{regime_result['confidence'] * 100:.1f}% confidence",
        )

    logger.complete_step(6, "Prediction ready  ✓")
    logger.close()

    # ── Results display ──
    st.subheader(f"Results — {mode}")
    st.divider()

    display_time_series(price_df)

    if is_polygon_source:
        meta = raw_results.get("OPEN_CLOSE", {}).get("_meta")
        if meta:
            display_open_close_meta(meta)

    if sentiment_df is not None and not sentiment_df.empty:
        with st.expander("News Sentiment Data", expanded=False):
            display_news_sentiment(sentiment_df)

    if earnings_df is not None and not earnings_df.empty:
        with st.expander("Earnings Estimates Data", expanded=False):
            display_earnings(earnings_df)

    if leakage_warnings:
        with st.expander("Leakage Check Warnings", expanded=False):
            for w in leakage_warnings:
                st.warning(w)

    if mode == "Event Risk":
        delivery.display_volatility_gauge(probs)
        with st.expander("Model Details", expanded=False):
            st.json(model_metadata)
            st.metric("Test Accuracy", f"{train_result['accuracy']:.2%}")
            delivery.display_feature_importance(model)

    elif mode == "Earnings Momentum":
        delivery.display_pead_chart(pred)
        with st.expander("Model Details", expanded=False):
            st.json(model_metadata)
            st.metric("Test MAE", f"{train_result['mae']:.4f}")
            st.metric("Test R²", f"{train_result['r2']:.4f}")
            delivery.display_feature_importance(model)

    elif mode == "Macro Regime":
        indicators = {}
        if "yield_spread_10y_3m" in latest:
            indicators["yield_spread_10y_3m"] = latest["yield_spread_10y_3m"]
        if "gold_momentum_20d" in latest:
            indicators["gold_momentum_20d"] = latest["gold_momentum_20d"]
        delivery.display_macro_dashboard(indicators, regime_result)
        with st.expander("Model Details", expanded=False):
            st.json(model_metadata)
            st.metric("Test Accuracy", f"{train_result['accuracy']:.2%}")
            delivery.display_feature_importance(model)

    st.divider()
    st.subheader("Downloads")
    download_data = {"features": feature_df}
    if price_df is not None:
        download_data["prices"] = price_df
    if sentiment_df is not None:
        download_data["sentiment"] = sentiment_df
    if earnings_df is not None:
        download_data["earnings"] = earnings_df
    serve_downloads(download_data, f"{mode}_{symbol}")

# ── MAIN: Single Endpoint mode ─────────────────────────────────

elif fetch:
    if source_key == "polygon":
        poly_cfg = POLYGON_ENDPOINT_MAP.get(endpoint, {})
        required = poly_cfg.get("required_params", [])
    else:
        required = req

    missing_required = [p for p in required if p not in endpoint_params or not endpoint_params.get(p)]
    if missing_required:
        display_error(f"Missing required parameters: {', '.join(missing_required)}")
        st.stop()

    if source_key == "alphavantage":
        track_api_calls(1)

    extractor = ExtractionAgent()
    transformer = TransformationAgent()

    source_label = "Polygon" if source_key == "polygon" else "Alpha Vantage"
    with st.spinner(f"Fetching data from {source_label}..."):
        raw = extractor.fetch(source_key, endpoint, **endpoint_params)

    with st.spinner("Transforming data..."):
        transformed = transformer.transform(endpoint, raw)

    if "error" in transformed:
        display_error(transformed["error"])
        st.stop()

    display_name = ENDPOINT_DISPLAY_NAMES.get(endpoint, endpoint)
    st.subheader(f"Results — {display_name}")
    st.divider()

    if endpoint == "OPEN_CLOSE":
        df = transformed.get("time_series_daily")
        if df is not None:
            display_time_series(df)
        meta = transformed.get("_meta")
        if meta:
            display_open_close_meta(meta)

    elif endpoint == "NEWS":
        df = transformed.get("news_sentiment")
        if df is not None:
            display_news_sentiment(df)

    elif endpoint == "TREASURY_YIELDS":
        df_all = transformed.get("all_yields")
        df_3m = transformed.get("treasury_yield_3month")
        df_10y = transformed.get("treasury_yield_10year")
        display_treasury_yields(df_all, df_3m, df_10y)

    elif endpoint == "FX_GROUPED":
        df = transformed.get("currency_exchange")
        if df is not None:
            display_fx_rates(df)

    elif endpoint == "TIME_SERIES_DAILY":
        df = transformed.get("time_series_daily")
        if df is not None:
            display_time_series(df)

    elif endpoint == "TOP_GAINERS_LOSERS":
        display_gainers_losers(transformed)

    elif endpoint == "OVERVIEW":
        df = transformed.get("overview")
        if df is not None:
            display_overview(df)

    elif endpoint == "ETF_PROFILE":
        df = transformed.get("etf_profile")
        if df is not None:
            display_etf_profile(df)
        sectors = transformed.get("sectors")
        if sectors is not None and not sectors.empty:
            with st.expander("Sector Allocation", expanded=True):
                st.dataframe(sectors, use_container_width=True, hide_index=True)
        holdings = transformed.get("holdings")
        if holdings is not None and not holdings.empty:
            with st.expander("Top Holdings", expanded=True):
                st.dataframe(holdings, use_container_width=True, hide_index=True)

    elif endpoint == "NEWS_SENTIMENT":
        df = transformed.get("news_sentiment")
        if df is not None:
            display_news_sentiment(df)

    elif endpoint == "EARNINGS_ESTIMATES":
        df = transformed.get("earnings_estimates")
        if df is not None:
            display_earnings(df)

    elif endpoint.startswith("TREASURY_YIELD"):
        for key in ("treasury_yield_3month", "treasury_yield_10year"):
            df = transformed.get(key)
            if df is not None and not df.empty:
                st.subheader(key.replace("_", " ").title())
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["Date"], y=df["Value"], mode="lines", name="Yield"))
                fig.update_layout(template="plotly_white", height=300,
                                  xaxis_title="Date", yaxis_title="Yield (%)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

    elif endpoint == "GOLD_SILVER_HISTORY":
        df = transformed.get("gold_silver_history")
        if df is not None:
            st.subheader("Gold/Silver History")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
            fig.update_layout(template="plotly_white", height=350,
                              xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.tail(10), use_container_width=True, hide_index=True)

    elif endpoint == "GOLD_SILVER_SPOT":
        df = transformed.get("gold_silver_spot")
        if df is not None and not df.empty:
            st.subheader("Gold/Silver Spot Price")
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif endpoint == "CURRENCY_EXCHANGE_RATE":
        df = transformed.get("currency_exchange")
        if df is not None and not df.empty:
            st.subheader("Currency Exchange Rate")
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Downloads")
    if "error" not in transformed:
        serve_downloads(transformed, endpoint)
