import streamlit as st
import plotly.graph_objects as go

from src.extraction import ExtractionAgent, ENDPOINT_MAP
from src.transformation import TransformationAgent
from src.delivery import DeliveryAgent

st.set_page_config(page_title="Alpha Vantage ETL", layout="wide")

st.title("Alpha Vantage ETL Pipeline")
st.caption("Extracts financial data via Alpha Vantage, transforms it, and displays results.")

with st.sidebar:
    st.header("Configuration")

    endpoint = st.selectbox(
        "Endpoint",
        options=list(ENDPOINT_MAP.keys()),
        index=0,
    )

    show_symbol = endpoint != "TOP_GAINERS_LOSERS"
    symbol = ""
    if show_symbol:
        symbol = st.text_input("Symbol", placeholder="e.g. IBM").strip().upper()

    show_outputsize = endpoint == "TIME_SERIES_DAILY"
    outputsize = "compact"
    if show_outputsize:
        outputsize = st.selectbox("Output Size", options=["compact", "full"], index=0)

    fetch = st.button("Fetch Data", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.info(
    "Alpha Vantage free tier: **5 calls/min, 500 calls/day**. "
    "Each submission triggers a live API call."
)


def display_error(msg):
    st.error(msg, icon="🚨")


def display_time_series(df):
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.subheader("Close Price Over Time")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close")
        )
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


def serve_downloads(dataframes, endpoint):
    delivery = DeliveryAgent()
    csv_data = delivery.to_csv(dataframes)
    json_data = delivery.to_json(dataframes)

    tab_csv, tab_json = st.tabs(["CSV", "JSON"])

    with tab_csv:
        for name, content in csv_data.items():
            st.download_button(
                label=f"Download {name}.csv",
                data=content,
                file_name=f"{endpoint}_{name}.csv",
                mime="text/csv",
                key=f"csv_{name}",
            )

    with tab_json:
        for name, content in json_data.items():
            st.download_button(
                label=f"Download {name}.json",
                data=content,
                file_name=f"{endpoint}_{name}.json",
                mime="application/json",
                key=f"json_{name}",
            )


if fetch:
    if show_symbol and not symbol:
        display_error("Please enter a valid stock symbol.")
        st.stop()

    extractor = ExtractionAgent()
    transformer = TransformationAgent()

    with st.spinner("Fetching data from Alpha Vantage..."):
        raw = extractor.fetch(endpoint, symbol=symbol, outputsize=outputsize)

    with st.spinner("Transforming data..."):
        transformed = transformer.transform(endpoint, raw)

    if "error" in transformed:
        display_error(transformed["error"])
        st.stop()

    st.subheader(f"Results — {endpoint}")
    st.divider()

    if endpoint == "TIME_SERIES_DAILY":
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

    st.divider()
    st.subheader("Downloads")
    if "error" not in transformed:
        serve_downloads(transformed, endpoint)
