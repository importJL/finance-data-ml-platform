import io
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st


class DeliveryAgent:
    def to_csv(self, dataframes):
        result = {}
        for name, df in dataframes.items():
            if not hasattr(df, "to_csv") or (hasattr(df, "empty") and df.empty):
                continue
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            result[name] = buf.getvalue()
        return result

    def to_json(self, dataframes):
        result = {}
        for name, df in dataframes.items():
            if not hasattr(df, "to_json") or (hasattr(df, "empty") and df.empty):
                continue
            if isinstance(df, dict):
                payload = df
            else:
                payload = json.loads(df.to_json(orient="records", date_format="iso"))
            result[name] = json.dumps(payload, indent=2).encode("utf-8")
        return result

    def display_volatility_gauge(self, probs):
        labels = ["Low", "Medium", "High"]
        values = [probs.get("low", 0) * 100, probs.get("medium", 0) * 100, probs.get("high", 0) * 100]
        colors = ["#2ECC71", "#F39C12", "#E74C3C"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        ))
        fig.update_layout(
            title="Volatility Shock Probability",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            template="plotly_white",
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        dominant = max(probs, key=probs.get)
        dominant_label = dominant.capitalize()
        dominant_color = {"low": "green", "medium": "orange", "high": "red"}.get(dominant, "gray")
        st.markdown(
            f"**Predicted Regime:** :{dominant_color}[{dominant_label}] "
            f"({probs[dominant] * 100:.1f}% confidence)"
        )

    def display_pead_chart(self, prediction, historical_returns=None):
        drift = prediction.get("expected_drift_pct", 0)
        color = "#2ECC71" if drift > 0 else "#E74C3C"

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Projected PEAD Return"],
            y=[drift],
            marker_color=color,
            text=f"{drift:.2f}%",
            textposition="auto",
        ))
        fig.update_layout(
            title="Post-Earnings Announcement Drift (5-Day Projection)",
            yaxis_title="Expected Return (%)",
            template="plotly_white",
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        if historical_returns is not None and not historical_returns.empty:
            st.caption(f"Based on {len(historical_returns)} training samples. | MAE: {prediction.get('_mae', 'N/A')}")

    def display_macro_dashboard(self, indicators, regime_result):
        regime = regime_result.get("regime", "unknown")
        confidence = regime_result.get("confidence", 0) * 100
        colors = {"expansion": "#2ECC71", "neutral": "#F39C12", "contraction": "#E74C3C"}

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Regime", regime.upper(), delta=f"{confidence:.1f}% confidence",
                      delta_color="off")
        with col2:
            spread = indicators.get("yield_spread_10y_3m", None)
            if spread is not None:
                st.metric("Yield Spread (10Y-3M)", f"{spread:.2f}%")
            else:
                st.metric("Yield Spread", "N/A")
        with col3:
            gold_mom = indicators.get("gold_momentum_20d", None)
            if gold_mom is not None:
                st.metric("Gold Momentum (20D)", f"{gold_mom*100:.2f}%")
            else:
                st.metric("Gold Momentum", "N/A")

        regime_color = colors.get(regime, "gray")
        st.markdown(
            f"<div style='padding:1rem; border-radius:0.5rem; background:{regime_color}20; "
            f"border-left:4px solid {regime_color};'>"
            f"<span style='font-size:1.2rem; font-weight:600; color:{regime_color};'>"
            f"{regime.upper()} REGIME</span> — Model confidence: {confidence:.1f}%"
            f"</div>",
            unsafe_allow_html=True,
        )

    def display_feature_importance(self, model, n=10):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            names = model.feature_names
            if names and len(names) == len(importances):
                sorted_idx = np.argsort(importances)[::-1][:n]
                top_names = [names[i] for i in sorted_idx]
                top_vals = [importances[i] for i in sorted_idx]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_vals[::-1],
                    y=top_names[::-1],
                    orientation="h",
                    marker_color="#3498DB",
                ))
                fig.update_layout(
                    title=f"Top {n} Feature Importances",
                    xaxis_title="Importance",
                    template="plotly_white",
                    height=50 * min(n, len(top_names)) + 50,
                )
                st.plotly_chart(fig, use_container_width=True)
