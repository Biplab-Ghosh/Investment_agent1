"""
Visualization module — all charts returned as Plotly Figure objects.

Charts:
  - DCF waterfall (value buildup)
  - Sensitivity heatmap (WACC × TGR)
  - Tornado diagram (key driver impacts)
  - Moat radar chart (5 dimensions)
  - Monte Carlo histogram
  - Time series (ROIC, margins, revenue growth)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False
    logger = logging.getLogger(__name__)
    logger.warning("plotly not installed; charts will be unavailable")

logger = logging.getLogger(__name__)


def _require_plotly() -> None:
    if not _PLOTLY_OK:
        raise ImportError("plotly is required for visualizations. Run: pip install plotly kaleido")


# ── DCF Waterfall ─────────────────────────────────────────────────────────────

def dcf_waterfall(
    dcf_results: dict[str, Any],
    ticker: str = "",
    currency: str = "$",
) -> Any:
    """
    Show how PV of FCF stages + PV of terminal value combine to give
    enterprise value, then bridge to equity value and per-share price.
    """
    _require_plotly()

    pv_fcfs = dcf_results.get("pv_fcf_stages", [])
    pv_tv = dcf_results.get("pv_terminal_value", 0)
    cash = 0  # not stored in results; shown as bridge if desired
    total_debt = 0
    ev = dcf_results.get("enterprise_value", 0)
    equity_val = dcf_results.get("equity_value", 0)
    per_share = dcf_results.get("intrinsic_value_per_share", 0)
    current_price = dcf_results.get("current_price", 0)

    labels = [f"PV FCF Y{i+1}" for i in range(len(pv_fcfs))] + ["PV Terminal Value", "Intrinsic Value/Share"]
    values = list(pv_fcfs) + [pv_tv, per_share]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * (len(pv_fcfs) + 1) + ["total"],
        x=labels,
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#3498db"}},
        text=[f"{currency}{v:,.0f}" if abs(v) > 1e6 else f"{currency}{v:,.2f}" for v in values],
        textposition="outside",
    ))

    if current_price > 0:
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Current Price: {currency}{current_price:.2f}",
        )

    fig.update_layout(
        title=f"{ticker} DCF Value Buildup" if ticker else "DCF Value Buildup",
        yaxis_title=f"Value ({currency})",
        showlegend=False,
        template="plotly_dark",
        height=500,
    )
    return fig


# ── Sensitivity Heatmap ───────────────────────────────────────────────────────

def sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    base_intrinsic: float,
    current_price: float,
    ticker: str = "",
    currency: str = "$",
) -> Any:
    """Heatmap of intrinsic value vs WACC (rows) and terminal growth rate (columns)."""
    _require_plotly()

    z = sensitivity_df.values.astype(float)
    x_labels = list(sensitivity_df.columns)
    y_labels = list(sensitivity_df.index)

    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            if np.isnan(val):
                text = "N/A"
                font_color = "white"
            else:
                text = f"{currency}{val:.0f}"
                font_color = "white" if abs(val - base_intrinsic) / max(base_intrinsic, 1) > 0.15 else "black"
            annotations.append(dict(x=x_labels[j], y=y_labels[i], text=text, showarrow=False, font=dict(color=font_color)))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale="RdYlGn",
        zmid=current_price if current_price > 0 else base_intrinsic,
        colorbar=dict(title=f"Intrinsic ({currency})"),
    ))
    fig.update_layout(
        title=f"{ticker} Sensitivity: Intrinsic Value vs WACC & Terminal Growth" if ticker else "Sensitivity Table",
        xaxis_title="Terminal Growth Rate",
        yaxis_title="WACC",
        annotations=annotations,
        template="plotly_dark",
        height=450,
    )
    return fig


# ── Tornado Diagram ───────────────────────────────────────────────────────────

def tornado_chart(
    tornado_data: list[dict[str, Any]],
    base_intrinsic: float,
    ticker: str = "",
    currency: str = "$",
) -> Any:
    """Horizontal bar chart showing value impact of each assumption ±1σ."""
    _require_plotly()

    if not tornado_data:
        return go.Figure().update_layout(title="No tornado data")

    variables = [d["variable"] for d in tornado_data]
    low_vals = [d["low_value"] - base_intrinsic for d in tornado_data]
    high_vals = [d["high_value"] - base_intrinsic for d in tornado_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Downside",
        y=variables,
        x=low_vals,
        orientation="h",
        marker_color="#e74c3c",
    ))
    fig.add_trace(go.Bar(
        name="Upside",
        y=variables,
        x=high_vals,
        orientation="h",
        marker_color="#2ecc71",
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"{ticker} Sensitivity Tornado" if ticker else "Sensitivity Tornado",
        xaxis_title=f"Change in Intrinsic Value ({currency})",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=400,
    )
    fig.add_vline(x=0, line_dash="solid", line_color="white")
    return fig


# ── Moat Radar Chart ──────────────────────────────────────────────────────────

def moat_radar(
    moat_results: dict[str, Any],
    ticker: str = "",
) -> Any:
    """Radar chart showing scores across 5 moat dimensions."""
    _require_plotly()

    dimensions = ["Intangible Assets", "Switching Costs", "Network Effects",
                  "Cost Advantages", "Efficient Scale"]
    scores = [
        moat_results.get("intangible_assets", 0),
        moat_results.get("switching_costs", 0),
        moat_results.get("network_effects", 0),
        moat_results.get("cost_advantages", 0),
        moat_results.get("efficient_scale", 0),
    ]
    scores_closed = scores + [scores[0]]
    dimensions_closed = dimensions + [dimensions[0]]

    rating = moat_results.get("rating", "Unknown")
    total = moat_results.get("total_score", 0)

    color_map = {"Wide": "#2ecc71", "Narrow": "#f39c12", "None": "#e74c3c"}
    color = color_map.get(rating, "#3498db")

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=dimensions_closed,
        fill="toself",
        fillcolor=color,
        opacity=0.4,
        line=dict(color=color, width=2),
        name=f"{rating} Moat (Score: {total}/50)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[10] * len(dimensions_closed),
        theta=dimensions_closed,
        fill="toself",
        fillcolor="rgba(255,255,255,0.05)",
        line=dict(color="rgba(255,255,255,0.2)", width=1),
        name="Maximum (10)",
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        title=f"{ticker} Economic Moat — {rating} ({total}/50)" if ticker else f"Economic Moat — {rating}",
        showlegend=True,
        template="plotly_dark",
        height=500,
    )
    return fig


# ── Monte Carlo Histogram ─────────────────────────────────────────────────────

def monte_carlo_histogram(
    mc_results: dict[str, Any],
    current_price: float,
    base_intrinsic: float,
    ticker: str = "",
    currency: str = "$",
) -> Any:
    """Distribution of Monte Carlo simulated intrinsic values."""
    _require_plotly()

    raw = mc_results.get("raw_values", [])
    if not raw:
        return go.Figure().update_layout(title="No Monte Carlo data")

    pcts = mc_results.get("monte_carlo_percentiles", {})

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=raw,
        nbinsx=80,
        marker_color="#3498db",
        opacity=0.75,
        name="Simulated Values",
    ))

    for label, key, color in [
        ("10th pct", "p10", "#e74c3c"),
        ("50th pct", "p50", "#f39c12"),
        ("90th pct", "p90", "#2ecc71"),
    ]:
        if key in pcts:
            fig.add_vline(x=pcts[key], line_dash="dash", line_color=color,
                          annotation_text=f"{label}: {currency}{pcts[key]:.0f}")

    if current_price > 0:
        fig.add_vline(x=current_price, line_dash="solid", line_color="white",
                      annotation_text=f"Market: {currency}{current_price:.0f}")

    fig.update_layout(
        title=f"{ticker} Monte Carlo Intrinsic Value Distribution" if ticker else "Monte Carlo Distribution",
        xaxis_title=f"Intrinsic Value per Share ({currency})",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=450,
    )
    return fig


# ── Time Series Charts ────────────────────────────────────────────────────────

def historical_metrics_chart(
    company_data: dict[str, Any],
    dcf_metrics: dict[str, Any],
    ticker: str = "",
) -> Any:
    """Multi-panel chart showing revenue, margins, and FCF history."""
    _require_plotly()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Revenue ($B)",
            "Operating Margin (%)",
            "Free Cash Flow ($B)",
            "ROIC (latest)",
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "indicator"}],
        ],
    )

    revenues = dcf_metrics.get("revenues")
    if revenues is not None and not revenues.empty:
        years = [str(d.year) for d in revenues.index]
        fig.add_trace(go.Bar(x=years, y=(revenues / 1e9).round(2).tolist(), name="Revenue", marker_color="#3498db"), row=1, col=1)

    margins = dcf_metrics.get("operating_margins")
    if margins:
        rev = dcf_metrics.get("revenues")
        if rev is not None:
            years_m = [str(d.year) for d in rev.index[-len(margins):]]
            fig.add_trace(go.Scatter(x=years_m, y=[m * 100 for m in margins], mode="lines+markers",
                                     name="Op Margin", marker_color="#2ecc71"), row=1, col=2)

    fcf = dcf_metrics.get("historical_fcf")
    if fcf is not None and not fcf.empty:
        years_f = [str(d.year) for d in fcf.index]
        fig.add_trace(go.Bar(x=years_f, y=(fcf / 1e9).round(2).tolist(), name="FCF",
                             marker_color="#9b59b6"), row=2, col=1)

    roic = dcf_metrics.get("roic_latest")
    if roic:
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=roic * 100,
                title={"text": "ROIC (latest)"},
                number={"suffix": "%"},
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=f"{ticker} Historical Financial Performance" if ticker else "Historical Performance",
        template="plotly_dark",
        height=600,
        showlegend=False,
    )
    return fig


# ── Scenario Comparison Bar ───────────────────────────────────────────────────

def scenario_comparison_chart(
    scenario_results: dict[str, Any],
    current_price: float,
    ticker: str = "",
    currency: str = "$",
) -> Any:
    """Bar chart comparing Bear / Base / Bull intrinsic values against market price."""
    _require_plotly()

    names = []
    values = []
    colors = []
    for scenario, color in [("bear", "#e74c3c"), ("base", "#3498db"), ("bull", "#2ecc71")]:
        if scenario in scenario_results and "intrinsic_value_per_share" in scenario_results[scenario]:
            names.append(scenario.title())
            values.append(scenario_results[scenario]["intrinsic_value_per_share"])
            colors.append(color)

    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors, text=[f"{currency}{v:.2f}" for v in values], textposition="outside"))

    if current_price > 0:
        fig.add_hline(y=current_price, line_dash="dash", line_color="white",
                      annotation_text=f"Current: {currency}{current_price:.2f}")

    fig.update_layout(
        title=f"{ticker} Scenario Analysis" if ticker else "Scenario Analysis",
        yaxis_title=f"Intrinsic Value ({currency})",
        template="plotly_dark",
        height=400,
    )
    return fig
