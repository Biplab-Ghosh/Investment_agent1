"""
LangGraph agent state schema.

All nodes read from and write to InvestmentAnalysisState.
Using TypedDict so LangGraph can serialise/deserialise cleanly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd
from typing_extensions import TypedDict


# ── Sub-state types ────────────────────────────────────────────────────────────

class DCFAssumptions(TypedDict, total=False):
    wacc: float
    terminal_growth_rate: float
    revenue_growth_rates: list[float]    # per forecast year
    operating_margin_target: float
    capex_percent_revenue: float
    nwc_percent_revenue: float
    tax_rate: float
    model_type: str                       # "2-stage" | "3-stage"
    lifecycle_stage: str
    forecast_years: int
    # Rationale strings for each assumption (for LLM explanation)
    rationale: dict[str, str]


class DCFResults(TypedDict, total=False):
    enterprise_value: float
    equity_value: float
    intrinsic_value_per_share: float
    current_price: float
    upside_downside_pct: float
    margin_of_safety: float
    pv_fcf_stages: list[float]
    pv_terminal_value: float
    terminal_value: float
    wacc_used: float
    terminal_growth_used: float
    validation: dict[str, Any]           # sanity-check warnings from valuation_checks
    model_type_used: str
    revenue_growth_rates_used: list[float]


class MoatScore(TypedDict, total=False):
    intangible_assets: float       # 0-10
    switching_costs: float
    network_effects: float
    cost_advantages: float
    efficient_scale: float
    total_score: float             # 0-50
    rating: str                    # "Wide" | "Narrow" | "None"
    qualitative_summary: str
    roic_5yr_avg: float
    sustainability_assessment: str


class SensitivityResults(TypedDict, total=False):
    wacc_vs_tgr: Any               # 2D DataFrame: WACC rows × terminal growth cols
    scenario_base: float
    scenario_bull: float
    scenario_bear: float
    monte_carlo_percentiles: dict[str, float]   # "p10", "p25", "p50", "p75", "p90"
    monte_carlo_mean: float


# ── Master agent state ─────────────────────────────────────────────────────────

class InvestmentAnalysisState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────────
    ticker_symbols: list[str]
    analysis_date: str                    # ISO date string
    user_overrides: dict[str, Any]        # user-supplied assumption overrides
    manual_data_paths: dict[str, str]     # ticker → path to manual financials CSV

    # ── Raw data ────────────────────────────────────────────────────────────────
    financial_data: dict[str, Any]        # ticker → company data dict from DataManager
    market_data: dict[str, Any]           # macro snapshot from FRED

    # ── Analysis outputs ────────────────────────────────────────────────────────
    company_profiles: dict[str, dict]     # ticker → {stage, industry_class, ...}
    dcf_assumptions: dict[str, DCFAssumptions]
    dcf_results: dict[str, DCFResults]
    moat_analysis: dict[str, MoatScore]
    sensitivity_results: dict[str, SensitivityResults]

    # ── Human-in-the-loop ───────────────────────────────────────────────────────
    conversation_history: list[dict[str, str]]
    pending_user_approval: Optional[str]  # name of the node waiting for approval
    assumption_presentation: Optional[str]  # formatted text shown to user

    # ── Outputs ─────────────────────────────────────────────────────────────────
    final_reports: dict[str, str]         # ticker → markdown report
    visualizations: dict[str, Any]        # ticker → figure objects

    # ── Control flow ────────────────────────────────────────────────────────────
    current_ticker: Optional[str]         # ticker currently being processed
    errors: list[str]                     # non-fatal errors accumulated during run
    status: str                           # "running" | "awaiting_input" | "complete"
