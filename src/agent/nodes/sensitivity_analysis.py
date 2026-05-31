"""
Node 5 — Sensitivity Analysis

Runs WACC × TGR data tables, scenario analysis, Monte Carlo simulation,
and tornado chart data for each ticker.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import InvestmentAnalysisState, SensitivityResults
from src.analysis.dcf_engine import DCFEngine
from src.analysis.sensitivity_engine import SensitivityEngine

logger = logging.getLogger(__name__)

_dcf_engine = DCFEngine()
_sens_engine = SensitivityEngine()


def sensitivity_analysis_node(state: InvestmentAnalysisState) -> dict[str, Any]:
    """
    Generate full sensitivity analysis for each ticker.
    Populates state.sensitivity_results.
    """
    financial_data: dict[str, Any] = state.get("financial_data", {})
    market_data: dict[str, Any] = state.get("market_data", {})
    dcf_assumptions: dict[str, Any] = state.get("dcf_assumptions", {})
    dcf_results: dict[str, Any] = state.get("dcf_results", {})
    errors: list[str] = list(state.get("errors", []))

    sensitivity_results: dict[str, SensitivityResults] = {}

    from config.settings import MONTE_CARLO_SIMULATIONS

    for ticker, company_data in financial_data.items():
        if ticker not in dcf_assumptions:
            continue

        assumptions = dcf_assumptions[ticker]
        base_iv = dcf_results.get(ticker, {}).get("intrinsic_value_per_share", 0)
        result: SensitivityResults = {}

        # ── 2D WACC × Terminal Growth table ──────────────────────────────────
        try:
            table = _sens_engine.wacc_vs_tgr_table(
                dcf_engine=_dcf_engine,
                company_data=company_data,
                base_assumptions=assumptions,
                market_data=market_data,
            )
            result["wacc_vs_tgr"] = table
            result["scenario_base"] = base_iv
            logger.info("Sensitivity table computed for %s (%dx%d)", ticker, *table.shape)
        except Exception as exc:
            errors.append(f"Sensitivity table failed for {ticker}: {exc}")
            logger.error("Sensitivity table error for %s: %s", ticker, exc)

        # ── Scenario analysis ─────────────────────────────────────────────────
        try:
            scenarios = _sens_engine.scenario_analysis(
                dcf_engine=_dcf_engine,
                company_data=company_data,
                base_assumptions=assumptions,
                market_data=market_data,
            )
            result["scenario_bull"] = scenarios.get("bull", {}).get("intrinsic_value_per_share", 0)
            result["scenario_base"] = scenarios.get("base", {}).get("intrinsic_value_per_share", base_iv)
            result["scenario_bear"] = scenarios.get("bear", {}).get("intrinsic_value_per_share", 0)
            result["scenarios"] = scenarios
            logger.info("Scenarios for %s: bear=$%.2f, base=$%.2f, bull=$%.2f",
                        ticker, result["scenario_bear"], result["scenario_base"], result["scenario_bull"])
        except Exception as exc:
            errors.append(f"Scenario analysis failed for {ticker}: {exc}")

        # ── Monte Carlo ───────────────────────────────────────────────────────
        try:
            mc = _sens_engine.monte_carlo_simulation(
                dcf_engine=_dcf_engine,
                company_data=company_data,
                base_assumptions=assumptions,
                market_data=market_data,
                n_simulations=MONTE_CARLO_SIMULATIONS,
            )
            result["monte_carlo_percentiles"] = mc.get("monte_carlo_percentiles", {})
            result["monte_carlo_mean"] = mc.get("mean", 0)
            result["monte_carlo_raw"] = mc  # full results including raw values for histogram
            logger.info(
                "Monte Carlo for %s: P50=$%.2f, mean=$%.2f (%d valid sims)",
                ticker,
                mc.get("monte_carlo_percentiles", {}).get("p50", 0),
                mc.get("mean", 0),
                mc.get("n_valid", 0),
            )
        except Exception as exc:
            errors.append(f"Monte Carlo failed for {ticker}: {exc}")

        # ── Tornado data ──────────────────────────────────────────────────────
        try:
            tornado = _sens_engine.tornado_data(
                dcf_engine=_dcf_engine,
                company_data=company_data,
                base_assumptions=assumptions,
                market_data=market_data,
            )
            result["tornado_data"] = tornado
        except Exception as exc:
            errors.append(f"Tornado analysis failed for {ticker}: {exc}")

        sensitivity_results[ticker] = result

    return {
        "sensitivity_results": sensitivity_results,
        "errors": errors,
        "status": "running",
    }
