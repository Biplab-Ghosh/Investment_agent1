"""
Node 3 — DCF Calculator

Computes DCF assumptions for each ticker using historical metrics + LLM reasoning,
then runs the full DCF valuation. Populates state.dcf_assumptions and state.dcf_results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.state import InvestmentAnalysisState, DCFAssumptions
from src.analysis.dcf_engine import DCFEngine
from src.analysis.valuation_checks import check_dcf_valuation
from src.data.data_manager import DataManager
from src.utils.prompts import SYSTEM_PROMPT, DCF_ASSUMPTIONS_PROMPT

logger = logging.getLogger(__name__)

_engine = DCFEngine()


def _build_llm():
    from langchain_openai import ChatOpenAI
    from config.settings import DEFAULT_LLM_MODEL, LLM_TEMPERATURE
    return ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.05)  # low temp for numbers


def _heuristic_assumptions(
    metrics: dict[str, Any],
    wacc_dict: dict[str, float],
    lifecycle_stage: str,
    market_data: dict[str, Any],
) -> DCFAssumptions:
    """Build sensible assumptions without LLM when API is unavailable."""
    rev_cagr = metrics.get("revenue_cagr", 0.05)
    avg_margin = metrics.get("avg_operating_margin", 0.15)
    capex_pct = metrics.get("capex_pct_revenue", 0.04)
    nwc_pct = metrics.get("nwc_pct_revenue", 0.02)
    tax_rate = metrics.get("avg_tax_rate", 0.21)
    wacc = wacc_dict["wacc"]

    model_type = "3-stage" if lifecycle_stage in {"Early Growth", "High Growth"} else "2-stage"
    terminal_g = min(market_data.get("nominal_gdp_growth", 0.05), 0.03)
    if lifecycle_stage in {"Mature", "Declining"}:
        terminal_g = min(market_data.get("gdp_growth_real", 0.02), 0.02)

    if model_type == "3-stage":
        growth_rates = _engine.build_three_stage_growth_rates(
            rev_cagr, lifecycle_stage, forecast_years=5, terminal_growth=terminal_g
        )
    else:
        growth_rates = _engine.build_two_stage_growth_rates(rev_cagr, lifecycle_stage, 5)

    return DCFAssumptions(
        wacc=round(wacc, 4),
        terminal_growth_rate=round(terminal_g, 4),
        revenue_growth_rates=growth_rates,
        operating_margin_target=round(avg_margin, 4),
        capex_percent_revenue=round(capex_pct, 4),
        nwc_percent_revenue=round(nwc_pct, 4),
        tax_rate=round(tax_rate, 4),
        forecast_years=len(growth_rates),
        model_type=model_type,
        lifecycle_stage=lifecycle_stage,
        rationale={
            "wacc": f"Computed via CAPM: Ke={wacc_dict['cost_of_equity']:.2%}, Kd={wacc_dict['after_tax_cost_of_debt']:.2%}",
            "terminal_growth_rate": f"Capped at nominal GDP growth ({terminal_g:.1%})",
            "revenue_growth": f"Based on 5yr CAGR of {rev_cagr:.1%}, decelerating with lifecycle",
            "operating_margin": f"Historical avg {avg_margin:.1%}",
        },
    )


def dcf_calculator_node(state: InvestmentAnalysisState) -> dict[str, Any]:
    """
    Compute DCF assumptions and run valuation for each ticker.
    User overrides from state.user_overrides are applied on top of computed assumptions.
    """
    financial_data: dict[str, Any] = state.get("financial_data", {})
    market_data: dict[str, Any] = state.get("market_data", {})
    company_profiles: dict[str, dict] = state.get("company_profiles", {})
    user_overrides: dict[str, Any] = state.get("user_overrides", {})
    errors: list[str] = list(state.get("errors", []))

    dcf_assumptions: dict[str, DCFAssumptions] = {}
    dcf_results: dict[str, Any] = {}

    from config.settings import MARKET_RISK_PREMIUM

    dm = DataManager()

    for ticker, company_data in financial_data.items():
        profile = company_profiles.get(ticker, {})
        lifecycle_stage = profile.get("lifecycle_stage", "Mature")
        name = company_data.get("name", ticker)
        key_metrics = company_data.get("key_metrics", {})

        # ── Extract historical metrics ────────────────────────────────────────
        metrics = _engine.extract_historical_metrics(company_data)

        # ── Compute WACC ──────────────────────────────────────────────────────
        beta = float(company_data.get("beta") or 1.0)
        rf = float(market_data.get("risk_free_rate", 0.045))
        kd = float(metrics.get("cost_of_debt", 0.05))
        tax = float(metrics.get("avg_tax_rate", 0.21))
        mkt_cap = float(company_data.get("market_cap") or 0)
        total_debt = float(metrics.get("latest_total_debt", 0))

        wacc_dict = _engine.calculate_wacc(
            beta=beta,
            risk_free_rate=rf,
            market_risk_premium=MARKET_RISK_PREMIUM,
            cost_of_debt=kd,
            tax_rate=tax,
            market_cap=mkt_cap,
            total_debt=total_debt,
        )

        # ── Build assumptions (LLM → heuristic fallback) ─────────────────────
        assumptions: DCFAssumptions | None = None
        try:
            llm = _build_llm()
            rev = metrics.get("revenues")
            latest_rev = float(rev.iloc[-1]) if rev is not None and not rev.empty else 0

            prompt = DCF_ASSUMPTIONS_PROMPT.format(
                name=name,
                ticker=ticker,
                revenue_cagr=metrics.get("revenue_cagr", 0),
                avg_operating_margin=metrics.get("avg_operating_margin", 0.15),
                latest_operating_margin=metrics.get("latest_operating_margin", 0.15),
                capex_pct=metrics.get("capex_pct_revenue", 0.04),
                tax_rate=tax,
                nwc_pct=metrics.get("nwc_pct_revenue", 0.02),
                current_price=float(company_data.get("current_price") or 0),
                shares_billions=float(company_data.get("shares_outstanding") or 0) / 1e9,
                beta=beta,
                risk_free_rate=rf,
                market_risk_premium=MARKET_RISK_PREMIUM,
                cost_of_equity=wacc_dict["cost_of_equity"],
                after_tax_kd=wacc_dict["after_tax_cost_of_debt"],
                wacc=wacc_dict["wacc"],
                sector=company_data.get("sector", "Unknown"),
                lifecycle_stage=lifecycle_stage,
                model_type=profile.get("dcf_model_type", "2-stage"),
            )

            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            text = response.content
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            llm_assumptions = json.loads(text.strip())
            assumptions = DCFAssumptions(**{k: v for k, v in llm_assumptions.items() if k in DCFAssumptions.__annotations__})
            logger.info("LLM suggested WACC=%.2f%% for %s", assumptions.get("wacc", 0) * 100, ticker)

        except Exception as exc:
            logger.warning("LLM assumption generation failed for %s (%s), using heuristics", ticker, exc)
            errors.append(f"DCF LLM fallback for {ticker}: {exc}")
            assumptions = _heuristic_assumptions(metrics, wacc_dict, lifecycle_stage, market_data)

        # ── Manual CSV + user overrides ───────────────────────────────────────
        manual_overrides = dm.get_manual_assumption_overrides(ticker)
        if manual_overrides:
            assumptions = {**assumptions, **manual_overrides}
            logger.info("Applied manual CSV overrides for %s", ticker)

        ticker_overrides = user_overrides.get(ticker, {})
        if ticker_overrides:
            assumptions = {**assumptions, **ticker_overrides}
            logger.info("Applied %d user overrides for %s", len(ticker_overrides), ticker)

        assumptions = {
            **assumptions,
            "model_type": profile.get("dcf_model_type", assumptions.get("model_type", "2-stage")),
            "lifecycle_stage": lifecycle_stage,
        }
        dcf_assumptions[ticker] = assumptions

        # ── Run DCF ───────────────────────────────────────────────────────────
        try:
            result = _engine.run_full_dcf(company_data, assumptions, market_data)
            validation = check_dcf_valuation(ticker, result, company_data)
            result["validation"] = validation
            for warning in validation.get("warnings", []):
                errors.append(warning)
            dcf_results[ticker] = result
            logger.info(
                "DCF for %s: IV=$%.2f, current=$%.2f, upside=%.1f%%",
                ticker,
                result.get("intrinsic_value_per_share", 0),
                result.get("current_price", 0),
                result.get("upside_downside_pct", 0) * 100,
            )
        except Exception as exc:
            errors.append(f"DCF calculation failed for {ticker}: {exc}")
            logger.error("DCF failed for %s: %s", ticker, exc, exc_info=True)

    return {
        "dcf_assumptions": dcf_assumptions,
        "dcf_results": dcf_results,
        "errors": errors,
        "status": "awaiting_input",
        "pending_user_approval": "assumption_review",
    }
