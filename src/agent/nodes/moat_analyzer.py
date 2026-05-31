"""
Node 6 — Moat Analyzer

Scores competitive moat across 5 dimensions using financial data,
then generates qualitative assessment via LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import InvestmentAnalysisState, MoatScore
from src.analysis.moat_engine import MoatEngine
from src.analysis.dcf_engine import DCFEngine
from src.utils.prompts import SYSTEM_PROMPT, MOAT_QUALITATIVE_PROMPT, MOAT_SUSTAINABILITY_PROMPT

logger = logging.getLogger(__name__)

_moat_engine = MoatEngine()
_dcf_engine = DCFEngine()


def _get_qualitative_assessment(
    ticker: str,
    company_data: dict[str, Any],
    moat_scores: dict[str, Any],
) -> str:
    """Generate LLM qualitative moat narrative."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from config.settings import DEFAULT_LLM_MODEL

        llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.3)

        prompt = MOAT_QUALITATIVE_PROMPT.format(
            name=company_data.get("name", ticker),
            ticker=ticker,
            sector=company_data.get("sector", "Unknown"),
            industry=company_data.get("industry", "Unknown"),
            market_cap_billions=(company_data.get("market_cap") or 0) / 1e9,
            intangible_assets=moat_scores.get("intangible_assets", 0),
            switching_costs=moat_scores.get("switching_costs", 0),
            network_effects=moat_scores.get("network_effects", 0),
            cost_advantages=moat_scores.get("cost_advantages", 0),
            efficient_scale=moat_scores.get("efficient_scale", 0),
            total_score=moat_scores.get("total_score", 0),
            rating=moat_scores.get("rating", "Unknown"),
            gross_margin=moat_scores.get("gross_margin", 0),
            operating_margin=moat_scores.get("operating_margin", 0),
            roe=moat_scores.get("roe", 0),
            roic=moat_scores.get("roic_latest", 0),
        )

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content

    except Exception as exc:
        logger.warning("LLM moat qualitative failed for %s: %s", ticker, exc)
        rating = moat_scores.get("rating", "Unknown")
        total = moat_scores.get("total_score", 0)
        return (
            f"**{ticker} Moat Summary ({rating}, {total}/50)**\n\n"
            "Quantitative scoring suggests the competitive position shown above. "
            "Full qualitative analysis requires LLM access."
        )


def moat_analyzer_node(state: InvestmentAnalysisState) -> dict[str, Any]:
    """
    Run full moat analysis for each ticker.
    Populates state.moat_analysis.
    """
    financial_data: dict[str, Any] = state.get("financial_data", {})
    dcf_assumptions: dict[str, Any] = state.get("dcf_assumptions", {})
    errors: list[str] = list(state.get("errors", []))
    moat_analysis: dict[str, MoatScore] = {}

    for ticker, company_data in financial_data.items():
        try:
            # Extract historical metrics for ROIC enrichment
            dcf_metrics = _dcf_engine.extract_historical_metrics(company_data)

            # Run quantitative scoring
            scores = _moat_engine.run_full_moat_analysis(company_data, dcf_metrics)

            # Generate qualitative narrative
            qualitative = _get_qualitative_assessment(ticker, company_data, scores)
            scores["qualitative_summary"] = qualitative

            moat_analysis[ticker] = MoatScore(
                intangible_assets=scores["intangible_assets"],
                switching_costs=scores["switching_costs"],
                network_effects=scores["network_effects"],
                cost_advantages=scores["cost_advantages"],
                efficient_scale=scores["efficient_scale"],
                total_score=scores["total_score"],
                rating=scores["rating"],
                qualitative_summary=qualitative,
                roic_5yr_avg=scores.get("roic_5yr_avg", 0),
                sustainability_assessment=scores.get("roic_note", ""),
            )

            logger.info(
                "Moat analysis for %s: %s (%.1f/50)",
                ticker, scores["rating"], scores["total_score"]
            )

        except Exception as exc:
            errors.append(f"Moat analysis failed for {ticker}: {exc}")
            logger.error("Moat analysis error for %s: %s", ticker, exc, exc_info=True)

    return {
        "moat_analysis": moat_analysis,
        "errors": errors,
        "status": "running",
    }
