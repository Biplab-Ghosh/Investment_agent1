"""
Node 2 — Company Analysis

Determines lifecycle stage and DCF model type for each ticker
using the LLM (GPT-4o mini) with structured JSON output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.state import InvestmentAnalysisState
from src.utils.prompts import SYSTEM_PROMPT, COMPANY_STAGE_PROMPT

logger = logging.getLogger(__name__)


def _build_llm():
    from langchain_openai import ChatOpenAI
    from config.settings import DEFAULT_LLM_MODEL, LLM_TEMPERATURE
    return ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=LLM_TEMPERATURE)


def _classify_stage_heuristic(key_metrics: dict[str, Any]) -> dict[str, Any]:
    """Fallback heuristic classification when LLM is unavailable."""
    rev_growth = key_metrics.get("revenue_growth") or 0
    op_margin = key_metrics.get("operating_margins") or 0
    market_cap = key_metrics.get("market_cap") or 0

    if rev_growth > 0.25 and op_margin < 0.05:
        stage, model = "Early Growth", "3-stage"
    elif rev_growth > 0.15:
        stage, model = "High Growth", "3-stage"
    elif rev_growth > 0.05:
        stage, model = "Mature Growth", "2-stage"
    else:
        stage, model = "Mature", "2-stage"

    return {
        "lifecycle_stage": stage,
        "dcf_model_type": model,
        "stage_rationale": f"Heuristic: revenue growth {rev_growth:.1%}, margin {op_margin:.1%}",
        "key_valuation_factors": ["revenue growth trajectory", "margin expansion potential", "capital efficiency"],
        "growth_profile": f"Revenue growing at {rev_growth:.1%} with {op_margin:.1%} operating margin",
    }


def company_analysis_node(state: InvestmentAnalysisState) -> dict[str, Any]:
    """
    Classify each company's lifecycle stage and recommend a DCF model structure.
    Populates state.company_profiles.
    """
    financial_data: dict[str, Any] = state.get("financial_data", {})
    errors: list[str] = list(state.get("errors", []))
    company_profiles: dict[str, dict] = {}

    for ticker, company_data in financial_data.items():
        key_metrics = company_data.get("key_metrics", {})
        name = company_data.get("name", ticker)

        # ── Try LLM classification ───────────────────────────────────────────
        profile = None
        try:
            llm = _build_llm()
            prompt = COMPANY_STAGE_PROMPT.format(
                name=name,
                ticker=ticker,
                sector=company_data.get("sector", "Unknown"),
                industry=company_data.get("industry", "Unknown"),
                revenue_growth=key_metrics.get("revenue_growth") or 0,
                operating_margin=key_metrics.get("operating_margins") or 0,
                market_cap_billions=(key_metrics.get("market_cap") or 0) / 1e9,
                revenue_cagr=0.05,  # will be refined by DCF engine
                pe_ratio=key_metrics.get("trailing_pe") or "N/A",
            )

            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            text = response.content
            # Extract JSON block
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            profile = json.loads(text.strip())
            logger.info("LLM classified %s as %s (%s)", ticker,
                        profile.get("lifecycle_stage"), profile.get("dcf_model_type"))

        except Exception as exc:
            logger.warning("LLM company analysis failed for %s (%s), using heuristics", ticker, exc)
            errors.append(f"Company analysis LLM fallback for {ticker}: {exc}")
            profile = _classify_stage_heuristic(key_metrics)

        company_profiles[ticker] = {
            "ticker": ticker,
            "name": name,
            "sector": company_data.get("sector"),
            "industry": company_data.get("industry"),
            **profile,
        }

    return {
        "company_profiles": company_profiles,
        "errors": errors,
        "status": "running",
    }
