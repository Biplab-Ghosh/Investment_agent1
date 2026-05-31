"""
Node 4 — Assumption Review (Human-in-the-Loop interrupt point)

This node formats assumptions for human review and generates an LLM explanation.
The graph is configured to INTERRUPT_BEFORE this node, so the Jupyter notebook
can display the presentation, collect user modifications, and then resume.

After the graph resumes, any updates in state.user_overrides are merged back
into state.dcf_assumptions and the DCF is recalculated.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import InvestmentAnalysisState
from src.analysis.dcf_engine import DCFEngine
from src.utils.prompts import SYSTEM_PROMPT, DCF_ASSUMPTION_EXPLANATION_PROMPT

logger = logging.getLogger(__name__)

_engine = DCFEngine()


def _format_assumptions_table(ticker: str, assumptions: dict, results: dict) -> str:
    """Build a human-readable assumptions table for display."""
    growth_rates = assumptions.get("revenue_growth_rates", [])
    growth_str = ", ".join(f"Y{i+1}: {g:.1%}" for i, g in enumerate(growth_rates))

    iv = results.get("intrinsic_value_per_share", 0)
    price = results.get("current_price", 0)
    upside = results.get("upside_downside_pct", 0)
    mos = results.get("margin_of_safety", 0)

    wacc_components = results.get("wacc_components", {})

    table = f"""
╔══════════════════════════════════════════════════════════════════════╗
║           DCF ASSUMPTIONS REVIEW — {ticker:<10}                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  WACC COMPONENTS                                                     ║
║  ├─ Beta Used:              {wacc_components.get('beta_used', 0):.2f}                               ║
║  ├─ Risk-Free Rate:         {wacc_components.get('risk_free_rate', 0):.2%}                            ║
║  ├─ Cost of Equity (CAPM):  {wacc_components.get('cost_of_equity', 0):.2%}                            ║
║  ├─ After-Tax Cost of Debt: {wacc_components.get('after_tax_cost_of_debt', 0):.2%}                    ║
║  └─ WACC:                   {assumptions.get('wacc', 0):.2%}                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  DCF ASSUMPTIONS                                                     ║
║  ├─ Model Type:             {assumptions.get('model_type', '2-stage'):<30}       ║
║  ├─ Terminal Growth Rate:   {assumptions.get('terminal_growth_rate', 0):.2%}                          ║
║  ├─ Operating Margin:       {assumptions.get('operating_margin_target', 0):.2%}                       ║
║  ├─ CapEx % Revenue:        {assumptions.get('capex_percent_revenue', 0):.2%}                         ║
║  ├─ Tax Rate:               {assumptions.get('tax_rate', 0):.2%}                              ║
║  └─ Revenue Growth:         {growth_str[:40]:<40}       ║
╠══════════════════════════════════════════════════════════════════════╣
║  VALUATION RESULTS                                                   ║
║  ├─ Intrinsic Value:        ${iv:>8.2f}/share                           ║
║  ├─ Current Price:          ${price:>8.2f}/share                         ║
║  ├─ Upside / (Downside):    {upside:>+.1%}                              ║
║  └─ Margin of Safety:       {mos:.1%}                               ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    return table


def _get_llm_explanation(ticker: str, assumptions: dict, results: dict) -> str:
    """Generate an LLM explanation of the assumptions."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from config.settings import DEFAULT_LLM_MODEL

        llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.2)

        growth_rates = assumptions.get("revenue_growth_rates", [])
        growth_str = ", ".join(f"Y{i+1}: {g:.1%}" for i, g in enumerate(growth_rates))

        prompt = DCF_ASSUMPTION_EXPLANATION_PROMPT.format(
            name=ticker,
            ticker=ticker,
            wacc=assumptions.get("wacc", 0),
            terminal_growth=assumptions.get("terminal_growth_rate", 0),
            n_years=len(growth_rates),
            growth_rates_str=growth_str,
            op_margin=assumptions.get("operating_margin_target", 0),
            capex_pct=assumptions.get("capex_percent_revenue", 0),
            tax_rate=assumptions.get("tax_rate", 0),
            intrinsic_value=results.get("intrinsic_value_per_share", 0),
            current_price=results.get("current_price", 0),
            upside=results.get("upside_downside_pct", 0),
            margin_of_safety=results.get("margin_of_safety", 0),
        )

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return response.content

    except Exception as exc:
        logger.warning("LLM explanation failed: %s", exc)
        rationale = assumptions.get("rationale", {})
        lines = [f"**{k}**: {v}" for k, v in rationale.items()]
        return "\n".join(lines) or "Assumptions based on historical financial analysis."


def assumption_review_node(state: InvestmentAnalysisState) -> dict[str, Any]:
    """
    Prepare assumption presentation for human review.
    This node runs AFTER the LangGraph interrupt, so by the time it executes
    the user has already had a chance to update state.user_overrides.

    If user_overrides contains ticker-level overrides, merge them and
    re-run the DCF before marking approval complete.
    """
    dcf_assumptions: dict[str, Any] = dict(state.get("dcf_assumptions", {}))
    dcf_results: dict[str, Any] = dict(state.get("dcf_results", {}))
    financial_data: dict[str, Any] = state.get("financial_data", {})
    market_data: dict[str, Any] = state.get("market_data", {})
    user_overrides: dict[str, Any] = state.get("user_overrides", {})
    errors: list[str] = list(state.get("errors", []))

    # ── Apply any new overrides and re-run DCF ────────────────────────────────
    for ticker, overrides in user_overrides.items():
        if ticker in dcf_assumptions and overrides:
            merged = {**dcf_assumptions[ticker], **overrides}
            dcf_assumptions[ticker] = merged
            try:
                result = _engine.run_full_dcf(financial_data[ticker], merged, market_data)
                dcf_results[ticker] = result
                logger.info("Re-ran DCF for %s with user overrides", ticker)
            except Exception as exc:
                errors.append(f"DCF re-calculation failed for {ticker}: {exc}")

    # ── Build presentation text ───────────────────────────────────────────────
    presentation_parts: list[str] = []
    for ticker in dcf_assumptions:
        assumptions = dcf_assumptions[ticker]
        results = dcf_results.get(ticker, {})
        table = _format_assumptions_table(ticker, assumptions, results)
        explanation = _get_llm_explanation(ticker, assumptions, results)
        presentation_parts.append(f"{table}\n\n**Agent Analysis:**\n{explanation}")

    presentation = "\n\n" + "=" * 70 + "\n\n".join(presentation_parts)

    return {
        "dcf_assumptions": dcf_assumptions,
        "dcf_results": dcf_results,
        "assumption_presentation": presentation,
        "pending_user_approval": None,
        "errors": errors,
        "status": "running",
    }
