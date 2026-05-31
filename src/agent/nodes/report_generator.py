"""
Node 7 — Report Generator

Produces a comprehensive markdown investment research report for each ticker
and builds the full visualization suite.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import InvestmentAnalysisState
from src.utils.prompts import SYSTEM_PROMPT, INVESTMENT_REPORT_PROMPT

logger = logging.getLogger(__name__)


def _build_llm():
    from langchain_openai import ChatOpenAI
    from config.settings import DEFAULT_LLM_MODEL
    return ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.3, max_tokens=2000)


def _generate_report(
    ticker: str,
    company_data: dict[str, Any],
    dcf_results: dict[str, Any],
    dcf_assumptions: dict[str, Any],
    sensitivity_results: dict[str, Any],
    moat_analysis: dict[str, Any],
) -> str:
    """Generate markdown report via LLM, with fallback to templated report."""
    assumptions = dcf_assumptions.get(ticker, {})
    results = dcf_results.get(ticker, {})
    sensitivity = sensitivity_results.get(ticker, {})
    moat = moat_analysis.get(ticker, {})

    key_metrics = company_data.get("key_metrics", {})
    mc_pcts = sensitivity.get("monte_carlo_percentiles", {})
    wacc_components = results.get("wacc_components", {})

    # Find strongest moat dimension
    dimensions = ["intangible_assets", "switching_costs", "network_effects", "cost_advantages", "efficient_scale"]
    strongest_dim = max(dimensions, key=lambda d: moat.get(d, 0)) if moat else "N/A"
    strongest_label = strongest_dim.replace("_", " ").title()

    try:
        llm = _build_llm()
        prompt = INVESTMENT_REPORT_PROMPT.format(
            name=company_data.get("name", ticker),
            ticker=ticker,
            sector=company_data.get("sector", "Unknown"),
            industry=company_data.get("industry", "Unknown"),
            current_price=results.get("current_price", 0),
            market_cap_billions=(company_data.get("market_cap") or 0) / 1e9,
            intrinsic_value=results.get("intrinsic_value_per_share", 0),
            upside=results.get("upside_downside_pct", 0),
            margin_of_safety=results.get("margin_of_safety", 0),
            wacc=assumptions.get("wacc", 0),
            terminal_growth=assumptions.get("terminal_growth_rate", 0),
            bear_case=sensitivity.get("scenario_bear", 0),
            base_case=sensitivity.get("scenario_base", results.get("intrinsic_value_per_share", 0)),
            bull_case=sensitivity.get("scenario_bull", 0),
            mc_p50=mc_pcts.get("p50", 0),
            moat_rating=moat.get("rating", "Unknown"),
            moat_score=moat.get("total_score", 0),
            strongest_moat_dimension=strongest_label,
            cost_of_equity=wacc_components.get("cost_of_equity", 0),
            after_tax_kd=wacc_components.get("after_tax_cost_of_debt", 0),
            beta=wacc_components.get("beta_used", 0),
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        return response.content

    except Exception as exc:
        logger.warning("LLM report generation failed for %s: %s", ticker, exc)
        return _fallback_report(ticker, company_data, results, assumptions, sensitivity, moat, strongest_label)


def _fallback_report(
    ticker: str,
    company_data: dict,
    results: dict,
    assumptions: dict,
    sensitivity: dict,
    moat: dict,
    strongest_moat: str,
) -> str:
    """Template-based fallback report when LLM is unavailable."""
    iv = results.get("intrinsic_value_per_share", 0)
    price = results.get("current_price", 0)
    upside = results.get("upside_downside_pct", 0)
    mos = results.get("margin_of_safety", 0)
    moat_rating = moat.get("rating", "Unknown")
    moat_score = moat.get("total_score", 0)

    bear = sensitivity.get("scenario_bear", 0)
    bull = sensitivity.get("scenario_bull", 0)
    mc_pcts = sensitivity.get("monte_carlo_percentiles", {})

    return f"""# Investment Analysis Report: {ticker}

*Analysis Date: {company_data.get('analysis_date', 'N/A')}*

---

## Executive Summary

**{company_data.get('name', ticker)}** ({ticker}) — {company_data.get('sector', 'Unknown')} / {company_data.get('industry', 'Unknown')}

| Metric | Value |
|--------|-------|
| Current Price | ${price:.2f} |
| Intrinsic Value (DCF) | ${iv:.2f} |
| Upside / (Downside) | {upside:+.1%} |
| Margin of Safety | {mos:.1%} |
| Moat Rating | {moat_rating} ({moat_score:.0f}/50) |

---

## Valuation Analysis

**WACC**: {assumptions.get('wacc', 0):.2%} | **Terminal Growth**: {assumptions.get('terminal_growth_rate', 0):.2%}

**Revenue Growth Assumptions**: {', '.join(f'Y{i+1}: {g:.1%}' for i, g in enumerate(assumptions.get('revenue_growth_rates', [])))}

**Operating Margin Target**: {assumptions.get('operating_margin_target', 0):.1%}

The DCF model suggests an intrinsic value of **${iv:.2f}** per share, representing a **{upside:+.1%}** {'premium' if upside < 0 else 'discount'} to the current market price of ${price:.2f}.

---

## Competitive Position

**Moat Rating: {moat_rating}** (Score: {moat_score:.0f}/50)

Strongest moat dimension: **{strongest_moat}**

{moat.get('qualitative_summary', 'Qualitative assessment unavailable.')}

---

## Sensitivity Analysis

| Scenario | Intrinsic Value |
|----------|----------------|
| Bear Case | ${bear:.2f} |
| Base Case | ${iv:.2f} |
| Bull Case | ${bull:.2f} |
| Monte Carlo P50 | ${mc_pcts.get('p50', 0):.2f} |
| Monte Carlo P10-P90 | ${mc_pcts.get('p10', 0):.2f} – ${mc_pcts.get('p90', 0):.2f} |

---

## Key Risks

- Model risk: DCF is highly sensitive to WACC and terminal growth rate assumptions
- Data quality: Analysis based on publicly available financial statements
- Macro risk: Changes in interest rates directly affect WACC
- Competitive risk: Moat erosion could compress margins and growth

---

*⚠️ Disclaimer: This analysis is for educational purposes only and does not constitute investment advice. Always conduct your own due diligence before making investment decisions.*
"""


def _build_visualizations(
    ticker: str,
    company_data: dict[str, Any],
    dcf_results: dict[str, Any],
    sensitivity_results: dict[str, Any],
    moat_analysis: dict[str, Any],
    dcf_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build all charts for a ticker. Returns dict of {chart_name: Figure}."""
    from src.utils import visualization as viz

    results = dcf_results.get(ticker, {})
    sensitivity = sensitivity_results.get(ticker, {})
    moat = moat_analysis.get(ticker, {})
    current_price = results.get("current_price", 0)
    iv = results.get("intrinsic_value_per_share", 0)
    figs: dict[str, Any] = {}

    try:
        figs["dcf_waterfall"] = viz.dcf_waterfall(results, ticker=ticker)
    except Exception as exc:
        logger.warning("DCF waterfall chart failed: %s", exc)

    try:
        if "wacc_vs_tgr" in sensitivity:
            figs["sensitivity_heatmap"] = viz.sensitivity_heatmap(
                sensitivity["wacc_vs_tgr"], iv, current_price, ticker=ticker
            )
    except Exception as exc:
        logger.warning("Sensitivity heatmap failed: %s", exc)

    try:
        tornado = sensitivity.get("tornado_data", [])
        if tornado:
            figs["tornado"] = viz.tornado_chart(tornado, iv, ticker=ticker)
    except Exception as exc:
        logger.warning("Tornado chart failed: %s", exc)

    try:
        if moat:
            figs["moat_radar"] = viz.moat_radar(moat, ticker=ticker)
    except Exception as exc:
        logger.warning("Moat radar failed: %s", exc)

    try:
        mc = sensitivity.get("monte_carlo_raw", {})
        if mc and "raw_values" in mc:
            figs["monte_carlo"] = viz.monte_carlo_histogram(mc, current_price, iv, ticker=ticker)
    except Exception as exc:
        logger.warning("Monte Carlo histogram failed: %s", exc)

    try:
        scenarios = sensitivity.get("scenarios", {})
        if scenarios:
            figs["scenarios"] = viz.scenario_comparison_chart(scenarios, current_price, ticker=ticker)
    except Exception as exc:
        logger.warning("Scenario chart failed: %s", exc)

    try:
        figs["historical"] = viz.historical_metrics_chart(company_data, dcf_metrics, ticker=ticker)
    except Exception as exc:
        logger.warning("Historical chart failed: %s", exc)

    return figs


def report_generator_node(state: InvestmentAnalysisState) -> dict[str, Any]:
    """
    Generate markdown reports and visualization figures for each ticker.
    Populates state.final_reports and state.visualizations.
    """
    financial_data: dict[str, Any] = state.get("financial_data", {})
    dcf_results: dict[str, Any] = state.get("dcf_results", {})
    dcf_assumptions: dict[str, Any] = state.get("dcf_assumptions", {})
    sensitivity_results: dict[str, Any] = state.get("sensitivity_results", {})
    moat_analysis: dict[str, Any] = state.get("moat_analysis", {})
    errors: list[str] = list(state.get("errors", []))

    final_reports: dict[str, str] = {}
    visualizations: dict[str, Any] = {}

    from src.analysis.dcf_engine import DCFEngine
    _engine = DCFEngine()

    for ticker, company_data in financial_data.items():
        try:
            report = _generate_report(
                ticker=ticker,
                company_data=company_data,
                dcf_results=dcf_results,
                dcf_assumptions=dcf_assumptions,
                sensitivity_results=sensitivity_results,
                moat_analysis=moat_analysis,
            )
            final_reports[ticker] = report
            logger.info("Report generated for %s (%d chars)", ticker, len(report))
        except Exception as exc:
            errors.append(f"Report generation failed for {ticker}: {exc}")
            logger.error("Report error for %s: %s", ticker, exc)

        try:
            dcf_metrics = _engine.extract_historical_metrics(company_data)
            figs = _build_visualizations(
                ticker=ticker,
                company_data=company_data,
                dcf_results=dcf_results,
                sensitivity_results=sensitivity_results,
                moat_analysis=moat_analysis,
                dcf_metrics=dcf_metrics,
            )
            visualizations[ticker] = figs
            logger.info("Built %d charts for %s", len(figs), ticker)
        except Exception as exc:
            errors.append(f"Visualization failed for {ticker}: {exc}")

    return {
        "final_reports": final_reports,
        "visualizations": visualizations,
        "errors": errors,
        "status": "complete",
        "analysis_complete": True,
    }
