"""
LLM prompt templates for all agent nodes.

All prompts are plain strings with {placeholder} substitution via str.format().
Kept separate so they can be tuned without touching business logic.
"""

from __future__ import annotations

# ── System prompt (shared across all nodes) ────────────────────────────────────

SYSTEM_PROMPT = """You are an expert equity research analyst and DCF valuation specialist with 20+ years of experience at top-tier investment banks and hedge funds.

Your role is to assist in investment analysis using Discounted Cash Flow (DCF) methodology and competitive moat analysis. You provide rigorous, data-driven analysis while clearly communicating your reasoning.

Guidelines:
- Be precise with numbers and always show your work
- Flag data quality issues or missing information
- Acknowledge uncertainty and provide ranges, not just point estimates
- Use finance industry terminology correctly
- Always add a disclaimer that this is educational analysis, not investment advice
- Keep explanations clear for both experts and learners
"""

# ── Company analysis prompts ───────────────────────────────────────────────────

COMPANY_STAGE_PROMPT = """Based on the following company data, determine the company's lifecycle stage and recommend an appropriate DCF model.

Company: {name} ({ticker})
Sector: {sector}
Industry: {industry}
Revenue Growth (TTM): {revenue_growth:.1%}
Operating Margin: {operating_margin:.1%}
Market Cap: ${market_cap_billions:.1f}B
Revenue CAGR (5yr): {revenue_cagr:.1%}
Current P/E: {pe_ratio}

Determine:
1. **Lifecycle Stage**: (Startup/Early Growth / High Growth / Mature Growth / Mature / Declining)
2. **Recommended DCF Model**: (2-stage / 3-stage) with justification
3. **Key Valuation Considerations**: 2-3 specific factors that will drive the DCF

Respond in JSON format:
{{
    "lifecycle_stage": "...",
    "dcf_model_type": "2-stage or 3-stage",
    "stage_rationale": "...",
    "key_valuation_factors": ["factor1", "factor2", "factor3"],
    "growth_profile": "description of expected growth trajectory"
}}"""

# ── DCF assumption prompts ─────────────────────────────────────────────────────

DCF_ASSUMPTIONS_PROMPT = """You are preparing DCF assumptions for {name} ({ticker}).

Historical Data:
- Revenue CAGR (5yr): {revenue_cagr:.1%}
- Average Operating Margin: {avg_operating_margin:.1%}
- Latest Operating Margin: {latest_operating_margin:.1%}
- CapEx as % of Revenue: {capex_pct:.1%}
- Effective Tax Rate: {tax_rate:.1%}
- Net Working Capital % Revenue: {nwc_pct:.1%}

Market Data:
- Current Stock Price: ${current_price:.2f}
- Shares Outstanding: {shares_billions:.2f}B
- Beta: {beta:.2f}
- Risk-Free Rate: {risk_free_rate:.2%}
- Market Risk Premium: {market_risk_premium:.2%}

Computed WACC Components:
- Cost of Equity (CAPM): {cost_of_equity:.2%}
- Cost of Debt (after-tax): {after_tax_kd:.2%}
- WACC: {wacc:.2%}

Sector: {sector}
Lifecycle Stage: {lifecycle_stage}
DCF Model: {model_type}

Recommend specific values for each DCF assumption with brief rationale. Be conservative and realistic.

Respond in JSON format:
{{
    "wacc": {wacc:.4f},
    "terminal_growth_rate": 0.025,
    "revenue_growth_rates": [year1_growth, year2_growth, ...],
    "operating_margin_target": {avg_operating_margin:.4f},
    "capex_percent_revenue": {capex_pct:.4f},
    "nwc_percent_revenue": {nwc_pct:.4f},
    "tax_rate": {tax_rate:.4f},
    "model_type": "{model_type}",
    "rationale": {{
        "wacc": "explanation",
        "terminal_growth_rate": "explanation",
        "revenue_growth": "explanation",
        "operating_margin": "explanation"
    }}
}}"""

DCF_ASSUMPTION_EXPLANATION_PROMPT = """Explain the following DCF assumptions for {name} ({ticker}) in plain English for a sophisticated investor:

Assumptions:
- WACC: {wacc:.2%}
- Terminal Growth Rate: {terminal_growth:.2%}
- Revenue Growth (Year 1-{n_years}): {growth_rates_str}
- Operating Margin Target: {op_margin:.2%}
- CapEx % Revenue: {capex_pct:.2%}
- Tax Rate: {tax_rate:.2%}

DCF Results:
- Intrinsic Value per Share: ${intrinsic_value:.2f}
- Current Price: ${current_price:.2f}
- Upside/(Downside): {upside:.1%}
- Margin of Safety: {margin_of_safety:.1%}

Provide:
1. A brief explanation of what each assumption means and why it was chosen
2. Key risks to the assumptions
3. A 2-3 sentence investment thesis summary

Keep the explanation concise (under 400 words total)."""

# ── Moat analysis prompts ──────────────────────────────────────────────────────

MOAT_QUALITATIVE_PROMPT = """Conduct a qualitative economic moat assessment for {name} ({ticker}).

Company Profile:
- Sector: {sector}
- Industry: {industry}
- Market Cap: ${market_cap_billions:.1f}B

Quantitative Moat Scores:
- Intangible Assets: {intangible_assets:.1f}/10
- Switching Costs: {switching_costs:.1f}/10
- Network Effects: {network_effects:.1f}/10
- Cost Advantages: {cost_advantages:.1f}/10
- Efficient Scale: {efficient_scale:.1f}/10
- Total Score: {total_score:.1f}/50
- Overall Rating: {rating}

Key Financial Metrics:
- Gross Margin: {gross_margin:.1%}
- Operating Margin: {operating_margin:.1%}
- ROE: {roe:.1%}
- ROIC (latest): {roic:.1%}

Provide a qualitative assessment covering:
1. **Primary Moat Source**: The strongest competitive advantage and why it persists
2. **Moat Durability**: How sustainable is this moat over 10 years?
3. **Key Threats**: Top 2-3 risks that could erode the moat
4. **Porter's Five Forces**: Brief assessment of competitive intensity
5. **Investment Implication**: How should the moat affect DCF discount rate and terminal growth?

Format as structured markdown with clear headers. Be specific to {name}'s actual business model."""

MOAT_SUSTAINABILITY_PROMPT = """In 3-4 sentences, assess the 10-year sustainability of {name}'s ({ticker}) economic moat, rated as {rating}.

Key factors:
- Primary advantage: {primary_advantage}
- Sector: {sector}
- ROIC trend: {roic_trend}

Focus on: what could erode the moat and what reinforces it."""

# ── Report generation prompts ──────────────────────────────────────────────────

INVESTMENT_REPORT_PROMPT = """Generate a comprehensive investment analysis report for {name} ({ticker}).

COMPANY DATA:
- Sector: {sector} | Industry: {industry}
- Current Price: ${current_price:.2f} | Market Cap: ${market_cap_billions:.1f}B

DCF VALUATION:
- Intrinsic Value: ${intrinsic_value:.2f}/share
- Upside/(Downside): {upside:.1%}
- Margin of Safety: {margin_of_safety:.1%}
- WACC Used: {wacc:.2%}
- Terminal Growth: {terminal_growth:.2%}

SENSITIVITY:
- Bear Case: ${bear_case:.2f}
- Base Case: ${base_case:.2f}
- Bull Case: ${bull_case:.2f}
- Monte Carlo P50: ${mc_p50:.2f}

MOAT ANALYSIS:
- Overall Rating: {moat_rating}
- Total Score: {moat_score:.1f}/50
- Strongest Dimension: {strongest_moat_dimension}

WACC COMPONENTS:
- Cost of Equity: {cost_of_equity:.2%}
- After-Tax Cost of Debt: {after_tax_kd:.2%}
- Beta: {beta:.2f}

Write a professional equity research note (400-600 words) with these sections:

## Executive Summary
One paragraph with investment thesis and rating.

## Valuation Analysis
DCF methodology, key assumptions, intrinsic value, and margin of safety.

## Competitive Position
Moat assessment and how it supports/challenges the valuation.

## Key Risks
Top 3-4 risks in bullet format.

## Sensitivity Analysis
What drives the range between bear and bull cases.

## Investment Conclusion
Clear recommendation with price target rationale.

---
*Disclaimer: This analysis is for educational purposes only and does not constitute investment advice.*"""

# ── Conversation / Q&A prompts ─────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are an investment research assistant with deep expertise in DCF valuation and competitive analysis.

You have just completed an analysis of {ticker}. You have access to:
- Full DCF valuation results
- Moat analysis scores
- Sensitivity analysis results
- Historical financial data

Answer questions accurately using the analysis data. If asked about something not in the analysis, say so clearly.
Keep answers concise and focused. When discussing numbers, be precise."""

QA_USER_PROMPT = """Based on the investment analysis for {ticker}, please answer this question:

{question}

Available Analysis Data:
{analysis_summary}

Answer in 2-4 paragraphs. Be specific and cite numbers from the analysis."""

WHAT_IF_PROMPT = """The user wants to explore a 'what-if' scenario for {ticker}:

Scenario: {scenario_description}

Base Case Results:
- Intrinsic Value: ${base_intrinsic:.2f}
- WACC: {base_wacc:.2%}
- Terminal Growth: {base_tgr:.2%}
- Revenue Growth (Yr1): {base_g1:.1%}

Modified Scenario Results:
- Intrinsic Value: ${scenario_intrinsic:.2f}
- Change: {change_pct:.1%}

Explain what this scenario means for the investment case in 2-3 sentences."""
