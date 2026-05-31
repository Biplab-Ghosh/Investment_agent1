"""
Node 8 — Conversation Manager

Handles follow-up questions, what-if scenarios, and multi-stock comparisons
after the main analysis is complete. Uses the full analysis state as context.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.state import InvestmentAnalysisState
from src.utils.prompts import SYSTEM_PROMPT, QA_SYSTEM_PROMPT, QA_USER_PROMPT, WHAT_IF_PROMPT

logger = logging.getLogger(__name__)


def _build_analysis_summary(state: InvestmentAnalysisState) -> str:
    """Compact summary of analysis results for LLM context."""
    financial_data = state.get("financial_data", {})
    dcf_results = state.get("dcf_results", {})
    dcf_assumptions = state.get("dcf_assumptions", {})
    moat_analysis = state.get("moat_analysis", {})
    sensitivity_results = state.get("sensitivity_results", {})

    parts: list[str] = []
    for ticker in financial_data:
        cd = financial_data[ticker]
        dr = dcf_results.get(ticker, {})
        da = dcf_assumptions.get(ticker, {})
        mo = moat_analysis.get(ticker, {})
        sr = sensitivity_results.get(ticker, {})
        mc = sr.get("monte_carlo_percentiles", {})

        parts.append(f"""
{ticker} — {cd.get('name', ticker)} ({cd.get('sector', 'N/A')})
  Current Price: ${dr.get('current_price', 0):.2f}
  Intrinsic Value: ${dr.get('intrinsic_value_per_share', 0):.2f}
  Upside: {dr.get('upside_downside_pct', 0):+.1%}  |  Margin of Safety: {dr.get('margin_of_safety', 0):.1%}
  WACC: {da.get('wacc', 0):.2%}  |  Terminal Growth: {da.get('terminal_growth_rate', 0):.2%}
  Moat: {mo.get('rating', 'N/A')} ({mo.get('total_score', 0):.0f}/50)
  Bear/Base/Bull: ${sr.get('scenario_bear', 0):.0f} / ${sr.get('scenario_base', 0):.0f} / ${sr.get('scenario_bull', 0):.0f}
  Monte Carlo P50: ${mc.get('p50', 0):.0f}  (P10: ${mc.get('p10', 0):.0f}, P90: ${mc.get('p90', 0):.0f})
""")

    return "\n".join(parts) if parts else "No analysis data available."


def _detect_what_if(question: str) -> bool:
    """Simple keyword detection for what-if questions."""
    keywords = ["what if", "what-if", "scenario", "if wacc", "if growth", "assume", "suppose", "hypothetical"]
    q_lower = question.lower()
    return any(kw in q_lower for kw in keywords)


def conversation_manager_node(state: InvestmentAnalysisState) -> dict[str, Any]:
    """
    Answer the latest user question using the full analysis context.
    The question is expected to be the last message in conversation_history
    with role='user'.
    """
    conversation_history: list[dict[str, str]] = list(state.get("conversation_history", []))
    errors: list[str] = list(state.get("errors", []))

    if not conversation_history:
        return {
            "conversation_history": conversation_history,
            "errors": errors,
            "status": "complete",
        }

    # Find the last user message
    user_messages = [m for m in conversation_history if m.get("role") == "user"]
    if not user_messages:
        return {"conversation_history": conversation_history, "status": "complete"}

    latest_question = user_messages[-1]["content"]
    tickers = list(state.get("financial_data", {}).keys())
    primary_ticker = tickers[0] if tickers else "the stock"

    # Build analysis context
    analysis_summary = _build_analysis_summary(state)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from config.settings import DEFAULT_LLM_MODEL

        llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.3)

        # Build message history for multi-turn conversation
        messages: list = [
            SystemMessage(content=SYSTEM_PROMPT + "\n\n" + QA_SYSTEM_PROMPT.format(ticker=primary_ticker)),
        ]

        # Add prior conversation turns (excluding the latest user message)
        for msg in conversation_history[:-1]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=msg["content"]))

        # Add the latest question with context
        user_prompt = QA_USER_PROMPT.format(
            ticker=primary_ticker,
            question=latest_question,
            analysis_summary=analysis_summary,
        )
        messages.append(HumanMessage(content=user_prompt))

        response = llm.invoke(messages)
        answer = response.content
        logger.info("Answered question for %s: %d chars", primary_ticker, len(answer))

    except Exception as exc:
        errors.append(f"Q&A failed: {exc}")
        logger.error("Conversation manager error: %s", exc)
        answer = (
            f"I encountered an error while processing your question: {exc}\n\n"
            "Please check that your OpenAI API key is configured correctly."
        )

    # Append assistant response to history
    conversation_history.append({"role": "assistant", "content": answer})

    return {
        "conversation_history": conversation_history,
        "errors": errors,
        "status": "complete",
    }
