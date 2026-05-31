"""
LangGraph agent graph definition.

Graph flow:
  START
    → data_acquisition
    → company_analysis
    → dcf_calculator
    [INTERRUPT] ← human reviews assumptions here in the Jupyter notebook
    → assumption_review      ← resumes after user approves/modifies
    → sensitivity_analysis
    → moat_analyzer
    → report_generator
    → [conversation_manager] ← optional Q&A loop
    → END

Human-in-the-loop is implemented via LangGraph's interrupt_before mechanism.
The Jupyter notebook calls graph.stream() until the interrupt, then lets
the user modify assumptions in state.user_overrides, then resumes.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import InvestmentAnalysisState
from src.agent.nodes.data_acquisition import data_acquisition_node
from src.agent.nodes.company_analysis import company_analysis_node
from src.agent.nodes.dcf_calculator import dcf_calculator_node
from src.agent.nodes.assumption_review import assumption_review_node
from src.agent.nodes.sensitivity_analysis import sensitivity_analysis_node
from src.agent.nodes.moat_analyzer import moat_analyzer_node
from src.agent.nodes.report_generator import report_generator_node
from src.agent.nodes.conversation_manager import conversation_manager_node

logger = logging.getLogger(__name__)


def _should_continue_to_qa(state: InvestmentAnalysisState) -> str:
    """Route to conversation_manager if there's a pending user question."""
    history = state.get("conversation_history", [])
    if not history:
        return "end"
    last = history[-1]
    if last.get("role") == "user":
        return "conversation_manager"
    return "end"


def build_graph(with_interrupt: bool = True) -> Any:
    """
    Build and compile the investment analysis LangGraph.

    Args:
        with_interrupt: If True, the graph will pause before assumption_review
                        for human-in-the-loop review. Set to False for fully
                        automated runs (testing / batch analysis).

    Returns:
        Compiled LangGraph (CompiledStateGraph).
    """
    workflow = StateGraph(InvestmentAnalysisState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("data_acquisition", data_acquisition_node)
    workflow.add_node("company_analysis", company_analysis_node)
    workflow.add_node("dcf_calculator", dcf_calculator_node)
    workflow.add_node("assumption_review", assumption_review_node)
    workflow.add_node("sensitivity_analysis", sensitivity_analysis_node)
    workflow.add_node("moat_analyzer", moat_analyzer_node)
    workflow.add_node("report_generator", report_generator_node)
    workflow.add_node("conversation_manager", conversation_manager_node)

    # ── Main analysis flow ────────────────────────────────────────────────────
    workflow.add_edge(START, "data_acquisition")
    workflow.add_edge("data_acquisition", "company_analysis")
    workflow.add_edge("company_analysis", "dcf_calculator")
    workflow.add_edge("dcf_calculator", "assumption_review")
    workflow.add_edge("assumption_review", "sensitivity_analysis")
    workflow.add_edge("sensitivity_analysis", "moat_analyzer")
    workflow.add_edge("moat_analyzer", "report_generator")

    # ── Post-analysis Q&A routing ─────────────────────────────────────────────
    workflow.add_conditional_edges(
        "report_generator",
        _should_continue_to_qa,
        {"conversation_manager": "conversation_manager", "end": END},
    )
    workflow.add_conditional_edges(
        "conversation_manager",
        _should_continue_to_qa,
        {"conversation_manager": "conversation_manager", "end": END},
    )

    # ── Compile (checkpointing only needed for human-in-the-loop interrupts) ─
    interrupt_nodes = ["assumption_review"] if with_interrupt else []

    if with_interrupt:
        graph = workflow.compile(
            checkpointer=MemorySaver(),
            interrupt_before=interrupt_nodes,
        )
    else:
        # No checkpointer: state may contain pandas/plotly objects unsuitable for serde.
        graph = workflow.compile()
    logger.info(
        "Graph compiled. Interrupt before: %s",
        interrupt_nodes if interrupt_nodes else "none",
    )
    return graph


# ── Convenience functions for notebook use ────────────────────────────────────

def create_initial_state(
    tickers: list[str],
    user_overrides: dict[str, Any] | None = None,
    manual_data_paths: dict[str, str] | None = None,
) -> InvestmentAnalysisState:
    """Create a fresh initial state for a new analysis run."""
    return InvestmentAnalysisState(
        ticker_symbols=[t.upper().strip() for t in tickers],
        analysis_date="",
        user_overrides=user_overrides or {},
        manual_data_paths=manual_data_paths or {},
        financial_data={},
        market_data={},
        company_profiles={},
        dcf_assumptions={},
        dcf_results={},
        moat_analysis={},
        sensitivity_results={},
        conversation_history=[],
        pending_user_approval=None,
        assumption_presentation=None,
        final_reports={},
        visualizations={},
        current_ticker=None,
        errors=[],
        status="running",
    )


def run_analysis(
    tickers: list[str],
    user_overrides: dict[str, Any] | None = None,
    thread_id: str = "default",
    with_interrupt: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Run the investment analysis graph.

    If with_interrupt=True, the graph will pause before assumption_review.
    Call resume_after_review() to continue after human review.

    Returns:
        (graph, config) tuple — keep both to resume the graph.
    """
    graph = build_graph(with_interrupt=with_interrupt)
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(tickers, user_overrides)

    print(f"Starting analysis for: {', '.join(tickers)}")
    print("-" * 60)

    events = []
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        status = event.get("status", "")
        errors = event.get("errors", [])
        if errors:
            for err in errors[-3:]:  # show last 3 errors only
                print(f"  ⚠ {err}")
        if status == "awaiting_input":
            presentation = event.get("assumption_presentation")
            if presentation:
                print("\n" + presentation)
            print("\n[PAUSED] Review assumptions above. Call resume_after_review() to continue.")
            break
        events.append(event)

    return graph, config


def resume_after_review(
    graph: Any,
    config: dict[str, Any],
    user_overrides: dict[str, Any] | None = None,
) -> InvestmentAnalysisState:
    """
    Resume the graph after human assumption review.

    Args:
        graph: The compiled graph returned by run_analysis()
        config: The config dict returned by run_analysis()
        user_overrides: Dict of {ticker: {assumption_key: value}} overrides.
                        Example: {"AAPL": {"wacc": 0.09, "terminal_growth_rate": 0.03}}

    Returns:
        Final state after analysis completes.
    """
    if user_overrides:
        graph.update_state(config, {"user_overrides": user_overrides})
        print(f"Applied overrides for: {list(user_overrides.keys())}")

    print("Resuming analysis...")
    print("-" * 60)

    final_state = None
    for event in graph.stream(None, config=config, stream_mode="values"):
        status = event.get("status", "")
        errors = event.get("errors", [])
        for err in errors[-2:]:
            print(f"  ⚠ {err}")
        if status == "complete":
            print("Analysis complete!")
        final_state = event

    return final_state or graph.get_state(config).values


def ask_question(
    graph: Any,
    config: dict[str, Any],
    question: str,
) -> str:
    """
    Ask a follow-up question about the completed analysis.

    Returns the assistant's answer as a string.
    """
    current_state = graph.get_state(config).values
    history = list(current_state.get("conversation_history", []))
    history.append({"role": "user", "content": question})

    graph.update_state(config, {"conversation_history": history})

    answer = ""
    for event in graph.stream(None, config=config, stream_mode="values"):
        conv = event.get("conversation_history", [])
        if conv and conv[-1]["role"] == "assistant":
            answer = conv[-1]["content"]

    return answer
