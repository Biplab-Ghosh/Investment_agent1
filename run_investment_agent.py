#!/usr/bin/env python3
"""
Run the AI Investment Analysis Agent for one or more tickers.

Loads OPENAI_API_KEY from config/openai_api_1.txt (or .env if present),
runs the full LangGraph pipeline without human-in-the-loop interrupts,
and saves markdown reports under output/reports/.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "reports"


def _configure_batch_defaults() -> None:
    """Faster defaults for scripted runs (spec default remains in settings.py)."""
    os.environ.setdefault("MONTE_CARLO_SIMULATIONS", "1000")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _load_api_keys() -> None:
    """Load keys from E:\\PyCharm Projects\\APIs, project .env, or config/."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from config.api_keys_loader import load_external_api_keys

    load_external_api_keys(sync_env=True)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        key_file = ROOT / "config" / "openai_api_1.txt"
        if key_file.is_file():
            raw = key_file.read_text(encoding="utf-8").strip()
            if raw.startswith("OPENAI_API_KEY="):
                raw = raw.split("=", 1)[1].strip().strip('"').strip("'")
            if raw:
                os.environ["OPENAI_API_KEY"] = raw
    if not os.getenv("OPENAI_API_KEY"):
        raise FileNotFoundError(
            "No OpenAI API key found. Add keys under E:\\PyCharm Projects\\APIs "
            "or set OPENAI_API_KEY in .env"
        )


def _ensure_project_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def run(tickers: list[str], thread_id: str | None = None) -> dict:
    _configure_batch_defaults()
    _ensure_project_on_path()
    _load_api_keys()

    from config.settings import validate_keys
    from src.agent.graph import build_graph, create_initial_state

    keys = validate_keys()
    if not keys["openai"]:
        raise RuntimeError("OPENAI_API_KEY is not configured after loading.")
    print(
        "API keys:",
        f"OpenAI={'yes' if keys['openai'] else 'no'}",
        f"AlphaVantage={'yes' if keys['alpha_vantage'] else 'no'}",
        f"FRED={'yes' if keys['fred'] else 'no'}",
    )

    tickers = [t.upper().strip() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required.")

    thread_id = thread_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    graph = build_graph(with_interrupt=False)
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = create_initial_state(tickers)

    print(f"Starting analysis for: {', '.join(tickers)}")
    print(f"Thread ID: {thread_id}")
    print("-" * 60)

    final_state = None
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        status = event.get("status", "")
        errors = event.get("errors", [])
        for err in errors[-3:]:
            print(f"  Warning: {err}")
        if status == "complete":
            print("Analysis complete.")
        final_state = event

    if final_state is None:
        final_state = graph.get_state(config).values

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = final_state.get("final_reports", {})
    saved: list[Path] = []

    for ticker, report in reports.items():
        path = OUTPUT_DIR / f"{ticker}_report.md"
        path.write_text(report, encoding="utf-8")
        saved.append(path)
        print(f"  Saved report: {path}")

    _print_summary(final_state, tickers)
    return final_state


def _print_summary(state: dict, tickers: list[str]) -> None:
    dcf = state.get("dcf_results", {})
    moat = state.get("moat_analysis", {})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ticker in tickers:
        r = dcf.get(ticker, {})
        m = moat.get(ticker, {})
        iv = r.get("intrinsic_value_per_share", 0)
        price = r.get("current_price", 0)
        upside = r.get("upside_downside_pct", 0)
        rating = m.get("rating", "N/A")
        score = m.get("total_score", 0)
        reliable = r.get("validation", {}).get("is_reliable", True)
        flag = "" if reliable else " [check data]"
        print(
            f"{ticker:6}  Price ${price:8.2f}  IV ${iv:8.2f}  "
            f"Upside {upside:+6.1%}  Moat {rating} ({score:.0f}/50){flag}"
        )

    errors = state.get("errors", [])
    if errors:
        print(f"\n{len(errors)} non-fatal error(s) recorded during run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI Investment Analysis Agent")
    parser.add_argument(
        "tickers",
        nargs="*",
        default=["MU", "TSM", "AVGO", "AMD"],
        help="Stock ticker symbols (default: MU TSM AVGO AMD)",
    )
    parser.add_argument("--thread-id", default=None, help="LangGraph thread id")
    args = parser.parse_args()
    run(args.tickers, thread_id=args.thread_id)


if __name__ == "__main__":
    main()
