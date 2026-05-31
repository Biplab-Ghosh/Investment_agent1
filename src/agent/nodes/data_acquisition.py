"""
Node 1 — Data Acquisition

Fetches company financial data and macro data for all requested tickers.
Uses DataManager which applies the fallback hierarchy: yfinance → Alpha Vantage → SEC EDGAR.
Results are cached in SQLite to avoid redundant API calls.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from src.agent.state import InvestmentAnalysisState
from src.data.data_manager import DataManager

logger = logging.getLogger(__name__)

_dm: DataManager | None = None


def _get_data_manager() -> DataManager:
    global _dm
    if _dm is None:
        _dm = DataManager()
    return _dm


def data_acquisition_node(state: InvestmentAnalysisState) -> dict[str, Any]:
    """
    Fetch financial data for all tickers in state.ticker_symbols.
    Populates state.financial_data and state.market_data.
    """
    tickers = state.get("ticker_symbols", [])
    manual_paths = state.get("manual_data_paths", {})
    if not tickers:
        return {
            "errors": ["No ticker symbols provided"],
            "status": "error",
        }

    dm = _get_data_manager()
    errors: list[str] = list(state.get("errors", []))
    financial_data: dict[str, Any] = {}

    logger.info("Data acquisition started for: %s", tickers)

    for ticker in tickers:
        ticker = ticker.upper().strip()
        try:
            if not dm.validate_ticker(ticker):
                errors.append(f"Ticker '{ticker}' not found or has no price data")
                logger.warning("Invalid ticker: %s", ticker)
                continue

            company_data = dm.get_company_data(
                ticker,
                manual_financials_path=manual_paths.get(ticker),
            )
            financial_data[ticker] = company_data
            for w in company_data.get("data_warnings", []):
                errors.append(f"{ticker} data: {w}")
            logger.info("Fetched data for %s: price=$%.2f, sector=%s",
                        ticker,
                        company_data.get("current_price", 0),
                        company_data.get("sector", "Unknown"))

        except Exception as exc:
            errors.append(f"Error fetching data for {ticker}: {exc}")
            logger.error("Failed to fetch %s: %s", ticker, exc, exc_info=True)

    if not financial_data:
        return {
            "errors": errors,
            "status": "error",
            "financial_data": {},
        }

    # ── Macro data ──────────────────────────────────────────────────────────
    try:
        macro_data = dm.get_macro_data()
        logger.info("Macro data fetched: risk-free rate=%.2f%%", macro_data.get("risk_free_rate", 0) * 100)
    except Exception as exc:
        errors.append(f"Macro data fetch failed: {exc}")
        macro_data = {
            "risk_free_rate": 0.045,
            "gdp_growth_real": 0.025,
            "inflation": 0.03,
            "nominal_gdp_growth": 0.055,
        }
        logger.warning("Using fallback macro data: %s", exc)

    return {
        "financial_data": financial_data,
        "market_data": macro_data,
        "analysis_date": date.today().isoformat(),
        "current_ticker": tickers[0] if tickers else None,
        "errors": errors,
        "status": "running",
    }
