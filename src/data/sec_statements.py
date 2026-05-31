"""
Build annual statement DataFrames from SEC EDGAR XBRL concept history.
Used when yfinance / Alpha Vantage statements are empty or incomplete.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.data.fetchers.sec_edgar_fetcher import SecEdgarFetcher

logger = logging.getLogger(__name__)

# XBRL concept -> income statement row label
_INCOME_MAP = {
    "revenue": "Total Revenue",
    "operating_income": "Operating Income",
    "net_income": "Net Income",
}
_BALANCE_MAP = {
    "total_assets": "Total Assets",
    "total_equity": "Stockholders Equity",
    "cash": "Cash And Cash Equivalents",
    "long_term_debt": "Long Term Debt",
}
_CASHFLOW_MAP = {
    "capex": "Capital Expenditure",
}


def _concept_to_row(xbrl: dict[str, pd.DataFrame], label_map: dict[str, str]) -> pd.DataFrame | None:
    rows: dict[Any, pd.Series] = {}
    for concept_key, row_name in label_map.items():
        df = xbrl.get(concept_key)
        if df is None or df.empty or "end" not in df.columns or "val" not in df.columns:
            continue
        series = df.set_index("end")["val"].sort_index()
        series.index = pd.to_datetime(series.index)
        # Annual: keep last filing per calendar year
        annual = series.groupby(series.index.year).last()
        annual.index = pd.to_datetime([f"{y}-12-31" for y in annual.index])
        rows[row_name] = annual

    if not rows:
        return None
    return pd.DataFrame(rows).T


def statements_from_sec_edgar(
    ticker: str,
    sec: SecEdgarFetcher | None = None,
) -> dict[str, Any]:
    """
    Fetch XBRL facts and return partial income/balance/cashflow DataFrames.
    """
    sec = sec or SecEdgarFetcher()
    try:
        xbrl = sec.get_key_financials_xbrl(ticker)
    except Exception as exc:
        logger.warning("SEC EDGAR XBRL fetch failed for %s: %s", ticker, exc)
        return {}

    income = _concept_to_row(xbrl, _INCOME_MAP)
    balance = _concept_to_row(xbrl, _BALANCE_MAP)
    cashflow = _concept_to_row(xbrl, _CASHFLOW_MAP)

    out: dict[str, Any] = {"sec_xbrl": xbrl}
    if income is not None and not income.empty:
        out["income_statement"] = income
    if balance is not None and not balance.empty:
        out["balance_sheet"] = balance
    if cashflow is not None and not cashflow.empty:
        out["cash_flow"] = cashflow

    shares_df = xbrl.get("shares_outstanding")
    if shares_df is not None and not shares_df.empty:
        out["shares_outstanding"] = float(shares_df["val"].iloc[-1])

    return out


def statements_empty(df: Any) -> bool:
    return df is None or (hasattr(df, "empty") and df.empty)
