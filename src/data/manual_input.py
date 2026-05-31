"""
Manual data input — CSV uploads for assumptions and financial statement overrides.

Supported files (place under data/manual/ or pass an explicit path):

1) Assumptions override — assumptions_overrides.csv
   ticker,wacc,terminal_growth_rate,revenue_growth_y1,revenue_growth_y2,...

2) Financial line items — financials_manual.csv
   ticker,statement,line_item,period,value
   (statement: income | balance | cashflow; period: YYYY-MM-DD or YYYY)

3) Projections — projections_manual.csv
   ticker,year,revenue,operating_income,capex,...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MANUAL_DIR = Path(__file__).resolve().parents[2] / "data" / "manual"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_assumption_overrides(
    ticker: str,
    manual_dir: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    """Return DCF assumption overrides for *ticker* from CSV."""
    path = csv_path or (manual_dir or DEFAULT_MANUAL_DIR) / "assumptions_overrides.csv"
    df = _read_csv(path)
    if df.empty or "ticker" not in df.columns:
        return {}

    row = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
    if row.empty:
        return {}

    r = row.iloc[0]
    overrides: dict[str, Any] = {}
    col_map = {
        "wacc": "wacc",
        "terminal_growth_rate": "terminal_growth_rate",
        "operating_margin_target": "operating_margin_target",
        "capex_percent_revenue": "capex_percent_revenue",
        "nwc_percent_revenue": "nwc_percent_revenue",
        "tax_rate": "tax_rate",
        "model_type": "model_type",
    }
    for col, key in col_map.items():
        if col in r.index and pd.notna(r[col]):
            overrides[key] = r[col]

    growth_cols = [c for c in df.columns if str(c).startswith("revenue_growth_y")]
    if growth_cols:
        rates = []
        for c in sorted(growth_cols, key=lambda x: int(str(x).replace("revenue_growth_y", "") or 0)):
            val = r.get(c)
            if pd.notna(val):
                rates.append(float(val))
        if rates:
            overrides["revenue_growth_rates"] = rates

    return overrides


def _pivot_financials(df: pd.DataFrame, ticker: str) -> dict[str, pd.DataFrame]:
    """Build statement DataFrames (index=line_item, columns=period) for one ticker."""
    sub = df[df["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if sub.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    for statement in sub["statement"].astype(str).str.lower().unique():
        stmt_df = sub[sub["statement"].astype(str).str.lower() == statement]
        wide = stmt_df.pivot_table(
            index="line_item",
            columns="period",
            values="value",
            aggfunc="last",
        )
        wide.columns = pd.to_datetime(wide.columns, errors="coerce")
        wide = wide.sort_index(axis=1)
        key = {
            "income": "income_statement",
            "balance": "balance_sheet",
            "cashflow": "cash_flow",
            "cash_flow": "cash_flow",
        }.get(statement, statement)
        result[key] = wide
    return result


def load_financial_overrides(
    ticker: str,
    manual_dir: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    path = csv_path or (manual_dir or DEFAULT_MANUAL_DIR) / "financials_manual.csv"
    df = _read_csv(path)
    required = {"ticker", "statement", "line_item", "period", "value"}
    if df.empty or not required.issubset(df.columns):
        return {}
    return _pivot_financials(df, ticker)


def merge_manual_into_company_data(
    company_data: dict[str, Any],
    manual_dir: Path | None = None,
    financials_path: Path | None = None,
) -> dict[str, Any]:
    """Overlay manual statement rows onto company_data when provided."""
    ticker = company_data.get("ticker", "")
    overrides = load_financial_overrides(ticker, manual_dir=manual_dir, csv_path=financials_path)
    if not overrides:
        return company_data

    merged = {**company_data, "data_sources": list(company_data.get("data_sources", []))}
    if "manual_csv" not in merged["data_sources"]:
        merged["data_sources"].append("manual_csv")

    for key in ("income_statement", "balance_sheet", "cash_flow"):
        if key in overrides and not overrides[key].empty:
            existing = merged.get(key)
            if existing is None or (hasattr(existing, "empty") and existing.empty):
                merged[key] = overrides[key]
            else:
                # Manual rows take precedence where provided
                combined = existing.combine_first(overrides[key])
                merged[key] = combined
            logger.info("Applied manual %s for %s", key, ticker)

    return merged
