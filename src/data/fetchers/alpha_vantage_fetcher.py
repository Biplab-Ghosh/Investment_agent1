"""
Alpha Vantage data fetcher.

Free tier: 500 API calls/day, 5 calls/minute.
Provides detailed income statement, balance sheet, cash flow, and earnings data.

Register for a free key at: https://www.alphavantage.co/support/#api-key
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

from config.settings import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_BASE_URL

logger = logging.getLogger(__name__)

# Rate-limit guard: free tier allows 5 requests/minute
_RATE_LIMIT_DELAY = 12.5  # seconds between calls on free tier


class AlphaVantageFetcher:
    """
    Fetches structured financial statements from Alpha Vantage.

    All methods return tidy pandas DataFrames with years as columns
    and line items as the index.
    """

    def __init__(self, api_key: str | None = None, respect_rate_limit: bool = True):
        self.api_key = api_key or ALPHA_VANTAGE_API_KEY
        self.respect_rate_limit = respect_rate_limit
        self._last_call: float = 0.0

        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key not set. "
                "Add ALPHA_VANTAGE_API_KEY to your .env file."
            )

    # ── Public entry points ────────────────────────────────────────────────────

    def get_income_statement(self, ticker: str, annual: bool = True) -> pd.DataFrame:
        """Annual (or quarterly) income statement."""
        func = "INCOME_STATEMENT"
        data = self._fetch(func, ticker)
        key = "annualReports" if annual else "quarterlyReports"
        return self._reports_to_df(data.get(key, []))

    def get_balance_sheet(self, ticker: str, annual: bool = True) -> pd.DataFrame:
        """Annual (or quarterly) balance sheet."""
        func = "BALANCE_SHEET"
        data = self._fetch(func, ticker)
        key = "annualReports" if annual else "quarterlyReports"
        return self._reports_to_df(data.get(key, []))

    def get_cash_flow(self, ticker: str, annual: bool = True) -> pd.DataFrame:
        """Annual (or quarterly) cash flow statement."""
        func = "CASH_FLOW"
        data = self._fetch(func, ticker)
        key = "annualReports" if annual else "quarterlyReports"
        return self._reports_to_df(data.get(key, []))

    def get_earnings(self, ticker: str) -> dict[str, pd.DataFrame]:
        """Annual and quarterly EPS data."""
        data = self._fetch("EARNINGS", ticker)
        return {
            "annual": self._reports_to_df(data.get("annualEarnings", [])),
            "quarterly": self._reports_to_df(data.get("quarterlyEarnings", [])),
        }

    def get_company_overview(self, ticker: str) -> dict[str, Any]:
        """Company metadata and key ratios from Alpha Vantage OVERVIEW endpoint."""
        raw = self._fetch("OVERVIEW", ticker)
        numeric_fields = {
            "MarketCapitalization", "EBITDA", "PERatio", "PEGRatio",
            "BookValue", "DividendPerShare", "DividendYield", "EPS",
            "RevenuePerShareTTM", "ProfitMargin", "OperatingMarginTTM",
            "ReturnOnAssetsTTM", "ReturnOnEquityTTM", "RevenueTTM",
            "GrossProfitTTM", "DilutedEPSTTM", "QuarterlyEarningsGrowthYOY",
            "QuarterlyRevenueGrowthYOY", "TrailingPE", "ForwardPE",
            "PriceToSalesRatioTTM", "PriceToBookRatio", "EVToRevenue",
            "EVToEBITDA", "Beta", "SharesOutstanding", "52WeekHigh", "52WeekLow",
        }
        result: dict[str, Any] = {}
        for k, v in raw.items():
            if k in numeric_fields:
                try:
                    result[k] = float(v)
                except (ValueError, TypeError):
                    result[k] = None
            else:
                result[k] = v
        return result

    def get_full_financials(self, ticker: str) -> dict[str, Any]:
        """Fetch all three statements + overview in sequence (rate-limited)."""
        logger.info("Fetching full Alpha Vantage financials for %s", ticker)
        return {
            "income_statement": self.get_income_statement(ticker),
            "balance_sheet": self.get_balance_sheet(ticker),
            "cash_flow": self.get_cash_flow(ticker),
            "overview": self.get_company_overview(ticker),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _fetch(self, function: str, symbol: str) -> dict[str, Any]:
        """Execute a single Alpha Vantage API call with optional rate limiting."""
        self._rate_limit()
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data: dict = response.json()

        if "Note" in data:
            raise RuntimeError(
                f"Alpha Vantage rate limit hit: {data['Note']}"
            )
        if "Information" in data:
            raise RuntimeError(
                f"Alpha Vantage API message: {data['Information']}"
            )
        if "Error Message" in data:
            raise ValueError(
                f"Alpha Vantage error for {symbol}/{function}: {data['Error Message']}"
            )

        logger.debug("AV %s/%s OK (%d top-level keys)", function, symbol, len(data))
        return data

    def _rate_limit(self) -> None:
        if not self.respect_rate_limit:
            return
        elapsed = time.time() - self._last_call
        if elapsed < _RATE_LIMIT_DELAY:
            sleep_time = _RATE_LIMIT_DELAY - elapsed
            logger.debug("Rate limiting: sleeping %.1fs", sleep_time)
            time.sleep(sleep_time)
        self._last_call = time.time()

    @staticmethod
    def _reports_to_df(reports: list[dict]) -> pd.DataFrame:
        """
        Convert a list of period report dicts into a DataFrame.
        Index = fiscal date, columns = line items, values cast to float where possible.
        """
        if not reports:
            return pd.DataFrame()

        df = pd.DataFrame(reports)
        date_col = "fiscalDateEnding" if "fiscalDateEnding" in df.columns else df.columns[0]
        df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=False)

        # Cast numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
