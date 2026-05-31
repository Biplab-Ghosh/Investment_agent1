"""
DataManager — single entry point for all data acquisition.

Implements the fallback hierarchy from the spec:
  manual CSV (if provided) → yfinance → Alpha Vantage → SEC EDGAR

All results are cached in SQLite to minimise API calls.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import FUNDAMENTAL_DATA_TTL_DAYS
from src.analysis.valuation_checks import reconcile_price_with_history
from src.data.cache.sqlite_cache import FinancialCache
from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
from src.data.fetchers.alpha_vantage_fetcher import AlphaVantageFetcher
from src.data.fetchers.fred_fetcher import FredFetcher
from src.data.fetchers.sec_edgar_fetcher import SecEdgarFetcher
from src.data.manual_input import (
    DEFAULT_MANUAL_DIR,
    load_assumption_overrides,
    merge_manual_into_company_data,
)
from src.data.sec_statements import statements_empty, statements_from_sec_edgar

logger = logging.getLogger(__name__)


class DataManager:
    """
    Orchestrates data fetching with caching and fallback logic.

    Usage:
        dm = DataManager()
        data = dm.get_company_data("AAPL")
        macro = dm.get_macro_data()
        overrides = dm.get_manual_assumption_overrides("AAPL")
    """

    def __init__(
        self,
        cache: FinancialCache | None = None,
        manual_dir: Path | str | None = None,
    ):
        self.cache = cache or FinancialCache()
        self.yf = YFinanceFetcher()
        self.fred = FredFetcher()
        self._av: AlphaVantageFetcher | None = None
        self._sec: SecEdgarFetcher | None = None
        self.manual_dir = Path(manual_dir) if manual_dir else DEFAULT_MANUAL_DIR

    # ── Primary public methods ─────────────────────────────────────────────────

    def get_company_data(
        self,
        ticker: str,
        force_refresh: bool = False,
        manual_financials_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Return a comprehensive data dict for *ticker*.

        Keys:
          ticker, name, sector, industry,
          current_price, shares_outstanding, market_cap, beta,
          key_metrics, income_statement, balance_sheet, cash_flow,
          historical_prices, data_sources, data_warnings
        """
        cache_key = f"dm:{ticker}:company_data"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info("Cache hit for %s company data", ticker)
                return cached

        logger.info("Fetching company data for %s …", ticker)
        ticker = ticker.upper()
        data_sources: list[str] = ["yfinance"]
        data_warnings: list[str] = []

        key_metrics = self.yf.get_key_metrics(ticker)
        yf_fundamentals = self.yf.get_fundamentals(ticker)
        income = yf_fundamentals["income_statement"]
        balance = yf_fundamentals["balance_sheet"]
        cashflow = yf_fundamentals["cash_flow"]

        if self._av_available():
            try:
                av_data = self._av_fetcher().get_full_financials(ticker)
                income = av_data["income_statement"]
                balance = av_data["balance_sheet"]
                cashflow = av_data["cash_flow"]
                data_sources.append("alpha_vantage")
                logger.info("Using Alpha Vantage statements for %s", ticker)
            except Exception as exc:
                logger.warning("Alpha Vantage failed for %s (%s), using yfinance", ticker, exc)

        if (
            statements_empty(income)
            or statements_empty(balance)
        ):
            try:
                sec_data = statements_from_sec_edgar(ticker, self._sec_fetcher())
                if sec_data:
                    data_sources.append("sec_edgar")
                    if statements_empty(income) and sec_data.get("income_statement") is not None:
                        income = sec_data["income_statement"]
                    if statements_empty(balance) and sec_data.get("balance_sheet") is not None:
                        balance = sec_data["balance_sheet"]
                    if statements_empty(cashflow) and sec_data.get("cash_flow") is not None:
                        cashflow = sec_data["cash_flow"]
                    if sec_data.get("shares_outstanding"):
                        key_metrics["shares_outstanding"] = sec_data["shares_outstanding"]
                    logger.info("Supplemented statements from SEC EDGAR for %s", ticker)
            except Exception as exc:
                logger.warning("SEC EDGAR fallback failed for %s: %s", ticker, exc)
                data_warnings.append(f"SEC EDGAR fallback failed: {exc}")

        prices = self.yf.get_historical_prices(ticker, period_years=5)

        beta = key_metrics.get("beta")
        if beta is None:
            try:
                beta = self.yf.calculate_beta_from_prices(ticker)
            except Exception:
                beta = 1.0
                logger.warning("Could not compute beta for %s, defaulting to 1.0", ticker)

        current_price, price_warnings = reconcile_price_with_history(
            key_metrics.get("current_price"),
            prices,
        )
        data_warnings.extend(price_warnings)
        if current_price is not None:
            key_metrics["current_price"] = current_price

        try:
            sec_filings = self._sec_fetcher().get_recent_filings(ticker, form_type="10-K", count=3)
        except Exception:
            sec_filings = pd.DataFrame()

        result: dict[str, Any] = {
            "ticker": ticker,
            "name": key_metrics.get("name", ticker),
            "sector": key_metrics.get("sector"),
            "industry": key_metrics.get("industry"),
            "current_price": current_price,
            "shares_outstanding": key_metrics.get("shares_outstanding"),
            "market_cap": key_metrics.get("market_cap"),
            "beta": beta,
            "key_metrics": key_metrics,
            "income_statement": income,
            "balance_sheet": balance,
            "cash_flow": cashflow,
            "historical_prices": prices,
            "sec_filings": sec_filings,
            "data_sources": data_sources,
            "data_warnings": data_warnings,
        }

        fin_path = Path(manual_financials_path) if manual_financials_path else None
        result = merge_manual_into_company_data(
            result,
            manual_dir=self.manual_dir,
            financials_path=fin_path,
        )

        self.cache.set(cache_key, result, ttl_days=FUNDAMENTAL_DATA_TTL_DAYS)
        logger.info("Company data for %s cached (sources: %s).", ticker, result.get("data_sources"))
        return result

    def get_manual_assumption_overrides(self, ticker: str) -> dict[str, Any]:
        """DCF assumption overrides from data/manual/assumptions_overrides.csv."""
        return load_assumption_overrides(ticker.upper(), manual_dir=self.manual_dir)

    def get_macro_data(self, force_refresh: bool = False) -> dict[str, Any]:
        """Return current macro indicators (risk-free rate, GDP growth, inflation)."""
        if not force_refresh:
            cached = self.cache.get_macro()
            if cached is not None:
                logger.info("Cache hit for macro data")
                return cached

        macro = self.fred.get_macro_snapshot()
        macro["fred_configured"] = bool(self.fred.api_key)
        self.cache.set_macro(macro)
        return macro

    def get_multi_company_data(
        self,
        tickers: list[str],
        force_refresh: bool = False,
        manual_paths: dict[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        manual_paths = manual_paths or {}
        return {
            t: self.get_company_data(
                t,
                force_refresh=force_refresh,
                manual_financials_path=manual_paths.get(t.upper()) or manual_paths.get(t),
            )
            for t in tickers
        }

    def get_sec_filings(self, ticker: str, form_type: str = "10-K") -> pd.DataFrame:
        """Return recent SEC filings for *ticker*."""
        return self._sec_fetcher().get_recent_filings(ticker, form_type=form_type)

    def validate_ticker(self, ticker: str) -> bool:
        try:
            price = self.yf.get_current_price(ticker)
            return price > 0
        except Exception:
            return False

    def _av_available(self) -> bool:
        from config.settings import ALPHA_VANTAGE_API_KEY
        return bool(ALPHA_VANTAGE_API_KEY)

    def _av_fetcher(self) -> AlphaVantageFetcher:
        if self._av is None:
            self._av = AlphaVantageFetcher()
        return self._av

    def _sec_fetcher(self) -> SecEdgarFetcher:
        if self._sec is None:
            self._sec = SecEdgarFetcher()
        return self._sec
