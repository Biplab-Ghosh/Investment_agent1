"""
Yahoo Finance data fetcher via yfinance.

Covers: historical prices, basic income/balance/cashflow statements,
        shares outstanding, current price, beta.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceFetcher:
    """Thin wrapper around yfinance with standardised column names."""

    # ── Public entry points ────────────────────────────────────────────────────

    def get_historical_prices(
        self,
        ticker: str,
        period_years: int = 5,
    ) -> pd.DataFrame:
        """Return OHLCV data for the given ticker, last *period_years* years."""
        end = datetime.today()
        start = end - timedelta(days=period_years * 365)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(f"No price data returned for {ticker}")
        df.index = pd.to_datetime(df.index)
        logger.info("Fetched %d price rows for %s", len(df), ticker)
        return df

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """
        Return a dict with:
          income_statement, balance_sheet, cash_flow  (annual DataFrames)
          info                                         (key statistics dict)
        """
        t = yf.Ticker(ticker)

        income = t.financials          # annual income statement
        balance = t.balance_sheet      # annual balance sheet
        cashflow = t.cashflow          # annual cash flow

        for name, df in [("income", income), ("balance", balance), ("cashflow", cashflow)]:
            if df is None or df.empty:
                logger.warning("Empty %s statement for %s", name, ticker)

        info = t.info or {}
        logger.info("Fetched fundamentals for %s", ticker)

        return {
            "income_statement": income,
            "balance_sheet": balance,
            "cash_flow": cashflow,
            "info": info,
        }

    def get_current_price(self, ticker: str) -> float:
        """Return the most recent closing price."""
        t = yf.Ticker(ticker)
        price = (t.info or {}).get("currentPrice") or (t.info or {}).get("regularMarketPrice")
        if price is None:
            hist = t.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if price is None:
            raise ValueError(f"Cannot determine current price for {ticker}")
        return float(price)

    def get_shares_outstanding(self, ticker: str) -> float:
        """Return shares outstanding (basic)."""
        info = yf.Ticker(ticker).info or {}
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if shares is None:
            raise ValueError(f"Cannot determine shares outstanding for {ticker}")
        return float(shares)

    def get_beta(self, ticker: str) -> float | None:
        """Return 5-year monthly beta vs S&P 500 (yfinance reported)."""
        return (yf.Ticker(ticker).info or {}).get("beta")

    def get_market_cap(self, ticker: str) -> float | None:
        return (yf.Ticker(ticker).info or {}).get("marketCap")

    # ── Derived helpers ────────────────────────────────────────────────────────

    def get_key_metrics(self, ticker: str) -> dict[str, Any]:
        """Collect the most-used scalar metrics in one call."""
        info = yf.Ticker(ticker).info or {}
        return {
            "ticker": ticker.upper(),
            "name": info.get("longName", ticker),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "beta": info.get("beta"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "free_cashflow": info.get("freeCashflow"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }

    def calculate_beta_from_prices(
        self,
        ticker: str,
        market_ticker: str = "^GSPC",
        period_years: int = 5,
    ) -> float:
        """Compute beta via OLS regression of monthly returns vs market."""
        import numpy as np

        stock_prices = self.get_historical_prices(ticker, period_years)
        market_prices = self.get_historical_prices(market_ticker, period_years)

        # Monthly resample
        stock_monthly = stock_prices["Close"].resample("ME").last().pct_change().dropna()
        market_monthly = market_prices["Close"].resample("ME").last().pct_change().dropna()

        aligned = pd.concat(
            [stock_monthly.rename("stock"), market_monthly.rename("market")], axis=1
        ).dropna()

        if len(aligned) < 12:
            raise ValueError(f"Insufficient data to calculate beta for {ticker}")

        cov = np.cov(aligned["stock"], aligned["market"])
        beta = cov[0, 1] / cov[1, 1]
        logger.info("Calculated beta for %s: %.3f", ticker, beta)
        return float(beta)
