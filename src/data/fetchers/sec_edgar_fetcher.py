"""
SEC EDGAR fetcher.

Uses the free SEC EDGAR JSON API (no key required).
Provides: 10-K / 10-Q filing metadata, structured XBRL financial facts,
          and direct links to full-text filings.

Docs: https://www.sec.gov/edgar/sec-api-documentation
Rate limit: max 10 requests/second (we stay well below that).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "InvestmentAgent/1.0 biplab.ghosh2003@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}
_BASE = "https://data.sec.gov"
_COMPANY_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2000-01-01&forms=10-K"
_SUBMISSIONS = _BASE + "/submissions/CIK{cik:010d}.json"
_FACTS = _BASE + "/api/xbrl/companyfacts/CIK{cik:010d}.json"
_CONCEPT = _BASE + "/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{concept}.json"

_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"


class SecEdgarFetcher:
    """
    Fetches financial data directly from SEC EDGAR.

    Primary use cases:
      - Retrieve the most recent 10-K/10-Q filings
      - Pull structured XBRL financial facts (revenue, net income, etc.)
      - Cross-check data from yfinance/Alpha Vantage
    """

    def __init__(self):
        self._ticker_cik_map: dict[str, int] | None = None

    # ── Public entry points ────────────────────────────────────────────────────

    def get_cik(self, ticker: str) -> int:
        """Return the CIK (Central Index Key) for a given ticker symbol."""
        mapping = self._load_ticker_cik_map()
        key = ticker.upper()
        if key not in mapping:
            raise ValueError(
                f"Ticker '{ticker}' not found in SEC EDGAR. "
                "It may be a non-US company or delisted."
            )
        return mapping[key]

    def get_recent_filings(
        self,
        ticker: str,
        form_type: str = "10-K",
        count: int = 5,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of recent filings (date, accession number, URL).
        form_type: "10-K", "10-Q", "8-K", etc.
        """
        cik = self.get_cik(ticker)
        data = self._get(_SUBMISSIONS.format(cik=cik))
        filings = data.get("filings", {}).get("recent", {})

        df = pd.DataFrame(
            {
                "filed": filings.get("filingDate", []),
                "form": filings.get("form", []),
                "accession": filings.get("accessionNumber", []),
                "primary_doc": filings.get("primaryDocument", []),
            }
        )
        df = df[df["form"] == form_type].head(count).copy()
        df["url"] = df.apply(
            lambda r: (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                + r["accession"].replace("-", "")
                + "/"
                + r["primary_doc"]
            ),
            axis=1,
        )
        return df.reset_index(drop=True)

    def get_financial_facts(self, ticker: str) -> dict[str, Any]:
        """
        Return all XBRL facts for the company (large payload).
        Useful for pulling specific line items by concept name.
        """
        cik = self.get_cik(ticker)
        return self._get(_FACTS.format(cik=cik))

    def get_concept_history(
        self,
        ticker: str,
        concept: str,
        taxonomy: str = "us-gaap",
    ) -> pd.DataFrame:
        """
        Return the history of a single XBRL concept as a tidy DataFrame.

        Common concepts:
          Revenues, NetIncomeLoss, OperatingIncomeLoss,
          ResearchAndDevelopmentExpense, CommonStockSharesOutstanding,
          LongTermDebt, CashAndCashEquivalentsAtCarryingValue
        """
        cik = self.get_cik(ticker)
        data = self._get(_CONCEPT.format(cik=cik, taxonomy=taxonomy, concept=concept))

        units = data.get("units", {})
        # Most financial concepts use USD; share concepts use "shares"
        unit_key = "USD" if "USD" in units else next(iter(units), None)
        if unit_key is None:
            return pd.DataFrame()

        rows = units[unit_key]
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["end"] = pd.to_datetime(df["end"])
        # Keep only 10-K annual filings when available
        if "form" in df.columns:
            annual = df[df["form"] == "10-K"].copy()
            df = annual if not annual.empty else df

        df = df.sort_values("end").drop_duplicates(subset=["end"], keep="last")
        return df[["end", "val", "form", "filed"] if "filed" in df.columns else ["end", "val"]].reset_index(drop=True)

    def get_key_financials_xbrl(self, ticker: str) -> dict[str, pd.DataFrame]:
        """
        Pull the most important annual financial concepts in one call.
        Returns a dict of concept_name -> DataFrame(end, val).
        """
        concepts = {
            "revenue": "Revenues",
            "net_income": "NetIncomeLoss",
            "operating_income": "OperatingIncomeLoss",
            "rd_expense": "ResearchAndDevelopmentExpense",
            "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
            "shares_outstanding": "CommonStockSharesOutstanding",
            "long_term_debt": "LongTermDebt",
            "cash": "CashAndCashEquivalentsAtCarryingValue",
            "total_assets": "Assets",
            "total_equity": "StockholdersEquity",
        }
        result: dict[str, pd.DataFrame] = {}
        for label, concept in concepts.items():
            try:
                result[label] = self.get_concept_history(ticker, concept)
                time.sleep(0.15)  # polite rate limiting
            except Exception as exc:
                logger.warning("Could not fetch %s for %s: %s", concept, ticker, exc)
                result[label] = pd.DataFrame()
        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get(self, url: str) -> dict:
        response = requests.get(url, headers=_HEADERS, timeout=30)
        response.raise_for_status()
        return response.json()

    def _load_ticker_cik_map(self) -> dict[str, int]:
        if self._ticker_cik_map is not None:
            return self._ticker_cik_map
        logger.info("Loading SEC EDGAR ticker→CIK map …")
        data = self._get(_TICKER_CIK_URL)
        self._ticker_cik_map = {
            v["ticker"].upper(): int(v["cik_str"])
            for v in data.values()
        }
        logger.info("Loaded %d tickers from SEC EDGAR", len(self._ticker_cik_map))
        return self._ticker_cik_map
