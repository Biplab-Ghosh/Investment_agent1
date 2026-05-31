"""
FRED (Federal Reserve Economic Data) fetcher.

Provides: risk-free rate (10Y Treasury), GDP growth, CPI/inflation,
          Fed Funds rate, and other macro indicators needed for WACC and DCF.

Free API — no call limits.
Register for a key at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from config.settings import FRED_API_KEY, FRED_SERIES

logger = logging.getLogger(__name__)


class FredFetcher:
    """
    Wraps the fredapi library and provides finance-ready helper methods.

    Falls back to hard-coded long-run averages when FRED is unavailable
    so the agent can still run without an internet connection.
    """

    # Long-run fallback values used when FRED is unreachable
    _FALLBACKS = {
        "risk_free_rate": 0.045,   # ~4.5% 10Y Treasury (2024 average)
        "gdp_growth": 0.025,       # ~2.5% real GDP growth (long-run US avg)
        "cpi": 0.03,               # ~3% CPI inflation
        "fed_funds": 0.053,        # ~5.3% Fed Funds (2024)
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or FRED_API_KEY
        self._fred = None  # lazy init

    # ── Public entry points ────────────────────────────────────────────────────

    def get_risk_free_rate(self) -> float:
        """
        Current 10-Year Treasury yield as a decimal (e.g. 0.045 = 4.5%).
        Used as the risk-free rate in CAPM / WACC.
        """
        try:
            series = self._get_series(FRED_SERIES["risk_free_rate"])
            rate = series.dropna().iloc[-1] / 100.0
            logger.info("10Y Treasury yield: %.4f", rate)
            return float(rate)
        except Exception as exc:
            logger.warning("FRED unavailable (%s), using fallback rate", exc)
            return self._FALLBACKS["risk_free_rate"]

    def get_gdp_growth_rate(self, years: int = 5) -> float:
        """
        Average real GDP growth over the last *years* years.
        Used as a sanity-check ceiling on terminal growth rates.
        """
        try:
            series = self._get_series(FRED_SERIES["gdp_growth"])
            recent = series.dropna().tail(years * 4)  # quarterly data
            avg = recent.mean() / 100.0
            logger.info("Avg real GDP growth (%dY): %.4f", years, avg)
            return float(avg)
        except Exception as exc:
            logger.warning("FRED unavailable (%s), using fallback GDP growth", exc)
            return self._FALLBACKS["gdp_growth"]

    def get_inflation_rate(self, years: int = 3) -> float:
        """
        Average CPI YoY change over *years* years (annualised).
        Useful for estimating nominal GDP growth = real GDP + inflation.
        """
        try:
            series = self._get_series(FRED_SERIES["cpi"])
            yoy = series.pct_change(12).dropna().tail(years * 12)
            avg = float(yoy.mean())
            logger.info("Avg CPI inflation (%dY): %.4f", years, avg)
            return avg
        except Exception as exc:
            logger.warning("FRED unavailable (%s), using fallback CPI", exc)
            return self._FALLBACKS["cpi"]

    def get_nominal_gdp_growth(self) -> float:
        """Real GDP growth + inflation — upper bound for terminal growth rate."""
        return self.get_gdp_growth_rate() + self.get_inflation_rate()

    def get_macro_snapshot(self) -> dict[str, Any]:
        """
        Return all macro inputs needed for DCF in a single dict.
        Safe — falls back gracefully if FRED is down.
        """
        return {
            "risk_free_rate": self.get_risk_free_rate(),
            "gdp_growth_real": self.get_gdp_growth_rate(),
            "inflation": self.get_inflation_rate(),
            "nominal_gdp_growth": self.get_nominal_gdp_growth(),
            "fed_funds_rate": self._safe_get_latest("fed_funds", scale=0.01),
            "snapshot_date": datetime.today().strftime("%Y-%m-%d"),
        }

    def get_series_history(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Retrieve an arbitrary FRED series by ID."""
        return self._get_series(series_id, start_date, end_date)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_fred(self):
        """Lazy-init fredapi.Fred instance."""
        if self._fred is None:
            try:
                from fredapi import Fred  # type: ignore
                if not self.api_key:
                    raise ValueError(
                        "FRED_API_KEY not set. Add it to .env file. "
                        "Register at https://fred.stlouisfed.org/docs/api/api_key.html"
                    )
                self._fred = Fred(api_key=self.api_key)
            except ImportError as exc:
                raise ImportError("fredapi not installed. Run: pip install fredapi") from exc
        return self._fred

    def _get_series(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        fred = self._get_fred()
        kwargs: dict[str, Any] = {}
        if start_date:
            kwargs["observation_start"] = start_date
        if end_date:
            kwargs["observation_end"] = end_date
        series = fred.get_series(series_id, **kwargs)
        series.index = pd.to_datetime(series.index)
        return series

    def _safe_get_latest(self, key: str, scale: float = 1.0) -> float:
        """Return latest value of a FRED_SERIES key, or fallback."""
        try:
            series = self._get_series(FRED_SERIES[key])
            return float(series.dropna().iloc[-1]) * scale
        except Exception:
            return self._FALLBACKS.get(key, 0.0)
