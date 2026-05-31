"""
Smoke tests for data fetchers.
Run with: python -m pytest tests/ -v

These tests hit real APIs (yfinance is free, no key needed).
Alpha Vantage / FRED tests are skipped when keys aren't configured.
"""

import os
import pytest
import pandas as pd

from src.data.fetchers.yfinance_fetcher import YFinanceFetcher
from src.data.fetchers.fred_fetcher import FredFetcher
from src.data.cache.sqlite_cache import FinancialCache


TICKER = "AAPL"


# ── yfinance tests (no key needed) ────────────────────────────────────────────

class TestYFinanceFetcher:
    def setup_method(self):
        self.fetcher = YFinanceFetcher()

    def test_get_current_price(self):
        price = self.fetcher.get_current_price(TICKER)
        assert isinstance(price, float)
        assert price > 0

    def test_get_historical_prices(self):
        df = self.fetcher.get_historical_prices(TICKER, period_years=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 200
        assert "Close" in df.columns

    def test_get_key_metrics(self):
        metrics = self.fetcher.get_key_metrics(TICKER)
        assert metrics["ticker"] == TICKER
        assert "sector" in metrics
        assert "market_cap" in metrics

    def test_get_beta(self):
        beta = self.fetcher.get_beta(TICKER)
        assert beta is None or isinstance(beta, float)

    def test_calculate_beta_from_prices(self):
        beta = self.fetcher.calculate_beta_from_prices(TICKER, period_years=2)
        assert isinstance(beta, float)
        assert 0.1 < beta < 5.0  # sanity range

    def test_invalid_ticker_raises(self):
        with pytest.raises(Exception):
            self.fetcher.get_current_price("XXXINVALIDXXX")


# ── FRED tests (key optional, falls back gracefully) ──────────────────────────

class TestFredFetcher:
    def setup_method(self):
        self.fetcher = FredFetcher()

    def test_get_risk_free_rate(self):
        rate = self.fetcher.get_risk_free_rate()
        assert isinstance(rate, float)
        assert 0.001 < rate < 0.20  # sanity: between 0.1% and 20%

    def test_get_gdp_growth(self):
        gdp = self.fetcher.get_gdp_growth_rate()
        assert isinstance(gdp, float)
        assert -0.1 < gdp < 0.15

    def test_get_macro_snapshot_keys(self):
        snap = self.fetcher.get_macro_snapshot()
        for key in ("risk_free_rate", "gdp_growth_real", "inflation", "nominal_gdp_growth"):
            assert key in snap


# ── Cache tests ────────────────────────────────────────────────────────────────

class TestFinancialCache:
    def setup_method(self, tmp_path=None):
        import tempfile, pathlib
        self.tmp = pathlib.Path(tempfile.mkdtemp()) / "test_cache.db"
        self.cache = FinancialCache(db_path=self.tmp)

    def test_set_and_get_dict(self):
        data = {"price": 150.0, "name": "Apple"}
        self.cache.set("test:key", data, ttl_hours=1)
        result = self.cache.get("test:key")
        assert result == data

    def test_set_and_get_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.cache.set("test:df", df, ttl_hours=1)
        result = self.cache.get("test:df")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    def test_expired_returns_none(self):
        self.cache.set("test:expired", {"x": 1}, ttl_hours=-1)
        assert self.cache.get("test:expired") is None

    def test_stats(self):
        self.cache.set("s:1", "v1", ttl_hours=1)
        self.cache.set("s:2", "v2", ttl_hours=1)
        stats = self.cache.stats()
        assert stats["total"] >= 2
