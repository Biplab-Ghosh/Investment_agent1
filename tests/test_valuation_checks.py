"""Tests for valuation sanity checks and growth schedules."""

import pandas as pd

from src.analysis.dcf_engine import DCFEngine
from src.analysis.valuation_checks import check_dcf_valuation, reconcile_price_with_history


def test_reconcile_price_uses_close_when_quote_diverges():
    hist = pd.DataFrame({"Close": [100.0, 102.0, 105.0]})
    price, warnings = reconcile_price_with_history(200.0, hist)
    assert price == 105.0
    assert warnings


def test_check_dcf_flags_negative_iv_and_extreme_ratio():
    result = check_dcf_valuation(
        "TEST",
        {"current_price": 100.0, "intrinsic_value_per_share": -5.0},
    )
    assert "negative_iv" in result["flags"]
    assert not result["is_reliable"]

    high = check_dcf_valuation(
        "TEST",
        {"current_price": 10.0, "intrinsic_value_per_share": 50.0},
    )
    assert "iv_price_high" in high["flags"]


def test_three_stage_growth_decelerates():
    engine = DCFEngine()
    rates = engine.build_three_stage_growth_rates(0.20, "High Growth", 5, 0.025)
    assert len(rates) == 5
    assert rates[0] >= rates[-1]
