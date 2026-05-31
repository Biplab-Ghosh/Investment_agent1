"""
Valuation and market-data sanity checks.

Flags implausible prices, negative intrinsic values, and extreme IV/price ratios
so reports and logs surface data-quality issues early.
"""

from __future__ import annotations

from typing import Any


# IV/price outside this band triggers a warning (spec: catch unrealistic DCF output)
IV_PRICE_RATIO_WARN_HIGH = 3.0
IV_PRICE_RATIO_WARN_LOW = 1.0 / 3.0


def reconcile_price_with_history(
    current_price: float | None,
    historical_prices: Any,
    max_deviation: float = 0.25,
) -> tuple[float | None, list[str]]:
    """
    Prefer the latest adjusted close when yfinance *info* price diverges sharply.
    """
    warnings: list[str] = []
    if historical_prices is None or getattr(historical_prices, "empty", True):
        return current_price, warnings

    try:
        close_col = historical_prices["Close"]
        if hasattr(close_col, "ndim") and close_col.ndim > 1:
            close_col = close_col.iloc[:, 0]
        last_close = float(close_col.iloc[-1])
    except (TypeError, ValueError, IndexError, KeyError):
        return current_price, warnings

    if last_close <= 0:
        return current_price, warnings

    if current_price is None or current_price <= 0:
        warnings.append(f"Using historical close ${last_close:.2f} (no valid quote in info)")
        return last_close, warnings

    deviation = abs(current_price - last_close) / last_close
    if deviation > max_deviation:
        warnings.append(
            f"Quote ${current_price:.2f} deviates {deviation:.0%} from last close "
            f"${last_close:.2f}; using close for valuation"
        )
        return last_close, warnings

    return float(current_price), warnings


def check_dcf_valuation(
    ticker: str,
    dcf_results: dict[str, Any],
    company_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run sanity checks on DCF output. Returns a dict with warnings and severity flags.
    """
    company_data = company_data or {}
    price = float(dcf_results.get("current_price") or 0)
    iv = float(dcf_results.get("intrinsic_value_per_share") or 0)
    warnings: list[str] = []
    flags: list[str] = []

    key_metrics = company_data.get("key_metrics") or {}
    low_52 = key_metrics.get("52w_low")
    high_52 = key_metrics.get("52w_high")

    if price <= 0:
        warnings.append(f"{ticker}: missing or non-positive market price")
        flags.append("invalid_price")

    if iv < 0:
        warnings.append(f"{ticker}: negative intrinsic value (${iv:.2f}) — check FCF inputs")
        flags.append("negative_iv")

    if price > 0 and iv > 0:
        ratio = iv / price
        if ratio > IV_PRICE_RATIO_WARN_HIGH:
            warnings.append(
                f"{ticker}: IV/price ratio {ratio:.1f}x exceeds {IV_PRICE_RATIO_WARN_HIGH}x "
                "(possible data or assumption error)"
            )
            flags.append("iv_price_high")
        elif ratio < IV_PRICE_RATIO_WARN_LOW:
            warnings.append(
                f"{ticker}: IV/price ratio {ratio:.2f}x below {IV_PRICE_RATIO_WARN_LOW:.2f}x "
                "(possible data or assumption error)"
            )
            flags.append("iv_price_low")

    if price > 0 and low_52 and high_52:
        try:
            low_f, high_f = float(low_52), float(high_52)
            if price < low_f * 0.5 or price > high_f * 2.0:
                warnings.append(
                    f"{ticker}: price ${price:.2f} outside plausible 52-week range "
                    f"${low_f:.2f}–${high_f:.2f}"
                )
                flags.append("price_outside_52w")
        except (TypeError, ValueError):
            pass

    shares = float(company_data.get("shares_outstanding") or 0)
    market_cap = float(company_data.get("market_cap") or 0)
    if shares > 0 and price > 0 and market_cap > 0:
        implied_cap = shares * price
        if abs(implied_cap - market_cap) / market_cap > 0.5:
            warnings.append(
                f"{ticker}: shares×price (${implied_cap/1e9:.1f}B) inconsistent with "
                f"market cap (${market_cap/1e9:.1f}B)"
            )
            flags.append("market_cap_mismatch")

    return {
        "ticker": ticker,
        "warnings": warnings,
        "flags": flags,
        "is_reliable": len(flags) == 0,
        "iv_price_ratio": (iv / price) if price > 0 and iv != 0 else None,
    }
