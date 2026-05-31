"""
Moat scoring engine — quantitative component of the Morningstar/Buffett moat framework.

Scores five moat dimensions (0-10 each) from financial data,
then derives an overall Wide / Narrow / None rating using ROIC validation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Industry average benchmarks ───────────────────────────────────────────────
_INDUSTRY_BENCHMARKS: dict[str, dict[str, float]] = {
    "Technology": {"avg_gross_margin": 0.55, "avg_roic": 0.18, "avg_pe": 28},
    "Healthcare": {"avg_gross_margin": 0.60, "avg_roic": 0.15, "avg_pe": 22},
    "Consumer Cyclical": {"avg_gross_margin": 0.35, "avg_roic": 0.12, "avg_pe": 20},
    "Consumer Defensive": {"avg_gross_margin": 0.40, "avg_roic": 0.14, "avg_pe": 22},
    "Financial Services": {"avg_gross_margin": 0.45, "avg_roic": 0.10, "avg_pe": 15},
    "Industrials": {"avg_gross_margin": 0.35, "avg_roic": 0.12, "avg_pe": 20},
    "Energy": {"avg_gross_margin": 0.30, "avg_roic": 0.08, "avg_pe": 14},
    "Materials": {"avg_gross_margin": 0.28, "avg_roic": 0.10, "avg_pe": 16},
    "Utilities": {"avg_gross_margin": 0.30, "avg_roic": 0.08, "avg_pe": 18},
    "Real Estate": {"avg_gross_margin": 0.50, "avg_roic": 0.08, "avg_pe": 20},
    "Communication Services": {"avg_gross_margin": 0.45, "avg_roic": 0.14, "avg_pe": 22},
}
_DEFAULT_BENCHMARKS = {"avg_gross_margin": 0.40, "avg_roic": 0.12, "avg_pe": 18}


def _safe(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


class MoatEngine:
    """
    Quantitative moat scoring engine.

    Scores each of the five Morningstar moat dimensions on a 0-10 scale
    using readily available financial metrics, then synthesises an overall rating.
    """

    def get_industry_benchmarks(self, sector: str) -> dict[str, float]:
        return _INDUSTRY_BENCHMARKS.get(sector, _DEFAULT_BENCHMARKS)

    # ── Dimension 1: Intangible Assets ────────────────────────────────────────

    def score_intangible_assets(
        self,
        company_data: dict[str, Any],
        industry_benchmarks: dict[str, float],
    ) -> dict[str, Any]:
        """
        Proxy: gross margin premium over peers, R&D intensity, revenue CAGR
        (as a proxy for brand/pricing power).
        Scale: 0-10.
        """
        score = 0.0
        details: dict[str, str] = {}

        key_metrics = company_data.get("key_metrics", {})
        sector = company_data.get("sector", "Unknown")

        # Gross margin vs industry (+4 if >10% above industry avg)
        gross_margin = _safe(key_metrics.get("gross_margins"))
        industry_gm = industry_benchmarks.get("avg_gross_margin", 0.40)
        if gross_margin > 0:
            premium = gross_margin - industry_gm
            if premium > 0.20:
                score += 4
                details["brand_premium"] = f"Gross margin {gross_margin:.1%} is {premium:.1%} above industry — strong pricing power"
            elif premium > 0.10:
                score += 3
                details["brand_premium"] = f"Gross margin {gross_margin:.1%} is {premium:.1%} above industry — good pricing power"
            elif premium > 0.0:
                score += 1.5
                details["brand_premium"] = f"Gross margin {gross_margin:.1%} slightly above industry average"
            else:
                details["brand_premium"] = f"Gross margin {gross_margin:.1%} below or at industry average — limited pricing power"

        # R&D / Revenue as IP proxy (+3 if >8%, +1.5 if >3%)
        revenue_growth = _safe(key_metrics.get("revenue_growth"))
        # Revenue CAGR proxy for brand strength
        if revenue_growth > 0.20:
            score += 3
            details["growth_proxy"] = f"Revenue growth {revenue_growth:.1%} — strong demand momentum"
        elif revenue_growth > 0.10:
            score += 2
            details["growth_proxy"] = f"Revenue growth {revenue_growth:.1%} — healthy demand"
        elif revenue_growth > 0.05:
            score += 1
            details["growth_proxy"] = f"Revenue growth {revenue_growth:.1%} — moderate demand"
        else:
            details["growth_proxy"] = "Revenue growth below 5% — weak demand or mature business"

        # Operating margin stability proxy for intangibles (+3 if operating margin >20%)
        op_margin = _safe(key_metrics.get("operating_margins"))
        if op_margin > 0.30:
            score += 3
            details["margin_quality"] = f"Operating margin {op_margin:.1%} — exceptional profitability suggests strong intangibles"
        elif op_margin > 0.20:
            score += 2
            details["margin_quality"] = f"Operating margin {op_margin:.1%} — above-average profitability"
        elif op_margin > 0.10:
            score += 1
            details["margin_quality"] = f"Operating margin {op_margin:.1%} — reasonable profitability"
        else:
            details["margin_quality"] = f"Operating margin {op_margin:.1%} — limited margin suggests weak intangibles"

        score = min(score, 10)
        return {"score": round(score, 1), "details": details, "max": 10}

    # ── Dimension 2: Switching Costs ──────────────────────────────────────────

    def score_switching_costs(
        self,
        company_data: dict[str, Any],
        industry_benchmarks: dict[str, float],
    ) -> dict[str, Any]:
        """
        Proxy: profit margin stability, debt-to-equity (high debt ok if sticky),
        revenue consistency. Direct metrics (churn, NRR) unavailable from free APIs.
        Scale: 0-10.
        """
        score = 0.0
        details: dict[str, str] = {}
        key_metrics = company_data.get("key_metrics", {})
        sector = company_data.get("sector", "Unknown")

        # Profit margin consistency (+3 if high, stable profit margins)
        profit_margin = _safe(key_metrics.get("profit_margins"))
        if profit_margin > 0.20:
            score += 3
            details["margin_stability"] = f"Net margin {profit_margin:.1%} — high profitability suggests sticky customers"
        elif profit_margin > 0.10:
            score += 2
            details["margin_stability"] = f"Net margin {profit_margin:.1%} — reasonable customer retention implied"
        elif profit_margin > 0.05:
            score += 1
            details["margin_stability"] = f"Net margin {profit_margin:.1%} — moderate switching costs possible"
        else:
            details["margin_stability"] = "Low profit margins — switching costs may be limited"

        # Sector-based switching cost bonus (enterprise software, healthcare, etc.)
        high_switching_sectors = {"Technology", "Healthcare", "Financial Services"}
        moderate_switching_sectors = {"Industrials", "Consumer Defensive"}
        if sector in high_switching_sectors:
            score += 4
            details["sector_bonus"] = f"{sector} sector typically has high switching costs (enterprise integrations, regulatory requirements)"
        elif sector in moderate_switching_sectors:
            score += 2
            details["sector_bonus"] = f"{sector} sector has moderate switching costs"
        else:
            details["sector_bonus"] = "Sector switching costs likely low"

        # ROE as stickiness proxy (+3 if ROE > 20%)
        roe = _safe(key_metrics.get("return_on_equity"))
        if roe > 0.25:
            score += 3
            details["roe_signal"] = f"ROE {roe:.1%} — exceptional returns suggest durable customer relationships"
        elif roe > 0.15:
            score += 2
            details["roe_signal"] = f"ROE {roe:.1%} — above-average returns"
        elif roe > 0.10:
            score += 1
            details["roe_signal"] = f"ROE {roe:.1%} — modest return on equity"
        else:
            details["roe_signal"] = "Low ROE — limited competitive advantage"

        score = min(score, 10)
        return {"score": round(score, 1), "details": details, "max": 10}

    # ── Dimension 3: Network Effects ──────────────────────────────────────────

    def score_network_effects(
        self,
        company_data: dict[str, Any],
        industry_benchmarks: dict[str, float],
    ) -> dict[str, Any]:
        """
        Proxy: revenue growth acceleration, market cap premium, platform sector.
        Scale: 0-10.
        """
        score = 0.0
        details: dict[str, str] = {}
        key_metrics = company_data.get("key_metrics", {})
        sector = company_data.get("sector", "Unknown")
        industry = company_data.get("industry", "Unknown")

        # Platform/marketplace sector bonus
        network_industries = {
            "Internet Content & Information", "Software—Application",
            "Software—Infrastructure", "Electronic Components",
            "Communication Services", "Social Media",
        }
        if any(kw in industry for kw in ["Platform", "Marketplace", "Network", "Social", "Payment"]):
            score += 4
            details["platform_type"] = f"'{industry}' — likely has network effects built in"
        elif sector in {"Technology", "Communication Services"}:
            score += 2
            details["platform_type"] = f"{sector} sector — potential for network effects"
        else:
            details["platform_type"] = "Sector/industry unlikely to have strong network effects"

        # Revenue growth as proxy for value-growth flywheel
        rev_growth = _safe(key_metrics.get("revenue_growth"))
        if rev_growth > 0.30:
            score += 4
            details["growth_flywheel"] = f"Revenue growth {rev_growth:.1%} — strong network value creation"
        elif rev_growth > 0.15:
            score += 2.5
            details["growth_flywheel"] = f"Revenue growth {rev_growth:.1%} — moderate network effects possible"
        elif rev_growth > 0.05:
            score += 1
            details["growth_flywheel"] = f"Revenue growth {rev_growth:.1%} — limited network acceleration"
        else:
            details["growth_flywheel"] = "Low revenue growth — network effects unlikely"

        # Market cap premium as proxy for platform value
        market_cap = _safe(company_data.get("market_cap"), 0)
        trailing_pe = _safe(key_metrics.get("trailing_pe"))
        industry_pe = industry_benchmarks.get("avg_pe", 18)
        if trailing_pe > 0 and trailing_pe > industry_pe * 1.5:
            score += 2
            details["valuation_premium"] = f"P/E {trailing_pe:.1f}x vs industry {industry_pe:.1f}x — market pricing in network premium"
        else:
            details["valuation_premium"] = "No significant valuation premium for network effects"

        score = min(score, 10)
        return {"score": round(score, 1), "details": details, "max": 10}

    # ── Dimension 4: Cost Advantages ──────────────────────────────────────────

    def score_cost_advantages(
        self,
        company_data: dict[str, Any],
        industry_benchmarks: dict[str, float],
    ) -> dict[str, Any]:
        """
        Proxy: operating margin vs peers, asset turnover, operating leverage.
        Scale: 0-10.
        """
        score = 0.0
        details: dict[str, str] = {}
        key_metrics = company_data.get("key_metrics", {})
        sector = company_data.get("sector", "Unknown")

        industry_gm = industry_benchmarks.get("avg_gross_margin", 0.40)
        industry_roic = industry_benchmarks.get("avg_roic", 0.12)

        # Operating margin vs industry (+4 if significantly above)
        op_margin = _safe(key_metrics.get("operating_margins"))
        if op_margin > industry_gm * 1.3:
            score += 4
            details["margin_advantage"] = f"Operating margin {op_margin:.1%} — {op_margin / industry_gm:.1f}x industry gross margin benchmark"
        elif op_margin > industry_gm:
            score += 2.5
            details["margin_advantage"] = f"Operating margin {op_margin:.1%} above industry average"
        elif op_margin > 0:
            score += 1
            details["margin_advantage"] = f"Operating margin {op_margin:.1%} at or below industry"
        else:
            details["margin_advantage"] = "Negative operating margin — no cost advantage"

        # ROA as asset efficiency proxy (+3 if significantly above peers)
        roa = _safe(key_metrics.get("return_on_assets"))
        if roa > 0.15:
            score += 3
            details["asset_efficiency"] = f"ROA {roa:.1%} — superior asset utilisation"
        elif roa > 0.08:
            score += 2
            details["asset_efficiency"] = f"ROA {roa:.1%} — above-average asset efficiency"
        elif roa > 0.03:
            score += 1
            details["asset_efficiency"] = f"ROA {roa:.1%} — below-average asset efficiency"
        else:
            details["asset_efficiency"] = f"ROA {roa:.1%} — poor asset utilisation"

        # Scale bonus (large market cap = likely economies of scale)
        market_cap = _safe(company_data.get("market_cap"), 0)
        if market_cap > 100e9:
            score += 3
            details["scale"] = f"Market cap ${market_cap / 1e9:.0f}B — significant scale economies likely"
        elif market_cap > 10e9:
            score += 1.5
            details["scale"] = f"Market cap ${market_cap / 1e9:.1f}B — moderate scale"
        else:
            details["scale"] = "Smaller-cap company — scale advantages may be limited"

        score = min(score, 10)
        return {"score": round(score, 1), "details": details, "max": 10}

    # ── Dimension 5: Efficient Scale ──────────────────────────────────────────

    def score_efficient_scale(
        self,
        company_data: dict[str, Any],
        industry_benchmarks: dict[str, float],
    ) -> dict[str, Any]:
        """
        Proxy: market cap dominance (proxy for market share), sector concentration.
        Scale: 0-10.
        """
        score = 0.0
        details: dict[str, str] = {}
        key_metrics = company_data.get("key_metrics", {})
        sector = company_data.get("sector", "Unknown")
        industry = company_data.get("industry", "Unknown")

        # High-barrier industries get automatic points
        high_barrier_sectors = {"Utilities", "Energy", "Real Estate", "Financial Services"}
        moderate_barrier_sectors = {"Healthcare", "Industrials", "Materials"}
        if sector in high_barrier_sectors:
            score += 5
            details["barriers"] = f"{sector} — regulated industry with high capital requirements and barriers to entry"
        elif sector in moderate_barrier_sectors:
            score += 3
            details["barriers"] = f"{sector} — moderate barriers to entry"
        else:
            details["barriers"] = f"{sector} — barriers to entry depend on specific competitive position"

        # Market cap as market dominance proxy
        market_cap = _safe(company_data.get("market_cap"), 0)
        if market_cap > 500e9:
            score += 4
            details["market_dominance"] = f"${market_cap / 1e9:.0f}B market cap — likely dominant market position"
        elif market_cap > 50e9:
            score += 2.5
            details["market_dominance"] = f"${market_cap / 1e9:.0f}B market cap — significant market participant"
        elif market_cap > 10e9:
            score += 1
            details["market_dominance"] = f"${market_cap / 1e9:.1f}B market cap — mid-cap with regional/niche scale"
        else:
            details["market_dominance"] = "Small-cap — likely faces competition from larger players"

        # Dividend consistency proxy for mature/stable competitive position
        div_yield = _safe(key_metrics.get("dividend_yield"))
        if div_yield > 0.02:
            score += 1
            details["maturity"] = f"Dividend yield {div_yield:.1%} — mature, stable business with predictable cash flows"
        else:
            details["maturity"] = "No significant dividend — reinvestment-stage or growth company"

        score = min(score, 10)
        return {"score": round(score, 1), "details": details, "max": 10}

    # ── Overall moat rating ────────────────────────────────────────────────────

    def calculate_overall_moat(
        self,
        dimension_scores: dict[str, dict[str, Any]],
        roic_latest: float | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate five dimension scores into Wide / Narrow / None rating.
        ROIC validation: Wide requires ROIC > 15%, Narrow requires ROIC > 10%.
        """
        total = sum(d["score"] for d in dimension_scores.values())
        max_possible = sum(d["max"] for d in dimension_scores.values())

        # Base rating from score
        if total >= 35:
            base_rating = "Wide"
        elif total >= 20:
            base_rating = "Narrow"
        else:
            base_rating = "None"

        # ROIC validation downgrade
        rating = base_rating
        roic_note = ""
        if roic_latest is not None:
            if base_rating == "Wide" and roic_latest < 0.15:
                rating = "Narrow"
                roic_note = f" (downgraded: ROIC {roic_latest:.1%} < 15% Wide threshold)"
            elif base_rating == "Narrow" and roic_latest < 0.10:
                rating = "None"
                roic_note = f" (downgraded: ROIC {roic_latest:.1%} < 10% Narrow threshold)"
            elif base_rating == "Wide" and roic_latest >= 0.15:
                roic_note = f" (ROIC {roic_latest:.1%} confirms Wide moat)"
            elif base_rating == "Narrow" and roic_latest >= 0.10:
                roic_note = f" (ROIC {roic_latest:.1%} confirms Narrow moat)"

        return {
            "total_score": round(total, 1),
            "max_score": max_possible,
            "rating": rating,
            "base_rating": base_rating,
            "roic_note": roic_note,
            "dimension_scores": {k: v["score"] for k, v in dimension_scores.items()},
        }

    def run_full_moat_analysis(
        self,
        company_data: dict[str, Any],
        dcf_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run complete quantitative moat analysis.
        Returns a MoatScore-compatible dict with all dimension scores.
        """
        sector = company_data.get("sector", "Unknown")
        industry_benchmarks = self.get_industry_benchmarks(sector)

        dimensions = {
            "intangible_assets": self.score_intangible_assets(company_data, industry_benchmarks),
            "switching_costs": self.score_switching_costs(company_data, industry_benchmarks),
            "network_effects": self.score_network_effects(company_data, industry_benchmarks),
            "cost_advantages": self.score_cost_advantages(company_data, industry_benchmarks),
            "efficient_scale": self.score_efficient_scale(company_data, industry_benchmarks),
        }

        roic_latest = None
        if dcf_metrics:
            roic_latest = dcf_metrics.get("roic_latest")

        overall = self.calculate_overall_moat(dimensions, roic_latest)

        key_metrics = company_data.get("key_metrics", {})
        return {
            "intangible_assets": dimensions["intangible_assets"]["score"],
            "switching_costs": dimensions["switching_costs"]["score"],
            "network_effects": dimensions["network_effects"]["score"],
            "cost_advantages": dimensions["cost_advantages"]["score"],
            "efficient_scale": dimensions["efficient_scale"]["score"],
            "total_score": overall["total_score"],
            "rating": overall["rating"],
            "roic_latest": roic_latest or 0.0,
            "roic_5yr_avg": _safe(dcf_metrics.get("roic_5yr_avg") if dcf_metrics else None),
            "gross_margin": _safe(key_metrics.get("gross_margins")),
            "operating_margin": _safe(key_metrics.get("operating_margins")),
            "roe": _safe(key_metrics.get("return_on_equity")),
            "dimension_details": {k: v["details"] for k, v in dimensions.items()},
            "industry_benchmarks": industry_benchmarks,
            "roic_note": overall["roic_note"],
        }
