"""
DCF valuation engine — pure computation, no LLM calls, no I/O.

Implements:
  - WACC via CAPM (cost of equity) + after-tax cost of debt
  - Historical FCF extraction from yfinance/Alpha Vantage DataFrames
  - 2-stage and 3-stage FCF projection
  - Gordon Growth Model terminal value
  - PV summation → intrinsic value per share
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Row-name aliases (yfinance can vary across versions) ───────────────────────
_ALIASES: dict[str, list[str]] = {
    "revenue": [
        "Total Revenue", "Revenue", "Revenues", "Net Revenue", "Total Revenues",
    ],
    "operating_income": [
        "Operating Income", "EBIT", "Operating Income Or Loss",
        "Total Operating Income As Reported",
    ],
    "interest_expense": [
        "Interest Expense", "Interest Expense Non Operating", "Net Interest Income",
    ],
    "tax_provision": [
        "Tax Provision", "Income Tax Expense",
        "Income Tax Expense Benefit", "Provision For Income Taxes",
    ],
    "depreciation": [
        "Reconciled Depreciation", "Depreciation",
        "Depreciation And Amortization", "Depreciation Amortization Depletion",
    ],
    "operating_cf": [
        "Operating Cash Flow", "Cash From Operations",
        "Total Cash From Operating Activities",
        "Net Cash Provided By Operating Activities",
    ],
    "capex": [
        "Capital Expenditure", "Purchase Of PPE", "Capital Expenditures",
        "Purchases Of Property Plant And Equipment",
        "Payments For Capital Expenditures",
    ],
    "total_debt": [
        "Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation",
    ],
    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash", "Total Cash",
    ],
    "total_assets": ["Total Assets", "Assets"],
    "total_equity": [
        "Stockholders Equity", "Total Stockholders Equity",
        "Common Stock Equity", "Total Equity Gross Minority Interest",
    ],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": [
        "Current Liabilities", "Total Current Liabilities",
        "Current Liabilities And Short Term Debt",
    ],
}


def _get_row(df: pd.DataFrame | None, field: str) -> pd.Series | None:
    """Return first matching row Series from *df*, or None."""
    if df is None or df.empty:
        return None
    for alias in _ALIASES.get(field, [field]):
        if alias in df.index:
            row = df.loc[alias]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.dropna()
    return None


def _safe(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


class DCFEngine:
    """Discounted Cash Flow valuation engine."""

    def extract_historical_metrics(self, company_data: dict[str, Any]) -> dict[str, Any]:
        """
        Parse financial statement DataFrames and return derived metrics
        used for suggesting DCF assumptions.
        """
        income = company_data.get("income_statement")
        balance = company_data.get("balance_sheet")
        cashflow = company_data.get("cash_flow")
        result: dict[str, Any] = {}

        # ── Revenue ──────────────────────────────────────────────────────────
        rev_row = _get_row(income, "revenue")
        if rev_row is not None and not rev_row.empty:
            rev = rev_row.sort_index()
            result["revenues"] = rev
            if len(rev) >= 2:
                n = len(rev) - 1
                result["revenue_cagr"] = float((rev.iloc[-1] / rev.iloc[0]) ** (1 / n) - 1)
            else:
                result["revenue_cagr"] = 0.05

        # ── Operating margin ─────────────────────────────────────────────────
        ebit_row = _get_row(income, "operating_income")
        if ebit_row is not None and rev_row is not None:
            aligned = pd.concat(
                [ebit_row.sort_index().rename("ebit"), rev_row.sort_index().rename("rev")],
                axis=1,
            ).dropna()
            if not aligned.empty:
                margins = (aligned["ebit"] / aligned["rev"]).clip(-1, 1)
                result["operating_margins"] = margins.tolist()
                result["avg_operating_margin"] = float(margins.mean())
                result["latest_operating_margin"] = float(margins.iloc[-1])

        # ── FCF history ──────────────────────────────────────────────────────
        ocf_row = _get_row(cashflow, "operating_cf")
        capex_row = _get_row(cashflow, "capex")
        if ocf_row is not None:
            ocf = ocf_row.sort_index()
            capex = capex_row.sort_index().abs() if capex_row is not None else pd.Series(0, index=ocf.index)
            result["historical_fcf"] = ocf - capex

        # ── CapEx % of revenue ───────────────────────────────────────────────
        if capex_row is not None and rev_row is not None:
            aligned = pd.concat(
                [capex_row.sort_index().abs().rename("c"), rev_row.sort_index().rename("r")],
                axis=1,
            ).dropna()
            if not aligned.empty and (aligned["r"] > 0).any():
                result["capex_pct_revenue"] = float(
                    (aligned["c"] / aligned["r"]).clip(0, 0.5).mean()
                )

        # ── Effective tax rate ───────────────────────────────────────────────
        tax_row = _get_row(income, "tax_provision")
        if tax_row is not None and ebit_row is not None:
            aligned = pd.concat(
                [tax_row.sort_index().abs().rename("t"), ebit_row.sort_index().rename("e")],
                axis=1,
            ).dropna()
            profitable = aligned[aligned["e"] > 0]
            if not profitable.empty:
                result["avg_tax_rate"] = float(
                    (profitable["t"] / profitable["e"]).clip(0, 0.5).mean()
                )

        # ── Balance sheet snapshots (most recent year) ───────────────────────
        for field, key in [
            ("total_debt", "latest_total_debt"),
            ("cash", "latest_cash"),
            ("total_equity", "latest_equity"),
            ("total_assets", "latest_total_assets"),
            ("current_assets", "latest_current_assets"),
            ("current_liabilities", "latest_current_liabilities"),
        ]:
            row = _get_row(balance, field)
            if row is not None and not row.empty:
                result[key] = _safe(row.sort_index().iloc[-1])

        # ── Cost of debt ─────────────────────────────────────────────────────
        interest_row = _get_row(income, "interest_expense")
        if interest_row is not None and not interest_row.empty:
            latest_interest = abs(_safe(interest_row.sort_index().iloc[-1]))
            total_debt = result.get("latest_total_debt", 0)
            result["cost_of_debt"] = (
                min(latest_interest / total_debt, 0.20) if total_debt > 0 else 0.04
            )

        # ── ROIC ─────────────────────────────────────────────────────────────
        assets = result.get("latest_total_assets", 0)
        cash = result.get("latest_cash", 0)
        cur_liab = result.get("latest_current_liabilities", 0)
        invested_capital = max(assets - cash - cur_liab, 1)
        if ebit_row is not None and not ebit_row.empty:
            tax_rate = result.get("avg_tax_rate", 0.21)
            nopat = _safe(ebit_row.sort_index().iloc[-1]) * (1 - tax_rate)
            result["roic_latest"] = nopat / invested_capital

        # ── NWC % revenue ────────────────────────────────────────────────────
        nwc = result.get("latest_current_assets", 0) - result.get("latest_current_liabilities", 0)
        latest_rev = _safe(rev_row.sort_index().iloc[-1]) if rev_row is not None and not rev_row.empty else 0.0
        if latest_rev > 0:
            result["nwc_pct_revenue"] = nwc / latest_rev

        # ── Depreciation % revenue ───────────────────────────────────────────
        dep_row = _get_row(income, "depreciation")
        if dep_row is None:
            dep_row = _get_row(cashflow, "depreciation")
        if dep_row is not None and not dep_row.empty and latest_rev > 0:
            result["depreciation_pct_revenue"] = abs(_safe(dep_row.sort_index().iloc[-1])) / latest_rev

        return result

    def calculate_wacc(
        self,
        beta: float,
        risk_free_rate: float,
        market_risk_premium: float,
        cost_of_debt: float,
        tax_rate: float,
        market_cap: float,
        total_debt: float,
    ) -> dict[str, float]:
        """CAPM-based WACC. Returns full component breakdown."""
        cost_of_equity = risk_free_rate + beta * market_risk_premium
        total_capital = market_cap + total_debt
        if total_capital <= 0:
            equity_weight, debt_weight = 1.0, 0.0
        else:
            equity_weight = market_cap / total_capital
            debt_weight = total_debt / total_capital

        after_tax_kd = cost_of_debt * (1 - tax_rate)
        wacc = equity_weight * cost_of_equity + debt_weight * after_tax_kd

        return {
            "wacc": round(float(wacc), 5),
            "cost_of_equity": round(float(cost_of_equity), 5),
            "after_tax_cost_of_debt": round(float(after_tax_kd), 5),
            "equity_weight": round(float(equity_weight), 4),
            "debt_weight": round(float(debt_weight), 4),
            "beta_used": float(beta),
            "risk_free_rate": float(risk_free_rate),
            "market_risk_premium": float(market_risk_premium),
        }

    @staticmethod
    def build_two_stage_growth_rates(
        revenue_cagr: float,
        lifecycle_stage: str = "Mature",
        forecast_years: int = 5,
    ) -> list[float]:
        """Decelerating growth path for mature / standard 2-stage DCF."""
        rev_cagr = float(revenue_cagr or 0.05)
        if lifecycle_stage in {"Early Growth", "High Growth"}:
            base = [
                min(rev_cagr * 1.1, 0.30),
                min(rev_cagr * 1.0, 0.25),
                min(rev_cagr * 0.85, 0.20),
                min(rev_cagr * 0.70, 0.15),
                min(rev_cagr * 0.60, 0.10),
            ]
        elif lifecycle_stage == "Mature Growth":
            base = [
                min(rev_cagr * 0.90, 0.12),
                min(rev_cagr * 0.80, 0.10),
                min(rev_cagr * 0.70, 0.08),
                min(rev_cagr * 0.60, 0.06),
                min(rev_cagr * 0.50, 0.05),
            ]
        else:
            base = [max(rev_cagr * 0.70, 0.01)] * forecast_years
        return [round(g, 4) for g in base[:forecast_years]]

    @staticmethod
    def build_three_stage_growth_rates(
        revenue_cagr: float,
        lifecycle_stage: str = "High Growth",
        forecast_years: int = 5,
        terminal_growth: float = 0.025,
    ) -> list[float]:
        """
        Three-stage revenue growth schedule:
          Stage 1 — high growth (years 1–2)
          Stage 2 — fade (years 3–4)
          Stage 3 — stable (final forecast year(s), approaching terminal)
        """
        rev_cagr = float(revenue_cagr or 0.05)
        terminal_growth = float(terminal_growth or 0.025)

        if lifecycle_stage in {"Early Growth", "High Growth"}:
            stage1 = min(rev_cagr * 1.20, 0.40)
            stage2 = min(rev_cagr * 0.95, 0.28)
            stage3 = min(rev_cagr * 0.70, 0.18)
        elif lifecycle_stage == "Mature Growth":
            stage1 = min(rev_cagr * 1.05, 0.15)
            stage2 = min(rev_cagr * 0.85, 0.10)
            stage3 = min(rev_cagr * 0.65, 0.07)
        else:
            stage1 = max(rev_cagr * 0.85, 0.03)
            stage2 = max(rev_cagr * 0.70, 0.02)
            stage3 = max(terminal_growth + 0.01, 0.015)

        fade_end = max(stage3, terminal_growth + 0.005)
        schedule: list[float] = []
        for year_idx in range(forecast_years):
            if year_idx < 2:
                schedule.append(stage1 if year_idx == 0 else (stage1 + stage2) / 2)
            elif year_idx < forecast_years - 1:
                t = (year_idx - 1) / max(forecast_years - 3, 1)
                schedule.append(stage2 + (fade_end - stage2) * t)
            else:
                schedule.append(fade_end)
        return [round(max(g, terminal_growth), 4) for g in schedule]

    def resolve_growth_schedule(
        self,
        assumptions: dict[str, Any],
        metrics: dict[str, Any],
        lifecycle_stage: str = "Mature",
    ) -> list[float]:
        """Pick or build revenue growth rates based on model_type."""
        explicit = assumptions.get("revenue_growth_rates")
        forecast_years = int(assumptions.get("forecast_years") or 5)
        model_type = (assumptions.get("model_type") or "2-stage").strip().lower()
        rev_cagr = float(metrics.get("revenue_cagr") or 0.05)
        terminal_g = float(assumptions.get("terminal_growth_rate") or 0.025)

        if explicit and len(explicit) >= forecast_years:
            return [float(g) for g in explicit[:forecast_years]]

        if model_type == "3-stage":
            return self.build_three_stage_growth_rates(
                rev_cagr, lifecycle_stage, forecast_years, terminal_g
            )
        return self.build_two_stage_growth_rates(rev_cagr, lifecycle_stage, forecast_years)

    def project_fcf(
        self,
        base_revenue: float,
        growth_rates: list[float],
        operating_margin: float,
        capex_pct_revenue: float,
        nwc_pct_revenue: float,
        tax_rate: float,
        depreciation_pct_revenue: float = 0.03,
    ) -> list[float]:
        """
        Project FCFF = NOPAT + D&A − CapEx − ΔNWC for each forecast year.
        Length of *growth_rates* determines number of projection years.
        """
        fcfs: list[float] = []
        prev_rev = base_revenue
        prev_nwc = base_revenue * nwc_pct_revenue

        for g in growth_rates:
            rev = prev_rev * (1 + g)
            ebit = rev * operating_margin
            nopat = ebit * (1 - tax_rate)
            dep = rev * depreciation_pct_revenue
            capex = rev * capex_pct_revenue
            nwc = rev * nwc_pct_revenue
            delta_nwc = nwc - prev_nwc
            fcfs.append(float(nopat + dep - capex - delta_nwc))
            prev_rev = rev
            prev_nwc = nwc

        return fcfs

    def calculate_terminal_value(
        self,
        terminal_fcf: float,
        wacc: float,
        terminal_growth_rate: float,
    ) -> float:
        """Gordon Growth Model: TV = FCF*(1+g) / (WACC-g)."""
        if wacc <= terminal_growth_rate:
            raise ValueError(
                f"WACC ({wacc:.2%}) must exceed terminal growth ({terminal_growth_rate:.2%})"
            )
        return float(terminal_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate))

    def calculate_intrinsic_value(
        self,
        projected_fcfs: list[float],
        terminal_value: float,
        wacc: float,
        cash: float,
        total_debt: float,
        shares_outstanding: float,
        minority_interest: float = 0.0,
    ) -> dict[str, float]:
        """Discount FCFs + terminal value; bridge to equity; derive per-share value."""
        n = len(projected_fcfs)
        pv_fcfs = [fcf / (1 + wacc) ** (t + 1) for t, fcf in enumerate(projected_fcfs)]
        pv_tv = terminal_value / (1 + wacc) ** n

        enterprise_value = sum(pv_fcfs) + pv_tv
        equity_value = enterprise_value + cash - total_debt - minority_interest
        intrinsic_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0.0

        return {
            "enterprise_value": float(enterprise_value),
            "equity_value": float(equity_value),
            "intrinsic_value_per_share": float(intrinsic_per_share),
            "pv_fcf_stages": [float(x) for x in pv_fcfs],
            "pv_terminal_value": float(pv_tv),
            "terminal_value": float(terminal_value),
        }

    def run_full_dcf(
        self,
        company_data: dict[str, Any],
        assumptions: dict[str, Any],
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a full DCF valuation using provided assumptions.
        Returns a DCFResults-compatible dict.
        """
        from config.settings import MARKET_RISK_PREMIUM

        metrics = self.extract_historical_metrics(company_data)

        beta = _safe(company_data.get("beta"), 1.0) or 1.0
        rf = _safe(market_data.get("risk_free_rate"), 0.045)
        kd = _safe(assumptions.get("cost_of_debt") or metrics.get("cost_of_debt"), 0.05)
        tax = _safe(assumptions.get("tax_rate") or metrics.get("avg_tax_rate"), 0.21)
        mkt_cap = _safe(company_data.get("market_cap"), 0)
        total_debt = _safe(metrics.get("latest_total_debt"), 0)

        wacc_dict = self.calculate_wacc(
            beta=beta,
            risk_free_rate=rf,
            market_risk_premium=MARKET_RISK_PREMIUM,
            cost_of_debt=kd,
            tax_rate=tax,
            market_cap=mkt_cap,
            total_debt=total_debt,
        )
        wacc = _safe(assumptions.get("wacc"), wacc_dict["wacc"])

        rev_series = metrics.get("revenues")
        base_rev = _safe(rev_series.iloc[-1]) if rev_series is not None and not rev_series.empty else 0.0

        lifecycle = assumptions.get("lifecycle_stage", "Mature")
        growth_rates = self.resolve_growth_schedule(assumptions, metrics, lifecycle)
        assumptions = {**assumptions, "revenue_growth_rates": growth_rates}
        op_margin = _safe(
            assumptions.get("operating_margin_target") or metrics.get("avg_operating_margin"), 0.15
        )
        capex_pct = _safe(
            assumptions.get("capex_percent_revenue") or metrics.get("capex_pct_revenue"), 0.04
        )
        nwc_pct = _safe(
            assumptions.get("nwc_percent_revenue") or metrics.get("nwc_pct_revenue"), 0.02
        )
        dep_pct = _safe(metrics.get("depreciation_pct_revenue"), 0.03)

        projected_fcfs = self.project_fcf(
            base_revenue=base_rev,
            growth_rates=growth_rates,
            operating_margin=op_margin,
            capex_pct_revenue=capex_pct,
            nwc_pct_revenue=nwc_pct,
            tax_rate=tax,
            depreciation_pct_revenue=dep_pct,
        )

        terminal_g = _safe(assumptions.get("terminal_growth_rate"), 0.025)
        tv = self.calculate_terminal_value(projected_fcfs[-1], wacc, terminal_g)

        cash = _safe(metrics.get("latest_cash"), 0)
        shares = _safe(company_data.get("shares_outstanding"), 1) or 1
        current_price = _safe(company_data.get("current_price"), 0)

        iv_dict = self.calculate_intrinsic_value(
            projected_fcfs=projected_fcfs,
            terminal_value=tv,
            wacc=wacc,
            cash=cash,
            total_debt=total_debt,
            shares_outstanding=shares,
        )

        intrinsic = iv_dict["intrinsic_value_per_share"]
        upside = (intrinsic / current_price - 1) if current_price > 0 else 0.0
        mos = 1 - (current_price / intrinsic) if intrinsic > 0 else 0.0

        return {
            **iv_dict,
            "current_price": current_price,
            "upside_downside_pct": upside,
            "margin_of_safety": mos,
            "wacc_used": wacc,
            "wacc_components": wacc_dict,
            "terminal_growth_used": terminal_g,
            "projected_fcfs": projected_fcfs,
            "base_revenue": base_rev,
            "tax_rate_used": tax,
            "model_type_used": assumptions.get("model_type", "2-stage"),
            "revenue_growth_rates_used": growth_rates,
        }
