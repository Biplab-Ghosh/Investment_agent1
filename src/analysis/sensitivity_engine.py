"""
Sensitivity analysis engine.

Implements:
  - 2D data table: WACC (rows) × Terminal Growth Rate (columns)
  - Monte Carlo simulation with configurable probability distributions
  - Bull / Base / Bear scenario analysis
  - Tornado chart data (absolute value impact per variable)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SensitivityEngine:
    """Sensitivity and scenario analysis for DCF valuations."""

    # ── 2D Sensitivity table ──────────────────────────────────────────────────

    def wacc_vs_tgr_table(
        self,
        dcf_engine: Any,
        company_data: dict[str, Any],
        base_assumptions: dict[str, Any],
        market_data: dict[str, Any],
        wacc_range: tuple[float, float] = (-0.02, 0.02),
        wacc_steps: int = 5,
        tgr_range: tuple[float, float] = (-0.01, 0.01),
        tgr_steps: int = 5,
    ) -> pd.DataFrame:
        """
        Build a matrix of intrinsic values varying WACC and terminal growth rate.
        Rows = WACC values; Columns = terminal growth rate values.
        """
        base_wacc = base_assumptions.get("wacc", 0.10)
        base_tgr = base_assumptions.get("terminal_growth_rate", 0.025)

        waccs = np.linspace(base_wacc + wacc_range[0], base_wacc + wacc_range[1], wacc_steps)
        tgrs = np.linspace(base_tgr + tgr_range[0], base_tgr + tgr_range[1], tgr_steps)

        rows = []
        for wacc in waccs:
            row = {}
            for tgr in tgrs:
                if wacc <= tgr:
                    row[f"{tgr:.2%}"] = np.nan
                    continue
                assumptions = {**base_assumptions, "wacc": float(wacc), "terminal_growth_rate": float(tgr)}
                try:
                    result = dcf_engine.run_full_dcf(company_data, assumptions, market_data)
                    row[f"{tgr:.2%}"] = round(result["intrinsic_value_per_share"], 2)
                except Exception:
                    row[f"{tgr:.2%}"] = np.nan
            rows.append(row)

        df = pd.DataFrame(rows, index=[f"{w:.2%}" for w in waccs])
        df.index.name = "WACC \\ TGR"
        return df

    # ── Monte Carlo simulation ────────────────────────────────────────────────

    def monte_carlo_simulation(
        self,
        dcf_engine: Any,
        company_data: dict[str, Any],
        base_assumptions: dict[str, Any],
        market_data: dict[str, Any],
        n_simulations: int = 10_000,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Run Monte Carlo simulation over key DCF assumptions.

        Distributions:
          - WACC: Normal(mean=base_wacc, σ=1%)
          - Terminal growth: Triangular(min=0%, mode=base_tgr, max=4%)
          - Revenue growth yr1: Normal(mean=base_g, σ=3%)
        """
        rng = np.random.default_rng(seed)
        base_wacc = base_assumptions.get("wacc", 0.10)
        base_tgr = base_assumptions.get("terminal_growth_rate", 0.025)
        growth_rates = base_assumptions.get("revenue_growth_rates", [0.08, 0.06, 0.05, 0.04, 0.03])
        base_g1 = growth_rates[0] if growth_rates else 0.06

        wacc_samples = rng.normal(loc=base_wacc, scale=0.01, size=n_simulations)
        tgr_samples = rng.triangular(0.0, base_tgr, 0.04, size=n_simulations)
        g1_samples = rng.normal(loc=base_g1, scale=0.03, size=n_simulations)

        intrinsic_values: list[float] = []

        for i in range(n_simulations):
            wacc_i = float(np.clip(wacc_samples[i], 0.04, 0.25))
            tgr_i = float(np.clip(tgr_samples[i], 0.0, wacc_i - 0.01))
            g1_i = float(np.clip(g1_samples[i], -0.20, 0.50))

            sim_growth = [g1_i] + growth_rates[1:]
            assumptions = {
                **base_assumptions,
                "wacc": wacc_i,
                "terminal_growth_rate": tgr_i,
                "revenue_growth_rates": sim_growth,
            }
            try:
                result = dcf_engine.run_full_dcf(company_data, assumptions, market_data)
                iv = result.get("intrinsic_value_per_share", 0)
                if np.isfinite(iv) and iv > 0:
                    intrinsic_values.append(iv)
            except Exception:
                pass

        if not intrinsic_values:
            return {"error": "Monte Carlo produced no valid results"}

        arr = np.array(intrinsic_values)
        percentiles = {
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }

        return {
            "n_simulations": n_simulations,
            "n_valid": len(intrinsic_values),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "monte_carlo_percentiles": percentiles,
            "raw_values": arr.tolist(),
        }

    # ── Scenario analysis ─────────────────────────────────────────────────────

    def scenario_analysis(
        self,
        dcf_engine: Any,
        company_data: dict[str, Any],
        base_assumptions: dict[str, Any],
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run Bull / Base / Bear scenarios.

        Bull:  lower WACC, higher growth, higher margin
        Base:  as-provided assumptions
        Bear:  higher WACC, lower growth, lower margin
        """
        scenarios: dict[str, dict[str, Any]] = {
            "bear": {
                "wacc": base_assumptions.get("wacc", 0.10) + 0.02,
                "terminal_growth_rate": max(
                    base_assumptions.get("terminal_growth_rate", 0.025) - 0.01, 0.01
                ),
                "revenue_growth_rates": [
                    max(g - 0.03, -0.10) for g in base_assumptions.get("revenue_growth_rates", [0.05] * 5)
                ],
                "operating_margin_target": max(
                    base_assumptions.get("operating_margin_target", 0.15) - 0.03, 0.01
                ),
            },
            "base": base_assumptions,
            "bull": {
                "wacc": max(base_assumptions.get("wacc", 0.10) - 0.015, 0.04),
                "terminal_growth_rate": min(
                    base_assumptions.get("terminal_growth_rate", 0.025) + 0.005, 0.04
                ),
                "revenue_growth_rates": [
                    min(g + 0.03, 0.50) for g in base_assumptions.get("revenue_growth_rates", [0.05] * 5)
                ],
                "operating_margin_target": min(
                    base_assumptions.get("operating_margin_target", 0.15) + 0.03, 0.50
                ),
            },
        }

        results: dict[str, Any] = {}
        for name, scenario_assumptions in scenarios.items():
            merged = {**base_assumptions, **scenario_assumptions}
            try:
                r = dcf_engine.run_full_dcf(company_data, merged, market_data)
                results[name] = {
                    "intrinsic_value_per_share": r.get("intrinsic_value_per_share", 0),
                    "upside_downside_pct": r.get("upside_downside_pct", 0),
                    "wacc": merged.get("wacc"),
                    "terminal_growth_rate": merged.get("terminal_growth_rate"),
                }
            except Exception as exc:
                results[name] = {"error": str(exc)}

        return results

    # ── Tornado chart data ────────────────────────────────────────────────────

    def tornado_data(
        self,
        dcf_engine: Any,
        company_data: dict[str, Any],
        base_assumptions: dict[str, Any],
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Calculate the impact on intrinsic value of a ±1-sigma shift in each variable.
        Returns a list sorted by absolute impact (largest first), for tornado chart.
        """
        try:
            base_result = dcf_engine.run_full_dcf(company_data, base_assumptions, market_data)
            base_iv = base_result["intrinsic_value_per_share"]
        except Exception:
            return []

        perturbations = {
            "WACC (±2%)": ("wacc", -0.02, +0.02),
            "Terminal Growth (±1%)": ("terminal_growth_rate", -0.01, +0.01),
            "Operating Margin (±3pp)": ("operating_margin_target", -0.03, +0.03),
            "Revenue Growth Yr1 (±5pp)": ("_revenue_growth_yr1", -0.05, +0.05),
            "CapEx % Revenue (±2pp)": ("capex_percent_revenue", -0.02, +0.02),
        }

        tornado_rows = []
        for label, (key, low_delta, high_delta) in perturbations.items():
            def _run(delta: float, key: str = key) -> float:
                if key == "_revenue_growth_yr1":
                    rates = list(base_assumptions.get("revenue_growth_rates", [0.05] * 5))
                    if rates:
                        rates[0] = rates[0] + delta
                    mod = {**base_assumptions, "revenue_growth_rates": rates}
                else:
                    base_val = base_assumptions.get(key, 0)
                    mod = {**base_assumptions, key: base_val + delta}
                try:
                    r = dcf_engine.run_full_dcf(company_data, mod, market_data)
                    return r.get("intrinsic_value_per_share", base_iv)
                except Exception:
                    return base_iv

            low_iv = _run(low_delta)
            high_iv = _run(high_delta)

            tornado_rows.append({
                "variable": label,
                "low_value": min(low_iv, high_iv),
                "high_value": max(low_iv, high_iv),
                "base_value": base_iv,
                "abs_impact": abs(max(low_iv, high_iv) - min(low_iv, high_iv)),
            })

        tornado_rows.sort(key=lambda x: x["abs_impact"], reverse=True)
        return tornado_rows
