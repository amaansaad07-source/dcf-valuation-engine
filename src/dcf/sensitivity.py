"""Sensitivity grids and coherent bear/base/bull scenarios.

Sensitivity varies one or two inputs; scenarios move a consistent set together. Both are
distinct from the Monte Carlo layer, which samples all inputs jointly.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .assumptions import Assumptions
from .config import CFG, log
from .dcf import run_dcf


def sensitivity_grid(
    drivers: pd.DataFrame,
    a: Assumptions,
    shares: float,
    wacc_range: Sequence[float],
    growth_range: Sequence[float],
    current_price: Optional[float] = None,
    metric: str = "per_share",
) -> pd.DataFrame:
    """Two-way sensitivity: rows = WACC, columns = terminal growth.

    metric: 'per_share' | 'upside' | 'enterprise_value'
    """
    grid = np.full((len(wacc_range), len(growth_range)), np.nan)

    for i, w in enumerate(wacc_range):
        for j, g in enumerate(growth_range):
            if g >= w - CFG.min_wacc_spread:
                continue                       # infeasible corner — leave it blank
            try:
                res = run_dcf(drivers, a.copy_with(terminal_growth=g), w, shares, current_price)
                if metric == "per_share":
                    grid[i, j] = res.value_per_share
                elif metric == "upside":
                    grid[i, j] = res.upside_vs_price if current_price else np.nan
                else:
                    grid[i, j] = res.enterprise_value
            except (ValueError, ZeroDivisionError):
                continue

    return pd.DataFrame(
        grid,
        index=[f"{w:.2%}" for w in wacc_range],
        columns=[f"{g:.2%}" for g in growth_range],
    )


def build_sensitivity_ranges(wacc: float, terminal_growth: float,
                             steps: int = 5, wacc_step: float = 0.005,
                             growth_step: float = 0.0025) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric ranges centred on the base case — the standard banker grid."""
    half = steps // 2
    wacc_range = np.array([wacc + (i - half) * wacc_step for i in range(steps)])
    growth_range = np.array([terminal_growth + (i - half) * growth_step for i in range(steps)])
    return wacc_range, np.clip(growth_range, -0.01, 0.05)


SCENARIOS: Dict[str, Dict[str, float]] = {
    "Bear":  {"growth_delta": -0.04, "margin_delta": -0.03, "wacc_delta": +0.010, "g_delta": -0.005},
    "Base":  {"growth_delta": 0.0,   "margin_delta": 0.0,   "wacc_delta": 0.0,    "g_delta": 0.0},
    "Bull":  {"growth_delta": +0.04, "margin_delta": +0.03, "wacc_delta": -0.010, "g_delta": +0.005},
}


def run_scenarios(drivers: pd.DataFrame, a: Assumptions, wacc: float, shares: float,
                  current_price: Optional[float] = None,
                  scenarios: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
    """Coherent bear/base/bull cases — inputs move together, as they do in reality."""
    scenarios = scenarios or SCENARIOS
    rows = []
    for name, deltas in scenarios.items():
        case = a.copy_with(
            revenue_growth_y1=a.revenue_growth_y1 + deltas["growth_delta"],
            ebit_margin_start=a.ebit_margin_start + deltas["margin_delta"],
            ebit_margin_target=(a.ebit_margin_target or a.ebit_margin_start) + deltas["margin_delta"],
            terminal_growth=max(a.terminal_growth + deltas["g_delta"], -0.005),
        )
        case_wacc = wacc + deltas["wacc_delta"]
        try:
            res = run_dcf(drivers, case, case_wacc, shares, current_price)
            rows.append({
                "Scenario": name,
                "Revenue growth (Y1)": case.revenue_growth_y1,
                "Terminal EBIT margin": case.ebit_margin_target,
                "WACC": case_wacc,
                "Terminal growth": case.terminal_growth,
                "Enterprise value": res.enterprise_value,
                "Equity value": res.equity_value,
                "Value per share": res.value_per_share,
                "Upside": res.upside_vs_price,
            })
        except ValueError as exc:
            log.warning("Scenario '%s' is infeasible: %s", name, exc)
    return pd.DataFrame(rows).set_index("Scenario")
