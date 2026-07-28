"""Fully vectorised Monte Carlo over the drivers that actually move the answer.

Builds the whole simulation as (n_sims x horizon) arrays rather than looping over
``run_dcf`` - roughly a 500x speed-up, 50,000 paths in about a tenth of a second.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .assumptions import Assumptions
from .config import CFG, log


@dataclass
class MonteCarloResult:
    values_per_share: np.ndarray
    percentiles: Dict[str, float]
    prob_above_price: Optional[float]
    mean: float
    median: float
    std: float
    n_valid: int
    n_discarded: int
    inputs: pd.DataFrame
    runtime_seconds: float
    share_negative: float = 0.0    # fraction of paths ending in negative equity value

    def summary_frame(self) -> pd.DataFrame:
        rows = {f"P{k}": v for k, v in self.percentiles.items()}
        rows.update({"Mean": self.mean, "Median": self.median, "Std dev": self.std})
        return pd.DataFrame({"Value per share": rows}).round(2)


def monte_carlo_dcf(
    drivers: pd.DataFrame,
    a: Assumptions,
    wacc: float,
    shares: float,
    bridge_adjustment: float,
    current_price: Optional[float] = None,
    n_sims: int = 50_000,
    growth_sd: float = 0.03,
    margin_sd: float = 0.02,
    wacc_sd: float = 0.010,
    terminal_growth_bounds: Tuple[float, float, float] = (0.010, 0.025, 0.032),
    capex_sd: float = 0.005,
    seed: int = 42,
) -> MonteCarloResult:
    """Fully vectorised Monte Carlo over the four inputs that actually move the answer.

    `bridge_adjustment` is (equity value − enterprise value) from the base case: net cash,
    minority interest and preferred are balance-sheet facts, not random variables, so they
    are held constant across draws rather than re-simulated.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    H = a.horizon

    # ---- Draw the joint sample ------------------------------------------------------
    g1 = rng.normal(a.revenue_growth_y1, growth_sd, n_sims)
    margin_target = rng.normal(
        a.ebit_margin_target if a.ebit_margin_target is not None else a.ebit_margin_start,
        margin_sd, n_sims,
    )
    wacc_draw = rng.normal(wacc, wacc_sd, n_sims)
    lo, mode, hi = terminal_growth_bounds
    g_terminal = rng.triangular(lo, mode, hi, n_sims)
    capex_pct = rng.normal(a.capex_pct_revenue, capex_sd, n_sims)

    # Keep draws inside the space where the model is defined
    margin_target = np.clip(margin_target, -0.25, 0.70)
    capex_pct = np.clip(capex_pct, 0.0, 0.50)
    wacc_draw = np.clip(wacc_draw, 0.03, 0.30)

    feasible = wacc_draw - g_terminal > CFG.min_wacc_spread
    n_discarded = int((~feasible).sum())

    g1, margin_target = g1[feasible], margin_target[feasible]
    wacc_draw, g_terminal = wacc_draw[feasible], g_terminal[feasible]
    capex_pct = capex_pct[feasible]
    n = len(g1)
    if n == 0:
        raise ValueError("Every draw was infeasible — tighten the WACC or terminal-growth ranges.")

    # ---- Build every revenue path at once: shape (n_sims, horizon) -------------------
    base_revenue = float(drivers["revenue"].iloc[-1])
    base_nwc = float(drivers["nwc"].iloc[-1]) if np.isfinite(drivers["nwc"].iloc[-1]) \
        else a.nwc_pct_revenue * base_revenue

    fade = np.linspace(0, 1, H)[None, :]                       # (1, H)
    growth_paths = g1[:, None] * (1 - fade) + g_terminal[:, None] * fade
    revenue = base_revenue * np.cumprod(1 + growth_paths, axis=1)

    margin_start = a.ebit_margin_start
    margin_paths = margin_start * (1 - fade) + margin_target[:, None] * fade

    ebit = revenue * margin_paths
    nopat = ebit - np.maximum(ebit, 0.0) * a.tax_rate   # no tax subsidy on losses
    d_and_a = revenue * a.da_pct_revenue
    capex = revenue * capex_pct[:, None]
    ebitda = ebit + d_and_a

    nwc = revenue * a.nwc_pct_revenue
    prior_nwc = np.concatenate([np.full((n, 1), base_nwc), nwc[:, :-1]], axis=1)
    delta_nwc = nwc - prior_nwc

    ufcf = nopat + d_and_a - capex - delta_nwc

    # ---- Discount ---------------------------------------------------------------------
    periods = np.arange(1, H + 1, dtype=float)
    if a.mid_year_convention:
        periods -= 0.5
    discount = (1 + wacc_draw[:, None]) ** (-periods[None, :])
    pv_forecast = (ufcf * discount).sum(axis=1)

    # ---- Terminal value ----------------------------------------------------------------
    if a.tv_method == "exit_multiple":
        tv = a.exit_ev_ebitda * ebitda[:, -1]
    else:
        tv = ufcf[:, -1] * (1 + g_terminal) / (wacc_draw - g_terminal)
    tv_period = periods[-1] if a.discount_tv_at_midyear and a.mid_year_convention else float(H)
    pv_tv = tv / (1 + wacc_draw) ** tv_period

    enterprise_value = pv_forecast + pv_tv
    equity_value = enterprise_value + bridge_adjustment
    per_share = equity_value / shares

    # Keep every finite outcome, INCLUDING negatives. A path where equity is worthless
    # is information, not noise — filtering it out silently inflated the distribution for
    # loss-making companies (and crashed outright when every path was negative).
    sane = np.isfinite(per_share)
    per_share_clean = per_share[sane]
    if per_share_clean.size == 0:
        raise ValueError(
            "Monte Carlo produced no finite outcomes — the input ranges are outside the "
            "space where this model is defined. This usually means a deeply loss-making "
            "company; a single-stage DCF is the wrong tool for it."
        )
    share_negative = float((per_share_clean <= 0).mean())

    pct_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles = {str(p): float(np.percentile(per_share_clean, p)) for p in pct_levels}

    runtime = time.perf_counter() - start
    log.info("Monte Carlo: %d valid paths in %.2fs (%d discarded as infeasible)",
             len(per_share_clean), runtime, n_discarded)

    inputs = pd.DataFrame({
        "Revenue growth (Y1)": g1[sane],
        "Terminal EBIT margin": margin_target[sane],
        "WACC": wacc_draw[sane],
        "Terminal growth": g_terminal[sane],
        "Capex % revenue": capex_pct[sane],
        "Value per share": per_share_clean,
    })

    return MonteCarloResult(
        values_per_share=per_share_clean,
        percentiles=percentiles,
        prob_above_price=float((per_share_clean > current_price).mean()) if current_price else None,
        mean=float(per_share_clean.mean()),
        median=float(np.median(per_share_clean)),
        std=float(per_share_clean.std()),
        n_valid=len(per_share_clean),
        n_discarded=n_discarded,
        share_negative=share_negative,
        inputs=inputs,
        runtime_seconds=runtime,
    )


def driver_attribution(mc: MonteCarloResult) -> pd.DataFrame:
    """Which input explains the spread? Rank-correlation of each driver to the output.

    Spearman rather than Pearson because the mapping from WACC to value is convex, not
    linear — rank correlation captures monotone relationships without assuming a shape.
    """
    target = mc.inputs["Value per share"]
    rows = []
    for col in mc.inputs.columns.drop("Value per share"):
        if mc.inputs[col].nunique() < 2 or target.nunique() < 2:
            continue                                     # constant input → no correlation
        rho, p_value = stats.spearmanr(mc.inputs[col], target)
        rows.append({"Driver": col, "Rank correlation": rho, "p-value": p_value,
                     "Contribution to variance": rho ** 2})
    frame = pd.DataFrame(rows).set_index("Driver")
    frame["Contribution to variance"] /= frame["Contribution to variance"].sum()
    return frame.sort_values("Contribution to variance", ascending=False)
