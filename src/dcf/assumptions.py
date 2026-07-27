"""The single assumption set that drives projection, sensitivity and simulation.

If a number is not in ``Assumptions``, it cannot be silently hardcoded downstream.
"""

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import CFG, fmt_pct, log
from .drivers import summarise_drivers


@dataclass
class Assumptions:
    """Every forecast input, in one auditable place."""

    horizon: int = 5

    # Revenue: either an explicit per-year path, or year-1 growth fading to terminal
    revenue_growth_y1: float = 0.08
    revenue_growth_path: Optional[List[float]] = None
    fade_to_terminal: bool = True

    # Margins: start at the current level and converge to a target by the final year
    ebit_margin_start: float = 0.20
    ebit_margin_target: Optional[float] = None

    # Intensity ratios, held flat as a percentage of revenue unless overridden
    da_pct_revenue: float = 0.04
    capex_pct_revenue: float = 0.05
    nwc_pct_revenue: float = 0.05

    tax_rate: float = 0.21

    # Terminal value
    terminal_growth: float = 0.025
    exit_ev_ebitda: float = 12.0
    tv_method: str = "gordon"          # 'gordon' | 'exit_multiple'

    # Conventions
    mid_year_convention: bool = True
    discount_tv_at_midyear: bool = True

    def revenue_growth_schedule(self) -> np.ndarray:
        """Per-year growth rates for the forecast horizon."""
        if self.revenue_growth_path is not None:
            path = np.asarray(self.revenue_growth_path, dtype=float)
            if len(path) != self.horizon:
                raise ValueError(f"revenue_growth_path has {len(path)} entries, need {self.horizon}")
            return path
        if not self.fade_to_terminal:
            return np.full(self.horizon, self.revenue_growth_y1)
        # Linear fade from year-1 growth to terminal growth by the final forecast year
        return np.linspace(self.revenue_growth_y1, self.terminal_growth, self.horizon)

    def margin_schedule(self) -> np.ndarray:
        """Per-year EBIT margin, linearly converging to the target."""
        target = self.ebit_margin_target if self.ebit_margin_target is not None else self.ebit_margin_start
        return np.linspace(self.ebit_margin_start, target, self.horizon)

    def validate(self, wacc: float) -> List[str]:
        """Catch the errors that make a DCF meaningless before it runs."""
        problems = []
        if self.terminal_growth >= wacc - CFG.min_wacc_spread:
            problems.append(
                f"Terminal growth ({self.terminal_growth:.2%}) must sit at least "
                f"{CFG.min_wacc_spread:.2%} below WACC ({wacc:.2%}) — otherwise the "
                "perpetuity diverges and the value is infinite."
            )
        if self.terminal_growth > 0.04:
            problems.append(
                f"Terminal growth of {self.terminal_growth:.2%} exceeds long-run nominal GDP; "
                "the company would eventually become the whole economy."
            )
        if not 0 <= self.tax_rate < 0.6:
            problems.append(f"Tax rate of {self.tax_rate:.1%} is implausible.")
        if self.horizon < 3 or self.horizon > 15:
            problems.append(f"A {self.horizon}-year horizon is unusual; 5–10 is standard.")
        if self.capex_pct_revenue < self.da_pct_revenue * 0.5:
            problems.append(
                "Capex is running well below D&A — the asset base is shrinking, which is "
                "inconsistent with a growing perpetuity."
            )
        return problems

    def copy_with(self, **overrides) -> "Assumptions":
        """Immutable-style update used by the sensitivity and scenario engines."""
        base = asdict(self)
        base.update(overrides)
        return Assumptions(**base)

    def to_frame(self) -> pd.DataFrame:
        growth = self.revenue_growth_schedule()
        margin = self.margin_schedule()
        years = [f"FY+{i}" for i in range(1, self.horizon + 1)]
        frame = pd.DataFrame({"Revenue growth": growth, "EBIT margin": margin}, index=years).T
        return frame.apply(lambda row: row.map(lambda v: fmt_pct(v, 1)), axis=1)


def seed_assumptions_from_history(
    drivers: pd.DataFrame,
    horizon: int = 5,
    lookback: int = 5,
    terminal_growth: float = 0.025,
    margin_convergence: float = 0.5,
) -> Assumptions:
    """Build a defensible base case from the company's own operating history.

    `margin_convergence` pulls the terminal-year margin from the latest reported level
    toward the historical median: 0 keeps today's margin, 1 fully mean-reverts. Half is a
    reasonable default — it assumes some, but not all, of any recent margin move sticks.
    """
    stats_ = summarise_drivers(drivers, lookback)

    growth = stats_["revenue_growth_3y"]
    if not np.isfinite(growth):
        growth = stats_["revenue_growth"]
    if not np.isfinite(growth):
        growth = 0.05
    growth = float(np.clip(growth, -0.10, 0.35))       # no 60% forever, no death spiral

    latest_margin = stats_["ebit_margin_latest"]
    median_margin = stats_["ebit_margin"]
    if not np.isfinite(latest_margin):
        latest_margin = median_margin
    target_margin = latest_margin + margin_convergence * (median_margin - latest_margin)

    def _clean(value: float, default: float, lo: float, hi: float) -> float:
        return float(np.clip(value, lo, hi)) if np.isfinite(value) else default

    assumptions = Assumptions(
        horizon=horizon,
        revenue_growth_y1=growth,
        ebit_margin_start=_clean(latest_margin, 0.15, -0.20, 0.65),
        ebit_margin_target=_clean(target_margin, 0.15, -0.20, 0.65),
        da_pct_revenue=_clean(stats_["da_pct_revenue"], 0.04, 0.0, 0.30),
        capex_pct_revenue=_clean(stats_["capex_pct_revenue"], 0.05, 0.0, 0.40),
        nwc_pct_revenue=_clean(stats_["nwc_pct_revenue"], 0.05, -0.30, 0.50),
        tax_rate=_clean(stats_["effective_tax_rate"], 0.21, CFG.tax_rate_floor, CFG.tax_rate_cap),
        terminal_growth=terminal_growth,
    )
    log.info(
        "Seeded base case │ growth %s → %s │ margin %s → %s │ capex %s of revenue",
        fmt_pct(assumptions.revenue_growth_y1), fmt_pct(assumptions.terminal_growth),
        fmt_pct(assumptions.ebit_margin_start), fmt_pct(assumptions.ebit_margin_target),
        fmt_pct(assumptions.capex_pct_revenue),
    )
    return assumptions
