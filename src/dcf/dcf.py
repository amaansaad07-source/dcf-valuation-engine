"""The valuation core: projection, discounting, terminal value and the equity bridge.

UFCF = EBIT(1-t) + D&A - Capex - dNWC, discounted at WACC under the mid-year convention.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .assumptions import Assumptions
from .config import fmt_money, fmt_pct, log, npf


@dataclass
class DCFResult:
    """A complete, auditable valuation."""

    projection: pd.DataFrame
    pv_forecast: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    value_per_share: float
    bridge: Dict[str, float]
    wacc: float
    terminal_growth: float
    tv_method: str
    implied_growth_from_multiple: Optional[float]
    implied_exit_multiple: Optional[float]
    tv_share_of_ev: float
    upside_vs_price: Optional[float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def headline(self) -> str:
        return (
            f"Intrinsic value {fmt_money(self.value_per_share, 2)}/share  │  "
            f"EV {fmt_money(self.enterprise_value)}  │  "
            f"WACC {fmt_pct(self.wacc, 2)}  │  g {fmt_pct(self.terminal_growth, 2)}  │  "
            f"TV = {fmt_pct(self.tv_share_of_ev, 0)} of EV"
        )


def project_financials(drivers: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """Build the forecast: revenue → EBIT → NOPAT → UFCF, one column per year."""
    base_revenue = float(drivers["revenue"].iloc[-1])
    base_nwc = float(drivers["nwc"].iloc[-1]) if np.isfinite(drivers["nwc"].iloc[-1]) else \
        a.nwc_pct_revenue * base_revenue

    growth = a.revenue_growth_schedule()
    margin = a.margin_schedule()

    revenue = base_revenue * np.cumprod(1 + growth)
    ebit = revenue * margin
    # Tax applies to positive EBIT only. Taxing a loss at 21% would book a cash refund
    # the company cannot claim — real losses become NOL carryforwards, not cash today.
    taxes = np.maximum(ebit, 0.0) * a.tax_rate
    nopat = ebit - taxes
    d_and_a = revenue * a.da_pct_revenue
    capex = revenue * a.capex_pct_revenue
    ebitda = ebit + d_and_a

    # Working capital scales with revenue; the *change* is the cash item.
    nwc = revenue * a.nwc_pct_revenue
    prior_nwc = np.concatenate([[base_nwc], nwc[:-1]])
    delta_nwc = nwc - prior_nwc

    ufcf = nopat + d_and_a - capex - delta_nwc

    years = np.arange(1, a.horizon + 1)
    discount_periods = years - 0.5 if a.mid_year_convention else years.astype(float)

    frame = pd.DataFrame(
        {
            "Revenue": revenue,
            "Revenue growth": growth,
            "EBIT": ebit,
            "EBIT margin": margin,
            "EBITDA": ebitda,
            "Taxes on EBIT": -taxes,
            "NOPAT": nopat,
            "(+) D&A": d_and_a,
            "(−) Capex": -capex,
            "(−) ΔNWC": -delta_nwc,
            "Unlevered FCF": ufcf,
            "Discount period": discount_periods,
        },
        index=[f"FY+{y}" for y in years],
    ).T
    return frame


def _terminal_value(ufcf_final: float, ebitda_final: float, a: Assumptions,
                    wacc: float) -> Tuple[float, Optional[float], Optional[float]]:
    """Return (terminal value, implied growth if exit-multiple, implied multiple if Gordon)."""
    gordon_tv = ufcf_final * (1 + a.terminal_growth) / (wacc - a.terminal_growth)

    if a.tv_method == "exit_multiple":
        tv = a.exit_ev_ebitda * ebitda_final
        # Back out the perpetual growth the market would have to believe
        implied_g = (tv * wacc - ufcf_final) / (tv + ufcf_final) if (tv + ufcf_final) != 0 else np.nan
        return tv, implied_g, a.exit_ev_ebitda

    implied_multiple = gordon_tv / ebitda_final if ebitda_final else np.nan
    return gordon_tv, a.terminal_growth, implied_multiple


def build_bridge(drivers: pd.DataFrame, enterprise_value: float,
                 include_operating_leases: bool = False) -> Dict[str, float]:
    """EV → equity value. Subtract claims senior to equity; add non-operating assets.

    The logic: enterprise value is the value of the *operating business*. Anything on the
    balance sheet that did not generate the projected UFCF gets added back; anything with
    a claim ahead of common shareholders gets deducted.
    """
    latest = drivers.iloc[-1]

    def val(col: str) -> float:
        v = latest.get(col, np.nan)
        return float(v) if np.isfinite(v) else 0.0

    total_debt = val("total_debt")
    if include_operating_leases:
        total_debt += val("operating_lease_liab")

    bridge = {
        "Enterprise value": enterprise_value,
        "(−) Total debt": -total_debt,
        "(−) Minority interest": -val("minority_interest"),
        "(−) Preferred stock": -val("preferred_stock"),
        "(+) Cash & equivalents": val("cash"),
        "(+) Short-term investments": val("short_term_investments"),
        "(+) Long-term investments": val("long_term_investments"),
        "(+) Equity method investments": val("equity_method_investments"),
    }
    bridge["Equity value"] = sum(bridge.values())
    return bridge


def run_dcf(
    drivers: pd.DataFrame,
    a: Assumptions,
    wacc: float,
    shares: float,
    current_price: Optional[float] = None,
    include_operating_leases: bool = False,
) -> DCFResult:
    """The core valuation: project, discount, terminal value, bridge, per share."""
    problems = a.validate(wacc)
    if any("perpetuity diverges" in p for p in problems):
        raise ValueError(problems[0])
    for p in problems:
        log.warning("Assumption check: %s", p)

    projection = project_financials(drivers, a)
    ufcf = projection.loc["Unlevered FCF"].to_numpy(dtype=float)
    periods = projection.loc["Discount period"].to_numpy(dtype=float)

    discount_factors = 1.0 / (1.0 + wacc) ** periods
    pv_ufcf = ufcf * discount_factors
    pv_forecast = float(pv_ufcf.sum())

    ebitda_final = float(projection.loc["EBITDA"].iloc[-1])
    tv, implied_growth, implied_multiple = _terminal_value(ufcf[-1], ebitda_final, a, wacc)

    # Terminal value is a perpetuity beginning after the final forecast year. Under the
    # mid-year convention the perpetuity's cash flows are also mid-year, so it is
    # discounted at n − 0.5; end-year convention discounts at n.
    tv_period = periods[-1] if a.discount_tv_at_midyear and a.mid_year_convention else float(a.horizon)
    pv_tv = tv / (1 + wacc) ** tv_period

    enterprise_value = pv_forecast + pv_tv
    bridge = build_bridge(drivers, enterprise_value, include_operating_leases)
    equity_value = bridge["Equity value"]
    per_share = equity_value / shares if shares else np.nan

    projection.loc["Discount factor"] = discount_factors
    projection.loc["PV of UFCF"] = pv_ufcf

    diagnostics = {
        "pv_ufcf_by_year": pv_ufcf,
        "ufcf_by_year": ufcf,
        "npv_crosscheck": float(npf.npv(wacc, np.concatenate([[0], ufcf]))) if npf else None,
        "ev_ebitda_entry": enterprise_value / float(drivers["ebitda"].iloc[-1])
        if np.isfinite(drivers["ebitda"].iloc[-1]) and drivers["ebitda"].iloc[-1] > 0 else np.nan,
        "assumption_warnings": problems,
    }

    return DCFResult(
        projection=projection,
        pv_forecast=pv_forecast,
        terminal_value=tv,
        pv_terminal_value=pv_tv,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=per_share,
        bridge=bridge,
        wacc=wacc,
        terminal_growth=a.terminal_growth,
        tv_method=a.tv_method,
        implied_growth_from_multiple=implied_growth,
        implied_exit_multiple=implied_multiple,
        tv_share_of_ev=pv_tv / enterprise_value if enterprise_value else np.nan,
        upside_vs_price=(per_share / current_price - 1) if current_price else None,
        diagnostics=diagnostics,
    )
