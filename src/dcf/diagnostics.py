"""Self-scoring quality report.

A valuation model has no R-squared. Quality is data integrity plus internal consistency,
and every row here is a check a reviewer would run before believing the number.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .config import CFG, fmt_money, fmt_pct
from .dcf import DCFResult
from .market_data import MarketData
from .wacc import WACCResult
from .xbrl_tags import ExtractionReport


def valuation_diagnostics(
    result: DCFResult,
    drivers: pd.DataFrame,
    report: ExtractionReport,
    market: MarketData,
    wacc_result: WACCResult,
) -> pd.DataFrame:
    """Self-scoring quality report. Every row is a check a reviewer would run."""
    checks: List[Dict[str, Any]] = []

    def add(metric: str, value: Any, benchmark: str, ok: bool) -> None:
        # Values are coerced to str so the column has a single dtype — mixed int/str
        # columns cannot be serialised to Arrow, which breaks Streamlit's dataframe render.
        checks.append({"Check": metric, "Value": str(value), "Benchmark": benchmark,
                       "Status": "PASS" if ok else "REVIEW"})

    coverage = report.coverage
    add("XBRL coverage — required items", fmt_pct(coverage, 0), "> 80%", coverage > 0.80)
    add("XBRL coverage — all mapped items", fmt_pct(report.coverage_all, 0), "context only", True)
    if report.missing_required:
        add("Required items missing", ", ".join(report.missing_required), "none", False)
    add("Fiscal years extracted", len(drivers), ">= 5", len(drivers) >= 5)

    tv_share = result.tv_share_of_ev
    add("Terminal value share of EV", fmt_pct(tv_share, 0), "< 85%", tv_share < CFG.max_terminal_value_share)

    entry_multiple = result.diagnostics.get("ev_ebitda_entry", np.nan)
    implied_exit = result.implied_exit_multiple
    if np.isfinite(entry_multiple) and implied_exit and np.isfinite(implied_exit):
        drift = implied_exit / entry_multiple - 1
        add("Implied exit vs entry EV/EBITDA",
            f"{implied_exit:.1f}× vs {entry_multiple:.1f}× ({drift:+.0%})",
            "within ±40%", abs(drift) < 0.40)

    if result.tv_method == "exit_multiple" and result.implied_growth_from_multiple is not None:
        g_implied = result.implied_growth_from_multiple
        add("Growth implied by exit multiple", fmt_pct(g_implied, 2), "0% – 4%",
            0 <= g_implied <= 0.04)

    add("WACC", fmt_pct(result.wacc, 2), "6% – 14%", 0.06 <= result.wacc <= 0.14)
    add("WACC − terminal growth spread", fmt_pct(result.wacc - result.terminal_growth, 2),
        "> 3%", (result.wacc - result.terminal_growth) > 0.03)

    # Reinvestment consistency: g ≈ ROIC × reinvestment rate
    roic = drivers["roic"].tail(3).median()
    reinvestment = (drivers["capex"] - drivers["d_and_a"] + drivers["delta_nwc"]).tail(3).median()
    nopat = (drivers["ebit"] * (1 - drivers["effective_tax_rate"])).tail(3).median()
    if np.isfinite(roic) and np.isfinite(nopat) and nopat > 0:
        implied_g = roic * (reinvestment / nopat)
        add("Reinvestment-implied growth", fmt_pct(implied_g, 1),
            f"vs forecast {fmt_pct(result.projection.loc['Revenue growth'].mean(), 1)}",
            abs(implied_g - result.projection.loc["Revenue growth"].mean()) < 0.10)

    # Back-test: our reconstructed UFCF vs reported CFO − capex
    if drivers["cfo"].notna().sum() >= 3:
        reported_fcf = (drivers["cfo"] - drivers["capex"]).tail(5)
        modelled = drivers["ufcf_historical"].tail(5)
        mape = float(np.nanmean(np.abs((modelled - reported_fcf) / reported_fcf.replace(0, np.nan))))
        add("Reconstructed UFCF vs reported FCF (MAPE)", fmt_pct(mape, 0), "< 30%", mape < 0.30)

    add("Implied equity value vs market cap",
        f"{fmt_money(result.equity_value)} vs {fmt_money(market.market_cap)}",
        "sanity check", np.isfinite(result.equity_value))

    if wacc_result.notes:
        add("WACC construction warnings", f"{len(wacc_result.notes)} flagged", "0", False)

    return pd.DataFrame(checks)
