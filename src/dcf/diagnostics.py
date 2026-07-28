"""Self-scoring quality report.

A valuation model has no R-squared. Quality is data integrity plus internal consistency,
and every row here is a check a reviewer would run before believing the number.
"""

from typing import Any, Dict, List, Tuple

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


def assess_dcf_suitability(drivers: pd.DataFrame) -> Tuple[str, List[str]]:
    """Is a single-stage unlevered DCF the right tool for this company at all?

    Returns a verdict in {"suitable", "caution", "unsuitable"} plus the reasons. This runs
    BEFORE anyone looks at a number, because the worst failure mode is not a crash — it is
    a confident valuation of a company the method does not apply to.
    """
    issues: List[str] = []
    verdict = "suitable"

    def escalate(level: str) -> None:
        nonlocal verdict
        order = {"suitable": 0, "caution": 1, "unsuitable": 2}
        if order[level] > order[verdict]:
            verdict = level

    latest = drivers.iloc[-1]
    ebit_latest = float(latest.get("ebit", np.nan))
    ebit_med3 = float(drivers["ebit"].tail(3).median())
    ebitda_latest = float(latest.get("ebitda", np.nan))

    if np.isfinite(ebit_latest) and np.isfinite(ebit_med3) and ebit_latest < 0 and ebit_med3 < 0:
        escalate("unsuitable")
        issues.append(
            "Operating losses in the latest year AND the 3-year median. A Gordon terminal "
            "value on negative cash flow projects the company burning cash in perpetuity — "
            "mathematically valid, financially meaningless. Pre-profitability names need a "
            "10–15 year horizon with an explicit margin ramp and a terminal value off "
            "normalised EBITDA, not this single-stage model."
        )
    elif (np.isfinite(ebit_latest) and ebit_latest < 0) or (np.isfinite(ebit_med3) and ebit_med3 < 0):
        escalate("caution")
        issues.append(
            "EBIT is negative in at least one recent period. The forecast anchors on "
            "medians that include losses — inspect the seeded margin path before trusting "
            "the output."
        )

    if np.isfinite(ebitda_latest) and ebitda_latest <= 0:
        escalate("unsuitable")
        issues.append(
            "EBITDA is not positive, so the exit-multiple terminal value is undefined and "
            "the Gordon cross-check has nothing to reconcile against."
        )

    capex_med = float(drivers["capex_pct_revenue"].tail(3).median())
    if np.isfinite(capex_med) and capex_med > 0.30:
        escalate("caution")
        issues.append(
            f"Capex is running at {capex_med:.0%} of revenue — build-out phase economics. "
            "Steady-state intensity is likely far lower; the seeded value will overstate "
            "perpetual reinvestment."
        )

    growth_med = float(drivers["revenue_growth"].tail(3).median())
    if np.isfinite(growth_med) and growth_med > 0.40:
        escalate("caution")
        issues.append(
            f"Median revenue growth of {growth_med:.0%} cannot persist. The five-year fade "
            "to terminal growth compresses a long transition into a short window."
        )

    if len(drivers) < 4:
        escalate("caution")
        issues.append(
            f"Only {len(drivers)} fiscal years of history — the medians anchoring the "
            "forecast are closer to anecdotes than statistics."
        )

    ufcf3 = drivers["ufcf_historical"].tail(3)
    if ufcf3.notna().all() and (ufcf3 < 0).all():
        escalate("unsuitable" if verdict == "unsuitable" else "caution")
        issues.append("Unlevered FCF has been negative in each of the last three years.")

    return verdict, issues
