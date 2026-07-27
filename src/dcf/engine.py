"""Orchestrator: ticker in, complete valuation out.

``run_valuation("MSFT")`` runs the full pipeline and returns a ``Valuation`` holding every
intermediate artefact for inspection.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .assumptions import Assumptions, seed_assumptions_from_history
from .charts import (
    plot_ev_to_equity_waterfall,
    plot_football_field,
    plot_historical_drivers,
    plot_monte_carlo,
    plot_sensitivity_heatmap,
    plot_ufcf_bridge,
)
from .config import fmt_money, fmt_pct, log
from .dcf import DCFResult, run_dcf
from .diagnostics import valuation_diagnostics
from .drivers import build_driver_table
from .market_data import MarketData, fetch_market_data
from .monte_carlo import MonteCarloResult, monte_carlo_dcf
from .sensitivity import build_sensitivity_ranges, run_scenarios, sensitivity_grid
from .wacc import WACCResult, compute_wacc
from .xbrl_tags import ExtractionReport, fetch_financial_history


@dataclass
class Valuation:
    """The complete output of one valuation run."""

    meta: Dict[str, Any]
    drivers: pd.DataFrame
    report: ExtractionReport
    market: MarketData
    wacc_result: WACCResult
    assumptions: Assumptions
    gordon: DCFResult
    exit_multiple: DCFResult
    sensitivity: pd.DataFrame
    scenarios: pd.DataFrame
    monte_carlo: Optional[MonteCarloResult]
    diagnostics: pd.DataFrame

    def tearsheet(self) -> None:
        """Print the one-screen summary you would paste into an email."""
        g, x, m = self.gordon, self.exit_multiple, self.market
        rule = "─" * 78
        print(rule)
        print(f"  {self.meta['name']} ({self.meta['ticker']})  │  CIK {self.meta['cik']}")
        print(rule)
        print(f"  Current price            {fmt_money(m.price, 2)}")
        print(f"  Market capitalisation    {fmt_money(m.market_cap)}")
        print(f"  WACC                     {fmt_pct(self.wacc_result.wacc, 2)}"
              f"   (Ke {fmt_pct(self.wacc_result.cost_of_equity, 2)}, "
              f"Kd(at) {fmt_pct(self.wacc_result.cost_of_debt_aftertax, 2)})")
        print(f"  Terminal growth          {fmt_pct(self.assumptions.terminal_growth, 2)}")
        print(rule)
        for label, res in (("Gordon growth", g), ("exit multiple", x)):
            upside = f"   ({res.upside_vs_price:+.1%} vs market)" if res.upside_vs_price is not None else ""
            print(f"  DCF — {label:<18s} {fmt_money(res.value_per_share, 2)} / share{upside}")
        if self.monte_carlo:
            mc = self.monte_carlo
            print(f"  Monte Carlo P10–P90      {fmt_money(mc.percentiles['10'], 2)}"
                  f" – {fmt_money(mc.percentiles['90'], 2)}")
            print(f"  P(intrinsic > price)     {mc.prob_above_price:.0%}"
                  f"   across {mc.n_valid:,} paths")
        print(rule)
        print(f"  Terminal value = {fmt_pct(g.tv_share_of_ev, 0)} of enterprise value")
        print(f"  Enterprise value         {fmt_money(g.enterprise_value)}")
        print(f"  Net debt & other         {fmt_money(g.equity_value - g.enterprise_value)}")
        print(f"  Equity value             {fmt_money(g.equity_value)}")
        print(rule)
        failures = self.diagnostics[self.diagnostics["Status"] == "REVIEW"]
        if not failures.empty:
            print("  ⚠️  Checks needing review:")
            for _, row in failures.iterrows():
                print(f"     • {row['Check']}: {row['Value']} (benchmark {row['Benchmark']})")
            print(rule)

    def dashboard(self, figsize: Tuple[int, int] = (17, 15)):
        """Six-panel summary figure — the thing you screenshot into a deck."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.26,
                              top=0.94, bottom=0.05, left=0.07, right=0.96)

        plot_historical_drivers(self.drivers, self.meta["ticker"], ax=fig.add_subplot(gs[0, 0]))
        plot_ufcf_bridge(self.gordon, ax=fig.add_subplot(gs[0, 1]))
        plot_ev_to_equity_waterfall(self.gordon, self.market.shares_diluted,
                                    ax=fig.add_subplot(gs[1, 0]))
        plot_sensitivity_heatmap(self.sensitivity, self.market.price, ax=fig.add_subplot(gs[1, 1]))
        plot_football_field(self.football_field_ranges(), self.market.price,
                            self.gordon.value_per_share, ax=fig.add_subplot(gs[2, 0]))
        if self.monte_carlo:
            plot_monte_carlo(self.monte_carlo, self.market.price,
                             self.gordon.value_per_share, ax=fig.add_subplot(gs[2, 1]))

        fig.suptitle(
            f"{self.meta['name']} ({self.meta['ticker']}) — DCF valuation summary   │   "
            f"{datetime.today():%d %b %Y}",
            fontsize=15, fontweight="bold", y=0.985,
        )
        return fig

    def football_field_ranges(self, peer_ev_ebitda: Tuple[float, float] = None
                              ) -> Dict[str, Tuple[float, float]]:
        """Assemble the comparison ranges: DCF, sensitivity, simulation, comps, 52-week."""
        ranges: Dict[str, Tuple[float, float]] = {}
        grid = self.sensitivity.to_numpy(dtype=float)
        if np.isfinite(grid).any():
            ranges["DCF — WACC × g grid"] = (float(np.nanmin(grid)), float(np.nanmax(grid)))
        ranges["DCF — Gordon vs exit multiple"] = (
            min(self.gordon.value_per_share, self.exit_multiple.value_per_share),
            max(self.gordon.value_per_share, self.exit_multiple.value_per_share),
        )
        if not self.scenarios.empty:
            ranges["Scenarios (bear–bull)"] = (
                float(self.scenarios["Value per share"].min()),
                float(self.scenarios["Value per share"].max()),
            )
        if self.monte_carlo:
            ranges["Monte Carlo P10–P90"] = (
                self.monte_carlo.percentiles["10"], self.monte_carlo.percentiles["90"],
            )
        if peer_ev_ebitda:
            ebitda = float(self.drivers["ebitda"].iloc[-1])
            adj = self.gordon.equity_value - self.gordon.enterprise_value
            shares = self.market.shares_diluted
            ranges["Comps — EV/EBITDA"] = (
                (peer_ev_ebitda[0] * ebitda + adj) / shares,
                (peer_ev_ebitda[1] * ebitda + adj) / shares,
            )
        if self.market.week52_low and self.market.week52_high:
            ranges["52-week trading range"] = (self.market.week52_low, self.market.week52_high)
        return ranges


def run_valuation(
    ticker: str,
    assumptions: Optional[Assumptions] = None,
    horizon: int = 5,
    years_history: int = 10,
    terminal_growth: float = 0.025,
    exit_ev_ebitda: Optional[float] = None,
    erp: Optional[float] = None,
    beta_override: Optional[float] = None,
    price_override: Optional[float] = None,
    shares_override: Optional[float] = None,
    size_premium: float = 0.0,
    n_sims: int = 50_000,
    run_monte_carlo: bool = True,
    peer_ev_ebitda: Optional[Tuple[float, float]] = None,
    verbose: bool = True,
) -> Valuation:
    """Ticker in, full valuation out. Every stage is logged and every artefact retained."""
    t0 = time.perf_counter()

    # 1 ── SEC filings -----------------------------------------------------------------
    history, report, meta = fetch_financial_history(ticker, years_history)
    drivers = build_driver_table(history, report)

    # 2 ── Market inputs ----------------------------------------------------------------
    sec_shares = float(drivers["diluted_shares"].iloc[-1]) \
        if np.isfinite(drivers["diluted_shares"].iloc[-1]) else None
    market = fetch_market_data(
        ticker, sec_shares=sec_shares, price_override=price_override,
        beta_override=beta_override, shares_override=shares_override, erp=erp,
    )

    # 3 ── Discount rate ------------------------------------------------------------------
    wacc_result = compute_wacc(drivers, market, size_premium=size_premium)

    # 4 ── Assumptions ---------------------------------------------------------------------
    a = assumptions or seed_assumptions_from_history(
        drivers, horizon=horizon, terminal_growth=terminal_growth
    )
    if exit_ev_ebitda is None:
        # Default the exit multiple to the company's own current trading multiple
        net_debt = drivers["net_debt"].iloc[-1]
        net_debt = float(net_debt) if np.isfinite(net_debt) else 0.0
        ebitda_now = float(drivers["ebitda"].iloc[-1])
        entry = market.market_cap + net_debt
        exit_ev_ebitda = float(np.clip(entry / ebitda_now, 4.0, 30.0)) if (
            np.isfinite(ebitda_now) and ebitda_now > 0) else 12.0
        log.info("Exit multiple defaulted to the current trading multiple: %.1f×", exit_ev_ebitda)
    a = a.copy_with(exit_ev_ebitda=exit_ev_ebitda)

    # 5 ── Valuation, both terminal-value methods --------------------------------------------
    gordon = run_dcf(drivers, a.copy_with(tv_method="gordon"), wacc_result.wacc,
                     market.shares_diluted, market.price)
    exit_case = run_dcf(drivers, a.copy_with(tv_method="exit_multiple"), wacc_result.wacc,
                        market.shares_diluted, market.price)

    # 6 ── Sensitivity and scenarios ------------------------------------------------------
    wacc_range, growth_range = build_sensitivity_ranges(wacc_result.wacc, a.terminal_growth)
    sensitivity = sensitivity_grid(drivers, a, market.shares_diluted, wacc_range,
                                   growth_range, market.price)
    scenarios = run_scenarios(drivers, a, wacc_result.wacc, market.shares_diluted, market.price)

    # 7 ── Simulation ------------------------------------------------------------------------
    mc = None
    if run_monte_carlo:
        hist_growth_vol = drivers["revenue_growth"].tail(5).std()
        hist_margin_vol = drivers["ebit_margin"].tail(5).std()
        mc = monte_carlo_dcf(
            drivers, a, wacc_result.wacc, market.shares_diluted,
            bridge_adjustment=gordon.equity_value - gordon.enterprise_value,
            current_price=market.price, n_sims=n_sims,
            growth_sd=float(np.clip(hist_growth_vol if np.isfinite(hist_growth_vol) else 0.03, 0.01, 0.12)),
            margin_sd=float(np.clip(hist_margin_vol if np.isfinite(hist_margin_vol) else 0.02, 0.005, 0.08)),
            terminal_growth_bounds=(max(a.terminal_growth - 0.015, 0.0), a.terminal_growth,
                                    a.terminal_growth + 0.007),
        )

    # 8 ── Self-assessment --------------------------------------------------------------------
    diagnostics = valuation_diagnostics(gordon, drivers, report, market, wacc_result)

    valuation = Valuation(
        meta=meta, drivers=drivers, report=report, market=market, wacc_result=wacc_result,
        assumptions=a, gordon=gordon, exit_multiple=exit_case, sensitivity=sensitivity,
        scenarios=scenarios, monte_carlo=mc, diagnostics=diagnostics,
    )
    if peer_ev_ebitda:
        valuation.meta["peer_ev_ebitda"] = peer_ev_ebitda

    log.info("Valuation complete in %.1fs", time.perf_counter() - t0)
    if verbose:
        valuation.tearsheet()
    return valuation


# ---------------------------------------------------------------------------------------
# RUN IT — change the ticker and execute.
# ---------------------------------------------------------------------------------------
# val = run_valuation("MSFT")
# fig = val.dashboard()
# plt.show()
