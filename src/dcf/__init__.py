"""Automated DCF valuation engine driven by SEC XBRL filings.

    >>> from dcf import run_valuation
    >>> val = run_valuation("MSFT")
    >>> val.tearsheet()
    >>> val.dashboard()
"""

__version__ = "1.0.0"

from .assumptions import Assumptions, seed_assumptions_from_history
from .charts import (
    plot_driver_tornado,
    plot_ev_to_equity_waterfall,
    plot_football_field,
    plot_historical_drivers,
    plot_monte_carlo,
    plot_sensitivity_heatmap,
    plot_ufcf_bridge,
)
from .config import CFG, PALETTE, Config, fmt_money, fmt_pct
from .dcf import DCFResult, build_bridge, project_financials, run_dcf
from .diagnostics import assess_dcf_suitability, valuation_diagnostics
from .drivers import DRIVER_VIEW, build_driver_table, display_drivers, summarise_drivers
from .engine import Valuation, run_valuation
from .export import export_valuation
from .market_data import (
    MarketData,
    fetch_market_data,
    fetch_risk_free_rate,
    peer_relevered_beta,
    relever_beta,
    unlever_beta,
)
from .monte_carlo import MonteCarloResult, driver_attribution, monte_carlo_dcf
from .sec_client import SEC, SECClient, SECError, TickerNotFound
from .sensitivity import build_sensitivity_ranges, run_scenarios, sensitivity_grid
from .wacc import WACCResult, compute_wacc
from .xbrl_tags import TAG_MAP, ExtractionReport, XBRLExtractor, fetch_financial_history

__all__ = [
    # Orchestration
    "run_valuation", "Valuation",
    # Assumptions
    "Assumptions", "seed_assumptions_from_history",
    # Valuation core
    "run_dcf", "DCFResult", "project_financials", "build_bridge",
    "compute_wacc", "WACCResult",
    # Risk
    "sensitivity_grid", "build_sensitivity_ranges", "run_scenarios",
    "monte_carlo_dcf", "MonteCarloResult", "driver_attribution",
    # Data
    "fetch_financial_history", "build_driver_table", "display_drivers", "summarise_drivers",
    "DRIVER_VIEW", "TAG_MAP", "ExtractionReport", "XBRLExtractor",
    "fetch_market_data", "fetch_risk_free_rate", "MarketData",
    "unlever_beta", "relever_beta", "peer_relevered_beta",
    "SEC", "SECClient", "SECError", "TickerNotFound",
    # Output
    "plot_ev_to_equity_waterfall", "plot_ufcf_bridge", "plot_sensitivity_heatmap",
    "plot_football_field", "plot_monte_carlo", "plot_historical_drivers",
    "plot_driver_tornado", "export_valuation", "valuation_diagnostics", "assess_dcf_suitability",
    # Configuration
    "CFG", "Config", "PALETTE", "fmt_money", "fmt_pct",
]
