"""Global configuration, house chart style and formatting helpers.

Everything the engine treats as a constant lives here, so there are no magic numbers
scattered through the valuation logic. Edit ``CFG.user_agent`` before making any SEC call.
"""
# --- Standard library ---------------------------------------------------------------
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib as mpl

# --- Third party --------------------------------------------------------------------
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

try:
    import numpy_financial as npf
except ImportError:  # pragma: no cover - engine still works without it
    npf = None

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)


# ====================================================================================
# GLOBAL CONFIGURATION
# ====================================================================================
@dataclass
class Config:
    """Single source of truth for every constant the engine touches."""

    # >>> EDIT THIS. The SEC requires a descriptive User-Agent with a real contact. <<<
    user_agent: str = "Amaan Student amaan.student@mail.utoronto.ca"

    # SEC endpoints (no key required)
    sec_ticker_map: str = "https://www.sec.gov/files/company_tickers.json"
    sec_companyfacts: str = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    sec_companyconcept: str = (
        "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"
    )

    # FRED CSV endpoint — keyless, returns the daily series as plain CSV
    fred_csv: str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    rf_series: str = "DGS10"          # 10-year constant maturity Treasury
    rf_fallback: float = 0.0425       # used if FRED is unreachable

    # Equity risk premium. Damodaran's implied US ERP; override per your own view.
    erp_default: float = 0.0450
    damodaran_erp_url: str = (
        "https://pages.stern.nyu.edu/~adamodar/pc/datasets/histimpl.xls"
    )

    # Networking discipline — SEC fair-access policy is 10 requests/second.
    request_timeout: int = 30
    max_retries: int = 4
    backoff_base: float = 1.6
    min_request_interval: float = 0.12

    # Caching. Colab wipes /content on disconnect, which is fine — it is only a cache.
    cache_dir: Path = Path("/content/dcf_cache") if Path("/content").exists() else Path("./dcf_cache")
    cache_ttl_hours: int = 24

    # Analysis defaults
    history_years: int = 10
    period_match_tolerance_days: int = 25   # tolerates 52/53-week fiscal calendars
    accepted_forms: Tuple[str, ...] = ("10-K", "10-K/A", "20-F", "40-F")

    # Guardrails — a DCF that violates these is telling you something is wrong
    min_wacc_spread: float = 0.005    # WACC must exceed g by at least 50bps
    max_terminal_value_share: float = 0.85
    tax_rate_floor: float = 0.08
    tax_rate_cap: float = 0.35


CFG = Config()
CFG.cache_dir.mkdir(parents=True, exist_ok=True)

# --- Logging: one audit trail for every assumption the engine makes automatically ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("dcf")
for noisy in ("matplotlib", "urllib3", "yfinance", "peewee"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ====================================================================================
# HOUSE CHART STYLE — one palette, applied everywhere, so the deck looks like a deck
# ====================================================================================
PALETTE = {
    "ink": "#12222E",
    "navy": "#1B3A5B",
    "blue": "#2E6F9E",
    "sky": "#7FB2D4",
    "teal": "#2E8B84",
    "gold": "#C8922A",
    "red": "#B4423A",
    "grey": "#8C97A0",
    "mist": "#E4E9ED",
    "paper": "#FFFFFF",
}

mpl.rcParams.update({
    "figure.facecolor": PALETTE["paper"],
    "axes.facecolor": PALETTE["paper"],
    "axes.edgecolor": PALETTE["grey"],
    "axes.labelcolor": PALETTE["ink"],
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": PALETTE["ink"],
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": PALETTE["mist"],
    "grid.linewidth": 0.8,
    "xtick.color": PALETTE["ink"],
    "ytick.color": PALETTE["ink"],
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def fmt_money(value: float, decimals: int = 1) -> str:
    """Format a USD figure the way a banker writes it: $1.2bn, $840.5mm, $12.30."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if v >= 1e12:
        return f"{sign}${v / 1e12:,.{decimals}f}tn"
    if v >= 1e9:
        return f"{sign}${v / 1e9:,.{decimals}f}bn"
    if v >= 1e6:
        return f"{sign}${v / 1e6:,.{decimals}f}mm"
    if v >= 1e3:
        return f"{sign}${v / 1e3:,.{decimals}f}k"
    return f"{sign}${v:,.2f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """0.0834 -> '8.3%'."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value * 100:,.{decimals}f}%"


MONEY_AXIS = FuncFormatter(lambda x, _: fmt_money(x, 0))

if "your.email" in CFG.user_agent or "@" not in CFG.user_agent:
    print("  ⚠️  Set CFG.user_agent to 'Your Name your@email.com' before calling the SEC.")
