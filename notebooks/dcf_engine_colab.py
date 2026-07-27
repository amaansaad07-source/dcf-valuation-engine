# %% [markdown]
# # Automated DCF Valuation Engine from SEC Filings
#
# **Ticker in → intrinsic value per share out.** Pulls 10 years of reported financials
# straight from the SEC's XBRL `companyfacts` API, rebuilds the unlevered free cash flow
# drivers, computes WACC from CAPM, discounts to enterprise value, bridges to equity
# value per share, then stress-tests the answer with a two-way sensitivity grid and a
# vectorised Monte Carlo simulation.
#
# ---
# **How to run in Google Colab**
# 1. `File → New notebook` (or `File → Upload notebook` and drop this `.ipynb` in).
# 2. Run the cells **in order, top to bottom**. Each cell is self-contained and prints
#    what it built so you can verify before moving on.
# 3. In **Cell 1**, replace `USER_AGENT` with your real name and email. The SEC blocks
#    requests without a valid contact string — this is their only "API key".
# 4. Jump to **Cell 13** to run a full valuation on any ticker.
#
# **Author:** _your name_ · **Licence:** MIT · **Not investment advice.**

# %% [markdown]
# ## Cell 1 — Environment setup, configuration and house style
#
# Colab already ships `pandas`, `numpy`, `scipy` and `matplotlib`. We add `yfinance`
# (market data) and `numpy-financial` (NPV/IRR cross-checks). Everything is pinned to a
# single config object so there are no magic numbers scattered through the engine.

# %%
# --- Install the two libraries Colab does not ship with -----------------------------
import importlib
import subprocess
import sys


def _ensure(package: str, import_name: str | None = None) -> None:
    """Install a package only if it is not already importable.

    Colab re-runs cells often; a naive `!pip install` costs ~15s every time.
    """
    try:
        importlib.import_module(import_name or package)
    except ImportError:
        print(f"Installing {package} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", package],
            check=False,
        )


_ensure("yfinance")
_ensure("numpy-financial", "numpy_financial")

# --- Standard library ---------------------------------------------------------------
import json
import logging
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --- Third party --------------------------------------------------------------------
import numpy as np
import pandas as pd
import requests
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

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

print("Environment ready.")
print(f"  numpy {np.__version__} │ pandas {pd.__version__} │ cache → {CFG.cache_dir}")
if "your.email" in CFG.user_agent or "@" not in CFG.user_agent:
    print("  ⚠️  Set CFG.user_agent to 'Your Name your@email.com' before calling the SEC.")


# %% [markdown]
# ## Cell 2 — SEC XBRL client
#
# The SEC exposes every tagged number in every filing through one JSON endpoint per
# company. This wrapper handles the three things that break naive scrapers:
#
# | Problem | Handling |
# |---|---|
# | No `User-Agent` → **403 Forbidden** | Required header injected on every call |
# | Fair-access rate limit (10 req/s) | Token-bucket sleep between requests |
# | Transient 429/503 | Exponential backoff, 4 attempts |
# | 40MB `companyfacts` payloads | Disk cache with a 24h TTL |
#
# The ticker → CIK map is a single ~1MB file covering every SEC registrant.

# %%
class SECError(RuntimeError):
    """Raised when the SEC API cannot satisfy a request."""


class TickerNotFound(SECError):
    """Raised when a ticker has no CIK in the SEC registrant map."""


class SECClient:
    """Thin, polite, cached client for the SEC XBRL REST API."""

    def __init__(self, config: Config = CFG):
        self.cfg = config
        self.session = requests.Session()
        # The SEC's only requirement is a descriptive User-Agent with a real contact.
        # `Host` is deliberately NOT set — requests derives it from the URL, and hardcoding
        # it breaks the moment you call www.sec.gov instead of data.sec.gov.
        self.session.headers.update({
            "User-Agent": config.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_call = 0.0
        self._ticker_map: Optional[pd.DataFrame] = None

    # ---------------------------------------------------------------- internals ----
    def _throttle(self) -> None:
        """Keep us comfortably inside the SEC's 10 requests/second ceiling."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.cfg.min_request_interval:
            time.sleep(self.cfg.min_request_interval - elapsed)
        self._last_call = time.monotonic()

    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:120]
        return self.cfg.cache_dir / f"{safe}.json"

    def _read_cache(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.cfg.cache_ttl_hours:
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                log.info("Cache hit: %s (age %.1fh)", key, age_hours)
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)   # corrupt cache entry — drop it
            return None

    def _write_cache(self, key: str, payload: dict) -> None:
        try:
            with self._cache_path(key).open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except OSError as exc:  # a full disk should never kill a valuation
            log.warning("Could not cache %s: %s", key, exc)

    def _get_json(self, url: str, cache_key: Optional[str] = None) -> dict:
        """GET with throttling, retries and optional disk cache."""
        if cache_key:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        last_error: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.cfg.request_timeout)
                if resp.status_code == 200:
                    payload = resp.json()
                    if cache_key:
                        self._write_cache(cache_key, payload)
                    return payload
                if resp.status_code == 404:
                    raise SECError(f"404 Not Found — the SEC has no data at {url}")
                if resp.status_code == 403:
                    raise SECError(
                        "403 Forbidden — the SEC rejected the User-Agent. "
                        "Set CFG.user_agent to 'Your Name your@email.com'."
                    )
                last_error = SECError(f"HTTP {resp.status_code}")
            except requests.RequestException as exc:
                last_error = exc

            wait = self.cfg.backoff_base ** attempt
            log.warning("SEC request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, self.cfg.max_retries, last_error, wait)
            time.sleep(wait)

        raise SECError(f"Gave up on {url} after {self.cfg.max_retries} attempts: {last_error}")

    # ------------------------------------------------------------------ public ----
    def ticker_map(self) -> pd.DataFrame:
        """Full SEC registrant map: ticker → CIK → company name."""
        if self._ticker_map is None:
            raw = self._get_json(self.cfg.sec_ticker_map, cache_key="company_tickers")
            frame = pd.DataFrame(raw).T
            frame.columns = [c.lower() for c in frame.columns]
            frame["ticker"] = frame["ticker"].str.upper()
            frame["cik_str"] = frame["cik_str"].astype(int)
            self._ticker_map = frame.rename(columns={"cik_str": "cik", "title": "name"})
            log.info("Loaded SEC registrant map: %d tickers", len(self._ticker_map))
        return self._ticker_map

    def resolve_cik(self, ticker: str) -> Tuple[int, str]:
        """'AAPL' → (320193, 'Apple Inc.'). Raises TickerNotFound if unlisted."""
        ticker = ticker.strip().upper()
        frame = self.ticker_map()
        hit = frame.loc[frame["ticker"] == ticker]
        if hit.empty:
            # Class-share tickers often use a dot or dash: BRK.B / BRK-B
            alt = ticker.replace(".", "-")
            hit = frame.loc[frame["ticker"] == alt]
        if hit.empty:
            raise TickerNotFound(
                f"'{ticker}' is not in the SEC registrant map. Foreign issuers without "
                "US listings and private companies will not appear."
            )
        row = hit.iloc[0]
        return int(row["cik"]), str(row["name"])

    def company_facts(self, cik: int) -> dict:
        """Every XBRL fact the company has ever tagged, in one payload."""
        url = self.cfg.sec_companyfacts.format(cik=cik)
        payload = self._get_json(url, cache_key=f"facts_{cik}")
        if "facts" not in payload:
            raise SECError(f"companyfacts payload for CIK {cik} contains no 'facts' block")
        return payload

    def company_concept(self, cik: int, tag: str) -> dict:
        """One tag's full history — useful for debugging a stubborn line item."""
        url = self.cfg.sec_companyconcept.format(cik=cik, tag=tag)
        return self._get_json(url, cache_key=f"concept_{cik}_{tag}")


SEC = SECClient()
print("SEC client initialised. Endpoints:")
print("  ticker map     →", CFG.sec_ticker_map)
print("  company facts  →", CFG.sec_companyfacts.format(cik=320193))


# %% [markdown]
# ## Cell 3 — XBRL tag dictionary and fact extraction
#
# **This is the part that eats 60% of the build time**, and it is the part that separates
# a working tool from a demo. There is no single "Revenue" tag: Apple reports
# `RevenueFromContractWithCustomerExcludingAssessedTax`, older filers use `Revenues` or
# `SalesRevenueNet`, and some use segment-level tags only.
#
# The approach: an **ordered fallback chain per line item**. The first tag that produces a
# usable series wins, and the choice is logged so the valuation is auditable.
#
# Two more subtleties handled here:
# - **Duration vs instant facts.** Income and cash-flow items span a period (`start`→`end`);
#   balance-sheet items are point-in-time (`end` only). Mixing them silently corrupts NWC.
# - **Fiscal-year alignment.** Apple's FY ends in late September and drifts by a few days
#   each year. We anchor on revenue's period-end dates and match every other tag to the
#   nearest anchor within ±25 days, instead of naively grouping by calendar year.

# %%
# Ordered fallback chains. Earlier entries are preferred; the engine walks down the list
# until it finds a tag with enough annual observations.
TAG_MAP: Dict[str, List[str]] = {
    # ---- Income statement (duration) ----------------------------------------------
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RevenuesNetOfInterestExpense",
    ],
    "cogs": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfServices",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_expenses": [
        "OperatingExpenses",
        "CostsAndExpenses",
    ],
    "ebit": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    "tax_expense": ["IncomeTaxExpenseBenefit", "CurrentIncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
        "InterestExpenseNonoperating",
        "InterestIncomeExpenseNet",
    ],
    # ---- Cash flow statement (duration) -------------------------------------------
    "d_and_a": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "amortisation_intangibles": ["AmortizationOfIntangibleAssets"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    ],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "diluted_shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstandingAdjustment",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ],
    # ---- Balance sheet (instant) ---------------------------------------------------
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndDueFromBanks",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
        "MarketableSecuritiesCurrent",
        "OtherShortTermInvestments",
    ],
    "long_term_investments": [
        "LongTermInvestments",
        "MarketableSecuritiesNoncurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent",
    ],
    "equity_method_investments": ["EquityMethodInvestments"],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsAndOtherReceivablesNetCurrent",
    ],
    "inventory": ["InventoryNet", "InventoryFinishedGoodsNetOfReserves"],
    "other_current_assets": ["OtherAssetsCurrent", "PrepaidExpenseAndOtherAssetsCurrent"],
    "current_assets": ["AssetsCurrent"],
    "payables": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "accrued_liabilities": [
        "AccruedLiabilitiesCurrent",
        "EmployeeRelatedLiabilitiesCurrent",
    ],
    "deferred_revenue_current": [
        "ContractWithCustomerLiabilityCurrent",
        "DeferredRevenueCurrent",
    ],
    "other_current_liabilities": ["OtherLiabilitiesCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "short_term_debt": [
        "ShortTermBorrowings",
        "OtherShortTermBorrowings",
        "CommercialPaper",
        "DebtCurrent",
    ],
    "current_portion_ltd": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "finance_lease_liab": [
        "FinanceLeaseLiabilityNoncurrent",
        "CapitalLeaseObligationsNoncurrent",
    ],
    "operating_lease_liab": ["OperatingLeaseLiabilityNoncurrent"],
    "minority_interest": ["MinorityInterest", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "preferred_stock": ["PreferredStockValue", "TemporaryEquityCarryingAmountAttributableToParent"],
    "total_assets": ["Assets"],
    "total_equity": ["StockholdersEquity"],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
}

# Which chains are period flows vs point-in-time balances.
DURATION_ITEMS = {
    "revenue", "cogs", "gross_profit", "operating_expenses", "ebit", "pretax_income",
    "tax_expense", "net_income", "interest_expense", "d_and_a", "amortisation_intangibles",
    "capex", "sbc", "cfo", "diluted_shares",
}
SHARE_ITEMS = {"diluted_shares"}

# Coverage is scored against the items the valuation genuinely cannot run without.
# Everything else (preferred stock, finance leases, equity-method investments) is
# legitimately absent for most filers and should not drag the quality score down.
REQUIRED_ITEMS = {
    "revenue", "ebit", "d_and_a", "capex", "tax_expense", "pretax_income",
    "cash", "long_term_debt", "current_assets", "current_liabilities", "diluted_shares",
}


@dataclass
class ExtractionReport:
    """Audit trail: which tag supplied which line item, and what was missing."""

    resolved: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    derived: Dict[str, str] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Share of *required* line items sourced from reported tags."""
        found = len(REQUIRED_ITEMS & set(self.resolved) | (REQUIRED_ITEMS & set(self.derived)))
        return found / len(REQUIRED_ITEMS)

    @property
    def coverage_all(self) -> float:
        """Share of every mapped line item, required and optional."""
        total = len(self.resolved) + len(self.missing)
        return len(self.resolved) / total if total else 0.0

    @property
    def missing_required(self) -> List[str]:
        return sorted(REQUIRED_ITEMS - set(self.resolved) - set(self.derived))

    def to_frame(self) -> pd.DataFrame:
        rows = [{"line_item": k, "status": "reported", "detail": v} for k, v in self.resolved.items()]
        rows += [{"line_item": k, "status": "derived", "detail": v} for k, v in self.derived.items()]
        rows += [{"line_item": k, "status": "missing", "detail": "no usable tag"} for k in self.missing]
        return pd.DataFrame(rows).sort_values(["status", "line_item"]).reset_index(drop=True)


class XBRLExtractor:
    """Turns a raw `companyfacts` payload into a clean annual financial history."""

    def __init__(self, facts: dict, config: Config = CFG):
        self.cfg = config
        self.entity = facts.get("entityName", "Unknown")
        self.us_gaap: dict = facts.get("facts", {}).get("us-gaap", {})
        self.dei: dict = facts.get("facts", {}).get("dei", {})
        self.report = ExtractionReport()
        if not self.us_gaap:
            raise SECError(
                f"{self.entity} reports no us-gaap facts — likely an IFRS foreign filer. "
                "Try a US-domiciled peer."
            )

    # ------------------------------------------------------------- fact plumbing ---
    @staticmethod
    def _units_for(tag_block: dict) -> Optional[List[dict]]:
        """Pick USD (or shares) from the tag's unit dictionary."""
        for unit in ("USD", "shares", "USD/shares", "pure"):
            if unit in tag_block.get("units", {}):
                return tag_block["units"][unit]
        return None

    def _annual_facts(self, tag: str, duration: bool) -> Dict[date, float]:
        """All annual observations for one tag, keyed by period-end date.

        Rules applied:
        - only annual report forms (10-K, 20-F, 40-F and amendments)
        - duration facts must span 300–400 days (i.e. a full year, not a quarter)
        - when the same period is restated, keep the most recently *filed* value
        """
        block = self.us_gaap.get(tag)
        if not block:
            return {}
        entries = self._units_for(block)
        if not entries:
            return {}

        best: Dict[date, Tuple[str, float]] = {}
        for item in entries:
            if item.get("form") not in self.cfg.accepted_forms:
                continue
            end_raw = item.get("end")
            val = item.get("val")
            if end_raw is None or val is None:
                continue
            try:
                end = datetime.strptime(end_raw, "%Y-%m-%d").date()
            except ValueError:
                continue

            if duration:
                start_raw = item.get("start")
                if not start_raw:
                    continue          # an instant fact masquerading in a flow chain
                start = datetime.strptime(start_raw, "%Y-%m-%d").date()
                span = (end - start).days
                if not 300 <= span <= 400:
                    continue          # quarterly, half-year or multi-year — discard
            elif item.get("start"):
                continue              # a flow fact in a balance chain

            filed = item.get("filed", "0000-00-00")
            prev = best.get(end)
            if prev is None or filed > prev[0]:
                best[end] = (filed, float(val))

        return {end: v for end, (_, v) in best.items()}

    def _resolve_chain(self, item: str, duration: bool, min_obs: int = 3) -> Dict[date, float]:
        """Walk the fallback chain for one line item; first adequate tag wins."""
        for tag in TAG_MAP[item]:
            series = self._annual_facts(tag, duration)
            if len(series) >= min_obs:
                self.report.resolved[item] = tag
                return series
        # Last resort: accept any tag with at least one observation
        for tag in TAG_MAP[item]:
            series = self._annual_facts(tag, duration)
            if series:
                self.report.resolved[item] = f"{tag} (sparse: {len(series)} obs)"
                return series
        self.report.missing.append(item)
        return {}

    # ------------------------------------------------------------------- public ---
    def anchor_periods(self, years: int) -> List[date]:
        """Fiscal year-end dates, taken from revenue — the one tag every filer reports."""
        revenue = self._resolve_chain("revenue", duration=True)
        if not revenue:
            raise SECError(
                f"No usable revenue tag for {self.entity}. Financials and REITs often "
                "need a bespoke tag chain."
            )
        return sorted(revenue.keys())[-years:]

    def build(self, years: int = None) -> pd.DataFrame:
        """Assemble the annual financial history, one column per line item."""
        years = years or self.cfg.history_years
        anchors = self.anchor_periods(years)
        tol = timedelta(days=self.cfg.period_match_tolerance_days)

        data: Dict[str, List[float]] = {}
        for item in TAG_MAP:
            series = self._resolve_chain(item, duration=item in DURATION_ITEMS)
            column: List[float] = []
            for anchor in anchors:
                # Match the observation whose period end is closest to this fiscal year end
                candidates = [(abs(d - anchor), v) for d, v in series.items() if abs(d - anchor) <= tol]
                column.append(min(candidates)[1] if candidates else np.nan)
            data[item] = column

        frame = pd.DataFrame(data, index=pd.DatetimeIndex(anchors, name="period_end"))
        frame.insert(0, "fiscal_year", [d.year if d.month > 6 else d.year - 1 for d in anchors])
        log.info(
            "Extracted %d fiscal years for %s │ required-tag coverage %.0f%% "
            "(%.0f%% across all mapped items)",
            len(frame), self.entity, self.report.coverage * 100,
            self.report.coverage_all * 100,
        )
        if self.report.missing_required:
            log.warning("Required items with no reported tag: %s — the driver table will "
                        "derive or zero-fill them", ", ".join(self.report.missing_required))
        return frame


def fetch_financial_history(ticker: str, years: int = CFG.history_years
                            ) -> Tuple[pd.DataFrame, ExtractionReport, Dict[str, Any]]:
    """End-to-end: ticker → cleaned annual financial history + audit report."""
    cik, name = SEC.resolve_cik(ticker)
    log.info("Resolved %s → CIK %d (%s)", ticker.upper(), cik, name)
    facts = SEC.company_facts(cik)
    extractor = XBRLExtractor(facts)
    history = extractor.build(years)
    meta = {"ticker": ticker.upper(), "cik": cik, "name": name, "entity": extractor.entity}
    return history, extractor.report, meta


print(f"Tag dictionary loaded: {len(TAG_MAP)} line items, "
      f"{sum(len(v) for v in TAG_MAP.values())} candidate XBRL tags.")

# %% [markdown]
# ## Cell 4 — Historical driver table
#
# Raw tags are not a model. This cell reconstructs the five inputs a DCF actually needs —
# **revenue, EBIT, D&A, capex, and change in net working capital** — and converts them into
# the ratios that anchor the forecast.
#
# Definitions used (state these in your README; interviewers ask):
# - **Operating NWC** = (current assets − cash − short-term investments)
#   − (current liabilities − short-term debt − current portion of long-term debt).
#   Cash and debt are excluded because they are *financing*, not operations — including
#   them would double-count what the EV→equity bridge already handles.
# - **ΔNWC** is the year-over-year *increase*, and an increase is a **use** of cash.
# - **Effective tax rate** = tax expense ÷ pre-tax income, winsorised to 8–35% so that a
#   one-off tax benefit does not produce a negative tax rate in the forecast.
# - **Total debt** = short-term debt + current portion of LTD + long-term debt + finance
#   leases. Operating leases are captured separately and included optionally.

# %%
def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that returns NaN instead of exploding on zeros."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def build_driver_table(history: pd.DataFrame, report: ExtractionReport) -> pd.DataFrame:
    """Convert raw tagged line items into the DCF driver table."""
    df = history.copy()

    # ---- Fill structural gaps with accounting identities ---------------------------
    # EBIT: if OperatingIncomeLoss is absent, rebuild it from pre-tax income + interest.
    if df["ebit"].isna().all() and df["pretax_income"].notna().any():
        df["ebit"] = df["pretax_income"] + df["interest_expense"].fillna(0)
        report.derived["ebit"] = "pretax income + interest expense"
    if df["ebit"].isna().all() and df["gross_profit"].notna().any():
        df["ebit"] = df["gross_profit"] - df["operating_expenses"].fillna(0)
        report.derived["ebit"] = "gross profit − operating expenses"

    # D&A: some filers only tag depreciation and amortisation separately.
    if df["d_and_a"].isna().all():
        combined = df["amortisation_intangibles"].fillna(0)
        if combined.abs().sum() > 0:
            df["d_and_a"] = combined
            report.derived["d_and_a"] = "amortisation of intangibles only (depreciation untagged)"

    # ---- Sign conventions ----------------------------------------------------------
    # XBRL reports capex as a positive "payment"; the model needs a positive outflow.
    df["capex"] = df["capex"].abs()
    df["d_and_a"] = df["d_and_a"].abs()
    df["interest_expense"] = df["interest_expense"].abs()

    # ---- Capital structure ---------------------------------------------------------
    debt_parts = ["short_term_debt", "current_portion_ltd", "long_term_debt", "finance_lease_liab"]
    df["total_debt"] = df[debt_parts].fillna(0).sum(axis=1)
    df.loc[df[debt_parts].isna().all(axis=1), "total_debt"] = np.nan

    df["cash_and_investments"] = (
        df["cash"].fillna(0) + df["short_term_investments"].fillna(0)
    )
    df["net_debt"] = df["total_debt"] - df["cash_and_investments"]

    # ---- Operating net working capital ---------------------------------------------
    detailed_assets = df[["receivables", "inventory", "other_current_assets"]].fillna(0).sum(axis=1)
    detailed_liabs = df[
        ["payables", "accrued_liabilities", "deferred_revenue_current", "other_current_liabilities"]
    ].fillna(0).sum(axis=1)

    have_detail = (detailed_assets > 0) & (detailed_liabs > 0)
    aggregate_nwc = (
        (df["current_assets"] - df["cash"].fillna(0) - df["short_term_investments"].fillna(0))
        - (df["current_liabilities"] - df["short_term_debt"].fillna(0) - df["current_portion_ltd"].fillna(0))
    )
    df["nwc"] = np.where(have_detail, detailed_assets - detailed_liabs, aggregate_nwc)
    report.derived["nwc"] = (
        "line-item build (AR + inventory + other CA − AP − accruals − deferred rev − other CL)"
        if have_detail.any()
        else "aggregate build (current assets − cash − ST inv − current liabs + ST debt)"
    )
    df["delta_nwc"] = df["nwc"].diff()          # positive = cash absorbed by working capital

    # ---- Profitability and intensity ratios ----------------------------------------
    df["revenue_growth"] = df["revenue"].pct_change()
    df["gross_margin"] = _safe_div(df["gross_profit"], df["revenue"])
    df["ebit_margin"] = _safe_div(df["ebit"], df["revenue"])
    df["ebitda"] = df["ebit"] + df["d_and_a"].fillna(0)
    df["ebitda_margin"] = _safe_div(df["ebitda"], df["revenue"])
    df["da_pct_revenue"] = _safe_div(df["d_and_a"], df["revenue"])
    df["capex_pct_revenue"] = _safe_div(df["capex"], df["revenue"])
    df["nwc_pct_revenue"] = _safe_div(df["nwc"], df["revenue"])
    df["delta_nwc_pct_delta_rev"] = _safe_div(df["delta_nwc"], df["revenue"].diff())

    # Effective tax rate, winsorised so a one-off benefit cannot poison the forecast
    raw_tax = _safe_div(df["tax_expense"], df["pretax_income"])
    df["effective_tax_rate"] = raw_tax.clip(CFG.tax_rate_floor, CFG.tax_rate_cap)

    # Historical unlevered FCF — the actual track record we are forecasting forward
    df["ufcf_historical"] = (
        df["ebit"] * (1 - df["effective_tax_rate"])
        + df["d_and_a"].fillna(0)
        - df["capex"].fillna(0)
        - df["delta_nwc"].fillna(0)
    )
    df["fcf_conversion"] = _safe_div(df["ufcf_historical"], df["ebitda"])
    df["roic"] = _safe_div(
        df["ebit"] * (1 - df["effective_tax_rate"]),
        (df["total_equity"].fillna(0) + df["total_debt"].fillna(0) - df["cash_and_investments"]),
    )

    return df


DRIVER_VIEW = [
    "fiscal_year", "revenue", "revenue_growth", "ebit", "ebit_margin", "ebitda",
    "d_and_a", "da_pct_revenue", "capex", "capex_pct_revenue", "nwc",
    "nwc_pct_revenue", "delta_nwc", "effective_tax_rate", "ufcf_historical",
]


def display_drivers(drivers: pd.DataFrame, unit: str = "mm") -> pd.DataFrame:
    """Human-readable driver table: dollars scaled to millions/billions, ratios as %."""
    scale = {"mm": 1e6, "bn": 1e9, "k": 1e3}.get(unit, 1e6)
    dollar_cols = ["revenue", "ebit", "ebitda", "d_and_a", "capex", "nwc",
                   "delta_nwc", "ufcf_historical"]
    ratio_cols = ["revenue_growth", "ebit_margin", "da_pct_revenue",
                  "capex_pct_revenue", "nwc_pct_revenue", "effective_tax_rate"]

    view = drivers[DRIVER_VIEW].copy()
    view.index = view["fiscal_year"].astype(int).astype(str)
    view = view.drop(columns="fiscal_year")
    for col in dollar_cols:
        if col in view:
            view[col] = (view[col] / scale).round(0)
            view = view.rename(columns={col: f"{col} ($" + unit + ")"})
    for col in ratio_cols:
        if col in view:
            view[col] = (view[col] * 100).round(1)
            view = view.rename(columns={col: f"{col} (%)"})
    return view.T


def summarise_drivers(drivers: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Median of the last N years for each forecast driver — the forecast anchor.

    Medians, not means: a single pandemic year or a large acquisition should not set
    the base case. Interviewers will ask why; this is the answer.
    """
    tail = drivers.tail(lookback)
    return pd.Series({
        "revenue_growth": tail["revenue_growth"].median(),
        "revenue_growth_3y": drivers.tail(3)["revenue_growth"].median(),
        "ebit_margin": tail["ebit_margin"].median(),
        "ebit_margin_latest": drivers["ebit_margin"].iloc[-1],
        "da_pct_revenue": tail["da_pct_revenue"].median(),
        "capex_pct_revenue": tail["capex_pct_revenue"].median(),
        "nwc_pct_revenue": tail["nwc_pct_revenue"].median(),
        "effective_tax_rate": tail["effective_tax_rate"].median(),
        "revenue_growth_vol": tail["revenue_growth"].std(),
        "ebit_margin_vol": tail["ebit_margin"].std(),
        "fcf_conversion": tail["fcf_conversion"].median(),
        "roic": tail["roic"].median(),
    })


print("Driver-table builder ready: revenue → EBIT → D&A → capex → ΔNWC → historical UFCF.")


# %% [markdown]
# ## Cell 5 — Market data: price, shares, beta, risk-free rate, ERP
#
# Three sources, each with a graceful fallback so a flaky API never kills a run:
#
# | Input | Primary | Fallback |
# |---|---|---|
# | Price, beta, 52-week range | Yahoo Finance (`yfinance`) | manual override argument |
# | Diluted share count | Yahoo | SEC weighted-average diluted shares |
# | Risk-free rate | FRED `DGS10` CSV (keyless) | `CFG.rf_fallback` |
# | Equity risk premium | Damodaran implied ERP | `CFG.erp_default` |
#
# **On beta:** Yahoo's beta is a 5-year monthly regression against the S&P 500. It is
# noisy for small caps and meaningless for recent IPOs. The engine flags anything outside
# 0.3–2.5 and lets you override — bankers usually use a peer-median unlevered beta,
# re-levered to the target capital structure. That helper is included below.

# %%
@dataclass
class MarketData:
    """Everything the valuation needs from outside the filings."""

    ticker: str
    price: float
    shares_diluted: float
    beta: float
    risk_free_rate: float
    equity_risk_premium: float
    market_cap: float
    week52_low: Optional[float] = None
    week52_high: Optional[float] = None
    currency: str = "USD"
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> pd.Series:
        return pd.Series({
            "Price": self.price,
            "Diluted shares": self.shares_diluted,
            "Market cap": self.market_cap,
            "Beta (levered)": self.beta,
            "Risk-free rate": self.risk_free_rate,
            "Equity risk premium": self.equity_risk_premium,
            "52-week low": self.week52_low,
            "52-week high": self.week52_high,
        })


def fetch_risk_free_rate(series: str = None) -> float:
    """Latest 10-year Treasury yield from FRED's keyless CSV endpoint."""
    series = series or CFG.rf_series
    try:
        url = CFG.fred_csv.format(series=series)
        frame = pd.read_csv(url)
        value_col = frame.columns[-1]
        clean = pd.to_numeric(frame[value_col], errors="coerce").dropna()
        if clean.empty:
            raise ValueError("FRED returned no numeric observations")
        rate = float(clean.iloc[-1]) / 100.0
        log.info("Risk-free rate (%s): %.2f%%", series, rate * 100)
        return rate
    except Exception as exc:
        log.warning("FRED unavailable (%s) — using fallback %.2f%%", exc, CFG.rf_fallback * 100)
        return CFG.rf_fallback


def fetch_market_data(
    ticker: str,
    sec_shares: Optional[float] = None,
    price_override: Optional[float] = None,
    beta_override: Optional[float] = None,
    shares_override: Optional[float] = None,
    erp: Optional[float] = None,
) -> MarketData:
    """Market inputs with layered fallbacks; every substitution is recorded."""
    notes: List[str] = []
    price = beta = shares = low = high = None
    currency = "USD"

    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:                                   # info blocks are flaky; use history
            hist = tk.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        beta = info.get("beta")
        shares = info.get("sharesOutstanding")
        low = info.get("fiftyTwoWeekLow")
        high = info.get("fiftyTwoWeekHigh")
        currency = info.get("currency", "USD")

        if low is None or high is None:                     # rebuild the range from prices
            hist = tk.history(period="1y")
            if not hist.empty:
                low, high = float(hist["Low"].min()), float(hist["High"].max())
    except Exception as exc:
        notes.append(f"Yahoo Finance unavailable ({type(exc).__name__}); using overrides/SEC data.")
        log.warning("yfinance failed: %s", exc)

    # ---- Overrides always win -------------------------------------------------------
    price = price_override or price
    beta = beta_override or beta
    shares = shares_override or shares or sec_shares

    if price is None:
        raise ValueError(
            "No price available. Pass price_override=<float> — Yahoo intermittently "
            "throttles Colab IP ranges."
        )
    if shares is None:
        raise ValueError("No share count available. Pass shares_override=<float>.")
    if beta is None:
        beta = 1.0
        notes.append("Beta unavailable — defaulted to 1.0. Override with a peer-median beta.")
    elif not 0.3 <= beta <= 2.5:
        notes.append(f"Beta of {beta:.2f} is outside 0.3–2.5; verify against peers before relying on it.")

    if sec_shares and shares and abs(shares - sec_shares) / sec_shares > 0.10:
        notes.append(
            f"Yahoo share count ({shares/1e6:,.0f}mm) differs from SEC diluted "
            f"({sec_shares/1e6:,.0f}mm) by >10% — check for buybacks or a recent issuance."
        )

    data = MarketData(
        ticker=ticker.upper(),
        price=float(price),
        shares_diluted=float(shares),
        beta=float(beta),
        risk_free_rate=fetch_risk_free_rate(),
        equity_risk_premium=erp if erp is not None else CFG.erp_default,
        market_cap=float(price) * float(shares),
        week52_low=float(low) if low else None,
        week52_high=float(high) if high else None,
        currency=currency,
        warnings=notes,
    )
    for note in notes:
        log.warning(note)
    return data


def unlever_beta(levered_beta: float, debt_to_equity: float, tax_rate: float) -> float:
    """Hamada: βu = βl / (1 + (1 − t)·D/E). Strips out financing risk."""
    return levered_beta / (1 + (1 - tax_rate) * debt_to_equity)


def relever_beta(unlevered_beta: float, target_de: float, tax_rate: float) -> float:
    """Hamada in reverse: βl = βu · (1 + (1 − t)·D/E) at the target structure."""
    return unlevered_beta * (1 + (1 - tax_rate) * target_de)


def peer_relevered_beta(peer_betas: Dict[str, Tuple[float, float]],
                        target_de: float, tax_rate: float) -> float:
    """Median peer beta, unlevered at each peer's own D/E then re-levered to the target.

    peer_betas: {"MSFT": (levered_beta, debt_to_equity), ...}
    """
    unlevered = [unlever_beta(b, de, tax_rate) for b, de in peer_betas.values()]
    return relever_beta(float(np.median(unlevered)), target_de, tax_rate)


print("Market-data layer ready (Yahoo → SEC → manual override; FRED risk-free rate).")


# %% [markdown]
# ## Cell 6 — WACC engine
#
# $$WACC = \frac{E}{D+E}\,\big(r_f + \beta\,ERP + \alpha\big) \;+\; \frac{D}{D+E}\,r_d(1-t)$$
#
# Three implementation points that come up in interviews:
# 1. **Weights are market value, not book.** Equity is priced by the market every second;
#    only debt uses book value, and only because most corporate debt trades close to par.
# 2. **Cost of debt is the *forward* rate, not the historical coupon.** Interest expense ÷
#    average debt gives the embedded rate on old issuance. The engine floors it at the
#    risk-free rate plus a spread, because no company borrows below Treasuries.
# 3. **The tax shield sits in the cost of debt, not in the cash flows.** UFCF is
#    deliberately unlevered — putting interest in both places double-counts the shield.

# %%
@dataclass
class WACCResult:
    cost_of_equity: float
    cost_of_debt_pretax: float
    cost_of_debt_aftertax: float
    weight_equity: float
    weight_debt: float
    wacc: float
    beta: float
    risk_free_rate: float
    equity_risk_premium: float
    size_premium: float
    tax_rate: float
    market_value_equity: float
    book_value_debt: float
    notes: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Component": [
                "Risk-free rate (10Y UST)", "Equity risk premium", "Levered beta",
                "Size / specific premium", "Cost of equity (CAPM)",
                "Pre-tax cost of debt", "Marginal tax rate", "After-tax cost of debt",
                "Market value of equity", "Book value of debt",
                "Weight of equity", "Weight of debt", "WACC",
            ],
            "Value": [
                fmt_pct(self.risk_free_rate, 2), fmt_pct(self.equity_risk_premium, 2),
                f"{self.beta:.2f}", fmt_pct(self.size_premium, 2),
                fmt_pct(self.cost_of_equity, 2), fmt_pct(self.cost_of_debt_pretax, 2),
                fmt_pct(self.tax_rate, 1), fmt_pct(self.cost_of_debt_aftertax, 2),
                fmt_money(self.market_value_equity), fmt_money(self.book_value_debt),
                fmt_pct(self.weight_equity, 1), fmt_pct(self.weight_debt, 1),
                fmt_pct(self.wacc, 2),
            ],
        })


def compute_wacc(
    drivers: pd.DataFrame,
    market: MarketData,
    tax_rate: Optional[float] = None,
    size_premium: float = 0.0,
    cost_of_debt_override: Optional[float] = None,
    target_debt_weight: Optional[float] = None,
) -> WACCResult:
    """CAPM cost of equity + effective cost of debt, weighted at market value."""
    notes: List[str] = []
    latest = drivers.iloc[-1]

    tax = tax_rate if tax_rate is not None else float(latest["effective_tax_rate"])
    if not np.isfinite(tax):
        tax = 0.21
        notes.append("Effective tax rate unavailable — defaulted to the 21% US federal rate.")

    # ---- Cost of equity (CAPM, plus an optional size/specific premium) --------------
    cost_equity = market.risk_free_rate + market.beta * market.equity_risk_premium + size_premium

    # ---- Cost of debt: interest expense over *average* debt balance ------------------
    debt_now = float(latest["total_debt"]) if np.isfinite(latest["total_debt"]) else 0.0
    debt_prior = float(drivers["total_debt"].iloc[-2]) if len(drivers) > 1 else debt_now
    avg_debt = np.nanmean([debt_now, debt_prior])
    interest = float(latest["interest_expense"]) if np.isfinite(latest["interest_expense"]) else 0.0

    if cost_of_debt_override is not None:
        cost_debt_pretax = cost_of_debt_override
    elif avg_debt > 0 and interest > 0:
        cost_debt_pretax = interest / avg_debt
        floor = market.risk_free_rate + 0.005          # nobody borrows below Treasuries
        ceiling = market.risk_free_rate + 0.10         # >10% spread implies distress
        if not floor <= cost_debt_pretax <= ceiling:
            notes.append(
                f"Implied cost of debt of {cost_debt_pretax:.1%} looks off (embedded legacy "
                f"coupons or capitalised interest) — clamped into [{floor:.1%}, {ceiling:.1%}]."
            )
            cost_debt_pretax = float(np.clip(cost_debt_pretax, floor, ceiling))
    else:
        cost_debt_pretax = market.risk_free_rate + 0.015
        notes.append("No meaningful debt or interest expense — cost of debt set to rf + 150bps.")

    cost_debt_aftertax = cost_debt_pretax * (1 - tax)

    # ---- Market-value weights --------------------------------------------------------
    equity_value = market.market_cap
    if target_debt_weight is not None:
        w_debt = float(np.clip(target_debt_weight, 0.0, 0.9))
        notes.append(f"Using a target capital structure of {w_debt:.0%} debt rather than the current mix.")
    else:
        total_cap = equity_value + debt_now
        w_debt = debt_now / total_cap if total_cap > 0 else 0.0
    w_equity = 1 - w_debt

    wacc = w_equity * cost_equity + w_debt * cost_debt_aftertax

    if wacc < 0.03 or wacc > 0.25:
        notes.append(f"WACC of {wacc:.1%} is far outside the usual 6–12% corridor — check beta and ERP.")

    return WACCResult(
        cost_of_equity=cost_equity,
        cost_of_debt_pretax=cost_debt_pretax,
        cost_of_debt_aftertax=cost_debt_aftertax,
        weight_equity=w_equity,
        weight_debt=w_debt,
        wacc=wacc,
        beta=market.beta,
        risk_free_rate=market.risk_free_rate,
        equity_risk_premium=market.equity_risk_premium,
        size_premium=size_premium,
        tax_rate=tax,
        market_value_equity=equity_value,
        book_value_debt=debt_now,
        notes=notes,
    )


print("WACC engine ready (CAPM cost of equity, effective cost of debt, market-value weights).")


# %% [markdown]
# ## Cell 7 — Assumption set
#
# Everything the forecast needs lives in **one dataclass**. That is the whole design
# principle: if a number is not in here, it cannot be silently hardcoded somewhere in the
# projection. Sliders, sensitivity grids and Monte Carlo all mutate copies of this object.
#
# `seed_from_history()` produces a defensible starting point from the company's own
# medians, then applies two pieces of discipline every credible model applies:
# - **Growth fades** from the recent run-rate toward the terminal rate — no company
#   compounds above GDP forever.
# - **Terminal growth is capped** at long-run nominal GDP (~2.5%). A terminal rate above
#   the discount rate is not aggressive, it is undefined: the geometric series diverges.

# %%
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


print("Assumption layer ready — one dataclass drives the projection, sensitivities and simulation.")

# %% [markdown]
# ## Cell 8 — Projection engine, UFCF and the DCF core
#
# $$UFCF_t = EBIT_t(1-\tau) + D\&A_t - Capex_t - \Delta NWC_t$$
#
# Unlevered, so it belongs to **all** capital providers — which is why it is discounted at
# WACC and produces enterprise value.
#
# **Mid-year convention.** Cash arrives throughout the year, not in a lump on 31 December,
# so cash flows are discounted at $t-0.5$. It lifts the valuation by roughly half a year
# of discounting — about 3–5% at a 9% WACC. Banks use it; academics often don't. The flag
# is exposed so you can show both.
#
# **Terminal value, both ways.** Gordon growth is the theoretically clean version; the exit
# multiple is what the market will actually pay. Computing both and backing out the growth
# rate implied by the multiple is the standard cross-check — if your exit multiple implies
# 6% perpetual growth, the multiple is too high.
#
# $$g_{implied} = \frac{TV \cdot WACC - UFCF_n}{TV + UFCF_n}$$

# %%
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
    nopat = ebit * (1 - a.tax_rate)
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
            "Taxes on EBIT": -ebit * a.tax_rate,
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


print("DCF core ready: projection → discounting → terminal value → EV→equity bridge.")


# %% [markdown]
# ## Cell 9 — Sensitivity and scenario analysis
#
# Three distinct techniques that get conflated constantly. Know the difference cold:
#
# | Technique | What it varies | What it answers |
# |---|---|---|
# | **Sensitivity** | one or two inputs across a grid | "How fragile is the answer?" |
# | **Scenario** | a coherent *set* of inputs together | "What if the bull case is right?" |
# | **Simulation** | all inputs jointly, from distributions | "What is the distribution of outcomes?" |
#
# The classic banker output is the WACC × terminal-growth grid, because those two inputs
# are the least observable and the most powerful. A grid that swings ±40% across a
# plausible range is telling you the terminal value is doing all the work.

# %%
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


print("Sensitivity + scenario engines ready.")


# %% [markdown]
# ## Cell 10 — Monte Carlo simulation (vectorised)
#
# This is the layer that turns a class assignment into a tool. Instead of one point
# estimate, it draws 50,000 joint samples of the key drivers and produces a **distribution
# of intrinsic values**, which supports statements the point estimate cannot:
#
# > "At a 9.2% base WACC the stock screens 18% undervalued, but across 50,000 draws only
# > 63% of outcomes clear the current price, and the 10th percentile implies 24% downside."
#
# **Implementation note — why this runs in under a second.** The naive version loops
# 50,000 times calling `run_dcf`. This version builds the entire simulation as
# `(n_sims × horizon)` NumPy arrays and computes every path at once. That is roughly a
# 500× speed-up, and it is exactly the vectorisation question quant interviews probe.
#
# **Distributions used:**
# - Revenue growth: **normal**, centred on the base case, σ from historical volatility.
# - EBIT margin: **normal**, truncated to sensible bounds — margins mean-revert.
# - WACC: **normal** — it is itself an estimate with standard error, not a known constant.
# - Terminal growth: **triangular** — bounded and asymmetric, which matches how analysts
#   actually think about it (a hard ceiling at GDP, more room on the downside).
#
# Infeasible draws where $g \geq WACC$ are discarded, and the discard rate is reported —
# if it exceeds ~5%, your input ranges are too wide to be credible.

# %%
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
    nopat = ebit * (1 - a.tax_rate)
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

    # Drop pathological tails (negative equity, or values >100× the base) before stats
    sane = np.isfinite(per_share) & (per_share > 0)
    per_share_clean = per_share[sane]

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
        rho, p_value = stats.spearmanr(mc.inputs[col], target)
        rows.append({"Driver": col, "Rank correlation": rho, "p-value": p_value,
                     "Contribution to variance": rho ** 2})
    frame = pd.DataFrame(rows).set_index("Driver")
    frame["Contribution to variance"] /= frame["Contribution to variance"].sum()
    return frame.sort_values("Contribution to variance", ascending=False)


print("Monte Carlo engine ready — vectorised, 50k paths in well under a second.")

# %% [markdown]
# ## Cell 11 — Visualisations
#
# Five charts, each answering a question a client or interviewer will actually ask:
#
# | Chart | Question it answers |
# |---|---|
# | **EV → equity waterfall** | "Where did the value go between enterprise and equity?" |
# | **UFCF bridge by year** | "What is driving cash flow — margin or capital intensity?" |
# | **WACC × g heatmap** | "How fragile is this number?" |
# | **Football field** | "How does the DCF compare to comps and where the stock has traded?" |
# | **Monte Carlo histogram** | "What is the probability this is actually undervalued?" |
#
# All use one palette, direct labelling instead of legends where possible, and no
# chartjunk — the aesthetic bankers and equity research desks expect.

# %%
def _annotate_source(ax, text: str = "Source: SEC XBRL companyfacts, FRED, Yahoo Finance") -> None:
    ax.annotate(text, xy=(0, -0.16), xycoords="axes fraction",
                fontsize=7.5, color=PALETTE["grey"], ha="left")


def plot_ev_to_equity_waterfall(result: DCFResult, shares: float, title_suffix: str = "",
                                ax=None):
    """Waterfall from enterprise value to equity value, with the per-share result."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(11, 5.5))

    items = [(k, v) for k, v in result.bridge.items() if k not in ("Enterprise value", "Equity value")]
    items = [(k, v) for k, v in items if abs(v) > 1e-9]        # hide zero-value rows

    labels = ["Enterprise\nvalue"] + [k.replace("(−) ", "− ").replace("(+) ", "+ ") for k, _ in items] \
             + ["Equity\nvalue"]
    running = result.bridge["Enterprise value"]
    bottoms, heights, colours = [0.0], [running], [PALETTE["navy"]]

    for _, value in items:
        bottoms.append(running if value >= 0 else running + value)
        heights.append(abs(value))
        colours.append(PALETTE["teal"] if value >= 0 else PALETTE["red"])
        running += value

    bottoms.append(0.0)
    heights.append(running)
    colours.append(PALETTE["gold"])

    x = np.arange(len(labels))
    ax.bar(x, heights, bottom=bottoms, color=colours, width=0.62, zorder=3)

    # Dashed connectors carry the running total across each step, which is what makes a
    # waterfall readable rather than just a stack of floating bars.
    running_total = result.bridge["Enterprise value"]
    for i, (_, value) in enumerate(items, start=1):
        ax.plot([i - 1 + 0.31, i + 0.31], [running_total, running_total],
                color=PALETTE["grey"], linewidth=0.9, linestyle="--", zorder=2)
        running_total += value

    # Label every bar with its own signed contribution, not the stacked height
    signed_values = [result.bridge["Enterprise value"]] + [v for _, v in items] + [result.bridge["Equity value"]]
    span = max(b + h for b, h in zip(bottoms, heights))
    for xi, (b, h, value) in enumerate(zip(bottoms, heights, signed_values)):
        ax.text(xi, b + h + span * 0.02, fmt_money(value), ha="center", va="bottom",
                fontsize=8.5, color=PALETTE["ink"], fontweight="medium")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=18 if len(labels) > 5 else 0,
                       ha="right" if len(labels) > 5 else "center")
    ax.yaxis.set_major_formatter(MONEY_AXIS)
    ax.set_ylabel("Value")
    ax.set_title(f"Enterprise value → equity value bridge{title_suffix}")
    ax.margins(y=0.16)
    ax.spines[["top", "right"]].set_visible(False)

    ax.annotate(
        f"Equity value per share: {fmt_money(result.value_per_share, 2)}\n"
        f"on {shares/1e6:,.0f}mm diluted shares",
        xy=(0.985, 0.94), xycoords="axes fraction", ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=PALETTE["mist"], edgecolor="none"),
    )
    if created:
        _annotate_source(ax)
        plt.tight_layout()
    return ax


def plot_ufcf_bridge(result: DCFResult, ax=None):
    """Per-year build from NOPAT to unlevered FCF — shows what consumes the cash."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(11, 5.5))

    proj = result.projection
    years = proj.columns.tolist()
    nopat = proj.loc["NOPAT"].to_numpy(float)
    da = proj.loc["(+) D&A"].to_numpy(float)
    capex = proj.loc["(−) Capex"].to_numpy(float)
    dnwc = proj.loc["(−) ΔNWC"].to_numpy(float)
    ufcf = proj.loc["Unlevered FCF"].to_numpy(float)

    x = np.arange(len(years))
    width = 0.2
    ax.bar(x - 1.5 * width, nopat, width, label="NOPAT", color=PALETTE["navy"], zorder=3)
    ax.bar(x - 0.5 * width, da, width, label="(+) D&A", color=PALETTE["sky"], zorder=3)
    ax.bar(x + 0.5 * width, capex, width, label="(−) Capex", color=PALETTE["red"], zorder=3)
    ax.bar(x + 1.5 * width, dnwc, width, label="(−) ΔNWC", color=PALETTE["gold"], zorder=3)
    ax.plot(x, ufcf, color=PALETTE["ink"], marker="o", markersize=6, linewidth=2,
            label="Unlevered FCF", zorder=4)

    for xi, val in zip(x, ufcf):
        ax.annotate(fmt_money(val), (xi, val), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, fontweight="bold", color=PALETTE["ink"])

    ax.axhline(0, color=PALETTE["ink"], linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.yaxis.set_major_formatter(MONEY_AXIS)
    ax.set_title("Unlevered free cash flow build by forecast year")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.spines[["top", "right"]].set_visible(False)
    if created:
        plt.tight_layout()
    return ax


def plot_sensitivity_heatmap(grid: pd.DataFrame, current_price: Optional[float] = None,
                             title: str = "Value per share: WACC × terminal growth", ax=None):
    """Two-way sensitivity heatmap, coloured by upside/downside against the market price."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9.5, 6))

    values = grid.to_numpy(dtype=float)
    if current_price:
        colour_basis = values / current_price - 1
        vlim = np.nanmax(np.abs(colour_basis)) or 0.01
        mesh = ax.imshow(colour_basis, cmap="RdYlGn", vmin=-vlim, vmax=vlim, aspect="auto")
        cbar_label = "Upside / (downside) vs current price"
    else:
        mesh = ax.imshow(values, cmap="Blues", aspect="auto")
        cbar_label = "Value per share"

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("Terminal growth rate")
    ax.set_ylabel("WACC")
    ax.set_title(title)
    ax.grid(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/m", ha="center", va="center", fontsize=8, color=PALETTE["grey"])
                continue
            label = f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5,
                    color=PALETTE["ink"], fontweight="medium")

    cbar = plt.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    if current_price:
        cbar.formatter = FuncFormatter(lambda v, _: f"{v:+.0%}")
        cbar.update_ticks()
    if created:
        plt.tight_layout()
    return ax


def plot_football_field(
    ranges: Dict[str, Tuple[float, float]],
    current_price: Optional[float] = None,
    base_case: Optional[float] = None,
    ax=None,
):
    """Football field: every valuation methodology as a horizontal range on one axis."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(10.5, 0.75 * len(ranges) + 2.4))

    labels = list(ranges.keys())
    y = np.arange(len(labels))[::-1]
    colours = [PALETTE["navy"], PALETTE["blue"], PALETTE["teal"], PALETTE["sky"],
               PALETTE["gold"], PALETTE["grey"]]

    for idx, (label, (low, high)) in enumerate(ranges.items()):
        if not (np.isfinite(low) and np.isfinite(high)):
            continue
        low, high = min(low, high), max(low, high)
        ax.barh(y[idx], high - low, left=low, height=0.52,
                color=colours[idx % len(colours)], alpha=0.88, zorder=3)
        ax.text(low, y[idx], f"  ${low:,.0f}  ", va="center", ha="right",
                fontsize=8.5, color=PALETTE["ink"])
        ax.text(high, y[idx], f"  ${high:,.0f}", va="center", ha="left",
                fontsize=8.5, color=PALETTE["ink"])

    if current_price:
        ax.axvline(current_price, color=PALETTE["red"], linestyle="--", linewidth=1.8, zorder=5)
        ax.annotate(f"Current price ${current_price:,.2f}",
                    xy=(current_price, len(labels) - 0.35), color=PALETTE["red"],
                    fontsize=9, fontweight="bold", ha="center", va="bottom")
    if base_case:
        ax.axvline(base_case, color=PALETTE["ink"], linestyle=":", linewidth=1.5, zorder=5)
        ax.annotate(f"DCF base ${base_case:,.2f}", xy=(base_case, -0.62),
                    color=PALETTE["ink"], fontsize=8.5, ha="center", annotation_clip=False)
    ax.set_ylim(-0.9, len(labels) - 0.25)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Implied value per share")
    ax.set_title("Football field — valuation range by methodology")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    if created:
        plt.tight_layout()
    return ax


def plot_monte_carlo(mc: MonteCarloResult, current_price: Optional[float] = None,
                     base_case: Optional[float] = None, ax=None):
    """Simulated value distribution with the market price and key percentiles marked."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(11, 5.5))

    values = mc.values_per_share
    # Trim the extreme tail so one 500× outlier does not flatten the whole histogram
    upper = np.percentile(values, 99.5)
    trimmed = values[values <= upper]

    ax.hist(trimmed, bins=90, color=PALETTE["sky"], edgecolor="white", linewidth=0.4, zorder=3)

    kde = stats.gaussian_kde(trimmed)
    grid = np.linspace(trimmed.min(), trimmed.max(), 400)
    scale = len(trimmed) * (trimmed.max() - trimmed.min()) / 90
    ax.plot(grid, kde(grid) * scale, color=PALETTE["navy"], linewidth=1.8, zorder=4)

    p10, p50, p90 = (mc.percentiles["10"], mc.percentiles["50"], mc.percentiles["90"])
    for value, label, colour, style in [
        (p10, f"P10 ${p10:,.0f}", PALETTE["grey"], ":"),
        (p50, f"Median ${p50:,.0f}", PALETTE["ink"], "-"),
        (p90, f"P90 ${p90:,.0f}", PALETTE["grey"], ":"),
    ]:
        ax.axvline(value, color=colour, linestyle=style, linewidth=1.4, zorder=5)
        ax.annotate(label, xy=(value, ax.get_ylim()[1] * 0.97), rotation=90,
                    fontsize=8, color=colour, ha="right", va="top")

    if current_price:
        ax.axvline(current_price, color=PALETTE["red"], linewidth=2.2, zorder=6)
        prob = mc.prob_above_price
        ax.annotate(
            f"Current price ${current_price:,.2f}\n"
            f"P(intrinsic > price) = {prob:.0%}",
            xy=(current_price, ax.get_ylim()[1] * 0.72), xytext=(14, 0),
            textcoords="offset points", fontsize=9, color=PALETTE["red"], fontweight="bold",
        )
    if base_case:
        ax.axvline(base_case, color=PALETTE["gold"], linewidth=1.8, linestyle="--", zorder=6)

    ax.set_xlabel("Intrinsic value per share")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Monte Carlo distribution of intrinsic value ({mc.n_valid:,} paths)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    if created:
        plt.tight_layout()
    return ax


def plot_historical_drivers(drivers: pd.DataFrame, name: str = "", ax=None):
    """Revenue bars with margin and capex-intensity lines — the forecast's anchor."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(11, 5.2))

    years = drivers["fiscal_year"].astype(int).astype(str)
    ax.bar(years, drivers["revenue"], color=PALETTE["mist"], zorder=2, label="Revenue (LHS)")
    ax.yaxis.set_major_formatter(MONEY_AXIS)
    ax.set_ylabel("Revenue")
    ax.set_title(f"Historical operating drivers{' — ' + name if name else ''}")

    twin = ax.twinx()
    twin.plot(years, drivers["ebit_margin"], color=PALETTE["navy"], marker="o",
              linewidth=2, label="EBIT margin")
    twin.plot(years, drivers["capex_pct_revenue"], color=PALETTE["red"], marker="s",
              linewidth=1.7, linestyle="--", label="Capex % revenue")
    twin.plot(years, drivers["da_pct_revenue"], color=PALETTE["teal"], marker="^",
              linewidth=1.5, linestyle=":", label="D&A % revenue")
    twin.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    twin.set_ylabel("% of revenue")
    twin.grid(False)

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    ax.spines[["top"]].set_visible(False)
    if created:
        plt.tight_layout()
    return ax


def plot_driver_tornado(attribution: pd.DataFrame, ax=None):
    """Tornado of which assumption explains the most variance in the simulated value."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(9, 3.6))

    data = attribution.sort_values("Contribution to variance")
    colours = [PALETTE["teal"] if r > 0 else PALETTE["red"] for r in data["Rank correlation"]]
    ax.barh(data.index, data["Contribution to variance"], color=colours, height=0.6, zorder=3)
    for y, (share, rho) in enumerate(zip(data["Contribution to variance"], data["Rank correlation"])):
        ax.text(share + 0.01, y, f"{share:.0%}  (ρ={rho:+.2f})", va="center", fontsize=8.5)

    ax.set_xlabel("Share of explained variance in value per share")
    ax.set_title("What actually drives the answer")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, max(data["Contribution to variance"]) * 1.35)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    if created:
        plt.tight_layout()
    return ax


print("Chart library ready: waterfall, UFCF bridge, heatmap, football field, Monte Carlo, tornado.")


# %% [markdown]
# ## Cell 12 — Diagnostics and quality metrics
#
# A valuation model has no R². Its quality is measured by **data integrity** and
# **internal consistency**, so the engine scores itself on both and flags anything a
# reviewer would challenge:
#
# - **Tag coverage** — share of required line items sourced from reported tags rather than
#   derived or missing. Below ~80% and the model is guessing.
# - **Terminal value share of EV** — above 85% and you are not valuing a business, you are
#   valuing an assumption. 60–80% is normal.
# - **Implied exit multiple vs entry multiple** — if Gordon growth implies an exit at 25×
#   EBITDA when the company trades at 12×, the terminal growth rate is too high.
# - **Reinvestment consistency** — sustainable growth ≈ ROIC × reinvestment rate. If your
#   forecast grows 12% on a 5% reinvestment rate, the growth is unfunded.
# - **Historical back-test** — rebuild UFCF from reported figures and compare against
#   reported CFO less capex. Large gaps mean the tag mapping is wrong.

# %%
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


print("Diagnostics ready — the model grades its own inputs before you present it.")


# %% [markdown]
# ## Cell 13 — Orchestrator: one call, end to end
#
# Everything above is composable on its own. This wraps it into a single entry point so a
# full valuation is one line, and returns an object holding every intermediate artefact
# for inspection.
#
# ```python
# val = run_valuation("MSFT")                      # base case, auto-seeded from filings
# val = run_valuation("MSFT", assumptions=my_case) # your own assumptions
# ```

# %%
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

print("Orchestrator ready. Run:  val = run_valuation('MSFT')  then  val.dashboard()")


# %% [markdown]
# ## Cell 14 — Interactive assumption panel (ipywidgets)
#
# Sliders turn the model from a script into something someone else can use. In Colab you
# must enable the custom widget manager once per session — the cell does that
# automatically. If widgets fail to render (they occasionally do in Colab), the fallback
# `revalue()` function does the same job from a plain function call.

# %%
def build_assumption_panel(val: Valuation):
    """Live sliders over the key assumptions, re-running the DCF on every change."""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        try:                                            # Colab-specific widget enablement
            from google.colab import output as colab_output
            colab_output.enable_custom_widget_manager()
        except ImportError:
            pass
    except ImportError:
        print("ipywidgets unavailable — use revalue(val, ...) instead.")
        return None

    a, w0 = val.assumptions, val.wacc_result.wacc

    style = {"description_width": "170px"}
    layout = widgets.Layout(width="460px")
    sliders = {
        "revenue_growth_y1": widgets.FloatSlider(
            value=a.revenue_growth_y1, min=-0.15, max=0.45, step=0.005,
            description="Year-1 revenue growth", readout_format=".1%", style=style, layout=layout),
        "ebit_margin_target": widgets.FloatSlider(
            value=a.ebit_margin_target or a.ebit_margin_start, min=-0.10, max=0.70, step=0.005,
            description="Terminal EBIT margin", readout_format=".1%", style=style, layout=layout),
        "capex_pct_revenue": widgets.FloatSlider(
            value=a.capex_pct_revenue, min=0.0, max=0.35, step=0.0025,
            description="Capex % of revenue", readout_format=".2%", style=style, layout=layout),
        "nwc_pct_revenue": widgets.FloatSlider(
            value=a.nwc_pct_revenue, min=-0.25, max=0.40, step=0.005,
            description="NWC % of revenue", readout_format=".1%", style=style, layout=layout),
        "tax_rate": widgets.FloatSlider(
            value=a.tax_rate, min=0.0, max=0.40, step=0.005,
            description="Tax rate", readout_format=".1%", style=style, layout=layout),
        "terminal_growth": widgets.FloatSlider(
            value=a.terminal_growth, min=0.0, max=0.045, step=0.0025,
            description="Terminal growth", readout_format=".2%", style=style, layout=layout),
    }
    wacc_slider = widgets.FloatSlider(
        value=w0, min=0.04, max=0.20, step=0.0025, description="WACC",
        readout_format=".2%", style=style, layout=layout)
    method = widgets.ToggleButtons(
        options=[("Gordon growth", "gordon"), ("Exit multiple", "exit_multiple")],
        value="gordon", description="Terminal value", style=style)
    midyear = widgets.Checkbox(value=a.mid_year_convention, description="Mid-year convention")
    out = widgets.Output()

    def refresh(_=None):
        with out:
            clear_output(wait=True)
            case = a.copy_with(tv_method=method.value, mid_year_convention=midyear.value,
                               **{k: s.value for k, s in sliders.items()})
            try:
                res = run_dcf(val.drivers, case, wacc_slider.value,
                              val.market.shares_diluted, val.market.price)
            except ValueError as exc:
                print(f"⚠️  {exc}")
                return
            upside = res.upside_vs_price
            verdict = "UNDERVALUED" if upside and upside > 0.15 else \
                      "OVERVALUED" if upside and upside < -0.15 else "FAIRLY VALUED"
            print(f"  Intrinsic value   {fmt_money(res.value_per_share, 2)} / share")
            print(f"  Market price      {fmt_money(val.market.price, 2)}")
            print(f"  Upside            {upside:+.1%}   →   {verdict}")
            print(f"  Enterprise value  {fmt_money(res.enterprise_value)}"
                  f"   │  TV = {fmt_pct(res.tv_share_of_ev, 0)} of EV")
            fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
            plot_ufcf_bridge(res, ax=axes[0])
            plot_ev_to_equity_waterfall(res, val.market.shares_diluted, ax=axes[1])
            plt.tight_layout()
            plt.show()

    for widget in list(sliders.values()) + [wacc_slider, method, midyear]:
        widget.observe(refresh, names="value")

    panel = widgets.VBox([
        widgets.HTML(f"<h3 style='margin-bottom:4px'>{val.meta['name']} "
                     f"({val.meta['ticker']}) — assumption panel</h3>"),
        widgets.HBox([widgets.VBox(list(sliders.values())[:3]),
                      widgets.VBox(list(sliders.values())[3:] + [wacc_slider])]),
        widgets.HBox([method, midyear]),
        out,
    ])
    display(panel)
    refresh()
    return panel


def revalue(val: Valuation, **overrides) -> DCFResult:
    """Non-widget fallback: revalue(val, terminal_growth=0.03, capex_pct_revenue=0.06)."""
    wacc = overrides.pop("wacc", val.wacc_result.wacc)
    case = val.assumptions.copy_with(**overrides)
    res = run_dcf(val.drivers, case, wacc, val.market.shares_diluted, val.market.price)
    print(res.headline())
    if res.upside_vs_price is not None:
        print(f"Upside vs {fmt_money(val.market.price, 2)}: {res.upside_vs_price:+.1%}")
    return res


print("Interactive panel ready:  build_assumption_panel(val)   or   revalue(val, terminal_growth=0.03)")


# %% [markdown]
# ## Cell 15 — Export deliverables
#
# Writes an Excel workbook with every tab a reviewer expects, plus the dashboard as a PNG.
# In Colab the files land in `/content/` — open the file browser on the left, or use
# `files.download()` to pull them locally.

# %%
def export_valuation(val: Valuation, output_dir: str = None, save_charts: bool = True) -> Dict[str, str]:
    """Write the full model to Excel and save the charts. Returns the paths written."""
    output_dir = Path(output_dir or ("/content/output" if Path("/content").exists() else "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.today().strftime("%Y%m%d")
    stem = f"{val.meta['ticker']}_DCF_{stamp}"
    written: Dict[str, str] = {}

    xlsx_path = output_dir / f"{stem}.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary = pd.DataFrame({
                "Metric": [
                    "Company", "Ticker", "CIK", "Valuation date", "Current price",
                    "Market cap", "WACC", "Cost of equity", "After-tax cost of debt",
                    "Terminal growth", "Exit EV/EBITDA", "Enterprise value (Gordon)",
                    "Equity value (Gordon)", "Value per share (Gordon)",
                    "Value per share (exit multiple)", "Upside vs market",
                    "TV share of EV", "Monte Carlo P10", "Monte Carlo median",
                    "Monte Carlo P90", "P(intrinsic > price)",
                ],
                "Value": [
                    val.meta["name"], val.meta["ticker"], val.meta["cik"],
                    datetime.today().strftime("%Y-%m-%d"), val.market.price,
                    val.market.market_cap, val.wacc_result.wacc,
                    val.wacc_result.cost_of_equity, val.wacc_result.cost_of_debt_aftertax,
                    val.assumptions.terminal_growth, val.assumptions.exit_ev_ebitda,
                    val.gordon.enterprise_value, val.gordon.equity_value,
                    val.gordon.value_per_share, val.exit_multiple.value_per_share,
                    val.gordon.upside_vs_price, val.gordon.tv_share_of_ev,
                    val.monte_carlo.percentiles["10"] if val.monte_carlo else None,
                    val.monte_carlo.percentiles["50"] if val.monte_carlo else None,
                    val.monte_carlo.percentiles["90"] if val.monte_carlo else None,
                    val.monte_carlo.prob_above_price if val.monte_carlo else None,
                ],
            })
            summary.to_excel(writer, sheet_name="Summary", index=False)
            val.drivers[DRIVER_VIEW].to_excel(writer, sheet_name="Historical drivers")
            val.gordon.projection.to_excel(writer, sheet_name="Projection (Gordon)")
            val.exit_multiple.projection.to_excel(writer, sheet_name="Projection (Exit)")
            val.wacc_result.to_frame().to_excel(writer, sheet_name="WACC", index=False)
            pd.DataFrame(val.gordon.bridge, index=["Value"]).T.to_excel(writer, sheet_name="EV-Equity bridge")
            val.sensitivity.to_excel(writer, sheet_name="Sensitivity")
            val.scenarios.to_excel(writer, sheet_name="Scenarios")
            val.diagnostics.to_excel(writer, sheet_name="Diagnostics", index=False)
            val.report.to_frame().to_excel(writer, sheet_name="XBRL tag audit", index=False)
            pd.DataFrame([asdict(val.assumptions)]).T.rename(columns={0: "Value"}).to_excel(
                writer, sheet_name="Assumptions")
            if val.monte_carlo:
                val.monte_carlo.summary_frame().to_excel(writer, sheet_name="Monte Carlo")
                val.monte_carlo.inputs.sample(min(5000, val.monte_carlo.n_valid), random_state=1)\
                    .to_excel(writer, sheet_name="MC draws (sample)", index=False)
        written["excel"] = str(xlsx_path)
        log.info("Wrote %s", xlsx_path)
    except Exception as exc:
        log.error("Excel export failed: %s", exc)

    if save_charts:
        try:
            fig = val.dashboard()
            png_path = output_dir / f"{stem}_dashboard.png"
            fig.savefig(png_path, dpi=200)
            plt.close(fig)
            written["dashboard"] = str(png_path)
            log.info("Wrote %s", png_path)
        except Exception as exc:
            log.error("Chart export failed: %s", exc)

    print("\nExported:")
    for kind, path in written.items():
        print(f"  {kind:10s} → {path}")
    return written


# In Colab, download the workbook with:
#   from google.colab import files; files.download(paths["excel"])

print("Export layer ready:  paths = export_valuation(val)")
