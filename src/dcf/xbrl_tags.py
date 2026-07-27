"""XBRL tag dictionary and fact extraction.

There is no single "revenue" tag. This module holds an ordered fallback chain per line
item, splits duration facts from instant facts, and anchors every tag to the fiscal
year-end dates taken from revenue so 52/53-week calendars align correctly.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CFG, Config, log
from .sec_client import SEC, SECError

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
