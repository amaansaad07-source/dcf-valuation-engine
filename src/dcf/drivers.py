"""Historical driver table: raw tags in, DCF inputs out.

Reconstructs revenue, EBIT, D&A, capex and change in net working capital, then derives the
margins and intensity ratios that anchor the forecast.
"""


import numpy as np
import pandas as pd

from .config import CFG
from .xbrl_tags import ExtractionReport


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
