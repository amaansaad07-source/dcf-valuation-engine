"""Weighted average cost of capital.

CAPM cost of equity, effective after-tax cost of debt, weighted at market value.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import fmt_money, fmt_pct
from .market_data import MarketData


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
