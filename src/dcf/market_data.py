"""Market inputs: price, shares, beta, risk-free rate and equity risk premium.

Every source has a fallback, so a throttled API degrades to a manual override rather than
killing the run.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CFG, log


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
