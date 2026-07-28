"""Chart library: waterfall, UFCF bridge, heatmap, football field, histogram, tornado.

One palette, direct labelling, banker money formatting, no chartjunk.
"""

from typing import Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats

from .config import MONEY_AXIS, PALETTE, fmt_money
from .dcf import DCFResult
from .monte_carlo import MonteCarloResult


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
        finite = colour_basis[np.isfinite(colour_basis)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (-0.01, 0.01)
        # Anchor the diverging map at zero upside but let each side stretch to the data.
        # The old symmetric ±max scaling saturated the whole grid deep red whenever every
        # cell sat below the market price, hiding all the structure within the grid.
        if lo < 0 < hi:
            norm = mpl.colors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
            mesh = ax.imshow(colour_basis, cmap="RdYlGn", norm=norm, aspect="auto")
        else:
            # Every cell on one side of the price: use the data range so gradation survives
            pad = (hi - lo) * 0.05 or abs(hi) * 0.05 or 0.01
            mesh = ax.imshow(colour_basis, cmap="RdYlGn", vmin=lo - pad, vmax=hi + pad,
                             aspect="auto")
        cbar_label = "Upside / (downside) vs current price"
    else:
        mesh = ax.imshow(values, cmap="Blues", aspect="auto")
        cbar_label = "Value per share"

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, fontsize=10.5)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=10.5)
    ax.set_xlabel("Terminal growth rate", fontsize=11)
    ax.set_ylabel("WACC", fontsize=11)
    ax.set_title(title, pad=12)
    ax.grid(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/m", ha="center", va="center", fontsize=9, color=PALETTE["grey"])
                continue
            label = f"${v:,.0f}" if abs(v) >= 100 else f"${v:,.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=10,
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
    """Football field: every valuation methodology as a horizontal range on one axis.

    Label placement is computed from the data extent rather than hardcoded, so range
    labels sit clear of their bars and the current-price callout stays clear of the title
    regardless of how wide or narrow the ranges turn out to be.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(11, 0.8 * len(ranges) + 2.6))

    valid = {k: (min(v), max(v)) for k, v in ranges.items()
             if np.isfinite(v[0]) and np.isfinite(v[1])}
    if not valid:
        ax.text(0.5, 0.5, "No valid ranges to plot", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["grey"])
        return ax

    labels = list(valid.keys())
    y = np.arange(len(labels))[::-1]
    colours = [PALETTE["navy"], PALETTE["blue"], PALETTE["teal"], PALETTE["sky"],
               PALETTE["gold"], PALETTE["grey"]]

    # Work out the full horizontal extent first, including the price markers, then reserve
    # margin on each side for the end labels. Without this the labels collide with the bars.
    points = [v for pair in valid.values() for v in pair]
    if current_price:
        points.append(current_price)
    if base_case:
        points.append(base_case)
    lo_x, hi_x = min(points), max(points)
    span = (hi_x - lo_x) or max(abs(hi_x), 1.0)
    pad = span * 0.16
    ax.set_xlim(lo_x - pad, hi_x + pad)
    gap = span * 0.015                                  # breathing room around each label

    for idx, (label, (low, high)) in enumerate(valid.items()):
        ax.barh(y[idx], high - low, left=low, height=0.5,
                color=colours[idx % len(colours)], alpha=0.9, zorder=3)
        ax.text(low - gap, y[idx], f"${low:,.0f}", va="center", ha="right",
                fontsize=8.5, color=PALETTE["ink"])
        ax.text(high + gap, y[idx], f"${high:,.0f}", va="center", ha="left",
                fontsize=8.5, color=PALETTE["ink"])

    # Price markers get their own row of headroom above the bars so the callout text can
    # never reach the title.
    top = len(labels) - 0.45
    ax.set_ylim(-0.85, top + 0.75)

    if current_price:
        ax.axvline(current_price, color=PALETTE["red"], linestyle="--", linewidth=1.8, zorder=5)
        ax.annotate(f"Current price  ${current_price:,.2f}",
                    xy=(current_price, top + 0.30), color=PALETTE["red"], fontsize=9,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["paper"],
                              edgecolor=PALETTE["red"], linewidth=0.8))
    if base_case:
        ax.axvline(base_case, color=PALETTE["ink"], linestyle=":", linewidth=1.5, zorder=5)
        ax.annotate(f"DCF base  ${base_case:,.2f}", xy=(base_case, -0.62),
                    color=PALETTE["ink"], fontsize=8.5, ha="center", va="center",
                    annotation_clip=False)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Implied value per share")
    ax.set_title("Football field — valuation range by methodology", pad=14)
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
    # Trim both extreme tails so a handful of outlier paths cannot flatten the histogram
    lo_t, hi_t = np.percentile(values, [0.5, 99.5])
    trimmed = values[(values >= lo_t) & (values <= hi_t)]
    if trimmed.size < 10:
        trimmed = values

    ax.hist(trimmed, bins=90, color=PALETTE["sky"], edgecolor="white", linewidth=0.4, zorder=3)

    # KDE overlay only when it is well defined; skip silently for degenerate samples
    if trimmed.size >= 30 and float(np.std(trimmed)) > 1e-9:
        kde = stats.gaussian_kde(trimmed)
        grid = np.linspace(trimmed.min(), trimmed.max(), 400)
        scale = len(trimmed) * (trimmed.max() - trimmed.min()) / 90
        ax.plot(grid, kde(grid) * scale, color=PALETTE["navy"], linewidth=1.8, zorder=4)

    if trimmed.min() < 0:
        ax.axvline(0, color=PALETTE["ink"], linewidth=1.0, alpha=0.6, zorder=4)
    if getattr(mc, "share_negative", 0.0) > 0.01:
        ax.annotate(f"{mc.share_negative:.0%} of paths end in negative equity value",
                    xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8.5,
                    color=PALETTE["red"], ha="left", va="top")

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
