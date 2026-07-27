"""Streamlit front end for the DCF valuation engine.

    streamlit run app.py

Type a ticker, get a full valuation. Expensive work (the 5-40MB SEC payload) is cached, so
dragging an assumption slider re-runs only the valuation maths — which takes milliseconds.
"""
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dcf import (  # noqa: E402
    CFG,
    build_driver_table,
    build_sensitivity_ranges,
    compute_wacc,
    display_drivers,
    driver_attribution,
    fetch_financial_history,
    fetch_market_data,
    fmt_money,
    monte_carlo_dcf,
    plot_driver_tornado,
    plot_ev_to_equity_waterfall,
    plot_football_field,
    plot_historical_drivers,
    plot_monte_carlo,
    plot_sensitivity_heatmap,
    plot_ufcf_bridge,
    run_dcf,
    run_scenarios,
    seed_assumptions_from_history,
    sensitivity_grid,
    valuation_diagnostics,
)
from dcf.sec_client import SEC  # noqa: E402

st.set_page_config(page_title="DCF Valuation Engine", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")


# ══════════════════════════════════════════════════════════════════════════════════════
# CACHED DATA LAYER
# The SEC payload is the only slow step. Cache it hard, then let the valuation maths run
# live on every slider change — that is what makes the sliders feel instant.
# ══════════════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_filings(ticker: str, user_agent: str, years: int):
    """Pull and normalise the filing history. Cached for an hour per ticker."""
    CFG.user_agent = user_agent
    SEC.session.headers["User-Agent"] = user_agent      # the live session needs it too
    history, report, meta = fetch_financial_history(ticker, years)
    drivers = build_driver_table(history, report)
    meta["fetched_at"] = datetime.now()
    return drivers, report, meta


@st.cache_data(ttl=900, show_spinner=False)
def load_market(ticker: str, sec_shares, price_o, beta_o, shares_o, erp):
    """Price, beta, shares and the risk-free rate. Cached for 15 minutes."""
    data = fetch_market_data(ticker, sec_shares=sec_shares, price_override=price_o,
                             beta_override=beta_o, shares_override=shares_o, erp=erp)
    return data, datetime.now()


def pct_slider(label, lo, hi, value, step, fmt="%.2f%%", help=None):
    """A percentage slider that displays honestly.

    Streamlit's ``format`` is printf-style and is applied to the *raw* stored value, so a
    slider holding 0.0882 with format "%.2f%%" renders as "0.09%" — a factor of 100 out.
    (Python's f-string ``:.2%`` scales by 100; printf's ``%%`` is only a literal sign.)
    So the widget works in percentage points and we divide on the way out — the label and
    the maths then agree.
    """
    raw = st.sidebar.slider(label, float(lo), float(hi), float(value), float(step),
                            format=fmt, help=help)
    return raw / 100.0


def show(ax):
    """Render a chart function's axis into Streamlit and free the figure."""
    st.pyplot(ax.figure, width="stretch")
    plt.close(ax.figure)


# ══════════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — inputs
# ══════════════════════════════════════════════════════════════════════════════════════
st.sidebar.title("DCF Valuation Engine")
st.sidebar.caption("Reported financials straight from SEC XBRL filings.")

def _default_user_agent() -> str:
    """Read from Streamlit secrets if deployed, otherwise leave blank for the user."""
    try:
        return st.secrets.get("SEC_USER_AGENT", "")
    except Exception:                                   # no secrets.toml configured
        return ""


user_agent = st.sidebar.text_input(
    "SEC User-Agent (required)",
    value=_default_user_agent(),
    placeholder="Your Name your@email.com",
    help="The SEC returns 403 without a real name and email. This is their only API key.",
)

ticker = st.sidebar.text_input("Ticker", value="MSFT", max_chars=8).strip().upper()
years_history = st.sidebar.slider("Years of history", 5, 12, 10)

with st.sidebar.expander("Market data overrides", expanded=False):
    st.caption("Yahoo occasionally throttles cloud IPs. Fill these in if the run fails.")
    price_o = st.number_input("Price", value=0.0, step=1.0, format="%.2f") or None
    shares_o = st.number_input("Diluted shares (mm)", value=0.0, step=100.0) or None
    beta_o = st.number_input("Levered beta", value=0.0, step=0.05, format="%.2f") or None
    shares_o = shares_o * 1e6 if shares_o else None

erp = pct_slider("Equity risk premium", 3.00, 8.00, CFG.erp_default * 100, 0.25,
                 help="Damodaran's implied US ERP is a good anchor — usually 4.0-5.5%.")

run = st.sidebar.button("Run valuation", type="primary", width="stretch")

if run:
    st.session_state["ran"] = True

if not user_agent or "@" not in user_agent:
    st.info(
        "**Set your SEC User-Agent in the sidebar to begin.**\n\n"
        "The SEC requires a descriptive contact string on every request — "
        "`Your Name your@email.com`. Requests without one are rejected with a 403. "
        "It is not a registration or an API key; nothing is sent anywhere but the SEC."
    )
    st.stop()

if not st.session_state.get("ran"):
    st.title("DCF Valuation Engine")
    st.markdown(
        "Enter a ticker in the sidebar and press **Run valuation**.\n\n"
        "The engine pulls a decade of reported financials from the SEC XBRL API, "
        "reconstructs the unlevered free cash flow drivers, computes WACC from CAPM, "
        "discounts to enterprise value, bridges to equity value per share, and stress-tests "
        "the result with a sensitivity grid and a 50,000-path Monte Carlo simulation."
    )
    st.caption(
        "Works on US operating companies. Skip banks and insurers — capex and working "
        "capital do not mean the same thing for financials, so an unlevered DCF is the "
        "wrong tool there. Also skip foreign issuers filing 20-F under IFRS."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════════════
try:
    with st.spinner(f"Pulling {ticker} filings from the SEC…"):
        drivers, report, meta = load_filings(ticker, user_agent, years_history)
except Exception as exc:                                    # noqa: BLE001
    msg = str(exc)
    st.error(f"**Could not load {ticker}.**\n\n{msg}")
    if "403" in msg:
        st.info("Fix the User-Agent in the sidebar — it needs a real name and email.")
    elif "registrant map" in msg:
        st.info("Not a US SEC filer. Try a US-listed operating company.")
    elif "revenue tag" in msg:
        st.info("Usually a bank, insurer or IFRS filer. Those need a bespoke tag chain.")
    st.stop()

sec_shares = float(drivers["diluted_shares"].iloc[-1]) if pd.notna(
    drivers["diluted_shares"].iloc[-1]) else None

try:
    market, price_fetched_at = load_market(ticker, sec_shares, price_o, beta_o, shares_o, erp)
except Exception as exc:                                    # noqa: BLE001
    st.error(f"**Market data unavailable.** {exc}")
    st.info("Open *Market data overrides* in the sidebar and enter price and share count.")
    st.stop()

wacc_result = compute_wacc(drivers, market)
base = seed_assumptions_from_history(drivers, horizon=5)


# ══════════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — assumptions (rendered only once the company has loaded, so the defaults are
# seeded from its own operating history rather than arbitrary constants)
# ══════════════════════════════════════════════════════════════════════════════════════
st.sidebar.divider()
st.sidebar.subheader("Assumptions")
st.sidebar.caption("Seeded from this company's 5-year medians.")

horizon = st.sidebar.slider("Forecast horizon (years)", 3, 10, 5)
g1 = pct_slider("Year-1 revenue growth", -15.0, 45.0,
                base.revenue_growth_y1 * 100, 0.5, "%.1f%%")
margin = pct_slider("Terminal EBIT margin", -10.0, 70.0,
                    base.ebit_margin_target * 100, 0.5, "%.1f%%")
capex_pct = pct_slider("Capex % of revenue", 0.0, 35.0,
                       base.capex_pct_revenue * 100, 0.25)
nwc_pct = pct_slider("NWC % of revenue", -25.0, 40.0,
                     base.nwc_pct_revenue * 100, 0.5, "%.1f%%")
da_pct = pct_slider("D&A % of revenue", 0.0, 30.0,
                    base.da_pct_revenue * 100, 0.25)
tax = pct_slider("Tax rate", 0.0, 40.0, base.tax_rate * 100, 0.5, "%.1f%%")
term_g = pct_slider("Terminal growth", 0.0, 4.50, 2.50, 0.25,
                    help="Capped near long-run nominal GDP. Must stay below WACC.")
wacc = pct_slider("WACC", 4.00, 20.00, wacc_result.wacc * 100, 0.25,
                  help="Overrides the computed WACC shown in the WACC tab.")
mid_year = st.sidebar.checkbox("Mid-year convention", value=True,
                               help="Discount at t−0.5. Worth roughly +4% at a 9% WACC.")
n_sims = st.sidebar.select_slider("Monte Carlo paths", [5_000, 10_000, 25_000, 50_000, 100_000],
                                  value=25_000)

assumptions = base.copy_with(
    horizon=horizon, revenue_growth_y1=g1, ebit_margin_target=margin,
    capex_pct_revenue=capex_pct, nwc_pct_revenue=nwc_pct, da_pct_revenue=da_pct,
    tax_rate=tax, terminal_growth=term_g, mid_year_convention=mid_year,
)

# Exit multiple defaults to the company's own current trading multiple
ebitda_now = float(drivers["ebitda"].iloc[-1])
net_debt = float(drivers["net_debt"].iloc[-1]) if pd.notna(drivers["net_debt"].iloc[-1]) else 0.0
entry_multiple = (market.market_cap + net_debt) / ebitda_now if ebitda_now > 0 else 12.0
exit_multiple = st.sidebar.slider("Exit EV/EBITDA", 3.0, 35.0,
                                  float(min(max(entry_multiple, 3.0), 35.0)), 0.5)
assumptions = assumptions.copy_with(exit_ev_ebitda=exit_multiple)


# ══════════════════════════════════════════════════════════════════════════════════════
# VALUE IT
# ══════════════════════════════════════════════════════════════════════════════════════
if term_g >= wacc - CFG.min_wacc_spread:
    st.error(
        f"**Terminal growth ({term_g:.2%}) must sit below WACC ({wacc:.2%}).** "
        "Otherwise the perpetuity diverges and the value is infinite, not large."
    )
    st.stop()

gordon = run_dcf(drivers, assumptions.copy_with(tv_method="gordon"), wacc,
                 market.shares_diluted, market.price)
exit_case = run_dcf(drivers, assumptions.copy_with(tv_method="exit_multiple"), wacc,
                    market.shares_diluted, market.price)

wacc_range, growth_range = build_sensitivity_ranges(wacc, term_g)
grid = sensitivity_grid(drivers, assumptions, market.shares_diluted,
                        wacc_range, growth_range, market.price)
scenarios = run_scenarios(drivers, assumptions, wacc, market.shares_diluted, market.price)
mc = monte_carlo_dcf(
    drivers, assumptions, wacc, market.shares_diluted,
    bridge_adjustment=gordon.equity_value - gordon.enterprise_value,
    current_price=market.price, n_sims=n_sims,
    terminal_growth_bounds=(max(term_g - 0.015, 0.0), term_g, term_g + 0.007),
)
diagnostics = valuation_diagnostics(gordon, drivers, report, market, wacc_result)


# ══════════════════════════════════════════════════════════════════════════════════════
# HEADLINE
# ══════════════════════════════════════════════════════════════════════════════════════
st.title(f"{meta['name']}")
age_min = (datetime.now() - price_fetched_at).total_seconds() / 60
stamp_col, refresh_col = st.columns([5, 1])
stamp_col.caption(
    f"{meta['ticker']} · CIK {meta['cik']} · {len(drivers)} fiscal years from SEC XBRL · "
    f"filings pulled {meta['fetched_at']:%H:%M} · "
    f"price {market.currency} as of {price_fetched_at:%H:%M} "
    f"({'live' if age_min < 1 else f'{age_min:.0f} min old'}) · "
    f"every chart on this page uses this one price"
)
if refresh_col.button("Refresh data", width="stretch"):
    load_filings.clear()
    load_market.clear()
    st.rerun()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current price", f"${market.price:,.2f}")
c2.metric("DCF — Gordon", f"${gordon.value_per_share:,.2f}",
          f"{gordon.upside_vs_price:+.1%}" if gordon.upside_vs_price is not None else None)
c3.metric("DCF — exit multiple", f"${exit_case.value_per_share:,.2f}",
          f"{exit_case.upside_vs_price:+.1%}" if exit_case.upside_vs_price is not None else None)
c4.metric("WACC", f"{wacc:.2%}", f"Ke {wacc_result.cost_of_equity:.1%}", delta_color="off")
c5.metric("P(undervalued)", f"{mc.prob_above_price:.0%}",
          f"P10–P90 ${mc.percentiles['10']:,.0f}–${mc.percentiles['90']:,.0f}",
          delta_color="off")

flagged = diagnostics[diagnostics["Status"] == "REVIEW"]
if not flagged.empty:
    with st.expander(f"⚠️ {len(flagged)} diagnostic check(s) need review", expanded=False):
        for _, row in flagged.iterrows():
            st.write(f"**{row['Check']}** — {row['Value']}  *(benchmark: {row['Benchmark']})*")

for note in market.warnings + wacc_result.notes:
    st.caption(f"ⓘ {note}")


# ══════════════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["Valuation", "Sensitivity", "Monte Carlo", "Football field",
                "History", "WACC", "Audit"])

with tabs[0]:
    left, right = st.columns(2)
    with left:
        show(plot_ev_to_equity_waterfall(gordon, market.shares_diluted))
    with right:
        show(plot_ufcf_bridge(gordon))

    st.subheader("Projection")
    proj = gordon.projection.copy()
    styled = proj.style.format(
        lambda v: f"{v:.1%}" if abs(v) < 5 else f"{v:,.0f}", na_rep="—")
    st.dataframe(styled, width="stretch")

    st.caption(
        f"Terminal value is {gordon.tv_share_of_ev:.0%} of enterprise value "
        f"({fmt_money(gordon.pv_terminal_value)} of {fmt_money(gordon.enterprise_value)}). "
        f"Gordon growth implies an exit multiple of {gordon.implied_exit_multiple:.1f}× EBITDA; "
        f"the {exit_multiple:.1f}× exit multiple implies "
        f"{exit_case.implied_growth_from_multiple:.2%} perpetual growth."
    )

with tabs[1]:
    show(plot_sensitivity_heatmap(grid, market.price))
    st.subheader("Scenarios")
    st.dataframe(
        scenarios.style.format({
            "Revenue growth (Y1)": "{:.1%}", "Terminal EBIT margin": "{:.1%}",
            "WACC": "{:.2%}", "Terminal growth": "{:.2%}",
            "Enterprise value": lambda v: fmt_money(v), "Equity value": lambda v: fmt_money(v),
            "Value per share": "${:,.2f}", "Upside": "{:+.1%}",
        }), width="stretch")
    st.caption(
        "Sensitivity varies one or two inputs across a grid; scenarios move a coherent set "
        "together. Both differ from the Monte Carlo, which samples every input jointly."
    )

with tabs[2]:
    show(plot_monte_carlo(mc, market.price, gordon.value_per_share))
    left, right = st.columns([1, 1.4])
    with left:
        st.dataframe(mc.summary_frame(), width="stretch")
        st.caption(f"{mc.n_valid:,} valid paths in {mc.runtime_seconds:.2f}s · "
                   f"{mc.n_discarded:,} discarded as infeasible (g ≥ WACC)")
    with right:
        show(plot_driver_tornado(driver_attribution(mc)))

with tabs[3]:
    st.caption("Add peer multiples to include a comps range.")
    p1, p2 = st.columns(2)
    peer_lo = p1.number_input("Peer EV/EBITDA — low", value=float(max(entry_multiple - 3, 1)), step=0.5)
    peer_hi = p2.number_input("Peer EV/EBITDA — high", value=float(entry_multiple + 3), step=0.5)

    adj = gordon.equity_value - gordon.enterprise_value
    ranges = {
        "DCF — WACC × g grid": (float(grid.min().min()), float(grid.max().max())),
        "DCF — Gordon vs exit": (min(gordon.value_per_share, exit_case.value_per_share),
                                 max(gordon.value_per_share, exit_case.value_per_share)),
        "Scenarios (bear–bull)": (float(scenarios["Value per share"].min()),
                                  float(scenarios["Value per share"].max())),
        "Monte Carlo P10–P90": (mc.percentiles["10"], mc.percentiles["90"]),
        "Comps — EV/EBITDA": ((peer_lo * ebitda_now + adj) / market.shares_diluted,
                              (peer_hi * ebitda_now + adj) / market.shares_diluted),
    }
    if market.week52_low and market.week52_high:
        ranges["52-week trading range"] = (market.week52_low, market.week52_high)
    show(plot_football_field(ranges, market.price, gordon.value_per_share))

with tabs[4]:
    show(plot_historical_drivers(drivers, meta["ticker"]))
    st.subheader("Driver table")
    st.dataframe(display_drivers(drivers), width="stretch")
    st.caption("Dollar figures in millions. Ratios in percent. Source: SEC XBRL companyfacts.")

with tabs[5]:
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(wacc_result.to_frame(), width="stretch", hide_index=True)
    with right:
        st.markdown(
            "**Cost of equity** is CAPM: risk-free rate plus levered beta times the equity "
            "risk premium.\n\n"
            "**Cost of debt** is interest expense over the average debt balance — the "
            "embedded rate on existing issuance — floored at the risk-free rate, since no "
            "company borrows below Treasuries.\n\n"
            "**Weights** are market value for equity and book value for debt, because most "
            "corporate debt trades close to par.\n\n"
            "**The tax shield** sits in the after-tax cost of debt, not in the cash flows. "
            "UFCF is deliberately unlevered — putting interest in both places double-counts it."
        )

with tabs[6]:
    st.subheader("Diagnostics")

    def _status_colour(row):
        """Green for PASS, amber for REVIEW.

        Explicit semantic colour rather than relying on the Streamlit theme: the default
        primary colour is red, so a theme that fails to load makes every focused row look
        like a failure.
        """
        tint = "#E9F3EC" if row["Status"] == "PASS" else "#FCF2DE"
        return [f"background-color: {tint}; color: #12222E"] * len(row)

    st.dataframe(diagnostics.style.apply(_status_colour, axis=1),
                 width="stretch", hide_index=True)
    st.caption("Green = passed. Amber = worth a look before you present the number. "
               "A REVIEW row is not necessarily an error — it flags something a reviewer "
               "would ask about.")

    st.subheader("XBRL tag audit")
    st.caption("Exactly which tag produced which line item. This is the table that survives "
               "a 'where did this number come from?' question.")
    st.dataframe(report.to_frame(), width="stretch", hide_index=True)

    st.subheader("Assumptions in force")
    # Every value is cast to string: a mixed int/bool/str column cannot be serialised
    # to Arrow, which is what Streamlit uses to ship dataframes to the browser.
    st.dataframe(pd.DataFrame([{
        "Horizon": str(horizon), "Y1 growth": f"{g1:.2%}", "Terminal margin": f"{margin:.2%}",
        "Capex % rev": f"{capex_pct:.2%}", "NWC % rev": f"{nwc_pct:.2%}",
        "D&A % rev": f"{da_pct:.2%}", "Tax rate": f"{tax:.2%}",
        "Terminal growth": f"{term_g:.2%}", "WACC": f"{wacc:.2%}",
        "Exit multiple": f"{exit_multiple:.1f}×", "Mid-year": str(mid_year),
    }]).T.rename(columns={0: "Value"}), width="stretch")


st.divider()
st.caption(
    "Built on the SEC XBRL `companyfacts` API, FRED and Yahoo Finance. "
    "**Not investment advice** — a DCF is a formalised opinion, not a measurement."
)
