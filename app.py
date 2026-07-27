
Claude finished the response

Diagnosed decimal scaling mismatch in slider formatting logic
Diagnosed decimal scaling mismatch in slider formatting logic
Sharp catch — and no, your math is fine. The valuation is correct; the slider labels are wrong. That's a bug in my app, and you found it.


Diagnosed decimal scaling inconsistency across percentage-based sliders
Diagnosed decimal scaling inconsistency across percentage-based sliders
Confirmed. Streamlit's format is printf-style applied to the raw stored value — "%.2f%%" on 0.0882 prints 0.09%. The %% is just a literal percent sign; it doesn't scale anything. Meanwhile the metric tiles at the top use Python f-strings (f"{wacc:.2%}"), which do multiply by 100. So the same number renders two different ways in two places, which is exactly what you spotted.

The stored values were always right — WACC really was 0.0882 = 8.82%, and every calculation used that. Only the sidebar labels lied. Let me fix it properly.


Ran 3 commands, read a file
Ran 3 commands, read a file
Fixed and verified — the sidebar now reads what it means:

Slider	Was showing	Now shows	Value fed to model
WACC	0.09%	8.82%	0.0882
Terminal growth	0.03%	2.50%	0.0250
Year-1 revenue growth	0.09%	9.00%	0.0900
Equity risk premium	0.04%	4.50%	0.0450
Terminal EBIT margin	0.28%	27.60%	0.2760
Same for capex, D&A, NWC and tax. All nine.

What was actually wrong
The sliders were storing decimals (0.0882) but labelling them with a printf format string. "%.2f%%" prints the number to two decimals and then appends a literal % sign — it never scales by 100. So 0.0882 rendered as 0.09%.

The metric tiles used Python f-strings instead, and f"{0.0882:.2%}" does scale by 100, giving 8.82%. Two formatting systems, one number, two answers. The fix is a pct_slider() helper that runs the widget in percentage points and divides by 100 on the way out, so the label and the arithmetic finally agree.

None of your valuations were wrong. Every calculation used the stored decimal, which was always correct. If you took a screenshot of a valuation earlier, the numbers in it are still good — the sidebar was just mislabelling its own inputs.

Updating your deployed app
You only need to replace one file:

On your GitHub repo page, click app.py
Click the pencil icon (top right) to edit
Select all (Cmd + A) and delete
Open the new app.py above, copy everything, paste it in
Scroll down, Commit changes
Streamlit Cloud watches the repo and redeploys automatically — about a minute. Refresh the app and the sliders will read properly.

Alternatively, re-download the zip and re-upload the whole repo, but editing the one file is faster.

Worth noticing
This is a good bug to have found, because the failure mode was silent. Nothing errored. The model was right, the labels were wrong, and the only way to catch it was for someone to look at a number and think "that's not what a WACC looks like." Which you did.

That instinct — knowing that 0.20% is not a plausible cost of capital and 2% is — is doing more work for you than most of the code in that repo. Keep applying it to the outputs too: if a DCF hands you a $400 price target on a $30 stock, the model isn't clever, it's broken.


App
PY 

Dcf valuation engine
ZIP 
Financial Logic: Broken Projections & Extreme WACC
Looking specifically at the first screenshot (the TSLA dashboard), the valuation outputs are completely broken, which usually indicates an error in the assumption-seeding logic.

Negative Free Cash Flow: Look at the "Unlevered free cash flow build by forecast year" chart in Image 1. Your projected UFCF is massively negative (around -$4.3bn) for every single forecast year (FY+1 through FY+7). A DCF with perpetual negative cash flows will obviously result in a near-zero valuation, which is why your implied share price is $4.76. You need to check how your app is seeding CapEx or NWC assumptions based on historical medians—it seems to be projecting unsustainable capital destruction.
20% WACC: The dashboard in Image 1 shows a WACC of 20.00%. For a mega-cap stock like Tesla, this is an astronomical discount rate. (Meanwhile, your other screenshots show a much more reasonable 8.89% WACC). You need to audit why the default CAPM calculation for TSLA spiked to 20%—perhaps an error in how it fetched the levered beta?

Data Consistency

Conflicting Current Prices: Image 1 shows the current price as $307.33. However, your Monte Carlo chart (Image 4) and Football Field chart (Image 6) show a "Current price" of $391.73. Your live market data fetcher needs to be synced so the price is uniform across all tabs and charts.

UI/UX Tweaks

Confusing Color Status: In the Diagnostics tab (Image 10), the row for "XBRL coverage — all mapped items" has a status of "PASS", but it is surrounded by a red bounding box. In UI design, red heavily implies an error, failure, or warning. Even if this is just a "selected" or "active" focus state, you should change the accent color to blue, gray, or green to avoid causing the user immediate panic that a check failed.
Chart Overlaps: On the Football Field chart (Image 6), the text labels for the ranges (e.g., "$180" on the Scenarios bar) overlap slightly with the bar itself, making it a bit cluttered. Similarly, the red "Current price $391.73" label overlaps the chart title. Adding a bit more padding or dynamically positioning those labels will clean that up.
Show more

Diagnosed slider display bugs, validated financial logic, and pinpointed chart rendering issues
Diagnosed slider display bugs, validated financial logic, and pinpointed chart rendering issues
Two things before I dig in: no images came through on this message — I can see your text but not the screenshots. And separately, I think three of these five have the same root cause, which is the bug you caught last turn.

Let me test that rather than assert it.


Investigated root causes of negative cash flow projections and valuation discrepancies
Investigated root causes of negative cash flow projections and valuation discrepancies
The seeder is not the problem — that's the key finding. Let me confirm what actually produced −$4.3bn.


Diagnosed financial model bugs and architected UI refinements systematically
Diagnosed financial model bugs and architected UI refinements systematically
Diagnosis confirmed. Now the fixes.


Identified layout completion and prioritized financial logic debugging next
Identified layout completion and prioritized financial logic debugging next
Clean — labels sit outside the bars, and the price callout has its own headroom band clear of the title. Now the app-side fixes.


Ran 3 commands, read a file
Ran 3 commands, read a file
Good report. Three of the five turn out to be the same bug you already caught, and two were real defects I've now fixed.

The 20% WACC and negative UFCF are one thing, not two
Both are downstream of the slider display bug from last turn — and the tell is that 20.00% is exactly the slider maximum. That's not a CAPM output; that's a slider pinned to the right edge.

Here's what happened. Before the fix, the WACC slider held 0.1456 and displayed 0.15%. You saw a number that looked absurdly low, dragged it right to make it look sensible, and hit the ceiling — 0.20, which meant 20%. Same story on the other sliders. Your horizon showing FY+7 confirms it: the default is 5, so you were definitely moving things.

I tested whether the seeder was at fault by building a Tesla-shaped filer and running it with nothing touched:

AUTO-SEEDED BASE CASE
  revenue growth y1   18.77%
  EBIT margin          7.27% → 8.23%
  capex % revenue     10.16%
  D&A % revenue        5.39%
  NWC % revenue       -9.57%

UFCF PATH:  FY+1 $2.9bn · FY+2 $3.2bn · FY+3 $3.3bn · FY+4 $3.3bn · FY+5 $3.2bn
Positive every year. The seeding logic is sound. Then I searched for what produces your −$4.3bn:

Slider state	FY+1	FY+4	FY+7
defaults	$2.9bn	$3.8bn	$3.9bn
margin dragged 8.2% → 2%	−$1.9bn	−$3.8bn	−$5.8bn
capex dragged 10.2% → 20%	−$8.5bn	−$12.8bn	−$15.5bn
The margin case brackets your −$4.3bn. That's the one — you dragged the EBIT margin slider down because 0.08% looked wrong, and landed near 2%.

Fix the sliders and both symptoms disappear. No change needed to the seeding logic.

One caveat on my test: those Tesla figures are approximate, reconstructed from memory rather than pulled from the filings. The shape is right, which is all that's needed to test the seeder — but don't quote the specific outputs.

The two prices are from two different runs
Within a single render this is structurally impossible. The app fetches market once and every chart and metric reads market.price off that one object. There's no second fetch path.

So $307.33 and $391.73 came from different runs — either separate sessions, or the 15-minute market-data cache expiring between screenshots.

That said, you found a real gap: nothing told you which vintage you were looking at. Added a stamp under the headline:

TSLA · CIK 1318605 · 10 fiscal years from SEC XBRL · filings pulled 14:02 · price USD as of 14:02 (3 min old) · every chart on this page uses this one price

Plus a Refresh data button that clears both caches and reruns, for when you want to force a fresh pull.

The red box is almost certainly a missing file
I never set row colours on that table — it renders plain. Red on a PASS row points at Streamlit's default primary colour, 
#FF4B4B, used for focus and selection borders.

My .streamlit/config.toml overrides it to navy. But that folder starts with a dot, and I warned when you were uploading that browsers frequently skip dotfolders in drag-and-drop. Check your repo — if .streamlit/ isn't there, that's your red box.

Belt and braces, I've made the status column colour itself explicitly rather than depending on the theme loading: green tint for PASS, amber for REVIEW, with a caption noting that REVIEW means "a reviewer would ask about this," not "this failed."

Football field labels
Real bug, fixed. The old version hardcoded label offsets, so narrow bars collided with their own end labels and the price callout sat at a fixed height that ran into the title.

Now the extent is computed from the data first, 16% margin is reserved on each side for end labels, and the price callout gets a dedicated headroom band above the bars with a bordered box. I stress-tested it with three bars spanning $99.50–$100.50 — worst case for collisions — and it holds.

Updating
Two files changed: app.py and src/dcf/charts.py. Edit both on GitHub (pencil icon, select all, paste, commit) and Streamlit redeploys in about a minute. Or re-upload the whole zip.

While you're in there, confirm .streamlit/config.toml exists.

One thing worth sitting with. When I ran the Tesla-shaped test with defaults and a computed 14.6% WACC, it returned roughly $17/share against a $307 market price. My inputs are approximate so don't take the figure literally — but a DCF on auto-manufacturing fundamentals returning a fraction of Tesla's market price is not a malfunction. It's the model correctly reporting that the price embeds things a five-year unlevered FCF projection does not: robotaxi optionality, energy storage, licensing, a belief about what the company becomes.

That's the same lesson as Lucid from the other direction. The tool tells you what the cash flows are worth. When the market pays a large multiple of that, the interesting question isn't "what's wrong with my model" — it's "what is the market pricing that I haven't modelled?" That framing is worth considerably more in an interview than a matching price target.


Dcf valuation engine
ZIP 

App
PY 

Charts
PY 
alright how do i update my code on github


Orchestrated step-by-step GitHub file editing walkthrough with deployment details
Orchestrated step-by-step GitHub file editing walkthrough with deployment details
Two files changed: app.py (in the root) and charts.py (which lives at src/dcf/charts.py). Here's the click-by-click.

Updating app.py
1. Go to your repo: github.com/YOUR-USERNAME/dcf-valuation-engine

2. In the file list on the main page, click app.py.

3. Top right of the file view, there's a row of small icons. Click the pencil (✏️). Hovering shows "Edit this file." The file becomes an editable text box.

4. Click anywhere inside the code, then press Cmd + A to select everything, then Delete. The box should be completely empty.

5. Open the new app.py I posted above, select all of it, copy.

6. Click back in the empty GitHub editor and press Cmd + V.

7. Scroll to the bottom (or click the green Commit changes… button top right). A dialog appears. In the message box type something like Fix percentage slider display, add data vintage stamp. Leave Commit directly to the main branch selected.

8. Click Commit changes.

Updating charts.py
Same process, but you have to navigate into the folders first.

1. From the repo main page, click the src folder.

2. Click the dcf folder.

3. Click charts.py.

4. Pencil icon → Cmd+A → Delete → paste the new charts.py → Commit changes.

Make sure you're editing the one at src/dcf/charts.py. I named the file I posted just charts.py, but it belongs in that folder — don't create a new one at the root.

While you're in there
Check whether .streamlit/ made it into the repo. On the main repo page, look for a folder called .streamlit in the file list. GitHub does show dotfolders, so if it's not listed, it never uploaded — and that's your red bounding box.

If it's missing:

On the repo main page, click Add file → Create new file
In the filename box type exactly: .streamlit/config.toml — typing the slash automatically creates the folder
Paste this in:
toml
[theme]
primaryColor = "#1B3A5B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#E4E9ED"
textColor = "#12222E"
font = "sans serif"

[browser]
gatherUsageStats = false
Commit changes
Watching the redeploy
Streamlit Cloud watches your repo and rebuilds automatically on every commit. Takes 1–2 minutes.

Open your app URL. In the bottom right corner there's a Manage app button — click it to see the build log streaming. When it stops and the app reloads, you're on the new version.

Quickest confirmation it worked: the WACC slider in the sidebar should read something like 8.82%, not 0.09%.

If nothing seems to change, hit Manage app → ⋮ → Reboot app. Occasionally it needs the nudge. And do a hard browser refresh (Cmd + Shift + R) since Streamlit caches aggressively.

If you'd rather just replace everything
Redownload the zip, unzip it, then:

Repo main page → Add file → Upload files
Open the fresh dcf-valuation-engine folder in Finder, Cmd + A to select the contents
Drag onto the upload area
Commit changes
Files with matching paths get overwritten. This is more clicks but avoids any copy-paste mistakes, and it catches any file I've changed that you'd otherwise miss.

Worth setting up: GitHub Desktop
You're going to keep iterating on this, and browser editing gets old fast. desktop.github.com — free, no Terminal.

Install it, sign in, File → Clone repository, pick dcf-valuation-engine, choose where to save it. Now you have a local folder linked to GitHub. Edit files however you like, and GitHub Desktop shows you exactly what changed, lets you write a commit message, and pushes with one button.

The real benefit is that you can run the app locally (streamlit run app.py) to test a change before pushing it, instead of deploying and hoping. Given how the last few bugs surfaced, that loop is worth the twenty minutes of setup.



Want to be notified when Claude responds?





Claude is AI and can make mistakes. Please double-check responses.
App · PY
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
 
