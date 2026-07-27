# Automated DCF Valuation Engine from SEC Filings

**Ticker in → intrinsic value per share out.** A Python engine that pulls a decade of
reported financials directly from the SEC's XBRL API, reconstructs the unlevered free cash
flow drivers, computes WACC from CAPM, discounts to enterprise value, bridges to equity
value per share, and stress-tests the answer with a two-way sensitivity grid and a
vectorised Monte Carlo simulation.

No hardcoded financials. No manual data entry. No Excel.

```python
val = run_valuation("MSFT")
val.dashboard()
export_valuation(val)
```

**Three ways to run it**

| | Command | Best for |
|---|---|---|
| **Web app** | `streamlit run app.py` | Typing a ticker and getting an answer |
| **Notebook** | open `notebooks/dcf_engine_colab.ipynb` in Colab | Learning the model line by line |
| **Library** | `from dcf import run_valuation` | Batch runs, your own scripts |

```bash
git clone https://github.com/<your-username>/dcf-valuation-engine.git
cd dcf-valuation-engine
pip install -r requirements.txt
streamlit run app.py          # opens at localhost:8501
```

---

## 1. Project Overview

A discounted cash flow model is the single most tested technical skill in investment
banking and equity research recruiting. Most candidates have built one in Excel, where a
hardcode can hide anywhere and the model breaks the moment you change the company.

This project rebuilds the same analysis in Python with three deliberate constraints:

1. **Every input traces to a source.** Financials come from XBRL tags in the filings
   themselves; the engine logs which tag supplied which line item.
2. **Every assumption lives in one place.** A single `Assumptions` dataclass drives the
   projection, the sensitivity grid and the simulation, so nothing can be silently
   overridden downstream.
3. **The output is a distribution, not a number.** A point estimate implies a precision the
   inputs do not support. The Monte Carlo layer replaces "the stock is worth $312" with
   "63% of 50,000 plausible outcomes clear the current price."

| | |
|---|---|
| **Difficulty** | Intermediate |
| **Build time** | 20–30 hours (the XBRL tag mapping is the time sink) |
| **Lines of code** | ~1,600, fully documented |
| **Runtime** | 3–8 seconds cold, under 1 second cached |
| **Monte Carlo** | 50,000 paths in ~0.1s (fully vectorised) |

---

## 2. Real-World Finance Use Case

| Where it is used | What it answers |
|---|---|
| **IB coverage groups** | Fairness opinions, pitch materials, the DCF tab of every model |
| **Equity research** | Price targets and the published valuation methodology section |
| **Private equity** | Entry/exit underwriting, LBO downside cases |
| **Corporate development** | Build-vs-buy, target screening, board approval memos |
| **Long-only / hedge funds** | Position sizing against a distribution of intrinsic values |

The Monte Carlo layer maps to what risk teams actually do. A single-point DCF answers
"what is it worth?" A simulated distribution answers "how confident should I be, and what
has to be true for me to be wrong?" — which is the question that determines position size.

**What you can say in an interview:** *"I automated a DCF off the SEC XBRL API so the model
rebuilds itself for any US filer in about five seconds. The interesting part was the tag
mapping — there's no single revenue tag, so I built ordered fallback chains and anchored
every line item to the fiscal-year-end date from revenue, which handles 52/53-week fiscal
calendars. Then I layered a Monte Carlo over growth, margin and WACC, and rank-correlated
each driver to the output. On most large caps, WACC alone explains ~70% of the variance in
value — which tells you where to spend your diligence time."*

---

## 3. System Architecture

```
                        ┌──────────────────────────────────────┐
                        │            INPUT: "MSFT"             │
                        └──────────────────┬───────────────────┘
                                           ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  LAYER 1 — DATA ACQUISITION                                          │
    │  ┌────────────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐  │
    │  │ SEC ticker→CIK │→ │ companyfacts │  │ Yahoo     │  │ FRED      │  │
    │  │ company_tickers│  │ XBRL JSON    │  │ px/β/share│  │ DGS10 rf  │  │
    │  └────────────────┘  └──────────────┘  └───────────┘  └───────────┘  │
    │            disk cache (24h TTL) · retry w/ backoff · 10 req/s cap     │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  LAYER 2 — NORMALISATION                                             │
    │  tag fallback chains → duration vs instant split → fiscal-year        │
    │  anchoring (±25d) → restatement dedupe → sign conventions            │
    │  OUTPUT: historical driver table (10y × 38 line items)               │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  LAYER 3 — FEATURE ENGINEERING                                       │
    │  margins · capex intensity · NWC % revenue · effective tax rate      │
    │  · historical UFCF · FCF conversion · ROIC                           │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  LAYER 4 — VALUATION CORE          Assumptions (single dataclass)    │
    │  projection → UFCF → WACC → discount → TV (Gordon + exit) → bridge   │
    └───────────┬──────────────────┬───────────────────┬───────────────────┘
                ▼                  ▼                   ▼
    ┌────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
    │ Sensitivity    │  │ Scenarios        │  │ Monte Carlo          │
    │ WACC × g grid  │  │ bear/base/bull   │  │ 50k vectorised paths │
    └───────┬────────┘  └────────┬─────────┘  └──────────┬───────────┘
            └────────────────────┴───────────────────────┘
                                 ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │  LAYER 5 — OUTPUT                                                    │
    │  tearsheet · 6-panel dashboard · ipywidgets panel · Excel · PNG      │
    │  · self-scoring diagnostics report                                   │
    └──────────────────────────────────────────────────────────────────────┘
```

**Design principle:** each layer returns a plain `DataFrame` or dataclass, so any stage can
be inspected, unit-tested, or swapped (e.g. point Layer 1 at Bloomberg or CapIQ without
touching Layers 2–5).

---

## 4. Required APIs and Data Sources

| Source | Endpoint | Key? | Used for |
|---|---|---|---|
| **SEC ticker map** | `sec.gov/files/company_tickers.json` | No | ticker → CIK |
| **SEC companyfacts** | `data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | No | every tagged number in every filing |
| **SEC companyconcept** | `data.sec.gov/api/xbrl/companyconcept/.../{tag}.json` | No | debugging one stubborn tag |
| **FRED** | `fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10` | No | 10-year Treasury (risk-free rate) |
| **Yahoo Finance** | `yfinance` | No | price, beta, shares, 52-week range |
| **Damodaran** | `pages.stern.nyu.edu/~adamodar/pc/datasets/` | No | implied ERP, industry betas, country premia |

**The SEC's only requirement is a `User-Agent` header with a real name and email.** Requests
without one get a 403. Their fair-access policy caps you at 10 requests/second — the client
throttles to ~8/s and caches responses for 24 hours.

Coverage caveat: `companyfacts` covers US GAAP filers (10-K, 10-Q). Foreign private issuers
filing 20-F under IFRS use different tags; banks and insurers need a bespoke tag chain
because "revenue" and "capex" do not mean the same thing for them.

---

## 5. Required Python Libraries

```
pandas>=2.0        # driver tables, projections, Excel export
numpy>=1.24        # vectorised Monte Carlo, discounting
requests>=2.31     # SEC/FRED HTTP with retry and backoff
numpy-financial    # NPV/IRR cross-checks
scipy>=1.11        # distributions, Spearman rank correlation, KDE
matplotlib>=3.7    # all charts
yfinance>=0.2      # price, beta, shares, 52-week range
ipywidgets>=8.0    # interactive assumption sliders
openpyxl>=3.1      # Excel workbook export
```

Colab ships everything except `yfinance` and `numpy-financial`; Cell 1 installs those two
only if they are missing.

---

## 6. Folder / File Structure

Each module corresponds to one cell of the Colab notebook, so you can learn it in the
notebook and ship it as a package.

```
dcf-valuation-engine/
├── app.py                            # Streamlit front end — the "type a ticker" app
├── README.md
├── LICENSE                           # MIT
├── pyproject.toml                    # installable package metadata
├── requirements.txt                  # runtime deps (what Streamlit Cloud reads)
├── requirements-dev.txt              # + pytest, ruff
├── .gitignore                        # excludes output/, dcf_cache/, secrets.toml
├── .streamlit/
│   ├── config.toml                   # theme matching the chart palette
│   └── secrets.toml.example          # template for SEC_USER_AGENT
├── .github/workflows/tests.yml       # CI: ruff + pytest on 3.10 and 3.12
├── notebooks/
│   ├── dcf_engine_colab.ipynb        # the full engine, runnable in Colab
│   └── dcf_engine_colab.py           # same code, cell-marked for copy-paste
├── src/dcf/
│   ├── __init__.py                   # public API
│   ├── config.py                     # Config dataclass, palette, formatters
│   ├── sec_client.py                 # CIK lookup, caching, retries, rate limiting
│   ├── xbrl_tags.py                  # TAG_MAP + XBRLExtractor
│   ├── drivers.py                    # build_driver_table, ratios, historical UFCF
│   ├── market_data.py                # Yahoo/FRED, beta unlever/relever
│   ├── wacc.py                       # compute_wacc
│   ├── assumptions.py                # Assumptions, seed_from_history
│   ├── dcf.py                        # project_financials, run_dcf, EV→equity bridge
│   ├── sensitivity.py                # grids + scenarios
│   ├── monte_carlo.py                # vectorised simulation + driver attribution
│   ├── charts.py                     # the seven chart functions
│   ├── diagnostics.py                # self-scoring quality report
│   ├── engine.py                     # Valuation, run_valuation orchestrator
│   └── export.py                     # 13-tab Excel workbook + PNG
├── tests/
│   └── test_engine.py                # 21 tests against a synthetic SEC payload
└── output/                           # generated workbooks and PNGs (gitignored)
```

**Testing without the network.** `tests/test_engine.py` builds a synthetic `companyfacts`
payload — including a quarterly fact that must be rejected and a stale restatement that must
lose to the later filing — then patches the SEC client at class level. Every layer from tag
extraction to Excel export runs exactly as it would in production, with no HTTP calls:

```bash
pytest -v          # 21 passed in ~5s
```

---

## 7. Step-by-Step Build

| # | Step | What you learn |
|---|---|---|
| 1 | **SEC wrapper** — ticker→CIK, `companyfacts`, cache, retry | API etiquette, idempotent caching |
| 2 | **Tag dictionary** — ordered fallback chains per line item | Why financial data engineering is hard |
| 3 | **Period alignment** — anchor on revenue's FY-end, match ±25 days | 52/53-week fiscal calendars, restatements |
| 4 | **Driver table** — revenue, EBIT, D&A, capex, ΔNWC | What actually drives a DCF |
| 5 | **Ratios** — margins, capex intensity, NWC % revenue | How forecasts get anchored to history |
| 6 | **Assumption dataclass** — one object, validated | Why hardcodes are the enemy |
| 7 | **Projection engine** — 5-year build with growth/margin fade | Forecast discipline |
| 8 | **UFCF** — `EBIT(1−t) + D&A − Capex − ΔNWC` | Why unlevered, why WACC |
| 9 | **WACC** — CAPM, after-tax Kd, market-value weights | Capital structure, the tax shield |
| 10 | **Terminal value** — Gordon *and* exit multiple, reconciled | The 70% of your value you can't forecast |
| 11 | **EV→equity bridge** — debt, minorities, preferred, investments | Where analysts lose points |
| 12 | **Sensitivity grid** — WACC × g | Fragility, not precision |
| 13 | **Monte Carlo** — vectorised, 50k paths | Simulation ≠ sensitivity ≠ scenario |
| 14 | **Charts** — waterfall, heatmap, football field, histogram | Presenting to a non-technical audience |
| 15 | **Diagnostics** — self-scoring quality report | Defending the model under questioning |

---

## 8. Data Collection Pipeline

```
ticker
  └─ GET company_tickers.json ──────────► CIK (zero-padded to 10 digits)
       └─ GET companyfacts/CIK##########.json  (~5–40 MB)
            └─ facts.us-gaap.{tag}.units.USD[]  → list of observations:
                 { start, end, val, form, filed, fy, fp, accn, frame }
```

Each observation is filtered through four gates before it enters the model:

1. **Form gate** — `10-K`, `10-K/A`, `20-F`, `40-F` only. Quarterly forms are discarded.
2. **Duration gate** — flow items must span 300–400 days. This is what kills the classic
   bug where a Q4 figure gets treated as a full year.
3. **Instant gate** — balance items must have no `start` key.
4. **Restatement gate** — when the same period appears twice, the most recently `filed`
   value wins.

Then the surviving facts are aligned: revenue's fiscal-year-end dates become the anchors,
and every other tag is matched to the nearest anchor within ±25 days.

**Resilience:** 24-hour disk cache, exponential backoff over four attempts, explicit 403
handling that tells you to fix your `User-Agent`, and graceful degradation to manual
overrides if Yahoo or FRED are unreachable.

---

## 9. Data Cleaning and Feature Engineering

**Cleaning**

| Problem | Fix |
|---|---|
| No universal revenue tag | Ordered fallback chain, first adequate tag wins, choice logged |
| `OperatingIncomeLoss` missing | Derive: pre-tax income + interest expense, or gross profit − opex |
| Capex signed as a positive "payment" | `abs()`, then subtract in the UFCF build |
| One-off tax benefit → negative tax rate | Winsorise the effective rate to 8–35% |
| Detailed NWC tags absent | Fall back to `AssetsCurrent`/`LiabilitiesCurrent` aggregates |
| Beta unavailable or absurd | Flag outside 0.3–2.5; peer unlever/relever helper provided |
| Interest ÷ debt implies 0.3% or 40% | Clamp to `[rf + 50bps, rf + 1000bps]` and log the clamp |

**Engineered features**

```
revenue_growth        = revenue.pct_change()
ebit_margin           = ebit / revenue
ebitda                = ebit + d_and_a
da_pct_revenue        = d_and_a / revenue
capex_pct_revenue     = capex / revenue
nwc                   = (AR + inventory + other CA) − (AP + accruals + deferred rev + other CL)
nwc_pct_revenue       = nwc / revenue
delta_nwc             = nwc.diff()                     # an increase is a USE of cash
effective_tax_rate    = clip(tax expense / pre-tax income, 8%, 35%)
ufcf_historical       = EBIT(1−t) + D&A − capex − ΔNWC
fcf_conversion        = ufcf_historical / ebitda
roic                  = EBIT(1−t) / (equity + debt − cash)
```

**Why operating NWC excludes cash and debt:** enterprise value is the value of the operating
business. Cash and debt are financing items handled explicitly in the EV→equity bridge —
including them in working capital double-counts them.

**Why medians, not means, anchor the forecast:** one pandemic year, one large acquisition or
one impairment should not set the base case. Medians over a 5-year lookback are robust to
exactly the outliers that show up in reported financials.

---

## 10. Core Models and Algorithms

### Unlevered free cash flow
$$UFCF_t = EBIT_t(1-\tau) + D\&A_t - Capex_t - \Delta NWC_t$$

Unlevered because it belongs to *all* capital providers — which is why it is discounted at
WACC and yields enterprise value. Interest is deliberately absent: the tax shield is already
in the after-tax cost of debt, and putting it in both places double-counts it.

### WACC
$$WACC = \frac{E}{D+E}\big(r_f + \beta\,ERP + \alpha\big) + \frac{D}{D+E}\,r_d(1-\tau)$$

Weights at **market value** for equity, book for debt (most corporate debt trades near par).
Cost of debt is interest expense ÷ *average* debt balance, floored at the risk-free rate.

### Discounting with the mid-year convention
$$PV = \sum_{t=1}^{n} \frac{UFCF_t}{(1+WACC)^{\,t-0.5}}$$

Cash arrives throughout the year, not in a lump on 31 December. Worth roughly +4% at a 9%
WACC. Banks use it; the flag is exposed so you can show both.

### Terminal value — computed both ways and reconciled
$$TV_{Gordon} = \frac{UFCF_n(1+g)}{WACC-g} \qquad TV_{exit} = EBITDA_n \times \text{multiple}$$

The cross-check that separates a real model from a class assignment — back out the growth
the exit multiple implies:
$$g_{implied} = \frac{TV \cdot WACC - UFCF_n}{TV + UFCF_n}$$

If your 15× exit multiple implies 5% perpetual growth, the multiple is too high.

### EV → equity bridge
```
Enterprise value
  − total debt (incl. finance leases)
  − minority interest
  − preferred stock
  + cash & equivalents
  + short- and long-term investments
  + equity-method investments
= Equity value  ÷  diluted shares  =  value per share
```

### Monte Carlo — vectorised
The naive implementation loops 50,000 times calling `run_dcf`. This version builds the whole
simulation as `(n_sims × horizon)` NumPy arrays and computes every path at once — roughly a
**500× speed-up**, and exactly the vectorisation question quant interviews probe.

| Driver | Distribution | Rationale |
|---|---|---|
| Year-1 revenue growth | Normal, σ from the company's own 5-year volatility | Symmetric, empirically estimable |
| Terminal EBIT margin | Normal, truncated | Margins mean-revert |
| WACC | Normal, σ ≈ 100bps | It is an *estimate* with standard error, not a constant |
| Terminal growth | Triangular | Hard ceiling at GDP, more room on the downside |
| Capex % revenue | Normal, σ ≈ 50bps | Capital intensity is sticky but not fixed |

Draws where $g \geq WACC$ are discarded and the discard rate reported — above ~5% and your
input ranges are not credible.

**Driver attribution** uses Spearman rank correlation rather than Pearson, because the map
from WACC to value is convex; rank correlation captures monotone relationships without
assuming linearity.

---

## 11. Visualisations and Dashboard Components

| Chart | Question it answers |
|---|---|
| **Historical driver panel** | Revenue bars with margin, capex- and D&A-intensity lines — what history says |
| **UFCF bridge by year** | Is cash flow driven by margin, or eaten by capital intensity? |
| **EV → equity waterfall** | Where did value go between enterprise and equity? |
| **WACC × g heatmap** | How fragile is this number? Cells coloured by upside vs market |
| **Football field** | DCF range vs scenarios vs Monte Carlo vs comps vs 52-week range |
| **Monte Carlo histogram** | KDE overlay, P10/median/P90 marked, current price and P(undervalued) |
| **Driver tornado** | Which assumption actually explains the variance? |

Plus a **six-panel composite dashboard** (`val.dashboard()`), a printed **tearsheet**, and an
**ipywidgets panel** with live sliders that re-run the DCF on every change.

House style: one palette, direct labelling instead of legends where possible, banker money
formatting (`$1.2bn`, `$840.5mm`), source annotations, no chartjunk.

---

## 12. Performance Metrics

A valuation model has no R². Quality is measured on **data integrity** and **internal
consistency**, and the engine grades itself before you present it:

| Metric | Benchmark | Why it matters |
|---|---|---|
| XBRL coverage — required items | > 80% | Below this the model is guessing |
| Fiscal years extracted | ≥ 5 | Fewer and the medians are noise |
| Terminal value share of EV | < 85% | Above it you are valuing an assumption, not a business |
| Implied exit vs entry EV/EBITDA | within ±40% | Catches a terminal growth rate that is too aggressive |
| Growth implied by exit multiple | 0–4% | The Gordon/exit reconciliation |
| WACC | 6–14% | Outside it, check beta and ERP |
| WACC − g spread | > 3% | Thin spreads make the perpetuity explode |
| Reinvestment-implied growth | within 10pp of forecast | `g ≈ ROIC × reinvestment rate` — is the growth funded? |
| Reconstructed UFCF vs reported FCF | MAPE < 30% | Back-test of the tag mapping itself |
| Monte Carlo infeasible-draw rate | < 5% | Sanity of the input ranges |

**Engineering benchmarks:** cold run 3–8s (SEC payloads are 5–40MB), cached run < 1s,
50,000-path Monte Carlo in ~0.1s, full six-panel dashboard in ~2s.

---

## 13. Final Deliverables

1. **`dcf_engine_colab.ipynb`** — the full engine, 15 cells, runs top to bottom in Colab.
2. **`dcf_engine_colab.py`** — the same code as a cell-marked script for copy-paste.
3. **Excel workbook** (`export_valuation`) with tabs: Summary · Historical drivers ·
   Projection (Gordon) · Projection (Exit) · WACC · EV-Equity bridge · Sensitivity ·
   Scenarios · Diagnostics · **XBRL tag audit** · Assumptions · Monte Carlo · MC draws.
4. **Dashboard PNG** — the six-panel summary, deck-ready.
5. **Printed tearsheet** — the one-screen summary you paste into an email.
6. **Diagnostics report** — every check, with PASS/REVIEW status.

The **XBRL tag audit tab** is the differentiator. It shows exactly which tag produced every
line item, which were derived and which were missing. That is the tab that survives a
"where did this number come from?" question.

---

## 14. Résumé Description

> **Automated DCF Valuation Engine** — *Python, SEC XBRL API, NumPy, SciPy, Matplotlib*
> Built an end-to-end equity valuation engine that ingests 10 years of reported financials
> from the SEC XBRL API for any US filer and produces an intrinsic value per share in under
> five seconds. Engineered an ordered tag-resolution layer with fiscal-period anchoring to
> normalise inconsistent filer tagging across 38 line items. Implemented WACC via CAPM,
> dual terminal-value methodologies with implied-growth reconciliation, a full EV-to-equity
> bridge, and a fully vectorised 50,000-path Monte Carlo simulation (~500× faster than a
> naive loop) with Spearman-based driver attribution.

**Short version (one line):**
> Automated DCF engine in Python: SEC XBRL ingestion → UFCF projection → WACC → EV-to-equity
> bridge, with sensitivity grids and a vectorised 50k-path Monte Carlo.

**Interview talking points to have ready**
- Why unlevered FCF and WACC, not levered FCF and cost of equity
- Where the tax shield sits, and why it appears exactly once
- Mid-year convention: what it is worth (+3–5%) and why banks use it
- Reconciling Gordon growth against the exit multiple
- Why NWC excludes cash and debt
- Which driver actually moves the answer (usually WACC, ~70% of variance) and what that
  implies about where to spend diligence time

---

## 15. Potential Upgrades

**Analytical**
- **Segment-level DCF** — value each reporting segment separately using the segment tags in
  `companyfacts`, then sum the parts. Standard for conglomerates.
- **Operating-lease capitalisation** — capitalise leases at 8× rent, add to debt and to the
  asset base. Materially changes retail and airline valuations.
- **Stock-based compensation** — treat SBC as a real cost and model dilution explicitly,
  rather than the add-back that flatters most tech DCFs.
- **Damodaran industry betas and country risk premia** — pull his tables directly instead of
  relying on Yahoo's regression beta.
- **Correlated Monte Carlo** — sample growth and margin from a joint distribution with the
  empirical correlation, since they are not independent in practice. Cholesky decomposition
  on the historical covariance matrix.
- **Reverse DCF** — solve for the growth and margin the *current price* implies. Often more
  useful than the forward DCF: it tells you what the market believes.

**Engineering**
- Batch mode across a whole sector, ranked by implied upside
- Full-text 10-K parsing to pull guidance and management commentary into the assumptions
- Streamlit or Dash front end, deployed, with a shareable link
- SQLite or Parquet layer so historical runs are queryable over time
- CI with `pytest` on the synthetic-payload fixtures and a nightly run against 20 tickers
- Comps module scraping peer multiples so the football field is fully automated

**Presentation**
- Auto-generate a one-page PDF tearsheet with `reportlab`
- Write straight into a formatted Excel template with `xlsxwriter` (banker formatting,
  blue inputs / black formulas)

---

## Running It as an App

```bash
streamlit run app.py
```

Enter your SEC User-Agent once in the sidebar, type a ticker, press **Run valuation**. The
5–40MB SEC payload is cached for an hour per ticker, so dragging an assumption slider
re-runs only the valuation maths — milliseconds, not seconds.

**Seven tabs:** Valuation (waterfall, UFCF bridge, projection) · Sensitivity (heatmap,
scenarios) · Monte Carlo (histogram, driver tornado) · Football field (with editable peer
multiples) · History (driver chart and table) · WACC (component build) · Audit (diagnostics
and the XBRL tag trail).

### Deploying to Streamlit Community Cloud (free)

1. Push the repo to GitHub (public).
2. Go to **share.streamlit.io** → **New app** → pick the repo, branch `main`, main file
   `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   SEC_USER_AGENT = "Your Name your@email.com"
   ```
   The app reads this so visitors don't have to supply their own.
4. Deploy. You get a permanent `https://<name>.streamlit.app` URL — put it on your CV.

**Two things that bite on deployment.** Yahoo throttles cloud IP ranges more aggressively
than residential ones, so the sidebar has manual price/share/beta overrides — keep them
visible. And free-tier apps sleep after inactivity; the first visitor waits ~30 seconds for
the container to wake.

---

## Running It in Google Colab

1. **Upload the notebook** — `File → Upload notebook` and drop in `dcf_engine_colab.ipynb`.
   (Or paste the cells from `dcf_engine_colab.py`; each `# %%` marks a new cell.)
2. **Set your User-Agent** in Cell 1:
   ```python
   user_agent: str = "Your Name your.email@university.edu"
   ```
   The SEC returns 403 without it. This is their only "API key".
3. **Run Cells 1–12** in order (~30 seconds, mostly the `yfinance` install).
4. **Run the valuation** in Cell 13:
   ```python
   val = run_valuation("MSFT")
   val.dashboard()
   ```
5. **Adjust assumptions live** in Cell 14: `build_assumption_panel(val)`
6. **Export** in Cell 15: `export_valuation(val)` — files land in `/content/output/`.

**Enable widgets in Colab** (Cell 14 does this automatically):
```python
from google.colab import output
output.enable_custom_widget_manager()
```

**Tickers that work well:** `MSFT`, `AAPL`, `GOOGL`, `NVDA`, `HD`, `NKE`, `COST`, `UNH`, `CAT`, `LMT`.
**Tickers that will not:** banks and insurers (`JPM`, `BAC`, `AIG`) — capex and working
capital do not mean the same thing for financials, so a DCF on them is conceptually wrong
without a dividend-discount or excess-return rebuild. Also skip pre-revenue biotech and any
foreign issuer filing 20-F under IFRS.

**If Yahoo throttles Colab** (it sometimes does):
```python
val = run_valuation("MSFT", price_override=430.0, shares_override=7.43e9, beta_override=0.90)
```

---

## Licence and Disclaimer

MIT. **Not investment advice.** This is an educational tool. Every valuation is only as good
as its assumptions, and a DCF is a formalised opinion, not a measurement.
