"""End-to-end tests against a synthetic SEC ``companyfacts`` payload.

No network access required — the SEC client is patched at the class level, so every layer
from tag extraction through to the Excel export runs exactly as it would in production.

    pytest -v
"""
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import dcf as engine
from dcf import market_data, sec_client

# ----------------------------------------------------------------------------- fixtures
YEARS = 10
END_DATES = [date(2015 + i, 9, 30) + timedelta(days=(i % 3) - 1) for i in range(YEARS + 1)]
BASE_REVENUE = 90e9
REVENUES = [BASE_REVENUE * (1.09 ** i) for i in range(YEARS + 1)]


def _duration(start, end, val, fy):
    return {"start": start.isoformat(), "end": end.isoformat(), "val": val,
            "form": "10-K", "filed": f"{fy}-11-01", "fy": fy, "fp": "FY"}


def _instant(end, val, fy):
    return {"end": end.isoformat(), "val": val, "form": "10-K",
            "filed": f"{fy}-11-01", "fy": fy, "fp": "FY"}


def _build_facts():
    """A realistic 10-K history, including quarterly noise the extractor must reject."""
    dur, inst = {}, {}
    for i in range(1, YEARS + 1):
        end, start = END_DATES[i], END_DATES[i - 1] + timedelta(days=1)
        rev, fy = REVENUES[i], END_DATES[i].year

        flows = {
            "RevenueFromContractWithCustomerExcludingAssessedTax": rev,
            "CostOfGoodsAndServicesSold": rev * 0.58,
            "GrossProfit": rev * 0.42,
            "OperatingIncomeLoss": rev * (0.24 + 0.004 * i),
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": rev * 0.23,
            "IncomeTaxExpenseBenefit": rev * 0.23 * 0.19,
            "NetIncomeLoss": rev * 0.23 * 0.81,
            "InterestExpense": 1.1e9,
            "DepreciationDepletionAndAmortization": rev * 0.045,
            "PaymentsToAcquirePropertyPlantAndEquipment": rev * 0.062,
            "NetCashProvidedByUsedInOperatingActivities": rev * 0.27,
        }
        for tag, val in flows.items():
            dur.setdefault(tag, []).append(_duration(start, end, val, fy))

        # A quarterly fact that must be filtered out by the 300-400 day duration gate
        dur["RevenueFromContractWithCustomerExcludingAssessedTax"].append(
            _duration(end - timedelta(days=90), end, rev * 0.26, fy))
        # A stale restatement that must lose to the later filing
        dur["OperatingIncomeLoss"].append(
            {"start": start.isoformat(), "end": end.isoformat(), "val": rev * 0.05,
             "form": "10-K", "filed": f"{fy - 1}-11-01"})

        balances = {
            "CashAndCashEquivalentsAtCarryingValue": 22e9,
            "ShortTermInvestments": 55e9,
            "AccountsReceivableNetCurrent": rev * 0.16,
            "InventoryNet": rev * 0.03,
            "OtherAssetsCurrent": rev * 0.02,
            "AccountsPayableCurrent": rev * 0.11,
            "AccruedLiabilitiesCurrent": rev * 0.05,
            "OtherLiabilitiesCurrent": rev * 0.02,
            "AssetsCurrent": rev * 0.75,
            "LiabilitiesCurrent": rev * 0.45,
            "LongTermDebtNoncurrent": 42e9,
            "LongTermDebtCurrent": 6e9,
            "MinorityInterest": 0.4e9,
            "StockholdersEquity": rev * 0.55,
            "Assets": rev * 1.6,
            "PropertyPlantAndEquipmentNet": rev * 0.4,
        }
        for tag, val in balances.items():
            inst.setdefault(tag, []).append(_instant(end, val, fy))

        dur.setdefault("WeightedAverageNumberOfDilutedSharesOutstanding", []).append(
            _duration(start, end, 7.6e9 - i * 6e7, fy))

    facts = {"entityName": "Synthetic Test Corp",
             "facts": {"us-gaap": {t: {"units": {"USD": r}} for t, r in {**dur, **inst}.items()}}}
    facts["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
        "units": {"shares": dur["WeightedAverageNumberOfDilutedSharesOutstanding"]}}
    return facts


FACTS = _build_facts()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Patch every network call at the class level so no test touches the internet."""
    monkeypatch.setattr(sec_client.SECClient, "resolve_cik",
                        lambda self, ticker: (1234567, "Synthetic Test Corp"))
    monkeypatch.setattr(sec_client.SECClient, "company_facts", lambda self, cik: FACTS)
    monkeypatch.setattr(market_data, "fetch_risk_free_rate", lambda series=None: 0.0428)


@pytest.fixture
def valuation():
    return engine.run_valuation(
        "TEST", price_override=185.0, shares_override=7.0e9, beta_override=1.05,
        n_sims=20_000, verbose=False,
    )


# -------------------------------------------------------------------------- extraction
def test_extracts_ten_fiscal_years():
    history, report, meta = engine.fetch_financial_history("TEST", 10)
    assert len(history) == 10
    assert meta["cik"] == 1234567
    assert report.coverage == 1.0, f"missing required tags: {report.missing_required}"


def test_quarterly_facts_are_rejected():
    """A Q4 figure must never be mistaken for a full year."""
    history, _, _ = engine.fetch_financial_history("TEST", 10)
    assert history["revenue"].iloc[-1] == pytest.approx(REVENUES[YEARS], rel=1e-9)


def test_latest_filing_wins_over_restatement():
    history, _, _ = engine.fetch_financial_history("TEST", 10)
    expected = REVENUES[YEARS] * (0.24 + 0.004 * YEARS)
    assert history["ebit"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_driver_table_signs_and_ratios():
    history, report, _ = engine.fetch_financial_history("TEST", 10)
    drivers = engine.build_driver_table(history, report)
    assert (drivers["capex"] > 0).all(), "capex must be a positive outflow"
    assert drivers["ebit_margin"].between(0.20, 0.35).all()
    assert drivers["nwc"].iloc[-1] > 0, "operating NWC should be positive here"
    assert drivers["effective_tax_rate"].between(0.08, 0.35).all()


# ------------------------------------------------------------------------- dcf identity
def test_enterprise_value_is_pv_of_flows_plus_terminal(valuation):
    g = valuation.gordon
    assert g.enterprise_value == pytest.approx(g.pv_forecast + g.pv_terminal_value, abs=1.0)


def test_bridge_sums_to_equity_value(valuation):
    g = valuation.gordon
    components = sum(v for k, v in g.bridge.items() if k != "Equity value")
    assert g.equity_value == pytest.approx(components, abs=1.0)


def test_per_share_is_equity_over_diluted_shares(valuation):
    g = valuation.gordon
    assert g.value_per_share == pytest.approx(
        g.equity_value / valuation.market.shares_diluted, rel=1e-9)


def test_terminal_value_is_a_sane_share_of_ev(valuation):
    assert 0.4 < valuation.gordon.tv_share_of_ev < 0.9


def test_mid_year_convention_lifts_value(valuation):
    """Discounting at t-0.5 instead of t is worth roughly 3-6% at a normal WACC."""
    kwargs = dict(wacc=valuation.wacc_result.wacc, shares=valuation.market.shares_diluted)
    mid = engine.run_dcf(valuation.drivers,
                         valuation.assumptions.copy_with(mid_year_convention=True), **kwargs)
    end = engine.run_dcf(valuation.drivers,
                         valuation.assumptions.copy_with(mid_year_convention=False,
                                                         discount_tv_at_midyear=False), **kwargs)
    uplift = mid.value_per_share / end.value_per_share - 1
    assert 0.01 < uplift < 0.10


def test_growth_above_wacc_is_rejected(valuation):
    """A perpetuity with g >= WACC diverges; it must raise, not return a negative value."""
    with pytest.raises(ValueError, match="perpetuity"):
        engine.run_dcf(valuation.drivers,
                       valuation.assumptions.copy_with(terminal_growth=0.30),
                       valuation.wacc_result.wacc, valuation.market.shares_diluted)


def test_higher_wacc_lowers_value(valuation):
    kwargs = dict(a=valuation.assumptions, shares=valuation.market.shares_diluted)
    low = engine.run_dcf(valuation.drivers, kwargs["a"], 0.08, kwargs["shares"])
    high = engine.run_dcf(valuation.drivers, kwargs["a"], 0.11, kwargs["shares"])
    assert high.value_per_share < low.value_per_share


# -------------------------------------------------------------------------------- wacc
def test_wacc_sits_between_cost_of_debt_and_cost_of_equity(valuation):
    w = valuation.wacc_result
    assert w.cost_of_debt_aftertax < w.wacc <= w.cost_of_equity
    assert w.weight_equity + w.weight_debt == pytest.approx(1.0)


def test_beta_relevering_round_trips():
    unlevered = engine.unlever_beta(1.20, 0.45, 0.21)
    assert engine.relever_beta(unlevered, 0.45, 0.21) == pytest.approx(1.20)


# ------------------------------------------------------------------------ monte carlo
def test_monte_carlo_is_reproducible(valuation):
    kwargs = dict(
        drivers=valuation.drivers, a=valuation.assumptions, wacc=valuation.wacc_result.wacc,
        shares=valuation.market.shares_diluted,
        bridge_adjustment=valuation.gordon.equity_value - valuation.gordon.enterprise_value,
        n_sims=5_000,
    )
    first = engine.monte_carlo_dcf(seed=7, **kwargs)
    second = engine.monte_carlo_dcf(seed=7, **kwargs)
    assert first.median == pytest.approx(second.median)


def test_monte_carlo_percentiles_are_ordered(valuation):
    mc = valuation.monte_carlo
    assert mc.percentiles["10"] < mc.percentiles["50"] < mc.percentiles["90"]
    assert mc.n_valid > 0
    assert mc.n_discarded / (mc.n_valid + mc.n_discarded) < 0.05


def test_wacc_dominates_the_variance(valuation):
    """On a stable large cap, the discount rate should explain most of the spread."""
    attribution = engine.driver_attribution(valuation.monte_carlo)
    assert attribution.index[0] == "WACC"
    assert attribution["Rank correlation"].loc["WACC"] < 0


# ----------------------------------------------------------------------------- outputs
def test_sensitivity_grid_is_monotonic(valuation):
    grid = valuation.sensitivity.to_numpy(dtype=float)
    assert np.all(np.diff(grid[:, 0]) < 0), "value must fall as WACC rises"
    assert np.all(np.diff(grid[0, :]) > 0), "value must rise as terminal growth rises"


def test_scenarios_are_ordered(valuation):
    values = valuation.scenarios["Value per share"]
    assert values["Bear"] < values["Base"] < values["Bull"]


def test_diagnostics_run(valuation):
    assert not valuation.diagnostics.empty
    assert set(valuation.diagnostics["Status"]) <= {"PASS", "REVIEW"}


def test_dashboard_renders(valuation):
    import matplotlib.pyplot as plt
    fig = valuation.dashboard()
    assert len(fig.axes) >= 6
    plt.close(fig)


def test_excel_export(valuation, tmp_path):
    paths = engine.export_valuation(valuation, output_dir=str(tmp_path), save_charts=False)
    assert "excel" in paths
    import pandas as pd
    sheets = pd.ExcelFile(paths["excel"]).sheet_names
    for expected in ("Summary", "Historical drivers", "WACC", "Sensitivity", "XBRL tag audit"):
        assert expected in sheets


# ------------------------------------------------------------------- loss-maker regime
def _loss_maker_facts():
    """Minimal pre-profitability filer: 4 years, deep operating losses, heavy capex."""
    from datetime import date
    dur, ins = {}, {}
    revs = [0.5e9, 1.2e9, 1.6e9, 1.9e9]
    for k in range(4):
        end, fy = date(2021 + k, 12, 31), 2021 + k
        start = date(end.year, 1, 1)
        rev, ebit = revs[k], revs[k] * (-2.5 + 0.4 * k)
        def dur_f(v, start=start, end=end, fy=fy):
            return {"start": start.isoformat(), "end": end.isoformat(), "val": v,
                    "form": "10-K", "filed": f"{fy+1}-02-15", "fy": fy, "fp": "FY"}

        def inst_f(v, end=end, fy=fy):
            return {"end": end.isoformat(), "val": v, "form": "10-K",
                    "filed": f"{fy+1}-02-15", "fy": fy, "fp": "FY"}
        for t, v in {
            "RevenueFromContractWithCustomerExcludingAssessedTax": rev,
            "OperatingIncomeLoss": ebit,
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": ebit,
            "IncomeTaxExpenseBenefit": 1e6, "NetIncomeLoss": ebit,
            "InterestExpense": 60e6,
            "DepreciationDepletionAndAmortization": rev * 0.12,
            "PaymentsToAcquirePropertyPlantAndEquipment": rev * 0.7,
            "NetCashProvidedByUsedInOperatingActivities": ebit * 0.8,
        }.items():
            dur.setdefault(t, []).append(dur_f(v))
        for t, v in {
            "CashAndCashEquivalentsAtCarryingValue": 3e9, "AccountsReceivableNetCurrent": rev * 0.1,
            "InventoryNet": rev * 0.9, "AccountsPayableCurrent": rev * 0.3,
            "AssetsCurrent": 3e9 + rev, "LiabilitiesCurrent": rev * 0.5,
            "LongTermDebtNoncurrent": 2e9, "StockholdersEquity": 2e9, "Assets": 8e9,
        }.items():
            ins.setdefault(t, []).append(inst_f(v))
        dur.setdefault("WeightedAverageNumberOfDilutedSharesOutstanding", []).append(dur_f(2e9))
    f = {"entityName": "LossCo",
         "facts": {"us-gaap": {t: {"units": {"USD": r}} for t, r in {**dur, **ins}.items()}}}
    f["facts"]["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
        "units": {"shares": dur["WeightedAverageNumberOfDilutedSharesOutstanding"]}}
    return f


@pytest.fixture
def loss_maker(monkeypatch):
    facts = _loss_maker_facts()
    monkeypatch.setattr(sec_client.SECClient, "resolve_cik", lambda self, t: (77, "LossCo"))
    monkeypatch.setattr(sec_client.SECClient, "company_facts", lambda self, cik: facts)
    monkeypatch.setattr(market_data, "fetch_risk_free_rate", lambda series=None: 0.0428)


def test_no_tax_subsidy_on_losses(loss_maker):
    """NOPAT must equal EBIT when EBIT is negative — losses are not taxed into refunds."""
    val = engine.run_valuation("LOSS", price_override=3.0, shares_override=2e9,
                               beta_override=1.7, n_sims=2000, verbose=False)
    proj = val.gordon.projection
    ebit = proj.loc["EBIT"]
    taxes = proj.loc["Taxes on EBIT"]
    for e, t in zip(ebit, taxes):
        if e < 0:
            assert t == 0.0
        else:
            assert t == pytest.approx(-e * val.assumptions.tax_rate)


def test_loss_maker_flagged_unsuitable(loss_maker):
    val = engine.run_valuation("LOSS", price_override=3.0, shares_override=2e9,
                               beta_override=1.7, n_sims=2000, verbose=False)
    verdict, issues = val.suitability
    assert verdict == "unsuitable"
    assert issues, "unsuitable verdict must come with reasons"


def test_monte_carlo_survives_all_negative_paths(loss_maker):
    """The old code filtered per_share > 0 and crashed (or lied) on loss-makers."""
    val = engine.run_valuation("LOSS", price_override=3.0, shares_override=2e9,
                               beta_override=1.7, n_sims=10_000, verbose=False)
    mc = val.monte_carlo
    assert mc.n_valid > 9_000
    assert mc.share_negative > 0.90
    assert mc.median < 0
    assert mc.percentiles["5"] < mc.percentiles["95"]


def test_mature_company_is_suitable(valuation):
    verdict, issues = valuation.suitability
    assert verdict == "suitable"
    assert issues == []


def test_heatmap_renders_when_all_cells_below_price(valuation):
    """One-sided grids used to saturate solid red; must render with visible gradation."""
    import matplotlib.pyplot as plt
    grid = valuation.sensitivity
    ax = engine.plot_sensitivity_heatmap(grid, current_price=10_000.0)  # absurdly high
    assert ax is not None
    plt.close("all")


def test_monte_carlo_chart_handles_negative_distribution(loss_maker):
    import matplotlib.pyplot as plt
    val = engine.run_valuation("LOSS", price_override=3.0, shares_override=2e9,
                               beta_override=1.7, n_sims=5_000, verbose=False)
    ax = engine.plot_monte_carlo(val.monte_carlo, current_price=3.0)
    assert ax is not None
    plt.close("all")
