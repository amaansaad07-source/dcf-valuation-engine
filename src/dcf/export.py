"""Excel workbook and chart export."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from .config import log
from .drivers import DRIVER_VIEW
from .engine import Valuation


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
