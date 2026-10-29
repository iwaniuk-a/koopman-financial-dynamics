import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Koopman Spectral Analysis of Financial Factor Dynamics
    ## 01 — Data acquisition and validation

    We build the daily panel used by the later notebooks.
    The source archives report percentage returns \(R_t\), whereas the research panel stores
    decimal returns \(r_t = R_t / 100\).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    import sys
    from pathlib import Path
    project_root = Path.cwd()
    if project_root.name == "notebooks":
        project_root = project_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data import (
        FILES,
        build_factor_panel,
        download_raw,
        validate_panel,
        verify_raw_snapshot,
    )

    return (
        FILES,
        Path,
        build_factor_panel,
        download_raw,
        mo,
        pd,
        validate_panel,
        verify_raw_snapshot,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reproducible raw inputs

    Existing archives in `data/raw` are treated as an immutable snapshot.
    Require `CHECKSUMS.txt` before downloading, and download both official archives
    only when neither is present. Partial snapshots must be restored.
    `verify_raw_snapshot()` checks both archives before parsing and missing entries
    or mismatches stop execution.
    """)
    return


@app.cell
def _(FILES, Path, download_raw, pd, verify_raw_snapshot):
    raw_dir = Path("data/raw")
    raw_paths = [raw_dir / filename for filename in FILES.values()]

    if not (raw_dir / "CHECKSUMS.txt").is_file():
        raise ValueError(f"Missing checksum manifest: {raw_dir / 'CHECKSUMS.txt'}")
    if not any(path.exists() for path in raw_paths):
        raw_paths = download_raw(raw_dir=raw_dir, timeout=60)

    actual_checksums = verify_raw_snapshot(raw_dir)

    raw_inventory = pd.DataFrame(
        {
            "file": [path.name for path in raw_paths],
            "bytes": [path.stat().st_size for path in raw_paths],
            "sha256": [actual_checksums[path.name][:12] for path in raw_paths],
        }
    )
    raw_inventory
    return actual_checksums, raw_dir, raw_inventory, raw_paths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build and validate the factor panel

    `build_factor_panel()` parses the daily rows, joins the five-factor and momentum
    archives, drops the risk-free rate, converts percentage returns to decimals, and
    rejects missing-value sentinels or broken joins. `validate_panel()` then checks
    column order, dates, finiteness, return bounds, and long calendar gaps.
    """)
    return


@app.cell
def _(build_factor_panel, raw_dir, validate_panel):
    factors = build_factor_panel(raw_dir=raw_dir)
    summary = validate_panel(factors)
    return factors, summary


@app.cell
def _(pd, summary):
    validation_summary = pd.DataFrame(
        {
            "value": [
                summary["start"],
                summary["end"],
                summary["n_obs"],
                summary["n_gaps_gt_7_days"],
                ", ".join(summary["gap_end_dates"]) or "none",
            ]
        },
        index=[
            "start",
            "end",
            "observations",
            "gaps > 7 calendar days",
            "gap end dates",
        ],
    )
    validation_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Descriptive statistics

    For each date, the panel is the six-dimensional return vector
    \[
    x_t = (r^{MKT}_t, r^{SMB}_t, r^{HML}_t,
           r^{RMW}_t, r^{CMA}_t, r^{MOM}_t)^\top.
    \]
    """)
    return


@app.cell
def _(factors):
    descriptive_statistics = factors.describe().T.round(4)
    descriptive_statistics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpretation

    The factor means are close to zero, while their daily variation is much larger. Market and momentum are the most volatile factors, whereas RMW and CMA are more stable. The wide gap between the typical values and the extreme minimums and maximums also shows that the dataset contains occasional large shocks.

    This motivates studying both signed returns and squared returns. The former capture directional dynamics, while the latter capture volatility and the changing size of market movements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save the validated panel

    Saving the validated panel lets other notebooks load the same data without
    downloading or reparsing the raw archives.
    """)
    return


@app.cell
def _(Path, factors):
    processed_path = Path("data/processed/factors_daily.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(processed_path, index_label="Date")
    return


if __name__ == "__main__":
    app.run()
