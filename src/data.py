"""Download, parse and validate the Fama-French daily factor panel.

Units: decimal returns. Columns: MKT (= Mkt-RF), SMB, HML, RMW, CMA, MOM.
RF is dropped.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "five_factors": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum": "F-F_Momentum_Factor_daily_CSV.zip",
}


def verify_raw_snapshot(raw_dir="data/raw") -> dict[str, str]:
    """Verify both local archives against CHECKSUMS.txt without downloading.

    Return full SHA-256 digests keyed by archive filename. Missing manifests,
    checksum entries, or archives and mismatched digests raise ValueError.
    Metadata lines (such as the download timestamp) are ignored.
    """
    raw_dir = Path(raw_dir)
    checksum_path = raw_dir / "CHECKSUMS.txt"
    if not checksum_path.is_file():
        raise ValueError(f"Missing checksum manifest: {checksum_path}")

    expected = {}
    for line in checksum_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            expected[parts[1]] = parts[0]

    verified = {}
    for name in FILES.values():
        if name not in expected:
            raise ValueError(f"Missing checksum entry for {name} in {checksum_path}")
        archive_path = raw_dir / name
        if not archive_path.is_file():
            raise ValueError(f"Missing raw archive: {archive_path}")
        actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual != expected[name]:
            raise ValueError(
                f"Checksum mismatch for {name}: expected {expected[name]}, got {actual}"
            )
        verified[name] = actual
    return verified


def download_raw(raw_dir="data/raw", timeout=60) -> list[Path]:     # Downloads the zip files from Dartmouth
    """Download the official archives to raw_dir."""
    import requests

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for fname in FILES.values():
        r = requests.get(BASE_URL + fname, timeout=timeout)
        r.raise_for_status()
        target = raw_dir / fname
        target.write_bytes(r.content)
        saved_paths.append(target)
    return saved_paths


def _daily_lines(zip_path) -> list[str]:        # Opens a raw zip file and filters out all the text headers/footers, keeping only valid daily data rows.
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith((".csv", ".txt")))
        text = zf.read(name).decode("utf-8-sig", errors="replace")
    out = []
    for line in text.splitlines():
        first = line.split(",")[0].strip()
        if len(first) == 8 and first.isdigit():
            out.append(line)
    return out


def _lines_to_frame(lines, columns) -> pd.DataFrame:        # Converts the filtered list of text strings into a clean Pandas DataFrame indexed by date.
    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        header=None,
        names=["Date"] + list(columns),
        skipinitialspace=True,
    )
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
    return df.set_index("Date")


def validate_panel(df: pd.DataFrame) -> dict:
    """Raise on any structural problem; return a summary dict. Never imputes."""
    if list(df.columns) != FACTOR_COLUMNS:
        raise ValueError(f"columns must be {FACTOR_COLUMNS}, got {list(df.columns)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not df.index.is_unique:
        raise ValueError("duplicate dates")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index not monotonic")
    arr = df.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values")
    if (np.abs(arr) >= 0.99).any():
        raise ValueError("|return| >= 99%: probably a missing-value sentinel")
    if (df.index.dayofweek >= 5).any():
        raise ValueError("weekend dates present")

    gaps = df.index.to_series().diff().dt.days.dropna()
    big = gaps[gaps > 7]
    return {
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "n_obs": len(df),
        "n_abs_gt_20pct": {c: int((df[c].abs() > 0.20).sum()) for c in df.columns},
        "max_abs_return": {c: float(df[c].abs().max()) for c in df.columns},
        "date_of_max_abs": {c: str(df[c].abs().idxmax().date()) for c in df.columns},
        "n_gaps_gt_7_days": len(big),
        "gap_end_dates": [str(d.date()) for d in big.index],
    }


def build_factor_panel(raw_dir="data/raw") -> pd.DataFrame:     # Combines the 5-factor file and the Momentum file into the final unified dataset.
    raw_dir = Path(raw_dir)
    ff = _lines_to_frame(
        _daily_lines(raw_dir / FILES["five_factors"]),
        ["MKT", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    mom = _lines_to_frame(_daily_lines(raw_dir / FILES["momentum"]), ["MOM"])

    for name, frame in (("five_factors", ff), ("momentum", mom)):
        if ((frame <= -99.0).any()).any():
            raise ValueError(f"{name}: missing-value sentinel (<= -99) present; handle explicitly")

    # The join must not silently drop dates inside the overlapping range.
    lost = ff.index[ff.index >= mom.index.min()].difference(mom.index)
    if len(lost) > 0:
        raise ValueError(f"{len(lost)} five-factor dates inside momentum range have no momentum row")

    panel = ff.drop(columns="RF").join(mom, how="inner")[FACTOR_COLUMNS] / 100.0
    validate_panel(panel)
    return panel


def load_factors(path="data/processed/factors_daily.csv") -> pd.DataFrame:
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    validate_panel(df)
    return df
