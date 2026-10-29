import ast
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data as D


def _zip(path, name, body):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, body)


FF = """This file was created by CMPT_ME_BEME_OP_INV_RETS_DAILY using the 2026 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,  -0.67,   0.00,  -0.33,   0.03,   0.13, 0.012
19630702,   0.79,  -0.27,   0.27,  -0.07,  -0.19, 0.012
19630703,   0.63,  -0.17,   0.19,  -0.12,  -0.08, 0.012

Copyright 2026 Kenneth R. French
"""
MOM = """Daily Momentum

,Mom
19630702,   0.37
19630703,   0.20
19630704,   0.10
"""


def test_parse_and_join(tmp_path):
    _zip(tmp_path / D.FILES["five_factors"], "ff.csv", FF)
    _zip(tmp_path / D.FILES["momentum"], "mom.csv", MOM)
    panel = D.build_factor_panel(tmp_path)
    assert list(panel.columns) == D.FACTOR_COLUMNS
    assert list(panel.index.strftime("%Y%m%d")) == ["19630702", "19630703"]
    assert panel.loc["1963-07-02", "MKT"] == pytest.approx(0.0079)
    assert panel.loc["1963-07-03", "MOM"] == pytest.approx(0.0020)


def test_sentinel_raises(tmp_path):
    _zip(tmp_path / D.FILES["five_factors"], "ff.csv", FF.replace("0.79", "-99.99"))
    _zip(tmp_path / D.FILES["momentum"], "mom.csv", MOM)
    with pytest.raises(ValueError):
        D.build_factor_panel(tmp_path)


def _good():
    idx = pd.date_range("2000-01-03", periods=50, freq="B")
    return pd.DataFrame(np.random.default_rng(0).standard_normal((50, 6)) * 0.01,
                        index=idx, columns=D.FACTOR_COLUMNS)


def test_validate_accepts_good():
    assert D.validate_panel(_good())["n_obs"] == 50


def test_validate_rejects_bad():
    df = _good()
    dup = pd.concat([df, df.iloc[[0]]]).sort_index()
    with pytest.raises(ValueError):
        D.validate_panel(dup)
    nan = df.copy(); nan.iloc[3, 2] = np.nan
    with pytest.raises(ValueError):
        D.validate_panel(nan)
    with pytest.raises(ValueError):
        D.validate_panel(df[["SMB"] + [c for c in df.columns if c != "SMB"]])


def test_verify_raw_snapshot_accepts_matching_checksums(tmp_path, monkeypatch):
    import hashlib

    def no_download(*args, **kwargs):
        pytest.fail("snapshot verification must not download archives")

    monkeypatch.setattr(D, "download_raw", no_download)
    checksums = {}
    for name in D.FILES.values():
        content = name.encode()
        (tmp_path / name).write_bytes(content)
        checksums[name] = hashlib.sha256(content).hexdigest()
    (tmp_path / "CHECKSUMS.txt").write_text(
        "downloaded: 2026-10-21T11:31:35+00:00\n\n"
        + "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
    )

    assert D.verify_raw_snapshot(tmp_path) == checksums
    monkeypatch.chdir(tmp_path)
    assert D.verify_raw_snapshot(".") == checksums


@pytest.mark.parametrize("missing_name", D.FILES.values())
def test_verify_raw_snapshot_rejects_missing_checksum_entry(tmp_path, missing_name):
    import hashlib

    for name in D.FILES.values():
        (tmp_path / name).write_bytes(b"archive")
    digest = hashlib.sha256(b"archive").hexdigest()
    (tmp_path / "CHECKSUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name in D.FILES.values() if name != missing_name)
    )

    with pytest.raises(ValueError, match="[Mm]issing.*" + missing_name):
        D.verify_raw_snapshot(tmp_path)


@pytest.mark.parametrize("mismatched_name", D.FILES.values())
def test_verify_raw_snapshot_rejects_mismatch(tmp_path, mismatched_name):
    import hashlib

    digest = hashlib.sha256(b"archive").hexdigest()
    for name in D.FILES.values():
        (tmp_path / name).write_bytes(b"archive")
    (tmp_path / "CHECKSUMS.txt").write_text(
        "".join(
            f"{'0' * 64 if name == mismatched_name else digest}  {name}\n"
            for name in D.FILES.values()
        )
    )

    with pytest.raises(ValueError, match="[Mm]ismatch.*" + mismatched_name):
        D.verify_raw_snapshot(tmp_path)


def test_verify_raw_snapshot_rejects_missing_manifest(tmp_path):
    with pytest.raises(ValueError, match=r"[Mm]issing.*CHECKSUMS\.txt"):
        D.verify_raw_snapshot(tmp_path)


def test_verify_raw_snapshot_rejects_empty_manifest(tmp_path):
    (tmp_path / "CHECKSUMS.txt").write_text("")
    with pytest.raises(ValueError, match="[Mm]issing.*" + D.FILES["five_factors"]):
        D.verify_raw_snapshot(tmp_path)


@pytest.mark.parametrize("missing_name", D.FILES.values())
def test_verify_raw_snapshot_rejects_missing_archive(tmp_path, missing_name):
    import hashlib

    digest = hashlib.sha256(b"archive").hexdigest()
    for name in D.FILES.values():
        if name != missing_name:
            (tmp_path / name).write_bytes(b"archive")
    (tmp_path / "CHECKSUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name in D.FILES.values())
    )
    with pytest.raises(ValueError, match="[Mm]issing.*" + missing_name):
        D.verify_raw_snapshot(tmp_path)


def _raw_snapshot(raw_dir):
    raw_dir.mkdir()
    _zip(raw_dir / D.FILES["five_factors"], "ff.csv", FF)
    _zip(raw_dir / D.FILES["momentum"], "mom.csv", MOM)
    (raw_dir / "CHECKSUMS.txt").write_text(
        "".join(
            f"{hashlib.sha256((raw_dir / name).read_bytes()).hexdigest()}  {name}\n"
            for name in D.FILES.values()
        )
    )


def _notebook_raw_inputs():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/01_data_exploration.py"
    tree = ast.parse(notebook.read_text())
    cell = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
        and any(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
                and item.id == "actual_checksums" for item in ast.walk(node))
    )
    cell.decorator_list = []
    namespace = {}
    # Execute the real, trusted input cell without starting a notebook server.
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(notebook), "exec"), namespace)  # noqa: S102
    return namespace[cell.name]


@pytest.mark.parametrize("case, name", [
    ("missing_manifest", None),
    ("missing_manifest_no_archives", None),
    *[("missing_archive", name) for name in D.FILES.values()],
    *[("mismatch", name) for name in D.FILES.values()],
    ("valid", None),
])
def test_data_notebook_preserves_existing_snapshot(tmp_path, monkeypatch, case, name):
    (tmp_path / "data").mkdir()
    raw_dir = tmp_path / "data/raw"
    _raw_snapshot(raw_dir)
    if case.startswith("missing_manifest"):
        (raw_dir / "CHECKSUMS.txt").unlink()
        if case == "missing_manifest_no_archives":
            for archive in D.FILES.values():
                (raw_dir / archive).unlink()
    elif case == "missing_archive":
        (raw_dir / name).unlink()
    elif case == "mismatch":
        (raw_dir / name).write_bytes(b"corrupt archive")
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_dir.iterdir()}
    monkeypatch.chdir(tmp_path)

    def no_download(*args, **kwargs):
        pytest.fail("existing snapshots or missing manifests must not trigger downloads")

    def run():
        return _notebook_raw_inputs()(
            FILES=D.FILES, Path=Path, download_raw=no_download,
            pd=pd, verify_raw_snapshot=D.verify_raw_snapshot,
        )

    if case == "valid":
        checksums, _, inventory, _ = run()
        assert set(checksums) == set(D.FILES.values())
        assert inventory.shape == (2, 3)
    else:
        message = {
            "missing_manifest": "Missing checksum manifest:",
            "missing_manifest_no_archives": "Missing checksum manifest:",
            "missing_archive": "Missing raw archive:",
            "mismatch": "Checksum mismatch for",
        }[case]
        with pytest.raises(ValueError, match=message):
            run()
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_dir.iterdir()} == before


def _fetch(raw_dir, processed_dir):
    return subprocess.run(
        [
            sys.executable, "-m", "scripts.fetch_data",
            "--raw-dir", str(raw_dir), "--processed-dir", str(processed_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_fetch_script_reuses_snapshot_and_writes_processed_files(tmp_path):
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    _raw_snapshot(raw_dir)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_dir.iterdir()}

    result = _fetch(raw_dir, processed_dir)

    assert result.returncode == 0, result.stderr
    assert {path.name for path in processed_dir.iterdir()} == {"factors_daily.csv", "data_summary.json"}
    panel = D.load_factors(processed_dir / "factors_daily.csv")
    assert list(panel.index.strftime("%Y%m%d")) == ["19630702", "19630703"]
    assert panel.loc["1963-07-02", "MKT"] == pytest.approx(0.0079)
    assert panel.loc["1963-07-03", "MOM"] == pytest.approx(0.0020)
    summary = json.loads((processed_dir / "data_summary.json").read_text())
    assert summary["n_obs"] == 2
    assert summary["start"] == "1963-07-02"
    assert summary["end"] == "1963-07-03"
    assert summary["max_abs_return"]["MKT"] == pytest.approx(0.0079)
    assert json.loads(result.stdout) == summary
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_dir.iterdir()} == before


@pytest.mark.parametrize("name", D.FILES.values())
def test_fetch_script_rejects_checksum_mismatch_without_writing(tmp_path, name):
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    _raw_snapshot(raw_dir)
    (raw_dir / name).write_bytes(b"corrupt archive")
    processed_dir.mkdir()
    for output in ("factors_daily.csv", "data_summary.json"):
        (processed_dir / output).write_text("existing output")

    result = _fetch(raw_dir, processed_dir)

    assert result.returncode != 0
    assert f"Checksum mismatch for {name}" in result.stderr
    assert "expected" in result.stderr and "got" in result.stderr
    assert "Traceback" not in result.stderr
    assert (raw_dir / name).read_bytes() == b"corrupt archive"
    assert all(path.read_text() == "existing output" for path in processed_dir.iterdir())


def test_fetch_script_downloads_into_empty_snapshot_directory(tmp_path, monkeypatch, capsys):
    import requests

    from scripts.fetch_data import main

    source_dir, raw_dir = tmp_path / "source", tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _raw_snapshot(source_dir)
    raw_dir.mkdir()
    manifest = (source_dir / "CHECKSUMS.txt").read_bytes()
    (raw_dir / "CHECKSUMS.txt").write_bytes(manifest)

    class ArchiveResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    def get_archive(url, timeout):
        assert timeout == 60
        assert url.startswith(D.BASE_URL)
        return ArchiveResponse((source_dir / url.removeprefix(D.BASE_URL)).read_bytes())

    monkeypatch.setattr(requests, "get", get_archive)
    assert main(["--raw-dir", str(raw_dir), "--processed-dir", str(processed_dir)]) == 0
    panel = D.load_factors(processed_dir / "factors_daily.csv")
    assert panel.shape == (2, 6)
    assert panel.loc["1963-07-02", "MKT"] == pytest.approx(0.0079)
    assert json.loads((processed_dir / "data_summary.json").read_text())["n_obs"] == 2
    assert json.loads(capsys.readouterr().out)["n_obs"] == 2
    assert (raw_dir / "CHECKSUMS.txt").read_bytes() == manifest
    D.verify_raw_snapshot(raw_dir)


def test_fetch_script_requires_manifest_before_downloading(tmp_path, monkeypatch):
    from scripts.fetch_data import main

    def no_download(*args, **kwargs):
        pytest.fail("a missing manifest must be rejected before downloading")

    monkeypatch.setattr(D, "download_raw", no_download)
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    with pytest.raises(ValueError, match=r"Missing checksum manifest:.*CHECKSUMS\.txt"):
        main(["--raw-dir", str(raw_dir), "--processed-dir", str(processed_dir)])
    assert not raw_dir.exists()
    assert not processed_dir.exists()


def test_fetch_script_preserves_incomplete_snapshot(tmp_path, monkeypatch):
    from scripts.fetch_data import main

    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    _raw_snapshot(raw_dir)
    (raw_dir / D.FILES["momentum"]).unlink()
    before = {path.name: path.read_bytes() for path in raw_dir.iterdir()}

    def no_download(*args, **kwargs):
        pytest.fail("an existing snapshot must not be overwritten by a download")

    monkeypatch.setattr(D, "download_raw", no_download)
    with pytest.raises(ValueError, match="Missing raw archive:"):
        main(["--raw-dir", str(raw_dir), "--processed-dir", str(processed_dir)])
    assert {path.name: path.read_bytes() for path in raw_dir.iterdir()} == before
    assert not processed_dir.exists()


def test_fetch_script_validates_panel_before_writing(tmp_path):
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    _raw_snapshot(raw_dir)
    _zip(raw_dir / D.FILES["five_factors"], "ff.csv", FF.replace("0.79", "-99.99"))
    (raw_dir / "CHECKSUMS.txt").write_text(
        "".join(
            f"{hashlib.sha256((raw_dir / name).read_bytes()).hexdigest()}  {name}\n"
            for name in D.FILES.values()
        )
    )

    result = _fetch(raw_dir, processed_dir)

    assert result.returncode != 0
    assert "missing-value sentinel" in result.stderr
    assert not processed_dir.exists()
