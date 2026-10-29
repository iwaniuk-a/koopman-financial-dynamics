import argparse
import json
from pathlib import Path

from src import data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)

    # Require the recorded snapshot identity before any network or output writes.
    manifest = args.raw_dir / "CHECKSUMS.txt"
    if not manifest.is_file():
        raise ValueError(f"Missing checksum manifest: {manifest}")
    raw_paths = [args.raw_dir / name for name in data.FILES.values()]
    # download_raw writes both archives, so never call it on a partial snapshot.
    if not any(path.exists() for path in raw_paths):
        data.download_raw(args.raw_dir)
    data.verify_raw_snapshot(args.raw_dir)

    panel = data.build_factor_panel(args.raw_dir)
    summary = json.dumps(data.validate_panel(panel), indent=2)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.processed_dir / "factors_daily.csv", index_label="Date")
    (args.processed_dir / "data_summary.json").write_text(summary + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        raise SystemExit(str(error)) from error
