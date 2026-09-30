"""
Build the committed dataset sample from the full IoT-23 capture.

The full capture (CTU-IoT-Malware-Capture-1-1) is 1,008,748 flows / 138 MB,
which is too large to commit. This script takes a systematic sample and writes
`data/iot23_sample.csv.gz`.

Systematic (every Nth row), not random, for two reasons:
  1. The flows are time-ordered and the project uses a TEMPORAL train/test
     split. A random sample would still preserve order after sorting, but a
     systematic stride also thins out the dense scan bursts evenly across the
     whole 4-day capture instead of clustering draws.
  2. Adjacent flows are near-duplicates of the same scan burst. Taking every
     Nth row reduces how many near-identical rows land next to each other.

You do NOT need to run this: `data/iot23_sample.csv.gz` is committed. Re-run it
only to regenerate the sample or change its size.

Usage:
    python src/make_sample.py [--source PATH] [--rows N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path(
    r"C:\Users\Rudraraj Radhwani\projects\iot-botnet-detection\data\clean.parquet"
)
OUT = Path(__file__).resolve().parents[1] / "data" / "iot23_sample.csv.gz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="cleaned full-capture parquet/pickle")
    ap.add_argument("--rows", type=int, default=80_000,
                    help="approximate number of rows to keep")
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(
            f"Source capture not found: {args.source}\n"
            "The sample at data/iot23_sample.csv.gz is already committed, so you "
            "only need this script to regenerate it. Pass --source to point at "
            "your own copy of the cleaned IoT-23 capture."
        )

    print(f"reading {args.source}")
    df = (pd.read_parquet(args.source) if args.source.suffix == ".parquet"
          else pd.read_pickle(args.source))
    print(f"  full capture: {len(df):,} rows x {df.shape[1]} cols")

    df = df.sort_values("ts").reset_index(drop=True)

    step = max(1, len(df) // args.rows)
    sample = df.iloc[::step].reset_index(drop=True)
    print(f"  systematic stride: every {step}th row -> {len(sample):,} rows")

    full_rate = df["is_malicious"].mean()
    samp_rate = sample["is_malicious"].mean()
    print(f"  malicious rate: full {full_rate*100:.2f}%  sample {samp_rate*100:.2f}%")
    assert abs(full_rate - samp_rate) < 0.02, "sample class balance drifted from the full capture"

    # detailed-label is the multi-class ground truth; keep it so the EDA can
    # show that 'Malicious' here is almost entirely one attack type.
    print("  detailed-label in sample:")
    for k, v in sample["detailed-label"].value_counts().items():
        print(f"    {k:<28} {v:,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False, compression="gzip")
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT}  ({size_mb:.2f} MB, {len(sample):,} rows)")


if __name__ == "__main__":
    main()
