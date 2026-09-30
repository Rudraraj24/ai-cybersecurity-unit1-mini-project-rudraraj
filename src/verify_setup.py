"""
Task 1 - environment validation.

Confirms every required package imports, prints its version, and checks that
the dataset loads. Run this immediately after creating the virtual environment
and installing requirements.txt.

Run:  python src/verify_setup.py
"""
from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "jupyter"]

# Import name -> the name used on PyPI, where they differ.
PYPI_NAME = {"sklearn": "scikit-learn"}


def main() -> int:
    print("=" * 62)
    print("Environment check - AI in Cyber Security, Unit 1 mini project")
    print("=" * 62)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"Platform    : {platform.system()} {platform.release()}")
    print(f"Interpreter : {sys.executable}")

    if sys.prefix == sys.base_prefix:
        print("\nWARNING: not running inside a virtual environment.")
        print("Activate it first - see README.md, Task 1.")

    print("\nRequired packages:")
    missing = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "(no __version__)")
            print(f"  OK    {PYPI_NAME.get(name, name):<14} {version}")
        except ImportError:
            print(f"  FAIL  {PYPI_NAME.get(name, name):<14} not installed")
            missing.append(PYPI_NAME.get(name, name))

    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print("Install with:  pip install -r requirements.txt")
        return 1

    print("\nDataset:")
    try:
        from common import DATA, load_data
        df = load_data()
        print(f"  OK    {DATA.name}")
        print(f"        {len(df):,} rows x {df.shape[1]} columns")
        print(f"        capture window {df.ts.min()} -> {df.ts.max()}")
        print(f"        malicious {df.is_malicious.mean()*100:.2f}%")
    except SystemExit as exc:
        print(f"  FAIL  {exc}")
        return 1

    print("\nSetup is valid. Next:")
    print("  python src/task2_eda.py")
    print("  python src/task3_ml.py")
    print("  python src/task4_dl.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
