"""
Shared loading, feature definitions, splitting and evaluation.

Imported by task2_eda.py, task3_ml.py and task4_dl.py so that all three scripts
use exactly the same data, the same split and the same metrics. If the ML and
DL scripts each built their own split, their scores would not be comparable.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "iot23_sample.csv.gz"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70

# --------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------
# Flow measurements. All are counts or byte totals from the Zeek conn.log.
NUMERIC = ["duration", "orig_bytes", "resp_bytes",
           "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]

# Explicit missingness flags. duration/orig_bytes/resp_bytes are NaN when
# nobody answered the connection, which is a different event from "answered
# with zero bytes". These two columns carry that distinction so it survives
# imputation -- see the note in build_preprocessor().
BOOLEAN = ["no_response", "never_established"]

# Protocol-level categories describing HOW the connection behaved.
CATEGORICAL = ["proto", "conn_state", "history"]

# Identifier-like columns that leak the label. See docs in task2_eda.py: the
# published ground truth was derived from the destination port, so including
# these lets a model recover the labelling rule instead of learning behaviour.
LEAKY_NUMERIC = ["id.resp_p", "id.orig_p"]
LEAKY_CATEGORICAL = ["service"]

TARGET = "is_malicious"


def feature_sets() -> dict[str, dict[str, list[str]]]:
    """The two feature configurations compared throughout the project."""
    return {
        "with_leak": {
            "numeric": NUMERIC + LEAKY_NUMERIC,
            "boolean": BOOLEAN,
            "categorical": CATEGORICAL + LEAKY_CATEGORICAL,
        },
        "honest": {
            "numeric": NUMERIC,
            "boolean": BOOLEAN,
            "categorical": CATEGORICAL,
        },
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_data(path: Path = DATA) -> pd.DataFrame:
    """Load the sample and restore dtypes lost in the CSV round-trip."""
    if not path.exists():
        raise SystemExit(
            f"Dataset not found: {path}\n"
            "It should be committed with the repository. If it is missing, "
            "regenerate it with:  python src/make_sample.py"
        )
    df = pd.read_csv(path, parse_dates=["ts"], low_memory=False)

    # CSV writes booleans as the strings "True"/"False".
    for col in BOOLEAN:
        if df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False}).astype(bool)

    # Ports are integers, but pandas reads them as float when NaN is present.
    for col in LEAKY_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # A missing `history` or `service` is itself meaningful (ICMP flows carry no
    # TCP history, and Zeek only names a service it recognises), so it becomes
    # its own category rather than being dropped.
    for col in CATEGORICAL + LEAKY_CATEGORICAL:
        df[col] = df[col].fillna("(none)").astype(str)

    df = df.sort_values("ts").reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    """
    Split on time, not at random.

    The flows are time-ordered and adjacent rows are near-duplicates of the same
    scan burst. A random split puts near-identical rows on both sides, so the
    test score measures memorisation rather than generalisation. Training on the
    earlier period and testing on the later one is also what a deployed detector
    actually faces: it must classify traffic it has not seen yet.
    """
    cut = int(len(df) * train_fraction)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    return train, test, cut


def build_preprocessor(numeric: list[str], boolean: list[str],
                       categorical: list[str]) -> ColumnTransformer:
    """
    Preprocessing required by Task 3: missing values, encoding, normalisation.

    On imputation: scikit-learn's LogisticRegression and MLPClassifier cannot
    accept NaN, so the missing duration/byte values must be filled. Filling them
    with the median alone would destroy real information -- 'nobody answered'
    would become indistinguishable from 'answered with a median-sized reply'.
    `add_indicator=True` therefore appends a binary was-missing column for each
    imputed feature, so the model still sees which rows were originally empty.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])

    # min_frequency collapses rare `history` strings into a single "infrequent"
    # level. Without it the 100+ rare values become 100+ near-empty columns that
    # the models can use to memorise individual flows.
    categorical_pipe = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False
    )

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("bool", "passthrough", boolean),
        ("cat", categorical_pipe, categorical),
    ])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(name: str, y_true, y_pred, y_prob, train_seconds: float) -> dict:
    """
    Per-class precision/recall/F1 plus absolute confusion counts.

    Accuracy is recorded but deliberately reported last. On this dataset a
    single-feature rule already reaches ~96%, so accuracy alone cannot tell you
    whether a model learned anything. The false-positive count is the number
    that matters operationally: every one is a legitimate flow that would raise
    an alert.
    """
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "train_seconds": round(train_seconds, 3),
        "precision_benign": p[0], "recall_benign": r[0], "f1_benign": f[0],
        "precision_malicious": p[1], "recall_malicious": r[1], "f1_malicious": f[1],
        "pr_auc_malicious": average_precision_score(y_true, y_prob),
        "macro_f1": f.mean(),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "false_positives": int(fp),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "support_benign": int(s[0]), "support_malicious": int(s[1]),
    }


def print_report(row: dict) -> None:
    """Console summary matching the columns stored in the results table."""
    print(f"\n  --- {row['model']} ---")
    print(f"  {'class':<12}{'precision':>11}{'recall':>10}{'F1':>10}{'support':>10}")
    for cls in ("benign", "malicious"):
        print(f"  {cls:<12}{row[f'precision_{cls}']:>11.4f}"
              f"{row[f'recall_{cls}']:>10.4f}{row[f'f1_{cls}']:>10.4f}"
              f"{row[f'support_{cls}']:>10,}")
    print(f"  PR-AUC (malicious) : {row['pr_auc_malicious']:.4f}")
    print(f"  confusion          : TP {row['TP']:,}  FP {row['FP']:,}  "
          f"FN {row['FN']:,}  TN {row['TN']:,}")
    print(f"  FALSE POSITIVES    : {row['false_positives']:,} "
          f"(rate {row['false_positive_rate']*100:.2f}%)")
    print(f"  training time      : {row['train_seconds']:.3f} s")
    print(f"  accuracy           : {row['accuracy']*100:.2f}%   <- reported last, on purpose")


class Timer:
    """Context manager used to record training time for the Task 4 comparison."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        return False


def save_table(df: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def save_figure(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  wrote {path.relative_to(ROOT)}")
    return path
