"""
Task 4 - Introduction to Deep Learning Concepts (Prototype)

Trains a neural network (scikit-learn's MLPClassifier) on the same data, the
same temporal split and the same metrics as Task 3, then compares ML and DL on
training time, detection accuracy and practical feasibility.

Two architectures are trained so the comparison is not a single lucky
configuration:
    MLP-small  one hidden layer  (32,)
    MLP-deep   three hidden layers (128, 64, 32)

Requires task3_ml.py to have been run first (it reads its results table).

Run:  python src/task3_ml.py && python src/task4_dl.py
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from common import (RANDOM_STATE, TABLES, TARGET, Timer, build_preprocessor,
                    evaluate, feature_sets, load_data, print_report,
                    save_figure, save_table, temporal_split)

ML_RESULTS = TABLES / "task3_ml_results.csv"


def header(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def make_networks() -> dict:
    """
    Two MLPs of very different capacity.

    early_stopping holds out 10% of the training data and stops when validation
    score stops improving, so the runtime comparison against Task 3 reflects
    time to a usable model rather than time to burn through max_iter.
    """
    common = dict(max_iter=100, random_state=RANDOM_STATE, early_stopping=True,
                  n_iter_no_change=8, validation_fraction=0.10)
    return {
        "MLP-small": MLPClassifier(hidden_layer_sizes=(32,), **common),
        "MLP-deep": MLPClassifier(hidden_layer_sizes=(128, 64, 32), **common),
    }


def main() -> None:
    header("1. LOAD AND SPLIT  (identical to Task 3)")
    df = load_data()
    train, test, _ = temporal_split(df)
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    print(f"  train {len(train):,}   test {len(test):,}")
    print("  Same split, same preprocessing, same metrics as Task 3 - otherwise")
    print("  the ML and DL numbers would not be comparable.")

    cols = feature_sets()["honest"]
    feature_cols = cols["numeric"] + cols["boolean"] + cols["categorical"]
    print(f"\n  Using the HONEST feature set ({len(feature_cols)} columns).")
    print("  Task 3 showed the port columns turn the problem into a lookup; a")
    print("  neural network trained on them would only memorise the same table.")

    X_train, X_test = train[feature_cols], test[feature_cols]

    header("2. TRAINING THE NEURAL NETWORKS")
    results, histories, fitted = [], {}, {}
    for name, net in make_networks().items():
        pipe = Pipeline([("prep", build_preprocessor(**cols)), ("model", net)])
        with warnings.catch_warnings():
            # A converged early stop still raises ConvergenceWarning; that is
            # expected here and not a problem worth printing.
            warnings.simplefilter("ignore", ConvergenceWarning)
            with Timer() as t:
                pipe.fit(X_train, y_train)

        model = pipe.named_steps["model"]
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        row = evaluate(f"{name} (honest)", y_test, y_pred, y_prob, t.seconds)
        row["algorithm"] = name
        row["config"] = "honest"
        row["family"] = "Deep Learning"
        row["epochs_run"] = int(model.n_iter_)
        row["parameters"] = int(sum(c.size for c in model.coefs_)
                                + sum(b.size for b in model.intercepts_))
        results.append(row)
        histories[name] = model.loss_curve_
        fitted[name] = pipe

        print_report(row)
        print(f"  epochs run         : {row['epochs_run']} "
              f"(early stopping, cap 100)")
        print(f"  trainable params   : {row['parameters']:,}")

    header("3. ML vs DL COMPARISON")
    if not ML_RESULTS.exists():
        raise SystemExit(
            f"Missing {ML_RESULTS}.\nRun `python src/task3_ml.py` first - Task 4 "
            "compares against its results."
        )
    ml = pd.read_csv(ML_RESULTS)
    ml = ml[ml.config == "honest"].copy()
    ml["family"] = "Machine Learning"

    dl = pd.DataFrame(results)
    combined = pd.concat([ml, dl], ignore_index=True)

    view = combined[["family", "algorithm", "accuracy", "precision_malicious",
                     "recall_malicious", "f1_malicious", "pr_auc_malicious",
                     "false_positives", "train_seconds"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fastest_ml = ml.loc[ml.train_seconds.idxmin()]
    best_dl = dl.loc[dl.pr_auc_malicious.idxmax()]
    slowdown = best_dl.train_seconds / fastest_ml.train_seconds
    acc_gain = (best_dl.accuracy - ml.accuracy.max()) * 100

    header("4. ANALYSIS")
    print("  Training time")
    print(f"    fastest ML ({fastest_ml.algorithm}): {fastest_ml.train_seconds:.3f} s")
    print(f"    best DL    ({best_dl.algorithm}): {best_dl.train_seconds:.3f} s")
    print(f"    the network costs {slowdown:.1f}x the training time")

    print("\n  Detection accuracy")
    print(f"    best ML accuracy : {ml.accuracy.max()*100:.2f}%")
    print(f"    best DL accuracy : {dl.accuracy.max()*100:.2f}%")
    print(f"    difference       : {acc_gain:+.2f} percentage points")
    print(f"    best ML PR-AUC   : {ml.pr_auc_malicious.max():.4f}")
    print(f"    best DL PR-AUC   : {dl.pr_auc_malicious.max():.4f}")

    print("\n  Practical feasibility")
    print(f"    The deep network carries {int(best_dl.parameters):,} trainable")
    print("    parameters against a depth-8 decision tree that can be printed on")
    print("    one page and read by an analyst. In a SOC, an alert that cannot be")
    print("    explained is an alert that gets ignored, so the interpretable model")
    print("    wins unless the network buys a real accuracy margin.")

    # Does the network escape the one-feature collapse Task 3 found?
    header("5. DOES THE NETWORK ESCAPE THE COLLAPSE?")
    tcp_rule = (test["proto"] == "tcp").to_numpy().astype(int)
    collapse = []
    for name, pipe in fitted.items():          # reuse the networks trained above
        agreement = float((pipe.predict(X_test) == tcp_rule).mean())
        collapse.append({"model": name, "rule": "proto == tcp",
                         "agreement": agreement,
                         "identical": bool(agreement == 1.0)})
        flag = "   <-- IDENTICAL" if agreement == 1.0 else ""
        print(f"    {name:<12} vs proto == tcp : {agreement*100:7.3f}%{flag}")

    if all(c["identical"] for c in collapse):
        print("\n  Both networks reproduce the same single-feature rule as the ML")
        print("  models. Extra capacity did not help, because the ceiling is not")
        print("  the model - it is the data. Benign traffic here is UDP that got")
        print("  answered and malicious traffic is TCP SYNs that did not, so there")
        print("  is no richer pattern left for a deeper network to find.")
    else:
        print("\n  The networks differ from the single-feature rule, so they are")
        print("  using more than protocol alone - compare the false-positive")
        print("  counts above to see whether that difference is worth anything.")

    header("6. FIGURES AND TABLES")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for name, curve in histories.items():
        ax1.plot(range(1, len(curve) + 1), curve, lw=1.6, label=name)
    ax1.set_title("Training loss per epoch")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("log-loss")
    ax1.legend()

    order = view.sort_values("train_seconds")
    colours = ["#4c72b0" if f == "Machine Learning" else "#c44e52"
               for f in order.family]
    ax2.barh(order.algorithm, order.train_seconds, color=colours)
    ax2.set_title("Training time (blue = ML, red = DL)")
    ax2.set_xlabel("seconds")
    for i, (t, a) in enumerate(zip(order.train_seconds, order.accuracy)):
        ax2.text(t, i, f"  {t:.2f}s / {a*100:.2f}%", va="center", fontsize=8)
    ax2.margins(x=0.28)
    fig.suptitle("Task 4 - ML vs DL: cost against benefit", fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "fig6_ml_vs_dl.png")
    plt.close(fig)

    save_table(combined, "task4_ml_vs_dl.csv")
    save_table(pd.DataFrame(collapse), "task4_dl_collapse_check.csv")
    print("\nTask 4 complete.")


if __name__ == "__main__":
    main()
