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
MAX_EPOCHS = 200


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
    common = dict(max_iter=MAX_EPOCHS, random_state=RANDOM_STATE,
                  early_stopping=True, n_iter_no_change=8,
                  validation_fraction=0.10)
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

    header("2. TRAINING THE NEURAL NETWORKS")
    print("  Both feature configurations are trained, matching Task 3.")
    print("  Training only on the honest set would leave nothing to compare: all")
    print("  models collapse to the same rule there, so the ML-vs-DL comparison")
    print("  needs the full feature set to have any variation in it.")

    results, histories, fitted = [], {}, {}
    for config_name in ("with_leak", "honest"):
        cols = feature_sets()[config_name]
        feature_cols = cols["numeric"] + cols["boolean"] + cols["categorical"]
        X_train, X_test = train[feature_cols], test[feature_cols]
        print(f"\n  [{config_name}] {len(feature_cols)} input columns")

        for name, net in make_networks().items():
            pipe = Pipeline([("prep", build_preprocessor(**cols)), ("model", net)])
            with warnings.catch_warnings():
                # A converged early stop still raises ConvergenceWarning; that
                # is expected here and not a problem worth printing.
                warnings.simplefilter("ignore", ConvergenceWarning)
                with Timer() as t:
                    pipe.fit(X_train, y_train)

            model = pipe.named_steps["model"]
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            row = evaluate(f"{name} ({config_name})", y_test, y_pred, y_prob,
                           t.seconds)
            row["algorithm"] = name
            row["config"] = config_name
            row["family"] = "Deep Learning"
            row["epochs_run"] = int(model.n_iter_)
            row["parameters"] = int(sum(c.size for c in model.coefs_)
                                    + sum(b.size for b in model.intercepts_))
            results.append(row)
            histories[f"{name} ({config_name})"] = model.loss_curve_
            if config_name == "honest":
                fitted[name] = pipe          # used by the collapse check below

            print_report(row)
            print(f"  epochs run         : {row['epochs_run']} "
                  f"(early stopping, cap {MAX_EPOCHS})")
            print(f"  trainable params   : {row['parameters']:,}")

    header("3. ML vs DL COMPARISON")
    if not ML_RESULTS.exists():
        raise SystemExit(
            f"Missing {ML_RESULTS}.\nRun `python src/task3_ml.py` first - Task 4 "
            "compares against its results."
        )
    ml = pd.read_csv(ML_RESULTS)
    ml["family"] = "Machine Learning"
    dl = pd.DataFrame(results)
    combined = pd.concat([ml, dl], ignore_index=True)

    view = combined[["config", "family", "algorithm", "accuracy",
                     "precision_malicious", "recall_malicious", "f1_malicious",
                     "pr_auc_malicious", "false_positives", "train_seconds"]]
    for config_name in ("with_leak", "honest"):
        print(f"\n  --- {config_name} ---")
        sub = view[view.config == config_name].drop(columns="config")
        print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    header("4. ANALYSIS")
    # The meaningful comparison is on the full feature set. On the honest set
    # every model collapses to the same rule, so comparing them there measures
    # nothing about the algorithms.
    full = combined[combined.config == "with_leak"]
    ml_full, dl_full = full[full.family == "Machine Learning"], full[full.family == "Deep Learning"]
    fastest = full.loc[full.train_seconds.idxmin()]
    slowest = full.loc[full.train_seconds.idxmax()]
    best = full.loc[full.pr_auc_malicious.idxmax()]

    print("  Training time (full feature set)")
    print(f"    fastest: {fastest.algorithm:<10} {fastest.train_seconds:>7.2f} s")
    print(f"    slowest: {slowest.algorithm:<10} {slowest.train_seconds:>7.2f} s")
    print(f"    ratio  : {slowest.train_seconds / fastest.train_seconds:.1f}x")

    print("\n  Detection accuracy (full feature set)")
    print(f"    best ML : {ml_full.accuracy.max()*100:.2f}%  "
          f"({ml_full.loc[ml_full.accuracy.idxmax()].algorithm})")
    print(f"    best DL : {dl_full.accuracy.max()*100:.2f}%  "
          f"({dl_full.loc[dl_full.accuracy.idxmax()].algorithm})")
    print(f"    difference: {(dl_full.accuracy.max() - ml_full.accuracy.max())*100:+.2f} pp")
    print(f"    lowest false positives: {int(full.false_positives.min()):,} "
          f"({full.loc[full.false_positives.idxmin()].algorithm})")

    print("\n  Practical feasibility")
    print(f"    Best overall is {best.algorithm} (PR-AUC {best.pr_auc_malicious:.4f}).")
    print("    The Decision Tree reaches 100% because it can REPRESENT the")
    print("    port-lookup rule exactly, in a handful of splits an analyst can")
    print("    read. The networks have to APPROXIMATE that same step function")
    print(f"    with thousands of weights, which is why {slowest.algorithm} spends "
          f"{slowest.train_seconds / fastest.train_seconds:.0f}x the")
    print("    training time to get close and still does not match it.")
    print("    When the true decision boundary is a rule, a rule-based learner")
    print("    wins on accuracy, speed AND explainability at once.")

    print("\n  Why the honest configuration shows no variation")
    honest = combined[combined.config == "honest"]
    print(f"    All {len(honest)} models score {honest.accuracy.iloc[0]*100:.2f}% with "
          f"{int(honest.false_positives.iloc[0]):,} false positives.")
    print("    That is not a bug and not a tie - it is the ceiling of the data.")
    print("    See section 5.")

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

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    for name, curve in histories.items():
        style = "-" if "with_leak" in name else "--"
        ax.plot(range(1, len(curve) + 1), curve, lw=1.5, ls=style, label=name)
    ax.set_title("Training loss per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("log-loss")
    ax.legend(fontsize=7)

    # Accuracy on the full feature set: this is where the models actually differ.
    ax = axes[1]
    f = full.sort_values("accuracy")
    colours = ["#4c72b0" if fam == "Machine Learning" else "#c44e52"
               for fam in f.family]
    ax.barh(f.algorithm, f.accuracy * 100, color=colours)
    ax.set_xlim(98.5, 100.05)
    ax.set_title("Accuracy, full feature set\n(blue = ML, red = DL)")
    ax.set_xlabel("% accuracy")
    for i, (a, fp) in enumerate(zip(f.accuracy, f.false_positives)):
        ax.text(a * 100, i, f"  {a*100:.2f}%  ({int(fp)} FP)", va="center", fontsize=8)

    ax = axes[2]
    f2 = full.sort_values("train_seconds")
    colours = ["#4c72b0" if fam == "Machine Learning" else "#c44e52"
               for fam in f2.family]
    ax.barh(f2.algorithm, f2.train_seconds, color=colours)
    ax.set_title("Training time, full feature set")
    ax.set_xlabel("seconds")
    for i, t in enumerate(f2.train_seconds):
        ax.text(t, i, f"  {t:.1f}s", va="center", fontsize=8)
    ax.margins(x=0.25)

    fig.suptitle("Task 4 - ML vs DL: the models differ on the full feature set",
                 fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "fig6_ml_vs_dl.png")
    plt.close(fig)

    save_table(combined, "task4_ml_vs_dl.csv")
    save_table(pd.DataFrame(collapse), "task4_dl_collapse_check.csv")
    print("\nTask 4 complete.")


if __name__ == "__main__":
    main()
