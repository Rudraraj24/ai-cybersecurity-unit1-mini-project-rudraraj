"""
Task 3 - Basic Machine Learning-Based Threat Classification

Preprocesses the dataset (missing values, categorical encoding, numerical
normalisation) and trains two supervised baselines - Logistic Regression and a
Decision Tree - to classify each flow as normal or suspicious.

Each model is trained twice:
  with_leak  includes the destination/source port and Zeek's `service` field
  honest     excludes them

Task 2 showed those three columns encode the rule the dataset's labels were
generated from. Training both makes the size of that effect measurable instead
of assumed, and shows why the near-perfect scores published for this dataset
are not evidence of a working detector.

Run:  python src/task3_ml.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import (RANDOM_STATE, TARGET, Timer, build_preprocessor, evaluate,
                    feature_sets, load_data, print_report, save_figure,
                    save_table, temporal_split)


def header(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def make_models() -> dict:
    """
    Two classifiers required by the brief.

    class_weight='balanced' is not used: the classes are near 50/50, so
    reweighting would only distort the decision threshold.

    max_depth=8 on the tree is a deliberate limit. Unrestricted, it grows until
    it isolates individual flows and reports a near-perfect training score that
    collapses on the later time period.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, random_state=RANDOM_STATE
        ),
    }


def main() -> None:
    header("1. LOAD AND SPLIT")
    df = load_data()
    train, test, cut = temporal_split(df)
    print(f"  total rows : {len(df):,}")
    print(f"  train      : {len(train):,}  ({train.ts.min()} -> {train.ts.max()})")
    print(f"  test       : {len(test):,}  ({test.ts.min()} -> {test.ts.max()})")
    print(f"  malicious rate  train {train[TARGET].mean()*100:.2f}%  "
          f"test {test[TARGET].mean()*100:.2f}%")
    print("\n  TEMPORAL split, not random: the model is trained on the earlier")
    print("  period and tested on the later one, which is what a deployed")
    print("  detector actually faces.")

    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()

    header("2. PREPROCESSING")
    print("  numeric      : median imputation (+ was-missing indicator) -> StandardScaler")
    print("  boolean      : passed through unchanged")
    print("  categorical  : one-hot, rare levels (<30 rows) collapsed together")
    print("\n  The was-missing indicator matters here. A missing duration means the")
    print("  connection was never answered; filling it with the median alone would")
    print("  make that look like an ordinary short connection.")

    results, fitted = [], {}

    header("3. TRAINING")
    for config_name, cols in feature_sets().items():
        feature_cols = cols["numeric"] + cols["boolean"] + cols["categorical"]
        print(f"\n  [{config_name}] {len(feature_cols)} input columns")
        if config_name == "honest":
            print("    excludes id.resp_p, id.orig_p, service")

        X_train = train[feature_cols]
        X_test = test[feature_cols]

        for model_name, estimator in make_models().items():
            pipe = Pipeline([
                ("prep", build_preprocessor(**cols)),
                ("model", estimator),
            ])
            with Timer() as t:
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            row = evaluate(f"{model_name} ({config_name})", y_test, y_pred,
                           y_prob, t.seconds)
            row["config"] = config_name
            row["algorithm"] = model_name
            row["n_features_in"] = len(feature_cols)
            results.append(row)
            fitted[row["model"]] = pipe
            print_report(row)

    header("4. COMPARISON")
    table = pd.DataFrame(results)
    display = table[["algorithm", "config", "precision_malicious",
                     "recall_malicious", "f1_malicious", "pr_auc_malicious",
                     "false_positives", "accuracy", "train_seconds"]]
    print(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    leak_rows = table[table.config == "with_leak"].set_index("algorithm")
    honest_rows = table[table.config == "honest"].set_index("algorithm")
    print("\n  Effect of removing the three port/service columns:")
    for algo in leak_rows.index:
        d_acc = (leak_rows.loc[algo, "accuracy"] - honest_rows.loc[algo, "accuracy"]) * 100
        d_fp = honest_rows.loc[algo, "false_positives"] - leak_rows.loc[algo, "false_positives"]
        print(f"    {algo:<20} accuracy -{d_acc:5.2f} pp,  "
              f"false positives +{d_fp:,}")

    print("\n  INTERPRETATION: with the port columns the task is a lookup, so any")
    print("  model scores near-perfectly. Without them the same algorithms fall")
    print("  back on protocol and connection-state behaviour, and the false")
    print("  positives that a real SOC would have to triage appear.")

    header("5. COLLAPSE CHECK - IS A MODEL JUST A ONE-FEATURE RULE?")
    print("  Two models scoring identically is suspicious. This compares every")
    print("  model's predictions against two hand-written rules. 100% agreement")
    print("  means the model makes no decision the rule does not already make.")

    rules = {
        "id.resp_p in [23,2323,8080,9527]":
            test["id.resp_p"].isin([23, 2323, 8080, 9527]).to_numpy().astype(int),
        "proto == tcp": (test["proto"] == "tcp").to_numpy().astype(int),
    }
    collapse = []
    for model_name, pipe in fitted.items():
        cols = feature_sets()["honest" if "(honest)" in model_name else "with_leak"]
        feature_cols = cols["numeric"] + cols["boolean"] + cols["categorical"]
        pred = pipe.predict(test[feature_cols])
        for rule_name, rule_pred in rules.items():
            agreement = float((pred == rule_pred).mean())
            collapse.append({"model": model_name, "rule": rule_name,
                             "agreement": agreement,
                             "identical": bool(agreement == 1.0)})
            flag = "   <-- IDENTICAL" if agreement == 1.0 else ""
            print(f"    {model_name:<34} vs {rule_name:<34} "
                  f"{agreement*100:7.3f}%{flag}")

    if any(c["identical"] for c in collapse):
        print("\n  FINDING: at least one model reproduces a single-feature rule")
        print("  exactly. It did not learn behaviour - it learned that TCP traffic")
        print("  in this capture is the scanning traffic. This is the most important")
        print("  result in the project: the dataset has two different activities")
        print("  (UDP with replies vs TCP SYNs with no reply) wearing two class")
        print("  labels, so high accuracy here is not evidence of threat detection.")

    header("6. FIGURES AND TABLES")

    # Confusion matrices, honest configuration only
    honest_models = [m for m in fitted if "(honest)" in m]
    fig, axes = plt.subplots(1, len(honest_models), figsize=(5.2 * len(honest_models), 4.4))
    if len(honest_models) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, honest_models):
        row = next(r for r in results if r["model"] == model_name)
        cm = [[row["TN"], row["FP"]], [row["FN"], row["TP"]]]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(model_name, fontsize=10)
        ax.set_xticks([0, 1], ["pred Benign", "pred Malicious"])
        ax.set_yticks([0, 1], ["true Benign", "true Malicious"])
        total = sum(sum(r) for r in cm)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i][j]:,}\n{cm[i][j]/total*100:.1f}%",
                        ha="center", va="center",
                        color="white" if cm[i][j] > total * 0.4 else "black",
                        fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Task 3 - confusion matrices (honest feature set, temporal test split)",
                 fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "fig4_ml_confusion_matrices.png")
    plt.close(fig)

    # Leak vs honest accuracy / false positives
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    pivot_acc = table.pivot(index="algorithm", columns="config", values="accuracy") * 100
    pivot_acc = pivot_acc[["with_leak", "honest"]]
    pivot_acc.plot(kind="bar", ax=ax1, color=["#c44e52", "#4c72b0"])
    ax1.set_title("Accuracy: with vs without the port columns")
    ax1.set_ylabel("% accuracy")
    ax1.set_ylim(80, 101)
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=0)
    ax1.legend(title="")

    pivot_fp = table.pivot(index="algorithm", columns="config", values="false_positives")
    pivot_fp = pivot_fp[["with_leak", "honest"]]
    pivot_fp.plot(kind="bar", ax=ax2, color=["#c44e52", "#4c72b0"])
    ax2.set_title("False positives on the test period")
    ax2.set_ylabel("benign flows wrongly alerted")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=0)
    ax2.legend(title="")
    fig.suptitle("Task 3 - what the leaking columns are worth", fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "fig5_leak_vs_honest.png")
    plt.close(fig)

    save_table(table, "task3_ml_results.csv")
    save_table(pd.DataFrame(collapse), "task3_collapse_check.csv")
    print("\nTask 3 complete.")


if __name__ == "__main__":
    main()
