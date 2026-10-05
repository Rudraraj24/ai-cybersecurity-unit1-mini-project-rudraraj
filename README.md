# AI in Cyber Security — Unit 1 Mini Project

An AI-enabled cyber security analysis and detection prototype built on real
labelled network-traffic data: exploratory threat analysis, a supervised
machine-learning baseline, and a neural-network prototype compared against it.

**Dataset:** Aposemat IoT-23, capture `CTU-IoT-Malware-Capture-1-1`
(Stratosphere Laboratory, CC BY 4.0). 84,063 Zeek flow records sampled from a
1,008,748-flow capture. See [`data/README.md`](data/README.md) for provenance,
column meanings and licence.

**Scope:** Tasks 1–4. Task 5 (ethical reflection) is not included.

---

## Headline result

Every model is evaluated on two feature configurations: the **full** set, and an
**honest** set with the three label-leaking columns removed. The two tell
opposite stories, and that contrast is the project's result.

### Full feature set — the algorithms differ

| Family | Model | Accuracy | Precision (mal.) | F1 | PR-AUC | False positives | Train time |
|---|---|---|---|---|---|---|---|
| ML | Logistic Regression | 99.01% | 0.9813 | 0.9906 | 0.9945 | 250 | 0.5 s |
| ML | **Decision Tree** | **100.00%** | 1.0000 | 1.0000 | 1.0000 | **0** | 0.5 s |
| DL | MLP-small (32) | 99.01% | 0.9814 | 0.9906 | 0.9945 | 249 | 2.4 s |
| DL | MLP-deep (128,64,32) | 99.62% | 0.9927 | 0.9963 | 0.9980 | 97 | 26.2 s |

The Decision Tree wins outright — perfect accuracy, zero false positives, and
joint-fastest training — because it can **represent** the port-lookup rule
exactly in a few splits. The networks must **approximate** that same step
function with thousands of weights; MLP-deep spends 25–55× the training time
(it varies per run) and still doesn't match it. When the true boundary is a
rule, a rule-based learner wins on accuracy, speed and explainability
simultaneously.

### Honest feature set — the algorithms become indistinguishable

| Family | Model | Accuracy | Precision (mal.) | F1 | PR-AUC | False positives |
|---|---|---|---|---|---|---|
| ML | Logistic Regression | 95.55% | 0.9213 | 0.9590 | 0.9244 | 1,121 |
| ML | Decision Tree | 95.55% | 0.9213 | 0.9590 | 0.9211 | 1,121 |
| DL | MLP-small | 95.55% | 0.9213 | 0.9590 | 0.9192 | 1,121 |
| DL | MLP-deep | 95.55% | 0.9213 | 0.9590 | 0.9243 | 1,121 |

Identical to four decimal places, and all four agree with the rule
`proto == tcp` on **100.000%** of test predictions.

**This is not a tie and not a bug — it is the ceiling of the data**, and
`src/diagnose_identical_scores.py` proves it with three experiments:

1. **Threshold sweep.** Each model emits only 17–25 distinct probabilities
   across 25,219 test rows. No threshold separates them.
2. **Remove the dominating feature** (TCP-only subset). Every model returns
   recall 1.0000 and FN 0 — they predict "malicious" for all 14,597 TCP flows,
   i.e. they give up and return the majority class.
3. **Remove the class imbalance** (balanced training set). The models finally
   differ — and all become useless, landing within ±0.01 of the no-skill PR-AUC
   baseline of 0.9214 while accuracy collapses as low as 7.86%.

Once the destination port is gone, Zeek's flow metadata in this capture carries
essentially **one usable bit — the protocol** — and every algorithm finds it
immediately.

---

## Task 1 — Setup

### Requirements
Python 3.11 or newer (developed on 3.14.5).

### Install

```bash
git clone https://github.com/Rudraraj24/ai-cybersecurity-unit1-mini-project-rudraraj.git
cd ai-cybersecurity-unit1-mini-project-rudraraj

python -m venv .venv

# Windows (PowerShell / Git Bash)
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Validate the setup

```bash
python src/verify_setup.py
```

Checks every required package imports, prints its version, warns if you are not
inside a virtual environment, and loads the dataset.

### Run everything

```bash
python src/task2_eda.py                   # EDA, figures 1-3
python src/task3_ml.py                    # ML models, figures 4-5
python src/task4_dl.py                    # neural networks, figure 6 (needs task3 first)
python src/diagnose_identical_scores.py   # why the honest scores are identical
```

Total runtime is roughly three minutes, most of it MLP-deep on the full feature
set. Figures land in `outputs/figures/`, result tables in `outputs/tables/`.

### Layout

```
├── data/
│   ├── iot23_sample.csv.gz       84,063 flows (committed)
│   └── README.md                 provenance, licence, column reference
├── src/
│   ├── common.py                 shared loading, split, preprocessing, metrics
│   ├── verify_setup.py           Task 1 environment check
│   ├── make_sample.py            rebuilds the sample from the full capture
│   ├── task2_eda.py              Task 2
│   ├── task3_ml.py               Task 3
│   ├── task4_dl.py               Task 4
│   └── diagnose_identical_scores.py   why the honest-set scores match
├── notebooks/
│   └── mini_project.ipynb        all four tasks in notebook form
├── outputs/
│   ├── figures/                  6 exported plots
│   └── tables/                   result tables as CSV
└── requirements.txt
```

---

## Task 2 — Threat landscape analysis

`python src/task2_eda.py` → `outputs/figures/fig1–fig3`

### What the traffic looks like

| Property | Value |
|---|---|
| Flows | 84,063 |
| Capture window | 2018-05-09 → 2018-05-14 (4 days) |
| Class balance | 53.47% Malicious / 46.53% Benign |
| Distinct destination IPs | 78,596 |
| Distinct destination ports | 28,361 |
| `conn_state == S0` (no reply at all) | 96.18% of all flows |

### Threat indicators found

**Connection state.** 96.18% of flows are `S0` — a connection attempt that never
received a reply. That is the signature of horizontal scanning: spray SYNs
across many hosts, most never answer.

**Protocol.** TCP is 92.37% malicious; UDP is 0.00% malicious (1 flow of
33,937); ICMP is 0.00%. The protocol alone almost separates the classes.

**Fan-out.** The infected host contacts thousands of distinct destination IPs
per 5-minute window (`fig3_scanning_fanout.png`). This is the behavioural
signal that a port lookup cannot fake, and the one a real detector should use.

### The key insight: the labels are derived from the destination port

```
Rule: predict Malicious if id.resp_p in [23, 2323, 8080, 9527]
  TP 44,945   FP 0   FN 1   TN 39,117
  accuracy 99.9988%   precision 100.0000%   recall 99.9978%
```

A four-entry lookup table scores what published notebooks on this dataset
report as a machine-learning result. Stratosphere labelled a flow as scan
traffic when enough flows converged on one port, so the destination port **is**
the labelling rule.

Two consequences drive the rest of the project:

1. Any model given `id.resp_p` will recover this table and report ~100%. That
   number measures the labelling process, not detection ability.
2. "Benign" does not mean legitimate. Most benign rows are unanswered
   connection attempts from the same infected device that simply fell below the
   flagging threshold.

So Tasks 3 and 4 train on an **honest feature set** with `id.resp_p`,
`id.orig_p` and `service` removed, and report the leaking configuration only as
a comparison.

---

## Task 3 — Machine-learning threat classification

`python src/task3_ml.py` → `outputs/figures/fig4–fig5`

### Preprocessing

| Step | Treatment | Why |
|---|---|---|
| Missing values | median imputation **+ was-missing indicator** | A missing `duration` means the connection was never answered. Median-filling alone would make that look like an ordinary short connection; the indicator column preserves the distinction. |
| Categorical encoding | one-hot, levels with <30 rows collapsed | `history` has 100+ rare values; without collapsing them the models get near-empty columns they can use to memorise individual flows. |
| Numerical normalisation | `StandardScaler` | Logistic Regression and the MLP are both scale-sensitive; byte counts span six orders of magnitude. |
| Train/test split | **temporal**, first 70% by time | Adjacent flows are near-duplicates of the same scan burst. A random split puts near-identical rows on both sides and measures memorisation. |

### Results

| Model | Config | Accuracy | Precision | Recall | F1 | FP |
|---|---|---|---|---|---|---|
| Logistic Regression | with leak | 99.01% | 0.9813 | 1.0000 | 0.9906 | 250 |
| Decision Tree | with leak | **100.00%** | 1.0000 | 1.0000 | 1.0000 | **0** |
| Logistic Regression | honest | 95.55% | 0.9213 | 1.0000 | 0.9590 | 1,121 |
| Decision Tree | honest | 95.55% | 0.9213 | 1.0000 | 0.9590 | 1,121 |

Removing three columns costs the Decision Tree 4.45 points of accuracy and adds
1,121 false positives.

### Collapse check

Two models scoring *identically* is suspicious, so the script compares every
model's predictions against hand-written rules:

| Model | vs port lookup | vs `proto == tcp` |
|---|---|---|
| Decision Tree (with leak) | **100.000%** | 95.555% |
| Logistic Regression (honest) | 95.555% | **100.000%** |
| Decision Tree (honest) | 95.555% | **100.000%** |

The leaky Decision Tree *is* the port lookup table. Both honest models *are*
the rule "TCP means malicious". Neither learned behaviour.

---

## Task 4 — Deep-learning prototype and comparison

`python src/task4_dl.py` → `outputs/figures/fig6`

Two `MLPClassifier` architectures on the same split, preprocessing and metrics
as Task 3, and on **both** feature configurations — otherwise the comparison has
nothing to vary.

### Training time (full feature set)

Wall-clock, one representative run. These are the **only** numbers in this
README that change between runs — across repeated runs MLP-deep ranged 26–54 s
while the classical models stayed under ~1.5 s, so the ratio below is indicative
rather than exact:

| Model | Time | vs fastest |
|---|---|---|
| Logistic Regression | 0.47 s | 1× |
| Decision Tree | 0.53 s | 1.1× |
| MLP-small (32) | 2.38 s | 5.1× |
| MLP-deep (128,64,32) | 26.22 s | **56×** |

### Detection accuracy (full feature set)

| | Best | Model |
|---|---|---|
| Machine Learning | **100.00%** | Decision Tree |
| Deep Learning | 99.62% | MLP-deep |
| Difference | **−0.38 pp** | DL is *worse* |
| Fewest false positives | **0** | Decision Tree |

The deep network is both slower and less accurate than a depth-8 tree.

### Practical feasibility

MLP-deep carries **14,593 trainable parameters**; the Decision Tree is depth-8
and fits on one page an analyst can read. In a SOC an alert that cannot be
explained is an alert that gets ignored — so here the interpretable model wins
on explainability, accuracy *and* speed at the same time. There is no trade-off
to make.

The reason is structural: the true decision boundary in this dataset is a
**rule** (`port ∈ {23, 2323, 8080, 9527}`). A tree represents rules natively and
finds it exactly. A neural network has to approximate a step function with
continuous weights, which is why more capacity and ~55× the compute still lands
short of a perfect score.

### Does extra capacity help on the honest set?

No. Both networks also agree with `proto == tcp` on 100.000% of predictions, and
all three diagnostic experiments confirm the limit is the data rather than the
model. Benign traffic in this capture is UDP that got answered; malicious
traffic is TCP SYNs that did not. Those are two different activities wearing two
class labels, so there is no richer pattern for a deeper network to find.

---

## What this project concludes

Row-level supervised classification on this capture is not a solved detection
problem — it is a mislabelled one.

Given the leaking columns, the four algorithms separate cleanly and a Decision
Tree wins by reproducing the labelling rule exactly. Take those columns away and
all four converge on the same single-feature rule and the same 1,121 false
positives, because the label was generated from a property of the data rather
than observed independently. The algorithms are only distinguishable while
they have access to the answer key.

The honest next step is not a bigger model. It is **windowed behavioural
features** — per-host fan-out, connection rate, share unanswered, and entropy
of destination IPs over time — which `fig3_scanning_fanout.png` shows are
strongly present in this traffic and which a port lookup cannot reproduce.

### Limitations

One device, one malware family, one network, under five days. The sample is
1-in-12 of the full capture and retains only 1 of its 8 command-and-control
flows, so nothing here supports a multi-class or C&C-detection claim.

---

## Reproducibility

All results were produced with the pinned versions in `requirements.txt`
(Python 3.14.5, scikit-learn 1.9.1, pandas 3.0.6) and `random_state=42`
throughout.

**Training times are wall-clock and vary with hardware and between runs** — they
are the only non-deterministic figures quoted here. Every other number (accuracy,
precision, recall, F1, PR-AUC, confusion counts, agreement percentages) is
deterministic and reproduces exactly.

The notebook was executed with the venv registered as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=ai-cyber-unit1 --display-name "Python (ai-cyber-unit1)"
python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=ai-cyber-unit1 notebooks/mini_project.ipynb
```

If you have Anaconda installed, use `python -m nbconvert` rather than
`jupyter nbconvert` — the latter dispatches to Anaconda's executable on PATH,
whose pyzmq fails to connect to the venv kernel and silently produces an
unexecuted notebook.
