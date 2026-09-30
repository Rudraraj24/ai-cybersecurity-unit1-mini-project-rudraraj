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

All four models — two classical, two neural — reach **exactly the same
performance**, and each one turns out to be reproducing a single-feature rule:

| Family | Model | Accuracy | Precision (mal.) | Recall (mal.) | PR-AUC | False positives | Train time |
|---|---|---|---|---|---|---|---|
| ML | Logistic Regression | 95.55% | 0.9213 | 1.0000 | 0.9244 | 1,121 | 0.39 s |
| ML | Decision Tree | 95.55% | 0.9213 | 1.0000 | 0.9211 | 1,121 | 0.33 s |
| DL | MLP-small (32) | 95.55% | 0.9213 | 1.0000 | 0.9192 | 1,121 | 1.69 s |
| DL | MLP-deep (128,64,32) | 95.55% | 0.9213 | 1.0000 | 0.9243 | 1,121 | 10.78 s |

The deep network costs **roughly 25× the training time for +0.00 percentage
points** (20–33× across runs; timings are the one non-deterministic number here).

All four agree with the rule `proto == tcp` on **100.000%** of test predictions.
They did not learn to detect threats; they learned that in this capture the TCP
traffic is the scanning traffic.

That is the finding the project reports, and the reasoning behind it is in
[Task 2](#task-2--threat-landscape-analysis) below.

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
python src/task2_eda.py     # EDA, figures 1-3
python src/task3_ml.py      # ML models, figures 4-5
python src/task4_dl.py      # neural networks, figure 6  (needs task3 first)
```

Total runtime is roughly 30 seconds. Figures land in `outputs/figures/`, result
tables in `outputs/tables/`.

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
│   └── task4_dl.py               Task 4
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
as Task 3 — otherwise the comparison would be meaningless.

### Training time

One representative run (these are wall-clock times and vary between runs):

| Model | Time | vs fastest ML |
|---|---|---|
| Decision Tree | 0.33 s | 1× |
| Logistic Regression | 0.39 s | 1.2× |
| MLP-small (32) | 1.69 s | 5.1× |
| MLP-deep (128,64,32) | 10.78 s | **~33×** |

### Detection accuracy

Identical. Best ML 95.55%, best DL 95.55%, difference **+0.00 pp**. PR-AUC is
marginally *worse* for the networks (0.9243 vs 0.9244).

### Practical feasibility

The deep network carries **14,593 trainable parameters**; the Decision Tree is
depth-8 and can be printed on one page and read by an analyst. In a SOC an
alert that cannot be explained is an alert that gets ignored, so the
interpretable model wins unless the network buys a real accuracy margin. Here
it buys none.

### Does extra capacity help?

No. Both networks also agree with `proto == tcp` on 100.000% of predictions.
**The ceiling is not the model — it is the data.** Benign traffic in this
capture is UDP that got answered; malicious traffic is TCP SYNs that did not.
Those are two different activities wearing two class labels, so there is no
richer pattern left for a deeper network to find.

---

## What this project concludes

Row-level supervised classification on this capture is not a solved detection
problem — it is a mislabelled one. Every model tested, from a linear classifier
to a three-layer network, converges on the same single-feature rule and the
same 1,121 false positives, because the label was generated from a property of
the data rather than observed independently.

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
