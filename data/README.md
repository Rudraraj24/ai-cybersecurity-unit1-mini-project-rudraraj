# Dataset

## Source

**Aposemat IoT-23**, capture `CTU-IoT-Malware-Capture-1-1`, produced by the
Stratosphere Laboratory, CTU University, Czech Republic.

* Project page: <https://www.stratosphereips.org/datasets-iot23>
* Licence: **CC BY 4.0** — free to use with attribution.
* Citation: Sebastian Garcia, Agustin Parmisano & Maria Jose Erquiaga (2020),
  *IoT-23: A labeled dataset with malicious and benign IoT network traffic*,
  Stratosphere Laboratory. DOI: 10.5281/zenodo.4743746

Rows are Zeek `conn.log` flow summaries — connection metadata only. There is no
packet payload in this dataset, so nothing here contains message contents.

## What is committed here

`iot23_sample.csv.gz` — 84,063 flows, 1.83 MB.

The full capture is 1,008,748 flows / 138 MB, which is too large for a Git
repository. The sample takes **every 12th flow** in time order, which:

* preserves the chronological ordering the temporal train/test split needs,
* keeps the class balance intact (53.48% malicious in the full capture,
  53.47% in the sample),
* thins the dense scan bursts evenly across the whole capture rather than
  drawing clusters.

Regenerate with `python src/make_sample.py --source <path-to-full-capture>`.

## Columns

| Column | Meaning |
|---|---|
| `ts` | flow start time |
| `id.orig_h` / `id.orig_p` | source IP and port |
| `id.resp_h` / `id.resp_p` | destination IP and port |
| `proto` | tcp / udp / icmp |
| `service` | application protocol Zeek recognised, if any |
| `duration` | connection length in seconds — **NaN when never answered** |
| `orig_bytes` / `resp_bytes` | payload bytes each way — **NaN when never answered** |
| `conn_state` | Zeek connection state (S0 = no reply, SF = normal completion, …) |
| `history` | per-packet connection history string |
| `orig_pkts` / `resp_pkts` | packet counts |
| `orig_ip_bytes` / `resp_ip_bytes` | total IP bytes each way |
| `label` | Benign / Malicious |
| `detailed-label` | attack type, or Benign |
| `no_response` / `never_established` | explicit missingness flags |
| `is_malicious` | 1 / 0 target used by Tasks 3 and 4 |

## Two things to know before modelling

**1. Missing is not unknown.** `duration`, `orig_bytes` and `resp_bytes` are
absent on the same rows — the ones where nobody answered the connection. A
missing value here means "no reply", which is a *different event* from
"answered with zero bytes". Filling both with 0 merges two opposite outcomes.
The scripts impute with the median but add a was-missing indicator column so
the distinction survives.

**2. The labels are derived from the destination port.** Stratosphere labelled
a flow as scan traffic when enough flows converged on one port, so
`id.resp_p ∈ {23, 2323, 8080, 9527}` reproduces the label at 100% precision
with zero false positives. "Benign" therefore does **not** mean legitimate —
most benign rows are unanswered connection attempts from the same infected
device that fell below the flagging threshold. Task 2 measures this and
Tasks 3–4 model with and without those columns.

## Class distribution in the sample

| detailed-label | flows |
|---|---|
| PartOfAHorizontalPortScan | 44,945 |
| Benign | 39,117 |
| C&C | 1 |

Only 1 of the full capture's 8 command-and-control flows survives the 1-in-12
stride. That is expected and does not affect the binary task, but it does mean
this sample cannot support a multi-class C&C experiment.

## Scope limitation

One device, one malware family, one network, under five days. Nothing measured
here should be assumed to generalise to other networks or attack types.
