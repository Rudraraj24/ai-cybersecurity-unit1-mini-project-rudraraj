"""
Task 2 - Cyber Security Threat Landscape Analysis (EDA)

Loads the IoT-23 network-traffic sample, characterises the threat landscape and
writes the figures and tables used in the report.

Run:  python src/task2_eda.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")          # write files, no interactive window needed

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ROOT, load_data, save_figure, save_table

sns.set_theme(style="whitegrid", palette="deep")
pd.set_option("display.width", 120)

SCAN_PORTS = [23, 2323, 8080, 9527]      # telnet, alt-telnet, http-alt, Mirai-family

# One colour per class, used consistently across every figure.
CLASS_ORDER = ["Benign", "Malicious"]
CLASS_COLOURS = {"Benign": "#4c72b0", "Malicious": "#c44e52"}


def header(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def main() -> None:
    header("1. DATASET OVERVIEW")
    df = load_data()
    print(f"  rows            : {len(df):,}")
    print(f"  columns         : {df.shape[1]}")
    print(f"  capture window  : {df.ts.min()}  ->  {df.ts.max()}")
    print(f"  duration        : {(df.ts.max() - df.ts.min()).days} days")
    print(f"  source hosts    : {df['id.orig_h'].nunique():,}")
    print(f"  destination IPs : {df['id.resp_h'].nunique():,}")
    print(f"  destination ports: {df['id.resp_p'].nunique():,}")

    print("\n  Class balance:")
    for label, n in df["label"].value_counts().items():
        print(f"    {label:<12} {n:>8,}  ({n/len(df)*100:5.2f}%)")

    print("\n  Attack types (detailed-label):")
    for label, n in df["detailed-label"].value_counts().items():
        print(f"    {label:<28} {n:>8,}  ({n/len(df)*100:6.3f}%)")

    print("\n  Missing values per column:")
    missing = df.isna().sum()
    for col, n in missing[missing > 0].items():
        print(f"    {col:<18} {n:>8,}  ({n/len(df)*100:5.2f}%)")
    if missing.sum() == 0:
        print("    (none)")

    print("\n  INSIGHT: duration/orig_bytes/resp_bytes are missing together on the")
    print("  same rows. In Zeek these are absent when the connection was never")
    print("  answered -- a missing value here means 'no reply', not 'unknown'.")
    print("  That is why the project keeps explicit no_response / never_established")
    print("  flags rather than silently filling zeros.")

    # ----------------------------------------------------------------- threat indicators
    header("2. COMMON THREAT INDICATORS")

    print("\n  Connection state by class:")
    conn = pd.crosstab(df["conn_state"], df["label"])
    print(conn.to_string())
    s0_share = (df["conn_state"] == "S0").mean()
    print(f"\n  INSIGHT: {s0_share*100:.1f}% of ALL flows are conn_state S0 -- a")
    print("  connection attempt that got no reply at all. That is the signature of")
    print("  horizontal scanning: spray SYNs across many hosts, most never answer.")

    print("\n  Protocol by class:")
    proto = pd.crosstab(df["proto"], df["label"])
    proto["malicious_%"] = (proto["Malicious"] /
                            proto.sum(axis=1) * 100).round(2)
    print(proto.to_string())

    print("\n  Top 10 destination ports:")
    top_ports = (df.groupby("id.resp_p")
                   .agg(flows=("label", "size"),
                        malicious=("is_malicious", "sum"))
                   .assign(malicious_pct=lambda d: (d.malicious / d.flows * 100).round(2))
                   .sort_values("flows", ascending=False)
                   .head(10))
    print(top_ports.to_string())

    # ----------------------------------------------------------------- the leak
    header("3. KEY SECURITY INSIGHT - THE LABELS ARE DERIVED FROM THE PORT")

    on_scan_ports = df["id.resp_p"].isin(SCAN_PORTS)
    tp = int((on_scan_ports & (df.is_malicious == 1)).sum())
    fp = int((on_scan_ports & (df.is_malicious == 0)).sum())
    fn = int((~on_scan_ports & (df.is_malicious == 1)).sum())
    tn = int((~on_scan_ports & (df.is_malicious == 0)).sum())
    acc = (tp + tn) / len(df)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"\n  Rule: predict Malicious if destination port in {SCAN_PORTS}")
    print(f"    TP {tp:,}   FP {fp:,}   FN {fn:,}   TN {tn:,}")
    print(f"    accuracy {acc*100:.4f}%   precision {prec*100:.4f}%   recall {rec*100:.4f}%")
    print("\n  A four-entry lookup table scores what published notebooks on this")
    print("  dataset report as a machine-learning result. The Stratosphere team")
    print("  labelled a flow as scan traffic when enough flows converged on one")
    print("  port, so the destination port IS the labelling rule.")
    print("\n  CONSEQUENCE FOR THIS PROJECT: 'Benign' here does not mean legitimate.")
    print("  Most benign rows are unanswered connection attempts from the same")
    print("  infected device -- they just fell below the flagging threshold. Task 3")
    print("  therefore trains with AND without the port features, and treats the")
    print("  no-port result as the honest one.")

    # ----------------------------------------------------------------- figures
    header("4. VISUALISATIONS")

    # Figure 1 - normal vs suspicious overview (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("IoT-23 (CTU-Malware-Capture-1-1): normal vs suspicious behaviour",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    # Fixed order and colour mapping: value_counts() orders by frequency, which
    # would otherwise colour Malicious blue here and red in every other panel.
    counts = df["label"].value_counts().reindex(CLASS_ORDER)
    ax.bar(counts.index.astype(str), counts.values,
           color=[CLASS_COLOURS[c] for c in CLASS_ORDER])
    ax.set_title("Class balance")
    ax.set_ylabel("flows")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.18)

    ax = axes[0, 1]
    conn_pct = (pd.crosstab(df["conn_state"], df["label"], normalize="index") * 100)
    conn_pct = conn_pct.loc[conn.sum(axis=1).sort_values(ascending=False).head(6).index]
    conn_pct = conn_pct.reindex(columns=CLASS_ORDER)
    conn_pct.plot(kind="barh", stacked=True, ax=ax,
                  color=[CLASS_COLOURS[c] for c in CLASS_ORDER], legend=True)
    ax.set_title("Connection state composition (top 6)")
    ax.set_xlabel("% of flows in that state")
    ax.set_ylabel("")
    ax.legend(title="", loc="lower right", fontsize=8)

    ax = axes[1, 0]
    proto_ct = pd.crosstab(df["proto"], df["label"]).reindex(columns=CLASS_ORDER)
    proto_ct.plot(kind="bar", ax=ax,
                  color=[CLASS_COLOURS[c] for c in CLASS_ORDER], legend=True)
    ax.set_title("Protocol by class")
    ax.set_ylabel("flows")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="", fontsize=8)

    ax = axes[1, 1]
    hourly = (df.set_index("ts")
                .is_malicious.resample("1h")
                .agg(["size", "mean"]).dropna())
    ax.plot(hourly.index, hourly["size"], color="#4c72b0", lw=1.2, label="flows/hour")
    ax.set_ylabel("flows per hour", color="#4c72b0")
    ax2 = ax.twinx()
    ax2.plot(hourly.index, hourly["mean"] * 100, color="#c44e52", lw=1.2,
             label="% malicious")
    ax2.set_ylabel("% malicious", color="#c44e52")
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    ax.set_title("Activity over the capture window")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    save_figure(fig, "fig1_threat_overview.png")
    plt.close(fig)

    # Figure 2 - attack-related feature distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle("Distribution of attack-related features",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    port_counts = (df[df.is_malicious == 1]["id.resp_p"].value_counts().head(8))
    ax.bar(port_counts.index.astype(str), port_counts.values, color="#c44e52")
    ax.set_title("Destination ports targeted (malicious flows)")
    ax.set_xlabel("destination port")
    ax.set_ylabel("flows")

    ax = axes[1]
    for label in CLASS_ORDER:
        vals = df.loc[df.label == label, "orig_ip_bytes"].clip(upper=200)
        ax.hist(vals, bins=40, alpha=0.6, label=label, color=CLASS_COLOURS[label])
    ax.set_title("Bytes sent by originator (clipped at 200)")
    ax.set_xlabel("orig_ip_bytes")
    ax.set_ylabel("flows")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[2]
    resp_share = (df.groupby("label", observed=True)["no_response"]
                    .mean().mul(100).reindex(CLASS_ORDER))
    ax.bar(resp_share.index.astype(str), resp_share.values,
           color=[CLASS_COLOURS[c] for c in CLASS_ORDER])
    ax.set_title("Flows that received no response")
    ax.set_ylabel("% of class")
    ax.set_ylim(0, 105)
    for i, v in enumerate(resp_share.values):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_figure(fig, "fig2_feature_distributions.png")
    plt.close(fig)

    # Figure 3 - fan-out, the behavioural signal a port lookup cannot fake
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fanout = (df.set_index("ts")
                .groupby(pd.Grouper(freq="5min"))
                .agg(distinct_dst_ip=("id.resp_h", "nunique"),
                     distinct_dst_port=("id.resp_p", "nunique"))
                .dropna())
    ax.plot(fanout.index, fanout["distinct_dst_ip"], lw=1.1,
            color="#c44e52", label="distinct destination IPs")
    ax.plot(fanout.index, fanout["distinct_dst_port"], lw=1.1,
            color="#4c72b0", label="distinct destination ports")
    ax.set_title("Scanning fan-out per 5-minute window", fontweight="bold")
    ax.set_ylabel("distinct values contacted")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_figure(fig, "fig3_scanning_fanout.png")
    plt.close(fig)

    # ----------------------------------------------------------------- tables
    header("5. TABLES SAVED FOR THE REPORT")
    save_table(conn.reset_index(), "task2_conn_state_by_class.csv")
    save_table(proto.reset_index(), "task2_protocol_by_class.csv")
    save_table(top_ports.reset_index(), "task2_top_destination_ports.csv")
    save_table(pd.DataFrame([{
        "rule": f"id.resp_p in {SCAN_PORTS}",
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": acc, "precision": prec, "recall": rec,
    }]), "task2_port_lookup_baseline.csv")

    print(f"\nTask 2 complete. Figures in outputs/figures, tables in outputs/tables.")


if __name__ == "__main__":
    main()
