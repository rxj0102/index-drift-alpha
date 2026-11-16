"""Run the event study on the frozen returns and write results/.

  results/metrics.json        everything the README reports
  results/events.csv          every event with its group and abnormal return windows
  results/figures/path.png    average abnormal return path around the announcement
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import events, study  # noqa: E402

LABEL = {"addition_new": "New additions", "addition_from_sp600": "Moved up from the S&P 600",
         "deletion_to_sp600": "Moved down to the S&P 600", "deletion_to_sp500": "Moved up to the S&P 500",
         "deletion_other": "Other deletions", "early_additions": "All additions, 2012 to 2019",
         "early_deletions": "All deletions, 2012 to 2019"}


def rounded(x):
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def paths(df: pd.DataFrame, returns: pd.DataFrame, group: str, lo: int = -5, hi: int = 8) -> list:
    """Mean cumulative abnormal return by trading day relative to A0 (0 at the close before A0)."""
    ids = df[(df["group_key"] == group) & df["A0"].notna()]
    by_id = {k: g for k, g in returns.groupby("event_id", sort=False)}
    rows = []
    for e in ids.itertuples():
        w = by_id[e.event_id]
        a0 = int(np.searchsorted(w["day"].to_numpy(), np.datetime64(e.A0)))
        ar = (w["ret"] - w["bench_ret"]).to_numpy()
        if a0 + lo < 0 or a0 + hi >= len(ar):
            continue
        seg = ar[a0 + lo: a0 + hi + 1]
        rows.append(np.cumsum(seg) - np.cumsum(seg)[-lo - 1])
    return [float(x) for x in np.mean(rows, axis=0)] if rows else []


def figure(m: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    colors = {"addition_new": "#2a78d6", "deletion_to_sp600": "#1baf7a", "addition_from_sp600": "#eb6834"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "xtick.color": muted, "ytick.color": muted,
                         "text.color": ink, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(m["path"]["lo"], m["path"]["hi"] + 1)
    for g, c in colors.items():
        y = np.array(m["path"][g]) * 100
        ax.plot(x, y, color=c, linewidth=2, marker="o", markersize=4)
        ax.annotate(LABEL[g], (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=ink, fontsize=9)
    ax.axhline(0, color=ink, linewidth=0.8)
    ax.axvspan(-1, 0, color=grid, alpha=0.6, linewidth=0)
    ax.set_xlim(x[0], x[-1] + 4.2)
    ax.set_xticks(x)
    ax.set_xlabel("Trading days from the first session after the announcement (shaded: the reaction)", color=muted)
    ax.set_ylabel("Cumulative abnormal return, percent", color=muted)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_title("S&P 400 changes since 2020: the whole move happens in one session", loc="left", fontsize=10.5)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "path.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ev = events.build_events(*events.load_reference())
    n_mna = int((ev["group"] == "deletion_mna").sum())
    ev = ev[ev["group"] != "deletion_mna"].reset_index(drop=True)
    returns = study.load_returns()
    df = study.compute(ev, returns)
    late = df["date"] >= events.CLASSIFIED_FROM
    df["group_key"] = np.where(late, df["group"], np.where(df["type"] == "addition", "early_additions", "early_deletions"))
    groups = {k: study.summarize(df[df["group_key"] == k]) for k in LABEL}

    pvals = {f"{g}|{w}": groups[g][w]["p"] for g in study.FLOW_GROUPS for w in study.TRADABLE}
    adj = study.holm(pvals)
    for key, p in adj.items():
        g, w = key.split("|")
        groups[g][w]["p_holm"] = p

    with_ann = df["A0"].notna()
    payload = rounded({
        "meta": {"event_start": events.EVENT_START, "event_end": events.EVENT_END,
                 "classified_from": events.CLASSIFIED_FROM, "n_events": int(len(df)), "n_mna_excluded": n_mna,
                 "n_with_prices": int(df["has_prices"].sum()), "coverage": float(df["has_prices"].mean()),
                 "coverage_additions": float(df.loc[df["type"] == "addition", "has_prices"].mean()),
                 "coverage_deletions": float(df.loc[df["type"] == "deletion", "has_prices"].mean()),
                 "coverage_from_2020": float(df.loc[late, "has_prices"].mean()),
                 "coverage_before_2020": float(df.loc[~late, "has_prices"].mean()),
                 "n_with_announcement": int(with_ann.sum()),
                 "median_gap_days": float(df.loc[with_ann, "gap"].median()),
                 "n_tradable_tests": len(pvals)},
        "groups": groups, "labels": LABEL,
        "path": {"lo": -5, "hi": 8, **{g: paths(df, returns, g) for g in study.FLOW_GROUPS}},
    })
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    df.drop(columns=["group_key"]).round(6).to_csv(out / "events.csv", index=False)
    figure(payload, out / "figures")

    print(json.dumps(payload["meta"]))
    for k, g in payload["groups"].items():
        print(f"{k:22s} n {g['n_events']:3d} px {g['n_with_prices']:3d} ann {g['n_with_announcement']:3d}")
        for w in list(study.E_WINDOWS) + study.A_WINDOWS + ["REACTION", "FROM_OPEN"]:
            v = g.get(w)
            if v:
                print(f"     {w:10s} {v['mean']*100:+6.2f}% t {v['t']:+6.2f} dates {v['n_dates']:3d} med {v['median_event']*100:+5.2f}% pos {v['share_positive']:.2f}"
                      + (f" holm {v['p_holm']:.3f}" if "p_holm" in v else ""))


if __name__ == "__main__":
    main()
