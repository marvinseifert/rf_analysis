"""
Moving-edge orientation preference vs the radiating cone rows and the RF long axis (23/11/2026).

Input: edge_tuning_table.csv (edge_tuning_table.py: peak-response tuning per cell and condition, OSI / DSI with a
direction-label shuffle test) and rf_vs_cone_row_angle_all_recordings_cells.csv (RF long axis and the cone row
through each cell, rows assumed to radiate from the optic nerve head). Both converted to the polar frame
(0° = up, clockwise; RF/row table: polar = (projected + 90) % 180).

Cells: significantly orientation-tuned (p_OSI < --alpha), RF not near the stimulus edge, in the RF/row table.
Split into direction-selective (p_DSI < --alpha; their "orientation" is the axis of their preferred direction) and
orientation-only cells.
Per cell: |preferred motion axis - own row| and |preferred motion axis - RF long axis| (0-90°).
Test: within-recording shuffle (preferred axes permuted among the tuned cells of the same recording and condition,
5000x), as rf_row_strict_tests.py. Two-sided: p(across) and p(along).
Also: how many cells are tuned vs the --alpha fraction expected by chance (binomial test).

Run from the repo root (~1 min):
    MPLBACKEND=Agg poetry run python laura_scripts/OSI/edge_orientation_vs_cone_rows.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.05)
args = parser.parse_args()

OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/OSI")
ROWS_CSV = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_row_angle_prediction/"
                "rf_vs_cone_row_angle_all_recordings_cells.csv")
tun = pd.read_csv(OUT / "edge_tuning_table.csv")
rf = pd.read_csv(ROWS_CSV)
rf = rf[~rf.near_edge.astype(bool) & rf.inside_displayed_stimulus.astype(bool)]
rf["rf_axis_polar"] = (rf.rf_axis_projected_deg + 90) % 180
rf["row_axis_polar"] = (rf.cone_row_axis_projected_deg + 90) % 180
df = tun.merge(rf[["recording", "cell_index", "rf_axis_polar", "row_axis_polar", "elongated", "quality"]],
               on=["recording", "cell_index"], how="inner")


def diff(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


df["vs_row"] = diff(df.pref_motion_axis_polar, df.row_axis_polar)
df["vs_rf"] = diff(df.pref_motion_axis_polar, df.rf_axis_polar)
df["os_tuned"] = df.p_OSI < args.alpha
df["ds"] = df.p_DSI < args.alpha
rng = np.random.default_rng(0)

print("Tuned cells (all cells with spikes, before merging with the RF table):")
for (r, c), d in tun.groupby(["recording", "condition"]):
    k = int((d.p_OSI < args.alpha).sum())
    print(f"  {r} {c}: {k}/{len(d)} OSI p < {args.alpha} ({k / len(d) * 100:.1f} %, chance {args.alpha * 100:.0f} %, "
          f"binomial p = {binomtest(k, len(d), args.alpha, alternative='greater').pvalue:.3g}); "
          f"DSI: {int((d.p_DSI < args.alpha).sum())}")


def shuffle_test(sel, col_pref="pref_motion_axis_polar", col_ref="row_axis_polar"):
    real = diff(sel[col_pref], sel[col_ref]).mean()
    groups = [d for _, d in sel.groupby(["recording", "condition"])]
    null = np.array([np.concatenate([diff(rng.permutation(d[col_pref].values), d[col_ref].values) for d in groups]).mean()
                     for _ in range(5000)])
    return real, null, float(np.mean(null >= real)), float(np.mean(null <= real))


tuned = df[df.os_tuned]
sets = {"all tuned": tuned, "orientation-only (not DS)": tuned[~tuned.ds], "direction-selective": tuned[tuned.ds]}
results = {}
print(f"\nTuned cells in the RF/row table: {len(tuned)} (cell x condition; a cell can appear in several conditions)")
for k, sel in sets.items():
    if len(sel) < 4:
        print(f"  {k}: n = {len(sel)}, too few"); continue
    real, null, p_across, p_along = shuffle_test(sel)
    results[k] = (real, null, p_across, p_along)
    print(f"  {k} (n = {len(sel)}): mean |pref axis - own row| {real:.1f}° (shuffle {null.mean():.1f}°), "
          f"p(across) = {p_across:.3f}, p(along) = {p_along:.3f}; % across {(diff(sel.pref_motion_axis_polar, sel.row_axis_polar) > 45).mean() * 100:.0f} %; "
          f"vs RF median {sel.vs_rf.median():.0f}°")
print("\nPer recording x condition (all tuned):")
for (r, c), d in tuned.groupby(["recording", "condition"]):
    print(f"  {r} {c}: n {len(d)}, vs own row median {d.vs_row.median():.0f}° ({(d.vs_row > 45).mean() * 100:.0f} % across), "
          f"vs RF median {d.vs_rf.median():.0f}°")

# ------------------------------------------------------------------ figure
conds = sorted(tuned.groupby(["recording", "condition"]).groups.keys())
fig, axs = plt.subplots(2, len(conds) + 1, figsize=(3.6 * (len(conds) + 1), 7.5), squeeze=False)
bins = np.linspace(0, 90, 7)
for j, key in enumerate(conds):
    d = tuned[(tuned.recording == key[0]) & (tuned.condition == key[1])]
    for i, (col, lab) in enumerate([("vs_row", "preferred motion axis vs own cone row"),
                                    ("vs_rf", "preferred motion axis vs RF long axis")]):
        ax = axs[i, j]
        ax.hist([d[~d.ds][col], d[d.ds][col]], bins=bins, stacked=True, color=["tab:blue", "tab:orange"],
                label=["orientation-only", "direction-selective"])
        ax.axhline(len(d) / (len(bins) - 1), color="grey", ls="--", lw=0.8)
        ax.set_xticks([0, 45, 90]); ax.set_xlim(0, 90)
        ax.set_title(f"{key[0]} {key[1]} (n = {len(d)})\n{lab}", fontsize=8, loc="left")
        if j == 0:
            ax.set_ylabel("0 = along, 90 = across", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
ax = axs[0, -1]
for k, col in zip(results, ["k", "tab:blue", "tab:orange"]):
    real, null, pa, pl_ = results[k]
    ax.hist(null - null.mean(), bins=40, histtype="step", color=col, alpha=0.6)
    ax.axvline(real - null.mean(), color=col, lw=2, label=f"{k}: {real - null.mean():+.1f}° (p across {pa:.2f}, along {pl_:.2f})")
ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel("mean |pref axis - own row| minus shuffle mean (°)\n< 0: along the rows, > 0: across")
ax.set_title("Within-recording shuffle test", fontsize=9, loc="left")
ax.legend(fontsize=6, frameon=False, loc="upper left")
ax = axs[1, -1]
frac = tun.groupby(["recording", "condition"]).apply(lambda d: (d.p_OSI < args.alpha).mean() * 100)
ax.barh(range(len(frac)), frac.values, color="tab:blue")
ax.axvline(args.alpha * 100, color="k", ls="--", lw=1, label=f"chance ({args.alpha * 100:.0f} %)")
ax.set_yticks(range(len(frac)), [f"{r} {c}" for r, c in frac.index], fontsize=7)
ax.set_xlabel("% of cells with significant orientation tuning")
ax.legend(fontsize=7, frameon=False)
ax.set_title("How many cells are orientation-tuned?", fontsize=9, loc="left")
fig.suptitle(f"Moving-edge orientation preference vs the radiating cone rows (cells with significant OSI, "
             f"p < {args.alpha}, direction-label shuffle; RF not near the stimulus edge). Dashed = evenly spread.",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(OUT / f"edge_orientation_vs_cone_rows.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
df.to_csv(OUT / "edge_orientation_vs_cone_rows_cells.csv", index=False)
print("saved", OUT / "edge_orientation_vs_cone_rows.png")
