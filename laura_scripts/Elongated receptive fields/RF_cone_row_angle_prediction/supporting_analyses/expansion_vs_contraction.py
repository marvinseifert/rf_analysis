"""
Expansion vs contraction: do direction-selective cells over crystalline mosaic prefer motion AWAY from the optic
nerve head (outward = expanding, looming-like) or TOWARDS it (contraction)? (23/11/2026)

The bar-axis result (computational_hypotheses.py H3) shows a preference for radial motion (along the rows) but not
its sign. Here each cell's preferred DIRECTION (0-360°, polar frame: 0° = up, clockwise, projected frame, same as the
alignment) is compared with the OUTWARD radial direction at its RF centre (ONH -> cell).
  ONH position per recording: least-squares intersection of the cells' radiating row lines (exact for the
  ONH-radial rows of rf_vs_cone_row_angle_all_recordings_cells.csv).
  Moving bar (bar_orientation_vs_cone_rows_cells_peak_response.csv, peak response): direction-selective cells =
  vector DSI >= --min-dsi (no per-cell chance test available), >= 100 spikes, RF not near the edge.
  Moving edges (Box Figures/OSI/edge_tuning_table.csv; 22/07 ON/OFF 13 and 60 fps, 14/08 ON/OFF 13 fps):
  direction-selective = p_DSI < 0.05 (direction-label shuffle).
Regions as rf_row_bias_by_mosaic_regularity.py (crude median split of cone_regularity_index; larval < 200 µm).
Statistic: mean cos(preferred - outward) (+1 all outward, -1 all inward), % outward (|difference| < 90°);
two-sided p from preferred directions shuffled among cells of the same recording (and condition), 5000x.
The eye's optics invert the image through a point, which keeps "outward from a centre": outward on the retina =
expanding around the visual-field point that corresponds to the ONH.

Run from the repo root (~1 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/expansion_vs_contraction.py"
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--min-dsi", type=float, default=0.15)
args = parser.parse_args()

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_row_angle_prediction")
OSI_FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/OSI")
rng = np.random.default_rng(71)

rf = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
onh = {}
for rec, g in rf.groupby("recording"):
    th = np.deg2rad(g.cone_row_axis_projected_deg.values)
    n = np.c_[-np.sin(th), np.cos(th)]
    P = np.c_[g.x_projected_um, g.y_projected_um]
    onh[rec] = np.linalg.lstsq(n, (n * P).sum(1), rcond=None)[0]
rf["outward_polar"] = [np.degrees(np.arctan2(x - onh[r][0], -(y - onh[r][1]))) % 360
                       for r, x, y in zip(rf.recording, rf.x_projected_um, rf.y_projected_um)]
ok = rf[(rf.quality >= 10) & rf.inside_displayed_stimulus.astype(bool) & ~rf.near_edge.astype(bool)].copy()
feat = []
for rec in ok.recording.unique():
    f = Path(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
    if f.exists():
        x = pd.read_csv(f, usecols=["cell_index", "cone_regularity_index", "cone_crop_valid_fraction"])
        x["recording"] = rec
        feat.append(x)
ok = ok.merge(pd.concat(feat), on=["recording", "cell_index"], how="left")
ok["region"] = "other"
ok.loc[ok.distance_to_onh_um < 200, "region"] = "larval patch"
m = (ok.distance_to_onh_um >= 200) & ok.cone_regularity_index.notna() & (ok.cone_crop_valid_fraction > 0.8)
med = ok[m].groupby("recording").cone_regularity_index.median()
ok.loc[m, "region"] = np.where(ok.loc[m, "cone_regularity_index"] > ok.loc[m, "recording"].map(med), "crystalline",
                               "disrupted")
cells = ok[["recording", "cell_index", "outward_polar", "region"]]

bar = pd.read_csv(FIG / "bar_orientation_vs_cone_rows_cells_peak_response.csv")
bar = bar[(bar.dsi >= args.min_dsi) & (bar.n_spikes >= 100) & ~bar.near_edge.astype(bool)]
bar = bar.merge(cells, on=["recording", "cell_index"])
bar["pref"], bar["group"] = bar.bar_direction_polar, bar.recording
edge = pd.read_csv(OSI_FIG / "edge_tuning_table.csv")
edge = edge[edge.p_DSI < 0.05].merge(cells, on=["recording", "cell_index"])
edge["pref"], edge["group"] = edge.pref_direction_polar, edge.recording + "_" + edge.condition


def stat(pref, out):
    return float(np.cos(np.deg2rad(np.asarray(pref) - np.asarray(out))).mean())


def test(sel):
    real = stat(sel.pref, sel.outward_polar)
    null = np.array([stat(np.concatenate([rng.permutation(g.pref.values) for _, g in sel.groupby("group")]),
                          np.concatenate([g.outward_polar.values for _, g in sel.groupby("group")]))
                     for _ in range(5000)])
    d = (sel.pref - sel.outward_polar + 180) % 360 - 180
    return dict(n=len(sel), score=real, p=float(np.mean(np.abs(null - null.mean()) >= abs(real - null.mean()))),
                null_mean=float(null.mean()), pct_out=float((np.abs(d) < 90).mean() * 100), diffs=d.values)


res = {}
for name, tab in [("moving bar", bar), ("moving edges", edge)]:
    for reg in ["crystalline", "disrupted", "all regions"]:
        sel = tab if reg == "all regions" else tab[tab.region == reg]
        if len(sel) < 4:
            print(f"{name:12s} {reg:12s}: n = {len(sel)}, too few"); continue
        r = test(sel)
        res[(name, reg)] = r
        print(f"{name:12s} {reg:12s}: n = {r['n']:3d}, mean cos(pref - outward) {r['score']:+.3f} (shuffle "
              f"{r['null_mean']:+.3f}, p = {r['p']:.3f}); outward {r['pct_out']:.0f} %")

fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(projection="polar"))
for i, name in enumerate(["moving bar", "moving edges"]):
    for j, reg in enumerate(["crystalline", "disrupted", "all regions"]):
        ax = axs[i, j]
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        if (name, reg) not in res:
            ax.set_title(f"{name}, {reg}: too few cells", fontsize=9); continue
        r = res[(name, reg)]
        h, e = np.histogram(np.deg2rad(r["diffs"]), bins=np.linspace(-np.pi, np.pi, 13))
        ax.bar((e[:-1] + e[1:]) / 2, h, width=np.diff(e), color="#1e6fd9" if reg == "crystalline" else "0.6",
               edgecolor="k", alpha=0.8)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ["outward\n(expanding)", "", "inward\n(contracting)", ""],
                      fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(f"{name}, {reg} (n = {r['n']})\nmean cos {r['score']:+.2f} (shuffle {r['null_mean']:+.2f}), "
                     f"p = {r['p']:.3f}; {r['pct_out']:.0f} % outward", fontsize=9, pad=14)
fig.suptitle(f"Expansion vs contraction: preferred motion direction relative to OUTWARD from the optic nerve head "
             f"(bars: DSI >= {args.min_dsi}; edges: DSI shuffle p < 0.05)", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"expansion_vs_contraction.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "expansion_vs_contraction.png")
