"""
Expansion vs contraction in the recordings where the optic nerve head (ONH) sits inside/near the middle of the array
(14/05: row spread 0.08; 14/08: 0.16), so outward directions point every way and a shared preferred direction
cannot masquerade as inward or outward. (23/11/2026)

Same measures as expansion_vs_contraction.py (rows assumed to radiate from the ONH): preferred motion direction
(moving bar: vector DSI >= --min-dsi; edges (14/08 only): DSI shuffle p < 0.05) vs the outward direction ONH -> RF
centre; mean cos(pref - outward) (+1 outward/expanding, -1 inward/contracting), shuffle of preferred directions
among the recording's cells. Map: each direction-selective cell's preferred direction drawn as an arrow at its RF
centre around the ONH (blue = outward, red = inward), crystalline / disrupted / larval patch marked.

Run from the repo root (~1 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/expansion_vs_contraction_central_onh.py"
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--min-dsi", type=float, default=0.15)
args = parser.parse_args()

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
OSI_FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/OSI")
RECS = ["14_05_2026", "14_08_2026"]
rng = np.random.default_rng(81)

rf = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
rf = rf[rf.recording.isin(RECS)].copy()
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
for rec in RECS:
    x = pd.read_csv(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv",
                    usecols=["cell_index", "cone_regularity_index", "cone_crop_valid_fraction"])
    x["recording"] = rec
    feat.append(x)
ok = ok.merge(pd.concat(feat), on=["recording", "cell_index"], how="left")
ok["region"] = "other"
ok.loc[ok.distance_to_onh_um < 200, "region"] = "larval patch"
m = (ok.distance_to_onh_um >= 200) & ok.cone_regularity_index.notna() & (ok.cone_crop_valid_fraction > 0.8)
med = ok[m].groupby("recording").cone_regularity_index.median()
ok.loc[m, "region"] = np.where(ok.loc[m, "cone_regularity_index"] > ok.loc[m, "recording"].map(med), "crystalline",
                               "disrupted")
cells = ok[["recording", "cell_index", "outward_polar", "region", "x_projected_um", "y_projected_um", "distance_to_onh_um"]]

bar = pd.read_csv(FIG / "bar_orientation_vs_cone_rows_cells_peak_response.csv")
bar = bar[(bar.dsi >= args.min_dsi) & (bar.n_spikes >= 100) & ~bar.near_edge.astype(bool)]
bar = bar[["recording", "cell_index", "bar_direction_polar", "dsi"]].merge(cells, on=["recording", "cell_index"])
bar = bar.rename(columns={"bar_direction_polar": "pref"})
bar["stim"] = "moving bar"
edge = pd.read_csv(OSI_FIG / "edge_tuning_table.csv")
edge = edge[(edge.p_DSI < 0.05) & (edge.recording == "14_08_2026")]
edge = edge[["recording", "cell_index", "pref_direction_polar", "DSI", "condition"]].merge(cells, on=["recording", "cell_index"])
edge = edge.rename(columns={"pref_direction_polar": "pref", "DSI": "dsi"})
edge["stim"] = "edges " + edge.condition
allc = pd.concat([bar, edge], ignore_index=True)
allc["cosd"] = np.cos(np.deg2rad(allc.pref - allc.outward_polar))


def test(sel):
    real = sel.cosd.mean()
    null = np.array([np.cos(np.deg2rad(rng.permutation(sel.pref.values) - sel.outward_polar.values)).mean()
                     for _ in range(5000)])
    return len(sel), float(real), float(null.mean()), float(np.mean(np.abs(null - null.mean()) >= abs(real - null.mean()))), \
        float((sel.cosd > 0).mean() * 100)


rows = []
for (rec, stim), g in allc.groupby(["recording", "stim"]):
    for reg in ["all", "crystalline", "disrupted"]:
        sel = g if reg == "all" else g[g.region == reg]
        if len(sel) < 4:
            continue
        n, real, nm, p, pct = test(sel)
        rows.append(dict(recording=rec, stimulus=stim, region=reg, n=n, mean_cos=real, shuffle_mean=nm, p=p, pct_outward=pct))
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print({r: tuple(np.round(o, 0)) for r, o in onh.items()}, "(ONH, projected µm; array centre = 0, 0)")
print(res.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 3, figsize=(21, 7.5))
for ax, (rec, stim) in zip(axs, [("14_05_2026", "moving bar"), ("14_08_2026", "moving bar"), ("14_08_2026", "edges")]):
    g = allc[(allc.recording == rec) & (allc.stim.str.startswith(stim))]
    allcells = ok[ok.recording == rec]
    ax.scatter(allcells.x_projected_um, allcells.y_projected_um, s=8, color="0.85", zorder=1)
    for th in np.linspace(0, np.pi, 18, endpoint=False):
        ax.plot(onh[rec][0] + 1500 * np.cos(th) * np.array([-1, 1]), onh[rec][1] + 1500 * np.sin(th) * np.array([-1, 1]),
                color="#e08040", lw=0.5, alpha=0.5, zorder=0)
    ax.plot(*onh[rec], "o", color="k", ms=12, zorder=5)
    L = 70
    for _, c in g.iterrows():
        a = np.deg2rad(c.pref)
        dx, dy = L * np.sin(a), -L * np.cos(a)  # polar frame: 0 = up (-y), clockwise
        colr = "#1e6fd9" if c.cosd > 0 else "#d03030"
        mk = {"crystalline": "o", "disrupted": "s", "larval patch": "^"}.get(c.region, "x")
        ax.scatter(c.x_projected_um, c.y_projected_um, marker=mk, s=30, facecolor="white", edgecolor=colr, zorder=3)
        ax.annotate("", xy=(c.x_projected_um + dx, c.y_projected_um + dy), xytext=(c.x_projected_um, c.y_projected_um),
                    arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.6), zorder=4)
    r = res[(res.recording == rec) & (res.stimulus.str.startswith(stim)) & (res.region == "all")]
    txt = "; ".join(f"{rr.stimulus}: n={rr.n}, cos {rr.mean_cos:+.2f} (shuf {rr.shuffle_mean:+.2f}, p={rr.p:.2f}), "
                    f"{rr.pct_outward:.0f}% out" for _, rr in r.iterrows())
    xs = np.r_[allcells.x_projected_um, onh[rec][0]]; ys = np.r_[allcells.y_projected_um, onh[rec][1]]
    ax.set_xlim(xs.min() - 150, xs.max() + 150); ax.set_ylim(ys.max() + 150, ys.min() - 150)
    ax.set_aspect("equal")
    ax.set_title(f"{rec.replace('_', '/')}, {stim}: preferred direction of DS cells\n(blue = outward/expanding, "
                 f"red = inward/contracting; o crystalline, s disrupted, ^ larval)\n{txt}", loc="left", fontsize=8.5)
    ax.set_xlabel("µm (projected)")
fig.suptitle("Expansion vs contraction where the optic nerve head (black dot) is near the middle of the array; "
             "orange = assumed radiating rows", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"expansion_vs_contraction_central_onh.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "expansion_vs_contraction_central_onh.png")
