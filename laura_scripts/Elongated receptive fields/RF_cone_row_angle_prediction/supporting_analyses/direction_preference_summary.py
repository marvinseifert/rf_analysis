"""
Summary figure: preferred motion DIRECTION relative to the optic nerve head (ONH) - expanding vs contracting -
by mosaic region, and in the recordings with a central ONH. (23/11/2026)

Cells: moving-bar direction-selective cells (bar_orientation_vs_cone_rows_cells_peak_response.csv, vector DSI >=
MIN_DSI, >= 100 spikes, RF not near the stimulus edge; no per-cell chance test for direction selectivity).
Outward direction = ONH -> RF centre (ONH = least-squares intersection of the assumed radiating rows).
Score = cos(preferred - outward): +1 outward (expanding), -1 inward (contracting).
Regions as rf_row_bias_by_mosaic_regularity.py (crude cone_regularity_index median split; larval < 200 µm).
Baselines: preferred directions shuffled among cells of the same recording (keeps each recording's shared direction
and ONH geometry). Region comparison: region labels shuffled within recording (both groups share the same
recording-level biases, so this is the most robust comparison).
Central ONH (14/05, 14/08): outward directions point every way, so a shared preferred direction cannot fake
inward/outward; maps show each cell's preferred direction around the ONH.

Results (23/11/2026): crystalline -0.62 (baseline -0.47, p = 0.001, 8 % outward, n = 37); disrupted -0.27 (baseline
-0.17, p = 0.19, n = 30); larval 3 cells; crystalline - disrupted -0.35, p = 0.005. Central ONH: 14/05 12 of 13 cells
inward (-0.70 vs -0.11, p < 0.001); 14/08 -0.26 (p = 0.16).

Run from the repo root (~1 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/direction_preference_summary.py"
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
MIN_DSI, N_PERM = 0.15, 5000
COL = {"crystalline": "#1e6fd9", "disrupted": "0.55", "larval patch": "#d98c3a"}
rng = np.random.default_rng(91)

rf = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
onh = {}
for rec, g in rf.groupby("recording"):
    th = np.deg2rad(g.cone_row_axis_projected_deg.values)
    n = np.c_[-np.sin(th), np.cos(th)]
    P = np.c_[g.x_projected_um, g.y_projected_um]
    onh[rec] = np.linalg.lstsq(n, (n * P).sum(1), rcond=None)[0]
rf["outward"] = [np.degrees(np.arctan2(x - onh[r][0], -(y - onh[r][1]))) % 360
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
ok["region"] = "unclassified"
ok.loc[ok.distance_to_onh_um < 200, "region"] = "larval patch"
m = (ok.distance_to_onh_um >= 200) & ok.cone_regularity_index.notna() & (ok.cone_crop_valid_fraction > 0.8)
med = ok[m].groupby("recording").cone_regularity_index.median()
ok.loc[m, "region"] = np.where(ok.loc[m, "cone_regularity_index"] > ok.loc[m, "recording"].map(med), "crystalline",
                               "disrupted")
bar = pd.read_csv(FIG / "bar_orientation_vs_cone_rows_cells_peak_response.csv")
bar = bar[(bar.dsi >= MIN_DSI) & (bar.n_spikes >= 100) & ~bar.near_edge.astype(bool)][
    ["recording", "cell_index", "bar_direction_polar", "dsi"]]
b = bar.merge(ok[["recording", "cell_index", "outward", "region", "x_projected_um", "y_projected_um"]],
              on=["recording", "cell_index"])
b["cosd"] = np.cos(np.deg2rad(b.bar_direction_polar - b.outward))


def baseline(g):
    null = np.array([np.concatenate([np.cos(np.deg2rad(rng.permutation(x.bar_direction_polar.values) - x.outward.values))
                                     for _, x in g.groupby("recording")]).mean() for _ in range(N_PERM)])
    r = g.cosd.mean()
    return dict(n=len(g), score=float(r), null=null, p=float(np.mean(np.abs(null - null.mean()) >= abs(r - null.mean()))),
                pct_out=float((g.cosd > 0).mean() * 100))


REG = {reg: baseline(b[b.region == reg]) for reg in ["crystalline", "disrupted", "larval patch"] if (b.region == reg).sum() >= 3}
cd = b[b.region.isin(["crystalline", "disrupted"])]
diff_real = cd[cd.region == "crystalline"].cosd.mean() - cd[cd.region == "disrupted"].cosd.mean()
dn = []
for _ in range(N_PERM):
    lab = cd.groupby("recording").region.transform(lambda v: rng.permutation(v.values)).values
    dn.append(cd.cosd[lab == "crystalline"].mean() - cd.cosd[lab == "disrupted"].mean())
p_diff = float(np.mean(np.abs(dn) >= abs(diff_real)))
CEN = {rec: baseline(b[b.recording == rec]) for rec in ["14_05_2026", "14_08_2026"]}
for k, v in {**REG, **CEN}.items():
    print(f"{k:13s} n = {v['n']:3d}: mean cos {v['score']:+.2f} (baseline {v['null'].mean():+.2f}, p = {v['p']:.3f}); "
          f"outward {v['pct_out']:.0f} %")
print(f"crystalline - disrupted: {diff_real:+.2f}, p = {p_diff:.3f}")

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(22, 9.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.25, 1.25, 1.05], wspace=0.3)
ax = fig.add_subplot(gs[0])
for k, reg in enumerate(REG):
    v = REG[reg]
    parts = ax.violinplot([v["null"]], positions=[k], showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("0.85")
    ax.scatter(k, v["score"], s=170, color=COL[reg], edgecolor="k", zorder=3)
    ax.text(k, 0.55, f"n = {v['n']}\np = {v['p']:.3f}\n{v['pct_out']:.0f} % outward", ha="center", fontsize=8.5)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylim(-1, 0.85)
ax.set_xticks(range(len(REG)), list(REG))
ax.set_ylabel("mean cos(preferred direction - outward)\n+1 outward (expanding), -1 inward (contracting)")
ax.set_title(f"A  Moving-bar DS cells by mosaic region (all recordings)\n(dot = data, grey = shuffle baseline)\n"
             f"crystalline vs disrupted: {diff_real:+.2f}, p = {p_diff:.3f}", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

for i, rec in enumerate(["14_05_2026", "14_08_2026"]):
    ax = fig.add_subplot(gs[1 + i])
    allc = ok[ok.recording == rec]
    ax.scatter(allc.x_projected_um, allc.y_projected_um, s=8, color="0.85", zorder=1)
    for th in np.linspace(0, np.pi, 18, endpoint=False):
        ax.plot(onh[rec][0] + 1500 * np.cos(th) * np.array([-1, 1]), onh[rec][1] + 1500 * np.sin(th) * np.array([-1, 1]),
                color="#e08040", lw=0.5, alpha=0.5, zorder=0)
    ax.plot(*onh[rec], "o", color="k", ms=12, zorder=5)
    g = b[b.recording == rec]
    for _, c in g.iterrows():
        a = np.deg2rad(c.bar_direction_polar)
        colr = "#1e6fd9" if c.cosd > 0 else "#d03030"
        mk = {"crystalline": "o", "disrupted": "s", "larval patch": "^"}.get(c.region, "x")
        ax.scatter(c.x_projected_um, c.y_projected_um, marker=mk, s=36, facecolor="white", edgecolor=colr, zorder=3)
        ax.annotate("", xy=(c.x_projected_um + 80 * np.sin(a), c.y_projected_um - 80 * np.cos(a)),
                    xytext=(c.x_projected_um, c.y_projected_um), arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.8),
                    zorder=4)
    v = CEN[rec]
    xs = np.r_[allc.x_projected_um, onh[rec][0]]; ys = np.r_[allc.y_projected_um, onh[rec][1]]
    ax.set_xlim(xs.min() - 150, xs.max() + 150); ax.set_ylim(ys.max() + 150, ys.min() - 150)
    ax.set_aspect("equal")
    ax.set_xlabel("µm (projected)")
    ax.set_title(f"{'BC'[i]}  {rec.replace('_', '/')}: ONH (black dot) near the array centre\n"
                 f"DS cells: n = {v['n']}, {100 - v['pct_out']:.0f} % inward\nmean cos {v['score']:+.2f} "
                 f"(baseline {v['null'].mean():+.2f}, p = {v['p']:.3f})", loc="left", fontsize=9.5)
    if i == 0:
        ax.text(0, -0.13, "arrows = preferred bar direction; red = inward (contracting), blue = outward (expanding)\n"
                "o crystalline, s disrupted, ^ larval patch, x unclassified; orange = assumed radiating rows; "
                "grey = all RFs", transform=ax.transAxes, fontsize=8.5, va="top")

ax = fig.add_subplot(gs[3]); ax.axis("off")
ax.text(0, 1, "Summary (tentative)\n\n"
        "- Direction-selective cells over crystalline mosaic\n  prefer motion along the rows TOWARDS the ONH\n"
        "  (inward = contracting around the ONH's\n  visual-field point).\n"
        "- Crystalline cells are more inward than disrupted\n  cells in the same recordings (p = 0.005): the\n"
        "  preference goes with mosaic order.\n"
        "- Where the ONH is central (14/05), a shared\n  preferred direction cannot fake this: 12 of 13\n"
        "  DS cells point inward (p < 0.001). 14/08 leans\n  the same way (n.s.).\n\n"
        "Caveats: moving-bar preferred directions only (no\ncontracting stimulus shown); DSI >= 0.15 without a\n"
        "per-cell chance test; small n (37 crystalline;\n13-14 per central recording); larval patch 3 cells;\n"
        "many analyses today - replicate.\n\n"
        "Possible meaning: self-motion / position holding\n(drift away from that direction), tracking receding\n"
        "objects, or a developmental by-product (growth\nat the margin); not mutually exclusive.\n\n"
        "Direct test: contracting vs expanding (and rotating)\nstimuli centred on the ONH's visual-field point,\n"
        "with crystalline and larval-patch cells on the array.",
        va="top", fontsize=9.3, linespacing=1.3, bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef5fd",
                                                          edgecolor="#1e6fd9"))
fig.suptitle("Preferred motion direction relative to the optic nerve head: contracting (inward) preference over "
             "crystalline mosaic", fontsize=13, x=0.01, ha="left")
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"direction_preference_summary.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "direction_preference_summary.png")
