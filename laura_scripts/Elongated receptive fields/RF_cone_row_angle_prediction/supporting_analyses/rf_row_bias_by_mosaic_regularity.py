"""
RF-vs-cone-row bias split by the local regularity of the cone mosaic (23/11/2026). Key figure of the
RF-orientation-vs-mosaic result.

Part A (4 recordings with cone_regularity_index: 14/05, 14/08, 15/05, 22/07; crude regions):
  larval patch = RF centre < LARVAL_UM from the optic nerve head (all recordings with RF/row data);
  outside it, cells with cone_crop_valid_fraction > 0.8 are split at their recording's median
  cone_regularity_index (fff_features_v2.csv, extract_cone_regularity.py) into "crystalline" and "disrupted".
  RF orientation vs the ONH-radial row (the same reference in every region).
  Also: the same split within RF-size halves (RF centre area above / below the recording's median), and a
  regression of the score on region adjusted for log RF size, quality, elongation, ONH distance and recording.
Part B (15/05 only, rows traced by Laura; rf_vs_measured_rows_15_05_2026_cells.csv from rf_vs_measured_rows_15_05.py):
  regular vs disrupted (median split of cone_regularity_index, outside the larval patch); RF orientation vs
    (1) the traced rows as imaged,
    (2) the ONH-radial rows (simple prediction of the rows), and
    (3) for disrupted cells: the rows PREDICTED from the surrounding intact mosaic = ONH-radial direction + the
        local twist of the traced rows away from radial at the regular cells (axial Gaussian-weighted mean,
        sigma SIGMA_UM).
  If the mosaic was distorted AFTER recording, RFs in disrupted patches should follow (3) better than (1).
Tests: across > along score = mean -cos(2 x angle) (+1 all across, -1 all along), one-sided p from RF axes
shuffled within recording (N_PERM x); fractions along (0-15°) / across (75-90°); 15° bins with shuffle chance.

Result (23/11/2026): A: crystalline +0.29 (p < 0.001, n = 113), disrupted -0.03 (n = 115), larval patch -0.16
(n = 19); holds within RF-size halves and after adjustment (+0.22, p = 0.004). B: regular vs traced rows +0.46
(p = 0.002; 35 % at 75-90°, 3 % along); disrupted vs traced rows +0.09 (p = 0.32), vs ONH-radial +0.20, vs rows
predicted from the intact mosaic +0.28 (p = 0.01). NB Part B's "disrupted" cells are the less regular half of
cells WITH traced rows nearby (mildly irregular mosaic; 28 of them are "disrupted" in Part A, 8 "crystalline"); the 17
most disrupted 15/05 cells (no traceable rows, regularity 0.06) are not in Part B and show no alignment (-0.30). So
this does NOT show post-recording distortion of strongly disrupted regions: alignment is graded (crystalline > mildly
irregular > strongly disrupted) and the timing of disruption is unresolved.
Output: Box Figures/RF_cone_row_angle_prediction/rf_row_bias_by_mosaic_regularity.png/.pdf

Run from the repo root (~3 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_row_bias_by_mosaic_regularity.py"
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
import xarray as xr

from laura_scripts.Alignment_pipeline.alignment_recordings import noise_data_v2_file

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
LARVAL_UM, SIGMA_UM, N_PERM = 200.0, 150.0, 5000
BINS = np.arange(0, 91, 15)
COL = {"crystalline": "#1e6fd9", "disrupted": "0.55", "larval patch": "#d98c3a", "regular": "#1e6fd9"}
rng = np.random.default_rng(23)


def diff(a, b):
    x = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(x, 180 - x)


def score(a):
    return float(-np.cos(2 * np.deg2rad(np.asarray(a))).mean())


def test(g, row_col, by_recording=True):
    a = diff(g.rf_axis_projected_deg, g[row_col])
    groups = [x for _, x in g.groupby("recording")] if by_recording else [g]
    nulls, nb = [], []
    for _ in range(N_PERM):
        aa = np.concatenate([diff(rng.permutation(x.rf_axis_projected_deg.values), x[row_col].values) for x in groups])
        nulls.append(score(aa)); nb.append(np.histogram(aa, BINS)[0] / len(aa) * 100)
    nulls = np.array(nulls)
    return dict(n=len(g), score=score(a), p=float(np.mean(nulls >= score(a))), null=nulls,
                along=(a < 15).mean() * 100, across=(a >= 75).mean() * 100,
                bins=np.histogram(a, BINS)[0] / max(len(a), 1) * 100, bins_null=np.median(nb, 0),
                bins_lo=np.percentile(nb, 2.5, 0), bins_hi=np.percentile(nb, 97.5, 0))


def report(name, r):
    print(f"  {name:48s} n = {r['n']:3d}: across>along {r['score']:+.3f} (p = {r['p']:.3f}); along {r['along']:.0f} %, "
          f"across {r['across']:.0f} % | 15° bins " + " ".join(f"{v:.0f}" for v in r["bins"]))


# ------------------------------------------------------------------ Part A
df = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
ok = df[(df.quality >= 10) & df.inside_displayed_stimulus.astype(bool) & ~df.near_edge.astype(bool)].copy()
feat = []
for rec in ok.recording.unique():
    f = Path(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
    if f.exists():
        x = pd.read_csv(f, usecols=["cell_index", "cone_regularity_index", "cone_crop_valid_fraction"])
        x["recording"] = rec
        feat.append(x)
d = ok.merge(pd.concat(feat), on=["recording", "cell_index"], how="left")
d["region"] = None
d.loc[d.distance_to_onh_um < LARVAL_UM, "region"] = "larval patch"
out = (d.distance_to_onh_um >= LARVAL_UM) & d.cone_regularity_index.notna() & (d.cone_crop_valid_fraction > 0.8)
med = d[out].groupby("recording").cone_regularity_index.median()
d.loc[out, "region"] = np.where(d.loc[out, "cone_regularity_index"] > d.loc[out, "recording"].map(med),
                                "crystalline", "disrupted")
size = {}
for (rec, ch), g in d.groupby(["recording", "channel"]):
    ds = xr.open_dataset(noise_data_v2_file(rec))
    for c in g.cell_index:
        size[(rec, c)] = float(ds.center_size_mm2.sel(channel=ch, cell_index=int(c)).values) * 1e6
    ds.close()
d["rf_size"] = [size[(r, c)] for r, c in zip(d.recording, d.cell_index)]
d["size_half"] = d.groupby("recording").rf_size.transform(lambda x: np.where(x > x.median(), "large", "small"))
print("Part A: regions (cells per recording)")
print(pd.crosstab(d.recording, d.region))
print("regularity index median:", d.groupby("region").cone_regularity_index.median().round(3).to_dict())
A = {}
for reg in ["crystalline", "disrupted", "larval patch"]:
    A[reg] = test(d[d.region == reg], "cone_row_axis_projected_deg")
    report(reg + " vs ONH-radial rows", A[reg])
S = {}
for reg in ["crystalline", "disrupted"]:
    for h in ["small", "large"]:
        S[(reg, h)] = test(d[(d.region == reg) & (d.size_half == h)], "cone_row_axis_projected_deg")
        report(f"{reg}, {h} RFs", S[(reg, h)])
r = d[d.region.isin(["crystalline", "disrupted"])].copy()
y = -np.cos(2 * np.deg2rad(r.rf_vs_row_difference_deg.values))
X = np.c_[np.ones(len(r)), (r.region == "crystalline").astype(float), np.log(r.rf_size.clip(lower=1)), np.log(r.quality),
          1 / r.tilt, r.distance_to_onh_um / 1000, pd.get_dummies(r.recording, drop_first=True).values.astype(float)]
beta = la.lstsq(X, y, rcond=None)[0]
nullb = []
for _ in range(2000):
    lab = r.groupby("recording").region.transform(lambda v: rng.permutation(v.values))
    X2 = X.copy(); X2[:, 1] = (lab == "crystalline").astype(float)
    nullb.append(la.lstsq(X2, y, rcond=None)[0][1])
p_adj = float(np.mean(np.array(nullb) >= beta[1]))
print(f"  region effect adjusted for RF size, quality, elongation, ONH distance, recording: {beta[1]:+.3f}, p = {p_adj:.4f}")

# ------------------------------------------------------------------ Part B
c = pd.read_csv(FIG / "rf_vs_measured_rows_15_05_2026_cells.csv")
f = pd.read_csv("/mnt/mea_nas/Laura/zebrafish_15_05_2026/Phase_00/fff_analysis/fff_features_v2.csv",
                usecols=["cell_index", "cone_regularity_index", "cone_crop_valid_fraction"])
c = c.merge(f, on="cell_index")
c = c[(c.cone_crop_valid_fraction > 0.8) & (c.distance_to_onh_um >= LARVAL_UM)].copy()
c["region"] = np.where(c.cone_regularity_index > c.cone_regularity_index.median(), "regular", "disrupted")
reg, dis = c[c.region == "regular"], c[c.region == "disrupted"].copy()
twist = ((reg.measured_row_deg - reg.cone_row_axis_projected_deg + 90) % 180 - 90).values
pred = []
for _, rr in dis.iterrows():
    w = np.exp(-np.hypot(reg.x_projected_um - rr.x_projected_um, reg.y_projected_um - rr.y_projected_um) ** 2
               / (2 * SIGMA_UM ** 2))
    pred.append((rr.cone_row_axis_projected_deg + np.degrees(np.angle(np.sum(w * np.exp(2j * np.deg2rad(twist))))) / 2) % 180)
dis["predicted_row_deg"] = pred
print("\nPart B: 15/05, traced rows")
B = {("regular", "traced"): test(reg, "measured_row_deg", False),
     ("regular", "radial"): test(reg, "cone_row_axis_projected_deg", False),
     ("disrupted", "traced"): test(dis, "measured_row_deg", False),
     ("disrupted", "radial"): test(dis, "cone_row_axis_projected_deg", False),
     ("disrupted", "predicted"): test(dis, "predicted_row_deg", False)}
for k, v in B.items():
    report(f"{k[0]} vs {k[1]} rows", v)
print(f"traced vs predicted rows at disrupted cells: median {np.median(diff(dis.measured_row_deg, dis.predicted_row_deg)):.0f}°")

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(22, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 1.25], hspace=0.5, wspace=0.3)
ax = fig.add_subplot(gs[0, 0])
regs = ["crystalline", "disrupted", "larval patch"]
for k, rg in enumerate(regs):
    v = A[rg]
    parts = ax.violinplot([v["null"]], positions=[k], showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("0.85")
    ax.scatter(k, v["score"], s=160, color=COL[rg], edgecolor="k", zorder=3)
    ax.text(k, max(v["score"], np.percentile(v["null"], 99)) + 0.05, f"n = {v['n']}\np = {v['p']:.3f}", ha="center",
            fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(3), ["crystalline", "disrupted", "larval patch\n(< 200 µm from ONH)"])
ax.set_ylabel("across > along score vs own (ONH-radial) row\n+1 all across, -1 all along")
ax.set_title("A  RF orientation vs cone rows by mosaic region\n(dots = data, grey = RF axes shuffled within recording)",
             loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[0, 1])
xc = BINS[:-1] + 7.5
for rg in regs:
    v = A[rg]
    ax.plot(xc, v["bins"], "o-", color=COL[rg], lw=2, label=f"{rg} (n = {v['n']})")
    if rg != "larval patch":
        ax.fill_between(xc, v["bins_lo"], v["bins_hi"], color=COL[rg], alpha=0.1)
ax.set_xticks(BINS); ax.set_xlim(0, 90)
ax.set_xlabel("angle between RF long axis and own row (°), 15° bins\n0 = along, 90 = across")
ax.set_ylabel("% of cells (band = shuffle 95 % range)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("B  Where RF axes fall: crystalline cells avoid the row direction", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[0, 2])
x = np.arange(2)
for k, rg in enumerate(["crystalline", "disrupted"]):
    vals = [S[(rg, h)]["score"] for h in ["small", "large"]]
    ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=COL[rg], edgecolor="k", label=rg)
    for xi, h in zip(x + (k - 0.5) * 0.38, ["small", "large"]):
        v = S[(rg, h)]
        ax.text(xi, max(v["score"], 0) + 0.02, f"n={v['n']}\np={v['p']:.3f}", ha="center", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x, ["small RFs", "large RFs"])
ax.set_ylim(-0.15, 0.55)
ax.set_ylabel("across > along score")
ax.legend(fontsize=8, frameon=False)
ax.set_title(f"C  Not explained by RF size: region effect adjusted for size,\nquality, elongation, ONH distance, "
             f"recording = {beta[1]:+.2f} (p = {p_adj:.3f})", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[1, 0:2])
order = [("regular", "traced", "regular patch vs traced rows"), ("regular", "radial", "regular patch vs ONH-radial rows"),
         ("disrupted", "traced", "disrupted patch vs traced rows (as imaged)"),
         ("disrupted", "radial", "disrupted patch vs ONH-radial rows"),
         ("disrupted", "predicted", "disrupted patch vs rows predicted from intact mosaic")]
yy = np.arange(len(order))[::-1]
for yi, (rg, ref, lab) in zip(yy, order):
    v = B[(rg, ref)]
    ax.barh(yi, v["score"], color=COL[rg] if ref != "predicted" else "#3bb273", edgecolor="k")
    ax.text(max(v["score"], 0) + 0.01, yi, f"{v['score']:+.2f}, p = {v['p']:.3f}; along {v['along']:.0f} %, across "
            f"{v['across']:.0f} % (n = {v['n']})", va="center", fontsize=9)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(yy, [o[2] for o in order])
ax.set_xlim(-0.1, 0.85)
ax.set_xlabel("across > along score")
ax.set_title("D  15/05 with hand-traced rows: regular-patch RFs lie across the REAL rows. 'Disrupted' here = less regular "
             "half of cells WITH traced rows\n(mildly irregular mosaic): they still align, best with rows predicted from "
             "the surroundings. The most disrupted cells (no traceable rows) show none:\ntiming of disruption unresolved",
             loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
ax.text(0, 1, "Summary\n\n"
        "- Over crystalline cone mosaic, RGC RFs avoid lying along\n  the cone rows and tend to lie across them.\n"
        "- Over disrupted mosaic and in the (non-crystalline) larval\n  patch, there is no such bias.\n"
        "- Not explained by RF size, quality, elongation or distance\n  from the optic nerve head.\n"
        "- With measured rows (15/05) the effect is stronger and\n  centred on 'across' (75-90°).\n\n"
        "Caveats: crude median split of cone_regularity_index;\n4 recordings (rows ONH-radial except 15/05); larval\n"
        "patch only 19 cells with lower RF quality; regions\nchosen after seeing the data - needs blind marking\n"
        "and replication.", va="top", fontsize=9.5, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef5fd", edgecolor="#1e6fd9"))
fig.suptitle("RGC RF orientation follows the cone rows only where the cone mosaic is crystalline", fontsize=13,
             x=0.01, ha="left")
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"rf_row_bias_by_mosaic_regularity.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "rf_row_bias_by_mosaic_regularity.png")
