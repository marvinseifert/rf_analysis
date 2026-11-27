"""
Could the RF-vs-cone-row alignment matter computationally? Tests with existing data (23/11/2026).

H1 matched anti-aliasing: is the labelled-cone lattice (15/05, traced rows) anisotropic enough, and on the right
   scale, for RF elongation to compensate its direction-dependent sampling limit?
H2 population layout: with the measured RF orientations (elongation, size, axis), how much more sensitive is the
   population to RING patterns centred on the optic nerve head (bars around the ONH, varying radially) than to
   SPOKE patterns, compared with RF axes shuffled within recording? Gaussian-ellipse RF model.
H3 looming-type motion: moving-bar preferred motion axis vs own row, crystalline vs disrupted mosaic.
H4 orientation-dependent resolution: RF width along vs across the own row from each cell's ellipse (sigma from
   centre area and elongation), and the implied difference in spatial cut-off.
H5 colour sampling: simulated cone-type mix inside Gaussian RFs of the real sizes, regular row mosaic (Laura's
   layout) vs the same positions with cone types shuffled; spread of the UV+blue fraction between RFs.
Cells: rf_vs_cone_row_angle_all_recordings_cells.csv (quality >= 10, inside the stimulus, not near the edge);
crystalline / disrupted as rf_row_bias_by_mosaic_regularity.py.

Run from the repo root (~5 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/computational_hypotheses.py"
"""
import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from laura_scripts.Alignment_pipeline import alignment_geometry as g
from laura_scripts.Alignment_pipeline.alignment_recordings import (alignment_cache_v2_file, cone_image_file,
                                                                   noise_data_v2_file)

Image.MAX_IMAGE_PIXELS = None
FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
rng = np.random.default_rng(61)
out = {}


def diff(a, b):
    x = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(x, 180 - x)


# ------------------------------------------------------------------ cells
df = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
d = df[(df.quality >= 10) & df.inside_displayed_stimulus.astype(bool) & ~df.near_edge.astype(bool)].copy()
size = {}
for (rec, ch), gg in d.groupby(["recording", "channel"]):
    ds = xr.open_dataset(noise_data_v2_file(rec))
    for c in gg.cell_index:
        size[(rec, c)] = float(ds.center_size_mm2.sel(channel=ch, cell_index=int(c)).values) * 1e6
    ds.close()
d["area"] = [size[(r, c)] for r, c in zip(d.recording, d.cell_index)]
d = d[d.area > 0].copy()
d["elong"] = (1 / d.tilt).clip(lower=1)
# ellipse: area = pi a b (half-axes), a/b = elongation; Gaussian sigma ~ half-axis / 2 (outline ~ 2 sigma)
d["a_half"] = np.sqrt(d.area * d.elong / np.pi)
d["b_half"] = np.sqrt(d.area / (np.pi * d.elong))
feat = []
for rec in d.recording.unique():
    f = Path(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
    if f.exists():
        x = pd.read_csv(f, usecols=["cell_index", "cone_regularity_index", "cone_crop_valid_fraction"])
        x["recording"] = rec
        feat.append(x)
d = d.merge(pd.concat(feat), on=["recording", "cell_index"], how="left")
d["region"] = None
d.loc[d.distance_to_onh_um < 200, "region"] = "larval patch"
m = (d.distance_to_onh_um >= 200) & d.cone_regularity_index.notna() & (d.cone_crop_valid_fraction > 0.8)
med = d[m].groupby("recording").cone_regularity_index.median()
d.loc[m, "region"] = np.where(d.loc[m, "cone_regularity_index"] > d.loc[m, "recording"].map(med), "crystalline",
                              "disrupted")

# ------------------------------------------------------------------ H1: lattice anisotropy (15/05)
REC = "15_05_2026"
with contextlib.redirect_stdout(io.StringIO()):
    geo = g.load_alignment_geometry(alignment_cache_v2_file(REC))
Ai = geo.A_image1_to_projected_um
um_px = float(np.sqrt(abs(np.linalg.det(Ai[:2, :2]))))
img = np.asarray(Image.open(cone_image_file(REC)))[..., 1].astype(np.float32)
sm = gaussian_filter(img - gaussian_filter(img, 6 / um_px), 1 / um_px)
pk = peak_local_max(sm, min_distance=int(5 / um_px), threshold_abs=np.percentile(sm, 80))
cones = (np.c_[pk[:, ::-1], np.ones(len(pk))] @ Ai.T)[:, :2]
rows = pd.read_csv("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Experiments/Imaging/20260515_dragonfly_microscope/"
                   "15_05_2026_cones_rows.csv")
rp = (np.c_[rows.x_pixels, rows.y_pixels, np.ones(len(rows))] @ Ai.T)[:, :2]
rt = cKDTree(rp)
cones = cones[rt.query(cones)[0] <= 15]
rd = np.full(len(cones), np.nan)
for i, p in enumerate(cones):
    idx = rt.query_ball_point(p, 25)
    if len(idx) >= 10:
        X = rp[idx] - rp[idx].mean(0)
        v = np.linalg.eigh(X.T @ X)[1][:, -1]
        rd[i] = np.arctan2(v[1], v[0])
ok = np.isfinite(rd)
cones, rd = cones[ok], rd[ok]
tree = cKDTree(cones)
nn = np.median(tree.query(cones, 2)[0][:, 1])
pairs = np.array(sorted(tree.query_pairs(1.8 * nn)))
vec = cones[pairs[:, 1]] - cones[pairs[:, 0]]
th = rd[pairs[:, 0]]
rx = vec[:, 0] * np.cos(th) + vec[:, 1] * np.sin(th)
ry = -vec[:, 0] * np.sin(th) + vec[:, 1] * np.cos(th)
rx, ry = np.r_[rx, -rx], np.r_[ry, -ry]
along = np.abs(ry) < 3
a1 = np.median(np.abs(rx[along & (np.abs(rx) > 6)]))           # same-row spacing
up = (ry > 6) & (np.abs(rx) < a1)
a2 = np.array([np.median(np.abs(rx[up])), np.median(ry[up])])  # adjacent-row neighbour
A = np.array([[a1, 0.0], [a2[0], a2[1]]])                      # lattice basis (rows)
B = np.linalg.inv(A).T                                         # reciprocal basis (cycles/µm)
recip = np.array([i * B[0] + j * B[1] for i in range(-2, 3) for j in range(-2, 3) if (i, j) != (0, 0)])
angs = np.deg2rad(np.arange(0, 180, 1))
nyq = []
for t in angs:  # first-zone boundary along direction t (Voronoi cell of the reciprocal lattice)
    u = np.array([np.cos(t), np.sin(t)])
    proj = recip @ u
    lim = np.min([(np.dot(k, k) / 2) / pp for k, pp in zip(recip, proj) if pp > 1e-9])
    nyq.append(lim)
nyq = np.array(nyq)
out["H1"] = dict(row_spacing=a1, adjacent=a2, row_separation=a2[1], nyq_min=nyq.min(), nyq_max=nyq.max(),
                 dir_min=np.degrees(angs[nyq.argmin()]), dir_max=np.degrees(angs[nyq.argmax()]),
                 lattice_anisotropy=nyq.max() / nyq.min())
cr = d[d.region == "crystalline"]
out["H1"]["rf_short_width_um"] = float((2 * cr.b_half).median())
out["H1"]["rf_elong"] = float(cr.elong.median())

# ------------------------------------------------------------------ H2: rings vs spokes around the ONH
def sens(sig_a, sig_b, theta, k_dir, f):
    """Gaussian ellipse RF amplitude for a grating with wave vector direction k_dir (deg), frequency f."""
    t = np.deg2rad(k_dir - theta)
    s2 = (sig_a * np.cos(t)) ** 2 + (sig_b * np.sin(t)) ** 2
    return np.exp(-2 * np.pi ** 2 * f ** 2 * s2)


def ring_spoke(sel, theta):
    sa, sb = sel.a_half.values / 2, sel.b_half.values / 2
    f = 1 / (2 * np.pi * np.sqrt(sa * sb))  # spatial frequency near each RF's own cut-off
    radial = sel.cone_row_axis_projected_deg.values  # ONH -> cell direction (= assumed row)
    ring = sens(sa, sb, theta, radial, f)            # rings: bars around the ONH, wave vector radial
    spoke = sens(sa, sb, theta, radial + 90, f)      # spokes: wave vector around the ONH
    return float(np.mean(np.log(ring / spoke)))


h2, H2_NULL = {}, {}
for name, sel in [("all", d), ("crystalline", d[d.region == "crystalline"]), ("disrupted", d[d.region == "disrupted"])]:
    real = ring_spoke(sel, sel.rf_axis_projected_deg.values)
    null = [ring_spoke(sel, sel.groupby("recording").rf_axis_projected_deg.transform(
        lambda x: rng.permutation(x.values)).values) for _ in range(2000)]
    h2[name] = (len(sel), np.exp(real), float(np.mean(np.array(null) >= real)))
    H2_NULL[name] = (np.exp(real), np.exp(np.array(null)))
out["H2"] = h2

# ------------------------------------------------------------------ H3: moving bars, crystalline vs disrupted
bar = pd.read_csv(FIG / "bar_orientation_vs_cone_rows_cells_peak_response.csv")
bar = bar[(bar.osi >= 0.1) & (bar.n_spikes >= 100) & ~bar.near_edge.astype(bool)]
bar = bar.merge(d[["recording", "cell_index", "region"]], on=["recording", "cell_index"], how="left")
h3, H3_NULL = {}, {}
for reg in ["crystalline", "disrupted", "larval patch"]:
    b = bar[bar.region == reg]
    if len(b) < 4:
        h3[reg] = (len(b), np.nan, np.nan, np.nan); continue
    a = diff(b.bar_axis_polar, b.row_axis_polar)
    s = float(-np.cos(2 * np.deg2rad(a)).mean())  # + = motion ACROSS rows, - = motion ALONG rows
    null = [np.concatenate([-np.cos(2 * np.deg2rad(diff(rng.permutation(x.bar_axis_polar.values), x.row_axis_polar.values)))
                            for _, x in b.groupby("recording")]).mean() for _ in range(3000)]
    null = np.array(null)
    h3[reg] = (len(b), s, float(np.mean(null >= s)), float(np.mean(null <= s)))
    H3_NULL[reg] = (s, null)
out["H3"] = h3

# ------------------------------------------------------------------ H4: resolution along vs across rows
def width(ah, bh, theta, direction):
    t = np.deg2rad(direction - theta)
    return 2 * np.sqrt((ah * np.cos(t)) ** 2 + (bh * np.sin(t)) ** 2)


h4, H4_RATIOS = {}, {}
for name, sel in [("all", d), ("crystalline", d[d.region == "crystalline"]), ("disrupted", d[d.region == "disrupted"])]:
    w_along = width(sel.a_half, sel.b_half, sel.rf_axis_projected_deg, sel.cone_row_axis_projected_deg)
    w_across = width(sel.a_half, sel.b_half, sel.rf_axis_projected_deg, sel.cone_row_axis_projected_deg + 90)
    ratio = np.median(w_across / w_along)
    H4_RATIOS[name] = (w_across / w_along).values
    h4[name] = (len(sel), float(np.median(w_along)), float(np.median(w_across)), float(ratio),
                float(np.mean(w_across > w_along) * 100))
out["H4"] = h4

# ------------------------------------------------------------------ H5: colour sampling, regular vs random mosaic
P, HS, HD = 6.0, 6.0, 3.0
period = 2 * HS + 4 * HD
rows_def = [(0.0, [("UV", 0), ("B", P)]), (HS / 2 + HD / 2, [("G", 0), ("R", P)]), (HS / 2 + 1.5 * HD, [("R", 0), ("G", P)]),
            (HS + 2 * HD, [("B", 0), ("UV", P)]), (1.5 * HS + 2.5 * HD, [("R", 0), ("G", P)]),
            (1.5 * HS + 3.5 * HD, [("G", 0), ("R", P)])]
pts, typ = [], []
for j in range(-30, 31):
    for i in range(-60, 61):
        for y, cs in rows_def:
            for t, x in cs:
                pts.append((i * 2 * P + x, j * period + y)); typ.append(t)
pts, typ = np.array(pts), np.array(typ)
short = np.isin(typ, ["UV", "B"]).astype(float)
h5 = {}
for label, area in [("small (10th pct)", d.area.quantile(0.1)), ("median", d.area.median()),
                    ("large (90th pct)", d.area.quantile(0.9))]:
    sig = np.sqrt(area / np.pi) / 2
    frac_reg, frac_rand = [], []
    shuffled = rng.permutation(short)
    for _ in range(300):
        c = rng.uniform(-200, 200, 2)
        w = np.exp(-np.sum((pts - c) ** 2, 1) / (2 * sig ** 2))
        frac_reg.append((w * short).sum() / w.sum())
        frac_rand.append((w * rng.permutation(short)).sum() / w.sum())
    h5[label] = (area, np.std(frac_reg), np.std(frac_rand), np.std(frac_rand) / max(np.std(frac_reg), 1e-12))
out["H5"] = h5
spread = []
for rec in ["14_05_2026", "14_08_2026", "15_05_2026", "22_07_2026"]:
    f = pd.read_csv(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
    wl = [660, 610, 560, 535, 500, 460, 420, 365]
    amp = f[[f"scf_{w}nm_amplitude_hz" for w in wl]].values
    amp0 = amp - np.nanmin(amp, 1, keepdims=True)
    s_, l_ = amp0[:, 5:].mean(1), amp0[:, :3].mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = (s_ - l_) / (s_ + l_)
    rng_ = (np.nanmax(amp, 1) - np.nanmin(amp, 1)) >= 3
    spread.append(np.nanstd(ci[rng_]))
out["H5_observed_colour_index_sd"] = float(np.mean(spread))

# ------------------------------------------------------------------ report
o = out["H1"]
print("H1 anti-aliasing (15/05 labelled-cone lattice)")
print(f"  same-row spacing {o['row_spacing']:.1f} µm; adjacent-row neighbour ({o['adjacent'][0]:.1f}, {o['adjacent'][1]:.1f}) µm "
      f"-> row separation {o['row_separation']:.1f} µm")
print(f"  sampling limit by direction: {o['nyq_min']:.4f}-{o['nyq_max']:.4f} cycles/µm (periods "
      f"{1 / o['nyq_max']:.1f}-{1 / o['nyq_min']:.1f} µm); anisotropy {o['lattice_anisotropy']:.2f}x; coarsest at "
      f"{o['dir_min']:.0f}°, finest at {o['dir_max']:.0f}° from the row")
print(f"  crystalline RFs: median short-axis width {o['rf_short_width_um']:.0f} µm, elongation {o['rf_elong']:.2f}")
print("H2 rings vs spokes around the ONH (population sensitivity ratio, real vs RF axes shuffled within recording)")
for k, (n, r, p) in out["H2"].items():
    print(f"  {k:12s} n = {n}: ring / spoke sensitivity {r:.3f} (p = {p:.3f})")
print("H3 moving-bar preferred motion axis vs own row (+ across, - along)")
for k, (n, s, pa, pl) in out["H3"].items():
    print(f"  {k:12s} n = {n}: score {s:+.3f}, p(across) = {pa:.3f}, p(along) = {pl:.3f}")
print("H4 RF width along vs across the own row")
for k, (n, wa, wc, r, pc) in out["H4"].items():
    print(f"  {k:12s} n = {n}: width along row {wa:.0f} µm, across row {wc:.0f} µm, median ratio across/along {r:.3f}; "
          f"across wider in {pc:.0f} %")
print("H5 colour sampling: SD of the UV+blue fraction between RFs, regular mosaic vs random arrangement")
for k, (area, sr, sx, ratio) in out["H5"].items():
    print(f"  {k:16s} RF area {area:.0f} µm²: regular SD {sr:.4f}, random SD {sx:.4f} (random / regular {ratio:.1f}x)")
print(f"  observed SD of the flash colour index between responsive cells: {out['H5_observed_colour_index_sd']:.2f} "
      "(index range -1..1, includes measurement noise and cell types)")


# ------------------------------------------------------------------ summary figure
BLUE, GREY = "#1e6fd9", "0.6"
fig, axs = plt.subplots(1, 5, figsize=(26, 5.6))
ax = axs[0]
o = out["H1"]
ax.plot(np.degrees(angs), 1 / nyq, color="k", lw=2, label="labelled-cone lattice: finest\nresolvable period (µm)")
ax.axhline(o["rf_short_width_um"], color=BLUE, ls="--", label=f"RF short-axis width ({o['rf_short_width_um']:.0f} µm)")
ax.set_yscale("log")
ax.set_xticks([0, 45, 90, 135, 180], ["0\nalong row", "45", "90\nacross", "135", "180"])
ax.set_xlabel("direction of the pattern's variation vs the row (°)")
ax.set_ylabel("period (µm, log)")
ax.legend(fontsize=8, frameon=False, loc="center right")
ax.set_title(f"H1 anti-aliasing (15/05)\nlattice coarsest ACROSS rows ({o['lattice_anisotropy']:.2f}x), RFs wider across "
             f"(right direction)\nbut RFs ~{o['rf_short_width_um'] * o['nyq_min']:.0f}x coarser than the lattice: "
             "aliasing irrelevant", loc="left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

ax = axs[1]
for k, (name, colr) in enumerate([("crystalline", BLUE), ("disrupted", GREY)]):
    real, null = H2_NULL[name]
    ax.violinplot([null], positions=[k], showextrema=False)
    ax.scatter(k, real, s=120, color=colr, edgecolor="k", zorder=3)
    ax.text(k, real + 0.004, f"{(real - 1) * 100:+.1f} %\np = {h2[name][2]:.3f}", ha="center", fontsize=9)
ax.axhline(1, color="k", lw=0.8)
ax.set_xticks([0, 1], ["crystalline", "disrupted"])
ax.set_ylabel("population sensitivity: rings / spokes\naround the ONH (violin = shuffled RF axes)")
ax.set_title("H2 population layout\nslightly more sensitive to ring patterns\nover crystalline mosaic only",
             loc="left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

ax = axs[2]
for k, (name, colr) in enumerate([("crystalline", BLUE), ("disrupted", GREY)]):
    if name not in H3_NULL:
        continue
    real, null = H3_NULL[name]
    ax.violinplot([null], positions=[k], showextrema=False)
    ax.scatter(k, real, s=120, color=colr, edgecolor="k", zorder=3)
    n, s3, pa, pl = h3[name]
    ax.text(k + 0.12, real, f"n = {n}\np(along) = {pl:.3f}\np(across) = {pa:.3f}", fontsize=8.5, va="center")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks([0, 1], ["crystalline", "disrupted"])
ax.set_ylabel("preferred bar motion vs own row\n(+ across, - along; violin = shuffle)")
ax.set_title("H3 radial (looming-like) motion\ncrystalline cells prefer bars moving ALONG the rows\n"
             "(radially, edges around the ONH)", loc="left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

ax = axs[3]
bins = np.linspace(0.6, 1.6, 26)
for name, colr in [("crystalline", BLUE), ("disrupted", GREY)]:
    r = H4_RATIOS[name]
    ax.hist(r, bins=bins, histtype="step", lw=2, color=colr, density=True,
            label=f"{name}: median {np.median(r):.3f}, {np.mean(r > 1) * 100:.0f} % > 1")
ax.axvline(1, color="k", lw=0.8)
ax.set_xlabel("RF width across the row / along the row")
ax.set_ylabel("density")
ax.legend(fontsize=8, frameon=False)
ax.set_title("H4 orientation-dependent resolution\nRFs ~5 % wider across rows (crystalline only)\n"
             "-> ~5 % finer resolution along rows", loc="left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

ax = axs[4]
labels = list(h5.keys())
x = np.arange(len(labels))
ax.bar(x - 0.2, [h5[k][1] for k in labels], 0.4, color=BLUE, edgecolor="k", label="regular row mosaic (model)")
ax.bar(x + 0.2, [h5[k][2] for k in labels], 0.4, color=GREY, edgecolor="k", label="same cones, random arrangement")
ax.set_xticks(x, [k.split(" ")[0] + "\nRFs" for k in labels])
ax.set_ylabel("SD of the UV+blue cone fraction between RFs")
ax.legend(fontsize=8, frameon=False)
ax.set_title(f"H5 colour sampling (claim 2)\nregular mosaic removes 1-3 % sampling variation;\nobserved colour-index SD "
             f"between cells {out['H5_observed_colour_index_sd']:.2f} (much larger)", loc="left", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Could the RF-vs-cone-row alignment matter computationally? Tests with existing data (crystalline vs "
             "disrupted mosaic: crude regularity split, 4 recordings; rows ONH-radial except H1)", fontsize=12,
             x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"computational_hypotheses_summary.{fmt}", dpi=150 if fmt == "png" else None,
                bbox_inches="tight")
print("saved", FIG / "computational_hypotheses_summary.png")
