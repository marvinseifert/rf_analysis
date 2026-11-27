"""
Do the cells whose RFs lie closest to ACROSS (or ALONG) their cone row, over crystalline mosaic, have particular
properties? (23/11/2026)

Cells: crystalline region as rf_row_bias_by_mosaic_regularity.py (cone_regularity_index above the recording's
median, > 200 µm from the optic nerve head, cone_crop_valid_fraction > 0.8; 14/05, 14/08, 15/05, 22/07).
Groups by RF long axis vs own ONH-radial row: across 75-90°, along 0-15° (few cells: descriptive only), oblique
15-75°. Continuous: alignment = -cos(2 x angle) (+1 across, -1 along).
Properties (fff_features_v2.csv + RF data): RF quality, elongation (1/tilt), RF centre size; contrast steps (ON / OFF
transient and sustained rates, polarity index, transience, contrast sensitivity = response at +/-30 % divided by
+/-100 %, preferred-polarity side); chirp (max-power frequency, threshold frequency, power ratio, max power); single
colour flashes (preferred amplitude, preferred wavelength, colour index as rf_row_bias_by_colour.py); moving bar
OSI / DSI (22/07, 14/05, 15/05, 14/08 where present).
Every property is converted to its rank within the recording (0-1) so recording differences cannot create effects.
Tests (two-sided): across vs oblique (permutation of group labels within recording, 5000x) and the rank correlation
of alignment with the property; Benjamini-Hochberg FDR over properties for each test.

Run from the repo root (~2 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_alignment_vs_properties_crystalline.py"
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from laura_scripts.Alignment_pipeline.alignment_recordings import noise_data_v2_file

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
N_PERM = 5000
rng = np.random.default_rng(41)
WL = [660, 610, 560, 535, 500, 460, 420, 365]

df = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
df = df[(df.quality >= 10) & df.inside_displayed_stimulus.astype(bool) & ~df.near_edge.astype(bool)].copy()
feat = []
for rec in df.recording.unique():
    f = Path(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
    if f.exists():
        x = pd.read_csv(f)
        x["recording"] = rec
        feat.append(x)
feat = pd.concat(feat)
d = df.merge(feat, on=["recording", "cell_index"], how="inner")
d = d[(d.distance_to_onh_um >= 200) & d.cone_regularity_index.notna() & (d.cone_crop_valid_fraction > 0.8)].copy()
d = d[d.cone_regularity_index > d.groupby("recording").cone_regularity_index.transform("median")].copy()

sizes = {}
for (rec, ch), g in d.groupby(["recording", "channel"]):
    ds = xr.open_dataset(noise_data_v2_file(rec))
    for c in g.cell_index:
        sizes[(rec, c)] = float(ds.center_size_mm2.sel(channel=ch, cell_index=int(c)).values) * 1e6
    ds.close()
d["RF centre size (µm²)"] = [sizes[(r, c)] for r, c in zip(d.recording, d.cell_index)]
d["RF quality"] = d.quality
d["RF elongation"] = 1 / d.tilt
with np.errstate(invalid="ignore", divide="ignore"):
    d["contrast sensitivity (30 % / 100 %)"] = ((d.csteps_pos_30_response_hz + d.csteps_neg_30_response_hz)
                                               / (d.csteps_pos_100_response_hz + d.csteps_neg_100_response_hz))
    amp = d[[f"scf_{w}nm_amplitude_hz" for w in WL]].values
    amp0 = amp - np.nanmin(amp, 1, keepdims=True)
    s, l = amp0[:, [5, 6, 7]].mean(1), amp0[:, [0, 1, 2]].mean(1)
    d["colour index (UV/blue +, red/green -)"] = (s - l) / (s + l)
PROPS = {
    "RF quality": "RF quality", "RF elongation": "RF elongation", "RF centre size (µm²)": "RF centre size (µm²)",
    "csteps ON transient (Hz)": "csteps_on_transient_hz", "csteps ON sustained (Hz)": "csteps_on_sustained_hz",
    "csteps OFF transient (Hz)": "csteps_off_transient_hz", "csteps OFF sustained (Hz)": "csteps_off_sustained_hz",
    "csteps polarity index (ON +)": "csteps_polarity_index", "csteps transience": "csteps_compound_transience_index",
    "contrast sensitivity (30 % / 100 %)": "contrast sensitivity (30 % / 100 %)",
    "chirp max-power frequency": "chirp_power_max_frequency", "chirp threshold frequency": "chirp_threshold_frequency",
    "chirp power ratio": "chirp_power_ratio", "chirp max power": "chirp_power_max",
    "scf preferred amplitude (Hz)": "scf_preferred_amplitude_hz", "scf preferred wavelength (nm)": "scf_preferred_wavelength_nm",
    "colour index (UV/blue +, red/green -)": "colour index (UV/blue +, red/green -)",
    "moving bar OSI": "moving_bar_OSI", "moving bar DSI": "moving_bar_DSI",
}
a = d.rf_vs_row_difference_deg.values
d["group"] = np.where(a >= 75, "across", np.where(a < 15, "along", "oblique"))
d["alignment"] = -np.cos(2 * np.deg2rad(a))
print(f"crystalline cells: {len(d)} ({d.recording.value_counts().to_dict()}); groups {d.group.value_counts().to_dict()}")


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty_like(pv); out[o] = np.minimum(r, 1)
    q[ok] = out
    return q


rows = []
for name, col in PROPS.items():
    x = d[["recording", "group", "alignment", col]].dropna().copy()
    if len(x) < 15 or x.group.eq("across").sum() < 4:
        continue
    x["rank"] = x.groupby("recording")[col].rank(pct=True)
    acr, obl = x[x.group == "across"], x[x.group == "oblique"]
    real = acr["rank"].mean() - obl["rank"].mean()
    sub = x[x.group.isin(["across", "oblique"])]
    null = []
    for _ in range(N_PERM):
        lab = sub.groupby("recording").group.transform(lambda v: rng.permutation(v.values)).values
        null.append(sub["rank"][lab == "across"].mean() - sub["rank"][lab == "oblique"].mean())
    null = np.array(null)
    p_grp = float(np.mean(np.abs(null) >= abs(real)))
    r_al = np.corrcoef(x.alignment, x["rank"])[0, 1]
    nullr = [np.corrcoef(x.groupby("recording").alignment.transform(lambda v: rng.permutation(v.values)), x["rank"])[0, 1]
             for _ in range(2000)]
    p_cor = float(np.mean(np.abs(nullr) >= abs(r_al)))
    alg = x[x.group == "along"][col]
    rows.append(dict(property=name, n=len(x), n_across=len(acr), n_along=len(alg),
                     median_across=acr[col].median(), median_oblique=obl[col].median(),
                     median_along=alg.median() if len(alg) else np.nan,
                     rank_diff_across_minus_oblique=real, p_across_vs_oblique=p_grp, corr_alignment=r_al,
                     p_corr=p_cor))
res = pd.DataFrame(rows)
res["q_across_vs_oblique"] = bh(res.p_across_vs_oblique)
res["q_corr"] = bh(res.p_corr)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
res.to_csv(FIG / "rf_alignment_vs_properties_crystalline.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 0.42 * len(res) + 1.5))
y = np.arange(len(res))[::-1]
ax.barh(y, res.rank_diff_across_minus_oblique, color=np.where(res.q_across_vs_oblique < 0.1, "#1e6fd9", "0.7"),
        edgecolor="k")
for yi, (_, r) in zip(y, res.iterrows()):
    ax.text(0.27, yi, f"p = {r.p_across_vs_oblique:.3f} (q = {r.q_across_vs_oblique:.2f}); r = {r.corr_alignment:+.2f}, "
            f"p = {r.p_corr:.3f}", va="center", fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y, res.property, fontsize=9)
ax.set_xlim(-0.25, 0.65)
ax.set_xlabel("mean within-recording rank: ACROSS cells minus OBLIQUE cells\n(> 0: across cells have higher values)")
ax.set_title(f"Crystalline-mosaic cells: do RFs lying across their row (n = {(d.group == 'across').sum()}) differ from "
             f"oblique ones (n = {(d.group == 'oblique').sum()})?\nBlue = FDR q < 0.1; r = rank correlation with "
             "alignment (-cos 2θ, all cells). Along cells (n = "
             f"{(d.group == 'along').sum()}) too few to test.", loc="left", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"rf_alignment_vs_properties_crystalline.{fmt}", dpi=150 if fmt == "png" else None,
                bbox_inches="tight")
print("saved", FIG / "rf_alignment_vs_properties_crystalline.png")
