"""
RF-vs-cone-row bias split by the local regularity of the cone mosaic (23/11/2026).

Part A (4 recordings with cone_regularity_index: 14/05, 14/08, 15/05, 22/07; crude regions):
  larval patch = RF centre < LARVAL_UM from the optic nerve head (all recordings with RF/row data);
  outside it, cells with cone_crop_valid_fraction > 0.8 are split at their recording's median
  cone_regularity_index (fff_features_v2.csv, extract_cone_regularity.py) into "crystalline" and "disrupted".
  RF orientation vs the ONH-radial row (the same reference in every region).
Part B (15/05 only, rows traced by Laura; rf_vs_measured_rows_15_05_2026_cells.csv from rf_vs_measured_rows_15_05.py):
  regular vs disrupted (median split of cone_regularity_index, outside the larval patch); RF orientation vs
    (1) the traced rows as imaged,
    (2) the ONH-radial rows (simple prediction of the rows), and
    (3) for disrupted cells: the rows PREDICTED from the surrounding intact mosaic = ONH-radial direction + the
        local twist of the traced rows away from radial at the regular cells (axial Gaussian-weighted mean,
        sigma SIGMA_UM).
  If the mosaic was distorted AFTER recording, RFs in disrupted patches should follow (3) better than (1).
Tests: across > along score = mean -cos(2 x angle) (+1 all across, -1 all along), one-sided p from RF axes
shuffled within recording (5000x); fractions along (0-15°) / across (75-90°); 15° bins.

Result (23/11/2026): A: crystalline +0.29 (p < 0.001, n = 113), disrupted -0.03 (n = 115), larval patch -0.16
(n = 19); crystalline - larval p = 0.003, crystalline - disrupted p < 0.001. B: regular vs traced rows +0.46
(p = 0.002; 35 % at 75-90°, 3 % along); disrupted vs traced rows +0.09 (p = 0.32), vs ONH-radial +0.20 (p = 0.056),
vs rows interpolated from the intact mosaic +0.28 (p = 0.013), i.e. disrupted-patch RFs follow the predicted
pre-distortion rows better than the imaged ones (suggesting mostly post-recording distortion in 15/05).

Run from the repo root (~1 min):
    poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_row_bias_by_mosaic_regularity.py"
"""
from pathlib import Path

import numpy as np
import pandas as pd

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_row_angle_prediction")
LARVAL_UM, SIGMA_UM, N_PERM = 200.0, 150.0, 5000
rng = np.random.default_rng(23)


def diff(a, b):
    x = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(x, 180 - x)


def score(a):
    return float(-np.cos(2 * np.deg2rad(np.asarray(a))).mean())


def test(g, row_col, by_recording=True):
    a = diff(g.rf_axis_projected_deg, g[row_col])
    groups = [x for _, x in g.groupby("recording")] if by_recording else [g]
    null = [score(np.concatenate([diff(rng.permutation(x.rf_axis_projected_deg.values), x[row_col].values)
                                  for x in groups])) for _ in range(N_PERM)]
    return dict(n=len(g), score=score(a), p=float(np.mean(np.array(null) >= score(a))), along=(a < 15).mean() * 100,
                across=(a >= 75).mean() * 100, bins=np.histogram(a, np.arange(0, 91, 15))[0] / max(len(a), 1) * 100)


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
print("Part A: regions (cells per recording)")
print(pd.crosstab(d.recording, d.region))
print("regularity index median:", d.groupby("region").cone_regularity_index.median().round(3).to_dict())
for reg in ["crystalline", "disrupted", "larval patch"]:
    report(reg + " vs ONH-radial rows", test(d[d.region == reg], "cone_row_axis_projected_deg"))

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
for _, r in dis.iterrows():
    w = np.exp(-np.hypot(reg.x_projected_um - r.x_projected_um, reg.y_projected_um - r.y_projected_um) ** 2
               / (2 * SIGMA_UM ** 2))
    off = np.degrees(np.angle(np.sum(w * np.exp(2j * np.deg2rad(twist))))) / 2
    pred.append((r.cone_row_axis_projected_deg + off) % 180)
dis["predicted_row_deg"] = pred
print("\nPart B: 15/05, traced rows")
report("regular vs traced rows", test(reg, "measured_row_deg", by_recording=False))
report("regular vs ONH-radial rows", test(reg, "cone_row_axis_projected_deg", by_recording=False))
report("disrupted vs traced rows (as imaged)", test(dis, "measured_row_deg", by_recording=False))
report("disrupted vs ONH-radial rows", test(dis, "cone_row_axis_projected_deg", by_recording=False))
report("disrupted vs rows predicted from intact mosaic", test(dis, "predicted_row_deg", by_recording=False))
print(f"traced vs predicted rows at disrupted cells: median {np.median(diff(dis.measured_row_deg, dis.predicted_row_deg)):.0f}°")
