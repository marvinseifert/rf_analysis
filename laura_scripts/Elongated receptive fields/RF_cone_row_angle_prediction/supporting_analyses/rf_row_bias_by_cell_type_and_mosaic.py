"""
Cell-type groupings of the RF-vs-cone-row bias WITHIN mosaic regions (23/11/2026).

Regions as rf_row_bias_by_mosaic_regularity.py (crude): larval patch = RF centre < 200 µm from the optic nerve head;
outside it, cone_regularity_index above / below the recording's median = crystalline / disrupted (cells with
cone_crop_valid_fraction > 0.8; recordings with the index: 14/05, 14/08, 15/05, 22/07).
Groupings as rf_row_bias_by_cell_type.py: RF polarity (noise STA), RF centre size (median split per recording),
contrast-step polarity (ON / ON-OFF / OFF), transience and chirp frequency preference (median splits per recording).
Per region x group: across > along score = mean -cos(2 x angle) vs the ONH-radial row (+1 all across, -1 all along),
one-sided p from RF axes shuffled within recording; % along (0-15°) and across (75-90°); group difference p from
group labels shuffled within recording (two-sided). The larval patch (19 cells) is too small to split.

Run from the repo root (~3 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_row_bias_by_cell_type_and_mosaic.py"
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from laura_scripts.Alignment_pipeline.alignment_recordings import noise_data_v2_file

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_row_angle_prediction")
LARVAL_UM, N_PERM = 200.0, 3000
rng = np.random.default_rng(31)

df = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
df = df[(df.quality >= 10) & df.inside_displayed_stimulus.astype(bool) & ~df.near_edge.astype(bool)].copy()
feat = []
for rec in df.recording.unique():
    f = Path(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
    if f.exists():
        x = pd.read_csv(f, usecols=["cell_index", "cone_regularity_index", "cone_crop_valid_fraction",
                                    "csteps_polarity_index", "csteps_compound_transience_index",
                                    "chirp_power_max_frequency"])
        x["recording"] = rec
        feat.append(x)
d = df.merge(pd.concat(feat), on=["recording", "cell_index"], how="left")
d["region"] = None
d.loc[d.distance_to_onh_um < LARVAL_UM, "region"] = "larval patch"
out = (d.distance_to_onh_um >= LARVAL_UM) & d.cone_regularity_index.notna() & (d.cone_crop_valid_fraction > 0.8)
med = d[out].groupby("recording").cone_regularity_index.median()
d.loc[out, "region"] = np.where(d.loc[out, "cone_regularity_index"] > d.loc[out, "recording"].map(med),
                                "crystalline", "disrupted")
d = d[d.region.notna()].copy()

# RF polarity and size from the noise data
pol, size = {}, {}
for (rec, ch), g in d.groupby(["recording", "channel"]):
    ds = xr.open_dataset(noise_data_v2_file(rec))
    t = ds.time_max.values
    win = (t >= -400) & (t <= -20)
    for c in g.cell_index:
        s = ds.sta_single_pixel.sel(channel=ch, cell_index=int(c)).values[win] - 0.5
        pol[(rec, c)] = ("ON" if s[np.nanargmax(np.abs(s))] > 0 else "OFF") if np.isfinite(s).any() else None
        size[(rec, c)] = float(ds.center_size_mm2.sel(channel=ch, cell_index=int(c)).values)
    ds.close()
d["rf_polarity"] = [pol[(r, c)] for r, c in zip(d.recording, d.cell_index)]
d["rf_size"] = [size[(r, c)] for r, c in zip(d.recording, d.cell_index)]
d["rf_size_group"] = d.groupby("recording").rf_size.transform(lambda x: np.where(x > x.median(), "large", "small"))
d["csteps_polarity"] = np.select([d.csteps_polarity_index > 0.3, d.csteps_polarity_index < -0.3], ["ON", "OFF"],
                                 default="ON-OFF")
d.loc[d.csteps_polarity_index.isna(), "csteps_polarity"] = None
for col, new, labels in [("csteps_compound_transience_index", "transience", ("sustained", "transient")),
                         ("chirp_power_max_frequency", "chirp_freq", ("low freq.", "high freq."))]:
    d[new] = d.groupby("recording")[col].transform(
        lambda x: pd.Series(np.where(x > x.median(), labels[1], labels[0]), index=x.index).where(x.notna()))

GROUPINGS = [("rf_polarity", "RF polarity", ["ON", "OFF"]), ("rf_size_group", "RF size", ["small", "large"]),
             ("csteps_polarity", "Contrast-step polarity", ["ON", "ON-OFF", "OFF"]),
             ("transience", "Transience", ["sustained", "transient"]),
             ("chirp_freq", "Chirp frequency", ["low freq.", "high freq."])]


def diff(a, b):
    x = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(x, 180 - x)


def score(a):
    return float(-np.cos(2 * np.deg2rad(np.asarray(a))).mean()) if len(a) else np.nan


def shuffle_p(g):
    s = score(g.rf_vs_row_difference_deg)
    null = [score(np.concatenate([diff(rng.permutation(x.rf_axis_projected_deg.values), x.cone_row_axis_projected_deg.values)
                                  for _, x in g.groupby("recording")])) for _ in range(N_PERM)]
    return s, float(np.mean(np.array(null) >= s))


def group_diff(g, col, levels):
    g = g[g[col].isin(levels)]
    def stat(lab):
        v = [score(g.rf_vs_row_difference_deg[lab == lv]) for lv in levels if (lab == lv).sum() >= 3]
        return np.ptp(v) if len(v) > 1 else np.nan
    real = stat(g[col].values)
    null = [stat(g.groupby("recording")[col].transform(lambda x: rng.permutation(x.values)).values) for _ in range(1000)]
    return float(np.nanmean(np.array(null) >= real)) if np.isfinite(real) else np.nan


rows = []
for region in ["crystalline", "disrupted"]:
    R = d[d.region == region]
    for col, title, levels in GROUPINGS:
        pd_ = group_diff(R, col, levels)
        for lv in levels:
            g = R[R[col] == lv]
            if len(g) < 4:
                rows.append(dict(region=region, grouping=title, group=lv, n=len(g))); continue
            s, p = shuffle_p(g)
            a = g.rf_vs_row_difference_deg.values
            rows.append(dict(region=region, grouping=title, group=lv, n=len(g), score=s, p=p,
                             along=(a < 15).mean() * 100, across=(a >= 75).mean() * 100, p_difference=pd_))
L = d[d.region == "larval patch"]
s, p = shuffle_p(L)
rows.append(dict(region="larval patch", grouping="all", group="all", n=len(L), score=s, p=p,
                 along=(L.rf_vs_row_difference_deg < 15).mean() * 100, across=(L.rf_vs_row_difference_deg >= 75).mean() * 100))
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("\nlarval patch composition:", L.rf_polarity.value_counts().to_dict(), L.csteps_polarity.value_counts().to_dict())
res.to_csv(FIG / "rf_row_bias_by_cell_type_and_mosaic.csv", index=False)

fig, axs = plt.subplots(1, len(GROUPINGS), figsize=(4.3 * len(GROUPINGS), 5.2), sharey=True)
for ax, (col, title, levels) in zip(axs, GROUPINGS):
    for k, (region, colr) in enumerate([("crystalline", "#1e6fd9"), ("disrupted", "0.6")]):
        r = res[(res.region == region) & (res.grouping == title)].set_index("group").reindex(levels)
        x = np.arange(len(levels)) + (k - 0.5) * 0.38
        ax.bar(x, r.score.fillna(0), 0.38, color=colr, edgecolor="k", label=region)
        for xi, (lv, rr) in zip(x, r.iterrows()):
            if pd.notna(rr.get("p", np.nan)):
                ax.text(xi, max(rr.score, 0) + 0.02, f"p={rr.p:.2f}\nn={int(rr.n)}", ha="center", fontsize=7)
            else:
                ax.text(xi, 0.02, f"n={int(rr.n) if pd.notna(rr.n) else 0}", ha="center", fontsize=7)
    pc = res[(res.region == "crystalline") & (res.grouping == title)].p_difference.dropna()
    ax.set_title(f"{title}\ngroups differ in crystalline: p = {pc.iloc[0]:.2f}" if len(pc) else title,
                 loc="left", fontsize=10)
    ax.set_xticks(np.arange(len(levels)), levels)
    ax.axhline(0, color="k", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("across > along score (vs ONH-radial row)")
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle("Cell-type groupings within crystalline vs disrupted mosaic (4 recordings; crude regularity split). "
             f"Larval patch (n = {len(L)}): score {s:+.2f}, p = {p:.2f}.", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"rf_row_bias_by_cell_type_and_mosaic.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "rf_row_bias_by_cell_type_and_mosaic.png")
