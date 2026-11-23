"""
Is the "RFs lie across the cone rows" bias specific to some cell types? (23/11/2026)

Cells as summary_figure.py (rf_vs_cone_row_angle_all_recordings_cells.csv: quality >= 10, inside the displayed
stimulus, not near its edge); angle = |RF long axis - own ONH-radial row| (0 along, 90 across).
Groupings (no saved cluster labels exist, so simple, interpretable properties):
  * RF polarity (all 7 recordings): sign of the single-pixel STA extremum 400-20 ms before the spike
    (sta_single_pixel of the RF channel in noise_data_v2.nc, value > 0.5 = light increment = ON)
  * RF centre size (all 7): center_size_mm2 of the RF channel, split at the median of each recording
  * contrast-step polarity (fff_features_v2.csv; 14/08, 22/07, 14/05, 15/05): csteps_polarity_index
    > 0.3 ON, < -0.3 OFF, else ON-OFF
  * transience (same 4): csteps_compound_transience_index, median split per recording
  * chirp temporal-frequency preference (same 4): chirp_power_max_frequency, median split per recording
Per group: % along (0-15°) / oblique (30-60°) / across (75-90°); across-vs-along score = mean -cos(2 x angle)
(+1 all across, -1 all along), p from RF axes shuffled among the group's cells within each recording (one-sided,
"more across than chance"). Difference between groups of a grouping: group labels shuffled within recordings
(two-sided). 5 groupings x up to 3 groups: p < 0.05 / 5 = 0.01 is the corrected threshold for the group-difference
tests (marked *).

Run from the repo root (~2 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_row_bias_by_cell_type.py"
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from laura_scripts.Alignment_pipeline.alignment_recordings import noise_data_v2_file

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_row_angle_prediction")
N_PERM = 5000
ACROSS, ALONG, OBLIQUE = "#1e6fd9", "#e0a800", "0.6"
rng = np.random.default_rng(11)

df = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
df = df[(df.quality >= 10) & df.inside_displayed_stimulus.astype(bool) & ~df.near_edge.astype(bool)].copy()
df["angle"] = df.rf_vs_row_difference_deg

# ---- RF polarity and size from the noise data
pol, size = {}, {}
for (rec, ch), d in df.groupby(["recording", "channel"]):
    ds = xr.open_dataset(noise_data_v2_file(rec))
    t = ds.time_max.values
    win = (t >= -400) & (t <= -20)
    for c in d.cell_index:
        s = ds.sta_single_pixel.sel(channel=ch, cell_index=int(c)).values[win] - 0.5
        pol[(rec, c)] = "ON" if np.isfinite(s).any() and s[np.nanargmax(np.abs(s))] > 0 else (
            "OFF" if np.isfinite(s).any() else np.nan)
        size[(rec, c)] = float(ds.center_size_mm2.sel(channel=ch, cell_index=int(c)).values)
    ds.close()
df["rf_polarity"] = [pol[(r, c)] for r, c in zip(df.recording, df.cell_index)]
df["rf_size"] = [size[(r, c)] for r, c in zip(df.recording, df.cell_index)]
df["rf_size_group"] = df.groupby("recording").rf_size.transform(lambda x: np.where(x > x.median(), "large", "small"))

# ---- full-field features (4 recordings)
feat = []
for rec in df.recording.unique():
    f = Path(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
    if f.exists():
        x = pd.read_csv(f, usecols=["cell_index", "csteps_polarity_index", "csteps_compound_transience_index",
                                    "chirp_power_max_frequency"])
        x["recording"] = rec
        feat.append(x)
df = df.merge(pd.concat(feat), on=["recording", "cell_index"], how="left")
df["csteps_polarity"] = np.select([df.csteps_polarity_index > 0.3, df.csteps_polarity_index < -0.3],
                                  ["ON", "OFF"], default="ON-OFF")
df.loc[df.csteps_polarity_index.isna(), "csteps_polarity"] = np.nan
for col, new, labels in [("csteps_compound_transience_index", "transience", ("sustained", "transient")),
                         ("chirp_power_max_frequency", "chirp_freq", ("low freq.", "high freq."))]:
    df[new] = df.groupby("recording")[col].transform(
        lambda x: pd.Series(np.where(x > x.median(), labels[1], labels[0]), index=x.index).where(x.notna()))

GROUPINGS = [("rf_polarity", "RF polarity (noise STA)", ["ON", "OFF"]),
             ("rf_size_group", "RF centre size", ["small", "large"]),
             ("csteps_polarity", "Contrast-step polarity", ["ON", "ON-OFF", "OFF"]),
             ("transience", "Contrast-step transience", ["sustained", "transient"]),
             ("chirp_freq", "Chirp frequency preference", ["low freq.", "high freq."])]


def score(a):
    return float(-np.cos(2 * np.deg2rad(a)).mean())


FINE_BINS = np.arange(0, 91, 15)  # six 15° bins: 0-15 (along) ... 75-90 (across)


def bin_pct(a):
    return np.histogram(a, bins=FINE_BINS)[0] / len(a) * 100


def shuffle_p(sel):
    """across>along score, its p, and per-15°-bin % with the shuffle's median and 2.5-97.5 % range."""
    real = score(sel.angle)
    groups = [g for _, g in sel.groupby("recording")]
    null, null_bins = [], []
    for _ in range(N_PERM):
        parts = []
        for g in groups:
            rf = rng.permutation(g.rf_axis_projected_deg.values)
            d = np.abs(rf - g.cone_row_axis_projected_deg.values) % 180
            parts.append(np.minimum(d, 180 - d))
        a = np.concatenate(parts)
        null.append(score(a)); null_bins.append(bin_pct(a))
    nb = np.array(null_bins)
    rb = bin_pct(sel.angle.values)
    p_bins_hi = (nb >= rb).mean(0)  # bin fuller than chance
    p_bins_lo = (nb <= rb).mean(0)  # bin emptier than chance
    return real, float(np.mean(np.array(null) >= real)), dict(real=rb, med=np.median(nb, 0),
                                                              lo=np.percentile(nb, 2.5, 0), hi=np.percentile(nb, 97.5, 0),
                                                              p_hi=p_bins_hi, p_lo=p_bins_lo)


def group_difference_p(sel, col, levels):
    sel = sel[sel[col].isin(levels)]
    def stat(lab):
        s = [score(sel.angle[lab == l]) for l in levels if (lab == l).sum() > 0]
        return np.ptp(s)
    real = stat(sel[col].values)
    null = []
    for _ in range(N_PERM // 5):
        lab = sel.groupby("recording")[col].transform(lambda x: rng.permutation(x.values)).values
        null.append(stat(lab))
    return float(np.mean(np.array(null) >= real))


rows, fine = [], {}
s_all, p_all, fine[("All cells", "all")] = shuffle_p(df)
print(f"All cells (n = {len(df)}): score {s_all:+.3f}, p = {p_all:.4f}")
for lo_, r_, m_, ph, pl in zip(FINE_BINS[:-1], fine[("All cells", "all")]["real"], fine[("All cells", "all")]["med"],
                               fine[("All cells", "all")]["p_hi"], fine[("All cells", "all")]["p_lo"]):
    print(f"   {lo_:2d}-{lo_ + 15:2d}°: {r_:5.1f} % (shuffle {m_:4.1f} %; p fuller {ph:.3f}, p emptier {pl:.3f})")
for col, title, levels in GROUPINGS:
    sub = df[df[col].isin(levels)]
    p_diff = group_difference_p(sub, col, levels)
    for l in levels:
        g = sub[sub[col] == l]
        if len(g) < 5:
            continue
        s, p, fine[(title, l)] = shuffle_p(g)
        a = g.angle.values
        rows.append(dict(grouping=title, group=l, n=len(g), n_recordings=g.recording.nunique(), score=s, p=p,
                         along=(a < 15).mean() * 100, oblique=((a >= 30) & (a < 60)).mean() * 100,
                         across=(a >= 75).mean() * 100, p_difference=p_diff))
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
res.to_csv(FIG / "rf_row_bias_by_cell_type.csv", index=False)

# ------------------------------------------------------------------ figure
fig, axs = plt.subplots(3, len(GROUPINGS) + 1, figsize=(4.2 * (len(GROUPINGS) + 1), 13))
# column 0: all cells pooled, fine bins
fb = fine[("All cells", "all")]
ax = axs[2, 0]
xc = FINE_BINS[:-1] + 7.5
ax.fill_between(xc, fb["lo"], fb["hi"], color="0.85", label="shuffle 95 % range")
ax.plot(xc, fb["med"], "k--", lw=1, label="shuffle median")
ax.plot(xc, fb["real"], "o-", color=ACROSS, lw=2, label=f"all cells (n = {len(df)})")
ax.set_xticks(FINE_BINS); ax.set_xlim(0, 90); ax.set_ylim(0, 32)
ax.set_xlabel("RF vs own row (°), 15° bins\n0 = along, 90 = across")
ax.set_ylabel("% of cells")
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.set_title("All cells, fine bins", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
axs[0, 0].axis("off"); axs[1, 0].axis("off")
axs = axs[:, 1:]
for j, (col, title, levels) in enumerate(GROUPINGS):
    r = res[res.grouping == title]
    ax = axs[0, j]
    x = np.arange(len(r))
    ax.bar(x, r.score, color=ACROSS, edgecolor="k")
    for xi, (_, rr) in zip(x, r.iterrows()):
        ax.text(xi, max(rr.score, 0) + 0.01, f"p = {rr.p:.3f}\nn = {rr.n}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, r.group)
    ax.set_ylim(-0.15, 0.45)
    pd_ = r.p_difference.iloc[0] if len(r) else np.nan
    ax.set_title(f"{title}\ngroups differ: p = {pd_:.3f}{' *' if pd_ < 0.01 else ''}", loc="left", fontsize=10)
    if j == 0:
        ax.set_ylabel("across > along score\n(+1 all across, -1 all along)")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1, j]
    w = 0.8 / 3
    for k, (band, colr) in enumerate([("along", ALONG), ("oblique", OBLIQUE), ("across", ACROSS)]):
        ax.bar(x + (k - 1) * w, r[band], w, color=colr, edgecolor="k", label=band if j == 0 else None)
    for k, ch in enumerate([100 / 6, 100 / 3, 100 / 6]):
        ax.scatter(x + (k - 1) * w, [ch] * len(x), marker="_", s=150, color="k", zorder=3)
    ax.set_xticks(x, r.group)
    ax.set_ylim(0, 55)
    if j == 0:
        ax.set_ylabel("% of cells (black ticks = chance)")
        ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
for j, (col, title, levels) in enumerate(GROUPINGS):
    ax = axs[2, j]
    xc = FINE_BINS[:-1] + 7.5
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(levels)))
    for l, colr in zip(levels, cols):
        if (title, l) not in fine:
            continue
        fb = fine[(title, l)]
        ax.fill_between(xc, fb["lo"], fb["hi"], color=colr, alpha=0.12)
        ax.plot(xc, fb["real"], "o-", color=colr, lw=1.8, label=l)
    ax.axhline(100 / 6, color="k", ls=":", lw=0.8)
    ax.set_xticks(FINE_BINS); ax.set_xlim(0, 90); ax.set_ylim(0, 40)
    ax.set_xlabel("RF vs own row (°), 15° bins")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Is the across-row RF bias specific to cell types? Each group: across > along score with p from RF axes "
             "shuffled within recordings (one-sided);\n'groups differ': group labels shuffled within recordings "
             "(* = below 0.01, the threshold after correcting for 5 groupings). RF polarity and size: 7 recordings; "
             "full-field features: 4 recordings.", fontsize=10, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"rf_row_bias_by_cell_type.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "rf_row_bias_by_cell_type.png")
