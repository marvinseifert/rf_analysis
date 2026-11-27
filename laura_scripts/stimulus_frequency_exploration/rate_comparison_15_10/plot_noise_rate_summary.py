"""Summary figure (15/10/2026): RF quality at 5/10/15/20 Hz noise, corrected for
  (1) the quality index's bias towards slow noise: quality is divided by the highest quality
      reached by time-shifted spikes at that rate (chance max: 14.17 / 9.46 / 10.17 / 9.94),
      so 1 = best chance value, at every rate;
  (2) retina health: per cell, health change = mean change in response reliability (quality
      index of scf, csteps, chirp) from before the noise (stim 6-8) to after it (13-15). For each
      rate, log(quality / quality at 5 Hz) is fitted against health change across RF cells
      (Theil-Sen); the health-related part (slope x health change) is removed, i.e. every cell
      is shown as if its responses had not changed. 5 Hz (first block, ~15 min after the
      'before' stimuli) is the reference and is not changed.
and linked to the chirp just before the noise (stim 8): highest frequency followed and
frequency followed most strongly (largest phase-locking amplitude among followed pieces).

Cells: RF above chance at >= 1 rate, follow the chirp, health known.
Inputs (stimulus_frequency_exploration/results/): quality_null_by_rate.csv, chirp_max_following_frequency.csv,
retina_health_bracket_stimuli.csv.
Also writes results/noise_rate_corrected_quality.csv (read by rf_timecourse_by_noise_rate.py).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parents[1] / "results"   # stimulus_frequency_exploration/results
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/noise_rate_comparison/noise_rate_summary.png")
RATES = [5, 10, 15, 20]
CHANCE_MAX = {5: 14.17, 10: 9.46, 15: 10.17, 20: 9.94}
GROUP_COL = ["#08519c", "#4292c6", "#41ab5d"]     # blue -> green
rng = np.random.default_rng(0)

# --- RF quality relative to chance -----------------------------------------------------------
q = pd.read_csv(HERE / "quality_null_by_rate.csv")
q["qn"] = q.q_real / q.rate.map(CHANCE_MAX)
qn = q.pivot(index="cell", columns="rate", values="qn").dropna()

# --- health change -------------------------------------------------------------------------
h = pd.read_csv(HERE / "retina_health_bracket_stimuli.csv")
p = h.pivot_table(index="cell", columns="stim", values="qi")
health = ((p[13] - p[6]) + (p[14] - p[7]) + (p[15] - p[8])) / 3

# --- chirp ---------------------------------------------------------------------------------
c = pd.read_csv(HERE / "chirp_max_following_frequency.csv")
c8 = c[c.chirp == 8].set_index("cell")
amp = [k for k in c8 if k.startswith("amp_")]
sig = [k for k in c8 if k.startswith("sig_")]
F = np.array([float(k[4:]) for k in amp])
S = c8[sig].values.astype(bool)
c8["best_hz"] = np.where(S.any(1), F[np.argmax(np.where(S, c8[amp].values, -1), 1)], np.nan)

cells = (qn.index[(qn > 1).any(axis=1)]
         .intersection(health.dropna().index)
         .intersection(c8.index[c8.max_followed_hz.notna()]))
qn, hc = qn.loc[cells], health.loc[cells]

# --- health correction ---------------------------------------------------------------------
logq = np.log(qn)
corr = logq.copy()
slopes = {}
for hz in RATES[1:]:
    y = (logq[hz] - logq[5]).values
    slope = stats.theilslopes(y, hc.values)[0]
    slopes[hz] = slope
    corr[hz] = logq[hz] - slope * hc
qc = np.exp(corr)


def median_ci(x):
    x = np.asarray(x)
    boots = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
    return np.median(x), *np.percentile(boots, [2.5, 97.5])


def group_panel(ax, labels, groups, title):
    for gi, (lab, m) in enumerate(zip(labels, groups)):
        sub = qc[m]
        med = [median_ci(sub[hz]) for hz in RATES]
        x = np.arange(4) + (gi - 1) * 0.08
        ax.errorbar(x, [v[0] for v in med], yerr=[[v[0] - v[1] for v in med], [v[2] - v[0] for v in med]],
                    color=GROUP_COL[gi], marker="o", lw=2, capsize=3, label=f"{lab} (n = {m.sum()})")
    ax.axhline(1, color="0.5", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{hz} Hz" for hz in RATES])
    ax.set_xlabel("noise rate")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

# A: every cell + median
ax = axes[0]
for cell in qc.index:
    ax.plot(range(4), qc.loc[cell, RATES], color="0.8", lw=0.6, zorder=1)
med = [median_ci(qc[hz]) for hz in RATES]
ax.errorbar(range(4), [v[0] for v in med], yerr=[[v[0] - v[1] for v in med], [v[2] - v[0] for v in med]],
            color="k", marker="o", lw=2.5, capsize=4, zorder=3, label="median (95% range)")
for i, hz in enumerate(RATES):
    n_rf = int((qc[hz] > 1).sum())
    ax.text(i, 0.33, f"{n_rf} cells\nabove chance", ha="center", fontsize=8, color="0.3")
ax.axhline(1, color="0.5", ls=":", lw=1)
ax.text(-0.3, 0.93, "best chance value", fontsize=7, color="0.4", ha="left", va="top")
ax.set_yscale("log")
ax.set_ylim(0.3, 30)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{hz} Hz" for hz in RATES])
ax.set_xlabel("noise rate")
ax.set_ylabel("RF quality ÷ best chance value\n(health corrected)")
ax.set_title(f"All cells with an RF (n = {len(qc)}), grey = one cell", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# B: by highest followed chirp frequency
mf = c8.loc[cells, "max_followed_hz"]
group_panel(axes[1], ["follows up to ≤ 15 Hz", "follows up to 19 Hz", "follows up to 24–30 Hz"],
            [mf <= 16, (mf > 16) & (mf < 20), mf >= 20],
            "Split by highest chirp frequency followed")

# C: by most strongly followed chirp frequency
bf = c8.loc[cells, "best_hz"]
group_panel(axes[2], ["strongest at ≤ 9 Hz", "strongest at 11–14 Hz", "strongest at ≥ 17 Hz"],
            [bf <= 9, (bf > 9) & (bf < 15), bf >= 15],
            "Split by chirp frequency followed most strongly")

# plain-words statistics: does chirp frequency predict how much quality is kept at 15-20 Hz?
keep = np.log(((qc[15] + qc[20]) / 2) / ((qc[5] + qc[10]) / 2))
for ax, var, name in [(axes[1], mf, "highest"), (axes[2], bf, "strongest")]:
    r, pv = stats.spearmanr(var, keep)
    ax.text(0.02, 0.03, f"Cells with a higher {name} chirp frequency keep\n"
            f"{'more' if r > 0 else 'less'} quality at 15–20 Hz? r = {r:.2f}, p = {pv:.3f}"
            f"\n({'no clear link' if pv >= 0.1 else 'borderline, weak' if pv >= 0.05 else 'weak link'})",
            transform=ax.transAxes, fontsize=8, color="0.25", va="bottom")

fig.suptitle("15/10/2026, 32px white noise at 4 rates (20 min each, played 5→20 Hz). Quality is divided by the best value "
             "time-shifted spikes reach at that rate,\nand corrected for each cell's change in full-field responses "
             "from before to after the noise (shown as if responses had not changed). Chirp = the one just before the noise.",
             fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(OUT)
print("health slopes (log quality per unit QI change):", {k: round(v, 2) for k, v in slopes.items()})
print("median corrected quality:", {hz: round(float(np.median(qc[hz])), 2) for hz in RATES})

# ---------------------------------------------------------------- per-cell table (used by other scripts)
# results/noise_rate_corrected_quality.csv: corrected quality per rate (q5..q20), best rate, ratio
# best/second best, chirp highest followed / strongest frequency, ON/OFF polarity (sign of the RF-centre
# peak of sta_single_pixel at the cell's best rate; next best rate if missing).
import xarray as xr
qc.columns = [f"q{h}" for h in RATES]
cols = [f"q{h}" for h in RATES]
qc["best"] = qc[cols].idxmax(axis=1)
qc["second_ratio"] = qc[cols].apply(lambda r: np.sort(r.values)[-1] / np.sort(r.values)[-2], axis=1)
qc["max_hz"] = c8.loc[qc.index, "max_followed_hz"]
qc["best_chirp"] = c8.loc[qc.index, "best_hz"]

ds = xr.open_dataset("/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00/noise_analysis_QI10/noise_data_v2.nc")
s = ds.sta_single_pixel.load(); t = s.time_max.values
pre, bl = (t >= -300) & (t <= 0), (t >= -800) & (t <= -400)
pol = {}
for c in qc.index:
    for ch in np.argsort(-qc.loc[c, cols].values):
        x = s.sel(cell_index=c).isel(channel=int(ch)).values
        if np.all(np.isnan(x[pre])):
            continue
        dlt = x - np.nanmean(x[bl])
        pol[c] = "ON" if dlt[pre][np.nanargmax(np.abs(dlt[pre]))] > 0 else "OFF"
        break
qc["polarity"] = pd.Series(pol)
qc.round(2).to_csv(HERE / "noise_rate_corrected_quality.csv")
print(qc.best.value_counts().to_dict(), qc.polarity.value_counts().to_dict())
