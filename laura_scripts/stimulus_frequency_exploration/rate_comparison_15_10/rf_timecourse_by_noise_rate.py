"""RF time course vs noise rate (15/10/2026, 32px white noise, 5/10/15/20 Hz, 20 min each).

Per cell and rate: STA from spikes + stimulus (8-px blocks, 10 ms bins, -800..+190 ms), time course
of the RF centre (3 x 3 blocks = 48 µm around the peak of temporal variance), sign-flipped so the
main peak is positive, normalised to peak 1. Measures: time of peak (parabolic interpolation),
width at half height (FWHM, linear interpolation), and frames inside that width.

Why frames blur the time course: a frame stays on screen for its whole duration D, so the stimulus
is correlated with itself for +-D (a triangle, base 2D). The measured time course = true response
smeared by that triangle -> wider, peak moved towards the slow side, and a tail after the spike.

Cells: RF above chance at that rate (health-corrected quality > 1, noise_rate_corrected_quality.csv).
Writes rf_timecourse_by_noise_rate.npz + figure in Box Figures/noise_rate_comparison/.
"""
import pickle
from pathlib import Path
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import torch
from scipy import ndimage, stats
import matplotlib.pyplot as plt

ROOT = "/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00"
STIM = "/mnt/mea_nas/Laura/stimuli/stimuli_Oct_2026/32px_{}Hz_20mins_shuffle_x4.h5"
RATES = {5: 9, 10: 10, 15: 11, 20: 12}
COL = {5: "#08306b", 10: "#2171b5", 15: "#41ab5d", 20: "#a1d99b"}
HERE = Path(__file__).resolve().parents[1] / "results"   # stimulus_frequency_exploration/results
OUT_FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/stimulus_frequency_exploration/rate_comparison_15_10/rf_timecourse_by_noise_rate_same_cells.png")
B, FS = 8, 20000
LAGS = -800 + 10 * np.arange(100)            # ms, negative = before the spike
dev = "cuda"

qc = pd.read_csv(HERE / "noise_rate_corrected_quality.csv", index_col="cell")
cache = HERE / "rf_timecourse_by_noise_rate.npz"

if not cache.exists():
    o = pickle.load(open(f"{ROOT}/overview", "rb"))
    sp = pd.read_parquet(f"{ROOT}/alldata.parquet")
    tcs = {}
    for hz, idx in RATES.items():
        r = o.stimulus_df[o.stimulus_df.stimulus_index == idx].iloc[0]
        trig = np.asarray(r.trigger_fr_relative)
        with h5py.File(STIM.format(hz), "r") as f:
            S = f["Noise"][:]
        nf = S.shape[0]
        S = (S[:, ::B, ::B] == 255).astype(np.float32)
        hb, wb = S.shape[1:]
        S_t = torch.from_numpy(S.reshape(nf, -1)).to(dev)
        S_t -= S_t.mean(0, keepdim=True)
        del S
        end = trig[nf]
        lag_samp = (LAGS * FS // 1000).astype(np.int64)
        m = (sp.times >= r.begin_fr) & (sp.times < r.begin_fr + end)
        spikes = {c: g.times.values - r.begin_fr for c, g in sp[m].groupby("cell_index")}
        for cell in qc.index:
            t = spikes.get(cell)
            if t is None or len(t) < 100:
                continue
            fi = np.searchsorted(trig[: nf + 1], t[:, None] + lag_samp[None, :], side="right") - 1
            ok = (fi >= 0) & (fi < nf)
            W = np.zeros((100, nf), np.float32)
            for li in range(100):
                W[li] = np.bincount(fi[ok[:, li], li], minlength=nf)
            sta = (torch.from_numpy(W).to(dev) @ S_t).cpu().numpy().reshape(100, hb, wb) / len(t)
            v = ndimage.gaussian_filter(sta, (2, 1, 1)).var(axis=0)
            cy, cx = np.unravel_index(v.argmax(), v.shape)
            tcs[(cell, hz)] = sta[:, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].mean(axis=(1, 2))
        print(hz, "Hz done", flush=True)
        del S_t; torch.cuda.empty_cache()
    keys = list(tcs)
    np.savez(cache, cells=[k[0] for k in keys], rates=[k[1] for k in keys], tc=np.array([tcs[k] for k in keys]))

z = np.load(cache)
rows, norm = [], {}
for cell, hz, tc in zip(z["cells"], z["rates"], z["tc"]):
    pre = LAGS <= 0
    i = np.argmax(np.abs(tc) * pre)
    y = tc * np.sign(tc[i])
    y = y / y[i]
    # parabolic peak
    if 0 < i < 99:
        a, b, c = y[i - 1], y[i], y[i + 1]
        off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0
    else:
        off = 0
    t_peak = LAGS[i] + 10 * off
    # FWHM around the main peak
    lo = i
    while lo > 0 and y[lo] > 0.5:
        lo -= 1
    hi = i
    while hi < 99 and y[hi] > 0.5:
        hi += 1
    t_lo = LAGS[lo] + 10 * (0.5 - y[lo]) / (y[lo + 1] - y[lo]) if y[lo] <= 0.5 else np.nan
    t_hi = LAGS[hi - 1] + 10 * (y[hi - 1] - 0.5) / (y[hi - 1] - y[hi]) if y[hi] <= 0.5 else np.nan
    fwhm = t_hi - t_lo
    rf = qc.loc[cell, f"q{hz}"] > 1
    rows.append(dict(cell=cell, rate=hz, rf=rf, t_peak=t_peak, fwhm=fwhm, t_half_end=t_hi,
                     frames_in_fwhm=fwhm / (1000 / hz), polarity=qc.loc[cell, "polarity"]))
    norm[(cell, hz)] = y
df = pd.DataFrame(rows)
df.to_csv(HERE / "rf_timecourse_by_noise_rate.csv", index=False)
good = df[df.rf]
# SAME cells in every panel: RF above chance at all four rates
paired = good.pivot(index="cell", columns="rate", values="t_peak").dropna()
good = good[good.cell.isin(paired.index)]
pw = good.pivot(index="cell", columns="rate", values="fwhm").dropna()
N_ON = good[good.polarity == "ON"].cell.nunique(); N_OFF = good[good.polarity == "OFF"].cell.nunique()

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(4, 3, width_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.42)
for row, hz in enumerate(RATES):
    ax = fig.add_subplot(gs[row, 0])
    D = 1000 / hz
    for k, start in enumerate(np.arange(-800, 200, D)):      # frame grid (phase is arbitrary)
        if k % 2 == 0:
            ax.axvspan(start, start + D, color="0.92", lw=0)
    g = good[good.rate == hz]
    lines = []
    for pol, ls in [("ON", "-"), ("OFF", "--")]:
        gp = g[g.polarity == pol]
        Y = np.array([norm[(c, hz)] for c in gp.cell])
        q1, q3 = np.percentile(Y, [25, 75], axis=0)
        ax.fill_between(LAGS, q1, q3, color=COL[hz], alpha=0.18 if pol == "ON" else 0.10, lw=0)
        ax.plot(LAGS, np.median(Y, axis=0), color=COL[hz], lw=2.2, ls=ls,
                label=f"{pol} (n = {len(gp)})" if row == 0 else None)
        lines.append(f"{pol}: peak {np.median(gp.t_peak):.0f} ms, width {np.median(gp.fwhm):.0f} ms")
    # blur triangle (stimulus self-correlation), drawn under the curves at the median peak
    tp = np.median(g.t_peak)
    tri_x = np.array([tp - D, tp, tp + D])
    ax.plot(tri_x, [-0.35, -0.05, -0.35], color="k", lw=1.2)
    ax.text(tp + D + 8, -0.33, f"blur: ±{D:.0f} ms", fontsize=8, va="bottom")
    ax.text(-790, 0.95, f"{hz} Hz: frame = {D:.0f} ms (grey/white bands), "
            f"{np.median(g.frames_in_fwhm):.1f} frames in the width\n" + "\n".join(lines), fontsize=8, va="top")
    if row == 0:
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.45))
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlim(-800, 190); ax.set_ylim(-0.45, 1.15)
    ax.set_yticks([0, 1])
    ax.spines[["top", "right"]].set_visible(False)
    if row == 3:
        ax.set_xlabel("time relative to spike (ms)")
    else:
        ax.set_xticklabels([])
    if row == 0:
        ax.set_title("RF centre time course: solid = ON, dashed = OFF\n"
                     "(median, shaded = middle 50 %; flipped so the peak is up, peak = 1)", fontsize=10, loc="left")

# B: time to peak, paired
ax = fig.add_subplot(gs[:2, 1])
for c in paired.index:
    ax.plot(range(4), paired.loc[c, list(RATES)], color="0.8", lw=0.6)
for k, hz in enumerate(RATES):
    v = good[good.rate == hz].t_peak
    ax.plot(k, np.median(v), "o", color=COL[hz], ms=9, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{h} Hz" for h in RATES])
ax.set_ylabel("time of peak (ms before spike)")
fr = stats.friedmanchisquare(*[paired[h] for h in RATES])
off = int((paired < -260).values.sum())
ax.set_ylim(-260, -20)
ax.set_title(f"Peak time (grey = one cell, dot = median)\n"
             f"{off} values below -260 ms off scale", fontsize=9, loc="left")
ax.text(0.02, 0.02, f"rates differ? Friedman p = {fr.pvalue:.3f}\n10 vs 20 Hz: {np.median(paired[20] - paired[10]):+.0f} ms",
        transform=ax.transAxes, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# C: width
ax = fig.add_subplot(gs[2:, 1])
for c in pw.index:
    ax.plot(range(4), pw.loc[c, list(RATES)], color="0.8", lw=0.6)
for k, hz in enumerate(RATES):
    v = good[good.rate == hz].fwhm
    ax.plot(k, np.median(v), "o", color=COL[hz], ms=9, zorder=3)
ax.plot(range(4), [1000 / h for h in RATES], "k--", lw=1, label="one frame")
ax.set_xticks(range(4)); ax.set_xticklabels([f"{h} Hz" for h in RATES])
ax.set_ylabel("width at half height (ms)")
frw = stats.friedmanchisquare(*[pw[h] for h in RATES])
ax.set_title(f"Response width at half height\n(Friedman p = {frw.pvalue:.3g})", fontsize=9, loc="left")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# D: frames inside the response + ON/OFF peak times
ax = fig.add_subplot(gs[:2, 2])
for k, hz in enumerate(RATES):
    v = good[good.rate == hz].frames_in_fwhm
    ax.boxplot(v.dropna(), positions=[k], widths=0.5, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=COL[hz], alpha=0.6), medianprops=dict(color="k"))
ax.axhline(1, color="0.5", ls=":", lw=1)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{h} Hz" for h in RATES])
ax.set_ylabel("frames shown within the\nresponse width")
ax.set_title("Frames within the response width\n(~1 = the width is just one frame)", fontsize=9, loc="left")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[2:, 2])
for j, (pol, mk) in enumerate([("ON", "o"), ("OFF", "s")]):
    meds = [np.median(good[(good.rate == hz) & (good.polarity == pol)].t_peak) for hz in RATES]
    ns = [int(((good.rate == hz) & (good.polarity == pol)).sum()) for hz in RATES]
    ax.plot(range(4), meds, marker=mk, color="0.3" if pol == "ON" else "0.6", lw=1.5,
            label=f"{pol} cells (n = {max(ns)})")
ax.set_xticks(range(4)); ax.set_xticklabels([f"{h} Hz" for h in RATES])
ax.set_ylabel("median peak time (ms before spike)")
ax.set_title("Peak time, ON vs OFF cells", fontsize=9, loc="left")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("15/10/2026, 32px white noise: how the frame length smears the RF time course. "
             f"Same {len(paired)} cells in every panel (RF above chance at all four rates; {N_ON} ON, {N_OFF} OFF)",
             fontsize=11, x=0.01, ha="left")
fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
print(OUT_FIG)
print(good.groupby("rate")[["t_peak", "fwhm", "frames_in_fwhm", "t_half_end"]].median().round(1))
for hz in RATES:
    gg = good[good.rate == hz]
    on, offv = gg[gg.polarity == "ON"].t_peak, gg[gg.polarity == "OFF"].t_peak
    print(f"{hz} Hz ON-OFF peak {np.median(on) - np.median(offv):+.0f} ms, Mann-Whitney p={stats.mannwhitneyu(on, offv).pvalue:.3f}")
print("paired n", len(paired), "Friedman peak p", round(fr.pvalue, 4), "width p", round(frw.pvalue, 4))
for a, b in [(5, 10), (10, 15), (15, 20), (10, 20)]:
    print(f"peak {a} vs {b} Hz: median diff {np.median(paired[b] - paired[a]):+.0f} ms, Wilcoxon p={stats.wilcoxon(paired[a], paired[b]).pvalue:.3f}")
