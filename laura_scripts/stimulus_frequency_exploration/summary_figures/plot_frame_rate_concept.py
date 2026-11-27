"""Concept figure: how noise frame length interacts with a cell's integration window (27 Nov 2026).

Top row: schematic for one pixel's noise (black/white frames) and a cell's response window
(two-lobe shape, ~100 ms in total, peak ~100 ms before the spike) at 20, 10 and 5 Hz.
Bottom row: measured numbers from 15/10/2026 (32px white noise, same 27 cells at all rates):
signal per spike and signal/noise (signal_noise_by_rate.csv), time-course width
(rf_timecourse_by_noise_rate.csv); and a schematic for change-detecting cells.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parents[1] / "results"   # stimulus_frequency_exploration/results
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/stimulus_frequency_exploration/summary_figures/frame_rate_concept.png")
COL = {5: "#08306b", 10: "#2171b5", 15: "#41ab5d", 20: "#a1d99b"}
rng = np.random.default_rng(3)
PATTERN = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]   # fixed black/white sequence so every strip shows changes

t = np.arange(-450, 101)
d = -t.astype(float)


def gbump(x, tau, n):
    x = np.clip(x / tau, 1e-9, None)
    return x ** n * np.exp(-n * (x - 1))


kernel = gbump(d, 95, 7) - 0.35 * gbump(d, 175, 18)         # main lobe + earlier opposite lobe (~150 ms in total)
kernel[d < 0] = 0


def frames_strip(ax, frame_ms, y0, h, phase=17):
    start = -450 - phase
    k = 0
    while start < 100:
        val = PATTERN[k % len(PATTERN)]
        ax.add_patch(Rectangle((start, y0), frame_ms, h, facecolor="0.1" if val else "white",
                               edgecolor="0.6", lw=0.5))
        start += frame_ms
        k += 1


fig = plt.figure(figsize=(15, 15))
outer = fig.add_gridspec(2, 1, height_ratios=[2.15, 1], hspace=0.28)
top = outer[0].subgridspec(2, 3, height_ratios=[1, 1.05], hspace=0.95, wspace=0.3)
bottom = outer[1].subgridspec(1, 3, wspace=0.3)

tcw = pd.read_csv(HERE / "rf_timecourse_by_noise_rate.csv")
good = tcw[tcw.rf]
cells = good.pivot(index="cell", columns="rate", values="t_peak").dropna().index
w = good[good.cell.isin(cells)].groupby("rate").fwhm.median()

cases = [(20, "Fast noise (20 Hz, 50 ms frames)",
          "~2–3 frames inside the window are added together\n→ their patterns partly cancel: less signal per spike\n→ timing sharp (little smearing)"),
         (10, "Matched (10 Hz, 100 ms frames)",
          "about one frame fills the window\n→ each spike points clearly at one pattern\n→ timing smeared over ±100 ms"),
         (5, "Slow noise (5 Hz, 200 ms frames)",
          "one frame is longer than the window\n→ signal per spike stays high, but timing is lost:\n   time course smeared over ±200 ms, false tail after the spike")]
for i, (hz, title, text) in enumerate(cases):
    ax = fig.add_subplot(top[0, i])
    frames_strip(ax, 1000 / hz, 1.25, 0.35)
    ax.fill_between(t, 0, kernel, where=kernel > 0, color=COL[hz], alpha=0.6, lw=0)
    ax.fill_between(t, 0, kernel, where=kernel < 0, color=COL[hz], alpha=0.25, lw=0)
    ax.plot(t, kernel, color=COL[hz], lw=1.5)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.plot([0, 0], [-0.4, 1.7], color="k", lw=2)
    ax.text(5, -0.38, "spike", fontsize=8)
    ax.text(-445, 1.65, "noise frames (one pixel)", fontsize=8, va="bottom")
    ax.text(-445, 0.75, "cell's response\nwindow", fontsize=8, color=COL[hz] if hz != 20 else "#41ab5d")
    ax.set_xlim(-450, 100); ax.set_ylim(-0.45, 1.85)
    ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("time relative to spike (ms)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.text(0.0, -0.3, text, transform=ax.transAxes, fontsize=9, va="top")
    ax.text(0.0, -0.52, f"blur ±{1000 / hz:.0f} ms → measured width {w[hz]:.0f} ms",
            transform=ax.transAxes, fontsize=9, va="top", color=COL[hz] if hz != 20 else "#238b45", fontweight="bold")

# ---- measured: signal per spike + signal/noise
sn = pd.read_csv(HERE / "signal_noise_by_rate.csv")
med = sn.groupby("rate")[["signal", "snr", "n"]].median()
rel = med / med.loc[5]
rates = [5, 10, 15, 20]
ax = fig.add_subplot(top[1, 0])
ax.plot(rates, rel.signal, "o-", color="#2171b5", lw=2, label="signal per spike (blending)")
ax.plot(rates, rel.snr, "s--", color="#41ab5d", lw=2, label="signal ÷ noise of the RF")
ax.axhline(1, color="0.7", lw=0.6)
ax.set_xticks(rates); ax.set_xticklabels([f"{r} Hz" for r in rates])
ax.set_ylim(0, 1.15)
ax.set_ylabel("relative to 5 Hz (median)")
ax.set_title("Measured: faster noise → less signal per spike\n(15/10, same 27 cells)", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)

# ---- measured: width vs frame
ax = fig.add_subplot(top[1, 1])
ax.plot(rates, [w[r] for r in rates], "o-", color="#08306b", lw=2, label="measured time-course width")
ax.plot(rates, [1000 / r for r in rates], "k:", lw=1.2, label="one frame")
for r in rates:
    ax.annotate(f"blur ±{1000 / r:.0f} ms\nwidth {w[r]:.0f} ms", (r, w[r]), xytext=(8, 6), textcoords="offset points",
                fontsize=8, color="#08306b")
ax.set_ylim(30, 215)
ax.set_xticks(rates); ax.set_xticklabels([f"{r} Hz" for r in rates])
ax.set_ylabel("width at half height (ms)")
ax.set_title("Measured: slower noise → width is just the frame\n(timing blurred)", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# ---- change-detecting cells
ax = fig.add_subplot(top[1, 2])
y_rows = [(1.95, 100, "frames ≤ window (10–20 Hz): at least one frame change\ninside every window → the cell keeps being driven"),
          (0.45, 260, "frames ≫ window (5 Hz and slower): stretches with no\nchange → quiet between changes; fewer useful spikes")]
for y0, fm, txt in y_rows:
    frames_strip(ax, fm, y0 + 0.55, 0.25, phase=40)
    k = kernel * 0.4 + y0
    ax.fill_between(t, y0, k, where=kernel > 0, color="#2171b5", alpha=0.6, lw=0)
    ax.fill_between(t, y0, k, where=kernel < 0, color="#2171b5", alpha=0.25, lw=0)
    ax.axhline(y0, color="0.7", lw=0.5)
    ax.text(-445, y0 - 0.12, txt, fontsize=8.5, va="top")
ax.plot([0, 0], [-0.3, 2.85], color="k", lw=2)
ax.set_xlim(-450, 100); ax.set_ylim(-0.45, 2.9)
ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xlabel("time relative to spike (ms)")
ax.set_title("Change-detecting cells (two lobes of opposite sign):\nneed a frame change within the window",
             fontsize=10, loc="left")

# ---- which number to compare with the frame, per recording (all at 20 Hz, least blurred)
REC_LABEL = {"05_11_2025": "05/11/25", "02_12_2025": "02/12/25", "15_01_2026": "15/01/26",
             "26_02_2026": "26/02/26", "05_03_2026": "05/03/26", "15_05_2026": "15/05/26", "15_10_2026": "15/10/26"}
wi = pd.read_csv(HERE / "width_integration_20hz.csv")
recs = {r: (g.width_ms.dropna(), g.int_both_ms.dropna()) for r, g in wi.groupby("recording")}
i1510 = pd.read_csv(HERE / "width_integration_15_10.csv")
i1510 = i1510[i1510.rate == 20]
w1510 = good[(good.rate == 20) & good.cell.isin(cells)].fwhm.dropna()
recs["15_10_2026"] = (w1510, i1510.int_both_ms.dropna())
order = sorted(recs, key=lambda r: recs[r][1].median())
ax = fig.add_subplot(bottom[0, :2])
for k, r in enumerate(order):
    wv, iv = recs[r]
    if r == "15_10_2026":
        ax.axvspan(k - 0.45, k + 0.45, color="#a1d99b", alpha=0.35, lw=0)
        ax.text(k, 12, "15/10 (rows above):\non the fast side", ha="center", fontsize=8, color="#238b45")
    for v, x, c, mk in [(wv, k - 0.12, "#2171b5", "o"), (iv, k + 0.12, "#08306b", "s")]:
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.errorbar(x, q2, yerr=[[q2 - q1], [q3 - q2]], fmt=mk, color=c, ms=8, capsize=3)
    ax.text(k, -38, f"n = {len(iv)}", ha="center", fontsize=8, color="0.4")
for hz, ls in [(20, "-"), (15, "--"), (10, "-."), (5, ":")]:
    ax.axhline(1000 / hz, color=COL[hz] if hz != 20 else "#41ab5d", ls=ls, lw=1.3)
    ax.text(len(order) - 0.45, 1000 / hz + 3, f"{hz} Hz frame ({1000 / hz:.0f} ms)", fontsize=8, ha="right",
            va="bottom", color=COL[hz] if hz != 20 else "#238b45")
ax.plot([], [], "o", color="#2171b5", label="half-height width (main lobe) → compare for TIMING")
ax.plot([], [], "s", color="#08306b", label="whole-response integration time (both lobes) → compare for RF SIGNAL")
ax.set_xticks(range(len(order))); ax.set_xticklabels([REC_LABEL[r] for r in order])
ax.set_xlim(-0.6, len(order) - 0.4); ax.set_ylim(0, 260)
ax.set_ylabel("ms (median, bars = middle 50 % of cells)")
ax.set_title("Which number to compare with the frame? All recordings measured with 20 Hz noise "
             "(least blurred; still slightly inflated by the 50 ms frames)", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(bottom[0, 2])
ax.axis("off")
ax.text(0, 1, "How to use these numbers\n\n"
        "For RF maps (signal):\n"
        "  frame ≈ whole-response integration time\n"
        "  (about one frame per response → little blending)\n"
        "  recordings: ~96–186 ms (15/10: ~127 ms)\n"
        "  → about 10 Hz for the faster retinas, 5–8 Hz\n     would also suit the slowest\n\n"
        "For timing (shape, width, ON/OFF):\n"
        "  frame clearly shorter than the half-height width\n"
        "  recordings: ~65–107 ms (15/10: ~73 ms)\n"
        "  → 20 Hz for the faster retinas (05/11, 15/10, 02/12);\n"
        "     15 Hz is enough for the slower ones (≥ ~90 ms)\n\n"
        "Retinas differ ~2-fold: choose from values typical\nfor the retinas you will record.\n"
        "Clustering recordings: 12px (60 µm squares);\n15/10: 32px (64 µm squares).",
        va="top", fontsize=9.5)

fig.suptitle("Choosing the noise frame rate: compare the frame length with the cell's response window.\n"
             "Too short → frames blend (weak RF signal); too long → timing smeared. "
             "Two effects, one cause.", fontsize=11, x=0.01, ha="left")
CONCLUSION = (
    "Conclusion\n"
    "1. Longer frames (lower rate) raise signal ÷ noise: each spike is tied more clearly to one frame. The gain stops once the frame ≈ the whole response;\n"
    "    beyond that: fewer independent frames, adaptation during long frames, and change-detecting cells go quiet between frame changes.\n"
    "2. Change-detecting cells need at least one frame change inside their response window (frames ≈ one lobe drive them best) — but more than ~2 frames\n"
    "    per window still blends and costs signal.\n"
    "3. For signal ÷ noise, compare the frame with the WHOLE-RESPONSE integration time (both lobes).\n"
    "4. For timing, the frame must be shorter than the HALF-HEIGHT WIDTH: the measured width can never be shorter than ~1 frame (peak time is hardly affected).\n"
    "5. Unique, non-repeating frames, longer recordings and brighter light (e.g. a stronger red LED) raise signal ÷ noise and help compensate\n"
    "    for using a faster rate.\n"
    "6. Retinas differ ~2-fold, so choose on typical values. For these retinas, 15 Hz is a reasonable compromise between signal and timing.")
fig.text(0.01, 0.0, CONCLUSION, va="top", ha="left", fontsize=10,
         bbox=dict(facecolor="#f0f6ec", edgecolor="#a1d99b", boxstyle="round,pad=0.6"))
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(OUT)
