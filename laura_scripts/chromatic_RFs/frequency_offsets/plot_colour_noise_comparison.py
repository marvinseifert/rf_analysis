"""Figure: colour noise results (27 Nov 2026), from colour_noise_comparison.csv / _timecourses.npy.

22/10/2026 Phase_01: 32px, 10 Hz, 413 / 500 / 610 nm interleaved (500 frames each), 4000 unique frames
per colour x 9. 15/10/2026 Phase_00: 32px, 15 Hz, white / 610 / 500 / 420 nm interleaved (900 frames each),
4500 unique frames per colour x 4, plus white noise alone (stim 11).
Cells: RF above chance for that colour (chance = spikes shifted by whole colour cycles).
LED powers from 'LED power 21_10_2026.txt' (22/10 settings): 413 and 500 nm 4.5 nW; the 610 nm LED was
replaced by a 622 nm LED at 3.3 nW (originally 2 nW, i.e. what 15/10 most likely had).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent / "results"   # chromatic_RFs/frequency_offsets/results
SFE = Path(__file__).resolve().parents[2] / "stimulus_frequency_exploration" / "response_width_integration"
OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/chromatic_RFs/frequency_offsets/colour_noise_comparison.png")
LAGS = -800 + 10 * np.arange(100)
COL = {"413": "#2171b5", "420": "#2171b5", "500": "#41ab5d", "610": "#cb181d", "white": "0.25", "white alone": "0.6"}
NAME = {"413": "413 nm", "420": "420 nm", "500": "500 nm", "610": "610 nm", "white": "white", "white alone": "white alone"}

df = pd.read_csv(HERE / "colour_noise_comparison.csv")
tcs = np.load(HERE / "colour_noise_comparison_timecourses.npy")
df["row"] = np.arange(len(df))
rf = df[df.rf]
groups = [("22/10 (10 Hz)", 4, ["413", "500", "610"]),
          ("15/10 (15 Hz)", 5, ["white", "420", "500", "610"]),
          ("15/10 (15 Hz)", 11, ["white alone"])]

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3)

# ---- A: number of cells with an RF
ax = fig.add_subplot(gs[0, 0])
x, labels = 0, []
for rec, stim, cols in groups:
    for c in cols:
        n = int(((rf.recording == rec) & (rf.stim == stim) & (rf.colour == c)).sum())
        ax.bar(x, n, color=COL[c], width=0.75)
        ax.text(x, n + 1, str(n), ha="center", fontsize=8)
        labels.append(NAME[c]); x += 1
    x += 0.6
ticks = [0, 1, 2, 3.6, 4.6, 5.6, 6.6, 8.2]
ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.text(1, 73, "22/10, 10 Hz", ha="center", fontsize=9, fontweight="bold")
ax.text(5.1, 73, "15/10, 15 Hz\n(colour sequence)", ha="center", fontsize=9, fontweight="bold", va="center")
ax.set_ylim(0, 80)
ax.set_ylabel("cells with an RF above chance")
ax.set_title("RF yield per colour\n(red: 2–3.3 nW vs 4.5 nW for the others)", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# ---- B: signal / noise
ax = fig.add_subplot(gs[0, 1])
x = 0
for rec, stim, cols in groups:
    for c in cols:
        v = rf[(rf.recording == rec) & (rf.stim == stim) & (rf.colour == c)].snr
        if len(v):
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            ax.errorbar(x, q2, yerr=[[q2 - q1], [q3 - q2]], fmt="o", color=COL[c], ms=8, capsize=3)
        else:
            ax.text(x, 3, "none", ha="center", fontsize=8, color=COL[c])
        x += 1
    x += 0.6
ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylim(0, 22)
ax.set_ylabel("signal ÷ noise of the RF (median, middle 50 %)")
ax.set_title("Signal ÷ noise (cells with an RF)", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)

# ---- C: peak time, same cells across colours
ax = fig.add_subplot(gs[0, 2])
x0 = 0
for rec, stim, cols in groups[:2]:
    sub = rf[(rf.recording == rec) & (rf.stim == stim)]
    piv = sub.pivot_table(index="cell", columns="colour", values="peak_ms")
    xs = {c: x0 + k for k, c in enumerate(cols)}
    for _, r in piv.iterrows():
        pts = [(xs[c], r[c]) for c in cols if c in piv and np.isfinite(r.get(c, np.nan))]
        if len(pts) > 1:
            ax.plot(*zip(*pts), color="0.85", lw=0.6, zorder=1)
    for c in cols:
        if c in piv:
            v = piv[c].dropna()
            if len(v):
                ax.plot(xs[c], np.median(v), "o", color=COL[c], ms=9, zorder=3)
    x0 += len(cols) + 0.6
ax.set_xticks(ticks[:7]); ax.set_xticklabels(labels[:7], rotation=45, ha="right", fontsize=8)
ax.set_ylim(-320, -40)
ax.set_ylabel("peak time (ms before spike)")
ax.set_title("Speed: peak time (dot = median, grey = same cell)\n"
             "413 = 500 nm at 10 Hz; red (6 cells) later, not earlier", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)


# ---- D, E: median time courses
def tc_panel(ax, rec, stim_cols, frame_ms, title):
    for k, start in enumerate(np.arange(-800, 200, frame_ms)):
        if k % 2 == 0:
            ax.axvspan(start, start + frame_ms, color="0.94", lw=0)
    for stim, cols in stim_cols:
        for c in cols:
            sub = rf[(rf.recording == rec) & (rf.stim == stim) & (rf.colour == c)]
            if not len(sub):
                continue
            Y = []
            for row in sub.row:
                y = tcs[row].astype(float)
                j = np.argmax(np.abs(y) * (LAGS <= 0))
                Y.append(y * np.sign(y[j]) / np.abs(y[j]))
            med = np.median(Y, axis=0)
            g = sub
            ax.plot(LAGS, med, color=COL[c], lw=2,
                    label=f"{NAME[c]} (n={len(g)}): width {g.width_ms.median():.0f}, whole {g.int_both_ms.median():.0f} ms")
    ax.axvline(0, color="0.4", lw=0.8, ls=":"); ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlim(-500, 190); ax.set_ylim(-0.6, 1.15)
    ax.set_xlabel("time relative to spike (ms)"); ax.set_ylabel("scaled to peak 1 (flipped: peak up)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)


tc_panel(fig.add_subplot(gs[1, 0]), "22/10 (10 Hz)", [(4, ["413", "500", "610"])], 100,
         "22/10, 10 Hz (grey bands = 100 ms frames)\nwidth ≈ one frame → colour differences hidden")
tc_panel(fig.add_subplot(gs[1, 1]), "15/10 (15 Hz)", [(5, ["white", "420", "500"]), (11, ["white alone"])], 1000 / 15,
         "15/10, 15 Hz (bands = 67 ms frames)\n420 nm: longer whole response than 500 nm (+19 ms, p = 0.02)")

ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
ax.text(0, 1,
        "Summary\n\n"
        "• Red (610 nm) barely drove the cells: 6 RFs at 10 Hz,\n  none at 15 Hz. The red LED was the dimmest\n"
        "  (2 nW, later 622 nm at 3.3 nW; others 4.5 nW).\n\n"
        "• The few red RFs peaked later, not earlier (median\n  55 ms later than 500 nm, n = 6, not significant):\n"
        "  dim light slows cones — not evidence that red is slow.\n\n"
        "• 413 vs 500 nm at 10 Hz (54 cells): no speed difference.\n"
        "  15 Hz: 420 nm responses peak earlier than 500 nm and white\n"
        "  (by ~10–17 ms, p 0.01–0.03, faster in 8–11 of 9–12 cells),\n"
        "  with a longer whole response (+19 ms) — and opposite sign\n"
        "  to 500 nm/white in nearly every cell (colour-opponent input).\n"
        "  The LEDs do not isolate single cones (413 nm also drives\n  green cones), which hides cone-speed differences.\n\n"
        "• 10 Hz: widths ≈ one frame, so speed differences are\n  hidden; 15 Hz shows them slightly better. Different\n"
        "  retinas, so not a clean 10 vs 15 Hz comparison.\n\n"
        "• White inside the colour sequence ≈ white alone\n  (peak −110 vs −100 ms; width 81 vs 78 ms).\n\n"
        "→ For red-cone noise: raise red LED power first;\n   then test red speed at 10 vs 15 Hz in the pilot.",
        va="top", fontsize=9)

fig.suptitle("Colour noise: RF yield, signal ÷ noise and speed per colour (32px noise; RF above chance per colour)",
             fontsize=11, x=0.01, ha="left")
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(OUT)
