"""Schematic: how the frame length blurs the STA time course (27 Nov 2026).

A cell's true response (same two-lobe shape as frame_rate_concept.png: main lobe peak ~95 ms before
the spike, earlier opposite lobe, ~150 ms in total). With noise frames of length D, the stimulus is
correlated with itself over +-D (each value is held for a whole frame, and the frame start is random
relative to the spike), i.e. a triangle of base 2D. The STA = true response smeared by that triangle.
Illustration, not a fit; the 15/10 measured widths are given for comparison.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/noise_rate_comparison/sta_blur_schematic.png")
COL = {5: "#08306b", 10: "#2171b5", 15: "#41ab5d", 20: "#a1d99b"}
MEASURED_WIDTH = {5: 190, 10: 99, 15: 75, 20: 73}       # 15/10, same 27 cells (ms)

t = np.arange(-600, 301).astype(float)                  # stimulus time relative to spike (ms)


def gbump(x, tau, n):
    x = np.clip(x / tau, 1e-9, None)
    return x ** n * np.exp(-n * (x - 1))


d = -t
true = gbump(d, 95, 11) - 0.35 * gbump(d, 175, 18)
true[d < 0] = 0                                           # the cell cannot respond to the future


def smear(k, D):
    tri = np.clip(1 - np.abs(np.arange(-int(D), int(D) + 1)) / D, 0, None)
    tri /= tri.sum()
    return np.convolve(k, tri, mode="same")


def half_width(y):
    i = np.argmax(y)
    above = np.where(y >= 0.5 * y[i])[0]
    # contiguous run around the peak
    lo = i
    while lo > 0 and y[lo - 1] >= 0.5 * y[i]:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and y[hi + 1] >= 0.5 * y[i]:
        hi += 1
    return t[lo], t[hi]


fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.1], hspace=0.55, wspace=0.28)

# ---- A: why it happens
ax = fig.add_subplot(gs[0, :2])
D = 200
starts = [-560, -360, -160, 40]
vals = [1, 0, 1, 0]
for s, v in zip(starts, vals):
    ax.add_patch(Rectangle((s, 1.15), D, 0.3, facecolor="0.1" if v else "white", edgecolor="0.5"))
ax.fill_between(t, 0, true, where=true > 0, color="#2171b5", alpha=0.55, lw=0)
ax.fill_between(t, 0, true, where=true < 0, color="#2171b5", alpha=0.2, lw=0)
ax.plot(t, true, color="#2171b5", lw=1.5)
ax.plot([0, 0], [-0.4, 1.55], "k", lw=2)
ax.text(4, -0.38, "spike", fontsize=9)
ax.annotate("", xy=(-160, 1.05), xytext=(40, 1.05), arrowprops=dict(arrowstyle="<->", color="0.3"))
ax.text(-330, 0.93, "this frame (200 ms at 5 Hz) is on screen\nboth before AND after the spike",
        ha="center", va="top", fontsize=9, color="0.25")
ax.text(-590, 1.5, "noise frames (one pixel), 5 Hz", fontsize=9)
ax.text(-590, 0.55, "cell's true\nresponse", fontsize=9, color="#2171b5")
ax.set_xlim(-600, 300); ax.set_ylim(-0.45, 1.65)
ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xlabel("time relative to spike (ms)")
ax.set_title("Why the STA gets blurred: the frame that drove the spike ~100 ms earlier is often still on screen,\n"
             "so the average stimulus is changed before and after the true response (and even after the spike)",
             fontsize=10, loc="left")

# ---- B: the smearing triangles
ax = fig.add_subplot(gs[0, 2])
for hz in (5, 10, 15, 20):
    Dm = 1000 / hz
    x = np.array([-Dm, 0, Dm])
    ax.plot(x, [0, 1, 0], color=COL[hz], lw=2, label=f"{hz} Hz: ±{Dm:.0f} ms")
ax.set_xlim(-230, 230); ax.set_ylim(0, 1.15)
ax.set_yticks([])
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xlabel("time shift (ms)")
ax.set_title("Smearing applied to the true response\n(stimulus self-correlation: ±1 frame)", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8)

# ---- C: STA as you would measure it (same spikes, real amplitude)
ax = fig.add_subplot(gs[1, 0])
ax.plot(t, true, color="k", lw=2, label="true response")
for hz in (20, 15, 10, 5):
    ax.plot(t, smear(true, 1000 / hz), color=COL[hz], lw=2, label=f"STA at {hz} Hz")
ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.4", lw=0.8, ls=":")
ax.set_xlim(-450, 200)
ax.set_xlabel("time relative to spike (ms)"); ax.set_ylabel("STA (true response peak = 1)")
ax.set_title("Measured STA: lower, wider, opposite lobe\ncancelled, tail after the spike", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# ---- D: shape only (each scaled to peak 1) with half-height widths
ax = fig.add_subplot(gs[1, 1])
ax.plot(t, true / true.max(), color="k", lw=2)
lo, hi = half_width(true)
ax.plot([lo, hi], [0.5, 0.5], color="k", lw=4, solid_capstyle="butt")
rows = [f"true: {hi - lo:.0f} ms"]
for k, hz in enumerate((20, 15, 10, 5)):
    s = smear(true, 1000 / hz)
    s = s / s.max()
    ax.plot(t, s, color=COL[hz], lw=2)
    lo, hi = half_width(s)
    ax.plot([lo, hi], [0.42 - 0.08 * k] * 2, color=COL[hz], lw=4, solid_capstyle="butt")
    rows.append(f"{hz} Hz: {hi - lo:.0f} ms (measured 15/10: {MEASURED_WIDTH[hz]} ms)")
ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.4", lw=0.8, ls=":")
ax.set_xlim(-450, 200); ax.set_ylim(-0.45, 1.1)
ax.set_xlabel("time relative to spike (ms)"); ax.set_ylabel("scaled to peak 1")
ax.set_title("Same, shape only: bars = width at half height", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
ax.text(0, 1, "Width at half height\n(schematic cell vs measured)\n\n" + "\n".join(rows) +
        "\n\nWhat the blur does:\n"
        "• wider time course (≈ one frame at slow rates)\n"
        "• lower peak: the response is spread out\n"
        "• the opposite lobe is smeared into the main lobe\n   and partly cancelled → looks less two-lobed\n"
        "• signal after the spike (frame still on screen)\n"
        "• peak time moves little (the smear is symmetric)",
        va="top", fontsize=9.5)

fig.suptitle("How noise frames blur the STA time course (schematic; true response drawn to resemble the 15/10 cells)",
             fontsize=11, x=0.01, ha="left")
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(OUT)
