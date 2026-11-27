"""Finer timing per colour in the 15/10/2026 15 Hz colour noise (stim 5), same cells across colours
(27 Nov 2026). From results/colour_noise_comparison.csv + _timecourses.npy (colour_noise_comparison.py).

Per cell and colour (RF above chance): peak (parabolic interpolation between 10 ms bins), early and late
half-height crossings, centre of mass of the main lobe (ms before the spike; closer to 0 = faster).
Paired Wilcoxon tests 420 vs 500 nm, 420 nm vs white, 500 nm vs white; polarity per colour (colour
opponency check). 420 nm is not cone-isolating (also drives UV and green cones).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent / "results"
LAGS = -800 + 10 * np.arange(100)
df = pd.read_csv(HERE / "colour_noise_comparison.csv")
tcs = np.load(HERE / "colour_noise_comparison_timecourses.npy")
df["row"] = np.arange(len(df))
d = df[(df.recording == "15/10 (15 Hz)") & (df.stim == 5) & df.rf]


def measures(y):
    j = np.argmax(np.abs(y) * (LAGS <= 0)); y = y * np.sign(y[j]) / abs(y[j])
    a, b, c = y[j - 1], y[j], y[j + 1]
    off = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0
    lo = j
    while lo > 0 and y[lo] > 0.5:
        lo -= 1
    hi = j
    while hi < 99 and y[hi] > 0.5:
        hi += 1
    early = LAGS[lo] + 10 * (0.5 - y[lo]) / (y[lo + 1] - y[lo])
    late = LAGS[hi - 1] + 10 * (y[hi - 1] - 0.5) / (y[hi - 1] - y[hi])
    z = lo
    while z > 0 and y[z] > 0:
        z -= 1
    e = hi
    while e < 99 and y[e] > 0:
        e += 1
    w = np.clip(y[z:e + 1], 0, None)
    return LAGS[j] + 10 * off, early, late, (LAGS[z:e + 1] * w).sum() / w.sum()


m = pd.DataFrame([dict(cell=r.cell, colour=r.colour, pol=r.polarity,
                       **dict(zip(["peak", "early_half", "late_half", "centroid"], measures(tcs[r.row].astype(float)))))
                  for _, r in d.iterrows()])
m.to_csv(HERE / "colour_timing_detail_15_10.csv", index=False)
print(m.groupby("colour")[["peak", "early_half", "late_half", "centroid"]].median().round(1))
for a, b in [("500", "420"), ("white", "420"), ("white", "500")]:
    print(f"{b} vs {a}:")
    for k in ["peak", "early_half", "late_half", "centroid"]:
        pv = m.pivot(index="cell", columns="colour", values=k)[[a, b]].dropna()
        print(f"   {k}: {b}-{a} {np.median(pv[b] - pv[a]):+.1f} ms (n={len(pv)}, "
              f"p={stats.wilcoxon(pv[a], pv[b]).pvalue:.3f}, {b} faster in {int((pv[b] > pv[a]).sum())}/{len(pv)})")
print(m.pivot(index="cell", columns="colour", values="pol").fillna("-").value_counts().head(8))
