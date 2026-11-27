"""Shared time-course measures (27 Nov 2026). Time courses on the STA time axis (stimulus time relative
to the spike, -800..+190 ms in 10 ms bins; negative = before the spike), main peak before the spike.

width_and_peak : width at half height of the main lobe (linear interpolation) and peak time
integration    : integration time = area under the time course after scaling the main peak to 1
                 (as Sato & Kefalov 2025 for cones): main lobe (between the zero crossings around the
                 peak) and the opposite-sign lobe just before it (limited to 400 ms); 'both lobes' =
                 main + opposite = the whole response window.
Used by width_and_integration.py and chromatic_RFs/frequency_offsets/colour_noise_comparison.py.
"""
import numpy as np

LAGS = -800 + 10 * np.arange(100)


def width_and_peak(y, lags=LAGS, step=10.0):
    pre = lags <= 0
    i = np.argmax(np.abs(y) * pre)
    y = y * np.sign(y[i]) / np.abs(y[i])
    lo = i
    while lo > 0 and y[lo] > 0.5:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and y[hi] > 0.5:
        hi += 1
    if y[lo] > 0.5 or y[hi] > 0.5:
        return np.nan, lags[i], y
    t_lo = lags[lo] + step * (0.5 - y[lo]) / (y[lo + 1] - y[lo])
    t_hi = lags[hi - 1] + step * (y[hi - 1] - 0.5) / (y[hi - 1] - y[hi])
    return t_hi - t_lo, lags[i], y


def integration(y, t=LAGS, step=10.0):
    """Returns (peak time, main-lobe area, opposite-lobe area) in ms, main peak scaled to 1."""
    pre = t <= 0
    i = np.argmax(np.abs(y) * pre)
    y = y * np.sign(y[i]) / np.abs(y[i])
    ys = np.convolve(y, np.ones(3) / 3, mode="same") if step >= 10 else y
    lo = i
    while lo > 0 and ys[lo] > 0:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and ys[hi] > 0:
        hi += 1
    main = y[lo:hi + 1].clip(min=0).sum() * step
    op_end = lo
    op_start = op_end
    limit = int(400 / step)
    while op_start > 0 and ys[op_start] <= 0 and op_end - op_start < limit:
        op_start -= 1
    opp = (-y[op_start:op_end + 1]).clip(min=0).sum() * step
    return t[i], main, opp
