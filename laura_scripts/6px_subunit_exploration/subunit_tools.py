"""
Shared "subunit-like" score for covariance RF maps (used by plot_cells_with_stas.py and
compare_6px_8px_subunit_cells.py). Moved here from plot_cells_with_stas.py on 20/11/2026, unchanged.

A peak in the smoothed covariance map counts as an extra peak if it is
  - >= SUBUNIT_MIN_PEAK x the main peak,
  - >= SUBUNIT_MIN_SEPARATION_UM from any higher peak,
  - within the cell's largest centre-outline radius of the main peak,
  - separated from the main peak by a dip <= SUBUNIT_MAX_DIP x the smaller peak, and
  - that dip is >= SUBUNIT_MIN_DIP_NOISE_SD x the noise SD of the smoothed map
    (measured > SUBUNIT_NOISE_DISTANCE_UM from the main peak, only where the map has values).
The map's grid is the 2 µm stimulus-file grid (the 2 µm shuffle makes it finer than the noise pixel).
"""
import numpy as np
from scipy import ndimage

SUBUNIT_QUALITY_MIN = 10
SUBUNIT_SMOOTH_UM = 4.0
SUBUNIT_MIN_PEAK = 0.5
SUBUNIT_MIN_SEPARATION_UM = 12.0
SUBUNIT_MAX_DIP = 0.85
SUBUNIT_MIN_DIP_NOISE_SD = 3.0  # dip depth must be >= this many noise SDs
SUBUNIT_NOISE_DISTANCE_UM = 200.0  # noise SD measured on the smoothed map this far from the main peak


def subunit_score(cm_map, step_um, centre_radius_um):
    """cm_map (y, x), per-cell normalised. Returns (n_extra_peaks, dip_sum, peaks_yx)."""
    finite = np.isfinite(cm_map)
    m = np.nan_to_num(cm_map, nan=0.0)
    s = ndimage.gaussian_filter(m, SUBUNIT_SMOOTH_UM / step_um)
    s = s / s.max()
    iy, ix = np.unravel_index(np.argmax(s), s.shape)
    yy, xx = np.indices(s.shape)
    far = finite & (np.hypot(yy - iy, xx - ix) * step_um > SUBUNIT_NOISE_DISTANCE_UM)
    noise_sd = float(np.std(s[far])) if far.sum() > 100 else np.inf
    sep_px = max(1, int(round(SUBUNIT_MIN_SEPARATION_UM / step_um)))
    is_max = (s == ndimage.maximum_filter(s, size=2 * sep_px + 1)) & (s >= SUBUNIT_MIN_PEAK)
    cand = [(s[y, x], y, x) for y, x in zip(*np.nonzero(is_max))
            if (y, x) != (iy, ix) and np.hypot(y - iy, x - ix) * step_um <= centre_radius_um]
    peaks, dips = [(iy, ix)], []
    for v, y, x in sorted(cand, reverse=True):
        n = int(np.hypot(y - iy, x - ix)) * 2 + 2
        line = s[np.linspace(iy, y, n).round().astype(int), np.linspace(ix, x, n).round().astype(int)]
        dip = line.min() / v
        if dip <= SUBUNIT_MAX_DIP and (v - line.min()) >= SUBUNIT_MIN_DIP_NOISE_SD * noise_sd:
            peaks.append((y, x))
            dips.append(1 - dip)
    return len(peaks) - 1, float(np.sum(dips)), peaks


def smooth_normalised(cm_map, step_um):
    """Positive part of the map, smoothed as in subunit_score, scaled to max 1 (for contours)."""
    s = ndimage.gaussian_filter(np.nan_to_num(np.clip(cm_map, 0, None)), SUBUNIT_SMOOTH_UM / step_um)
    return s / s.max()
