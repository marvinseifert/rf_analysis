"""
cone_lattice_v3.py — lattice regularity that does not depend on the lattice SHAPE (draft, 25/11/2026).

Why: extract_cone_regularity.py (v2) only counts autocorrelation peaks within +-25-37 % of the
NEAREST peak distance ("first shell") and needs >= 4 of them. Zebrafish cone rows are often a
stretched lattice: cones are much closer along a row (e.g. 9 µm) than to the next row
(e.g. 13 µm), so only the 2 along-row peaks are in the shell and a perfectly regular mosaic
scores 0. Shells with unequal distances also lower v2 for stretched lattices.

v3 (same crop, preprocessing and autocorrelation as v2):
  1. a = nearest autocorrelation peak (outside the central peak).
  2. b = nearest peak that is NOT along a (line angle to a >= MIN_ANGLE_DEG).
     These are the two lattice vectors, whatever the lattice shape (square, hexagonal,
     stretched, oblique).
  3. A regular lattice must also have peaks at a + b and a - b (and the mirror images, which
     the autocorrelation always has). The autocorrelation is read at the 8 predicted points
     +-a, +-b, +-(a+b), +-(a-b) (maximum within SEARCH_FRACTION * |a| of each point).
  4. lattice_regularity = mean of those 8 values (0-1). A disordered mosaic has peaks at a
     and b only by chance, and its a+b / a-b points fall between peaks, so it scores low.
  Also returned: spacings |a|, |b|, stretch = |b| / |a|, the angle between a and b, the
  direction of a (the along-row / closest-neighbour direction, in crop coordinates), and how
  many of the 4 predicted a+-b points have a detected peak nearby (lattice_points_found, 0-4).
"""

from __future__ import annotations

import numpy as np

from extract_cone_regularity import (
    ConeRegularityConfig,
    central_peak_mask,
    compute_autocorrelation,
    detect_autocorr_peaks,
    preprocess_for_lattice,
)

MIN_ANGLE_DEG = 30.0  # b must point at least this far away from the line of a
MAX_B_OVER_A = 3.0  # b further away than this x |a| -> no second lattice direction
SEARCH_FRACTION = 0.15  # read the autocorrelation within this x |a| of each predicted point
MATCH_FRACTION = 0.2  # a detected peak this close (x |a|) counts as "found"
# Scale-adaptive high-pass: cone spacing differs between recordings (~12 µm in 15/05, ~18-24 µm in
# 14/08 and 15/10), and a fixed 13 px high-pass removes the cone bodies when cones are large, so
# the autocorrelation then follows texture inside the cones. Pass 1 (FIRST_PASS_SIGMA_PX)
# measures the spacing |a|; pass 2 uses sigma = SIGMA_FRACTION_OF_SPACING x |a| (clipped).
ADAPTIVE_SIGMA = True
FIRST_PASS_SIGMA_PX = 40.0
SIGMA_FRACTION_OF_SPACING = 0.4
SIGMA_LIMITS_PX = (13.0, 60.0)

V3_COLUMNS = (
    "lattice_regularity",
    "lattice_spacing_a_um",
    "lattice_spacing_b_um",
    "lattice_stretch",
    "lattice_angle_ab_deg",
    "lattice_direction_a_deg",
    "lattice_points_found",
    "lattice_highpass_sigma_px",
)


def _empty(regularity: float) -> dict[str, float]:
    out = {k: np.nan for k in V3_COLUMNS}
    out["lattice_regularity"] = regularity
    return out


def _sample_max(ac: np.ndarray, centre_xy: np.ndarray, radius_px: float) -> float:
    """Maximum of ac within radius_px of centre_xy (x, y relative to the ac centre)."""
    cy, cx = np.array(ac.shape) // 2
    x0, y0 = cx + centre_xy[0], cy + centre_xy[1]
    r = max(radius_px, 1.0)
    xs = np.arange(int(np.floor(x0 - r)), int(np.ceil(x0 + r)) + 1)
    ys = np.arange(int(np.floor(y0 - r)), int(np.ceil(y0 + r)) + 1)
    xs, ys = xs[(xs >= 0) & (xs < ac.shape[1])], ys[(ys >= 0) & (ys < ac.shape[0])]
    if len(xs) == 0 or len(ys) == 0:
        return np.nan
    X, Y = np.meshgrid(xs, ys)
    inside = (X - x0) ** 2 + (Y - y0) ** 2 <= r**2
    return float(ac[Y[inside], X[inside]].max()) if inside.any() else np.nan


def lattice_from_autocorrelation(ac: np.ndarray, um_per_px: float, config: ConeRegularityConfig,
                                 return_details: bool = False):
    exclude = central_peak_mask(ac, int(config.centre_exclusion_radius_px), config.central_peak_threshold)
    coords, vectors, radii, _ = detect_autocorr_peaks(
        ac, min_distance=config.ac_peak_min_distance_px, exclude_radius=int(config.centre_exclusion_radius_px),
        exclude_mask=exclude, num_peaks=config.max_num_ac_peaks, threshold_rel=config.ac_peak_threshold_rel)
    details = dict(vectors=vectors, a=None, b=None, predicted=None)
    if len(radii) < 2:
        return (_empty(0.0), details) if return_details else _empty(0.0)
    order = np.argsort(radii)
    vectors, radii = vectors[order], radii[order]
    a = vectors[0].astype(float)
    ra = float(radii[0])
    # b: nearest peak not along the line of a
    cos_ab = np.abs(vectors @ a) / (radii * ra)
    ok = (cos_ab <= np.cos(np.deg2rad(MIN_ANGLE_DEG))) & (radii <= MAX_B_OVER_A * ra)
    if not ok.any():
        return (_empty(0.0), details) if return_details else _empty(0.0)
    b = vectors[np.flatnonzero(ok)[0]].astype(float)
    rb = float(np.hypot(*b))
    predicted = np.array([a, -a, b, -b, a + b, -(a + b), a - b, -(a - b)])
    values = np.array([_sample_max(ac, p, SEARCH_FRACTION * ra) for p in predicted])
    # a+-b points with a detected peak nearby
    found = 0
    for p in predicted[4:8:2]:  # a+b and a-b (mirror images are identical)
        d = np.min(np.hypot(*(vectors - p).T))
        found += 2 * int(d <= MATCH_FRACTION * ra)
    angle_ab = float(np.degrees(np.arccos(np.clip(abs(a @ b) / (ra * rb), 0, 1))))
    out = dict(
        lattice_regularity=float(np.clip(np.nanmean(values), 0.0, 1.0)),
        lattice_spacing_a_um=ra * um_per_px,
        lattice_spacing_b_um=rb * um_per_px,
        lattice_stretch=rb / ra,
        lattice_angle_ab_deg=angle_ab,
        lattice_direction_a_deg=float(np.degrees(np.arctan2(a[1], a[0])) % 180),
        lattice_points_found=float(found),
    )
    details.update(vectors=vectors, a=a, b=b, predicted=predicted, values=values)
    return (out, details) if return_details else out


def analyse_cone_crop_v3(crop: np.ndarray, um_per_px: float, config: ConeRegularityConfig,
                         return_details: bool = False):
    """v3 lattice metrics for one cone crop (same preprocessing / autocorrelation as v2)."""
    finite = np.isfinite(crop)
    if not finite.any():
        return (_empty(np.nan), {}) if return_details else _empty(np.nan)
    filled = np.where(finite, crop, float(np.nanmedian(crop)))
    sigma = float(config.preprocess_sigma_px)
    if ADAPTIVE_SIGMA:
        ac1 = compute_autocorrelation(preprocess_for_lattice(filled, sigma_px=FIRST_PASS_SIGMA_PX))
        first = lattice_from_autocorrelation(ac1, um_per_px, config)
        if np.isfinite(first["lattice_spacing_a_um"]):
            sigma = float(np.clip(SIGMA_FRACTION_OF_SPACING * first["lattice_spacing_a_um"] / um_per_px, *SIGMA_LIMITS_PX))
        else:
            sigma = FIRST_PASS_SIGMA_PX
    ac = compute_autocorrelation(preprocess_for_lattice(filled, sigma_px=sigma))
    result = lattice_from_autocorrelation(ac, um_per_px, config, return_details=return_details)
    (result[0] if return_details else result)["lattice_highpass_sigma_px"] = sigma
    if return_details:
        out, details = result
        details["ac"] = ac
        return out, details
    return result
