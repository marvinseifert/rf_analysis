# AI code
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # put this before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path

from skimage import io, exposure, filters, feature, morphology, measure
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# =============================================================================
# User settings
# =============================================================================

IMAGE_PATH = r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish MEA + imaging\20260225_dragonfly_microscope_test\overlays\tissue_with_grid.tif"
EXPECTED_PITCH_UM = 100.0  # adjacent electrode centre-to-centre spacing
N_ROIS = 3

# If None, you will manually choose ROIs.
# If you want fixed ROIs, set this to a list like:
# FIXED_ROIS = [(r0, r1, c0, c1), (r0, r1, c0, c1), (r0, r1, c0, c1)]
FIXED_ROIS = None

# Autocorrelation peak finding
AC_PEAK_MIN_DISTANCE = 8
CENTER_EXCLUSION_RADIUS_PX = 20
MAX_NUM_AC_PEAKS = 80
FIRST_SHELL_TOL = 0.20

# Optional blob-based NN check
SHOW_BLOB_CHECK = True

# Plot settings
MAIN_FIG_WIDTH = 16
MAIN_FIG_ROW_HEIGHT = 4.8


# =============================================================================
# I/O
# =============================================================================


def load_image(path):
    path = str(Path(path))

    try:
        import tifffile

        img = tifffile.imread(path)
    except Exception:
        img = io.imread(path)

    img = np.asarray(img)

    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    else:
        img = img.astype(np.float32)
        if img.max() > 1:
            img = img / img.max()

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    return img


# =============================================================================
# Manual selection helpers
# =============================================================================


def manual_um_per_pixel(rgb, pitch_um=100.0):
    # Explore/zoom first
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.set_title("Zoom/pan to two adjacent electrodes, then close this window")
    plt.show()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Click at same zoom
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Click TWO adjacent electrode centres")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(pts) != 2:
        raise RuntimeError("You must click exactly two adjacent electrode centres.")

    pts = np.array(pts, dtype=float)
    dist_px = np.sqrt(np.sum((pts[1] - pts[0]) ** 2))
    um_per_px = pitch_um / dist_px
    return um_per_px, dist_px, pts


def manual_rois(rgb, n_rois=3):
    rois = []
    roi_pts = []

    for i in range(n_rois):
        # Explore/zoom first
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb)
        ax.set_title(
            f"ROI {i+1}/{n_rois}: zoom/pan to a clean region, then close this window"
        )
        plt.show()

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Click corners at same zoom
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"ROI {i+1}/{n_rois}: click top-left then bottom-right")
        pts = plt.ginput(2, timeout=0)
        plt.close(fig)

        if len(pts) != 2:
            raise RuntimeError(f"You must click exactly two points for ROI {i+1}.")

        pts = np.array(pts, dtype=float)
        c0, r0 = pts[0]
        c1, r1 = pts[1]

        r0, r1 = sorted([int(round(r0)), int(round(r1))])
        c0, c1 = sorted([int(round(c0)), int(round(c1))])

        rois.append((r0, r1, c0, c1))
        roi_pts.append(pts)

    return rois, roi_pts


# =============================================================================
# Image preprocessing
# =============================================================================


def make_retina_intensity_image(rgb):
    # suppress green overlay by using red+blue channels only
    rb = 0.5 * (rgb[..., 0] + rgb[..., 2])
    rb = exposure.rescale_intensity(rb, in_range="image", out_range=(0, 1))
    return rb


def crop_image(img, roi):
    r0, r1, c0, c1 = roi
    return img[r0:r1, c0:c1]


def auto_mask_retina(intensity):
    blur = filters.gaussian(intensity, sigma=6)
    thresh = filters.threshold_otsu(blur)
    mask = blur > max(thresh * 0.6, 0.03)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, min_size=5000)

    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        return np.ones_like(mask, dtype=bool)

    largest = max(props, key=lambda p: p.area)
    return labels == largest.label


def preprocess_for_lattice(intensity, mask):
    low = filters.gaussian(intensity, sigma=12)
    hp = intensity - low
    hp = hp * mask.astype(float)

    vals = hp[mask]
    hp = hp - np.median(vals)
    sd = np.std(vals)
    if sd > 0:
        hp = hp / sd

    wy = np.hanning(hp.shape[0])
    wx = np.hanning(hp.shape[1])
    window = np.outer(wy, wx)

    return hp * window


# =============================================================================
# Lattice analysis
# =============================================================================


def compute_fft_power(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(F) ** 2)


def compute_autocorrelation(img):
    F = np.fft.fft2(img)
    ac = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)).real)
    ac = ac / np.max(ac)
    return ac


def detect_autocorr_peaks(ac, min_distance=8, exclude_radius=20, num_peaks=80):
    cy, cx = np.array(ac.shape) // 2

    yy, xx = np.indices(ac.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    work = ac.copy()
    work[rr < exclude_radius] = 0

    coords = feature.peak_local_max(
        work,
        min_distance=min_distance,
        threshold_rel=0.08,
        num_peaks=num_peaks,
        exclude_border=False,
    )

    if len(coords) == 0:
        return coords, np.empty((0, 2)), np.array([]), np.array([])

    vals = ac[coords[:, 0], coords[:, 1]]
    order = np.argsort(vals)[::-1]
    coords = coords[order]

    vectors = np.column_stack([coords[:, 1] - cx, coords[:, 0] - cy])  # dx, dy
    radii = np.sqrt(np.sum(vectors**2, axis=1))
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))

    return coords, vectors, radii, angles


def first_shell_mask(radii, tol=0.20):
    if len(radii) == 0:
        return np.array([], dtype=bool)

    r0 = np.min(radii)
    mask = np.abs(radii - r0) <= tol * r0

    if mask.sum() < 4:
        mask = np.abs(radii - r0) <= (tol * 1.5) * r0

    return mask


def blob_nn_spacing(intensity, mask, um_per_px, debug=False):
    """
    Optional sanity-check only.
    Tries to detect one blob per bright cone-like object and compute
    nearest-neighbour distance.
    """
    from skimage.feature import blob_log

    sm = ndi.gaussian_filter(intensity, sigma=1.2)

    blobs = blob_log(
        sm,
        min_sigma=2.0,
        max_sigma=5.0,
        num_sigma=15,
        threshold=0.03,
        overlap=0.5,
    )

    if len(blobs) < 5:
        return None

    rows = blobs[:, 0]
    cols = blobs[:, 1]
    sigmas = blobs[:, 2]

    dist_to_edge = ndi.distance_transform_edt(mask)
    keep = dist_to_edge[rows.astype(int), cols.astype(int)] > 4
    rows = rows[keep]
    cols = cols[keep]
    sigmas = sigmas[keep]

    if len(rows) < 5:
        return None

    pts_xy = np.column_stack([cols, rows])
    tree = cKDTree(pts_xy)
    dists, _ = tree.query(pts_xy, k=2)
    nn_px = dists[:, 1]

    result = {
        "points_rc": np.column_stack([rows, cols]),
        "sigmas": sigmas,
        "nn_px_median": float(np.median(nn_px)),
        "nn_px_mean": float(np.mean(nn_px)),
        "nn_um_median": float(np.median(nn_px) * um_per_px),
        "nn_um_mean": float(np.mean(nn_px) * um_per_px),
    }

    if debug:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(intensity, cmap="gray")
        for r, c, s in zip(rows, cols, sigmas):
            circ = plt.Circle(
                (c, r), radius=np.sqrt(2) * s, fill=False, color="lime", linewidth=1
            )
            ax.add_patch(circ)
        ax.set_title(f"Blob check\nNN median = {result['nn_um_median']:.2f} µm")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return result


# =============================================================================
# Per-ROI analysis
# =============================================================================


def analyze_one_roi(retina, roi, um_per_px):
    retina_crop = crop_image(retina, roi)
    mask_crop = auto_mask_retina(retina_crop)
    proc = preprocess_for_lattice(retina_crop, mask_crop)

    fft_power = compute_fft_power(proc)
    ac = compute_autocorrelation(proc)

    peak_coords, peak_vecs, peak_radii_px, peak_angles_deg = detect_autocorr_peaks(
        ac,
        min_distance=AC_PEAK_MIN_DISTANCE,
        exclude_radius=CENTER_EXCLUSION_RADIUS_PX,
        num_peaks=MAX_NUM_AC_PEAKS,
    )

    shell_mask = first_shell_mask(peak_radii_px, tol=FIRST_SHELL_TOL)

    if shell_mask.sum() == 0:
        raise RuntimeError(
            f"No first-shell peaks found for ROI {roi}. Try a cleaner ROI."
        )

    shell_r_px = peak_radii_px[shell_mask]
    shell_r_um = shell_r_px * um_per_px
    shell_angles = np.mod(peak_angles_deg[shell_mask], 180.0)
    shell_vecs = peak_vecs[shell_mask]

    nn_check = (
        blob_nn_spacing(retina_crop, mask_crop, um_per_px, debug=False)
        if SHOW_BLOB_CHECK
        else None
    )

    return {
        "roi": roi,
        "retina_crop": retina_crop,
        "mask_crop": mask_crop,
        "proc": proc,
        "fft_power": fft_power,
        "ac": ac,
        "peak_coords": peak_coords,
        "peak_vecs": peak_vecs,
        "peak_radii_px": peak_radii_px,
        "peak_angles_deg": peak_angles_deg,
        "shell_mask": shell_mask,
        "shell_r_px": shell_r_px,
        "shell_r_um": shell_r_um,
        "shell_angles": shell_angles,
        "shell_vecs": shell_vecs,
        "nn_check": nn_check,
        "spacing_um_median": float(np.median(shell_r_um)),
        "spacing_um_mean": float(np.mean(shell_r_um)),
        "spacing_um_std": float(np.std(shell_r_um)),
    }


# =============================================================================
# Main
# =============================================================================

rgb = load_image(IMAGE_PATH)

# 1) Calibration once
um_per_px, pitch_px, cal_pts = manual_um_per_pixel(rgb, pitch_um=EXPECTED_PITCH_UM)

# 2) ROI selection
if FIXED_ROIS is not None:
    rois = FIXED_ROIS
    roi_pts_list = None
else:
    rois, roi_pts_list = manual_rois(rgb, n_rois=N_ROIS)

# 3) Convert to analysis image
retina = make_retina_intensity_image(rgb)

# 4) Analyze each ROI
results = []
for i, roi in enumerate(rois):
    print(f"\nAnalyzing ROI {i+1}/{len(rois)}: {roi}")
    result = analyze_one_roi(retina, roi, um_per_px)
    results.append(result)

# 5) Summary stats across ROIs
roi_medians = np.array([r["spacing_um_median"] for r in results], dtype=float)
roi_means = np.array([r["spacing_um_mean"] for r in results], dtype=float)

overall_mean = float(np.mean(roi_means))
overall_sd = float(np.std(roi_means, ddof=1)) if len(roi_means) > 1 else 0.0
overall_median = float(np.median(roi_medians))

# =============================================================================
# Print summary
# =============================================================================

print("\n=== Calibration ===")
print(f"Clicked electrode spacing: {pitch_px:.2f} px = {EXPECTED_PITCH_UM:.1f} µm")
print(f"Scale: {um_per_px:.4f} µm/px")

print("\n=== Per-ROI lattice spacing ===")
for i, r in enumerate(results):
    print(
        f"ROI {i+1}: median = {r['spacing_um_median']:.2f} µm, "
        f"mean = {r['spacing_um_mean']:.2f} µm, "
        f"std = {r['spacing_um_std']:.2f} µm"
    )

print("\n=== Overall across ROIs ===")
print(f"ROI medians (µm): {np.round(roi_medians, 2)}")
print(f"ROI means   (µm): {np.round(roi_means, 2)}")
print(f"Overall mean spacing = {overall_mean:.2f} ± {overall_sd:.2f} µm")
print(f"Overall median of ROI medians = {overall_median:.2f} µm")

# =============================================================================
# Figure 1: overview with calibration + all ROIs
# =============================================================================

fig0, ax0 = plt.subplots(figsize=(9, 9))
ax0.imshow(rgb)
ax0.plot(cal_pts[:, 0], cal_pts[:, 1], "yo-", linewidth=1.5, markersize=6)
mid = cal_pts.mean(axis=0)
ax0.text(
    mid[0],
    mid[1],
    f"{EXPECTED_PITCH_UM:.0f} µm",
    color="yellow",
    fontsize=10,
    ha="center",
    va="bottom",
    bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"),
)

roi_colors = ["cyan", "magenta", "lime", "orange", "white", "red"]
for i, roi in enumerate(rois):
    r0, r1, c0, c1 = roi
    color = roi_colors[i % len(roi_colors)]
    ax0.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0], color=color, linewidth=1.8)
    ax0.text(
        c0,
        r0 - 5,
        f"ROI {i+1}",
        color=color,
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.4, boxstyle="round"),
    )

ax0.set_title("Original image with calibration and selected ROIs")
ax0.axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# =============================================================================
# Figure 2: one row per ROI
# =============================================================================

fig, axes = plt.subplots(
    len(results), 3, figsize=(MAIN_FIG_WIDTH, MAIN_FIG_ROW_HEIGHT * len(results))
)

if len(results) == 1:
    axes = np.array([axes])

for i, r in enumerate(results):
    # Column 1: retina crop
    ax = axes[i, 0]
    ax.imshow(r["retina_crop"], cmap="gray")
    ax.contour(r["mask_crop"], levels=[0.5], colors="cyan", linewidths=0.7)
    ax.set_title(
        f"ROI {i+1}\nCrop used for analysis\nmedian spacing = {r['spacing_um_median']:.2f} µm"
    )
    ax.axis("off")

    # Column 2: FFT
    ax = axes[i, 1]
    ax.imshow(r["fft_power"], cmap="magma")
    cyf, cxf = np.array(r["fft_power"].shape) // 2
    ax.scatter([cxf], [cyf], c="cyan", s=15)
    ax.set_title(f"ROI {i+1}\n2D FFT power spectrum")
    ax.axis("off")

    # Column 3: autocorrelation
    ax = axes[i, 2]
    ax.imshow(r["ac"], cmap="viridis")
    cy, cx = np.array(r["ac"].shape) // 2
    ax.scatter([cx], [cy], c="red", s=20, label="center")

    if len(r["peak_coords"]) > 0:
        ax.scatter(
            r["peak_coords"][:, 1],
            r["peak_coords"][:, 0],
            s=20,
            facecolors="none",
            edgecolors="white",
            linewidths=0.7,
            label="all peaks",
        )

        ax.scatter(
            r["peak_coords"][r["shell_mask"], 1],
            r["peak_coords"][r["shell_mask"], 0],
            s=45,
            facecolors="none",
            edgecolors="yellow",
            linewidths=1.3,
            label="first shell",
        )

    for vec in r["shell_vecs"]:
        ax.plot([cx, cx + vec[0]], [cy, cy + vec[1]], color="yellow", linewidth=0.9)

    ax.set_title(
        f"ROI {i+1}\n2D autocorrelation\nmean spacing = {r['spacing_um_mean']:.2f} µm"
    )
    ax.axis("off")

    if i == 0:
        ax.legend(loc="lower right", fontsize=8)

fig.suptitle(
    f"Cone lattice analysis across {len(results)} ROIs\n"
    f"Overall mean spacing = {overall_mean:.2f} ± {overall_sd:.2f} µm",
    y=0.995,
    fontsize=14,
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# =============================================================================
# Figure 3: summary plot across ROIs
# =============================================================================

fig2, ax = plt.subplots(figsize=(6, 4))
x = np.arange(1, len(results) + 1)

ax.plot(x, roi_means, "o-", label="ROI mean spacing")
ax.axhline(overall_mean, linestyle="--", color="red", label="Overall mean")
ax.fill_between(
    [0.5, len(results) + 0.5],
    overall_mean - overall_sd,
    overall_mean + overall_sd,
    color="red",
    alpha=0.2,
    label="±1 SD",
)

ax.set_xticks(x)
ax.set_xlabel("ROI")
ax.set_ylabel("Spacing (µm)")
ax.set_title("Cone lattice spacing across ROIs")
ax.legend()
plt.tight_layout()
plt.show()
