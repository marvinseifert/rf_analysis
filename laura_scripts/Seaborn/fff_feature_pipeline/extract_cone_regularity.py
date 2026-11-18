"""
extract_cone_regularity.py — v2 version, using the v2 alignment (2026-11-18).
The original (v1) is in laura_scripts/legacy/Seaborn/fff_feature_pipeline/.

Differences:
  * Needs a v2 noise_data file (RF maps in stimulus coordinates; see
    combined_analysis/convert_noise_data_v2.py).
  * Geometry comes from alignment_geometry.load_alignment_geometry (one shared
    loader): displayed size and dataset pixel size are read from the alignment
    cache unless given, and the stimulus centre is recomputed from the clicked edge.
  * RF centre: strongest pixel in stimulus coordinates, placed in the projection
    with the recording's stimulus -> projection transform (date-dependent: before
    8 Oct 2026 (x, y) -> (-y, -x); from 8 Oct 2026 (x, y) -> (y, -x)). No data flip.
  * Autocorrelation: the centre exclusion follows the SHAPE of the central peak
    (connected positive region around zero shift) instead of a fixed 20 px disc,
    and peaks touching the excluded region are rejected. The fixed 4.9 µm disc let
    elongated central peaks through, giving a spurious ~4.9 µm "spacing" for every
    cell. New column: cone_centre_exclusion_um (furthest extent of the exclusion).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import imageio.v3 as iio
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation, label, map_coordinates
from skimage import exposure, feature, filters
from tqdm import tqdm

from laura_scripts.Alignment_pipeline.alignment_geometry import (
    dataset_um_per_px,
    load_alignment_geometry,
    require_v2_rf_frame,
)


@dataclass
class ConeRegularityConfig:
    """Settings for cone-mosaic regularity around each aligned RF centre."""

    noise_data_path: Path
    cone_image_path: Path
    alignment_cache_path: Path
    rf_channel: str

    # RF map used to define the centre (the same map the alignment script shows).
    # For covariance RFs the centre is the strongest absolute value.
    rf_variable: str = "cm_most_important"

    # 200 means a 200 x 200 um square crop.
    crop_size_um: float = 200.0

    # Per-recording geometry, normally left as None:
    #   dataset_assumed_um_per_px -> noise_pixel_size in noise_params.json
    #   displayed_stim_*          -> this recording's v2 alignment cache
    dataset_assumed_um_per_px: float | None = None
    displayed_stim_width_px: int | None = None
    displayed_stim_height_px: int | None = None

    # Autocorrelation settings.
    preprocess_sigma_px: float = 13.0
    ac_peak_min_distance_px: int = 10
    centre_exclusion_radius_px: int = 20
    # v2: exclude the central autocorrelation peak by its SHAPE: the connected
    # region around zero shift where the (normalised) autocorrelation exceeds
    # central_peak_threshold, plus a disc of centre_exclusion_radius_px as a
    # minimum. Peaks touching the excluded region are rejected (cut-off artefacts:
    # the central peak's tail just falls away from the edge). The threshold keeps
    # the region compact; at 0 it leaked along cone rows.
    # With a fixed 20 px (4.9 µm) disc, elongated cones made the central peak
    # stick out of the disc and its two tips were picked as the "first shell"
    # (spacing ~4.9 µm for every cell).
    adaptive_centre_exclusion: bool = True
    central_peak_threshold: float = 0.25
    max_num_ac_peaks: int = 80
    ac_peak_threshold_rel: float = 0.08
    first_shell_tol: float = 0.25
    min_first_shell_peaks: int = 4

    min_valid_crop_fraction: float = 0.95

    # None = use approximately the native cone-image physical resolution.
    resample_um_per_px: float | None = None


def _transform_points(A: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=float)
    hom = np.column_stack(
        [
            points_xy[:, 0],
            points_xy[:, 1],
            np.ones(len(points_xy), dtype=float),
        ]
    )
    return (hom @ np.asarray(A, dtype=float).T)[:, :2]


def _prepare_cone_intensity(image: np.ndarray) -> np.ndarray:
    """
    Convert the cone image to one normalised intensity plane.

    For RGB images, use mean(red, blue), matching the existing cone
    autocorrelation script and suppressing a green MEA overlay.
    """
    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim == 2:
        intensity = image.astype(float)

    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        rgb = image[..., :3].astype(float)
        intensity = rgb[..., [0, 2]].mean(axis=-1)

    elif image.ndim == 3 and image.shape[0] in (3, 4):
        rgb = np.moveaxis(image[:3], 0, -1).astype(float)
        intensity = rgb[..., [0, 2]].mean(axis=-1)

    else:
        raise ValueError(
            "Cone image must be a 2-D image or RGB/RGBA image. "
            f"Got shape {image.shape}."
        )

    finite = intensity[np.isfinite(intensity)]
    if finite.size == 0:
        raise ValueError("Cone image contains no finite pixels.")

    return exposure.rescale_intensity(
        intensity,
        in_range="image",
        out_range=(0.0, 1.0),
    ).astype(float)


def _load_alignment_geometry(config: ConeRegularityConfig) -> dict:
    """Cone-image <-> projected-µm geometry from the shared v2 loader."""
    geometry = load_alignment_geometry(
        config.alignment_cache_path,
        dataset_assumed_um_per_px=config.dataset_assumed_um_per_px,
        displayed_stim_width_px=config.displayed_stim_width_px,
        displayed_stim_height_px=config.displayed_stim_height_px,
    )
    A_image1_to_projected_um = geometry.A_image1_to_projected_um
    linear = A_image1_to_projected_um[:2, :2]
    return {
        "rf_scale_factor": float(geometry.rf_scale_factor),
        # date-dependent stimulus -> projection transform (see alignment_geometry)
        "A_raw_rf_to_projected_um": geometry.A_raw_rf_to_projected_um,
        "corrected_stim_um_per_px": float(geometry.corrected_stim_um_per_px),
        "A_image1_to_projected_um": A_image1_to_projected_um,
        "A_projected_um_to_image1": np.linalg.inv(A_image1_to_projected_um),
        # Equivalent isotropic physical pixel size for crop resampling.
        "native_cone_um_per_px": float(np.sqrt(abs(np.linalg.det(linear)))),
    }


def _rf_centre_projected_um(
    dataset: xr.Dataset,
    cell_index: int,
    config: ConeRegularityConfig,
    rf_scale_factor: float,
    A_raw_rf_to_projected_um: np.ndarray,
) -> tuple[float, float]:
    """
    Centre of the cell's RF in projected-stimulus µm (the frame the cone image
    is aligned to).

        1. Select config.rf_variable / config.rf_channel / the cell.
        2. Strongest pixel (|value| for signed covariance) in STIMULUS coordinates.
        3. Scale by rf_scale_factor and place in the projection with the
           recording's stimulus -> projection transform (date-dependent: before
           8 Oct 2026 (x, y) -> (-y, -x); from 8 Oct 2026 (x, y) -> (y, -x)).
    """
    if config.rf_variable not in dataset:
        raise KeyError(
            f"noise_data does not contain dataset[{config.rf_variable!r}]."
        )
    da = dataset[config.rf_variable]
    if "channel" in da.dims:
        da = da.sel(channel=config.rf_channel)
    if "cell_index" not in da.dims:
        raise ValueError(
            f"dataset[{config.rf_variable!r}] does not contain a cell_index "
            f"dimension. Dims: {da.dims}"
        )
    # ORIENTATION [AXIS ORDER only] rows = y, columns = x for indexing. Not a rotation.
    da = da.sel(cell_index=int(cell_index)).transpose("y", "x")

    values = np.asarray(da.values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan, np.nan
    peak_metric = np.where(finite, np.abs(values), -np.inf)
    peak_y_i, peak_x_i = np.unravel_index(np.argmax(peak_metric), peak_metric.shape)

    stimulus_xy = np.array(
        [
            float(da["x"].isel(x=peak_x_i).item()) * rf_scale_factor,
            float(da["y"].isel(y=peak_y_i).item()) * rf_scale_factor,
        ]
    )
    # ORIENTATION [RF ROTATION] RF peak (stimulus coords) -> projected with the recording's
    # date-dependent transform (before 8 Oct 2026 (x, y) -> (-y, -x); from 8 Oct (y, -x)).
    projected_x_um, projected_y_um = _transform_points(
        A_raw_rf_to_projected_um, stimulus_xy[None, :]
    )[0]
    return float(projected_x_um), float(projected_y_um)


def _sample_aligned_square_crop(
    cone_intensity: np.ndarray,
    centre_projected_um: tuple[float, float],
    A_projected_um_to_image1: np.ndarray,
    crop_size_um: float,
    requested_um_per_px: float,
) -> tuple[np.ndarray, float, float]:
    """
    Sample an exact physical square from the correctly aligned cone mosaic.

    crop_size_um=200 means a 200 x 200 um square centred on the aligned RF
    centre.

    The square is defined in projected RF/stimulus coordinates and transformed
    into the native cone-image pixels. This correctly handles rotation and the
    affine alignment.
    """
    if crop_size_um <= 0:
        raise ValueError("crop_size_um must be > 0")

    if requested_um_per_px <= 0:
        raise ValueError("resample_um_per_px must be > 0")

    n_px = max(
        32,
        int(round(crop_size_um / requested_um_per_px)),
    )

    actual_um_per_px = crop_size_um / n_px

    offsets = (
        np.arange(
            n_px,
            dtype=float,
        )
        - (n_px - 1) / 2.0
    ) * actual_um_per_px

    x_um = float(centre_projected_um[0]) + offsets

    y_um = float(centre_projected_um[1]) + offsets

    xx_um, yy_um = np.meshgrid(
        x_um,
        y_um,
    )

    projected_points = np.column_stack(
        [
            xx_um.ravel(),
            yy_um.ravel(),
        ]
    )

    cone_points = _transform_points(
        A_projected_um_to_image1,
        projected_points,
    )

    cone_x = cone_points[:, 0]
    cone_y = cone_points[:, 1]

    h, w = cone_intensity.shape

    valid = (cone_x >= 0) & (cone_x <= w - 1) & (cone_y >= 0) & (cone_y <= h - 1)

    valid_fraction = float(np.mean(valid))

    sampled = map_coordinates(
        cone_intensity,
        [
            cone_y,
            cone_x,
        ],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    ).reshape(
        n_px,
        n_px,
    )

    return (
        sampled,
        float(actual_um_per_px),
        valid_fraction,
    )


def preprocess_for_lattice(
    intensity: np.ndarray,
    sigma_px: float = 13.0,
) -> np.ndarray:
    low = filters.gaussian(
        intensity,
        sigma=sigma_px,
    )

    hp = intensity - low

    wy = np.hanning(hp.shape[0])

    wx = np.hanning(hp.shape[1])

    window = np.outer(
        wy,
        wx,
    )

    return hp * window


def compute_autocorrelation(
    img: np.ndarray,
) -> np.ndarray:
    F = np.fft.fft2(img)

    power_spectrum = F * np.conj(F)

    ac = np.fft.fftshift(np.fft.ifft2(power_spectrum).real)

    peak = float(np.nanmax(ac))

    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(
            ac,
            dtype=float,
        )

    return ac / peak


def detect_autocorr_peaks(
    ac: np.ndarray,
    min_distance: int,
    exclude_radius: int,
    num_peaks: int,
    threshold_rel: float,
    exclude_mask: np.ndarray | None = None,
):
    cy, cx = np.array(ac.shape) // 2

    yy, xx = np.indices(ac.shape)

    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    if exclude_mask is None:
        exclude_mask = rr < exclude_radius

    work = ac.copy()

    work[exclude_mask] = 0

    coords = feature.peak_local_max(
        work,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=num_peaks,
        exclude_border=False,
    )

    # v2: a maximum on the edge of the excluded region is an artefact of the
    # cut-off (values just fall away from the central peak), not a lattice peak.
    if len(coords):
        touching = binary_dilation(exclude_mask, iterations=2)
        coords = coords[~touching[coords[:, 0], coords[:, 1]]]

    if len(coords) == 0:
        return (
            coords,
            np.empty((0, 2)),
            np.array([]),
            np.array([]),
        )

    vals = ac[
        coords[:, 0],
        coords[:, 1],
    ]

    order = np.argsort(vals)[::-1]

    coords = coords[order]

    vectors = np.column_stack(
        [
            coords[:, 1] - cx,
            coords[:, 0] - cy,
        ]
    )

    radii = np.sqrt(
        np.sum(
            vectors**2,
            axis=1,
        )
    )

    angles = np.degrees(
        np.arctan2(
            vectors[:, 1],
            vectors[:, 0],
        )
    )

    return (
        coords,
        vectors,
        radii,
        angles,
    )


def central_peak_mask(
    ac: np.ndarray, min_radius_px: int, threshold: float = 0.25
) -> np.ndarray:
    """
    Boolean mask of the central (zero-shift) autocorrelation peak: the connected
    region where ac > threshold (ac is normalised so the centre is 1) that
    contains the centre, joined with a disc of min_radius_px. Follows the peak's
    real shape, so elongated cones are covered.
    """
    cy, cx = np.array(ac.shape) // 2
    yy, xx = np.indices(ac.shape)
    mask = np.hypot(yy - cy, xx - cx) < min_radius_px
    regions, _ = label(ac > threshold)
    centre_region = regions[cy, cx]
    if centre_region > 0:
        mask |= regions == centre_region
    return mask


def first_shell_mask(
    radii: np.ndarray,
    tol: float,
) -> np.ndarray:
    if len(radii) == 0:
        return np.array(
            [],
            dtype=bool,
        )

    r0 = float(np.min(radii))

    mask = np.abs(radii - r0) <= tol * r0

    if mask.sum() < 4:
        mask = np.abs(radii - r0) <= (tol * 1.5) * r0

    return mask


def analyse_cone_crop(
    crop: np.ndarray,
    um_per_px: float,
    config: ConeRegularityConfig,
) -> dict[str, float]:
    """
    Calculate cone spacing and the custom 0-1 cone regularity index.

    cone_regularity_index =
        mean first-shell autocorrelation strength
        x
        1 / (1 + first-shell spacing CV)

    This is a custom bounded index, not a standard named crystallinity metric.
    """
    finite = np.isfinite(crop)

    if not np.any(finite):
        return _empty_autocorr_metrics(regularity=np.nan)

    filled = crop.copy()

    median_value = float(np.nanmedian(filled))

    filled[~finite] = median_value

    proc = preprocess_for_lattice(
        filled,
        sigma_px=config.preprocess_sigma_px,
    )

    ac = compute_autocorrelation(proc)

    exclude_mask = None
    exclude_radius_px = float(config.centre_exclusion_radius_px)
    if config.adaptive_centre_exclusion:
        exclude_mask = central_peak_mask(
            ac, int(config.centre_exclusion_radius_px), config.central_peak_threshold
        )
        cy, cx = np.array(ac.shape) // 2
        my, mx = np.nonzero(exclude_mask)
        exclude_radius_px = float(np.max(np.hypot(my - cy, mx - cx)))  # furthest extent

    (
        peak_coords,
        _,
        peak_radii_px,
        _,
    ) = detect_autocorr_peaks(
        ac,
        min_distance=(config.ac_peak_min_distance_px),
        exclude_radius=int(config.centre_exclusion_radius_px),
        exclude_mask=exclude_mask,
        num_peaks=(config.max_num_ac_peaks),
        threshold_rel=(config.ac_peak_threshold_rel),
    )

    shell_mask = first_shell_mask(
        peak_radii_px,
        tol=config.first_shell_tol,
    )

    n_shell = int(shell_mask.sum()) if len(shell_mask) else 0

    if n_shell == 0:
        empty = _empty_autocorr_metrics(regularity=0.0)
        empty["cone_centre_exclusion_um"] = exclude_radius_px * um_per_px
        return empty

    shell_r_px = peak_radii_px[shell_mask]

    shell_r_um = shell_r_px * um_per_px

    shell_coords = peak_coords[shell_mask]

    shell_peak_values = ac[
        shell_coords[:, 0],
        shell_coords[:, 1],
    ]

    mean_spacing = float(np.mean(shell_r_um))

    median_spacing = float(np.median(shell_r_um))

    spacing_std = float(np.std(shell_r_um))

    spacing_cv = np.nan if mean_spacing <= 0 else float(spacing_std / mean_spacing)

    autocorr_strength = float(
        np.clip(
            np.mean(shell_peak_values),
            0.0,
            1.0,
        )
    )

    if n_shell < config.min_first_shell_peaks or not np.isfinite(spacing_cv):
        spacing_consistency = 0.0
        regularity = 0.0

    else:
        spacing_consistency = float(
            np.clip(
                1.0 / (1.0 + spacing_cv),
                0.0,
                1.0,
            )
        )

        regularity = float(
            np.clip(
                autocorr_strength * spacing_consistency,
                0.0,
                1.0,
            )
        )

    return {
        "cone_regularity_index": regularity,
        "cone_autocorr_strength": autocorr_strength,
        "cone_spacing_consistency": spacing_consistency,
        "cone_spacing_um": median_spacing,
        "cone_spacing_mean_um": mean_spacing,
        "cone_spacing_std_um": spacing_std,
        "cone_spacing_cv": spacing_cv,
        "cone_first_shell_n_peaks": float(n_shell),
        "cone_centre_exclusion_um": exclude_radius_px * um_per_px,
    }


def _empty_autocorr_metrics(
    regularity: float,
) -> dict[str, float]:
    return {
        "cone_regularity_index": regularity,
        "cone_autocorr_strength": (0.0 if regularity == 0.0 else np.nan),
        "cone_spacing_consistency": (0.0 if regularity == 0.0 else np.nan),
        "cone_spacing_um": np.nan,
        "cone_spacing_mean_um": np.nan,
        "cone_spacing_std_um": np.nan,
        "cone_spacing_cv": np.nan,
        "cone_first_shell_n_peaks": 0.0,
        "cone_centre_exclusion_um": np.nan,
    }


def calculate_cone_regularity_features(
    noise_data_path: str | Path,
    cone_image_path: str | Path,
    alignment_cache_path: str | Path,
    rf_channel: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate cone-mosaic regularity around every correctly aligned RF centre.

    For each cell:
        1. Select the same RF map/channel used by the alignment script.
        2. Find the strongest RF pixel (stimulus coordinates).
        3. Scale it and place it in the projection with the recording's (date-dependent) transform
           (v2 alignment; no data flip).
        5. Use the cached cone alignment to sample a physical square centred
           on that RF centre.
        6. Run autocorrelation on that cone crop.

    One output row is returned per cell.
    """
    config = ConeRegularityConfig(
        noise_data_path=Path(noise_data_path),
        cone_image_path=Path(cone_image_path),
        alignment_cache_path=Path(alignment_cache_path),
        rf_channel=rf_channel,
        **kwargs,
    )

    dataset = xr.load_dataset(config.noise_data_path)
    require_v2_rf_frame(dataset, str(config.noise_data_path))
    if config.dataset_assumed_um_per_px is None:
        config.dataset_assumed_um_per_px = dataset_um_per_px(
            config.noise_data_path.parent, dataset
        )

    cone_image = iio.imread(config.cone_image_path)

    cone_intensity = _prepare_cone_intensity(cone_image)

    alignment = _load_alignment_geometry(config)

    requested_um_per_px = (
        alignment["native_cone_um_per_px"]
        if config.resample_um_per_px is None
        else float(config.resample_um_per_px)
    )

    print("\nCone-mosaic regularity")

    print(f"RF channel: {config.rf_channel}")

    print(
        f"RF centre source: strongest plotted {config.rf_variable} pixel "
        "(stimulus coords -> projection, v2 alignment)"
    )

    print(f"Crop: {config.crop_size_um:.1f} x " f"{config.crop_size_um:.1f} um")

    print(f"Resampled scale: approximately " f"{requested_um_per_px:.4f} um/px")

    print(f"RF coordinate scale factor: " f"{alignment['rf_scale_factor']:.4f}")

    rows: list[dict] = []

    for cell_index in tqdm(
        dataset["cell_index"].values,
        desc="Cone regularity",
        unit="cell",
    ):
        cell_index = int(cell_index)

        centre_x_um, centre_y_um = _rf_centre_projected_um(
            dataset=dataset,
            cell_index=cell_index,
            config=config,
            rf_scale_factor=(alignment["rf_scale_factor"]),
            A_raw_rf_to_projected_um=alignment["A_raw_rf_to_projected_um"],
        )

        row = {
            "cell_index": cell_index,
            "cone_rf_center_x_um": centre_x_um,
            "cone_rf_center_y_um": centre_y_um,
            "cone_crop_size_um": float(config.crop_size_um),
        }

        if not (np.isfinite(centre_x_um) and np.isfinite(centre_y_um)):
            row.update(_empty_autocorr_metrics(regularity=np.nan))

            row["cone_crop_valid_fraction"] = np.nan

            rows.append(row)

            continue

        (
            crop,
            actual_um_per_px,
            valid_fraction,
        ) = _sample_aligned_square_crop(
            cone_intensity=(cone_intensity),
            centre_projected_um=(
                centre_x_um,
                centre_y_um,
            ),
            A_projected_um_to_image1=(alignment["A_projected_um_to_image1"]),
            crop_size_um=(config.crop_size_um),
            requested_um_per_px=(requested_um_per_px),
        )

        row["cone_crop_um_per_px"] = actual_um_per_px

        row["cone_crop_valid_fraction"] = valid_fraction

        if valid_fraction < config.min_valid_crop_fraction:
            row.update(_empty_autocorr_metrics(regularity=np.nan))

            rows.append(row)

            continue

        row.update(
            analyse_cone_crop(
                crop=crop,
                um_per_px=(actual_um_per_px),
                config=config,
            )
        )

        rows.append(row)

    dataset.close()

    output = pd.DataFrame(rows)

    output["cell_index"] = output["cell_index"].astype(int)

    return output


if __name__ == "__main__":
    raise SystemExit(
        "Import calculate_cone_regularity_features() " "from build_fff_dataset.py."
    )
