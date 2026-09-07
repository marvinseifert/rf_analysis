from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates
from skimage import exposure, feature, filters
from tqdm import tqdm


@dataclass
class ConeRegularityConfig:
    """Settings for cone-mosaic regularity around each aligned RF centre."""

    noise_data_path: Path
    cone_image_path: Path
    alignment_cache_path: Path
    rf_channel: str

    # RF map used to define the centre. This is the SAME RF map that is
    # displayed by the alignment script. For covariance RFs the centre is the
    # strongest absolute value after applying the same fixed left-right data flip.
    rf_variable: str = "cm_most_important"

    # 200 means a 200 x 200 um square crop.
    crop_size_um: float = 200.0

    # Same scale assumptions used by the alignment script.
    dataset_assumed_um_per_px: float = 2.0  # NEED TO NOT HARDCODE
    displayed_stim_height_px: int = 800  # NEED TO NOT HARDCODE

    # Autocorrelation settings.
    preprocess_sigma_px: float = 13.0
    ac_peak_min_distance_px: int = 10
    centre_exclusion_radius_px: int = 20
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


def _cached_stimulus_edge_points(cache) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the cached top/bottom points of the clicked vertical stimulus edge.

    Current caches:
        stim_edge_top_px / stim_edge_bottom_px

    Older right-edge caches:
        stim_top_right_px / stim_bottom_right_px
    """
    if "stim_edge_top_px" in cache.files and "stim_edge_bottom_px" in cache.files:
        return (
            np.asarray(cache["stim_edge_top_px"], dtype=float),
            np.asarray(cache["stim_edge_bottom_px"], dtype=float),
        )

    if "stim_top_right_px" in cache.files and "stim_bottom_right_px" in cache.files:
        return (
            np.asarray(cache["stim_top_right_px"], dtype=float),
            np.asarray(cache["stim_bottom_right_px"], dtype=float),
        )

    raise KeyError(
        "Alignment cache does not contain stimulus-edge points. "
        "Expected either stim_edge_top_px/stim_edge_bottom_px or "
        "stim_top_right_px/stim_bottom_right_px."
    )


def _load_alignment_geometry(config: ConeRegularityConfig) -> dict:
    """
    Reconstruct the same cone-image <-> projected-RF alignment used by the
    alignment script.
    """
    cache = np.load(
        config.alignment_cache_path,
        allow_pickle=True,
    )

    required = (
        "tform_2_to_3_params",
        "um_per_px_image3",
        "visual_stim_center_px_image3",
    )

    missing = [key for key in required if key not in cache.files]
    if missing:
        raise KeyError(
            "Alignment cache is missing required entries: "
            f"{missing}. Run the alignment script first."
        )

    tform_2_to_3_params = np.asarray(
        cache["tform_2_to_3_params"],
        dtype=float,
    )

    um_per_px_image3 = float(np.asarray(cache["um_per_px_image3"]).squeeze())

    visual_stim_center_px_image3 = np.asarray(
        cache["visual_stim_center_px_image3"],
        dtype=float,
    )

    stim_edge_top_px, stim_edge_bottom_px = _cached_stimulus_edge_points(cache)

    clicked_edge_vec_px = stim_edge_bottom_px - stim_edge_top_px

    clicked_edge_length_px = float(np.linalg.norm(clicked_edge_vec_px))

    if clicked_edge_length_px == 0:
        raise ValueError("Cached stimulus top/bottom edge points are identical.")

    # Same scale correction as the alignment script.
    clicked_edge_length_um = clicked_edge_length_px * um_per_px_image3

    corrected_stim_um_per_px = clicked_edge_length_um / float(
        config.displayed_stim_height_px
    )

    rf_scale_factor = corrected_stim_um_per_px / config.dataset_assumed_um_per_px

    # projected-stimulus um -> Image 3 pixels
    px_per_um_image3 = 1.0 / um_per_px_image3

    stim_down_unit_px = clicked_edge_vec_px / clicked_edge_length_px

    stim_right_unit_px = np.array(
        [
            stim_down_unit_px[1],
            -stim_down_unit_px[0],
        ],
        dtype=float,
    )

    M_projected_um_to_image3 = np.column_stack(
        [
            stim_right_unit_px * px_per_um_image3,
            stim_down_unit_px * px_per_um_image3,
        ]
    )

    A_projected_um_to_image3 = np.eye(
        3,
        dtype=float,
    )

    A_projected_um_to_image3[:2, :2] = M_projected_um_to_image3

    A_projected_um_to_image3[:2, 2] = visual_stim_center_px_image3

    A_image3_to_projected_um = np.linalg.inv(A_projected_um_to_image3)

    # Same chain as the RF alignment script:
    # cone/Image1 pixels -> Image3 pixels -> projected stimulus um
    A_image1_to_projected_um = A_image3_to_projected_um @ tform_2_to_3_params

    A_projected_um_to_image1 = np.linalg.inv(A_image1_to_projected_um)

    # Equivalent isotropic physical pixel size for crop resampling.
    linear = A_image1_to_projected_um[:2, :2]

    native_cone_um_per_px = float(np.sqrt(abs(np.linalg.det(linear))))

    cache.close()

    return {
        "rf_scale_factor": float(rf_scale_factor),
        "corrected_stim_um_per_px": float(corrected_stim_um_per_px),
        "A_image1_to_projected_um": A_image1_to_projected_um,
        "A_projected_um_to_image1": A_projected_um_to_image1,
        "native_cone_um_per_px": native_cone_um_per_px,
    }


def _flip_rf_left_right(da: xr.DataArray) -> xr.DataArray:
    """
    Exact left-right RF flip used by RF_alignment_simplified.py.

    The RF DATA are reversed along x, while the x-coordinate labels are left
    unchanged. This moves the RF contents to the opposite physical x positions.
    """
    if "x" not in da.dims:
        raise ValueError("Input DataArray must include an 'x' dimension.")

    x_axis = da.get_axis_num("x")
    flipped_values = np.flip(da.values, axis=x_axis)

    return da.copy(
        data=flipped_values,
        deep=True,
    )


def _rf_centre_projected_um(
    dataset: xr.Dataset,
    cell_index: int,
    config: ConeRegularityConfig,
    rf_scale_factor: float,
) -> tuple[float, float]:
    """
    Locate the centre of the ACTUAL RF that is plotted by the alignment script.

    This deliberately does NOT use dataset["positions"]. Instead it reproduces
    the spatial operations applied to the RF image in RF_alignment_simplified.py:

        1. Select config.rf_variable and config.rf_channel.
        2. Select the requested cell.
        3. Apply the fixed left-right RF DATA flip, leaving x coordinates fixed.
        4. Find the strongest visible RF pixel. For signed covariance this is the
           largest absolute value, so either a strong positive or negative RF can
           define the centre.
        5. Convert that plotted array position through dataset x/y coordinates.
        6. Apply the same rf_scale_factor used to draw the RF map.

    Raw RF -> projected-stimulus coordinates is identity in the current alignment
    script, so these corrected x/y values are already projected-stimulus um.
    """
    if config.rf_variable not in dataset:
        raise KeyError(
            f"noise_data.nc does not contain dataset[{config.rf_variable!r}]."
        )

    da = dataset[config.rf_variable]

    if "channel" in da.dims:
        da = da.sel(channel=config.rf_channel)

    if "cell_index" not in da.dims:
        raise ValueError(
            f"dataset[{config.rf_variable!r}] does not contain a cell_index "
            f"dimension. Dims: {da.dims}"
        )

    da = da.sel(cell_index=int(cell_index)).transpose("y", "x")

    # EXACT same fixed left-right RF mirror used before plotting.
    da = _flip_rf_left_right(da)

    values = np.asarray(da.values, dtype=float)

    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan, np.nan

    # cm_most_important is signed. The visible RF centre is defined by the
    # strongest magnitude, matching the single-cell plotted-RF peak logic.
    peak_metric = np.abs(values)
    peak_metric[~finite] = -np.inf

    if np.all(peak_metric == -np.inf):
        return np.nan, np.nan

    peak_y_i, peak_x_i = np.unravel_index(
        np.argmax(peak_metric),
        peak_metric.shape,
    )

    # Because the flip changed the DATA but deliberately left coordinate labels
    # unchanged, these are the physical coordinates at which that flipped RF
    # peak is actually plotted.
    peak_raw_x_dataset_units = float(da["x"].isel(x=peak_x_i).item())
    peak_raw_y_dataset_units = float(da["y"].isel(y=peak_y_i).item())

    peak_projected_x_um = peak_raw_x_dataset_units * rf_scale_factor
    peak_projected_y_um = peak_raw_y_dataset_units * rf_scale_factor

    return (
        float(peak_projected_x_um),
        float(peak_projected_y_um),
    )


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
):
    cy, cx = np.array(ac.shape) // 2

    yy, xx = np.indices(ac.shape)

    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    work = ac.copy()

    work[rr < exclude_radius] = 0

    coords = feature.peak_local_max(
        work,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=num_peaks,
        exclude_border=False,
    )

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

    (
        peak_coords,
        _,
        peak_radii_px,
        _,
    ) = detect_autocorr_peaks(
        ac,
        min_distance=(config.ac_peak_min_distance_px),
        exclude_radius=(config.centre_exclusion_radius_px),
        num_peaks=(config.max_num_ac_peaks),
        threshold_rel=(config.ac_peak_threshold_rel),
    )

    shell_mask = first_shell_mask(
        peak_radii_px,
        tol=config.first_shell_tol,
    )

    n_shell = int(shell_mask.sum()) if len(shell_mask) else 0

    if n_shell == 0:
        return _empty_autocorr_metrics(regularity=0.0)

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
        2. Apply the exact same fixed left-right RF DATA flip.
        3. Find the strongest visible RF pixel in that plotted map.
        4. Convert that plotted position through the RF x/y coordinates and apply
           the same RF scale correction as the alignment script.
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
        "after alignment-script left-right flip"
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
