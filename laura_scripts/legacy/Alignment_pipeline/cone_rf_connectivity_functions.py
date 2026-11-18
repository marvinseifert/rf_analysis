"""
Reusable functions for single-cell cone-mosaic/RF connectivity analysis.

Coordinate convention
---------------------
All final positions are represented in the alignment pipeline's
"projected-stimulus" coordinate system:

    x: rightwards, in µm
    y: downwards, in µm
    origin: centre of the displayed visual stimulus

The RF coordinate scale and orientation follow the current RF alignment script.
The cone image is warped from microscope pixels directly into projected-stimulus
µm coordinates before blue-cone detection and mosaic interpolation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist
from skimage import exposure, filters, measure, morphology


CONE_TYPES = ("UV", "Blue", "Green", "Red")
CONE_COLOURS = {
    "UV": "magenta",
    "Blue": "dodgerblue",
    "Green": "limegreen",
    "Red": "red",
}


@dataclass
class AlignmentGeometry:
    """Transforms and scale factors copied from the final alignment logic."""

    tform_image1_to_image3: np.ndarray
    um_per_px_image3: float
    corrected_stim_um_per_px: float
    rf_scale_factor: float
    A_projected_um_to_image3: np.ndarray
    A_image3_to_projected_um: np.ndarray
    A_image1_to_projected_um: np.ndarray
    A_raw_rf_to_projected_um: np.ndarray
    visual_stim_center_px_image3: np.ndarray
    stim_top_left_px: np.ndarray
    stim_bottom_left_px: np.ndarray


@dataclass
class AlignedCrop:
    """Cone image resampled onto a regular projected-µm grid."""

    image: np.ndarray
    valid_mask: np.ndarray
    x_centres_um: np.ndarray
    y_centres_um: np.ndarray
    extent_um: tuple[float, float, float, float]
    um_per_px: float
    central_extent_um: tuple[float, float, float, float]


@dataclass
class RFData:
    """One RF on its corrected raw-RF coordinate grid."""

    values: np.ndarray
    raw_values: np.ndarray
    x_raw_um: np.ndarray
    y_raw_um: np.ndarray
    peak_raw_um: np.ndarray
    peak_projected_um: np.ndarray
    variable_name: str
    signed: bool
    normalised: bool
    quality: float | None
    tilt: float | None


@dataclass
class ConeMosaic:
    """Interpolated cone mosaic in both crop pixels and projected µm."""

    points_px: dict[str, np.ndarray]
    points_projected_um: dict[str, np.ndarray]
    blue_radius_um: float
    uv_radius_um: float
    red_long_radius_um: float
    green_long_radius_um: float
    red_short_radii_um: np.ndarray
    green_short_radii_um: np.ndarray
    red_angles_deg: np.ndarray
    green_angles_deg: np.ndarray
    row_angle_deg: float
    row_spacing_um: float
    detection_image: np.ndarray
    binary_clean: np.ndarray
    blob_table: list[dict[str, float]]


@dataclass
class ConnectivityResults:
    """Per-cone samples plus one summary record per cone type."""

    per_cone: list[dict[str, Any]]
    summary: dict[str, dict[str, float | int]]


# -----------------------------------------------------------------------------
# GENERAL HELPERS
# -----------------------------------------------------------------------------


def load_image(path: str | Path) -> np.ndarray:
    """Read a microscopy image while preserving its channels and dynamic range."""
    image = np.asarray(iio.imread(Path(path)))
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return image


def transform_points(A: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine transform to Nx2 x/y points."""
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=float)
    points_xy = np.atleast_2d(points_xy)
    points_h = np.column_stack([points_xy, np.ones(len(points_xy))])
    transformed = points_h @ np.asarray(A, dtype=float).T
    return transformed[:, :2]


def coordinate_edges_from_centres(coords: np.ndarray) -> np.ndarray:
    """Convert evenly spaced coordinate centres to image edges."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 1 or len(coords) < 2:
        raise ValueError("coords must be a 1D array with at least two values")
    diffs = np.diff(coords)
    edges = np.empty(len(coords) + 1, dtype=float)
    edges[1:-1] = 0.5 * (coords[:-1] + coords[1:])
    edges[0] = coords[0] - 0.5 * diffs[0]
    edges[-1] = coords[-1] + 0.5 * diffs[-1]
    return edges


def _ensure_points(points: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.atleast_2d(arr)


def _normalise_for_display(image: np.ndarray) -> np.ndarray:
    """Robustly map an image to 0-1 for plotting, without altering analysis data."""
    arr = np.asarray(image, dtype=float)
    out = np.zeros_like(arr, dtype=float)

    if arr.ndim == 2:
        finite = np.isfinite(arr)
        if finite.any():
            p1, p99 = np.nanpercentile(arr[finite], [1, 99])
            if p99 > p1:
                out = np.clip((arr - p1) / (p99 - p1), 0, 1)
            else:
                out[finite] = arr[finite]
        return out

    for channel_i in range(arr.shape[-1]):
        channel = arr[..., channel_i]
        finite = np.isfinite(channel)
        if finite.any():
            p1, p99 = np.nanpercentile(channel[finite], [1, 99])
            if p99 > p1:
                out[..., channel_i] = np.clip((channel - p1) / (p99 - p1), 0, 1)
            else:
                out[..., channel_i][finite] = channel[finite]
    return out


# -----------------------------------------------------------------------------
# ALIGNMENT GEOMETRY
# -----------------------------------------------------------------------------


def estimate_transformed_cone_pixel_size_um(
    A_image1_to_projected_um: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate projected µm represented by one source cone-image pixel."""
    linear = np.asarray(A_image1_to_projected_um, dtype=float)[:2, :2]
    x_um_per_px = float(np.linalg.norm(linear[:, 0]))
    y_um_per_px = float(np.linalg.norm(linear[:, 1]))
    isotropic = float(np.sqrt(x_um_per_px * y_um_per_px))
    return isotropic, x_um_per_px, y_um_per_px


def warp_cone_crop_to_projected_grid(
    cone_image: np.ndarray,
    geometry: AlignmentGeometry,
    center_projected_um: np.ndarray,
    *,
    crop_size_um: float = 300.0,
    interpolation_margin_um: float = 20.0,
    output_um_per_px: float | None = None,
    interpolation_order: int = 1,
) -> AlignedCrop:
    """
    Warp a square Image-1 cone crop into the final projected-µm coordinates.

    `crop_size_um=300` means a 300 × 300 µm central analysis square. A margin
    is added only for interpolation, then removed before cone weighting.
    """
    cone_image = np.asarray(cone_image)
    center = np.asarray(center_projected_um, dtype=float)
    if center.shape != (2,):
        raise ValueError("center_projected_um must contain [x, y]")
    if crop_size_um <= 0 or interpolation_margin_um < 0:
        raise ValueError("crop_size_um must be > 0 and margin must be >= 0")

    estimated_scale, x_scale, y_scale = estimate_transformed_cone_pixel_size_um(
        geometry.A_image1_to_projected_um
    )
    requested_scale = (
        estimated_scale if output_um_per_px is None else float(output_um_per_px)
    )
    if requested_scale <= 0:
        raise ValueError("output_um_per_px must be positive")

    full_size_um = crop_size_um + 2.0 * interpolation_margin_um
    n_px = max(2, int(np.ceil(full_size_um / requested_scale)))
    actual_scale = full_size_um / n_px

    x_min = center[0] - full_size_um / 2.0
    x_max = center[0] + full_size_um / 2.0
    y_min = center[1] - full_size_um / 2.0
    y_max = center[1] + full_size_um / 2.0

    x_centres = x_min + (np.arange(n_px) + 0.5) * actual_scale
    y_centres = y_min + (np.arange(n_px) + 0.5) * actual_scale
    x_grid, y_grid = np.meshgrid(x_centres, y_centres)

    projected_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    source_points = transform_points(
        np.linalg.inv(geometry.A_image1_to_projected_um), projected_points
    )
    source_x = source_points[:, 0].reshape(n_px, n_px)
    source_y = source_points[:, 1].reshape(n_px, n_px)

    source_h, source_w = cone_image.shape[:2]
    valid_mask = (
        (source_x >= 0)
        & (source_x <= source_w - 1)
        & (source_y >= 0)
        & (source_y <= source_h - 1)
    )

    coordinates = np.vstack([source_y.ravel(), source_x.ravel()])
    if cone_image.ndim == 2:
        warped = map_coordinates(
            cone_image.astype(float),
            coordinates,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
            prefilter=interpolation_order > 1,
        ).reshape(n_px, n_px)
    elif cone_image.ndim == 3:
        channels = []
        for channel_i in range(cone_image.shape[-1]):
            warped_channel = map_coordinates(
                cone_image[..., channel_i].astype(float),
                coordinates,
                order=interpolation_order,
                mode="constant",
                cval=0.0,
                prefilter=interpolation_order > 1,
            ).reshape(n_px, n_px)
            channels.append(warped_channel)
        warped = np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Unexpected cone image shape: {cone_image.shape}")

    central_half = crop_size_um / 2.0
    central_extent = (
        center[0] - central_half,
        center[0] + central_half,
        center[1] - central_half,
        center[1] + central_half,
    )

    print("Transformed cone crop")
    print(f"  Source projected sampling: x={x_scale:.4f}, y={y_scale:.4f} µm/px")
    print(f"  Analysis sampling:         {actual_scale:.4f} µm/px")
    print(f"  Full interpolation crop:   {full_size_um:.1f} × {full_size_um:.1f} µm")
    print(f"  Central weighted crop:     {crop_size_um:.1f} × {crop_size_um:.1f} µm")
    print(f"  Output image shape:        {warped.shape[:2]}")

    return AlignedCrop(
        image=warped,
        valid_mask=valid_mask,
        x_centres_um=x_centres,
        y_centres_um=y_centres,
        extent_um=(x_min, x_max, y_min, y_max),
        um_per_px=actual_scale,
        central_extent_um=central_extent,
    )


# -----------------------------------------------------------------------------
# RF EXTRACTION AND SAMPLING
# -----------------------------------------------------------------------------


def _select_optional_scalar(
    dataset: xr.Dataset,
    variable_name: str,
    channel: str | int,
    cell_index: int,
) -> float | None:
    if variable_name not in dataset:
        return None
    da = dataset[variable_name]
    if "channel" in da.dims:
        da = da.sel(channel=channel)
    if "cell_index" in da.dims:
        da = da.sel(cell_index=cell_index)
    values = np.asarray(da.values).squeeze()
    if values.size != 1:
        return None
    value = float(values)
    return value if np.isfinite(value) else None


# -----------------------------------------------------------------------------
# CURRENT RF ALIGNMENT HELPERS
# -----------------------------------------------------------------------------
#
# These functions intentionally reproduce the CURRENT RF alignment script:
#   - corrected scale from cached clicked stimulus edge
#   - projected x=right, y=down, origin=displayed stimulus centre
#   - raw RF -> projected µm = identity
#   - RF data always mirrored left <-> right
#


def flip_rf_left_right(da: xr.DataArray) -> xr.DataArray:
    """
    Mirror RF DATA left <-> right while leaving the x-coordinate values unchanged.

    This is the same fixed horizontal RF flip used by the RF alignment script.
    """
    if "x" not in da.dims:
        raise ValueError("Input DataArray must include an 'x' dimension.")

    x_axis = da.get_axis_num("x")
    flipped_values = np.flip(da.values, axis=x_axis)
    return da.copy(data=flipped_values, deep=True)


def build_alignment_geometry_matching_rf_alignment(
    alignment_cache_path: str | Path,
    *,
    displayed_stim_height_px: int,
    dataset_assumed_um_per_px: float,
) -> AlignmentGeometry:
    """
    Recreate the CURRENT RF alignment script's geometry from its saved cache.

    Important: this deliberately does NOT use the older connectivity-helper
    orientation logic. The RF alignment script now uses:

        A_raw_rf_to_projected_um = identity

    and stores the clicked stimulus edge as stim_edge_top_px /
    stim_edge_bottom_px, irrespective of whether LEFT or RIGHT was clicked.
    """
    cache_path = Path(alignment_cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Alignment cache not found: {cache_path}\n"
            "Run the RF alignment script first so its .npz cache exists."
        )

    loaded = np.load(cache_path, allow_pickle=True)
    required = {
        "tform_2_to_3_params",
        "um_per_px_image3",
        "stim_edge_top_px",
        "stim_edge_bottom_px",
        "visual_stim_center_px_image3",
        "stimulus_edge_side",
    }
    missing = sorted(required.difference(loaded.files))
    if missing:
        raise KeyError(
            "Alignment cache is not in the format produced by the current RF "
            f"alignment script. Missing: {missing}"
        )

    tform_2_to_3_params = np.asarray(loaded["tform_2_to_3_params"], dtype=float)
    um_per_px_image3 = float(loaded["um_per_px_image3"])
    px_per_um_image3 = 1.0 / um_per_px_image3

    stim_edge_top_px = np.asarray(loaded["stim_edge_top_px"], dtype=float)
    stim_edge_bottom_px = np.asarray(loaded["stim_edge_bottom_px"], dtype=float)
    visual_stim_center_px_image3 = np.asarray(
        loaded["visual_stim_center_px_image3"], dtype=float
    )
    stimulus_edge_side = str(np.asarray(loaded["stimulus_edge_side"]).item()).lower()

    if stimulus_edge_side not in {"left", "right"}:
        raise ValueError(
            "Cached stimulus_edge_side must be 'left' or 'right'; "
            f"got {stimulus_edge_side!r}."
        )

    # ------------------------------------------------------------
    # EXACT RF ALIGNMENT SCALE CALCULATION
    # ------------------------------------------------------------
    clicked_edge_vec_px = stim_edge_bottom_px - stim_edge_top_px
    clicked_edge_length_px = float(np.linalg.norm(clicked_edge_vec_px))

    if clicked_edge_length_px == 0:
        raise ValueError("stim_edge_top_px and stim_edge_bottom_px are identical.")

    clicked_edge_length_um = clicked_edge_length_px * um_per_px_image3
    corrected_stim_um_per_px = clicked_edge_length_um / displayed_stim_height_px
    rf_scale_factor = corrected_stim_um_per_px / dataset_assumed_um_per_px

    print("------------------------------------------------------------")
    print("SCALE CHECK — MATCHING RF ALIGNMENT SCRIPT")
    print("------------------------------------------------------------")
    print(
        f"Clicked {stimulus_edge_side.upper()} vertical edge length: "
        f"{clicked_edge_length_px:.2f} Image 3 px"
    )
    print(
        f"Clicked {stimulus_edge_side.upper()} vertical edge length: "
        f"{clicked_edge_length_um:.2f} µm"
    )
    print(f"Displayed stimulus height: {displayed_stim_height_px} stimulus px")
    print(f"Corrected stimulus pixel size: {corrected_stim_um_per_px:.4f} µm/px")
    print(f"RF dataset assumed pixel size: {dataset_assumed_um_per_px:.4f} µm/px")
    print(f"RF coordinate scale factor: {rf_scale_factor:.4f}")

    # ------------------------------------------------------------
    # EXACT PROJECTED-STIMULUS COORDINATE SYSTEM
    # ------------------------------------------------------------
    # +x = right across displayed stimulus
    # +y = down along clicked vertical edge
    # origin = displayed stimulus centre
    stim_down_unit_px = clicked_edge_vec_px / clicked_edge_length_px
    stim_right_unit_px = np.array(
        [stim_down_unit_px[1], -stim_down_unit_px[0]],
        dtype=float,
    )

    M_projected_um_to_image3 = np.column_stack(
        [
            stim_right_unit_px * px_per_um_image3,
            stim_down_unit_px * px_per_um_image3,
        ]
    )

    A_projected_um_to_image3 = np.eye(3, dtype=float)
    A_projected_um_to_image3[:2, :2] = M_projected_um_to_image3
    A_projected_um_to_image3[:2, 2] = visual_stim_center_px_image3

    A_image3_to_projected_um = np.linalg.inv(A_projected_um_to_image3)
    A_image1_to_projected_um = A_image3_to_projected_um @ tform_2_to_3_params

    # CURRENT RF alignment behaviour: no extra raw-stimulus rotation/flip matrix.
    A_raw_rf_to_projected_um = np.eye(3, dtype=float)

    origin_check = A_projected_um_to_image3 @ np.array([0.0, 0.0, 1.0])

    print("------------------------------------------------------------")
    print("TRANSFORM CHECK — MATCHING RF ALIGNMENT SCRIPT")
    print("------------------------------------------------------------")
    print("Raw RF [0,0] maps to Image 3 / displayed stimulus centre:")
    print(f"x = {origin_check[0]:.2f}, y = {origin_check[1]:.2f}")
    print(
        "Clicked displayed stimulus centre: "
        f"x = {visual_stim_center_px_image3[0]:.2f}, "
        f"y = {visual_stim_center_px_image3[1]:.2f}"
    )
    print("Raw -> projected transform: identity")
    print(f"Stimulus edge used for clicks: {stimulus_edge_side.upper()}")

    # AlignmentGeometry still uses the historical field names stim_top_left_px /
    # stim_bottom_left_px. They are not used for the maths below; store the actual
    # clicked top/bottom edge points there only for compatibility with the helper
    # dataclass and output routines.
    return AlignmentGeometry(
        tform_image1_to_image3=tform_2_to_3_params,
        um_per_px_image3=um_per_px_image3,
        corrected_stim_um_per_px=corrected_stim_um_per_px,
        rf_scale_factor=rf_scale_factor,
        A_projected_um_to_image3=A_projected_um_to_image3,
        A_image3_to_projected_um=A_image3_to_projected_um,
        A_image1_to_projected_um=A_image1_to_projected_um,
        A_raw_rf_to_projected_um=A_raw_rf_to_projected_um,
        visual_stim_center_px_image3=visual_stim_center_px_image3,
        stim_top_left_px=stim_edge_top_px,
        stim_bottom_left_px=stim_edge_bottom_px,
    )


def extract_single_cell_rf_matching_rf_alignment(
    dataset: xr.Dataset,
    geometry: AlignmentGeometry,
    *,
    cell_index: int,
    channel: str | int,
    rf_variable_name: str,
    normalise_each_cell: bool,
    displayed_stim_width_px: int,
    displayed_stim_height_px: int,
    clip_rf_to_displayed_stimulus: bool,
    quality_variable_name: str = "quality",
    tilt_variable_name: str = "tilt",
) -> RFData:
    """
    Extract one RF using the SAME spatial operations as the current RF alignment.

    Order of operations is intentionally the same:
        1. select RF variable/channel
        2. mirror RF data left <-> right, preserving x-coordinate labels
        3. select the requested cell
        4. normalise that cell, if requested
        5. scale x/y coordinates by rf_scale_factor
        6. clip to the displayed stimulus, if requested
        7. locate the strongest actually plotted RF pixel
    """
    if rf_variable_name not in dataset:
        raise KeyError(f"dataset does not contain {rf_variable_name!r}")

    combined = dataset[rf_variable_name]
    if "channel" in combined.dims:
        combined = combined.sel(channel=channel)

    print("------------------------------------------------------------")
    print("RF DATA — MATCHING RF ALIGNMENT SCRIPT")
    print("------------------------------------------------------------")
    print("Variable:", rf_variable_name)
    print("Original dims:", combined.dims)
    print("Original shape:", combined.shape)
    print(
        "Applying fixed left-right RF mirror "
        "(horizontal flip about central vertical line)."
    )
    combined = flip_rf_left_right(combined)

    if "cell_index" in combined.dims:
        available_cells = combined["cell_index"].values
        if cell_index not in available_cells:
            raise ValueError(f"cell_index={cell_index} is not present")
        combined = combined.sel(cell_index=cell_index)

    combined = combined.squeeze(drop=True)
    if "y" not in combined.dims or "x" not in combined.dims:
        raise ValueError(
            f"RF must contain y and x dimensions after selection; got {combined.dims}."
        )
    combined = combined.transpose("y", "x")

    raw_values = np.asarray(combined.values, dtype=float).copy()
    signed = rf_variable_name == "cm_most_important" or (
        np.isfinite(raw_values).any() and np.nanmin(raw_values) < 0
    )

    values = raw_values.copy()
    if normalise_each_cell:
        if signed:
            scale = float(np.nanmax(np.abs(values)))
        else:
            scale = float(np.nanmax(values))

        if np.isfinite(scale) and scale != 0:
            values = values / scale
        values = np.nan_to_num(values, nan=0.0)

    rf_x = np.asarray(combined["x"].values, dtype=float)
    rf_y = np.asarray(combined["y"].values, dtype=float)

    # Same edge-based coordinate scaling as the RF alignment plot.
    rf_x_edges_original = coordinate_edges_from_centres(rf_x)
    rf_y_edges_original = coordinate_edges_from_centres(rf_y)
    rf_x_edges_corrected = rf_x_edges_original * geometry.rf_scale_factor
    rf_y_edges_corrected = rf_y_edges_original * geometry.rf_scale_factor

    x_raw_um = rf_x * geometry.rf_scale_factor
    y_raw_um = rf_y * geometry.rf_scale_factor

    # Same displayed-stimulus clipping as the RF alignment script.
    displayed_half_x_um = (
        displayed_stim_width_px * geometry.corrected_stim_um_per_px
    ) / 2.0
    displayed_half_y_um = (
        displayed_stim_height_px * geometry.corrected_stim_um_per_px
    ) / 2.0

    raw_x_grid_um, raw_y_grid_um = np.meshgrid(x_raw_um, y_raw_um)
    projected_x_grid_um = (
        geometry.A_raw_rf_to_projected_um[0, 0] * raw_x_grid_um
        + geometry.A_raw_rf_to_projected_um[0, 1] * raw_y_grid_um
        + geometry.A_raw_rf_to_projected_um[0, 2]
    )
    projected_y_grid_um = (
        geometry.A_raw_rf_to_projected_um[1, 0] * raw_x_grid_um
        + geometry.A_raw_rf_to_projected_um[1, 1] * raw_y_grid_um
        + geometry.A_raw_rf_to_projected_um[1, 2]
    )

    valid_displayed_mask_yx = (
        (projected_x_grid_um >= -displayed_half_x_um)
        & (projected_x_grid_um <= displayed_half_x_um)
        & (projected_y_grid_um >= -displayed_half_y_um)
        & (projected_y_grid_um <= displayed_half_y_um)
    )

    print(
        "Displayed stimulus region: "
        f"x={-displayed_half_x_um:.2f} to {displayed_half_x_um:.2f} µm; "
        f"y={-displayed_half_y_um:.2f} to {displayed_half_y_um:.2f} µm"
    )
    print(
        f"RF pixels inside displayed region: "
        f"{np.sum(valid_displayed_mask_yx)} / {valid_displayed_mask_yx.size}"
    )

    if clip_rf_to_displayed_stimulus:
        values = np.where(valid_displayed_mask_yx, values, np.nan)

    # EXACTLY as the RF alignment's single-cell zoom: find the strongest visible
    # pixel AFTER the fixed mirror, scaling and optional clipping.
    rf_for_peak = np.abs(values) if signed else values
    finite = np.isfinite(rf_for_peak)
    if not np.any(finite):
        raise ValueError("No finite RF pixels remain after alignment/clipping.")

    masked = np.where(finite, rf_for_peak, -np.inf)
    peak_y_i, peak_x_i = np.unravel_index(np.nanargmax(masked), masked.shape)

    peak_x_um = 0.5 * (
        rf_x_edges_corrected[peak_x_i] + rf_x_edges_corrected[peak_x_i + 1]
    )
    peak_y_um = 0.5 * (
        rf_y_edges_corrected[peak_y_i] + rf_y_edges_corrected[peak_y_i + 1]
    )

    peak_raw_um = np.array([peak_x_um, peak_y_um], dtype=float)
    peak_projected_um = transform_points(
        geometry.A_raw_rf_to_projected_um,
        peak_raw_um[None, :],
    )[0]

    quality = _select_optional_scalar(
        dataset, quality_variable_name, channel, cell_index
    )
    tilt = _select_optional_scalar(dataset, tilt_variable_name, channel, cell_index)

    print("------------------------------------------------------------")
    print("PLOTTED RF PEAK USED FOR CONNECTIVITY CROP")
    print("------------------------------------------------------------")
    print(f"Cell: {cell_index}")
    print(f"Peak array index: y={peak_y_i}, x={peak_x_i}")
    print(f"Peak raw RF coordinate: x={peak_x_um:.2f}, y={peak_y_um:.2f} µm")
    print(
        "Peak projected coordinate: "
        f"x={peak_projected_um[0]:.2f}, y={peak_projected_um[1]:.2f} µm"
    )

    return RFData(
        values=values,
        raw_values=raw_values,
        x_raw_um=x_raw_um,
        y_raw_um=y_raw_um,
        peak_raw_um=peak_raw_um,
        peak_projected_um=peak_projected_um,
        variable_name=rf_variable_name,
        signed=signed,
        normalised=normalise_each_cell,
        quality=quality,
        tilt=tilt,
    )


def make_rf_interpolator(rf: RFData) -> RegularGridInterpolator:
    """Create a bilinear interpolator in corrected raw RF coordinates."""
    x_order = np.argsort(rf.x_raw_um)
    y_order = np.argsort(rf.y_raw_um)
    x_sorted = rf.x_raw_um[x_order]
    y_sorted = rf.y_raw_um[y_order]
    values_sorted = rf.values[np.ix_(y_order, x_order)]
    return RegularGridInterpolator(
        (y_sorted, x_sorted),
        values_sorted,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def sample_rf_at_projected_points(
    rf: RFData,
    geometry: AlignmentGeometry,
    points_projected_um: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample one RF at projected-µm x/y positions."""
    points = _ensure_points(points_projected_um)
    if len(points) == 0:
        return np.empty(0, dtype=float)
    raw_points = transform_points(
        np.linalg.inv(geometry.A_raw_rf_to_projected_um), points
    )
    interpolator = make_rf_interpolator(rf)
    query_yx = np.column_stack([raw_points[:, 1], raw_points[:, 0]])
    return np.asarray(interpolator(query_yx), dtype=float)


def sample_rf_on_aligned_crop(
    rf: RFData,
    geometry: AlignmentGeometry,
    crop: AlignedCrop,
) -> np.ndarray:
    """Resample an RF onto the transformed cone-crop raster."""
    x_grid, y_grid = np.meshgrid(crop.x_centres_um, crop.y_centres_um)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    sampled = sample_rf_at_projected_points(rf, geometry, points)
    return sampled.reshape(x_grid.shape)


# -----------------------------------------------------------------------------
# CONE-MOSAIC INTERPOLATION (ADAPTED FROM THE USER'S DRAFT)
# -----------------------------------------------------------------------------


def make_retina_intensity_image(image: np.ndarray) -> np.ndarray:
    """Create the red/blue mean image used in the draft blue-cone detector."""
    image = np.asarray(image, dtype=float)
    if image.ndim == 2:
        intensity = image.copy()
    elif image.ndim == 3 and image.shape[-1] >= 3:
        intensity = image[..., [0, 2]].mean(axis=-1)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    finite = np.isfinite(intensity)
    if not finite.any():
        raise ValueError("The transformed cone crop contains no finite pixels")
    fill_value = float(np.nanmedian(intensity[finite]))
    intensity = np.where(finite, intensity, fill_value)
    return exposure.rescale_intensity(intensity, in_range="image", out_range=(0, 1))


def detect_blue_cones_from_blobs(
    intensity_crop: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    sigma: float = 2.5,
    threshold_rel: float = 0.30,
    min_blob_area_px: int = 20,
    max_blob_area_px: int | None = None,
    opening_radius_px: int = 0,
    closing_radius_px: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    detection_image = filters.gaussian(intensity_crop, sigma=sigma)
    detection_image = exposure.rescale_intensity(
        detection_image, in_range="image", out_range=(0, 1)
    )

    if valid_mask is None:
        valid_mask = np.ones_like(detection_image, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    valid_values = detection_image[valid_mask]
    if valid_values.size == 0:
        raise ValueError("No valid transformed cone-image pixels are available")
    threshold_value = threshold_rel * float(valid_values.max())
    binary = (detection_image > threshold_value) & valid_mask

    if opening_radius_px > 0:
        binary = morphology.binary_opening(binary, morphology.disk(opening_radius_px))
    if closing_radius_px > 0:
        binary = morphology.binary_closing(binary, morphology.disk(closing_radius_px))
    binary_clean = morphology.remove_small_objects(binary, min_size=min_blob_area_px)

    labels_raw = measure.label(binary_clean, connectivity=2)
    props = measure.regionprops(labels_raw, intensity_image=detection_image)
    centres: list[list[float]] = []
    blob_table: list[dict[str, float]] = []
    labels_kept = np.zeros_like(labels_raw, dtype=np.int32)
    next_label = 1

    for prop in props:
        area = int(prop.area)
        if area < min_blob_area_px:
            continue
        if max_blob_area_px is not None and area > max_blob_area_px:
            continue
        y, x = prop.centroid
        equivalent_radius_px = float(np.sqrt(area / np.pi))
        centres.append([float(x), float(y)])
        blob_table.append(
            {
                "label": next_label,
                "x_px": float(x),
                "y_px": float(y),
                "area_px": area,
                "equivalent_radius_px": equivalent_radius_px,
                "mean_intensity": float(prop.mean_intensity),
                "max_intensity": float(prop.max_intensity),
            }
        )
        labels_kept[labels_raw == prop.label] = next_label
        next_label += 1

    centres_array = _ensure_points(centres)
    print("Blue cone blob detection")
    print(f"  Threshold relative to max: {threshold_rel}")
    print(f"  Threshold value:           {threshold_value:.4f}")
    print(f"  Accepted blue blobs:       {len(centres_array)}")
    return centres_array, detection_image, binary_clean, labels_kept, blob_table


def estimate_blue_uv_sizes_from_blobs(
    blob_table: list[dict[str, float]],
    um_per_px: float,
    *,
    fallback_blue_radius_um: float = 1.5,
    blue_scale: float = 0.80,
    uv_scale: float = 1.0,
) -> tuple[float, float]:
    if blob_table:
        median_radius_px = float(
            np.median([row["equivalent_radius_px"] for row in blob_table])
        )
        measured_blue_radius_um = median_radius_px * um_per_px
        blue_radius_um = measured_blue_radius_um * blue_scale
    else:
        blue_radius_um = fallback_blue_radius_um
    uv_radius_um = blue_radius_um * uv_scale
    return float(blue_radius_um), float(uv_radius_um)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def wrap_angle_180(angle_deg: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(angle_deg) % 180.0


def circular_angle_diff_180(angle_deg: np.ndarray, reference_deg: float) -> np.ndarray:
    return (angle_deg - reference_deg + 90.0) % 180.0 - 90.0


def merge_duplicate_points_with_ellipse_params(
    points_xy: np.ndarray,
    angles_deg: np.ndarray,
    short_radii_um: np.ndarray,
    merge_distance_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_xy = _ensure_points(points_xy)
    if len(points_xy) == 0:
        return points_xy, np.asarray(angles_deg), np.asarray(short_radii_um)

    kept_points: list[np.ndarray] = []
    kept_angles: list[float] = []
    kept_radii: list[float] = []
    for point, angle, radius in zip(points_xy, angles_deg, short_radii_um):
        if not kept_points:
            keep = True
        else:
            distances = np.linalg.norm(np.asarray(kept_points) - point, axis=1)
            keep = bool(np.min(distances) >= merge_distance_px)
        if keep:
            kept_points.append(point)
            kept_angles.append(float(angle))
            kept_radii.append(float(radius))
    return _ensure_points(kept_points), np.asarray(kept_angles), np.asarray(kept_radii)


def estimate_blue_row_axis_and_spacing(
    blue_xy_px: np.ndarray,
    *,
    manual_angle_deg: float | None = None,
    peak_halfwidth_deg: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    blue_xy_px = _ensure_points(blue_xy_px)
    if len(blue_xy_px) < 3:
        raise ValueError("Need at least three detected blue cones")

    distances = cdist(blue_xy_px, blue_xy_px)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    nn_spacing_px = float(np.median(nearest))

    pair_vectors: list[np.ndarray] = []
    for i in range(len(blue_xy_px)):
        vectors = blue_xy_px[i + 1 :] - blue_xy_px[i]
        vector_lengths = np.linalg.norm(vectors, axis=1)
        keep = (vector_lengths >= 0.60 * nn_spacing_px) & (
            vector_lengths <= 1.80 * nn_spacing_px
        )
        pair_vectors.extend(vectors[keep])

    if not pair_vectors:
        raise ValueError("Could not find local blue-cone neighbour vectors")
    pair_vectors_array = np.asarray(pair_vectors, dtype=float)
    pair_angles = wrap_angle_180(
        np.degrees(np.arctan2(pair_vectors_array[:, 1], pair_vectors_array[:, 0]))
    )

    if manual_angle_deg is None:
        hist, edges = np.histogram(pair_angles, bins=np.linspace(0, 180, 181))
        centres = 0.5 * (edges[:-1] + edges[1:])
        dominant_angle = float(centres[np.argmax(hist)])
    else:
        dominant_angle = float(wrap_angle_180(manual_angle_deg))

    angle_diffs = circular_angle_diff_180(pair_angles, dominant_angle)
    kept_vectors = pair_vectors_array[np.abs(angle_diffs) <= peak_halfwidth_deg]
    if len(kept_vectors) < 2:
        raise ValueError(
            "Too few neighbour vectors near the chosen row angle; set "
            "manual_row_axis_angle_deg or adjust the crop/detection settings"
        )

    reference_unit = np.array(
        [np.cos(np.deg2rad(dominant_angle)), np.sin(np.deg2rad(dominant_angle))]
    )
    oriented_units = []
    for vector in kept_vectors:
        vector_use = -vector if np.dot(vector, reference_unit) < 0 else vector
        oriented_units.append(normalize_vector(vector_use))
    row_unit = normalize_vector(np.mean(oriented_units, axis=0))
    row_angle_deg = float(
        wrap_angle_180(np.degrees(np.arctan2(row_unit[1], row_unit[0])))
    )
    perp_unit = np.array([-row_unit[1], row_unit[0]])

    along = np.abs(kept_vectors @ row_unit)
    across = np.abs(kept_vectors @ perp_unit)
    aligned = across <= 0.35 * nn_spacing_px
    row_spacing_px = float(np.median(along[aligned] if aligned.any() else along))

    print("Blue/UV row geometry")
    print(f"  Row angle:                 {row_angle_deg:.2f}°")
    print(f"  Blue row spacing:          {row_spacing_px:.2f} px")
    print(f"  Nearest blue spacing:      {nn_spacing_px:.2f} px")
    return row_unit, perp_unit, row_spacing_px, row_angle_deg, nn_spacing_px


def build_blue_row_pairs_and_uv_cones(
    blue_xy_px: np.ndarray,
    row_unit: np.ndarray,
    perp_unit: np.ndarray,
    row_spacing_px: float,
    *,
    along_tol_fraction: float = 0.35,
    perp_tol_fraction: float = 0.30,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    blue_xy_px = _ensure_points(blue_xy_px)
    used_pairs: set[tuple[int, int]] = set()
    row_pairs: list[tuple[int, int]] = []

    for i, point in enumerate(blue_xy_px):
        differences = blue_xy_px - point
        along = differences @ row_unit
        across = differences @ perp_unit
        candidates = np.where(
            (along > 0.40 * row_spacing_px)
            & (np.abs(along - row_spacing_px) <= along_tol_fraction * row_spacing_px)
            & (np.abs(across) <= perp_tol_fraction * row_spacing_px)
        )[0]
        if len(candidates) == 0:
            continue
        scores = ((along[candidates] - row_spacing_px) / row_spacing_px) ** 2 + (
            across[candidates] / row_spacing_px
        ) ** 2
        best_j = int(candidates[np.argmin(scores)])
        key = tuple(sorted((i, best_j)))
        if key not in used_pairs:
            used_pairs.add(key)
            row_pairs.append(key)

    uv_xy_px = _ensure_points(
        [0.5 * (blue_xy_px[i] + blue_xy_px[j]) for i, j in row_pairs]
    )
    print(f"  UV cones interpolated:     {len(uv_xy_px)}")
    return row_pairs, uv_xy_px


def build_orthogonal_blue_uv_pairs(
    blue_xy_px: np.ndarray,
    uv_xy_px: np.ndarray,
    row_unit: np.ndarray,
    perp_unit: np.ndarray,
    row_spacing_px: float,
    *,
    row_tol_fraction: float = 0.35,
    min_perp_fraction: float = 0.25,
    max_perp_fraction: float = 1.50,
    use_both_sides: bool = True,
) -> list[dict[str, Any]]:
    blue_xy_px = _ensure_points(blue_xy_px)
    uv_xy_px = _ensure_points(uv_xy_px)
    if len(blue_xy_px) == 0 or len(uv_xy_px) == 0:
        return []

    max_row_offset_px = row_tol_fraction * row_spacing_px
    min_perp_px = min_perp_fraction * row_spacing_px
    max_perp_px = max_perp_fraction * row_spacing_px
    pairs: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()

    for blue_i, blue_point in enumerate(blue_xy_px):
        differences = uv_xy_px - blue_point
        row_projection = differences @ row_unit
        perp_projection = differences @ perp_unit
        for side in (-1, 1):
            candidate_mask = (
                (np.abs(row_projection) <= max_row_offset_px)
                & (side * perp_projection > min_perp_px)
                & (side * perp_projection < max_perp_px)
            )
            candidates = np.where(candidate_mask)[0]
            if len(candidates) == 0:
                continue
            denominator = max(max_row_offset_px, np.finfo(float).eps)
            scores = (row_projection[candidates] / denominator) ** 2 + (
                (np.abs(perp_projection[candidates]) - 0.5 * row_spacing_px)
                / row_spacing_px
            ) ** 2
            uv_i = int(candidates[np.argmin(scores)])
            key = (blue_i, uv_i)
            if key in used:
                continue
            used.add(key)
            vector = uv_xy_px[uv_i] - blue_point
            length = float(np.linalg.norm(vector))
            if length == 0:
                continue
            pairs.append(
                {
                    "blue_index": blue_i,
                    "uv_index": uv_i,
                    "blue_px": blue_point,
                    "uv_px": uv_xy_px[uv_i],
                    "vector_px": vector,
                    "angle_deg": float(np.degrees(np.arctan2(vector[1], vector[0]))),
                    "length_px": length,
                    "side": side,
                }
            )
            if not use_both_sides:
                break
    print(f"  Orthogonal blue-UV pairs:  {len(pairs)}")
    return pairs


def build_red_and_green_per_segment(
    blue_uv_pairs: list[dict[str, Any]],
    roi_shape: tuple[int, int],
    *,
    blue_radius_um: float,
    uv_radius_um: float,
    red_long_radius_um: float,
    green_long_radius_um: float,
    red_near_blue: bool = True,
    edge_clearance_um: float = 0.10,
    gap_fill_fraction: float = 0.98,
    min_short_radius_um: float = 0.05,
    max_short_axis_fraction_of_long_axis: float = 0.85,
    duplicate_merge_distance_um: float = 0.10,
    um_per_px: float = 1.0,
    angle_offset_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height_px, width_px = roi_shape
    red_points: list[np.ndarray] = []
    green_points: list[np.ndarray] = []
    red_angles: list[float] = []
    green_angles: list[float] = []
    red_short_radii: list[float] = []
    green_short_radii: list[float] = []

    for pair in blue_uv_pairs:
        blue = np.asarray(pair["blue_px"], dtype=float)
        vector = np.asarray(pair["vector_px"], dtype=float)
        length_px = float(pair["length_px"])
        if length_px <= 0:
            continue
        available_gap_um = (
            length_px * um_per_px
            - blue_radius_um
            - uv_radius_um
            - 2.0 * edge_clearance_um
        )
        if available_gap_um <= 0:
            continue
        local_short_um = 0.25 * available_gap_um * gap_fill_fraction
        local_short_um = float(
            np.clip(
                local_short_um,
                min_short_radius_um,
                red_long_radius_um * max_short_axis_fraction_of_long_axis,
            )
        )
        unit = vector / length_px
        red_distance = (blue_radius_um + edge_clearance_um + local_short_um) / um_per_px
        green_distance = (
            length_px - (uv_radius_um + edge_clearance_um + local_short_um) / um_per_px
        )
        if red_distance >= green_distance:
            continue
        near_point = blue + red_distance * unit
        far_point = blue + green_distance * unit
        red_point, green_point = (
            (near_point, far_point) if red_near_blue else (far_point, near_point)
        )
        angle = float(pair["angle_deg"] + angle_offset_deg)
        red_points.append(red_point)
        green_points.append(green_point)
        red_angles.append(angle)
        green_angles.append(angle)
        red_short_radii.append(local_short_um)
        green_short_radii.append(local_short_um)

    def keep_inside(
        points: np.ndarray, angles: np.ndarray, radii: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = _ensure_points(points)
        if len(points) == 0:
            return points, np.asarray(angles), np.asarray(radii)
        keep = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width_px)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height_px)
        )
        return points[keep], np.asarray(angles)[keep], np.asarray(radii)[keep]

    red_px, red_angles_arr, red_radii_arr = keep_inside(
        _ensure_points(red_points), np.asarray(red_angles), np.asarray(red_short_radii)
    )
    green_px, green_angles_arr, green_radii_arr = keep_inside(
        _ensure_points(green_points),
        np.asarray(green_angles),
        np.asarray(green_short_radii),
    )
    merge_px = duplicate_merge_distance_um / um_per_px
    red_px, red_angles_arr, red_radii_arr = merge_duplicate_points_with_ellipse_params(
        red_px, red_angles_arr, red_radii_arr, merge_px
    )
    (
        green_px,
        green_angles_arr,
        green_radii_arr,
    ) = merge_duplicate_points_with_ellipse_params(
        green_px, green_angles_arr, green_radii_arr, merge_px
    )
    print(f"  Red cones interpolated:    {len(red_px)}")
    print(f"  Green cones interpolated:  {len(green_px)}")
    return (
        red_px,
        green_px,
        red_angles_arr,
        green_angles_arr,
        red_radii_arr,
        green_radii_arr,
    )


def crop_px_to_projected_um(points_px: np.ndarray, crop: AlignedCrop) -> np.ndarray:
    """Map transformed-crop pixel centres to projected x/y µm."""
    points_px = _ensure_points(points_px)
    if len(points_px) == 0:
        return points_px
    x_min, _, y_min, _ = crop.extent_um
    return np.column_stack(
        [
            x_min + (points_px[:, 0] + 0.5) * crop.um_per_px,
            y_min + (points_px[:, 1] + 0.5) * crop.um_per_px,
        ]
    )


def _filter_points_to_extent(
    points_px: np.ndarray,
    points_um: np.ndarray,
    extent: tuple[float, float, float, float],
    *associated: np.ndarray,
) -> tuple[np.ndarray, ...]:
    x_min, x_max, y_min, y_max = extent
    points_px = _ensure_points(points_px)
    points_um = _ensure_points(points_um)
    if len(points_um) == 0:
        return (points_px, points_um, *[np.asarray(a) for a in associated])
    keep = (
        (points_um[:, 0] >= x_min)
        & (points_um[:, 0] <= x_max)
        & (points_um[:, 1] >= y_min)
        & (points_um[:, 1] <= y_max)
    )
    return (
        points_px[keep],
        points_um[keep],
        *[np.asarray(a)[keep] for a in associated],
    )


def interpolate_cone_mosaic_on_aligned_crop(
    crop: AlignedCrop,
    *,
    blue_detection_sigma: float = 2.5,
    blue_threshold_rel: float = 0.30,
    blue_min_blob_area_px: int = 20,
    blue_max_blob_area_px: int | None = None,
    blue_opening_radius_px: int = 0,
    blue_closing_radius_px: int = 1,
    manual_row_axis_angle_deg: float | None = None,
    row_angle_peak_halfwidth_deg: float = 12.0,
    row_pair_along_tol_fraction: float = 0.35,
    row_pair_perp_tol_fraction: float = 0.30,
    blue_uv_orthogonal_row_tol_fraction: float = 0.35,
    blue_uv_orthogonal_min_fraction: float = 0.25,
    blue_uv_orthogonal_max_fraction: float = 1.50,
    use_both_orthogonal_sides: bool = True,
    fallback_blue_radius_um: float = 1.5,
    blue_radius_scale: float = 0.8,
    uv_radius_scale: float = 1.0,
    red_green_long_axis_scale: float = 1.0,
    red_near_blue: bool = True,
    red_green_edge_clearance_um: float = 0.10,
    red_green_gap_fill_fraction: float = 0.98,
    duplicate_merge_distance_um: float = 0.10,
    red_green_angle_offset_deg: float = 90.0,
    red_green_max_short_axis_fraction_of_long_axis: float = 0.85,
    red_green_min_short_radius_um: float = 0.05,
) -> ConeMosaic:
    """Run the draft interpolation logic on a transformed, aligned cone crop."""
    intensity = make_retina_intensity_image(crop.image)
    blue_px, detection, binary, _, blob_table = detect_blue_cones_from_blobs(
        intensity,
        valid_mask=crop.valid_mask,
        sigma=blue_detection_sigma,
        threshold_rel=blue_threshold_rel,
        min_blob_area_px=blue_min_blob_area_px,
        max_blob_area_px=blue_max_blob_area_px,
        opening_radius_px=blue_opening_radius_px,
        closing_radius_px=blue_closing_radius_px,
    )
    if len(blue_px) < 5:
        raise ValueError(
            "Fewer than five blue cones were detected. Lower "
            "blue_threshold_rel/blue_min_blob_area_px or inspect the transformed crop."
        )

    blue_radius_um, uv_radius_um = estimate_blue_uv_sizes_from_blobs(
        blob_table,
        crop.um_per_px,
        fallback_blue_radius_um=fallback_blue_radius_um,
        blue_scale=blue_radius_scale,
        uv_scale=uv_radius_scale,
    )
    (
        row_unit,
        perp_unit,
        row_spacing_px,
        row_angle_deg,
        _,
    ) = estimate_blue_row_axis_and_spacing(
        blue_px,
        manual_angle_deg=manual_row_axis_angle_deg,
        peak_halfwidth_deg=row_angle_peak_halfwidth_deg,
    )
    _, uv_px = build_blue_row_pairs_and_uv_cones(
        blue_px,
        row_unit,
        perp_unit,
        row_spacing_px,
        along_tol_fraction=row_pair_along_tol_fraction,
        perp_tol_fraction=row_pair_perp_tol_fraction,
    )
    orthogonal_pairs = build_orthogonal_blue_uv_pairs(
        blue_px,
        uv_px,
        row_unit,
        perp_unit,
        row_spacing_px,
        row_tol_fraction=blue_uv_orthogonal_row_tol_fraction,
        min_perp_fraction=blue_uv_orthogonal_min_fraction,
        max_perp_fraction=blue_uv_orthogonal_max_fraction,
        use_both_sides=use_both_orthogonal_sides,
    )
    if not orthogonal_pairs:
        raise ValueError(
            "No orthogonal blue-UV pairs were found. Adjust the row angle or pairing tolerances."
        )

    red_long_radius_um = blue_radius_um * red_green_long_axis_scale
    green_long_radius_um = red_long_radius_um
    (
        red_px,
        green_px,
        red_angles,
        green_angles,
        red_short_radii,
        green_short_radii,
    ) = build_red_and_green_per_segment(
        orthogonal_pairs,
        crop.image.shape[:2],
        blue_radius_um=blue_radius_um,
        uv_radius_um=uv_radius_um,
        red_long_radius_um=red_long_radius_um,
        green_long_radius_um=green_long_radius_um,
        red_near_blue=red_near_blue,
        edge_clearance_um=red_green_edge_clearance_um,
        gap_fill_fraction=red_green_gap_fill_fraction,
        min_short_radius_um=red_green_min_short_radius_um,
        max_short_axis_fraction_of_long_axis=red_green_max_short_axis_fraction_of_long_axis,
        duplicate_merge_distance_um=duplicate_merge_distance_um,
        um_per_px=crop.um_per_px,
        angle_offset_deg=red_green_angle_offset_deg,
    )

    blue_um = crop_px_to_projected_um(blue_px, crop)
    uv_um = crop_px_to_projected_um(uv_px, crop)
    red_um = crop_px_to_projected_um(red_px, crop)
    green_um = crop_px_to_projected_um(green_px, crop)

    blue_px, blue_um = _filter_points_to_extent(
        blue_px, blue_um, crop.central_extent_um
    )[:2]
    uv_px, uv_um = _filter_points_to_extent(uv_px, uv_um, crop.central_extent_um)[:2]
    red_px, red_um, red_angles, red_short_radii = _filter_points_to_extent(
        red_px,
        red_um,
        crop.central_extent_um,
        red_angles,
        red_short_radii,
    )
    green_px, green_um, green_angles, green_short_radii = _filter_points_to_extent(
        green_px,
        green_um,
        crop.central_extent_um,
        green_angles,
        green_short_radii,
    )

    print("Central analysis-crop cone counts")
    print(
        f"  UV={len(uv_um)}, Blue={len(blue_um)}, Green={len(green_um)}, Red={len(red_um)}"
    )

    return ConeMosaic(
        points_px={"UV": uv_px, "Blue": blue_px, "Green": green_px, "Red": red_px},
        points_projected_um={
            "UV": uv_um,
            "Blue": blue_um,
            "Green": green_um,
            "Red": red_um,
        },
        blue_radius_um=blue_radius_um,
        uv_radius_um=uv_radius_um,
        red_long_radius_um=red_long_radius_um,
        green_long_radius_um=green_long_radius_um,
        red_short_radii_um=np.asarray(red_short_radii),
        green_short_radii_um=np.asarray(green_short_radii),
        red_angles_deg=np.asarray(red_angles),
        green_angles_deg=np.asarray(green_angles),
        row_angle_deg=row_angle_deg,
        row_spacing_um=row_spacing_px * crop.um_per_px,
        detection_image=detection,
        binary_clean=binary,
        blob_table=blob_table,
    )


# -----------------------------------------------------------------------------
# CONE WEIGHTS
# -----------------------------------------------------------------------------


def compute_cone_connectivity_weights(
    rf: RFData,
    geometry: AlignmentGeometry,
    mosaic: ConeMosaic,
    *,
    weight_floor_fraction: float = 0.0,
) -> ConnectivityResults:
    """
    Sample the RF at every cone centre and summarise each cone type.

    Primary strength = abs(RF value), so equal-magnitude positive and negative
    covariance values receive equal connection strength. Sign is retained in
    separate positive, negative and signed-sum columns.
    """
    if not 0 <= weight_floor_fraction < 1:
        raise ValueError("weight_floor_fraction must be in [0, 1)")

    per_cone: list[dict[str, Any]] = []
    cone_id = 0
    for cone_type in CONE_TYPES:
        points = mosaic.points_projected_um[cone_type]
        sampled = sample_rf_at_projected_points(rf, geometry, points)
        finite_abs = np.abs(sampled[np.isfinite(sampled)])
        type_scale = float(finite_abs.max()) if finite_abs.size else 0.0
        floor = weight_floor_fraction * type_scale

        for point, rf_value in zip(points, sampled):
            if not np.isfinite(rf_value):
                continue
            magnitude = abs(float(rf_value))
            if magnitude < floor:
                magnitude = 0.0
                effective_value = 0.0
            else:
                effective_value = float(rf_value)
            per_cone.append(
                {
                    "cone_id": cone_id,
                    "cone_type": cone_type,
                    "x_projected_um": float(point[0]),
                    "y_projected_um": float(point[1]),
                    "rf_value": float(rf_value),
                    "effective_rf_value": effective_value,
                    "magnitude_weight": magnitude,
                    "positive_weight": max(effective_value, 0.0),
                    "negative_weight": max(-effective_value, 0.0),
                }
            )
            cone_id += 1

    total_abs = sum(row["magnitude_weight"] for row in per_cone)
    summary: dict[str, dict[str, float | int]] = {}
    for cone_type in CONE_TYPES:
        rows = [row for row in per_cone if row["cone_type"] == cone_type]
        count = len(rows)
        sum_abs = float(sum(row["magnitude_weight"] for row in rows))
        positive = float(sum(row["positive_weight"] for row in rows))
        negative = float(sum(row["negative_weight"] for row in rows))
        signed_sum = float(sum(row["effective_rf_value"] for row in rows))
        mean_abs = sum_abs / count if count else 0.0
        summary[cone_type] = {
            "count": count,
            "sum_abs_weight": sum_abs,
            "fraction_abs_weight": sum_abs / total_abs if total_abs > 0 else 0.0,
            "mean_abs_weight_per_cone": mean_abs,
            "positive_weight": positive,
            "negative_weight": negative,
            "signed_sum": signed_sum,
        }

    return ConnectivityResults(per_cone=per_cone, summary=summary)


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------


def _plot_cones(
    ax: plt.Axes,
    mosaic: ConeMosaic,
    *,
    alpha: float = 0.85,
    draw_all_types: bool = True,
) -> None:
    for cone_type in CONE_TYPES:
        if not draw_all_types and cone_type not in {"UV", "Blue"}:
            continue
        points = mosaic.points_projected_um[cone_type]
        colour = CONE_COLOURS[cone_type]
        if cone_type == "Blue":
            for index, (x, y) in enumerate(points):
                ax.add_patch(
                    plt.Circle(
                        (x, y),
                        mosaic.blue_radius_um,
                        facecolor=colour,
                        edgecolor="none",
                        alpha=alpha,
                        label=cone_type if index == 0 else None,
                        zorder=8,
                    )
                )
        elif cone_type == "UV":
            for index, (x, y) in enumerate(points):
                ax.add_patch(
                    plt.Circle(
                        (x, y),
                        mosaic.uv_radius_um,
                        facecolor=colour,
                        edgecolor="none",
                        alpha=alpha,
                        label=cone_type if index == 0 else None,
                        zorder=8,
                    )
                )
        else:
            angles = (
                mosaic.green_angles_deg
                if cone_type == "Green"
                else mosaic.red_angles_deg
            )
            short_radii = (
                mosaic.green_short_radii_um
                if cone_type == "Green"
                else mosaic.red_short_radii_um
            )
            long_radius = (
                mosaic.green_long_radius_um
                if cone_type == "Green"
                else mosaic.red_long_radius_um
            )
            for index, ((x, y), angle, short_radius) in enumerate(
                zip(points, angles, short_radii)
            ):
                ax.add_patch(
                    Ellipse(
                        (x, y),
                        width=2.0 * long_radius,
                        height=2.0 * float(short_radius),
                        angle=float(angle),
                        facecolor=colour,
                        edgecolor="none",
                        alpha=alpha,
                        label=cone_type if index == 0 else None,
                        zorder=8,
                    )
                )


def _rf_norm(rf_grid: np.ndarray, signed: bool) -> Normalize:
    finite = rf_grid[np.isfinite(rf_grid)]
    if finite.size == 0:
        return Normalize(0, 1)
    if signed:
        vmax = float(np.nanmax(np.abs(finite)))
        vmax = vmax if vmax > 0 else 1.0
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    vmax = float(np.nanmax(finite))
    return Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)


def _set_projected_crop_limits(
    ax: plt.Axes, extent: tuple[float, float, float, float]
) -> None:
    x_min, x_max, y_min, y_max = extent
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Projected x (µm)")
    ax.set_ylabel("Projected y (µm, +down)")


def _add_scale_bar_projected(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    *,
    length_um: float = 50.0,
) -> None:
    x_min, x_max, y_min, y_max = extent
    x_end = x_max - 0.05 * (x_max - x_min)
    x_start = x_end - length_um
    y = y_max - 0.06 * (y_max - y_min)
    ax.plot([x_start, x_end], [y, y], color="white", linewidth=3, zorder=30)
    ax.text(
        0.5 * (x_start + x_end),
        y - 0.02 * (y_max - y_min),
        f"{length_um:g} µm",
        ha="center",
        va="bottom",
        color="white",
        fontsize=9,
        zorder=31,
    )


def _draw_spider_connections(
    ax: plt.Axes,
    rf: RFData,
    connectivity: ConnectivityResults,
    *,
    display_percentile: float = 90.0,
    max_lines: int = 250,
) -> int:
    nonzero = [row for row in connectivity.per_cone if row["magnitude_weight"] > 0]
    if not nonzero:
        return 0
    magnitudes = np.asarray([row["magnitude_weight"] for row in nonzero])
    threshold = float(np.percentile(magnitudes, display_percentile))
    selected = [row for row in nonzero if row["magnitude_weight"] >= threshold]
    selected = sorted(selected, key=lambda row: row["magnitude_weight"], reverse=True)
    selected = selected[:max_lines]
    max_weight = max(row["magnitude_weight"] for row in selected)

    for row in reversed(selected):
        relative = row["magnitude_weight"] / max_weight if max_weight > 0 else 0
        ax.plot(
            [rf.peak_projected_um[0], row["x_projected_um"]],
            [rf.peak_projected_um[1], row["y_projected_um"]],
            color=CONE_COLOURS[row["cone_type"]],
            linewidth=0.25 + 1.75 * relative,
            alpha=0.08 + 0.72 * relative,
            linestyle="-" if row["effective_rf_value"] >= 0 else "--",
            zorder=6,
        )
    return len(selected)


def create_connectivity_figure(
    crop: AlignedCrop,
    rf_grid: np.ndarray,
    rf: RFData,
    mosaic: ConeMosaic,
    connectivity: ConnectivityResults,
    *,
    cell_index: int,
    channel: str | int,
    corrected_stim_um_per_px: float,
    rf_alpha: float = 0.48,
    cone_alpha: float = 0.85,
    spider_display_percentile: float = 90.0,
    spider_max_lines: int = 250,
    scale_bar_um: float = 50.0,
    label_connected_cone_ids: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Create a four-part single-cell cone-connectivity summary.

    Layout
    ------
    1. Transformed cone mosaic + aligned RF.
    2. Transformed cone mosaic + ALL interpolated cones.
    3. Transformed cone mosaic + aligned RF + weighted connectivity lines,
       showing ONLY the cones selected for the spider connections.
    4. Cone-type weight radar plot + numerical summary.

    The connected cones in panel 3 are selected using exactly the same
    percentile and max-line rules as the spider connections.
    """

    # -------------------------------------------------------------------------
    # Shared plot preparation
    # -------------------------------------------------------------------------
    central_extent = crop.central_extent_um

    x_min, x_max, y_min, y_max = crop.extent_um
    image_extent = [
        x_min,
        x_max,
        y_max,
        y_min,
    ]

    display_image = _normalise_for_display(crop.image)

    norm = _rf_norm(
        rf_grid,
        rf.signed,
    )

    cmap = "coolwarm" if rf.signed else "Reds"

    # -------------------------------------------------------------------------
    # Select the cones considered "connected" for display.
    #
    # These use the same percentile and maximum-number settings as the
    # spider connections.
    # -------------------------------------------------------------------------
    nonzero_rows = [row for row in connectivity.per_cone if row["magnitude_weight"] > 0]

    if nonzero_rows:
        magnitudes = np.asarray(
            [row["magnitude_weight"] for row in nonzero_rows],
            dtype=float,
        )

        spider_threshold = float(
            np.percentile(
                magnitudes,
                spider_display_percentile,
            )
        )

        connected_rows = [
            row for row in nonzero_rows if row["magnitude_weight"] >= spider_threshold
        ]

        connected_rows = sorted(
            connected_rows,
            key=lambda row: row["magnitude_weight"],
            reverse=True,
        )

        connected_rows = connected_rows[:spider_max_lines]

    else:
        connected_rows = []

    # =========================================================================
    # FIGURE LAYOUT
    # =========================================================================
    fig = plt.figure(
        figsize=(25, 7),
        constrained_layout=True,
    )

    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[
            1.0,
            1.0,
            1.0,
            0.92,
        ],
    )

    ax_original = fig.add_subplot(grid[0, 0])

    ax_all_cones = fig.add_subplot(grid[0, 1])

    ax_spider = fig.add_subplot(grid[0, 2])

    summary_grid = grid[0, 3].subgridspec(
        2,
        1,
        height_ratios=[
            1.1,
            1.0,
        ],
    )

    ax_radar = fig.add_subplot(
        summary_grid[0, 0],
        projection="polar",
    )

    ax_text = fig.add_subplot(summary_grid[1, 0])

    # =========================================================================
    # PANEL 1
    # RF + transformed cone mosaic
    # =========================================================================
    ax_original.imshow(
        display_image,
        extent=image_extent,
        origin="upper",
        zorder=1,
    )

    rf_im = ax_original.imshow(
        rf_grid,
        extent=image_extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        alpha=rf_alpha,
        interpolation="nearest",
        zorder=4,
    )

    # RF peak
    ax_original.scatter(
        rf.peak_projected_um[0],
        rf.peak_projected_um[1],
        marker="x",
        s=70,
        linewidths=1.8,
        color="white",
        zorder=10,
    )

    _set_projected_crop_limits(
        ax_original,
        central_extent,
    )

    _add_scale_bar_projected(
        ax_original,
        central_extent,
        length_um=scale_bar_um,
    )

    ax_original.set_title("Cone mosaic + aligned RF")

    fig.colorbar(
        rf_im,
        ax=ax_original,
        fraction=0.045,
        pad=0.02,
        label="RF value",
    )

    # =========================================================================
    # PANEL 2
    # Cone mosaic + ALL interpolated cones
    # =========================================================================
    ax_all_cones.imshow(
        display_image,
        extent=image_extent,
        origin="upper",
        zorder=1,
    )

    _plot_cones(
        ax_all_cones,
        mosaic,
        alpha=cone_alpha,
        draw_all_types=True,
    )

    _set_projected_crop_limits(
        ax_all_cones,
        central_extent,
    )

    _add_scale_bar_projected(
        ax_all_cones,
        central_extent,
        length_um=scale_bar_um,
    )

    ax_all_cones.set_title("Cone mosaic + all interpolated cones")

    cone_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=CONE_COLOURS[cone_type],
            label=cone_type,
        )
        for cone_type in CONE_TYPES
    ]

    ax_all_cones.legend(
        handles=cone_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    # =========================================================================
    # PANEL 3
    # Same as PANEL 1 + weighted RF-to-cone connection lines
    # + small circles at connected cone locations
    # =========================================================================
    ax_spider.imshow(
        display_image,
        extent=image_extent,
        origin="upper",
        zorder=1,
    )

    # Same RF overlay as Panel 1
    ax_spider.imshow(
        rf_grid,
        extent=image_extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        alpha=rf_alpha,
        interpolation="nearest",
        zorder=4,
    )

    # Add weighted connection lines
    drawn_links = _draw_spider_connections(
        ax_spider,
        rf,
        connectivity,
        display_percentile=spider_display_percentile,
        max_lines=spider_max_lines,
    )

    # Add small circles at the cone locations for the displayed links
    for row in connected_rows:
        cone_type = row["cone_type"]
        x = row["x_projected_um"]
        y = row["y_projected_um"]

        ax_spider.scatter(
            x,
            y,
            s=28,  # same size for all cones
            facecolor=CONE_COLOURS[
                cone_type
            ],  # or set to "white" if you want all same colour
            edgecolor="white",
            linewidth=0.5,
            zorder=12,
        )

    # Same RF peak marker as Panel 1
    ax_spider.scatter(
        rf.peak_projected_um[0],
        rf.peak_projected_um[1],
        marker="x",
        s=70,
        linewidths=1.8,
        color="white",
        zorder=10,
    )

    _set_projected_crop_limits(
        ax_spider,
        central_extent,
    )

    _add_scale_bar_projected(
        ax_spider,
        central_extent,
        length_um=scale_bar_um,
    )

    ax_spider.set_title(
        "Cone mosaic + aligned RF + connections\n"
        f">= {spider_display_percentile:g}th percentile, "
        f"{drawn_links} links shown"
    )

    # Legend only explains line style
    spider_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            label="positive RF",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="negative RF",
        ),
    ]

    ax_spider.legend(
        handles=spider_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    # =========================================================================
    # PANEL 4A
    # Radar plot: cone-type RF weight fractions
    # =========================================================================
    fractions = np.array(
        [
            connectivity.summary[cone_type]["fraction_abs_weight"]
            for cone_type in CONE_TYPES
        ],
        dtype=float,
    )

    angles = np.linspace(
        0,
        2 * np.pi,
        len(CONE_TYPES),
        endpoint=False,
    )

    closed_angles = np.r_[
        angles,
        angles[0],
    ]

    closed_fractions = np.r_[
        fractions,
        fractions[0],
    ]

    ax_radar.plot(
        closed_angles,
        closed_fractions,
        linewidth=2,
    )

    ax_radar.fill(
        closed_angles,
        closed_fractions,
        alpha=0.15,
    )

    ax_radar.set_xticks(angles)

    ax_radar.set_xticklabels(CONE_TYPES)

    ax_radar.set_ylim(
        0,
        max(
            0.25,
            float(np.ceil(fractions.max() * 10) / 10),
        ),
    )

    ax_radar.set_title(
        "Fraction of total RF cone weight",
        pad=18,
    )

    # =========================================================================
    # PANEL 4B
    # Numerical summary
    # =========================================================================
    ax_text.axis("off")

    heading = f"Cell {cell_index} | " f"channel {channel}\n" f"RF: {rf.variable_name}; "

    if rf.quality is not None:
        heading += f"\nQI = " f"{rf.quality:.2f}"

    ax_text.text(
        0.0,
        1.0,
        heading,
        va="top",
        ha="left",
        fontsize=10,
        weight="bold",
    )

    y = 0.78

    for cone_type in CONE_TYPES:
        stats = connectivity.summary[cone_type]

        line = (
            f"{cone_type:>5}: "
            f"n={stats['count']:4d}\n"
            f"mean weight per cone="
            f"{stats['mean_abs_weight_per_cone']:.4f}"
        )

        ax_text.text(
            0.0,
            y,
            line,
            va="top",
            ha="left",
            fontsize=9.5,
            color=CONE_COLOURS[cone_type],
            family="monospace",
        )

        y -= 0.18

    # =========================================================================
    # Overall title
    # =========================================================================
    fig.suptitle(
        f"Single-cell cone input analysis — " f"cell {cell_index}",
        fontsize=15,
        weight="bold",
    )

    # =========================================================================
    # Save
    # =========================================================================
    if save_path is not None:
        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        print(f"Saved connectivity figure: " f"{save_path}")

    return fig


def create_detection_qc_figure(
    crop: AlignedCrop,
    mosaic: ConeMosaic,
    *,
    save_path: str | Path | None = None,
    dpi: int = 250,
) -> plt.Figure:
    """Optional QC plot for transformed crop, threshold and final interpolation."""
    x_min, x_max, y_min, y_max = crop.extent_um
    image_extent = [x_min, x_max, y_max, y_min]
    display_image = _normalise_for_display(crop.image)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    axes[0].imshow(display_image, extent=image_extent, origin="upper")
    axes[0].set_title("Transformed cone crop")
    axes[1].imshow(
        mosaic.binary_clean, extent=image_extent, origin="upper", cmap="gray"
    )
    axes[1].set_title("Detected blue-cone blobs")
    axes[2].imshow(display_image, extent=image_extent, origin="upper")
    _plot_cones(axes[2], mosaic)
    axes[2].set_title("Interpolated mosaic")
    for ax in axes:
        _set_projected_crop_limits(ax, crop.central_extent_um)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved interpolation QC figure: {save_path}")
    return fig


# -----------------------------------------------------------------------------
# OUTPUT TABLES
# -----------------------------------------------------------------------------


def save_connectivity_outputs(
    output_dir: str | Path,
    *,
    cell_index: int,
    connectivity: ConnectivityResults,
    geometry: AlignmentGeometry,
    rf: RFData,
    mosaic: ConeMosaic,
) -> dict[str, Path]:
    """Save one per-cone CSV, one summary CSV and one metadata JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"cell_{cell_index}_cone_connectivity"

    per_cone_path = output_dir / f"{prefix}_per_cone.csv"
    if connectivity.per_cone:
        fieldnames = list(connectivity.per_cone[0].keys())
        with per_cone_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(connectivity.per_cone)
    else:
        per_cone_path.write_text("", encoding="utf-8")

    summary_path = output_dir / f"{prefix}_summary.csv"
    summary_rows = [
        {"cone_type": cone_type, **connectivity.summary[cone_type]}
        for cone_type in CONE_TYPES
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata_path = output_dir / f"{prefix}_metadata.json"
    metadata = {
        "cell_index": cell_index,
        "rf_variable": rf.variable_name,
        "rf_signed": rf.signed,
        "rf_normalised": rf.normalised,
        "quality": rf.quality,
        "tilt": rf.tilt,
        "rf_peak_raw_um": rf.peak_raw_um.tolist(),
        "rf_peak_projected_um": rf.peak_projected_um.tolist(),
        "um_per_px_image3": geometry.um_per_px_image3,
        "corrected_stim_um_per_px": geometry.corrected_stim_um_per_px,
        "rf_scale_factor": geometry.rf_scale_factor,
        "mosaic_row_angle_deg": mosaic.row_angle_deg,
        "mosaic_row_spacing_um": mosaic.row_spacing_um,
        "cone_counts": {
            cone_type: len(mosaic.points_projected_um[cone_type])
            for cone_type in CONE_TYPES
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved per-cone table: {per_cone_path}")
    print(f"Saved summary table:  {summary_path}")
    print(f"Saved metadata:       {metadata_path}")
    return {
        "per_cone_csv": per_cone_path,
        "summary_csv": summary_path,
        "metadata_json": metadata_path,
    }


def _select_connected_cones_for_display(
    connectivity: ConnectivityResults,
    *,
    display_percentile: float = 90.0,
    max_cones: int = 250,
    use_only_nonzero_weights: bool = True,
) -> list[dict[str, Any]]:
    """
    Select cones to label in the 'connected cones' panel.

    By default this matches the spider-display logic approximately:
    keep cones with non-zero magnitude weight, then take the top
    display_percentile by magnitude_weight, capped at max_cones.
    """
    rows = connectivity.per_cone

    if use_only_nonzero_weights:
        rows = [row for row in rows if row["magnitude_weight"] > 0]

    if len(rows) == 0:
        return []

    weights = np.asarray([row["magnitude_weight"] for row in rows], dtype=float)
    cutoff = np.percentile(weights, display_percentile)

    selected = [row for row in rows if row["magnitude_weight"] >= cutoff]

    selected = sorted(
        selected,
        key=lambda row: row["magnitude_weight"],
        reverse=True,
    )

    return selected[:max_cones]


def _label_connected_cones(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    fontsize: float = 6,
    alpha: float = 0.95,
    show_ids: bool = True,
) -> None:
    """
    Plot and optionally label only the selected connected cones.
    """
    for row in rows:
        x = row["x_projected_um"]
        y = row["y_projected_um"]
        cone_type = row["cone_type"]
        cone_id = row["cone_id"]
        colour = CONE_COLOURS[cone_type]

        ax.scatter(
            x,
            y,
            s=28,
            color=colour,
            alpha=alpha,
            zorder=10,
        )

        if show_ids:
            ax.text(
                x + 1.5,
                y - 1.5,
                str(cone_id),
                color=colour,
                fontsize=fontsize,
                zorder=11,
            )
