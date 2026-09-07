from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.transforms import Affine2D
from polarspike import Overview

"""Create per-cell retinal response plots and a combined summary figure.

The script combines six analyses for one recorded retinal ganglion cell:

1. One receptive-field (RF) image for every channel present in the selected
   RF variable, aligned to the cone-mosaic image.
2. The temporal spike-triggered average (STA) from one reference channel.
3. Single chromatic flash responses.
4. Contrast-step responses.
5. Chirp responses.
6. Moving-bar direction/orientation responses.

The NetCDF dataset is deliberately loaded in the first executable PyCharm cell
at the bottom of this file. All plotting and analysis parameters are configured
afterwards. The already-open ``xarray.Dataset`` is then passed to
``CellPlotSession``. Consequently, changing figure settings, channel choices,
filter thresholds, stimulus indices, or alignment parameters does not require
reading the large NetCDF file again.

RF-channel behaviour
--------------------
The script examines the ``channel`` coordinate of ``cfg.rf.rf_variable``:

* one RF channel -> one RF subplot;
* two RF channels -> two separately labelled RF subplots;
* more than two channels -> all channels are plotted in a compact two-column
  RF grid.

``RFConfig.channel`` remains useful as the reference channel for the STA, RF
quality, tilt, cell filtering, and ranking. It no longer limits which channels
appear in the RF portion of the summary figure.
"""
# %% Constants

# Moving bar / edge directions, as projected on the MEA - using the original fps_py
DIRECTION_NAMES = np.array(
    [
        "up",
        "down",
        "right",
        "left",
        "up-right",
        "down-right",
        "up-left",
        "down-left",
    ]
)

# Where "up" is 0 degrees
DIRECTIONS_DEG = np.array(
    [
        0,
        180,
        90,
        270,
        45,
        135,
        315,
        225,
    ],
    dtype=float,
)
SCF_STIMULUS_COLORS = (
    "#fd1717",
    "#000000",
    "#fe7c7c",
    "#000000",
    "#fafe7c",
    "#000000",
    "#dffc7e",
    "#000000",
    "#8afe7c",
    "#000000",
    "#7cfcfe",
    "#000000",
    "#7c86fe",
    "#000000",
    "#fe7cfe",
    "#000000",
)
SCF_STIMULUS_LABELS = (
    "660",
    "",
    "610",
    "",
    "560",
    "",
    "535",
    "",
    "500",
    "",
    "460",
    "",
    "420",
    "",
    "365nm",
    "",
)


# %% Configuration dataclasses
@dataclass
class RFConfig:
    # Reference channel used for STA extraction, quality/tilt lookup, filtering
    # and cell ranking. All channels in ``rf_variable`` are still plotted.
    # Set this to None to use the first RF channel automatically.
    channel: Any | None = None
    rf_variable: str = "cm_most_important"
    quality_variable: str = "quality"
    tilt_variable: str = "tilt"
    transform_tilt_reciprocal: bool = False
    dataset_assumed_um_per_px: float = 2.0
    displayed_stim_width_px: int = 800
    displayed_stim_height_px: int = 800
    # Match the main alignment script exactly:
    # RF data are always mirrored left <-> right and raw RF coordinates are
    # used directly in projected-stimulus coordinates (identity transform).
    clip_rf_to_displayed_stimulus: bool = True
    normalise_rf: bool = True
    rf_alpha: float = 0.5
    cone_alpha: float = 0.9
    rf_cmap_positive: str = "Reds"
    rf_cmap_signed: str = "coolwarm"
    rf_vmax_percentile: float = 99.5
    zoom_half_width_um: float = 200.0
    show_peak: bool = False
    sta_variable_candidates: tuple[str, ...] = (
        "sta_single_pixel",
        "weighted_sta",
        "sta",
        "sta_timecourse",
    )
    sta_time_dim_candidates: tuple[str, ...] = (
        "time",
        "time_max",
        "sta_time",
        "lag",
        "frame",
    )
    sta_baseline: float | None = None
    sta_normalise: bool = False
    sta_reverse_time: bool = False
    sta_time_before_spike_s: float = 0.8
    sta_time_step_s: float | None = None


# Parameters describing the repeated single-chromatic-flash stimulus.
@dataclass
class SCFConfig:
    stimulus_index: int
    duration_s: float = 32.0
    expected_repeats: int = 5
    bin_size_s: float = 0.05
    time_col_candidates: tuple[str, ...] = (
        "times_relative",
        "times",
        "times_triggered",
        "time",
    )
    repeat_col_candidates: tuple[str, ...] = ("repeat", "repeats", "trial")
    stimulus_span_s: float = 2.0
    title: str = "Single chromatic flash"


# Parameters describing the sequence of positive and negative contrast steps.
@dataclass
class CStepsConfig:
    stimulus_index: int = 1
    duration_s: float = 40.0
    expected_repeats: int = 3
    bin_size_s: float = 0.05

    time_col_candidates: tuple[str, ...] = (
        "times_relative",
        "times",
        "times_triggered",
        "time",
    )

    repeat_col_candidates: tuple[str, ...] = (
        "repeat",
        "repeats",
        "trial",
    )

    step_span_s: float = 2.0

    contrasts_percent: tuple[float, ...] = (
        100,
        -100,
        90,
        -90,
        80,
        -80,
        70,
        -70,
        60,
        -60,
        50,
        -50,
        40,
        -40,
        30,
        -30,
        20,
        -20,
        10,
        -10,
    )

    title: str = "Contrast steps"


# Timing and frequency mapping for the temporal-frequency chirp stimulus.
@dataclass
class ChirpConfig:
    stimulus_index: int = 2

    expected_repeats: int = 3
    bin_size_s: float = 0.05

    # Plot from stimulus trigger to the end of the active chirp
    plot_start_s: float = 0.0
    plot_end_s: float = 33.0

    # Active chirp starts 3 seconds after the trigger
    chirp_offset_s: float = 3.0
    chirp_duration_s: float = 30.0

    # times_triggered is already aligned separately for every repeat
    time_col: str = "times_triggered"
    repeat_col: str = "repeat"

    start_frequency_hz: float = 1.0
    end_frequency_hz: float = 30.0

    frequency_tick_values: tuple[float, ...] = (
        1,
        2,
        5,
        10,
        20,
        30,
    )
    title: str = "Chirp"


# Moving-bar timing, direction labels, metrics and compact-panel geometry.
@dataclass
class OSConfig:
    stimulus_index: int
    frames_per_direction: int = 300
    seconds_per_frame: float = 1 / 60
    expected_repeats: int = 10
    sample_rate_hz: float = 20000.0
    time_unit: str = "seconds"
    trigger_col: str = "trigger"
    repeat_col: str = "repeat"
    time_col: str = "times_triggered"
    bin_size_s: float = 0.1
    response_metric: str = "total_spikes"
    direction_names: np.ndarray = field(default_factory=lambda: DIRECTION_NAMES.copy())
    directions_deg: np.ndarray = field(default_factory=lambda: DIRECTIONS_DEG.copy())
    raster_facecolor: str = "0.90"
    psth_facecolor: str = "white"
    show_peak_direction_line: bool = True
    show_orientation_axis: bool = True
    show_motion_direction_arrows: bool = True
    polar_arrow_radius: float = 0.60
    polar_arrow_length: float = 0.075
    polar_arrow_linewidth: float = 1.8
    polar_arrow_mutation_scale: float = 18
    raster_box_width: float = 0.16
    raster_box_height: float = 0.10
    raster_gap_from_arrow: float = 0.035
    psth_box_height: float = 0.045
    psth_gap: float = 0.006
    show_psth: bool = True
    show_psth_scale: bool = True
    title: str = "OS / DS"


# Top-level object that groups paths, sub-configurations and cell filters.
@dataclass
class PlotConfig:
    recording_name: str
    overview_path: Path
    noise_data_path: Path
    cone_image_path: Path
    alignment_cache_path: Path
    output_dir: Path
    rf: RFConfig
    scf: SCFConfig
    csteps: CStepsConfig
    chirp: ChirpConfig
    os: OSConfig
    selected_cells: Sequence[int] | None = None
    min_rf_quality: float | None = None
    min_tilt: float | None = None
    max_tilt: float | None = None
    central_cells_path: Path | None = None
    max_cells: int | None = None
    save: bool = True
    show: bool = False
    figure_dpi: int = 200
    save_dpi: int = 300
    image_format: str = "png"


# Precomputed image/RF transforms and scale values reused for every cell.
@dataclass
class AlignmentState:
    cone_image: np.ndarray
    A_image1_to_projected_um: np.ndarray
    A_raw_rf_to_projected_um: np.ndarray
    corrected_stim_um_per_px: float
    rf_scale_factor: float


# %% Shared conversion, selection and coordinate helpers
def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if isinstance(frame, pl.LazyFrame):
        return frame.collect().to_pandas()
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str:
    columns = set(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(
        f"None of these columns were found: {tuple(candidates)}. Available: {sorted(columns)}"
    )


def _safe_scalar(da: xr.DataArray | None) -> float:
    if da is None:
        return np.nan
    arr = np.asarray(da.values).squeeze()
    return float(arr) if arr.size == 1 else np.nan


def _select_cell(
    da: xr.DataArray,
    cell_id: int,
    channel: Any | None = None,
) -> xr.DataArray:
    """Select one cell and, where appropriate, one named channel.

    Not every variable has a channel dimension. For example, some derived
    metrics may be stored only by ``cell_index``. In that case ``channel`` is
    ignored and the cell is still selected normally.
    """

    out = da.sel(cell_index=int(cell_id))

    if channel is not None and ("channel" in out.dims or "channel" in out.coords):
        out = out.sel(channel=channel)

    return out.squeeze(drop=True)


def _python_scalar(value: Any) -> Any:
    """Convert NumPy scalar channel labels into ordinary Python values."""

    return value.item() if isinstance(value, np.generic) else value


def get_rf_channels(dataset: xr.Dataset, rf_variable: str) -> list[Any | None]:
    """Return the channels represented by one RF variable.

    The channel count is taken from the RF variable itself rather than from the
    dataset-wide ``channel`` coordinate. This matters when different variables
    contain different subsets of channels. A variable with no channel dimension
    is treated as a single, unlabelled RF and represented by ``None``.
    """

    if rf_variable not in dataset:
        raise KeyError(
            f"RF variable {rf_variable!r} was not found. "
            f"Available variables: {list(dataset.data_vars)}"
        )

    da = dataset[rf_variable]

    if "channel" not in da.dims:
        return [None]

    if "channel" in da.coords:
        values = np.asarray(da.coords["channel"].values).reshape(-1)
    else:
        # A channel dimension should normally have coordinate labels, but
        # positional labels make the function robust to an unlabelled dimension.
        values = np.arange(da.sizes["channel"])

    channels = [_python_scalar(value) for value in values]

    if not channels:
        raise ValueError(f"RF variable {rf_variable!r} has an empty channel dimension.")

    return channels


def resolve_reference_channel(
    dataset: xr.Dataset,
    rf_variable: str,
    requested_channel: Any | None,
) -> Any | None:
    """Resolve the channel used for STA/quality/tilt operations.

    RF plotting itself always uses every RF channel. The reference channel is
    needed only for analyses that are intentionally single-channel. If the user
    supplies ``None``, the first RF channel is selected automatically.
    """

    rf_channels = get_rf_channels(dataset, rf_variable)

    if rf_channels == [None]:
        return None

    if requested_channel is None:
        return rf_channels[0]

    if requested_channel not in rf_channels:
        raise KeyError(
            f"Reference channel {requested_channel!r} is not present in "
            f"RF variable {rf_variable!r}. Available RF channels: {rf_channels}"
        )

    return requested_channel


def describe_loaded_dataset(dataset: xr.Dataset) -> None:
    """Print the information most useful when configuring the script.

    Run this immediately after opening the NetCDF file. It exposes dimensions,
    variables and channel labels before the configuration object is created, so
    channel names can be copied directly into ``RFConfig.channel``.
    """

    print("Loaded NetCDF dataset")
    print("Dimensions:", dict(dataset.sizes))
    print("Data variables:", list(dataset.data_vars))

    if "channel" in dataset.coords:
        channels = [
            _python_scalar(value)
            for value in np.asarray(dataset.coords["channel"].values).reshape(-1)
        ]
        print("Dataset channel coordinate:", channels)
    else:
        print("Dataset has no global 'channel' coordinate.")


def _coordinate_edges_from_centres(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.size == 1:
        return np.array([coords[0] - 0.5, coords[0] + 0.5])
    mid = (coords[:-1] + coords[1:]) / 2
    first = coords[0] - (mid[0] - coords[0])
    last = coords[-1] + (coords[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _flip_rf_left_right(da: xr.DataArray) -> xr.DataArray:
    """Mirror RF data left <-> right while preserving x-coordinate labels.

    This is the same operation used by the main alignment script: the RF DATA
    are flipped along the x dimension, but the physical x-coordinate values are
    deliberately left unchanged.
    """
    if "x" not in da.dims:
        raise ValueError("Input DataArray must include an 'x' dimension.")

    x_axis = da.get_axis_num("x")
    return da.copy(data=np.flip(da.values, axis=x_axis), deep=True)


def _transform_points(A: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    hom = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    return (hom @ A.T)[:, :2]


def _normalise_repeat_labels(
    repeats: np.ndarray, expected_repeats: int
) -> tuple[np.ndarray, dict[int, int]]:
    unique_values = np.sort(np.unique(repeats.astype(int)))
    if len(unique_values) == 0:
        mapping: dict[int, int] = {}
    elif unique_values.min() == 0 and unique_values.max() == expected_repeats - 1:
        mapping = {int(x): int(x) for x in unique_values}
    elif unique_values.min() == 1 and unique_values.max() == expected_repeats:
        mapping = {int(x): int(x) - 1 for x in unique_values}
    else:
        mapping = {int(x): i for i, x in enumerate(unique_values)}
    indices = np.array([mapping.get(int(x), -1) for x in repeats], dtype=int)
    return (indices, mapping)


def _container_transform(container):
    return getattr(container, "transSubfigure", container.transFigure)


# %% Alignment loading, dataset inspection and cell selection
def load_alignment_state(cfg: PlotConfig) -> AlignmentState:
    cache = np.load(cfg.alignment_cache_path, allow_pickle=True)
    required = [
        "tform_2_to_3_params",
        "um_per_px_image3",
        "stim_edge_top_px",
        "stim_edge_bottom_px",
        "visual_stim_center_px_image3",
    ]
    missing = [key for key in required if key not in cache.files]
    if missing:
        raise KeyError(
            f"Alignment cache is missing keys {missing}. Available: {cache.files}"
        )
    tform_2_to_3 = np.asarray(cache["tform_2_to_3_params"], dtype=float)
    um_per_px_image3 = float(np.asarray(cache["um_per_px_image3"]).squeeze())
    stim_edge_top_px = np.asarray(cache["stim_edge_top_px"], dtype=float)
    stim_edge_bottom_px = np.asarray(cache["stim_edge_bottom_px"], dtype=float)
    stim_center = np.asarray(cache["visual_stim_center_px_image3"], dtype=float)
    blue_line_vec = stim_edge_bottom_px - stim_edge_top_px
    blue_line_length_px = np.linalg.norm(blue_line_vec)
    if blue_line_length_px == 0:
        raise ValueError("The cached stimulus-edge clicks are identical.")
    corrected_stim_um_per_px = (
        blue_line_length_px * um_per_px_image3 / cfg.rf.displayed_stim_height_px
    )
    rf_scale_factor = corrected_stim_um_per_px / cfg.rf.dataset_assumed_um_per_px
    down_unit = blue_line_vec / blue_line_length_px
    right_unit = np.array([down_unit[1], -down_unit[0]], dtype=float)
    px_per_um_image3 = 1.0 / um_per_px_image3
    M = np.column_stack([right_unit * px_per_um_image3, down_unit * px_per_um_image3])
    A_projected_to_image3 = np.eye(3)
    A_projected_to_image3[:2, :2] = M
    A_projected_to_image3[:2, 2] = stim_center
    A_image3_to_projected = np.linalg.inv(A_projected_to_image3)
    A_image1_to_projected = A_image3_to_projected @ tform_2_to_3

    # Match the main alignment script exactly: raw RF/stimulus coordinates are
    # already interpreted in projected-stimulus coordinates. No extra rotation
    # or axis swap is applied here.
    A_raw_to_projected = np.eye(3, dtype=float)
    return AlignmentState(
        cone_image=iio.imread(cfg.cone_image_path),
        A_image1_to_projected_um=A_image1_to_projected,
        A_raw_rf_to_projected_um=A_raw_to_projected,
        corrected_stim_um_per_px=corrected_stim_um_per_px,
        rf_scale_factor=rf_scale_factor,
    )


def get_cell_metric(
    dataset: xr.Dataset,
    variable: str,
    cell_id: int,
    channel: Any | None,
) -> float:
    """Read a scalar per-cell metric, optionally from a particular channel."""

    if variable not in dataset:
        return np.nan

    try:
        return _safe_scalar(_select_cell(dataset[variable], cell_id, channel))
    except (KeyError, ValueError, IndexError):
        return np.nan


def get_quality_and_tilt_for_channel(
    dataset: xr.Dataset,
    cell_id: int,
    cfg: PlotConfig,
    channel: Any | None,
) -> tuple[float, float]:
    """Return RF quality and tilt for one RF channel.

    These values are displayed above each RF subplot. If a metric is not stored
    for that channel, ``NaN`` is returned rather than preventing the RF image
    itself from being plotted.
    """

    quality = get_cell_metric(dataset, cfg.rf.quality_variable, cell_id, channel)
    tilt = get_cell_metric(dataset, cfg.rf.tilt_variable, cell_id, channel)

    if cfg.rf.transform_tilt_reciprocal and np.isfinite(tilt) and tilt != 0:
        tilt = 1 / tilt

    return quality, tilt


def get_quality_and_tilt(
    dataset: xr.Dataset,
    cell_id: int,
    cfg: PlotConfig,
) -> tuple[float, float]:
    """Return quality/tilt from the configured reference channel."""

    reference_channel = resolve_reference_channel(
        dataset=dataset,
        rf_variable=cfg.rf.rf_variable,
        requested_channel=cfg.rf.channel,
    )

    return get_quality_and_tilt_for_channel(
        dataset=dataset,
        cell_id=cell_id,
        cfg=cfg,
        channel=reference_channel,
    )


def select_cells(dataset: xr.Dataset, cfg: PlotConfig) -> list[int]:
    cells = [int(x) for x in dataset["cell_index"].values]
    if cfg.selected_cells is not None:
        requested = {int(x) for x in cfg.selected_cells}
        cells = [cell for cell in cells if cell in requested]
    if cfg.central_cells_path is not None:
        central = {int(x) for x in np.load(cfg.central_cells_path)}
        cells = [cell for cell in cells if cell in central]
    selected = []
    for cell in cells:
        quality, tilt = get_quality_and_tilt(dataset, cell, cfg)
        if cfg.min_rf_quality is not None and (
            not np.isfinite(quality) or quality < cfg.min_rf_quality
        ):
            continue
        if cfg.min_tilt is not None and (not np.isfinite(tilt) or tilt < cfg.min_tilt):
            continue
        if cfg.max_tilt is not None and (not np.isfinite(tilt) or tilt > cfg.max_tilt):
            continue
        selected.append(cell)
    if cfg.max_cells is not None:
        selected = selected[: cfg.max_cells]
    return selected


# %% Receptive-field analysis and drawing
def analyse_rf(
    dataset: xr.Dataset,
    cell_id: int,
    cfg: PlotConfig,
    alignment: AlignmentState,
    channel: Any | None,
) -> dict[str, Any]:
    """Extract, normalise and geometrically prepare one RF channel.

    The returned RF is not yet drawn. It contains the pixel values, coordinate
    edges, transformed peak position, colour normalisation and channel label.
    Keeping analysis separate from drawing lets the same result be reused in
    standalone and combined figures.
    """
    if cfg.rf.rf_variable not in dataset:
        raise KeyError(
            f"RF variable {cfg.rf.rf_variable!r} not found. Variables: {list(dataset.data_vars)}"
        )
    da = _select_cell(dataset[cfg.rf.rf_variable], cell_id, channel)
    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError(f"RF data must have x and y dims; got {da.dims}")

    # Same fixed RF orientation as the main alignment script. First put the
    # array into y,x display order, then mirror the DATA left <-> right while
    # retaining the original physical x-coordinate labels.
    da = da.transpose("y", "x")
    da = _flip_rf_left_right(da)
    values = np.asarray(da.values, dtype=float)
    if cfg.rf.normalise_rf:
        if cfg.rf.rf_variable == "cm_most_important":
            scale = np.nanmax(np.abs(values))
        else:
            scale = np.nanmax(values)
        if np.isfinite(scale) and scale != 0:
            values = values / scale
    x_centres = np.asarray(da["x"].values, dtype=float) * alignment.rf_scale_factor
    y_centres = np.asarray(da["y"].values, dtype=float) * alignment.rf_scale_factor
    x_edges = _coordinate_edges_from_centres(x_centres)
    y_edges = _coordinate_edges_from_centres(y_centres)

    # Same displayed-stimulus clipping used by the main alignment script. This
    # is applied before finding the visible RF peak, so the multipanel zoom is
    # centred on the same plotted RF pixel as the main alignment plot.
    if cfg.rf.clip_rf_to_displayed_stimulus:
        displayed_half_x_um = (
            cfg.rf.displayed_stim_width_px * alignment.corrected_stim_um_per_px
        ) / 2
        displayed_half_y_um = (
            cfg.rf.displayed_stim_height_px * alignment.corrected_stim_um_per_px
        ) / 2
        raw_x_grid_um, raw_y_grid_um = np.meshgrid(x_centres, y_centres)
        valid_displayed_mask = (
            (raw_x_grid_um >= -displayed_half_x_um)
            & (raw_x_grid_um <= displayed_half_x_um)
            & (raw_y_grid_um >= -displayed_half_y_um)
            & (raw_y_grid_um <= displayed_half_y_um)
        )
        values = np.where(valid_displayed_mask, values, np.nan)

    peak_projected = np.array([np.nan, np.nan])
    if np.any(np.isfinite(values)):
        if cfg.rf.rf_variable == "cm_most_important":
            peak_flat = np.nanargmax(np.abs(values))
        else:
            peak_flat = np.nanargmax(values)
        py, px = np.unravel_index(peak_flat, values.shape)
        peak_raw = np.array([[x_centres[px], y_centres[py]]])
        peak_projected = _transform_points(
            alignment.A_raw_rf_to_projected_um, peak_raw
        )[0]
    if cfg.rf.rf_variable == "cm_most_important":
        valid = np.abs(values[np.isfinite(values)])
        vmax = np.nanpercentile(valid, cfg.rf.rf_vmax_percentile) if valid.size else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = cfg.rf.rf_cmap_signed
    else:
        valid = values[np.isfinite(values)]
        vmax = np.nanpercentile(valid, cfg.rf.rf_vmax_percentile) if valid.size else 1.0
        vmax = 1.0 if cfg.rf.normalise_rf else vmax
        norm = Normalize(vmin=0, vmax=vmax if vmax != 0 else 1.0)
        cmap = cfg.rf.rf_cmap_positive
    return {
        "channel": channel,
        "values": values,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "peak_projected": peak_projected,
        "norm": norm,
        "cmap": cmap,
    }


def draw_rf(
    ax,
    rf: dict[str, Any],
    cell_id: int,
    quality: float,
    tilt: float,
    cfg: PlotConfig,
    alignment: AlignmentState,
):
    cone = alignment.cone_image
    if cone.ndim == 3 and cone.shape[-1] == 4:
        cone = cone[..., :3]
    h1, w1 = cone.shape[:2]
    ax.imshow(
        cone,
        alpha=cfg.rf.cone_alpha,
        extent=[0, w1, h1, 0],
        origin="upper",
        transform=Affine2D(alignment.A_image1_to_projected_um) + ax.transData,
        zorder=1,
    )
    im = ax.imshow(
        rf["values"],
        cmap=rf["cmap"],
        norm=rf["norm"],
        alpha=cfg.rf.rf_alpha,
        origin="upper",
        interpolation="nearest",
        extent=[
            rf["x_edges"][0],
            rf["x_edges"][-1],
            rf["y_edges"][-1],
            rf["y_edges"][0],
        ],
        transform=Affine2D(alignment.A_raw_rf_to_projected_um) + ax.transData,
        zorder=10,
    )
    peak = rf["peak_projected"]
    if cfg.rf.show_peak and np.all(np.isfinite(peak)):
        ax.scatter(
            peak[0], peak[1], marker="x", s=50, linewidths=1.4, color="black", zorder=20
        )
    if np.all(np.isfinite(peak)):
        half = cfg.rf.zoom_half_width_um
        ax.set_xlim(peak[0] - half, peak[0] + half)
        ax.set_ylim(peak[1] + half, peak[1] - half)
    ax.set_aspect("equal")
    ax.set_xlabel("projected x (µm)")
    ax.set_ylabel("projected y (µm, +down)")
    channel = rf.get("channel")
    channel_title = "Unlabelled RF channel" if channel is None else str(channel)

    ax.set_title(
        f"RF channel: {channel_title}\n"
        f"Aligned RF + cone mosaic | Cell {cell_id} | {cfg.rf.rf_variable}\n"
        f"RF QI: {quality:.3f} | tilt: {tilt:.3f} | corrected pixel size: "
        f"{alignment.corrected_stim_um_per_px:.3f} µm/px",
        fontsize=9,
    )
    return im


# %% STA analysis and drawing
def analyse_sta(
    dataset: xr.Dataset,
    cell_id: int,
    cfg: PlotConfig,
    channel: Any | None = None,
) -> dict[str, Any]:
    """Extract one STA timecourse for one cell and one RF channel.

    The RF and STA variables normally share the same ``channel`` coordinate.
    Passing the RF channel into this function therefore produces the STA that
    belongs to that specific RF. If the STA variable has no channel dimension,
    ``_select_cell`` simply ignores the channel and the same unchannelled STA is
    returned for each RF panel.

    When this function is called without an explicit channel, it falls back to
    ``RFConfig.channel`` (or the first RF channel if that setting is ``None``).
    This preserves the old single-channel behaviour for external code that calls
    ``analyse_sta`` directly.
    """

    variable = next((v for v in cfg.rf.sta_variable_candidates if v in dataset), None)
    if variable is None:
        raise KeyError(
            f"No STA variable found among {cfg.rf.sta_variable_candidates}. "
            f"Variables: {list(dataset.data_vars)}"
        )

    selected_channel = channel
    if selected_channel is None:
        selected_channel = resolve_reference_channel(
            dataset=dataset,
            rf_variable=cfg.rf.rf_variable,
            requested_channel=cfg.rf.channel,
        )

    da = _select_cell(dataset[variable], cell_id, selected_channel)

    time_dim = next((d for d in cfg.rf.sta_time_dim_candidates if d in da.dims), None)
    if time_dim is None:
        non_spatial = [
            d for d in da.dims if d not in {"x", "y", "channel", "cell_index"}
        ]
        if len(non_spatial) != 1:
            raise ValueError(f"Cannot identify STA time dimension from dims {da.dims}")
        time_dim = non_spatial[0]

    # Any remaining non-time dimensions are collapsed. Singleton dimensions are
    # simply removed; larger dimensions are averaged, matching the old script.
    for dim in list(da.dims):
        if dim != time_dim:
            if da.sizes[dim] == 1:
                da = da.isel({dim: 0})
            else:
                da = da.mean(dim=dim, skipna=True)

    values = np.asarray(da.values, dtype=float).squeeze()

    if cfg.rf.sta_baseline is not None:
        values = values - cfg.rf.sta_baseline

    if cfg.rf.sta_normalise:
        scale = np.nanmax(np.abs(values))
        if np.isfinite(scale) and scale != 0:
            values = values / scale

    if time_dim in da.coords and np.issubdtype(da[time_dim].dtype, np.number):
        times = np.asarray(da[time_dim].values, dtype=float)
        if np.nanmax(np.abs(times)) > 10 and cfg.rf.sta_time_step_s is None:
            times = np.linspace(-cfg.rf.sta_time_before_spike_s, 0, len(values))
    elif cfg.rf.sta_time_step_s is not None:
        times = -np.arange(len(values) - 1, -1, -1) * cfg.rf.sta_time_step_s
    else:
        times = np.linspace(-cfg.rf.sta_time_before_spike_s, 0, len(values))

    if cfg.rf.sta_reverse_time:
        values = values[::-1]
        times = times[::-1]

    return {
        "times": times,
        "values": values,
        "variable": variable,
        "channel": selected_channel,
    }


def draw_sta(ax, sta: dict[str, Any], cell_id: int, quality: float, tilt: float):
    """Draw one channel-specific STA and label it with its channel name."""

    ax.plot(sta["times"], sta["values"], color="#3f88c5", linewidth=1.2)
    ax.axvline(0, linestyle="--", color="#3f88c5", linewidth=0.8)
    ax.set_xlabel("time before spike (s)")
    ax.set_ylabel("STA")

    channel = sta.get("channel")
    channel_title = "Unlabelled STA channel" if channel is None else str(channel)

    ax.set_title(
        f"STA channel: {channel_title} ({sta['variable']})\n"
        f"Cell {cell_id} | RF QI: {quality:.3f} | tilt: {tilt:.3f}",
        fontsize=9,
    )


# %% Flash, contrast-step and chirp analysis/drawing
def analyse_scf(spikes_df: pd.DataFrame, cfg: SCFConfig) -> dict[str, Any]:
    if spikes_df.empty:
        bins = np.arange(0, cfg.duration_s + cfg.bin_size_s, cfg.bin_size_s)
        raster = [np.array([]) for _ in range(cfg.expected_repeats)]
        all_x_times = np.array([])
    else:
        time_col = _first_existing(spikes_df.columns, cfg.time_col_candidates)
        repeat_col = next(
            (c for c in cfg.repeat_col_candidates if c in spikes_df.columns), None
        )
        times = spikes_df[time_col].to_numpy(dtype=float)
        if repeat_col is not None:
            repeats_raw = spikes_df[repeat_col].to_numpy(dtype=int)
            repeat_idx, _ = _normalise_repeat_labels(repeats_raw, cfg.expected_repeats)
            x_times = times.copy()
            if np.nanmax(x_times) > cfg.duration_s + 1e-09:
                x_times = np.mod(x_times, cfg.duration_s)
        elif np.nanmax(times) > cfg.duration_s + 1e-09:
            repeat_idx = np.floor(times / cfg.duration_s).astype(int)
            x_times = np.mod(times, cfg.duration_s)
        else:
            repeat_idx = np.zeros(len(times), dtype=int)
            x_times = times.copy()
        valid = (
            np.isfinite(x_times)
            & (x_times >= 0)
            & (x_times <= cfg.duration_s)
            & (repeat_idx >= 0)
            & (repeat_idx < cfg.expected_repeats)
        )
        x_times = x_times[valid]
        repeat_idx = repeat_idx[valid]
        raster = [
            np.asarray(x_times[repeat_idx == rep], dtype=float)
            for rep in range(cfg.expected_repeats)
        ]
        bins = np.arange(0, cfg.duration_s + cfg.bin_size_s, cfg.bin_size_s)
        if bins[-1] < cfg.duration_s:
            bins = np.append(bins, cfg.duration_s)
        all_x_times = x_times
    counts, _ = np.histogram(all_x_times, bins=bins)
    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths / 2
    psth = counts / (bin_widths * max(1, cfg.expected_repeats))
    return {
        "raster_by_repeat": raster,
        "bins": bins,
        "bin_centers": bin_centers,
        "psth": psth,
    }


def analyse_csteps(
    spikes_df: pd.DataFrame,
    cfg: CStepsConfig,
) -> dict[str, Any]:
    """
    Csteps has the same repeated-stimulus raster/PSTH analysis as SCF.
    """

    return analyse_scf(
        spikes_df=spikes_df,
        cfg=cfg,
    )


def analyse_chirp(
    spikes_df: pd.DataFrame,
    cfg: ChirpConfig,
) -> dict[str, Any]:
    """
    Create a three-repeat chirp raster and an average PSTH.

    times_triggered is measured from the beginning of each repeat:
        0–3 s   = pre-chirp buffer
        3–33 s  = active chirp
    """

    bins = np.arange(
        cfg.plot_start_s,
        cfg.plot_end_s + cfg.bin_size_s,
        cfg.bin_size_s,
    )

    if bins[-1] < cfg.plot_end_s:
        bins = np.append(
            bins,
            cfg.plot_end_s,
        )

    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths / 2

    empty_raster = [np.array([]) for _ in range(cfg.expected_repeats)]

    if spikes_df.empty:
        return {
            "raster_by_repeat": empty_raster,
            "bins": bins,
            "bin_centers": bin_centers,
            "psth": np.zeros(len(bin_centers)),
        }

    if cfg.time_col not in spikes_df.columns:
        raise KeyError(
            f"Chirp time column {cfg.time_col!r} is missing. "
            f"Available columns: {list(spikes_df.columns)}"
        )

    if cfg.repeat_col not in spikes_df.columns:
        raise KeyError(
            f"Chirp repeat column {cfg.repeat_col!r} is missing. "
            f"Available columns: {list(spikes_df.columns)}"
        )

    times_in_repeat = spikes_df[cfg.time_col].to_numpy(dtype=float)

    repeat_values = spikes_df[cfg.repeat_col].to_numpy(dtype=int)

    repeat_indices, repeat_mapping = _normalise_repeat_labels(
        repeat_values,
        cfg.expected_repeats,
    )

    valid = (
        np.isfinite(times_in_repeat)
        & (times_in_repeat >= cfg.plot_start_s)
        & (times_in_repeat <= cfg.plot_end_s)
        & (repeat_indices >= 0)
        & (repeat_indices < cfg.expected_repeats)
    )

    times_in_repeat = times_in_repeat[valid]
    repeat_indices = repeat_indices[valid]

    raster_by_repeat = [
        np.asarray(
            times_in_repeat[repeat_indices == repeat_index],
            dtype=float,
        )
        for repeat_index in range(cfg.expected_repeats)
    ]

    counts, _ = np.histogram(
        times_in_repeat,
        bins=bins,
    )

    # Average firing rate across all three repeats
    psth = counts / (bin_widths * cfg.expected_repeats)

    return {
        "raster_by_repeat": raster_by_repeat,
        "bins": bins,
        "bin_centers": bin_centers,
        "psth": psth,
        "repeat_mapping": repeat_mapping,
    }


def draw_scf(
    container,
    scf: dict[str, Any],
    cell_id: int,
    quality: float,
    tilt: float,
    cfg: SCFConfig,
):
    gs = container.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
    ax_psth = container.add_subplot(gs[0])
    ax_raster = container.add_subplot(gs[1], sharex=ax_psth)
    for i, color in enumerate(SCF_STIMULUS_COLORS):
        t0 = i * cfg.stimulus_span_s
        t1 = t0 + cfg.stimulus_span_s
        alpha = 0.15 if color != "#000000" else 0.08
        ax_psth.axvspan(t0, t1, color=color, alpha=alpha, zorder=0)
        ax_raster.axvspan(t0, t1, color=color, alpha=alpha, zorder=0)
    ax_psth.plot(scf["bin_centers"], scf["psth"], color="black", linewidth=1.5)
    ax_psth.set_ylabel("Firing rate\n(spikes/s)", fontsize=12)
    ax_psth.tick_params(axis="y", labelsize=11)
    ax_psth.set_xlim(0, cfg.duration_s)
    ax_psth.set_xticks([])
    for i, label in enumerate(SCF_STIMULUS_LABELS):
        if not label:
            continue
        t0 = i * cfg.stimulus_span_s
        t1 = t0 + cfg.stimulus_span_s
        ax_psth.text(
            (t0 + t1) / 2,
            1.08,
            label,
            transform=ax_psth.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax_raster.eventplot(
        scf["raster_by_repeat"],
        lineoffsets=np.arange(cfg.expected_repeats),
        linelengths=0.7,
        linewidths=0.7,
        colors="black",
    )
    ax_raster.set_xlim(0, cfg.duration_s)
    ax_raster.set_ylim(-0.5, cfg.expected_repeats - 0.5)
    ax_raster.invert_yaxis()
    ax_raster.set_xlabel("Time (s)", fontsize=12)
    ax_raster.set_ylabel("Repeat", fontsize=12)
    ax_raster.set_yticks(np.arange(cfg.expected_repeats))
    ax_raster.set_xticks(np.arange(2, cfg.duration_s + 0.1, 2))
    ax_raster.tick_params(axis="x", labelsize=11)
    ax_raster.tick_params(axis="y", labelsize=10)
    ax_psth.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_psth.set_title(
        f"{cfg.title}\nCell {cell_id} | RF QI: {quality:.3f} | tilt: {tilt:.3f}",
        fontsize=12,
        pad=24,
    )
    return (ax_psth, ax_raster)


def _contrast_to_gray(contrast_percent: float) -> float:
    """
    Convert signed contrast into a grayscale value.

    -100 -> 0.0 = black
       0 -> 0.5 = grey
    +100 -> 1.0 = white
    """

    contrast_fraction = contrast_percent / 100

    return float(
        np.clip(
            0.5 * (1 + contrast_fraction),
            0,
            1,
        )
    )


def draw_csteps(
    container,
    csteps: dict[str, Any],
    cell_id: int,
    quality: float,
    tilt: float,
    cfg: CStepsConfig,
):
    """
    Draw the csteps stimulus strip, PSTH and repeat raster.
    """

    expected_duration = len(cfg.contrasts_percent) * cfg.step_span_s

    if not np.isclose(expected_duration, cfg.duration_s):
        raise ValueError(
            "The csteps timing is inconsistent:\n"
            f"{len(cfg.contrasts_percent)} intervals × "
            f"{cfg.step_span_s} s = {expected_duration} s, "
            f"but duration_s={cfg.duration_s} s."
        )

    gs = container.add_gridspec(
        3,
        1,
        height_ratios=[0.18, 1, 3],
        hspace=0.06,
    )

    ax_stimulus = container.add_subplot(gs[0])
    ax_psth = container.add_subplot(gs[1])
    ax_raster = container.add_subplot(
        gs[2],
        sharex=ax_psth,
    )
    # Draw each contrast interval
    for interval_index, contrast in enumerate(cfg.contrasts_percent):
        t0 = interval_index * cfg.step_span_s
        t1 = t0 + cfg.step_span_s

        gray_value = _contrast_to_gray(contrast)
        gray_colour = str(gray_value)

        # Exact grayscale stimulus strip.
        ax_stimulus.axvspan(
            t0,
            t1,
            color=gray_colour,
            alpha=1,
            ec="0.35",
            linewidth=0.5,
        )

        # Lighter stimulus shading behind data.
        for ax in (ax_psth, ax_raster):
            ax.axvspan(
                t0,
                t1,
                color=gray_colour,
                alpha=0.12,
                zorder=0,
            )

        # Label each interval.
        label_colour = "white" if gray_value < 0.35 else "black"

        ax_stimulus.text(
            (t0 + t1) / 2,
            0.5,
            f"{contrast:+.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            color=label_colour,
        )
    # Stimulus strip formatting
    ax_stimulus.set_xlim(0, cfg.duration_s)
    ax_stimulus.set_ylim(0, 1)
    ax_stimulus.set_xticks([])
    ax_stimulus.set_yticks([])

    for spine in ax_stimulus.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")
    # PSTH
    ax_psth.plot(
        csteps["bin_centers"],
        csteps["psth"],
        color="black",
        linewidth=1.5,
    )

    ax_psth.set_xlim(0, cfg.duration_s)
    ax_psth.set_xticks([])
    ax_psth.set_ylabel(
        "Firing rate\n(spikes/s)",
        fontsize=12,
    )
    ax_psth.tick_params(
        axis="y",
        labelsize=11,
    )
    # Raster
    ax_raster.eventplot(
        csteps["raster_by_repeat"],
        lineoffsets=np.arange(cfg.expected_repeats),
        linelengths=0.7,
        linewidths=0.7,
        colors="black",
    )

    ax_raster.set_xlim(0, cfg.duration_s)
    ax_raster.set_ylim(
        -0.5,
        cfg.expected_repeats - 0.5,
    )
    ax_raster.invert_yaxis()

    ax_raster.set_xlabel(
        "Time (s)",
        fontsize=12,
    )
    ax_raster.set_ylabel(
        "Repeat",
        fontsize=12,
    )

    ax_raster.set_yticks(np.arange(cfg.expected_repeats))
    ax_raster.set_xticks(
        np.arange(
            0,
            cfg.duration_s + 0.1,
            cfg.step_span_s,
        )
    )

    ax_raster.tick_params(
        axis="x",
        labelsize=11,
    )
    ax_raster.tick_params(
        axis="y",
        labelsize=10,
    )

    ax_stimulus.set_title(
        f"{cfg.title}\n"
        f"Cell {cell_id} | "
        f"RF QI: {quality:.3f} | "
        f"tilt: {tilt:.3f}",
        fontsize=12,
        pad=10,
    )

    return (
        ax_stimulus,
        ax_psth,
        ax_raster,
    )


def draw_chirp(
    container,
    chirp: dict[str, Any],
    cell_id: int,
    quality: float,
    tilt: float,
    cfg: ChirpConfig,
):
    """
    Plot the average firing rate above the three-repeat raster.
    """

    gs = container.add_gridspec(
        2,
        1,
        height_ratios=[1, 3],
        hspace=0.05,
    )

    ax_psth = container.add_subplot(gs[0])

    ax_raster = container.add_subplot(
        gs[1],
        sharex=ax_psth,
    )
    # AVERAGE PSTH
    ax_psth.plot(
        chirp["bin_centers"],
        chirp["psth"],
        color="black",
        linewidth=1.5,
    )

    ax_psth.set_xlim(
        cfg.plot_start_s,
        cfg.plot_end_s,
    )

    ax_psth.set_ylabel(
        "Firing rate\n(spikes/s)",
        fontsize=10,
    )

    ax_psth.tick_params(
        axis="y",
        labelsize=9,
    )

    ax_psth.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )
    # EXPONENTIAL FREQUENCY AXIS ABOVE THE PSTH
    frequency_ticks = np.asarray(
        cfg.frequency_tick_values,
        dtype=float,
    )

    # Only keep frequencies within the chirp range
    frequency_ticks = frequency_ticks[
        (frequency_ticks >= cfg.start_frequency_hz)
        & (frequency_ticks <= cfg.end_frequency_hz)
    ]

    # Exponential chirp:
    # f(t) = start_frequency * exp(beta * t)
    beta = np.log(cfg.end_frequency_hz / cfg.start_frequency_hz) / cfg.chirp_duration_s

    # Convert each frequency into time since chirp onset
    time_from_chirp_start = np.log(frequency_ticks / cfg.start_frequency_hz) / beta

    # Add the 3-second pre-chirp buffer
    frequency_tick_times = cfg.chirp_offset_s + time_from_chirp_start

    # Create an axis above the PSTH
    ax_frequency = ax_psth.twiny()

    # It must use the same stimulus-time coordinates as the PSTH
    ax_frequency.set_xlim(
        cfg.plot_start_s,
        cfg.plot_end_s,
    )

    ax_frequency.set_xticks(frequency_tick_times)

    ax_frequency.set_xticklabels([f"{frequency:g}" for frequency in frequency_ticks])

    ax_frequency.set_xlabel(
        "Stimulus frequency (Hz)",
        fontsize=9,
        labelpad=2,
    )

    ax_frequency.tick_params(
        axis="x",
        labelsize=8,
        length=3,
        pad=1,
    )

    # Hide unnecessary borders from the extra axis
    ax_frequency.spines["bottom"].set_visible(False)
    ax_frequency.spines["left"].set_visible(False)
    ax_frequency.spines["right"].set_visible(False)
    # THREE-REPEAT RASTER
    ax_raster.eventplot(
        chirp["raster_by_repeat"],
        lineoffsets=np.arange(cfg.expected_repeats),
        linelengths=0.7,
        linewidths=0.7,
        colors="black",
    )

    ax_raster.set_xlim(
        cfg.plot_start_s,
        cfg.plot_end_s,
    )

    ax_raster.set_ylim(
        -0.5,
        cfg.expected_repeats - 0.5,
    )

    ax_raster.invert_yaxis()

    ax_raster.set_ylabel(
        "Repeat",
        fontsize=10,
    )

    ax_raster.set_yticks(np.arange(cfg.expected_repeats))

    ax_raster.set_yticklabels(
        np.arange(
            1,
            cfg.expected_repeats + 1,
        )
    )

    ax_raster.tick_params(
        axis="y",
        labelsize=9,
    )
    # TIME RELATIVE TO CHIRP ONSET
    chirp_xticks = np.arange(
        cfg.chirp_offset_s,
        cfg.chirp_offset_s + cfg.chirp_duration_s + 0.1,
        5,
    )

    chirp_xticklabels = [f"{tick - cfg.chirp_offset_s:.0f}" for tick in chirp_xticks]

    ax_raster.set_xticks(chirp_xticks)

    ax_raster.set_xticklabels(chirp_xticklabels)

    ax_raster.set_xlabel(
        "Time from chirp start (s)",
        fontsize=10,
    )

    ax_raster.tick_params(
        axis="x",
        labelsize=9,
    )

    # Mark the beginning of the active chirp
    for ax in (ax_psth, ax_raster):
        ax.axvline(
            cfg.chirp_offset_s,
            color="0.35",
            linestyle="--",
            linewidth=0.8,
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

    ax_psth.set_title(
        f"{cfg.title}\n"
        f"Cell {cell_id} | "
        f"RF QI: {quality:.3f} | "
        f"tilt: {tilt:.3f}",
        fontsize=10,
        pad=5,
    )

    return ax_psth, ax_raster


# %% Moving-bar analysis
def split_os_spikes(spikes_df: pd.DataFrame, cfg: OSConfig) -> dict[str, Any]:
    for col in [cfg.time_col, cfg.trigger_col, cfg.repeat_col]:
        if col not in spikes_df.columns:
            raise KeyError(
                f"Column {col!r} not found in moving-bar table. Available: {list(spikes_df.columns)}"
            )
    direction_names = np.asarray(cfg.direction_names)
    n_directions = len(direction_names)
    direction_duration_s = cfg.frames_per_direction * cfg.seconds_per_frame
    frames_per_repeat = cfg.frames_per_direction * n_directions
    trigger_frames = spikes_df[cfg.trigger_col].to_numpy().astype(int)
    repeat_values = spikes_df[cfg.repeat_col].to_numpy().astype(int)
    times_triggered = spikes_df[cfg.time_col].to_numpy().astype(float)
    repeat_indices, _ = _normalise_repeat_labels(repeat_values, cfg.expected_repeats)
    valid = (
        np.isfinite(times_triggered)
        & (trigger_frames >= 0)
        & (trigger_frames < frames_per_repeat)
        & (repeat_indices >= 0)
        & (repeat_indices < cfg.expected_repeats)
    )
    trigger_frames = trigger_frames[valid]
    repeat_indices = repeat_indices[valid]
    times_triggered = times_triggered[valid]
    empty_rasters = {
        str(direction): [np.array([]) for _ in range(cfg.expected_repeats)]
        for direction in direction_names
    }
    if len(trigger_frames) == 0:
        return {
            "raster_by_direction": empty_rasters,
            "spike_counts": np.zeros(n_directions, dtype=int),
            "n_spikes": 0,
        }
    direction_indices = trigger_frames // cfg.frames_per_direction
    if cfg.time_unit == "seconds":
        time_in_repeat_s = times_triggered
    elif cfg.time_unit == "samples":
        time_in_repeat_s = times_triggered / cfg.sample_rate_hz
    elif cfg.time_unit == "auto":
        repeat_duration_s = direction_duration_s * n_directions
        finite_times = times_triggered[np.isfinite(times_triggered)]
        max_time = float(np.nanmax(finite_times)) if finite_times.size else 0.0
        if max_time <= repeat_duration_s * 2:
            time_in_repeat_s = times_triggered
            detected_unit = "seconds"
        else:
            time_in_repeat_s = times_triggered / cfg.sample_rate_hz
            detected_unit = "samples"
        print(
            f"OS time-unit detection: max {cfg.time_col}={max_time:.3f}; interpreting values as {detected_unit}."
        )
    else:
        raise ValueError("OSConfig.time_unit must be 'seconds', 'samples', or 'auto'.")
    direction_start_s = direction_indices * direction_duration_s
    time_in_direction_s = time_in_repeat_s - direction_start_s
    valid_direction = (
        (direction_indices >= 0)
        & (direction_indices < n_directions)
        & (time_in_direction_s >= 0)
        & (time_in_direction_s < direction_duration_s)
    )
    direction_indices = direction_indices[valid_direction]
    repeat_indices = repeat_indices[valid_direction]
    time_in_direction_s = time_in_direction_s[valid_direction]
    if len(direction_indices) == 0:
        return {
            "raster_by_direction": empty_rasters,
            "spike_counts": np.zeros(n_directions, dtype=int),
            "n_spikes": 0,
        }
    spike_counts = np.bincount(direction_indices, minlength=n_directions)
    raster_by_direction: dict[str, list[np.ndarray]] = {}
    for direction_index, direction_name in enumerate(direction_names):
        raster_by_direction[str(direction_name)] = [
            np.asarray(
                time_in_direction_s[
                    (direction_indices == direction_index)
                    & (repeat_indices == repeat_index)
                ],
                dtype=float,
            )
            for repeat_index in range(cfg.expected_repeats)
        ]
    return {
        "raster_by_direction": raster_by_direction,
        "spike_counts": spike_counts,
        "n_spikes": int(len(time_in_direction_s)),
    }


def circular_vector_direction_stats(
    responses: np.ndarray, directions_deg: np.ndarray
) -> dict[str, float]:
    responses = np.asarray(responses, dtype=float).copy()
    directions_deg = np.asarray(directions_deg, dtype=float)
    responses[~np.isfinite(responses)] = 0
    total_response = np.sum(responses)
    if total_response == 0:
        return {
            "vector_preferred_direction_deg": np.nan,
            "vector_DSI": np.nan,
            "preferred_orientation_deg": np.nan,
            "OSI": np.nan,
        }
    theta = np.deg2rad(directions_deg)
    direction_vector = np.sum(responses * np.exp(1j * theta))
    orientation_vector = np.sum(responses * np.exp(2j * theta))
    return {
        "vector_preferred_direction_deg": np.rad2deg(np.angle(direction_vector)) % 360,
        "vector_DSI": np.abs(direction_vector) / total_response,
        "preferred_orientation_deg": np.rad2deg(np.angle(orientation_vector)) / 2 % 180,
        "OSI": np.abs(orientation_vector) / total_response,
    }


def compute_peak_opposite_dsi(
    responses: np.ndarray, directions_deg: np.ndarray
) -> dict[str, float]:
    responses = np.asarray(responses, dtype=float)
    directions_deg = np.asarray(directions_deg, dtype=float)
    if len(responses) == 0 or np.all(~np.isfinite(responses)):
        return {"peak_opposite_DSI": np.nan, "peak_preferred_direction_deg": np.nan}
    responses_for_argmax = responses.copy()
    responses_for_argmax[~np.isfinite(responses_for_argmax)] = -np.inf
    preferred_index = int(np.argmax(responses_for_argmax))
    preferred_response = responses[preferred_index]
    preferred_direction = directions_deg[preferred_index]
    opposite_target = (preferred_direction + 180) % 360
    differences = np.abs(directions_deg - opposite_target)
    differences = np.minimum(differences, 360 - differences)
    opposite_index = int(np.argmin(differences))
    opposite_response = responses[opposite_index]
    denominator = preferred_response + opposite_response
    dsi = (
        np.nan
        if denominator == 0
        else (preferred_response - opposite_response) / denominator
    )
    return {
        "peak_opposite_DSI": dsi,
        "peak_preferred_direction_deg": preferred_direction,
    }


def compute_os_psths(
    raster_by_direction: dict[str, list[np.ndarray]], cfg: OSConfig
) -> dict[str, Any]:
    direction_duration_s = cfg.frames_per_direction * cfg.seconds_per_frame
    bins = np.arange(0, direction_duration_s + cfg.bin_size_s, cfg.bin_size_s)
    if bins[-1] < direction_duration_s:
        bins = np.append(bins, direction_duration_s)
    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths / 2
    peak_responses = np.zeros(len(cfg.direction_names), dtype=float)
    peak_times = np.full(len(cfg.direction_names), np.nan, dtype=float)
    psth_by_direction: dict[str, dict[str, np.ndarray]] = {}
    for direction_index, direction_name in enumerate(cfg.direction_names):
        direction_name = str(direction_name)
        non_empty = [
            np.asarray(times, dtype=float)
            for times in raster_by_direction[direction_name]
            if len(times) > 0
        ]
        all_spike_times = np.concatenate(non_empty) if non_empty else np.array([])
        if len(all_spike_times) == 0:
            psth = np.zeros(len(bin_centers), dtype=float)
            peak_response = 0.0
            peak_time = np.nan
        else:
            counts, _ = np.histogram(all_spike_times, bins=bins)
            psth = counts / (bin_widths * cfg.expected_repeats)
            peak_bin = int(np.argmax(psth))
            peak_response = float(psth[peak_bin])
            peak_time = float(bin_centers[peak_bin])
        peak_responses[direction_index] = peak_response
        peak_times[direction_index] = peak_time
        psth_by_direction[direction_name] = {"bin_centers": bin_centers, "psth": psth}
    return {
        "peak_response_by_direction": peak_responses,
        "peak_time_by_direction": peak_times,
        "psth_by_direction": psth_by_direction,
    }


def _direction_stats(
    responses: np.ndarray, directions_deg: np.ndarray
) -> dict[str, Any]:
    return {
        "responses": responses,
        **compute_peak_opposite_dsi(responses, directions_deg),
        **circular_vector_direction_stats(responses, directions_deg),
    }


def analyse_os(spikes_df: pd.DataFrame, cfg: OSConfig) -> dict[str, Any]:
    split = split_os_spikes(spikes_df, cfg)
    direction_duration_s = cfg.frames_per_direction * cfg.seconds_per_frame
    total_time_per_direction_s = cfg.expected_repeats * direction_duration_s
    total_spike_response = split["spike_counts"] / total_time_per_direction_s
    peak_data = compute_os_psths(split["raster_by_direction"], cfg)
    return {
        "raster_by_direction": split["raster_by_direction"],
        "spike_counts": split["spike_counts"],
        "n_spikes": split["n_spikes"],
        "psth_by_direction": peak_data["psth_by_direction"],
        "peak_time_by_direction": peak_data["peak_time_by_direction"],
        "total_spikes": _direction_stats(total_spike_response, cfg.directions_deg),
        "peak_response": _direction_stats(
            peak_data["peak_response_by_direction"], cfg.directions_deg
        ),
    }


def get_os_metric(os_data: dict[str, Any], cfg: OSConfig) -> dict[str, Any]:
    if cfg.response_metric not in {"total_spikes", "peak_response"}:
        raise ValueError("response_metric must be 'peak_response' or 'total_spikes'.")
    metric = os_data[cfg.response_metric]
    return {
        **metric,
        "label": "Peak response"
        if cfg.response_metric == "peak_response"
        else "Total spikes",
    }


# %% Moving-bar drawing
def _close_polar_curve(
    directions_deg: np.ndarray, responses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sort_idx = np.argsort(np.asarray(directions_deg) % 360)
    theta_sorted = np.deg2rad(np.asarray(directions_deg)[sort_idx])
    response_sorted = np.asarray(responses, dtype=float)[sort_idx]
    return (
        np.concatenate([theta_sorted, theta_sorted[:1]]),
        np.concatenate([response_sorted, response_sorted[:1]]),
    )


def _add_motion_arrows(ax, cfg: OSConfig):
    for motion_deg in np.arange(0, 360, 45):
        theta = np.deg2rad(motion_deg)
        x = 0.5 + cfg.polar_arrow_radius * np.sin(theta)
        y = 0.5 + cfg.polar_arrow_radius * np.cos(theta)
        dx = cfg.polar_arrow_length * np.sin(theta)
        dy = cfg.polar_arrow_length * np.cos(theta)
        ax.annotate(
            "",
            xy=(x + dx / 2, y + dy / 2),
            xytext=(x - dx / 2, y - dy / 2),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="->",
                linewidth=cfg.polar_arrow_linewidth,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=cfg.polar_arrow_mutation_scale,
            ),
            annotation_clip=False,
        )


def _draw_polar_tuning(ax, os_data: dict[str, Any], cfg: OSConfig):
    metric = get_os_metric(os_data, cfg)
    theta, responses = _close_polar_curve(cfg.directions_deg, metric["responses"])
    (line,) = ax.plot(
        theta, responses, linewidth=2, marker="o", markersize=4, linestyle="-"
    )
    ax.fill(theta, responses, alpha=0.1)
    max_response = (
        np.nanmax(metric["responses"]) if len(metric["responses"]) else np.nan
    )
    if np.isfinite(max_response) and max_response > 0:
        preferred_direction = metric["peak_preferred_direction_deg"]
        preferred_orientation = metric["preferred_orientation_deg"]
        if cfg.show_peak_direction_line and np.isfinite(preferred_direction):
            angle = np.deg2rad(preferred_direction)
            ax.plot(
                [angle, angle],
                [0, max_response],
                linewidth=1.5,
                linestyle=":",
                color=line.get_color(),
            )
        if cfg.show_orientation_axis and np.isfinite(preferred_orientation):
            for angle_deg in [preferred_orientation, preferred_orientation + 180]:
                angle = np.deg2rad(angle_deg % 360)
                ax.plot(
                    [angle, angle],
                    [0, max_response],
                    linewidth=1.2,
                    linestyle="--",
                    color=line.get_color(),
                    alpha=0.7,
                )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(-22.5)
    ticks = np.arange(0, 360, 45)
    ax.set_xticks(np.deg2rad(ticks))
    ax.set_xticklabels([""] * len(ticks))
    ax.tick_params(labelsize=7)
    if cfg.show_motion_direction_arrows:
        _add_motion_arrows(ax, cfg)


def _draw_small_raster(
    ax,
    repeat_spike_times: list[np.ndarray],
    direction_duration_s: float,
    cfg: OSConfig,
    show_xlabel: bool,
):
    ax.set_facecolor(cfg.raster_facecolor)
    ax.eventplot(
        repeat_spike_times,
        lineoffsets=np.arange(cfg.expected_repeats),
        linelengths=0.65,
        linewidths=0.6,
        colors="black",
    )
    ax.set_xlim(0, direction_duration_s)
    ax.set_ylim(-0.5, cfg.expected_repeats - 0.5)
    ax.invert_yaxis()
    ticks = np.arange(0, direction_duration_s + 0.01, 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.0f}" for tick in ticks], fontsize=5)
    ax.tick_params(axis="x", length=2, width=0.6, pad=1)
    if show_xlabel:
        ax.set_xlabel("s", fontsize=5, labelpad=1)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.25")


def _draw_small_psth(
    ax,
    psth_data: dict[str, np.ndarray],
    direction_duration_s: float,
    common_y_max: float,
    cfg: OSConfig,
    show_y_scale: bool,
):
    bin_centers = np.asarray(psth_data["bin_centers"], dtype=float)
    firing_rate = np.asarray(psth_data["psth"], dtype=float)
    ax.set_facecolor(cfg.psth_facecolor)
    ax.plot(bin_centers, firing_rate, linewidth=1.0, color="black")
    ax.fill_between(bin_centers, 0, firing_rate, alpha=0.15, color="black")
    ax.set_xlim(0, direction_duration_s)
    ax.set_ylim(0, common_y_max)
    ax.set_xticks([])
    if show_y_scale:
        ax.set_yticks([0, common_y_max])
        ax.set_yticklabels(["0", f"{common_y_max:.0f}"], fontsize=5)
        ax.set_ylabel("Hz", fontsize=5, labelpad=0)
        ax.tick_params(axis="y", length=2, width=0.6, pad=1)
    else:
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def _common_psth_y_max(os_data: dict[str, Any], cfg: OSConfig) -> float:
    values = []
    for direction_name in cfg.direction_names:
        psth = np.asarray(
            os_data["psth_by_direction"][str(direction_name)]["psth"], dtype=float
        )
        finite = psth[np.isfinite(psth)]
        if finite.size:
            values.append(finite)
    if not values:
        return 1.0
    maximum = float(np.max(np.concatenate(values)))
    return 1.0 if maximum <= 0 else maximum * 1.08


def _square_design_box(container) -> tuple[float, float, float, float]:
    """Return a centred, physically square region inside a Figure/SubFigure.

    The values are normalised coordinates relative to ``container``:
    ``(left, bottom, width, height)``.

    A square internal design space keeps the moving-bar arrangement identical
    in the standalone square figure and in the wider combined-figure panel.
    """

    root_figure = getattr(container, "figure", container)
    root_figure.canvas.draw()

    container_width_px = float(container.bbox.width)
    container_height_px = float(container.bbox.height)

    if container_width_px <= 0 or container_height_px <= 0:
        raise ValueError("The plotting container has zero width or height.")

    if container_width_px >= container_height_px:
        square_width = container_height_px / container_width_px
        square_height = 1.0
        square_left = (1.0 - square_width) / 2
        square_bottom = 0.0
    else:
        square_width = 1.0
        square_height = container_width_px / container_height_px
        square_left = 0.0
        square_bottom = (1.0 - square_height) / 2

    return square_left, square_bottom, square_width, square_height


def _map_box_from_square(
    square_box: tuple[float, float, float, float],
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> list[float]:
    """Map a rectangle from square-design coordinates into the container."""

    square_left, square_bottom, square_width, square_height = square_box

    return [
        square_left + left * square_width,
        square_bottom + bottom * square_height,
        width * square_width,
        height * square_height,
    ]


def _os_layout(container, cfg: OSConfig):
    """Calculate the polar, raster and PSTH geometry.

    This follows the layout of the original moving-bar summary script:

    * The polar axis occupies ``[0.30, 0.27, 0.40, 0.40]`` in a square canvas.
    * Cardinal tiles are centred on the horizontal/vertical arrow axes.
    * Diagonal tiles are displaced by ``gap / sqrt(2)`` in x and y, so the
      arrow trajectory points towards the nearest tile corner.
    * Raster and PSTH sizes remain physically square/consistent even when the
      surrounding combined-figure panel is rectangular.
    """

    square_box = _square_design_box(container)
    polar_local_box = (0.30, 0.27, 0.40, 0.40)
    polar_box = _map_box_from_square(square_box, *polar_local_box)

    raster_width = cfg.raster_box_width
    raster_height = cfg.raster_box_height

    gap = cfg.raster_gap_from_arrow

    total_tile_height = (
        raster_height + cfg.psth_gap + cfg.psth_box_height
        if cfg.show_psth
        else raster_height
    )

    direction_to_angle = {
        "up": 0,
        "up-right": 45,
        "right": 90,
        "down-right": 135,
        "down": 180,
        "down-left": 225,
        "left": 270,
        "up-left": 315,
    }

    polar_left, polar_bottom, polar_width, polar_height = polar_local_box

    # Arrow centres in the same square coordinate system as the tile layout.
    arrow_xy = {}
    for direction_name, motion_deg in direction_to_angle.items():
        theta = np.deg2rad(motion_deg)
        x_axes = 0.5 + cfg.polar_arrow_radius * np.sin(theta)
        y_axes = 0.5 + cfg.polar_arrow_radius * np.cos(theta)
        arrow_xy[direction_name] = (
            polar_left + x_axes * polar_width,
            polar_bottom + y_axes * polar_height,
        )

    positions_local: dict[str, tuple[float, float]] = {}

    # Cardinal directions: centre the nearest tile edge on the arrow axis.
    x, y = arrow_xy["up"]
    positions_local["up"] = (x - raster_width / 2, y + gap)

    x, y = arrow_xy["down"]
    positions_local["down"] = (
        x - raster_width / 2,
        y - gap - total_tile_height,
    )

    x, y = arrow_xy["right"]
    positions_local["right"] = (x + gap, y - total_tile_height / 2)

    x, y = arrow_xy["left"]
    positions_local["left"] = (
        x - gap - raster_width,
        y - total_tile_height / 2,
    )

    # Diagonal directions: aim the arrow trajectory at the nearest tile corner.
    diagonal_gap = gap / np.sqrt(2)

    x, y = arrow_xy["up-right"]
    positions_local["up-right"] = (x + diagonal_gap, y + diagonal_gap)

    x, y = arrow_xy["up-left"]
    positions_local["up-left"] = (
        x - diagonal_gap - raster_width,
        y + diagonal_gap,
    )

    x, y = arrow_xy["down-right"]
    positions_local["down-right"] = (
        x + diagonal_gap,
        y - diagonal_gap - total_tile_height,
    )

    x, y = arrow_xy["down-left"]
    positions_local["down-left"] = (
        x - diagonal_gap - raster_width,
        y - diagonal_gap - total_tile_height,
    )

    positions = {}
    for direction_name, (left, bottom) in positions_local.items():
        mapped = _map_box_from_square(
            square_box,
            left,
            bottom,
            raster_width,
            raster_height,
        )
        positions[direction_name] = (mapped[0], mapped[1])

    dimensions = {
        "raster_width": raster_width * square_box[2],
        "raster_height": raster_height * square_box[3],
        "psth_height": cfg.psth_box_height * square_box[3],
        "psth_gap": cfg.psth_gap * square_box[3],
    }

    return polar_box, positions, dimensions


def draw_os(
    container,
    os_data: dict[str, Any],
    cell_id: int,
    quality: float,
    tilt: float,
    cfg: OSConfig,
):
    direction_duration_s = cfg.frames_per_direction * cfg.seconds_per_frame
    metric = get_os_metric(os_data, cfg)
    common_y_max = _common_psth_y_max(os_data, cfg)

    polar_box, positions, dimensions = _os_layout(container, cfg)

    ax_polar = container.add_axes(polar_box, projection="polar")
    _draw_polar_tuning(ax_polar, os_data, cfg)

    raster_width = dimensions["raster_width"]
    raster_height = dimensions["raster_height"]
    psth_height = dimensions["psth_height"]
    psth_gap = dimensions["psth_gap"]

    for direction_name, (left, bottom) in positions.items():
        ax_raster = container.add_axes([left, bottom, raster_width, raster_height])
        _draw_small_raster(
            ax_raster,
            os_data["raster_by_direction"][direction_name],
            direction_duration_s,
            cfg,
            show_xlabel=direction_name == "left",
        )

        if cfg.show_psth:
            ax_psth = container.add_axes(
                [
                    left,
                    bottom + raster_height + psth_gap,
                    raster_width,
                    psth_height,
                ]
            )
            _draw_small_psth(
                ax_psth,
                os_data["psth_by_direction"][direction_name],
                direction_duration_s,
                common_y_max,
                cfg,
                show_y_scale=(cfg.show_psth_scale and direction_name == "left"),
            )

    container.suptitle(
        f"{cfg.title}\n"
        f"Cell {cell_id} | {metric['label']} | "
        f"Q={quality:.1f}, tilt={tilt:.2f}, "
        f"DSI={metric['peak_opposite_DSI']:.2f}, "
        f"OSI={metric['OSI']:.2f}",
        fontsize=12,
        y=0.98,
    )


# %% Plotting session and caches
class CellPlotSession:
    """Hold reusable data, caches and plotting methods for one recording.

    Parameters
    ----------
    cfg:
        All analysis, plotting and path settings.
    dataset:
        An already-loaded NetCDF dataset. Passing this object is the recommended
        workflow because configuration changes then do not trigger another disk
        read. ``None`` remains supported as a fallback and loads
        ``cfg.noise_data_path`` internally.
    """

    def __init__(self, cfg: PlotConfig, dataset: xr.Dataset | None = None):
        self.cfg = cfg

        # Track whether this class opened the dataset itself. An externally loaded
        # dataset belongs to the caller and is therefore not closed automatically.
        self._owns_dataset = dataset is None

        if dataset is None:
            print("No preloaded dataset supplied; loading NetCDF from disk...")
            self.dataset = xr.load_dataset(cfg.noise_data_path)
        else:
            print("Using the already-loaded NetCDF dataset.")
            self.dataset = dataset

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        for subfolder in [
            "rf",
            "sta",
            "scf",
            "csteps",
            "chirp",
            "os",
            "combined",
        ]:
            (cfg.output_dir / subfolder).mkdir(parents=True, exist_ok=True)

        print("Loading Polarspike recording overview...")
        self.recording = Overview.Recording.load(str(cfg.overview_path))

        # These objects depend on configuration but not on rereading the NetCDF.
        self.alignment: AlignmentState
        self.rf_channels: list[Any | None]
        self.reference_channel: Any | None
        self.available_cells: list[int]

        # Spike tables are cached by (cell, stimulus). Processed analyses are
        # cached by cell to make repeated plotting substantially faster.
        self._spike_cache: dict[tuple[int, int], pd.DataFrame] = {}
        self._analysis_cache: dict[int, dict[str, Any]] = {}

        self.refresh_from_config(clear_spike_cache=False)

        print(
            f"Session ready. {len(self.available_cells)} selected cells available:\n"
            f"{self.available_cells}"
        )

    def refresh_from_config(self, clear_spike_cache: bool = False) -> None:
        """Rebuild configuration-dependent state without reloading the dataset.

        Call this after changing settings that affect alignment, RF channels,
        filtering or stimulus analysis. The processed-cell cache is always
        cleared because its contents may no longer match the configuration.
        Spike data can normally remain cached because it is indexed by stimulus
        number; set ``clear_spike_cache=True`` after changing recording-level
        assumptions or when a completely clean refresh is preferred.
        """

        print("Refreshing alignment and channel information from configuration...")
        self.alignment = load_alignment_state(self.cfg)
        self.rf_channels = get_rf_channels(
            self.dataset,
            self.cfg.rf.rf_variable,
        )
        self.reference_channel = resolve_reference_channel(
            dataset=self.dataset,
            rf_variable=self.cfg.rf.rf_variable,
            requested_channel=self.cfg.rf.channel,
        )
        self.available_cells = select_cells(self.dataset, self.cfg)

        self._analysis_cache.clear()
        if clear_spike_cache:
            self._spike_cache.clear()

        print(f"RF channels to plot: {self.rf_channels}")
        print(f"Reference channel for STA/quality/tilt: {self.reference_channel!r}")

    def _check_cell(self, cell_id: int) -> int:
        """Validate a requested cell ID against the loaded dataset."""

        cell_id = int(cell_id)
        all_cells = {int(x) for x in self.dataset["cell_index"].values}
        if cell_id not in all_cells:
            raise ValueError(f"Cell {cell_id} is not present in the dataset.")
        return cell_id

    def get_spikes(self, cell_id: int, stimulus_index: int) -> pd.DataFrame:
        """Return one cell/stimulus spike table, loading it only once."""

        key = (int(cell_id), int(stimulus_index))
        if key not in self._spike_cache:
            condition = {
                "stimulus_index": [int(stimulus_index)],
                "cell_index": [int(cell_id)],
            }
            self._spike_cache[key] = _to_pandas(
                self.recording.get_spikes_triggered([condition])
            )
        return self._spike_cache[key]

    def analyse_cell(self, cell_id: int, refresh: bool = False) -> dict[str, Any]:
        """Run every analysis required by the summary figure for one cell."""

        cell_id = self._check_cell(cell_id)

        if not refresh and cell_id in self._analysis_cache:
            return self._analysis_cache[cell_id]

        # The reference-channel metrics are used for the STA and non-RF panels.
        quality, tilt = get_quality_and_tilt(self.dataset, cell_id, self.cfg)

        # Prepare one independent RF result for each RF channel in the variable.
        rf_panels = []
        for channel in self.rf_channels:
            channel_quality, channel_tilt = get_quality_and_tilt_for_channel(
                dataset=self.dataset,
                cell_id=cell_id,
                cfg=self.cfg,
                channel=channel,
            )
            rf_panels.append(
                {
                    "channel": channel,
                    "quality": channel_quality,
                    "tilt": channel_tilt,
                    "rf": analyse_rf(
                        dataset=self.dataset,
                        cell_id=cell_id,
                        cfg=self.cfg,
                        alignment=self.alignment,
                        channel=channel,
                    ),
                    # The STA is analysed with the same channel label as the RF,
                    # so every plotted RF has a directly corresponding STA.
                    "sta": analyse_sta(
                        dataset=self.dataset,
                        cell_id=cell_id,
                        cfg=self.cfg,
                        channel=channel,
                    ),
                }
            )

        scf_spikes = self.get_spikes(cell_id, self.cfg.scf.stimulus_index)
        csteps_spikes = self.get_spikes(cell_id, self.cfg.csteps.stimulus_index)
        chirp_spikes = self.get_spikes(cell_id, self.cfg.chirp.stimulus_index)
        os_spikes = self.get_spikes(cell_id, self.cfg.os.stimulus_index)

        # Use the configured reference channel for the old single-RF/single-STA
        # shortcuts. This keeps existing interactive code working while the new
        # ``rf_panels`` structure contains every channel-specific RF/STA pair.
        reference_panel = next(
            (
                panel
                for panel in rf_panels
                if panel["channel"] == self.reference_channel
            ),
            rf_panels[0],
        )

        result = {
            "cell_id": cell_id,
            "quality": quality,
            "tilt": tilt,
            "reference_channel": self.reference_channel,
            "rf_channels": list(self.rf_channels),
            "rf_panels": rf_panels,
            "sta_panels": [panel["sta"] for panel in rf_panels],
            # Backwards-compatible shortcuts point to the reference channel.
            "rf": reference_panel["rf"],
            "sta": reference_panel["sta"],
            "scf": analyse_scf(scf_spikes, self.cfg.scf),
            "csteps": analyse_csteps(csteps_spikes, self.cfg.csteps),
            "chirp": analyse_chirp(chirp_spikes, self.cfg.chirp),
            "os": analyse_os(os_spikes, self.cfg.os),
        }

        counts = dict(
            zip(self.cfg.os.direction_names, result["os"]["spike_counts"].tolist())
        )
        print(
            f"Cell {cell_id} OS spikes retained: {result['os']['n_spikes']} / "
            f"{len(os_spikes)}; counts by direction: {counts}"
        )

        self._analysis_cache[cell_id] = result
        return result

    def clear_cache(self, cell_id: int | None = None) -> None:
        """Clear all cached results, or only entries belonging to one cell."""

        if cell_id is None:
            self._spike_cache.clear()
            self._analysis_cache.clear()
        else:
            cell_id = int(cell_id)
            self._analysis_cache.pop(cell_id, None)
            for key in list(self._spike_cache):
                if key[0] == cell_id:
                    self._spike_cache.pop(key)

    def _save(self, fig, plot_type: str, cell_id: int) -> Path:
        """Save a figure into the matching output subfolder."""

        path = (
            self.cfg.output_dir
            / plot_type
            / f"{self.cfg.recording_name}_cell_{cell_id}_{plot_type}.{self.cfg.image_format}"
        )
        fig.savefig(path, dpi=self.cfg.save_dpi, bbox_inches="tight")
        print(f"Saved: {path}")
        return path

    def _finish(self, fig, plot_type: str, cell_id: int, save, show):
        """Apply the common save/show behaviour used by standalone plots."""

        if save is None:
            save = self.cfg.save
        if save:
            self._save(fig, plot_type, cell_id)
        if show:
            plt.show()
        return fig

    @staticmethod
    def _rf_grid_shape(n_channels: int) -> tuple[int, int]:
        """Return a compact grid shape for one or more RF panels."""

        if n_channels < 1:
            raise ValueError("At least one RF channel is required.")
        n_cols = 1 if n_channels == 1 else 2
        n_rows = int(np.ceil(n_channels / n_cols))
        return n_rows, n_cols

    def plot_rf(self, cell_id: int, save=None, show=True):
        """Plot every RF channel as a labelled standalone RF figure."""

        result = self.analyse_cell(cell_id)
        n_channels = len(result["rf_panels"])
        n_rows, n_cols = self._rf_grid_shape(n_channels)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(7.2 * n_cols, 7.0 * n_rows),
            dpi=self.cfg.figure_dpi,
            squeeze=False,
        )

        flat_axes = axes.ravel()
        for ax, panel in zip(flat_axes, result["rf_panels"]):
            im = draw_rf(
                ax=ax,
                rf=panel["rf"],
                cell_id=cell_id,
                quality=panel["quality"],
                tilt=panel["tilt"],
                cfg=self.cfg,
                alignment=self.alignment,
            )
            fig.colorbar(
                im,
                ax=ax,
                shrink=0.82,
                label=self.cfg.rf.rf_variable,
            )

        # Hide any unused grid cell when an odd number of channels exceeds one.
        for ax in flat_axes[n_channels:]:
            ax.set_visible(False)

        fig.tight_layout()
        return self._finish(fig, "rf", cell_id, save, show)

    def plot_sta(self, cell_id: int, save=None, show=True):
        """Plot one labelled STA subplot for every plotted RF channel."""

        result = self.analyse_cell(cell_id)
        n_channels = len(result["rf_panels"])
        n_rows, n_cols = self._rf_grid_shape(n_channels)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6.2 * n_cols, 4.2 * n_rows),
            dpi=self.cfg.figure_dpi,
            squeeze=False,
        )

        flat_axes = axes.ravel()
        for ax, panel in zip(flat_axes, result["rf_panels"]):
            draw_sta(
                ax=ax,
                sta=panel["sta"],
                cell_id=cell_id,
                quality=panel["quality"],
                tilt=panel["tilt"],
            )

        for ax in flat_axes[n_channels:]:
            ax.set_visible(False)

        fig.tight_layout()
        return self._finish(fig, "sta", cell_id, save, show)

    def plot_scf(self, cell_id: int, save=None, show=True):
        result = self.analyse_cell(cell_id)
        fig = plt.figure(figsize=(11.0, 6.2), dpi=self.cfg.figure_dpi)
        draw_scf(
            fig,
            result["scf"],
            cell_id,
            result["quality"],
            result["tilt"],
            self.cfg.scf,
        )
        fig.tight_layout()
        return self._finish(fig, "scf", cell_id, save, show)

    def plot_csteps(self, cell_id: int, save=None, show=True):
        result = self.analyse_cell(cell_id)
        fig = plt.figure(figsize=(11.0, 6.2), dpi=self.cfg.figure_dpi)
        draw_csteps(
            container=fig,
            csteps=result["csteps"],
            cell_id=cell_id,
            quality=result["quality"],
            tilt=result["tilt"],
            cfg=self.cfg.csteps,
        )
        fig.tight_layout()
        return self._finish(fig, "csteps", cell_id, save, show)

    def plot_chirp(self, cell_id: int, save=None, show=True):
        result = self.analyse_cell(cell_id)
        fig = plt.figure(figsize=(11.0, 5.0), dpi=self.cfg.figure_dpi)
        draw_chirp(
            container=fig,
            chirp=result["chirp"],
            cell_id=cell_id,
            quality=result["quality"],
            tilt=result["tilt"],
            cfg=self.cfg.chirp,
        )
        fig.tight_layout()
        return self._finish(fig, "chirp", cell_id, save, show)

    def plot_os(self, cell_id: int, save=None, show=True):
        result = self.analyse_cell(cell_id)
        fig = plt.figure(figsize=(10, 10), dpi=self.cfg.figure_dpi)
        draw_os(
            fig,
            result["os"],
            cell_id,
            result["quality"],
            result["tilt"],
            self.cfg.os,
        )
        return self._finish(fig, "os", cell_id, save, show)

    def plot_cell(
        self,
        cell_id: int,
        save_individual: bool = False,
        save_combined: bool = True,
        show_combined: bool = True,
        combined_figsize: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Create all standalone plots optionally, plus the combined summary.

        The left column is dynamic. Each RF channel occupies one cell in a
        one- or two-column channel grid. Inside each cell, the RF is placed on
        top and its matching channel-specific STA is placed immediately below.
        The middle and right columns preserve the original SCF/csteps/chirp and
        moving-bar layout. With two RF channels the overall figure is widened so
        both RF/STA pairs remain readable rather than being compressed.
        """

        result = self.analyse_cell(cell_id)

        individual = {}
        if save_individual:
            individual = {
                "rf": self.plot_rf(cell_id, save=True, show=False),
                "sta": self.plot_sta(cell_id, save=True, show=False),
                "scf": self.plot_scf(cell_id, save=True, show=False),
                "csteps": self.plot_csteps(cell_id, save=True, show=False),
                "chirp": self.plot_chirp(cell_id, save=True, show=False),
                "os": self.plot_os(cell_id, save=True, show=False),
            }

        n_rf_channels = len(result["rf_panels"])
        rf_rows, rf_cols = self._rf_grid_shape(n_rf_channels)

        if combined_figsize is None:
            # Preserve the original size for one RF. Add width only when a
            # second RF column is required.
            combined_figsize = (25 if rf_cols == 1 else 31, 16.5 + 7 * (rf_rows - 1))

        combined = plt.figure(
            figsize=combined_figsize,
            dpi=self.cfg.figure_dpi,
        )

        left_width = 0.9 if rf_cols == 1 else 1.65
        main_gs = combined.add_gridspec(
            1,
            3,
            width_ratios=[left_width, 1.35, 1.15],
            wspace=0.04,
            # Leave a clear margin above all three main columns for the
            # overall recording/cell title.
            top=0.92,
            # Small lower margin prevents bottom axis labels being clipped.
            bottom=0.04,
            # Retain a small margin at the outer sides of the figure.
            left=0.025,
            right=0.985,
        )

        left_subfig = combined.add_subfigure(main_gs[0, 0])
        middle_subfig = combined.add_subfigure(main_gs[0, 1])
        os_subfig = combined.add_subfigure(main_gs[0, 2])

        # The outer grid arranges channels. Each occupied grid cell contains a
        # small two-row subfigure: RF above and the matching STA below.
        channel_gs = left_subfig.add_gridspec(
            rf_rows,
            rf_cols,
            hspace=0.12,
            wspace=0.10,
        )

        for panel_index, panel in enumerate(result["rf_panels"]):
            row = panel_index // rf_cols
            col = panel_index % rf_cols

            channel_subfig = left_subfig.add_subfigure(channel_gs[row, col])
            pair_gs = channel_subfig.add_gridspec(
                2,
                1,
                height_ratios=[1.0, 0.58],
                hspace=0.20,
            )

            rf_subfig = channel_subfig.add_subfigure(pair_gs[0])
            ax_rf = rf_subfig.subplots()

            im = draw_rf(
                ax=ax_rf,
                rf=panel["rf"],
                cell_id=cell_id,
                quality=panel["quality"],
                tilt=panel["tilt"],
                cfg=self.cfg,
                alignment=self.alignment,
            )
            rf_subfig.colorbar(
                im,
                ax=ax_rf,
                shrink=0.80,
                label=self.cfg.rf.rf_variable,
            )

            sta_subfig = channel_subfig.add_subfigure(pair_gs[1])
            ax_sta = sta_subfig.subplots()
            draw_sta(
                ax=ax_sta,
                sta=panel["sta"],
                cell_id=cell_id,
                quality=panel["quality"],
                tilt=panel["tilt"],
            )

        # Empty cells are possible only for an odd number greater than one.
        for empty_index in range(n_rf_channels, rf_rows * rf_cols):
            row = empty_index // rf_cols
            col = empty_index % rf_cols
            empty_ax = left_subfig.add_subplot(channel_gs[row, col])
            empty_ax.axis("off")

        # The middle column keeps the three original repeated-stimulus panels.
        middle_gs = middle_subfig.add_gridspec(
            3,
            1,
            height_ratios=[1, 1, 1],
            hspace=0.22,
        )
        scf_subfig = middle_subfig.add_subfigure(middle_gs[0])
        csteps_subfig = middle_subfig.add_subfigure(middle_gs[1])
        chirp_subfig = middle_subfig.add_subfigure(middle_gs[2])

        draw_scf(
            scf_subfig,
            result["scf"],
            cell_id,
            result["quality"],
            result["tilt"],
            self.cfg.scf,
        )
        draw_csteps(
            csteps_subfig,
            result["csteps"],
            cell_id,
            result["quality"],
            result["tilt"],
            self.cfg.csteps,
        )
        draw_chirp(
            chirp_subfig,
            result["chirp"],
            cell_id,
            result["quality"],
            result["tilt"],
            self.cfg.chirp,
        )

        # The moving-bar layout still occupies the full right column.
        draw_os(
            os_subfig,
            result["os"],
            cell_id,
            result["quality"],
            result["tilt"],
            self.cfg.os,
        )

        left_subfig.suptitle(
            f"{self.cfg.recording_name} | Cell {cell_id}",
            fontsize=16,
            fontweight="bold",
            y=0.99,
        )

        if save_combined:
            self._save(combined, "combined", cell_id)
        if show_combined:
            plt.show()

        return {
            **individual,
            "combined": combined,
            "data": result,
            "rf_panels": result["rf_panels"],
            "sta_panels": result["sta_panels"],
        }

    def quality_table(self) -> pd.DataFrame:
        """Rank cells by RF quality in the configured reference channel."""

        da = self.dataset[self.cfg.rf.quality_variable]
        if "channel" in da.dims or "channel" in da.coords:
            da = da.sel(channel=self.reference_channel)

        table = (
            da.to_dataframe()
            .reset_index()[["cell_index", self.cfg.rf.quality_variable]]
            .drop_duplicates(subset=["cell_index"])
            .rename(columns={self.cfg.rf.quality_variable: "rf_quality"})
            .dropna(subset=["rf_quality"])
            .sort_values("rf_quality", ascending=False)
            .reset_index(drop=True)
        )
        table["rank"] = np.arange(1, len(table) + 1)
        return table

    def close(self, close_dataset: bool | None = None) -> None:
        """Close figures and optionally close the NetCDF dataset.

        By default, only a dataset loaded internally by this session is closed.
        Pass ``close_dataset=True`` when the externally loaded dataset is no
        longer needed either.
        """

        if close_dataset is None:
            close_dataset = self._owns_dataset
        if close_dataset:
            self.dataset.close()
        plt.close("all")
        print("Plotting session closed.")


# %% 1. Load the large NetCDF dataset first
# Change only this path when moving to another recording. This is deliberately
# separate from all later configuration. Rerun this cell only when the NetCDF
# path itself changes or when you explicitly want a fresh disk read.

NOISE_DATA_PATH = Path(
    r"F:\Laura\zebrafish_22_07_2026\Phase_00\noise_analysis\noise_data.nc"
)

# During interactive work, close an older dataset before replacing it to avoid
# keeping unnecessary file handles open. The NameError guard also allows this
# cell to run cleanly the first time.
try:
    dataset.close()
except (NameError, AttributeError):
    pass

dataset = xr.load_dataset(NOISE_DATA_PATH)
describe_loaded_dataset(dataset)


# %% 2. Set paths and all analysis/plotting parameters
# The NetCDF is already open at this point. You can rerun this cell repeatedly
# while adjusting channels, stimulus indices, thresholds, colours, alpha values,
# figure sizes or display settings without reloading the dataset.

config = PlotConfig(
    recording_name="zebrafish_22_07_2026_phase_00",
    overview_path=Path(r"F:\Laura\zebrafish_22_07_2026\Phase_00\overview"),
    noise_data_path=NOISE_DATA_PATH,
    cone_image_path=Path(
        r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish\Experiments\Imaging\20260722_dragonfly_microscope\cones(MAX_FLIPPED)_2026_07_22.jpg"
    ),
    alignment_cache_path=Path(
        r"F:\Laura\zebrafish_22_07_2026\alignment_videos\zebrafish_22_07_2026_phase_00_alignment_cache.npz"
    ),
    output_dir=Path(r"F:\Laura\zebrafish_22_07_2026\Phase_00\separate_cell_plots"),
    rf=RFConfig(
        # This is the reference channel for STA, quality, tilt and ranking.
        # It does not restrict RF plotting: every channel in rf_variable is shown.
        # Set to None to use the first RF channel printed by the session.
        channel="32px_15Hz_20mins_shuffle_x4",
        rf_variable="cm_most_important",
        transform_tilt_reciprocal=True,
        sta_baseline=None,
        show_peak=False,
    ),
    scf=SCFConfig(
        stimulus_index=9,
        duration_s=32,
        expected_repeats=5,
        bin_size_s=0.05,
        title="SCF",
    ),
    csteps=CStepsConfig(
        stimulus_index=8,
        duration_s=40.0,
        expected_repeats=3,
        bin_size_s=0.05,
        step_span_s=2.0,
        contrasts_percent=(
            100,
            -100,
            90,
            -90,
            80,
            -80,
            70,
            -70,
            60,
            -60,
            50,
            -50,
            40,
            -40,
            30,
            -30,
            20,
            -20,
            10,
            -10,
        ),
        title="Contrast steps",
    ),
    chirp=ChirpConfig(
        stimulus_index=17,
        expected_repeats=3,
        bin_size_s=0.05,
        plot_start_s=0.0,
        plot_end_s=33.0,
        chirp_offset_s=3.0,
        chirp_duration_s=30.0,
        time_col="times_triggered",
        repeat_col="repeat",
        title="Chirp",
    ),
    os=OSConfig(
        stimulus_index=3,
        frames_per_direction=300,
        seconds_per_frame=1 / 60,
        expected_repeats=10,
        time_unit="seconds",
        bin_size_s=0.10,
        response_metric="peak_response",
        title="Moving bar",
    ),
    selected_cells=None,
    min_rf_quality=None,
    max_cells=None,
    save=False,
    show=True,
)


# %% 3. Create or refresh the plotting session without reloading NetCDF
# The dataset object created in step 1 is passed directly into the session.
# Rerunning this cell reloads the smaller recording overview and alignment data,
# but it reuses the already-open NetCDF object.

try:
    session.close(close_dataset=False)
except (NameError, AttributeError):
    pass

session = CellPlotSession(
    cfg=config,
    dataset=dataset,
)

# Alternative after changing config in place:
# session.cfg = config
# session.refresh_from_config(clear_spike_cache=False)


# %% 4. Rank cells by RF quality
# Ranking uses RFConfig.channel, because quality ranking must remain a single
# well-defined quantity even when several RF channels are displayed.

quality_table = session.quality_table()
print(quality_table.head(20))

# Select a cell from the quality-ranked table. Rank is one-based for convenience.
rank = 10
cell_id = int(quality_table.iloc[rank - 1]["cell_index"])


# %% 5. Plot one cell
# Adjust plotting parameters above, call session.refresh_from_config(), and then
# rerun this cell. The NetCDF file remains loaded throughout.

plt.close("all")
figures = session.plot_cell(
    cell_id=260,
    save_individual=False,
    save_combined=False,
    show_combined=True,
)
