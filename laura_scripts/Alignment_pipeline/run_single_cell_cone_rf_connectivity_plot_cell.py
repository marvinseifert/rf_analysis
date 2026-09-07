# %% ============================================================================
# SINGLE-CELL RF ↔ INTERPOLATED CONE-MOSAIC CONNECTIVITY
# LOAD ONCE, THEN USE plot_cell(cell_index)
# ============================================================================
#
# Workflow
# --------
# 1. Run CELL 1 once to set recording parameters.
# 2. Run CELL 2 once to load the NetCDF, cone image and alignment geometry.
# 3. Run CELL 3 once to define plot_cell().
# 4. Thereafter only run CELL 4, e.g.:
#
#       plot_cell(116)
#       plot_cell(235)
#       plot_cell(284)
#
# The dataset, cone image and alignment cache are NOT reloaded between cells.

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from laura_scripts.Alignment_pipeline.cone_rf_connectivity_functions import (
    build_alignment_geometry_matching_rf_alignment,
    compute_cone_connectivity_weights,
    create_connectivity_figure,
    create_detection_qc_figure,
    extract_single_cell_rf_matching_rf_alignment,
    interpolate_cone_mosaic_on_aligned_crop,
    load_image,
    sample_rf_on_aligned_crop,
    save_connectivity_outputs,
    warp_cone_crop_to_projected_grid,
)


# %% ============================================================================
# CELL 1 — RECORDING PARAMETERS — RUN ONCE PER RECORDING
# ============================================================================

# -----------------------------------------------------------------------------
# Recording-specific input paths
# -----------------------------------------------------------------------------
recording_name = "zebrafish_15_05_2026_phase_00"

path_to_data = Path(
    r"F:\Laura\zebrafish_15_05_2026\Phase_00\noise_analysis\noise_data.nc"
)

# Exact Image 1 cone image used in the alignment pipeline.
image_1_cones_path = Path(
    r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish\Experiments\Imaging\20260515_dragonfly_microscope\15_05_2026_cones.jpg"
)

# .npz cache produced by the alignment pipeline.
alignment_cache_path = Path(
    rf"F:\Laura\zebrafish_15_05_2026\alignment_videos\{recording_name}_alignment_cache.npz"
)

output_dir = path_to_data.parent / "single_cell_cone_connectivity"


# -----------------------------------------------------------------------------
# RF / channel settings
# -----------------------------------------------------------------------------
channel_to_plot = "12px_20Hz_25mins_shuffle_white"

# Choose:
#   "cm_most_important" = signed covariance RF
#   "rms"               = positive-only RMS RF
rf_variable_name = "cm_most_important"

# For covariance: divide by max(abs(RF)) while preserving sign.
# For RMS: divide by max(RF).
normalise_each_cell = True


# -----------------------------------------------------------------------------
# Alignment / stimulus geometry — MUST match the RF alignment script
# -----------------------------------------------------------------------------
raw_stim_width_px = 1000
raw_stim_height_px = 1000

displayed_stim_width_px = 800
displayed_stim_height_px = 1000

clip_rf_to_displayed_stimulus = True

# Pixel size assumed when the RF dataset x/y coordinates were generated.
dataset_assumed_um_per_px = 2.5

# IMPORTANT:
# Raw RF -> projected stimulus is identity.
# RF orientation is handled by the fixed left-right data mirror in the helper.


# -----------------------------------------------------------------------------
# Crop geometry
# -----------------------------------------------------------------------------
# 200 × 200 µm central analysis crop.
crop_size_um = 200.0

# Extra tissue for cone interpolation around the central crop.
interpolation_margin_um = 25.0

# None             = preserve transformed microscope resolution
# "corrected_stim" = one crop pixel per corrected stimulus pixel
# float            = explicit µm/pixel
cone_crop_output_um_per_px = None


# -----------------------------------------------------------------------------
# Blue-cone detection
# -----------------------------------------------------------------------------
blue_detection_sigma = 0.1
blue_threshold_rel = 0.55
blue_min_blob_area_px = 60
blue_max_blob_area_px = None
blue_opening_radius_px = 0
blue_closing_radius_px = 0


# -----------------------------------------------------------------------------
# Blue/UV row interpolation
# -----------------------------------------------------------------------------
manual_row_axis_angle_deg = None
row_angle_peak_halfwidth_deg = 12.0
row_pair_along_tol_fraction = 0.35
row_pair_perp_tol_fraction = 0.30

blue_uv_orthogonal_row_tol_fraction = 0.35
blue_uv_orthogonal_min_fraction = 0.25
blue_uv_orthogonal_max_fraction = 1.50
use_both_orthogonal_sides = True


# -----------------------------------------------------------------------------
# Red/green interpolation and cone sizes
# -----------------------------------------------------------------------------
fallback_blue_radius_um = 1.0
blue_radius_scale = 1.0
uv_radius_scale = 1.0
red_green_long_axis_scale = 0.8

red_near_blue = True
red_green_edge_clearance_um = 0.05
red_green_gap_fill_fraction = 0.99
duplicate_merge_distance_um = 0.10
red_green_angle_offset_deg = 90.0
red_green_max_short_axis_fraction_of_long_axis = 0.9
red_green_min_short_radius_um = 0.1


# -----------------------------------------------------------------------------
# Weighting and display
# -----------------------------------------------------------------------------
rf_weight_floor_fraction = 0.0

spider_display_percentile = 85
spider_max_lines = 250

rf_alpha = 0.7
cone_alpha = 0.4
scale_bar_um = 50.0
save_dpi = 300
save_qc_figure = False
show_figures = True


# %% ============================================================================
# CELL 2 — LOAD DATASET, CONE IMAGE AND ALIGNMENT — RUN ONCE
# ============================================================================

dataset = xr.load_dataset(path_to_data)
cone_image = load_image(image_1_cones_path)

geometry = build_alignment_geometry_matching_rf_alignment(
    alignment_cache_path,
    displayed_stim_height_px=displayed_stim_height_px,
    dataset_assumed_um_per_px=dataset_assumed_um_per_px,
)

print("------------------------------------------------------------")
print("RECORDING LOADED")
print("------------------------------------------------------------")
print("Dataset:", path_to_data)
print("Cone image:", image_1_cones_path)
print("Use plot_cell(cell_index) for each cell from this recording.")


# %% ============================================================================
# CELL 3 — DEFINE plot_cell() — RUN ONCE
# ============================================================================


def plot_cell(
    cell_index: int,
    *,
    save_outputs: bool = True,
    show: bool | None = None,
    save_qc: bool | None = None,
):
    """
    Analyse, plot and optionally save one cell WITHOUT reloading the recording.

    Examples
    --------
    plot_cell(116)
    plot_cell(235)
    plot_cell(284)

    Parameters
    ----------
    cell_index
        Cell index from the already-loaded dataset.
    save_outputs
        If True, save per-cone CSV, summary CSV, metadata JSON and main figure.
    show
        If None, use global show_figures.
    save_qc
        If None, use global save_qc_figure.

    Returns
    -------
    dict
        The RF, crop, mosaic, connectivity result, figure and output paths.
    """

    if show is None:
        show = show_figures

    if save_qc is None:
        save_qc = save_qc_figure

    plt.close("all")

    # ------------------------------------------------------------
    # RF extraction — same alignment operations as RF alignment script
    # ------------------------------------------------------------
    rf = extract_single_cell_rf_matching_rf_alignment(
        dataset,
        geometry,
        cell_index=cell_index,
        channel=channel_to_plot,
        rf_variable_name=rf_variable_name,
        normalise_each_cell=normalise_each_cell,
        displayed_stim_width_px=displayed_stim_width_px,
        displayed_stim_height_px=displayed_stim_height_px,
        clip_rf_to_displayed_stimulus=clip_rf_to_displayed_stimulus,
    )

    # ------------------------------------------------------------
    # Cone crop
    # ------------------------------------------------------------
    if cone_crop_output_um_per_px == "corrected_stim":
        crop_sampling_um_per_px = geometry.corrected_stim_um_per_px
    elif cone_crop_output_um_per_px is None:
        crop_sampling_um_per_px = None
    else:
        crop_sampling_um_per_px = float(cone_crop_output_um_per_px)

    aligned_crop = warp_cone_crop_to_projected_grid(
        cone_image,
        geometry,
        rf.peak_projected_um,
        crop_size_um=crop_size_um,
        interpolation_margin_um=interpolation_margin_um,
        output_um_per_px=crop_sampling_um_per_px,
    )

    # ------------------------------------------------------------
    # Cone-mosaic interpolation
    # ------------------------------------------------------------
    mosaic = interpolate_cone_mosaic_on_aligned_crop(
        aligned_crop,
        blue_detection_sigma=blue_detection_sigma,
        blue_threshold_rel=blue_threshold_rel,
        blue_min_blob_area_px=blue_min_blob_area_px,
        blue_max_blob_area_px=blue_max_blob_area_px,
        blue_opening_radius_px=blue_opening_radius_px,
        blue_closing_radius_px=blue_closing_radius_px,
        manual_row_axis_angle_deg=manual_row_axis_angle_deg,
        row_angle_peak_halfwidth_deg=row_angle_peak_halfwidth_deg,
        row_pair_along_tol_fraction=row_pair_along_tol_fraction,
        row_pair_perp_tol_fraction=row_pair_perp_tol_fraction,
        blue_uv_orthogonal_row_tol_fraction=blue_uv_orthogonal_row_tol_fraction,
        blue_uv_orthogonal_min_fraction=blue_uv_orthogonal_min_fraction,
        blue_uv_orthogonal_max_fraction=blue_uv_orthogonal_max_fraction,
        use_both_orthogonal_sides=use_both_orthogonal_sides,
        fallback_blue_radius_um=fallback_blue_radius_um,
        blue_radius_scale=blue_radius_scale,
        uv_radius_scale=uv_radius_scale,
        red_green_long_axis_scale=red_green_long_axis_scale,
        red_near_blue=red_near_blue,
        red_green_edge_clearance_um=red_green_edge_clearance_um,
        red_green_gap_fill_fraction=red_green_gap_fill_fraction,
        duplicate_merge_distance_um=duplicate_merge_distance_um,
        red_green_angle_offset_deg=red_green_angle_offset_deg,
        red_green_max_short_axis_fraction_of_long_axis=(
            red_green_max_short_axis_fraction_of_long_axis
        ),
        red_green_min_short_radius_um=red_green_min_short_radius_um,
    )

    # ------------------------------------------------------------
    # RF sampling and cone weights
    # ------------------------------------------------------------
    rf_on_crop_grid = sample_rf_on_aligned_crop(
        rf,
        geometry,
        aligned_crop,
    )

    connectivity = compute_cone_connectivity_weights(
        rf,
        geometry,
        mosaic,
        weight_floor_fraction=rf_weight_floor_fraction,
    )

    print("\nCone-type RF weights")
    print("--------------------")
    for cone_type, stats in connectivity.summary.items():
        print(
            f"{cone_type:>5}: "
            f"fraction={100 * stats['fraction_abs_weight']:6.2f}% | "
            f"n={stats['count']:4d} | "
            f"sum|w|={stats['sum_abs_weight']:.4f} | "
            f"signed sum={stats['signed_sum']:+.4f} | "
            f"mean|w|={stats['mean_abs_weight_per_cone']:.5f}"
        )

    # ------------------------------------------------------------
    # Save tables / metadata
    # ------------------------------------------------------------
    saved_paths = {}

    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths.update(
            save_connectivity_outputs(
                output_dir,
                cell_index=cell_index,
                connectivity=connectivity,
                geometry=geometry,
                rf=rf,
                mosaic=mosaic,
            )
        )

        figure_path = output_dir / f"cell_{cell_index}_cone_connectivity_figure.png"
    else:
        figure_path = None

    # ------------------------------------------------------------
    # Main figure
    # ------------------------------------------------------------
    fig = create_connectivity_figure(
        aligned_crop,
        rf_on_crop_grid,
        rf,
        mosaic,
        connectivity,
        cell_index=cell_index,
        channel=channel_to_plot,
        corrected_stim_um_per_px=geometry.corrected_stim_um_per_px,
        rf_alpha=rf_alpha,
        cone_alpha=cone_alpha,
        spider_display_percentile=spider_display_percentile,
        spider_max_lines=spider_max_lines,
        scale_bar_um=scale_bar_um,
        save_path=figure_path,
        dpi=save_dpi,
    )

    if figure_path is not None:
        saved_paths["figure_png"] = figure_path

    # ------------------------------------------------------------
    # Optional QC figure
    # ------------------------------------------------------------
    qc_fig = None

    if save_qc:
        output_dir.mkdir(parents=True, exist_ok=True)
        qc_path = output_dir / f"cell_{cell_index}_cone_interpolation_QC.png"

        qc_fig = create_detection_qc_figure(
            aligned_crop,
            mosaic,
            save_path=qc_path,
            dpi=save_dpi,
        )

        saved_paths["qc_png"] = qc_path

    if show:
        plt.show()
    else:
        plt.close("all")

    return {
        "cell_index": cell_index,
        "rf": rf,
        "aligned_crop": aligned_crop,
        "mosaic": mosaic,
        "rf_on_crop_grid": rf_on_crop_grid,
        "connectivity": connectivity,
        "figure": fig,
        "qc_figure": qc_fig,
        "saved_paths": saved_paths,
    }


# %% ============================================================================
# CELL 4 — PLOT CELLS — RERUN ONLY THIS CELL
# ============================================================================

result = plot_cell(260)

# Then, for another cell from the SAME loaded recording:
# result = plot_cell(235)
# result = plot_cell(284)

# You can also keep several results in memory:
# cell_116 = plot_cell(116)
# cell_235 = plot_cell(235)
# cell_284 = plot_cell(284)
