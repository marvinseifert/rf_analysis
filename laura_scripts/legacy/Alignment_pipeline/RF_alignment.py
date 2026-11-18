# %% ============================================================
# CELL 1 — LOAD EVERYTHING ONCE + DEFINE HELPER FUNCTIONS
# ============================================================
# Run this cell once per recording/session.
# After this has run, you can repeatedly edit/run CELL 2 and CELL 3
# without reloading the NetCDF/images or re-clicking alignment points.

from pathlib import Path
import json

import imageio.v3 as iio
import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from matplotlib.transforms import Affine2D
from matplotlib.colors import Normalize
from skimage.transform import resize, rotate, estimate_transform


# ------------------------------------------------------------
# RECORDING-SPECIFIC PATHS — edit these once per recording
# ------------------------------------------------------------
recording_name = "zebrafish_15_05_2026_phase_00"

# RF dataset
path_to_data = Path(
    r"F:\Laura\zebrafish_15_05_2026\Phase_00\noise_analysis\noise_data.nc"
)

# Image 1: cone mosaic, same microscope coordinate space as Image 2
image_1_cones_path = Path(
    r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish\Experiments\Imaging\20260515_dragonfly_microscope\15_05_2026_cones.jpg"
)

# Image 2: MEA electrodes with retina on top, same microscope coordinate space as Image 1
image_2_electrodes_retina_path = Path(
    r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish\Experiments\Imaging\20260515_dragonfly_microscope\AVG_Sample Prefix_2026-05-15_Lsteel_mCherry_18.08.13_Laura Steel_FusionStitcher.jpg"
)

# Image 3: projected visual stimulus on empty MEA
image_3_visual_stim_path = Path(
    r"F:\Laura\zebrafish_15_05_2026\alignment_videos\visual_stim_on_MEA.png"
)

# Optional: create Image 3 from the alignment video before loading it.
# Set to False if image_3_visual_stim_path already exists and is correct.
create_visual_stim_image_from_video = False
video_path = Path(
    r"F:\Laura\zebrafish_15_05_2026\alignment_videos\WIN_20260515_09_55_39_Pro.mp4"
)
video_crop_values = (100, 700, 200, 700)  # y1, y2, x1, x2
video_scale_factor = 2
video_rotation_angle = 0  # positive = anticlockwise
video_start_prop = 0.0
video_end_prop = 1.0

# Alignment cache: stores electrode clicks, transform, and stimulus-edge clicks.
alignment_cache_path = Path(
    rf"F:\Laura\zebrafish_15_05_2026\alignment_videos\{recording_name}_alignment_cache.npz"
)

# Click options
redo_electrode_alignment = False
redo_stimulus_clicks = False

# Choose which vertical edge of the projected visual stimulus to click:
#   "left"  = click top-left, then bottom-left
#   "right" = click top-right, then bottom-right
stimulus_edge_to_click = "left"

n_electrode_points = 12
transform_type = "affine"
electrode_spacing_um = 100

# Diagnostic display while clicking/alignment
show_image_load_overview = False
show_electrode_alignment_check = False
show_stimulus_click_check = False


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------


def prepare_display_image(img):
    """Convert image to a Matplotlib-friendly array."""
    img = np.asarray(img)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def to_rgb_for_background(img):
    """Ensure image is RGB for background display."""
    img = prepare_display_image(img)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    return img


def projection_of_visual_stim(
    video_path,
    crop_values,
    output_path,
    scale_factor=2,
    rotation_angle=0,
    start_prop=0.0,
    end_prop=1.0,
    show_plots=True,
):
    """
    Create an average projection from a video, crop it, upscale it, rotate it,
    and save the final image.
    """
    y1, y2, x1, x2 = crop_values

    n_total_frames = 0
    for _ in iio.imiter(video_path):
        n_total_frames += 1

    print("Total frames read:", n_total_frames)

    if n_total_frames == 0:
        raise ValueError("No frames were read from the video.")

    start_frame = int(start_prop * n_total_frames)
    end_frame = int(end_prop * n_total_frames)

    start_frame = max(0, start_frame)
    end_frame = min(n_total_frames, end_frame)

    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}"
        )

    print(f"Averaging frames {start_frame} to {end_frame - 1}")

    sum_frame = None
    n_frames = 0

    for i, frame in enumerate(iio.imiter(video_path)):
        if i < start_frame:
            continue

        if i >= end_frame:
            break

        frame = frame.astype(float)

        if sum_frame is None:
            sum_frame = np.zeros_like(frame, dtype=float)

        sum_frame += frame
        n_frames += 1

    if n_frames == 0:
        raise ValueError("No frames were included. Check start_prop and end_prop.")

    avg_projection = sum_frame / n_frames
    print(f"Averaged {n_frames} frames")

    if avg_projection.ndim == 3:
        avg_projection_gray = avg_projection.mean(axis=2)
    else:
        avg_projection_gray = avg_projection

    if show_plots:
        plt.figure(figsize=(8, 8), dpi=150)
        plt.imshow(avg_projection_gray, cmap="gray")
        plt.axis("off")
        plt.title("Full average projection")
        plt.show()

    crop = avg_projection_gray[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError(
            f"Crop is empty. Check crop_values: y1={y1}, y2={y2}, x1={x1}, x2={x2}"
        )

    print("Original full image shape:", avg_projection_gray.shape)
    print("Crop shape:", crop.shape)

    if show_plots:
        plt.figure(figsize=(8, 8), dpi=150)
        plt.imshow(crop, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.title("Cropped visual stimulus")
        plt.show()

    crop_upscaled = resize(
        crop,
        (
            int(crop.shape[0] * scale_factor),
            int(crop.shape[1] * scale_factor),
        ),
        anti_aliasing=True,
        preserve_range=True,
    )

    p1, p99 = np.percentile(crop_upscaled, (1, 99))

    crop_upscaled_rotated = rotate(
        crop_upscaled,
        angle=rotation_angle,
        resize=True,
        preserve_range=True,
    )

    if show_plots:
        plt.figure(figsize=(8, 8), dpi=300)
        plt.imshow(
            crop_upscaled_rotated,
            cmap="gray",
            vmin=p1,
            vmax=p99,
            interpolation="nearest",
        )
        plt.axis("off")
        plt.title("Upscaled crop with contrast adjustment & rotation")
        plt.show()

    plt.imsave(
        output_path,
        crop_upscaled_rotated,
        cmap="gray",
        vmin=p1,
        vmax=p99,
    )

    print(f"Saved final image to: {output_path}")

    return crop_upscaled_rotated, avg_projection_gray, crop


def click_electrode_alignment(
    image2,
    image3,
    n_points=12,
    transform_type="affine",
    electrode_spacing_um=100,
    show_check=True,
):
    """
    Click matching electrode centres in Image 2 and Image 3.

    Returns a dict containing:
        tform_2_to_3_params, points_image2, points_image3,
        residuals, um_per_px_image3.
    """
    image2_display = prepare_display_image(image2)
    image3_display = prepare_display_image(image3)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image2_display, cmap="gray")
    ax.set_title(
        f"Image 2: click {n_points} electrode centres\n"
        "Use the same order when clicking Image 3"
    )
    ax.axis("off")
    points_image2 = np.array(plt.ginput(n_points, timeout=0))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image3_display, cmap="gray")
    ax.set_title(
        f"Image 3: click the SAME {n_points} electrode centres\n" "in the SAME order"
    )
    ax.axis("off")
    points_image3 = np.array(plt.ginput(n_points, timeout=0))
    plt.close(fig)

    tform_2_to_3 = estimate_transform(
        transform_type,
        src=points_image2,
        dst=points_image3,
    )

    predicted_points_image3 = tform_2_to_3(points_image2)
    residuals = np.linalg.norm(predicted_points_image3 - points_image3, axis=1)

    dist_01_px_image3 = np.linalg.norm(points_image3[1] - points_image3[0])
    um_per_px_image3 = electrode_spacing_um / dist_01_px_image3

    print("------------------------------------------------------------")
    print("ELECTRODE ALIGNMENT")
    print("------------------------------------------------------------")
    print(f"Transform type: {transform_type}")
    print(f"Mean residual error: {residuals.mean():.2f} Image 3 px")
    print(f"Median residual error: {np.median(residuals):.2f} Image 3 px")
    print(f"Max residual error: {residuals.max():.2f} Image 3 px")
    print(f"Distance between first two Image 3 electrodes: {dist_01_px_image3:.2f} px")
    print(f"Image 3 calibration: {um_per_px_image3:.4f} µm/px")
    print(f"Equivalent: {1 / um_per_px_image3:.4f} px/µm")

    if show_check:
        h2, w2 = image2_display.shape[:2]
        h3, w3 = image3_display.shape[:2]
        mpl_tform_2_to_3 = Affine2D(tform_2_to_3.params)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

        axes[0].imshow(image2_display, cmap="gray")
        axes[0].scatter(
            points_image2[:, 0],
            points_image2[:, 1],
            s=40,
            facecolors="none",
            edgecolors="red",
        )
        axes[0].set_title("Image 2 clicked electrodes")
        axes[0].axis("off")

        axes[1].imshow(image3_display, cmap="gray")
        axes[1].scatter(
            points_image3[:, 0],
            points_image3[:, 1],
            s=60,
            facecolors="none",
            edgecolors="lime",
            label="Clicked Image 3 electrodes",
        )
        axes[1].scatter(
            predicted_points_image3[:, 0],
            predicted_points_image3[:, 1],
            s=25,
            color="red",
            label="Transformed Image 2 points",
        )
        axes[1].legend()
        axes[1].set_title("Electrode alignment check")
        axes[1].axis("off")

        axes[2].imshow(image3_display, cmap="gray")
        axes[2].imshow(
            image2_display,
            cmap="gray" if image2_display.ndim == 2 else None,
            alpha=0.6,
            extent=[0, w2, h2, 0],
            transform=mpl_tform_2_to_3 + axes[2].transData,
        )
        axes[2].set_xlim(0, w3)
        axes[2].set_ylim(h3, 0)
        axes[2].set_title("Image 2 transformed onto Image 3")
        axes[2].axis("on")

        plt.tight_layout()
        plt.show()

    return {
        "tform_2_to_3_params": tform_2_to_3.params,
        "points_image2": points_image2,
        "points_image3": points_image3,
        "residuals": residuals,
        "um_per_px_image3": um_per_px_image3,
    }


def click_stimulus_vertical_edge(
    image3,
    displayed_width_px,
    displayed_height_px,
    edge_side="right",
    show_check=True,
):
    """
    Click either the LEFT-HAND or RIGHT-HAND vertical edge of the displayed stimulus.

    Parameters
    ----------
    edge_side : {"left", "right"}
        "left":
            click 1) top-left corner, 2) bottom-left corner
            and infer the right-hand edge.

        "right":
            click 1) top-right corner, 2) bottom-right corner
            and infer the left-hand edge.

    The displayed stimulus centre is calculated from the clicked vertical edge
    using the known displayed width:height aspect ratio.
    """

    edge_side = str(edge_side).lower()

    if edge_side not in {"left", "right"}:
        raise ValueError("edge_side must be 'left' or 'right'")

    image3_display = to_rgb_for_background(image3)

    if edge_side == "left":
        top_corner_name = "top-left"
        bottom_corner_name = "bottom-left"
        edge_title = "LEFT-HAND"
    else:
        top_corner_name = "top-right"
        bottom_corner_name = "bottom-right"
        edge_title = "RIGHT-HAND"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(image3_display)
    ax.set_title(
        f"Click {edge_title} edge of displayed visual stimulus:\n"
        f"1) {top_corner_name} corner\n"
        f"2) {bottom_corner_name} corner"
    )
    ax.axis("on")

    clicked = np.array(plt.ginput(2, timeout=0))
    plt.close(fig)

    if clicked.shape != (2, 2):
        raise ValueError(
            f"Expected exactly 2 clicks, but received array with shape {clicked.shape}."
        )

    stim_edge_top_px = clicked[0]
    stim_edge_bottom_px = clicked[1]

    # Direction from top -> bottom along whichever vertical edge was clicked.
    down_vec = stim_edge_bottom_px - stim_edge_top_px
    height_in_image_px = np.linalg.norm(down_vec)

    if height_in_image_px == 0:
        raise ValueError("Top and bottom stimulus-edge clicks are identical.")

    down_unit = down_vec / height_in_image_px

    # If down is (0, +1), right is (+1, 0) and left is (-1, 0).
    right_unit = np.array(
        [down_unit[1], -down_unit[0]],
        dtype=float,
    )
    left_unit = -right_unit

    width_in_image_px = height_in_image_px * (displayed_width_px / displayed_height_px)

    right_vec = right_unit * width_in_image_px
    left_vec = left_unit * width_in_image_px

    if edge_side == "left":
        # Clicked edge is the left edge; infer the right edge.
        stim_top_left_px = stim_edge_top_px
        stim_bottom_left_px = stim_edge_bottom_px
        stim_top_right_px = stim_top_left_px + right_vec
        stim_bottom_right_px = stim_bottom_left_px + right_vec

        # Centre = halfway down + halfway RIGHT from clicked left edge.
        visual_stim_center_px_image3 = (
            stim_top_left_px + 0.5 * down_vec + 0.5 * right_vec
        )

    else:
        # Clicked edge is the right edge; infer the left edge.
        stim_top_right_px = stim_edge_top_px
        stim_bottom_right_px = stim_edge_bottom_px
        stim_top_left_px = stim_top_right_px + left_vec
        stim_bottom_left_px = stim_bottom_right_px + left_vec

        # Centre = halfway down + halfway LEFT from clicked right edge.
        visual_stim_center_px_image3 = (
            stim_top_right_px + 0.5 * down_vec + 0.5 * left_vec
        )

    print("------------------------------------------------------------")
    print("DISPLAYED STIMULUS CLICKS")
    print("------------------------------------------------------------")
    print(f"Clicked edge: {edge_side.upper()}")
    print(f"Clicked {top_corner_name}:", stim_edge_top_px)
    print(f"Clicked {bottom_corner_name}:", stim_edge_bottom_px)
    print(f"Displayed stimulus height: {height_in_image_px:.2f} px")
    print(f"Displayed stimulus width: {width_in_image_px:.2f} px")
    print("Calculated stimulus centre:")
    print(f"x = {visual_stim_center_px_image3[0]:.2f}")
    print(f"y = {visual_stim_center_px_image3[1]:.2f}")

    if show_check:
        stim_corners = np.array(
            [
                stim_top_left_px,
                stim_top_right_px,
                stim_bottom_right_px,
                stim_bottom_left_px,
                stim_top_left_px,
            ],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.imshow(image3_display)

        ax.plot(
            stim_corners[:, 0],
            stim_corners[:, 1],
            color="magenta",
            linewidth=1.5,
            label="Calculated stimulus boundary",
        )

        ax.scatter(
            visual_stim_center_px_image3[0],
            visual_stim_center_px_image3[1],
            color="cyan",
            marker="+",
            s=140,
            linewidths=2,
            label="Displayed stimulus centre",
        )

        ax.plot(
            [stim_edge_top_px[0], stim_edge_bottom_px[0]],
            [stim_edge_top_px[1], stim_edge_bottom_px[1]],
            color="cyan",
            linewidth=1.5,
            label=f"Clicked {edge_side.upper()} edge",
        )

        ax.legend()
        ax.set_title("Displayed stimulus click check")
        ax.axis("on")
        plt.show()

    return {
        "stim_edge_top_px": stim_edge_top_px,
        "stim_edge_bottom_px": stim_edge_bottom_px,
        "visual_stim_center_px_image3": visual_stim_center_px_image3,
        "stimulus_edge_side": edge_side,
    }


def load_alignment_cache(cache_path):
    if not cache_path.exists():
        return {}

    loaded = np.load(cache_path, allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def save_alignment_cache(cache_path, **kwargs):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **kwargs)
    print(f"Saved alignment cache to: {cache_path}")


def transform_points(A, points_xy):
    """Apply a 3x3 affine matrix to Nx2 points."""
    points_xy = np.asarray(points_xy, dtype=float)

    points_h = np.column_stack(
        [
            points_xy[:, 0],
            points_xy[:, 1],
            np.ones(len(points_xy)),
        ]
    )

    out = points_h @ A.T
    return out[:, :2]


def image_corners(width, height):
    """Return image corner coordinates in pixel space."""
    return np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=float,
    )


def data_extent_corners(x_min, x_max, y_min, y_max, close=False):
    """Return corners of a rectangular data extent."""
    corners = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=float,
    )

    if close:
        corners = np.vstack([corners, corners[0]])

    return corners


def coordinate_edges_from_centres(coords):
    """Convert 1D coordinate centres to imshow-style image edges."""
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 1:
        raise ValueError("coords must be 1D")

    if len(coords) < 2:
        raise ValueError("Need at least two coordinates to infer edges")

    diffs = np.diff(coords)

    if not np.allclose(diffs, diffs[0]):
        print("Warning: coordinates are not perfectly evenly spaced.")

    edges = np.empty(len(coords) + 1, dtype=float)
    edges[1:-1] = 0.5 * (coords[:-1] + coords[1:])
    edges[0] = coords[0] - 0.5 * diffs[0]
    edges[-1] = coords[-1] + 0.5 * diffs[-1]

    return edges


def robust_vmax(values, percentile=99.5, fallback=1.0):
    """Compute a robust vmax for RF plotting."""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return fallback

    vmax = float(np.nanpercentile(finite, percentile))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback

    return vmax


def flip_rf_left_right(da: xr.DataArray) -> xr.DataArray:
    """
    Mirror an RF left <-> right about its central vertical line.

    This flips the DATA along the x dimension while leaving the x-coordinate
    values themselves unchanged.

    In image terminology this is a horizontal flip.

        left  <---->  right

    This is deliberately different from:
        da.isel(x=slice(None, None, -1))

    because that would also reverse the x-coordinate labels.
    Here we want the RF contents to move to the opposite x positions while
    retaining the existing physical coordinate system.
    """
    if "x" not in da.dims:
        raise ValueError("Input DataArray must include an 'x' dimension.")

    x_axis = da.get_axis_num("x")

    flipped_values = np.flip(
        da.values,
        axis=x_axis,
    )

    flipped = da.copy(
        data=flipped_values,
        deep=True,
    )

    return flipped


def label_axes_as_raw_rf_um(
    ax,
    A_raw_rf_to_image3,
    xlim_px,
    ylim_px,
    step_um=500,
):
    """
    Label Image 3 pixel axes in raw RF/stimulus µm coordinates.

    Raw RF coordinates:
        +x = right in raw HDF5 stimulus
        +y = down in raw HDF5 stimulus
    """
    A_image3_to_raw_rf = np.linalg.inv(A_raw_rf_to_image3)

    x_left, x_right = min(xlim_px), max(xlim_px)
    y_top, y_bottom = min(ylim_px), max(ylim_px)

    visible_corners_px = np.array(
        [
            [x_left, y_top],
            [x_right, y_top],
            [x_right, y_bottom],
            [x_left, y_bottom],
        ],
        dtype=float,
    )

    visible_corners_rf = transform_points(
        A_image3_to_raw_rf,
        visible_corners_px,
    )

    rf_x_visible_min = np.nanmin(visible_corners_rf[:, 0])
    rf_x_visible_max = np.nanmax(visible_corners_rf[:, 0])
    rf_y_visible_min = np.nanmin(visible_corners_rf[:, 1])
    rf_y_visible_max = np.nanmax(visible_corners_rf[:, 1])

    xticks_um = np.arange(
        step_um * np.floor(rf_x_visible_min / step_um),
        step_um * np.ceil(rf_x_visible_max / step_um) + step_um,
        step_um,
    )

    yticks_um = np.arange(
        step_um * np.floor(rf_y_visible_min / step_um),
        step_um * np.ceil(rf_y_visible_max / step_um) + step_um,
        step_um,
    )

    xtick_points_rf = np.column_stack([xticks_um, np.zeros_like(xticks_um)])
    ytick_points_rf = np.column_stack([np.zeros_like(yticks_um), yticks_um])

    xtick_points_px = transform_points(A_raw_rf_to_image3, xtick_points_rf)
    ytick_points_px = transform_points(A_raw_rf_to_image3, ytick_points_rf)

    xticks_px = xtick_points_px[:, 0]
    yticks_px = ytick_points_px[:, 1]

    xmask = (xticks_px >= x_left) & (xticks_px <= x_right)
    ymask = (yticks_px >= y_top) & (yticks_px <= y_bottom)

    ax.set_xticks(xticks_px[xmask])
    ax.set_xticklabels([f"{int(v)}" for v in xticks_um[xmask]])

    ax.set_yticks(yticks_px[ymask])
    ax.set_yticklabels([f"{int(v)}" for v in yticks_um[ymask]])

    ax.set_xlabel("Raw RF/stimulus x position (µm)")
    ax.set_ylabel("Raw RF/stimulus y position (µm)")

    rf_x_axis = np.array([[rf_x_visible_min, 0], [rf_x_visible_max, 0]], dtype=float)
    rf_y_axis = np.array([[0, rf_y_visible_min], [0, rf_y_visible_max]], dtype=float)

    rf_x_axis_px = transform_points(A_raw_rf_to_image3, rf_x_axis)
    rf_y_axis_px = transform_points(A_raw_rf_to_image3, rf_y_axis)

    ax.plot(
        rf_x_axis_px[:, 0],
        rf_x_axis_px[:, 1],
        color="cyan",
        linewidth=0.8,
        alpha=0.35,
        zorder=20,
    )

    ax.plot(
        rf_y_axis_px[:, 0],
        rf_y_axis_px[:, 1],
        color="cyan",
        linewidth=0.8,
        alpha=0.35,
        zorder=20,
    )


# ------------------------------------------------------------
# OPTIONAL: CREATE IMAGE 3 FROM VIDEO
# ------------------------------------------------------------
if create_visual_stim_image_from_video:
    projection_of_visual_stim(
        video_path=video_path,
        crop_values=video_crop_values,
        output_path=image_3_visual_stim_path,
        scale_factor=video_scale_factor,
        rotation_angle=video_rotation_angle,
        start_prop=video_start_prop,
        end_prop=video_end_prop,
        show_plots=True,
    )


# ------------------------------------------------------------
# LOAD DATASET AND IMAGES ONCE
# ------------------------------------------------------------
print("------------------------------------------------------------")
print("LOADING DATA")
print("------------------------------------------------------------")

dataset = xr.load_dataset(path_to_data)
cone_image = iio.imread(image_1_cones_path)
MEA_electrode_retina = iio.imread(image_2_electrodes_retina_path)
visual_stim_image = iio.imread(image_3_visual_stim_path)

cone_display = prepare_display_image(cone_image)
MEA_electrode_retina_display = prepare_display_image(MEA_electrode_retina)
visual_stim_rgb = to_rgb_for_background(visual_stim_image)

if MEA_electrode_retina_display.ndim == 3:
    MEA_electrode_retina_gray = MEA_electrode_retina_display.mean(axis=2)
else:
    MEA_electrode_retina_gray = MEA_electrode_retina_display

h1, w1 = cone_display.shape[:2]
h2, w2 = MEA_electrode_retina_display.shape[:2]
h3, w3 = visual_stim_rgb.shape[:2]

print("Dataset loaded:", path_to_data)
print("Cone image shape:", cone_display.shape)
print("MEA/electrode image shape:", MEA_electrode_retina_display.shape)
print("Visual stimulus Image 3 shape:", visual_stim_rgb.shape)
print(dataset)

if show_image_load_overview:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    axes[0].imshow(cone_display, cmap="gray" if cone_display.ndim == 2 else None)
    axes[0].set_title("Image 1: cone mosaic")
    axes[0].axis("off")

    axes[1].imshow(MEA_electrode_retina_display, cmap="gray")
    axes[1].set_title("Image 2: MEA/electrodes + retina")
    axes[1].axis("off")

    axes[2].imshow(visual_stim_rgb)
    axes[2].set_title("Image 3: projected stimulus on MEA")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# LOAD OR CREATE ALIGNMENT CACHE
# ------------------------------------------------------------
cache = load_alignment_cache(alignment_cache_path)

need_electrode_alignment = (
    redo_electrode_alignment
    or "tform_2_to_3_params" not in cache
    or "points_image2" not in cache
    or "points_image3" not in cache
    or "um_per_px_image3" not in cache
)

if need_electrode_alignment:
    electrode_alignment = click_electrode_alignment(
        image2=MEA_electrode_retina_gray,
        image3=visual_stim_rgb,
        n_points=n_electrode_points,
        transform_type=transform_type,
        electrode_spacing_um=electrode_spacing_um,
        show_check=show_electrode_alignment_check,
    )
    cache.update(electrode_alignment)
else:
    print("Using cached electrode alignment:", alignment_cache_path)

# Load electrode alignment variables from cache dict
tform_2_to_3_params = np.asarray(cache["tform_2_to_3_params"], dtype=float)
points_image2 = np.asarray(cache["points_image2"], dtype=float)
points_image3 = np.asarray(cache["points_image3"], dtype=float)
residuals = np.asarray(cache["residuals"], dtype=float)
um_per_px_image3 = float(cache["um_per_px_image3"])
px_per_um_image3 = 1 / um_per_px_image3

print("------------------------------------------------------------")
print("IMAGE 2 -> IMAGE 3 ALIGNMENT SUMMARY")
print("------------------------------------------------------------")
print("Alignment cache:", alignment_cache_path)
print(f"Mean residual: {residuals.mean():.2f} Image 3 px")
print(f"Median residual: {np.median(residuals):.2f} Image 3 px")
print(f"Max residual: {residuals.max():.2f} Image 3 px")
print(f"Image 3 scale: {um_per_px_image3:.4f} µm/px")
print(f"Image 3 scale: {px_per_um_image3:.4f} px/µm")


# ------------------------------------------------------------
# LOAD OR CLICK DISPLAYED STIMULUS EDGE
# ------------------------------------------------------------
# These are the hard assumptions you requested:
raw_stim_width_px_load = 1000
raw_stim_height_px_load = 1000
displayed_stim_width_px_load = 800
displayed_stim_height_px_load = 1000

stimulus_edge_to_click = str(stimulus_edge_to_click).lower()

if stimulus_edge_to_click not in {"left", "right"}:
    raise ValueError("stimulus_edge_to_click must be 'left' or 'right'")

cached_stimulus_edge_side = None
if "stimulus_edge_side" in cache:
    cached_stimulus_edge_side = str(
        np.asarray(cache["stimulus_edge_side"]).item()
    ).lower()

need_stimulus_clicks = (
    redo_stimulus_clicks
    or "stim_edge_top_px" not in cache
    or "stim_edge_bottom_px" not in cache
    or "visual_stim_center_px_image3" not in cache
    or cached_stimulus_edge_side != stimulus_edge_to_click
)

if need_stimulus_clicks:
    stimulus_clicks = click_stimulus_vertical_edge(
        image3=visual_stim_rgb,
        displayed_width_px=displayed_stim_width_px_load,
        displayed_height_px=displayed_stim_height_px_load,
        edge_side=stimulus_edge_to_click,
        show_check=show_stimulus_click_check,
    )
    cache.update(stimulus_clicks)
else:
    print(
        f"Using cached {stimulus_edge_to_click.upper()} stimulus-edge clicks:",
        alignment_cache_path,
    )

stim_edge_top_px = np.asarray(
    cache["stim_edge_top_px"],
    dtype=float,
)

stim_edge_bottom_px = np.asarray(
    cache["stim_edge_bottom_px"],
    dtype=float,
)

visual_stim_center_px_image3 = np.asarray(
    cache["visual_stim_center_px_image3"],
    dtype=float,
)

stimulus_edge_side = str(np.asarray(cache["stimulus_edge_side"]).item()).lower()

# Save merged cache after possible updates.
save_alignment_cache(
    alignment_cache_path,
    tform_2_to_3_params=tform_2_to_3_params,
    points_image2=points_image2,
    points_image3=points_image3,
    residuals=residuals,
    um_per_px_image3=um_per_px_image3,
    stim_edge_top_px=stim_edge_top_px,
    stim_edge_bottom_px=stim_edge_bottom_px,
    visual_stim_center_px_image3=visual_stim_center_px_image3,
    stimulus_edge_side=stimulus_edge_side,
)

# Matplotlib transform for cone image onto Image 3
mpl_tform_1_to_3 = Affine2D(tform_2_to_3_params)

print("------------------------------------------------------------")
print("CELL 1 COMPLETE")
print("------------------------------------------------------------")
print("You can now edit/run CELL 2 and CELL 3 repeatedly without reloading data.")

# %% ============================================================
# CELL 2 — PARAMETERS TO CHANGE BEFORE RERUNNING THE PLOT
# ============================================================

# ------------------------------------------------------------
# RF choices
# ------------------------------------------------------------
# channel_to_plot = "32px_15Hz_20mins_shuffle_x4"
# channel_to_plot = "8px_15Hz_35mins_shuffle_x8"
# channel_to_plot = "6px_15Hz_30mins_shuffle_x6"
channel_to_plot = "12px_20Hz_25mins_shuffle_white"

# Choose: "rms" or "cm_most_important"
rf_dataset_to_plot = "rms"


# ------------------------------------------------------------
# Plot mode
# ------------------------------------------------------------
# Choose:
#   "population"  = combine many cells using sum/mean/max
#   "single_cell" = plot one selected cell only
plot_mode = "population"

# ------------------------------------------------------------
# Single-cell automatic selection
# ------------------------------------------------------------
# Choose:
#   "manual"       = use single_cell_index directly
#   "quality_rank" = select nth-highest quality cell
#   "tilt_rank"    = select nth-lowest tilt cell, after quality threshold
single_cell_selection_mode = "manual"

# Used only if single_cell_selection_mode = "manual"
single_cell_index = 226

# Used for "quality_rank" and "tilt_rank"
# 1 = best-ranked cell
# 2 = second-best-ranked cell
# 10 = tenth-best-ranked cell
single_cell_rank = 6

# Used for automatic selection.
# For quality_rank: only rank cells with QI >= this value.
# For tilt_rank: only rank cells with QI >= this value.
single_cell_auto_quality_threshold = 20

quality_variable_name = "quality"
tilt_variable_name = "tilt"

# Used only if plot_mode = "single_cell"
#   False = use normal crop_mode below
#   True  = zoom around this cell's strongest RF pixel
single_cell_zoom = True

# Half-width of zoom window in projected-stimulus µm.
# Example: 500 gives a 1000 × 1000 µm plot.
single_cell_zoom_half_width_um = 1000

# Mark the strongest RF pixel in single-cell plots
draw_single_cell_peak = False

# In population mode, quality threshold filters cells.
# In single-cell mode, this controls whether a below-threshold selected cell errors.
apply_quality_threshold_to_single_cell = False


# ------------------------------------------------------------
# RF filtering
# ------------------------------------------------------------
use_quality_threshold = True
quality_threshold = 20
quality_variable_name = "quality"


# ------------------------------------------------------------
# Population tilt filtering
# ------------------------------------------------------------
use_population_tilt_threshold = False

# Keep cells with tilt < this value
population_tilt_threshold = 0.75

# ------------------------------------------------------------
# RF spatial orientation
# ------------------------------------------------------------
# RF data are ALWAYS mirrored left <-> right before plotting.
# There is intentionally no orientation option here.

# ------------------------------------------------------------
# RF normalisation and combination
# ------------------------------------------------------------
# Choose:
#   False = keep original RF amplitudes
#   True  = normalise each cell before combining/plotting
#
# Normalisation is selected automatically by RF type:
#   rms:
#       divide each cell by its largest RMS value
#   cm_most_important:
#       divide each cell by its largest absolute covariance value,
#       while preserving positive and negative signs
normalise_each_cell = True

# Choose: "sum", "mean", or "max"
# Only used in population mode.
#
# For signed covariance, "mean" is generally the clearest population summary.
# If "max" is selected for covariance, the code retains the value with the
# largest absolute magnitude at each pixel while preserving its sign.
combine_method = "max"

# Optional final display normalisation.
# This applies to RMS only. It is automatically ignored for signed covariance,
# because ordinary 0-to-1 normalisation would remove its signed interpretation.
normalise_final_projection_to_0_1 = True


# ------------------------------------------------------------
# Stimulus geometry assumptions
# ------------------------------------------------------------
# Raw HDF5 stimulus file: 800 × 800 pixels.
raw_stim_width_px = 1000
raw_stim_height_px = 1000

# Displayed final projected stimulus: 800 high × 800 wide.
displayed_stim_width_px = 800
displayed_stim_height_px = 1000

# Set True to mask RF pixels outside the displayed 800×800 stimulus.
clip_rf_to_displayed_stimulus = True

# Boundaries to draw on the final plot.
draw_raw_stimulus_boundary = False  # yellow square = full raw 800×800
draw_displayed_stimulus_boundary = False  # magenta rectangle = displayed 800×800

# Raw RF/stimulus coordinates are used directly in projected-stimulus coordinates.
# In other words, raw -> projected is ALWAYS the identity transform.

# This should be the pixel size used when the RF dataset x/y coordinates
# were generated by the RF pipeline.
dataset_assumed_um_per_px = 2.5


# ------------------------------------------------------------
# Plot appearance
# ------------------------------------------------------------
rf_alpha = 0.6
cone_alpha = 1

# Colour scale.
# RMS uses a positive scale. Signed covariance automatically uses equal
# negative and positive limits centred on zero.
rf_vmax_percentile_if_not_normalised = 99.5

# Crop mode:
#   "displayed_stimulus" = crop around magenta displayed boundary
#   "raw_stimulus"       = crop around yellow raw boundary
#   "rf"                 = crop around full RF canvas
#   "all"                = include Image 3, cone image, raw boundary, displayed boundary, and RFs
#   "manual"             = use manual projected-stimulus µm limits below
#
# If plot_mode = "single_cell" and single_cell_zoom = True,
# this crop_mode is ignored and the plot zooms around the selected cell's RF peak.
crop_mode = "rf"
pad_px = 0

manual_x_min_um = -1200
manual_x_max_um = 1200
manual_y_min_um = -1200
manual_y_max_um = 1200

# Axes / figure
projected_tick_step_um = 200
display_figsize = (8, 8)
display_dpi = 150
save_output = False
save_dpi = 300

output_path = Path(f"cone_mosaic_{rf_dataset_to_plot}_flip_left_right_{plot_mode}.png")


# %% ============================================================
# CELL 3 — PLOTTING ONLY
# ============================================================
# You can rerun this cell repeatedly after editing CELL 2.

# ------------------------------------------------------------
# Validate parameter choices
# ------------------------------------------------------------
if plot_mode not in {"population", "single_cell"}:
    raise ValueError("plot_mode must be 'population' or 'single_cell'")

if single_cell_selection_mode not in {"manual", "quality_rank", "tilt_rank"}:
    raise ValueError(
        "single_cell_selection_mode must be 'manual', 'quality_rank', or 'tilt_rank'"
    )

if single_cell_rank < 1:
    raise ValueError("single_cell_rank must be >= 1")

if rf_dataset_to_plot not in {"rms", "cm_most_important"}:
    raise ValueError("rf_dataset_to_plot must be 'rms' or 'cm_most_important'")

if combine_method not in {"sum", "mean", "max"}:
    raise ValueError("combine_method must be 'sum', 'mean', or 'max'")

# ------------------------------------------------------------
# Dataset-specific plotting settings
# ------------------------------------------------------------
if rf_dataset_to_plot == "rms":
    rf_variable_name = "rms"
    rf_cmap = "Reds"
    rf_title_name = "RF (RMS)"
    rf_colourbar_name = "RMS"
    rf_is_signed = False

elif rf_dataset_to_plot == "cm_most_important":
    rf_variable_name = "cm_most_important"
    rf_cmap = "coolwarm"
    rf_title_name = "RF (covariance)"
    rf_colourbar_name = "covariance"
    rf_is_signed = True


# ------------------------------------------------------------
# Calculate corrected stimulus scale from clicked vertical edge
# ------------------------------------------------------------
clicked_edge_vec_px = stim_edge_bottom_px - stim_edge_top_px
clicked_edge_length_px = np.linalg.norm(clicked_edge_vec_px)

if clicked_edge_length_px == 0:
    raise ValueError("stim_edge_top_px and stim_edge_bottom_px are identical.")
clicked_edge_length_um = clicked_edge_length_px * um_per_px_image3

# Clicked edge is the longer displayed edge, assumed to be 800 stimulus pixels.
corrected_stim_um_per_px = clicked_edge_length_um / displayed_stim_height_px
rf_scale_factor = corrected_stim_um_per_px / dataset_assumed_um_per_px

print("------------------------------------------------------------")
print("SCALE CHECK")
print("------------------------------------------------------------")
print(
    f"Clicked {stimulus_edge_side.upper()} vertical edge length: {clicked_edge_length_px:.2f} Image 3 px"
)
print(
    f"Clicked {stimulus_edge_side.upper()} vertical edge length: {clicked_edge_length_um:.2f} µm"
)
print(f"Displayed stimulus height: {displayed_stim_height_px} stimulus px")
print(f"Corrected stimulus pixel size: {corrected_stim_um_per_px:.4f} µm/px")
print(f"RF dataset assumed pixel size: {dataset_assumed_um_per_px:.4f} µm/px")
print(f"RF coordinate scale factor: {rf_scale_factor:.4f}")


# ------------------------------------------------------------
# Build projected-stimulus µm -> Image 3 pixel transform
# ------------------------------------------------------------
# Projected coordinate system:
#   +x = right across displayed stimulus
#   +y = down along clicked vertical edge
#   origin = displayed stimulus centre

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


# ------------------------------------------------------------
# Raw RF/stimulus -> projected-stimulus transform
# ------------------------------------------------------------
# Fixed behaviour: raw RF/stimulus coordinates are already interpreted
# in the projected-stimulus coordinate system, so this transform is identity.
A_raw_rf_to_projected_um = np.eye(3, dtype=float)

A_raw_rf_to_image3 = A_projected_um_to_image3 @ A_raw_rf_to_projected_um
mpl_rf_transform = Affine2D(A_raw_rf_to_image3)

origin_check = A_raw_rf_to_image3 @ np.array([0, 0, 1], dtype=float)

print("------------------------------------------------------------")
print("TRANSFORM CHECK")
print("------------------------------------------------------------")
print("Raw RF [0,0] maps to Image 3:")
print(f"x = {origin_check[0]:.2f}, y = {origin_check[1]:.2f}")
print("Clicked displayed stimulus centre:")
print(
    f"x = {visual_stim_center_px_image3[0]:.2f}, "
    f"y = {visual_stim_center_px_image3[1]:.2f}"
)
print("Raw -> projected transform: identity")
print(f"Stimulus edge used for clicks: {stimulus_edge_side.upper()}")

# ------------------------------------------------------------
# Get RF data
# ------------------------------------------------------------
combined = dataset[rf_variable_name].sel(channel=channel_to_plot)

print("------------------------------------------------------------")
print("RF DATA")
print("------------------------------------------------------------")
print("Plot mode:", plot_mode)
print("Variable:", rf_variable_name)
print("Original dims:", combined.dims)
print("Original shape:", combined.shape)

print(
    "Applying fixed left-right RF mirror "
    "(horizontal flip about central vertical line)."
)
combined = flip_rf_left_right(combined)
# ------------------------------------------------------------
# Get quality and tilt values, if present
# ------------------------------------------------------------
quality = None
tilt = None

if quality_variable_name in dataset:
    quality = dataset[quality_variable_name]

    if "channel" in quality.dims:
        quality = quality.sel(channel=channel_to_plot)

    if "cell_index" not in quality.dims:
        print(
            f"dataset['{quality_variable_name}'] exists but does not have "
            f"cell_index dimension. Dims: {quality.dims}"
        )
        quality = None
else:
    print(f"dataset['{quality_variable_name}'] was not found.")


if tilt_variable_name in dataset:
    tilt = dataset[tilt_variable_name]

    if "channel" in tilt.dims:
        tilt = tilt.sel(channel=channel_to_plot)

    if "cell_index" not in tilt.dims:
        print(
            f"dataset['{tilt_variable_name}'] exists but does not have "
            f"cell_index dimension. Dims: {tilt.dims}"
        )
        tilt = None
else:
    print(f"dataset['{tilt_variable_name}'] was not found.")
# ------------------------------------------------------------
# Population mode: optional quality + tilt filtering
# ------------------------------------------------------------
if plot_mode == "population":
    # Start with every cell included
    population_cell_mask = xr.DataArray(
        np.ones(
            combined.sizes["cell_index"],
            dtype=bool,
        ),
        coords={"cell_index": combined["cell_index"]},
        dims=["cell_index"],
    )

    print("")
    print("------------------------------------------------------------")
    print("POPULATION CELL FILTER")
    print("------------------------------------------------------------")

    print(f"Cells before filtering: " f"{combined.sizes['cell_index']}")

    # --------------------------------------------------------
    # QUALITY FILTER
    # --------------------------------------------------------
    if use_quality_threshold:
        if quality is None:
            raise KeyError(
                f"Quality filtering requested, but usable "
                f"dataset['{quality_variable_name}'] "
                "was not found."
            )

        quality_mask = np.isfinite(quality) & (quality >= quality_threshold)

        population_cell_mask = population_cell_mask & quality_mask

        print(f"Quality filter: " f"{quality_variable_name} >= " f"{quality_threshold}")

    else:
        print("Quality filtering: OFF")

    # --------------------------------------------------------
    # TILT FILTER
    # --------------------------------------------------------
    if use_population_tilt_threshold:
        if tilt is None:
            raise KeyError(
                f"Tilt filtering requested, but usable "
                f"dataset['{tilt_variable_name}'] "
                "was not found."
            )

        tilt_mask = np.isfinite(tilt) & (tilt < population_tilt_threshold)

        population_cell_mask = population_cell_mask & tilt_mask

        print(
            f"Tilt filter: " f"{tilt_variable_name} < " f"{population_tilt_threshold}"
        )

    else:
        print("Tilt filtering: OFF")

    # --------------------------------------------------------
    # COUNT SELECTED CELLS
    # --------------------------------------------------------
    n_selected_cells = int(population_cell_mask.sum().item())

    print(f"Cells after filtering: " f"{n_selected_cells}")

    if n_selected_cells == 0:
        raise ValueError("No cells passed the population " "quality/tilt filters.")

    # --------------------------------------------------------
    # PRINT CELLS THAT PASSED
    # --------------------------------------------------------
    selected_population_cell_ids = (
        combined["cell_index"]
        .where(
            population_cell_mask,
            drop=True,
        )
        .values
    )

    print("")
    print("Selected population cells:")

    for cell_id in selected_population_cell_ids:
        cell_id = int(cell_id)

        text = f"cell {cell_id:>4}"

        if quality is not None:
            q = float(quality.sel(cell_index=cell_id).values)

            text += f", QI={q:.3f}"

        if tilt is not None:
            t = float(tilt.sel(cell_index=cell_id).values)

            text += f", tilt={t:.3f}"

        print(text)

    # --------------------------------------------------------
    # APPLY FILTER
    # --------------------------------------------------------
    combined = combined.where(
        population_cell_mask,
        drop=True,
    )
# ------------------------------------------------------------
# Single-cell mode: select one cell
# ------------------------------------------------------------
elif plot_mode == "single_cell":
    if "cell_index" not in combined.dims:
        raise ValueError(
            "Cannot plot a single cell because RF data has no cell_index dimension."
        )

    available_cells = combined["cell_index"].values

    # --------------------------------------------------------
    # Option 1: manual cell choice
    # --------------------------------------------------------
    if single_cell_selection_mode == "manual":
        if single_cell_index not in available_cells:
            raise ValueError(
                f"single_cell_index={single_cell_index} was not found in dataset cell_index.\n"
                f"Available examples: {available_cells[:20]}"
            )

        selected_cell_index = int(single_cell_index)

    # --------------------------------------------------------
    # Option 2: rank cells by quality, highest first
    # --------------------------------------------------------
    elif single_cell_selection_mode == "quality_rank":
        if quality is None:
            raise KeyError(
                f"single_cell_selection_mode='quality_rank' requires "
                f"dataset['{quality_variable_name}']."
            )

        candidate_quality = quality.where(
            quality >= single_cell_auto_quality_threshold,
            drop=True,
        )

        if candidate_quality.sizes["cell_index"] == 0:
            raise ValueError(
                f"No cells have {quality_variable_name} >= "
                f"{single_cell_auto_quality_threshold}."
            )

        sorted_quality = candidate_quality.sortby(candidate_quality, ascending=False)
        n_candidates = sorted_quality.sizes["cell_index"]

        if single_cell_rank > n_candidates:
            raise ValueError(
                f"single_cell_rank={single_cell_rank} is out of range. "
                f"There are {n_candidates} candidate cells with "
                f"{quality_variable_name} >= {single_cell_auto_quality_threshold}."
            )

        selected_cell_index = int(
            sorted_quality["cell_index"].isel(cell_index=single_cell_rank - 1).item()
        )

        print("")
        print("------------------------------------------------------------")
        print("AUTOMATIC SINGLE-CELL SELECTION")
        print("------------------------------------------------------------")
        print("Selection mode: quality_rank")
        print(f"Rank requested: {single_cell_rank}")
        print(f"Minimum quality threshold: {single_cell_auto_quality_threshold}")
        print(f"Candidate cells: {n_candidates}")
        print(f"Selected cell: {selected_cell_index}")
        print(
            f"Selected quality: "
            f"{float(quality.sel(cell_index=selected_cell_index).values):.3f}"
        )

        print("")
        print("Top ranked quality cells:")
        top_n_to_print = min(20, n_candidates)

        for i in range(top_n_to_print):
            cell_i = int(sorted_quality["cell_index"].isel(cell_index=i).item())
            q_i = float(sorted_quality.isel(cell_index=i).values)
            print(f"rank {i + 1:>2}: cell {cell_i:>4}, quality={q_i:.3f}")

    # --------------------------------------------------------
    # Option 3: rank cells by tilt, lowest first, after quality threshold
    # --------------------------------------------------------
    elif single_cell_selection_mode == "tilt_rank":
        if quality is None:
            raise KeyError(
                f"single_cell_selection_mode='tilt_rank' requires "
                f"dataset['{quality_variable_name}']."
            )

        if tilt is None:
            raise KeyError(
                f"single_cell_selection_mode='tilt_rank' requires "
                f"dataset['{tilt_variable_name}']."
            )

        good_quality = quality >= single_cell_auto_quality_threshold
        finite_tilt = np.isfinite(tilt)

        candidate_tilt = tilt.where(
            good_quality & finite_tilt,
            drop=True,
        )

        if candidate_tilt.sizes["cell_index"] == 0:
            raise ValueError(
                f"No cells have {quality_variable_name} >= "
                f"{single_cell_auto_quality_threshold} and finite {tilt_variable_name}."
            )

        sorted_tilt = candidate_tilt.sortby(candidate_tilt, ascending=True)
        n_candidates = sorted_tilt.sizes["cell_index"]

        if single_cell_rank > n_candidates:
            raise ValueError(
                f"single_cell_rank={single_cell_rank} is out of range. "
                f"There are {n_candidates} candidate cells with "
                f"{quality_variable_name} >= {single_cell_auto_quality_threshold} "
                f"and finite {tilt_variable_name}."
            )

        selected_cell_index = int(
            sorted_tilt["cell_index"].isel(cell_index=single_cell_rank - 1).item()
        )

        print("")
        print("------------------------------------------------------------")
        print("AUTOMATIC SINGLE-CELL SELECTION")
        print("------------------------------------------------------------")
        print("Selection mode: tilt_rank")
        print("Ranking rule: lowest tilt first")
        print(f"Rank requested: {single_cell_rank}")
        print(f"Minimum quality threshold: {single_cell_auto_quality_threshold}")
        print(f"Candidate cells: {n_candidates}")
        print(f"Selected cell: {selected_cell_index}")
        print(
            f"Selected quality: "
            f"{float(quality.sel(cell_index=selected_cell_index).values):.3f}"
        )
        print(
            f"Selected tilt: "
            f"{float(tilt.sel(cell_index=selected_cell_index).values):.3f}"
        )

        print("")
        print("Top ranked low-tilt cells:")
        top_n_to_print = min(20, n_candidates)

        for i in range(top_n_to_print):
            cell_i = int(sorted_tilt["cell_index"].isel(cell_index=i).item())
            tilt_i = float(sorted_tilt.isel(cell_index=i).values)
            q_i = float(quality.sel(cell_index=cell_i).values)
            print(
                f"rank {i + 1:>2}: cell {cell_i:>4}, "
                f"tilt={tilt_i:.3f}, quality={q_i:.3f}"
            )

    else:
        raise ValueError(
            "single_cell_selection_mode must be 'manual', 'quality_rank', or 'tilt_rank'"
        )

    # Make selected cell available to the rest of the plotting code
    single_cell_index = selected_cell_index

    print("")
    print("------------------------------------------------------------")
    print("SINGLE CELL SELECTION")
    print("------------------------------------------------------------")
    print(f"Selected cell: {single_cell_index}")
    print(f"Selection mode: {single_cell_selection_mode}")
    print(
        f"Rank: {single_cell_rank if single_cell_selection_mode != 'manual' else 'manual'}"
    )

    if quality is not None:
        selected_quality = float(quality.sel(cell_index=single_cell_index).values)
        print(f"Selected cell quality: {selected_quality:.3f}")

    if tilt is not None:
        selected_tilt = float(tilt.sel(cell_index=single_cell_index).values)
        print(f"Selected cell tilt: {selected_tilt:.3f}")

    combined = combined.sel(cell_index=single_cell_index)


# ------------------------------------------------------------
# Optional per-cell / single-cell normalisation
# ------------------------------------------------------------
if normalise_each_cell:
    if rf_is_signed:
        print(
            "Normalising each covariance RF by its largest absolute value "
            "while preserving positive and negative signs."
        )

        combined_scale = np.abs(combined).max(
            dim=["x", "y"],
            skipna=True,
        )

    else:
        print("Normalising each RMS RF by its largest RMS value.")

        combined_scale = combined.max(
            dim=["x", "y"],
            skipna=True,
        )

    combined_scale = combined_scale.where(
        np.isfinite(combined_scale) & (combined_scale != 0)
    )

    combined = combined / combined_scale
    combined = combined.fillna(0)

else:
    print("Per-cell normalisation OFF. Keeping original RF amplitude.")

    if (
        plot_mode == "population"
        and combine_method == "sum"
        and "cell_index" in combined.dims
    ):
        combined = combined.fillna(0)


# ------------------------------------------------------------
# Combine cells, or keep selected single cell
# ------------------------------------------------------------
if "cell_index" in combined.dims:
    if combine_method == "sum":
        rf_projection = combined.sum(
            dim="cell_index",
            skipna=True,
        )

    elif combine_method == "mean":
        rf_projection = combined.mean(
            dim="cell_index",
            skipna=True,
        )

    elif combine_method == "max":
        if rf_is_signed:
            print(
                "Signed covariance with combine_method='max': selecting "
                "the largest absolute value at each pixel while preserving sign."
            )

            largest_magnitude_cell = np.abs(combined).argmax(
                dim="cell_index",
                skipna=True,
            )

            rf_projection = combined.isel(cell_index=largest_magnitude_cell)

        else:
            rf_projection = combined.max(
                dim="cell_index",
                skipna=True,
            )

else:
    rf_projection = combined


# ------------------------------------------------------------
# Ensure RF is in y,x order for imshow
# ------------------------------------------------------------
dims_before = rf_projection.dims
spatial_dims_before = tuple(d for d in dims_before if d in ("y", "x"))

if spatial_dims_before == ("y", "x"):
    print(
        'RF spatial dimensions already in ("y", "x") order, so no transpose was applied.'
    )
else:
    print(f"RF dims before display transpose: {dims_before}")
    rf_projection = rf_projection.transpose("y", "x")
    print('RF spatial dimensions transposed to ("y", "x") for imshow display.')

rf_to_plot = rf_projection.values
rf_x = rf_projection["x"].values
rf_y = rf_projection["y"].values

print("")
print("------------------------------------------------------------")
print("RF PROJECTION")
print("------------------------------------------------------------")
print("Plot mode:", plot_mode)

if plot_mode == "population":
    print("Combine method:", combine_method)
else:
    print("Single cell:", single_cell_index)

print("RF image shape:", rf_to_plot.shape, "= y,x")
print("Original RF x min/max:", np.nanmin(rf_x), np.nanmax(rf_x))
print("Original RF y min/max:", np.nanmin(rf_y), np.nanmax(rf_y))


# ------------------------------------------------------------
# RF coordinate edges and scale correction
# ------------------------------------------------------------
rf_x_edges_original = coordinate_edges_from_centres(rf_x)
rf_y_edges_original = coordinate_edges_from_centres(rf_y)

rf_x_edges_corrected = rf_x_edges_original * rf_scale_factor
rf_y_edges_corrected = rf_y_edges_original * rf_scale_factor

rf_x_min_edge = float(rf_x_edges_corrected[0])
rf_x_max_edge = float(rf_x_edges_corrected[-1])
rf_y_min_edge = float(rf_y_edges_corrected[0])
rf_y_max_edge = float(rf_y_edges_corrected[-1])

print("Corrected RF x edge min/max:", rf_x_min_edge, rf_x_max_edge)
print("Corrected RF y edge min/max:", rf_y_min_edge, rf_y_max_edge)
print("Corrected RF x span:", rf_x_max_edge - rf_x_min_edge)
print("Corrected RF y span:", rf_y_max_edge - rf_y_min_edge)

# ------------------------------------------------------------
# Boundaries: raw 800×800 and displayed 800×800
# ------------------------------------------------------------
raw_half_x_um = (raw_stim_width_px * corrected_stim_um_per_px) / 2
raw_half_y_um = (raw_stim_height_px * corrected_stim_um_per_px) / 2

displayed_half_x_um = (displayed_stim_width_px * corrected_stim_um_per_px) / 2
displayed_half_y_um = (displayed_stim_height_px * corrected_stim_um_per_px) / 2

# Raw square in raw RF coordinates
raw_stim_corners_raw_um = data_extent_corners(
    -raw_half_x_um,
    raw_half_x_um,
    -raw_half_y_um,
    raw_half_y_um,
    close=True,
)

raw_stim_corners_img3 = transform_points(
    A_raw_rf_to_image3,
    raw_stim_corners_raw_um,
)

# Displayed rectangle in projected-stimulus coordinates.
# This means the 800 px crop is centred in the final projected image.
displayed_stim_corners_projected_um = data_extent_corners(
    -displayed_half_x_um,
    displayed_half_x_um,
    -displayed_half_y_um,
    displayed_half_y_um,
    close=True,
)

displayed_stim_corners_img3 = transform_points(
    A_projected_um_to_image3,
    displayed_stim_corners_projected_um,
)


# ------------------------------------------------------------
# Optional mask to displayed stimulus region
# ------------------------------------------------------------
rf_x_corrected_centres = rf_x * rf_scale_factor
rf_y_corrected_centres = rf_y * rf_scale_factor

raw_x_grid_um, raw_y_grid_um = np.meshgrid(
    rf_x_corrected_centres,
    rf_y_corrected_centres,
)

projected_x_grid_um = (
    A_raw_rf_to_projected_um[0, 0] * raw_x_grid_um
    + A_raw_rf_to_projected_um[0, 1] * raw_y_grid_um
    + A_raw_rf_to_projected_um[0, 2]
)

projected_y_grid_um = (
    A_raw_rf_to_projected_um[1, 0] * raw_x_grid_um
    + A_raw_rf_to_projected_um[1, 1] * raw_y_grid_um
    + A_raw_rf_to_projected_um[1, 2]
)

valid_displayed_mask_yx = (
    (projected_x_grid_um >= -displayed_half_x_um)
    & (projected_x_grid_um <= displayed_half_x_um)
    & (projected_y_grid_um >= -displayed_half_y_um)
    & (projected_y_grid_um <= displayed_half_y_um)
)

print("Displayed stimulus region:")
print(f"projected x = {-displayed_half_x_um:.2f} to {displayed_half_x_um:.2f} µm")
print(f"projected y = {-displayed_half_y_um:.2f} to {displayed_half_y_um:.2f} µm")
print(
    f"RF pixels inside displayed region: "
    f"{np.sum(valid_displayed_mask_yx)} / {valid_displayed_mask_yx.size}"
)

if clip_rf_to_displayed_stimulus:
    rf_to_plot = np.where(valid_displayed_mask_yx, rf_to_plot, np.nan)


# ------------------------------------------------------------
# Get the stored most-important-pixel position for a single cell
# ------------------------------------------------------------
# dataset["positions"] stores the x and y ARRAY INDICES selected by the
# RF pipeline as the most-important pixel. These indices are converted to
# physical coordinates using dataset["x"] and dataset["y"].
single_cell_peak_projected_um = None
single_cell_peak_raw_um = None

if plot_mode == "single_cell":
    if "positions" not in dataset:
        raise KeyError("Dataset does not contain variable 'positions'.")

    positions = dataset["positions"]

    if "channel" in positions.dims:
        positions = positions.sel(channel=channel_to_plot)

    if "cell_index" not in positions.dims:
        raise ValueError(
            "dataset['positions'] does not contain a cell_index dimension. "
            f"Dims: {positions.dims}"
        )

    if "pos_dim" not in positions.dims:
        raise ValueError(
            "dataset['positions'] does not contain a pos_dim dimension. "
            f"Dims: {positions.dims}"
        )

    peak_x_index = int(
        positions.sel(
            cell_index=single_cell_index,
            pos_dim="x",
        ).item()
    )

    peak_y_index = int(
        positions.sel(
            cell_index=single_cell_index,
            pos_dim="y",
        ).item()
    )

    # RF data are always mirrored left <-> right, so mirror the stored
    # most-important-pixel x index in the same way.
    original_peak_x_index = peak_x_index
    peak_x_index = dataset.sizes["x"] - 1 - peak_x_index

    print(
        "Mirroring stored RF peak x index: "
        f"{original_peak_x_index} -> {peak_x_index}"
    )

    if not (0 <= peak_x_index < dataset.sizes["x"]):
        raise IndexError(
            f"Stored x index {peak_x_index} is outside the dataset x range "
            f"0 to {dataset.sizes['x'] - 1}."
        )

    if not (0 <= peak_y_index < dataset.sizes["y"]):
        raise IndexError(
            f"Stored y index {peak_y_index} is outside the dataset y range "
            f"0 to {dataset.sizes['y'] - 1}."
        )

    peak_raw_x_dataset_units = float(dataset["x"].isel(x=peak_x_index).item())

    peak_raw_y_dataset_units = float(dataset["y"].isel(y=peak_y_index).item())

    # Apply the same scale correction used for the RF image coordinates.
    peak_raw_x_um = peak_raw_x_dataset_units * rf_scale_factor
    peak_raw_y_um = peak_raw_y_dataset_units * rf_scale_factor

    single_cell_peak_raw_um = np.array(
        [peak_raw_x_um, peak_raw_y_um],
        dtype=float,
    )

    single_cell_peak_projected_um = transform_points(
        A_raw_rf_to_projected_um,
        single_cell_peak_raw_um[None, :],
    )[0]

    print("")
    print("------------------------------------------------------------")
    print("STORED MOST-IMPORTANT-PIXEL POSITION")
    print("------------------------------------------------------------")
    print(f"Cell: {single_cell_index}")
    print(f"Stored RF array index: x={peak_x_index}, y={peak_y_index}")
    print(
        f"Dataset coordinate before scale correction: "
        f"x={peak_raw_x_dataset_units:.2f}, "
        f"y={peak_raw_y_dataset_units:.2f}"
    )
    print(
        f"Corrected raw RF position: "
        f"x={peak_raw_x_um:.2f}, y={peak_raw_y_um:.2f} µm"
    )
    print(
        f"Projected stimulus position: "
        f"x={single_cell_peak_projected_um[0]:.2f}, "
        f"y={single_cell_peak_projected_um[1]:.2f} µm"
    )


# ------------------------------------------------------------
# Single-cell zoom centre from ACTUAL plotted RF values
# ------------------------------------------------------------
# This fixes the issue where single_cell_zoom centres on dataset["positions"],
# but the visible RF has already been transformed/flipped for plotting.
# It finds the strongest visible RF pixel in rf_to_plot and uses that as the
# zoom centre.

if plot_mode == "single_cell" and single_cell_zoom:
    rf_abs = np.abs(rf_to_plot) if rf_is_signed else rf_to_plot

    finite = np.isfinite(rf_abs)

    if np.any(finite):
        masked = np.where(finite, rf_abs, -np.inf)

        peak_y_i, peak_x_i = np.unravel_index(
            np.nanargmax(masked),
            masked.shape,
        )

        # Convert image pixel index to RF coordinate centre.
        peak_x_um = 0.5 * (
            rf_x_edges_corrected[peak_x_i] + rf_x_edges_corrected[peak_x_i + 1]
        )

        peak_y_um = 0.5 * (
            rf_y_edges_corrected[peak_y_i] + rf_y_edges_corrected[peak_y_i + 1]
        )

        # Convert raw RF coordinates into projected-stimulus coordinates.
        single_cell_peak_projected_um = transform_points(
            A_raw_rf_to_projected_um,
            np.array([[peak_x_um, peak_y_um]], dtype=float),
        )[0]

        print("")
        print("------------------------------------------------------------")
        print("PLOTTED RF PEAK USED FOR SINGLE-CELL ZOOM")
        print("------------------------------------------------------------")
        print(f"Peak array index: y={peak_y_i}, x={peak_x_i}")
        print(f"Peak raw RF coordinate: x={peak_x_um:.2f}, y={peak_y_um:.2f} µm")
        print(
            "Peak projected coordinate: "
            f"x={single_cell_peak_projected_um[0]:.2f}, "
            f"y={single_cell_peak_projected_um[1]:.2f} µm"
        )

# ------------------------------------------------------------
# Final display normalisation and colour limits
# ------------------------------------------------------------
finite = rf_to_plot[np.isfinite(rf_to_plot)]

if rf_is_signed:
    # Signed covariance always uses a symmetric colour scale centred on zero.
    # Do not apply ordinary 0-to-1 final normalisation.
    if finite.size == 0:
        rf_vmax = 1.0

    elif plot_mode == "single_cell" and normalise_each_cell:
        # A per-cell normalised covariance RF is bounded by -1 and +1.
        rf_vmax = 1.0

    else:
        rf_vmax = float(
            np.nanpercentile(
                np.abs(finite),
                rf_vmax_percentile_if_not_normalised,
            )
        )

        if not np.isfinite(rf_vmax) or rf_vmax <= 0:
            rf_vmax = 1.0

    rf_vmin = -rf_vmax

else:
    # RMS is positive-only.
    if normalise_final_projection_to_0_1:
        if finite.size > 0:
            final_max = np.nanmax(finite)

            if np.isfinite(final_max) and final_max > 0:
                rf_to_plot = rf_to_plot / final_max

        rf_vmin = 0.0
        rf_vmax = 1.0

    elif normalise_each_cell:
        rf_vmin = 0.0
        rf_vmax = 1.0

    else:
        rf_vmin = 0.0
        rf_vmax = robust_vmax(
            rf_to_plot,
            percentile=rf_vmax_percentile_if_not_normalised,
            fallback=1.0,
        )

rf_norm = Normalize(vmin=rf_vmin, vmax=rf_vmax)

print("Colour scale:")
print("rf_vmin:", rf_vmin)
print("rf_vmax:", rf_vmax)


# ------------------------------------------------------------
# PLOT IN PROJECTED-STIMULUS µm COORDINATES
# ------------------------------------------------------------
# In this coordinate system:
#   x = projected stimulus x position in µm
#   y = projected stimulus y position in µm
#   origin = displayed stimulus / RF centre
#   +y = down the projected stimulus

from matplotlib.ticker import MultipleLocator

# Image 3 pixel coordinates -> projected-stimulus µm
A_image3_to_projected_um = np.linalg.inv(A_projected_um_to_image3)

# Image 1/cone coordinates -> Image 3 -> projected-stimulus µm
A_image1_to_projected_um = A_image3_to_projected_um @ tform_2_to_3_params

# RF raw coordinates -> projected-stimulus µm
mpl_image3_to_projected_um = Affine2D(A_image3_to_projected_um)
mpl_image1_to_projected_um = Affine2D(A_image1_to_projected_um)
mpl_rf_to_projected_um = Affine2D(A_raw_rf_to_projected_um)


# ------------------------------------------------------------
# Boundaries in projected-stimulus µm
# ------------------------------------------------------------
raw_stim_corners_projected_um = transform_points(
    A_raw_rf_to_projected_um,
    raw_stim_corners_raw_um,
)

# displayed_stim_corners_projected_um already exists and is already
# in projected-stimulus µm coordinates.


# ------------------------------------------------------------
# Clicked vertical edge in projected-stimulus µm
# ------------------------------------------------------------
clicked_edge_projected_um = transform_points(
    A_image3_to_projected_um,
    np.vstack([stim_edge_top_px, stim_edge_bottom_px]),
)
# ------------------------------------------------------------
# Crop limits in projected-stimulus µm
# ------------------------------------------------------------
image3_corners_projected_um = transform_points(
    A_image3_to_projected_um,
    image_corners(w3, h3),
)

cone_corners_projected_um = transform_points(
    A_image1_to_projected_um,
    image_corners(w1, h1),
)

rf_corners_raw_um = data_extent_corners(
    rf_x_min_edge,
    rf_x_max_edge,
    rf_y_min_edge,
    rf_y_max_edge,
    close=False,
)

rf_corners_projected_um = transform_points(
    A_raw_rf_to_projected_um,
    rf_corners_raw_um,
)

if (
    plot_mode == "single_cell"
    and single_cell_zoom
    and single_cell_peak_projected_um is not None
):
    peak_x_um = single_cell_peak_projected_um[0]
    peak_y_um = single_cell_peak_projected_um[1]

    crop_pts_um = data_extent_corners(
        peak_x_um - single_cell_zoom_half_width_um,
        peak_x_um + single_cell_zoom_half_width_um,
        peak_y_um - single_cell_zoom_half_width_um,
        peak_y_um + single_cell_zoom_half_width_um,
        close=False,
    )

elif crop_mode == "all":
    crop_pts_um = np.vstack(
        [
            image3_corners_projected_um,
            cone_corners_projected_um,
            rf_corners_projected_um,
            raw_stim_corners_projected_um,
            displayed_stim_corners_projected_um,
        ]
    )

elif crop_mode == "rf":
    crop_pts_um = rf_corners_projected_um

elif crop_mode == "raw_stimulus":
    crop_pts_um = raw_stim_corners_projected_um

elif crop_mode == "displayed_stimulus":
    crop_pts_um = displayed_stim_corners_projected_um

elif crop_mode == "manual":
    crop_pts_um = data_extent_corners(
        manual_x_min_um,
        manual_x_max_um,
        manual_y_min_um,
        manual_y_max_um,
        close=False,
    )

else:
    raise ValueError(
        "crop_mode must be 'all', 'rf', 'raw_stimulus', "
        "'displayed_stimulus', or 'manual'"
    )

# Convert pixel padding into µm padding.
# Do not add extra pad if single-cell zoom is explicitly requested.
if (
    plot_mode == "single_cell"
    and single_cell_zoom
    and single_cell_peak_projected_um is not None
):
    pad_um = 0
else:
    pad_um = pad_px * um_per_px_image3

x_min_um = np.nanmin(crop_pts_um[:, 0]) - pad_um
x_max_um = np.nanmax(crop_pts_um[:, 0]) + pad_um
y_min_um = np.nanmin(crop_pts_um[:, 1]) - pad_um
y_max_um = np.nanmax(crop_pts_um[:, 1]) + pad_um

print("Plot limits in projected-stimulus µm:")
print("x:", x_min_um, x_max_um)
print("y:", y_min_um, y_max_um)


# ------------------------------------------------------------
# Final plot
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=display_figsize,
    dpi=display_dpi,
    constrained_layout=True,
)

# ------------------------------------------------------------
# Image 3 background transformed into projected µm coordinates
# ------------------------------------------------------------
# ax.imshow(
#     visual_stim_rgb,
#     extent=[0, w3, h3, 0],
#     transform=mpl_image3_to_projected_um + ax.transData,
#     zorder=4,
# )

# # ------------------------------------------------------------
# # Cone mosaic transformed into projected µm coordinates
# # ------------------------------------------------------------
ax.imshow(
    cone_display,
    alpha=cone_alpha,
    extent=[0, w1, h1, 0],
    origin="upper",
    transform=mpl_image1_to_projected_um + ax.transData,
    zorder=2,
)

# ------------------------------------------------------------
# RF overlay transformed from raw RF µm -> projected µm
# ------------------------------------------------------------
rf_im = ax.imshow(
    rf_to_plot,
    cmap=rf_cmap,
    norm=rf_norm,
    alpha=rf_alpha,
    origin="upper",
    extent=[
        rf_x_min_edge,
        rf_x_max_edge,
        rf_y_max_edge,
        rf_y_min_edge,
    ],
    transform=mpl_rf_to_projected_um + ax.transData,
    interpolation="nearest",
    zorder=10,
)

# ------------------------------------------------------------
# Raw 800 × 800 HDF5 stimulus boundary
# ------------------------------------------------------------
if draw_raw_stimulus_boundary:
    ax.plot(
        raw_stim_corners_projected_um[:, 0],
        raw_stim_corners_projected_um[:, 1],
        color="yellow",
        linewidth=1.5,
        alpha=0.95,
        label="Raw HDF5 stimulus boundary 800×800",
        zorder=30,
    )

# ------------------------------------------------------------
# Displayed 800 × 800 stimulus boundary
# # ------------------------------------------------------------
if draw_displayed_stimulus_boundary:
    ax.plot(
        displayed_stim_corners_projected_um[:, 0],
        displayed_stim_corners_projected_um[:, 1],
        color="magenta",
        linewidth=1.8,
        alpha=0.95,
        label="Displayed stimulus boundary 800×800",
        zorder=31,
    )

# ------------------------------------------------------------
# Clicked vertical edge
# ------------------------------------------------------------
ax.plot(
    clicked_edge_projected_um[:, 0],
    clicked_edge_projected_um[:, 1],
    color="cyan",
    linewidth=1.2,
    alpha=0.9,
    label="Clicked displayed vertical edge",
    zorder=32,
)

# ------------------------------------------------------------
# RF origin / displayed stimulus centre
# ------------------------------------------------------------
ax.scatter(
    0,
    0,
    s=140,
    marker="+",
    color="cyan",
    linewidths=2.2,
    label="RF origin / displayed stimulus centre",
    zorder=33,
)

# ------------------------------------------------------------
# Single-cell RF peak marker
# ------------------------------------------------------------
if (
    plot_mode == "single_cell"
    and draw_single_cell_peak
    and single_cell_peak_projected_um is not None
):
    ax.scatter(
        single_cell_peak_projected_um[0],
        single_cell_peak_projected_um[1],
        s=90,
        marker="x",
        color="white",
        linewidths=2.0,
        label=f"cell {single_cell_index} RF peak",
        zorder=34,
    )


# ------------------------------------------------------------
# Axes are directly in µm
# ------------------------------------------------------------
ax.set_xlim(x_min_um, x_max_um)

# Invert y-axis so +y is visually downward, matching image/stimulus convention.
ax.set_ylim(y_max_um, y_min_um)

ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("Projected stimulus x position (µm)")
ax.set_ylabel("Projected stimulus y position (µm)")

ax.xaxis.set_major_locator(MultipleLocator(projected_tick_step_um))
ax.yaxis.set_major_locator(MultipleLocator(projected_tick_step_um))

ax.tick_params(axis="both", labelsize=9)


# ------------------------------------------------------------
# Title, legend, colourbar
# ------------------------------------------------------------
xy_label = "RF mirrored left-right"
norm_label = (
    "per-cell normalised" if normalise_each_cell else "no per-cell normalisation"
)

clip_label = (
    "clipped to displayed stimulus"
    if clip_rf_to_displayed_stimulus
    else "full RF canvas"
)

if plot_mode == "single_cell":
    if quality is not None:
        selected_quality = float(quality.sel(cell_index=single_cell_index).values)
        quality_label = f"cell {single_cell_index}; QI={selected_quality:.2f}"
    else:
        quality_label = f"cell {single_cell_index}"
elif use_quality_threshold:
    quality_label = f"QI ≥ {quality_threshold}"
else:
    quality_label = "all cells"

if plot_mode == "population":
    plot_mode_label = f"combine={combine_method}; {norm_label}; {clip_label}"
else:
    zoom_label = (
        f"zoom ±{single_cell_zoom_half_width_um} µm"
        if single_cell_zoom
        else f"crop={crop_mode}"
    )
    plot_mode_label = f"single-cell plot; {norm_label}; {clip_label}; {zoom_label}"

ax.set_title(
    (
        f"sws2 cone mosaic and {rf_title_name} overlay\n"
        f"{recording_name}; {channel_to_plot}\n"
        f"{quality_label};corrected stimulus scale = {corrected_stim_um_per_px:.3f} µm/px"
    ),
    fontsize=10,
)

ax.legend(loc="upper right", frameon=True, fontsize=7)

cbar = fig.colorbar(
    rf_im,
    ax=ax,
    fraction=0.04,
    pad=0.03,
)

cbar.ax.tick_params(labelsize=8)

if plot_mode == "single_cell":
    cbar_prefix = "Single-cell"
elif combine_method == "sum":
    cbar_prefix = "Sum"
elif combine_method == "mean":
    cbar_prefix = "Mean"
else:
    cbar_prefix = "Max"

if normalise_each_cell:
    cbar_norm_text = "per-cell normalised"
elif normalise_final_projection_to_0_1 and not rf_is_signed:
    cbar_norm_text = "final projection normalised"
else:
    cbar_norm_text = "unnormalised"

cbar.set_label(
    f"{cbar_prefix} {cbar_norm_text} {rf_colourbar_name}",
    fontsize=9,
)

if save_output:
    fig.savefig(
        output_path,
        dpi=save_dpi,
        bbox_inches="tight",
        transparent=True,
    )
    print(f"Saved figure to: {output_path}")

plt.show()

# %%
# %% ============================================================
# CELL 4 — PLOT TOP-N QUALITY CELLS FOR SELECTED CHANNEL
# ============================================================
# Run CELL 1, CELL 2 and CELL 3 first.
#
# This uses:
#   - channel_to_plot from CELL 2
#   - rf_dataset_to_plot from CELL 2
#   - the same left-right RF flip as CELL 3
#   - the same corrected stimulus scale as CELL 3
#   - the same cone -> projected stimulus transform as CELL 3
#   - the same RF -> projected stimulus transform as CELL 3
#
# Each panel is centred on the strongest visible RF pixel.


# ------------------------------------------------------------
# OPTIONS
# ------------------------------------------------------------

# Number of highest-quality cells to plot
top_n_quality_cells = 30

# Number of columns in the grid
top_quality_n_cols = 5

# Quality threshold before ranking.
#
# None = simply take the top X cells, regardless of absolute QI.
#
# For example, to only consider cells with QI >= 20:
# top_quality_min_qi = 20
#
# Or to use the same threshold as the automatic single-cell selection:
# top_quality_min_qi = single_cell_auto_quality_threshold
top_quality_min_qi = None

# Zoom around each cell's strongest RF pixel.
# Uses the same value as the single-cell plot by default.
top_quality_zoom_half_width_um = single_cell_zoom_half_width_um

# Show RF peak marker
top_quality_draw_peak = False

# Show coordinate ticks on every panel.
# False is usually much cleaner for 30 cells.
top_quality_show_axes = False

# Saving
top_quality_save = False

top_quality_output_path = Path(
    f"top_{top_n_quality_cells}_quality_cells_"
    f"{channel_to_plot}_{rf_dataset_to_plot}.png"
)


# ------------------------------------------------------------
# GET QUALITY FOR SELECTED CHANNEL
# ------------------------------------------------------------

if quality_variable_name not in dataset:
    raise KeyError(f"dataset['{quality_variable_name}'] was not found.")

quality_top = dataset[quality_variable_name]

if "channel" in quality_top.dims:
    quality_top = quality_top.sel(channel=channel_to_plot)

if "cell_index" not in quality_top.dims:
    raise ValueError(
        f"dataset['{quality_variable_name}'] does not contain "
        f"a cell_index dimension. Dims: {quality_top.dims}"
    )


# ------------------------------------------------------------
# GET RF DATA FOR SELECTED CHANNEL
# ------------------------------------------------------------

rf_all_top = dataset[rf_variable_name].sel(channel=channel_to_plot)

if "cell_index" not in rf_all_top.dims:
    raise ValueError(
        f"dataset['{rf_variable_name}'] does not contain " "a cell_index dimension."
    )


# ------------------------------------------------------------
# MAKE SURE QUALITY AND RF DATA CONTAIN THE SAME CELLS
# ------------------------------------------------------------

quality_top, rf_all_top = xr.align(
    quality_top,
    rf_all_top,
    join="inner",
)


# ------------------------------------------------------------
# REMOVE NON-FINITE QUALITY VALUES
# ------------------------------------------------------------

candidate_quality_top = quality_top.where(
    np.isfinite(quality_top),
    drop=True,
)


# ------------------------------------------------------------
# OPTIONAL QUALITY THRESHOLD
# ------------------------------------------------------------

if top_quality_min_qi is not None:
    candidate_quality_top = candidate_quality_top.where(
        candidate_quality_top >= top_quality_min_qi,
        drop=True,
    )


if candidate_quality_top.sizes["cell_index"] == 0:
    raise ValueError("No cells remain after applying the quality selection.")


# ------------------------------------------------------------
# SORT HIGHEST QUALITY -> LOWEST QUALITY
# ------------------------------------------------------------

sorted_quality_top = candidate_quality_top.sortby(
    candidate_quality_top,
    ascending=False,
)

n_available_top = sorted_quality_top.sizes["cell_index"]

n_plot_top = min(
    top_n_quality_cells,
    n_available_top,
)

if n_plot_top < top_n_quality_cells:
    print(
        f"Requested {top_n_quality_cells} cells, but only "
        f"{n_available_top} cells are available."
    )


top_cell_ids = [
    int(sorted_quality_top["cell_index"].isel(cell_index=i).item())
    for i in range(n_plot_top)
]


top_cell_quality = [
    float(sorted_quality_top.isel(cell_index=i).item()) for i in range(n_plot_top)
]


print("")
print("------------------------------------------------------------")
print("TOP QUALITY CELLS")
print("------------------------------------------------------------")
print("Channel:", channel_to_plot)
print("RF dataset:", rf_dataset_to_plot)
print("Number plotted:", n_plot_top)

if top_quality_min_qi is None:
    print("Minimum QI: none")
else:
    print("Minimum QI:", top_quality_min_qi)

print("")

for rank, (cell_id, q) in enumerate(
    zip(top_cell_ids, top_cell_quality),
    start=1,
):
    print(f"rank {rank:>2}: " f"cell {cell_id:>4}, " f"quality={q:.3f}")


# ------------------------------------------------------------
# CREATE GRID
# ------------------------------------------------------------

top_quality_n_rows = int(np.ceil(n_plot_top / top_quality_n_cols))

fig, axes = plt.subplots(
    nrows=top_quality_n_rows,
    ncols=top_quality_n_cols,
    figsize=(
        top_quality_n_cols * 3.2,
        top_quality_n_rows * 3.2,
    ),
    dpi=display_dpi,
    squeeze=False,
)

axes_flat = axes.flatten()


# ------------------------------------------------------------
# LOOP THROUGH TOP CELLS
# ------------------------------------------------------------

for rank_i, (
    ax,
    cell_id,
    cell_quality,
) in enumerate(
    zip(
        axes_flat,
        top_cell_ids,
        top_cell_quality,
    ),
    start=1,
):
    # --------------------------------------------------------
    # Get this cell's RF
    # --------------------------------------------------------

    rf_cell = rf_all_top.sel(cell_index=cell_id)

    # --------------------------------------------------------
    # SAME fixed left-right RF mirror as main plot
    # --------------------------------------------------------

    rf_cell = flip_rf_left_right(rf_cell)

    # --------------------------------------------------------
    # Per-cell normalisation
    # --------------------------------------------------------

    if normalise_each_cell:
        if rf_is_signed:
            cell_scale = np.abs(rf_cell).max(
                dim=["x", "y"],
                skipna=True,
            )

        else:
            cell_scale = rf_cell.max(
                dim=["x", "y"],
                skipna=True,
            )

        cell_scale_value = float(cell_scale.values)

        if np.isfinite(cell_scale_value) and cell_scale_value != 0:
            rf_cell = rf_cell / cell_scale_value

        rf_cell = rf_cell.fillna(0)

    # --------------------------------------------------------
    # Ensure y,x order for imshow
    # --------------------------------------------------------

    rf_cell = rf_cell.transpose(
        "y",
        "x",
    )

    rf_cell_values = rf_cell.values.copy()

    # --------------------------------------------------------
    # Apply SAME displayed-stimulus mask
    # --------------------------------------------------------

    if clip_rf_to_displayed_stimulus:
        rf_cell_values = np.where(
            valid_displayed_mask_yx,
            rf_cell_values,
            np.nan,
        )

    # --------------------------------------------------------
    # RMS final normalisation, if requested
    # --------------------------------------------------------

    if not rf_is_signed and normalise_final_projection_to_0_1:
        finite_values = rf_cell_values[np.isfinite(rf_cell_values)]

        if finite_values.size > 0:
            cell_max = np.nanmax(finite_values)

            if np.isfinite(cell_max) and cell_max > 0:
                rf_cell_values = rf_cell_values / cell_max

    # --------------------------------------------------------
    # Find strongest ACTUAL plotted RF pixel
    # --------------------------------------------------------

    if rf_is_signed:
        rf_for_peak = np.abs(rf_cell_values)
    else:
        rf_for_peak = rf_cell_values

    finite_peak = np.isfinite(rf_for_peak)

    if not np.any(finite_peak):
        ax.set_title(
            f"Rank {rank_i} | cell {cell_id}\n"
            f"QI={cell_quality:.2f}\n"
            "No finite RF",
            fontsize=8,
        )

        ax.axis("off")
        continue

    masked_peak = np.where(
        finite_peak,
        rf_for_peak,
        -np.inf,
    )

    peak_y_i, peak_x_i = np.unravel_index(
        np.argmax(masked_peak),
        masked_peak.shape,
    )

    # Convert RF pixel index to corrected physical coordinate
    peak_x_um = 0.5 * (
        rf_x_edges_corrected[peak_x_i] + rf_x_edges_corrected[peak_x_i + 1]
    )

    peak_y_um = 0.5 * (
        rf_y_edges_corrected[peak_y_i] + rf_y_edges_corrected[peak_y_i + 1]
    )

    # Same raw RF -> projected-stimulus transform
    peak_projected_um = transform_points(
        A_raw_rf_to_projected_um,
        np.array(
            [[peak_x_um, peak_y_um]],
            dtype=float,
        ),
    )[0]

    # --------------------------------------------------------
    # Colour limits — same logic as single-cell plot
    # --------------------------------------------------------

    finite_values = rf_cell_values[np.isfinite(rf_cell_values)]

    if rf_is_signed:
        if normalise_each_cell:
            cell_vmax = 1.0

        elif finite_values.size > 0:
            cell_vmax = float(
                np.nanpercentile(
                    np.abs(finite_values),
                    rf_vmax_percentile_if_not_normalised,
                )
            )

            if not np.isfinite(cell_vmax) or cell_vmax <= 0:
                cell_vmax = 1.0

        else:
            cell_vmax = 1.0

        cell_vmin = -cell_vmax

    else:
        if normalise_each_cell or normalise_final_projection_to_0_1:
            cell_vmin = 0.0
            cell_vmax = 1.0

        else:
            cell_vmin = 0.0

            cell_vmax = robust_vmax(
                rf_cell_values,
                percentile=(rf_vmax_percentile_if_not_normalised),
                fallback=1.0,
            )

    cell_norm = Normalize(
        vmin=cell_vmin,
        vmax=cell_vmax,
    )

    # --------------------------------------------------------
    # CONE MOSAIC — SAME transform as main plot
    # --------------------------------------------------------

    ax.imshow(
        cone_display,
        alpha=cone_alpha,
        extent=[
            0,
            w1,
            h1,
            0,
        ],
        origin="upper",
        transform=(mpl_image1_to_projected_um + ax.transData),
        zorder=2,
    )

    # --------------------------------------------------------
    # RF — SAME transform/origin/extent as main plot
    # --------------------------------------------------------

    ax.imshow(
        rf_cell_values,
        cmap=rf_cmap,
        norm=cell_norm,
        alpha=rf_alpha,
        origin="upper",
        extent=[
            rf_x_min_edge,
            rf_x_max_edge,
            rf_y_max_edge,
            rf_y_min_edge,
        ],
        transform=(mpl_rf_to_projected_um + ax.transData),
        interpolation="nearest",
        zorder=10,
    )

    # --------------------------------------------------------
    # Optional peak marker
    # --------------------------------------------------------

    if top_quality_draw_peak:
        ax.scatter(
            peak_projected_um[0],
            peak_projected_um[1],
            marker="x",
            s=50,
            linewidths=1.5,
            color="white",
            zorder=20,
        )

    # --------------------------------------------------------
    # Zoom around this cell's strongest RF pixel
    # --------------------------------------------------------

    peak_x_projected = peak_projected_um[0]

    peak_y_projected = peak_projected_um[1]

    ax.set_xlim(
        peak_x_projected - top_quality_zoom_half_width_um,
        peak_x_projected + top_quality_zoom_half_width_um,
    )

    # Inverted y-axis, same as main plot
    ax.set_ylim(
        peak_y_projected + top_quality_zoom_half_width_um,
        peak_y_projected - top_quality_zoom_half_width_um,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(
        f"Rank {rank_i} | cell {cell_id}\n" f"QI = {cell_quality:.2f}",
        fontsize=9,
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    if top_quality_show_axes:
        ax.set_xlabel(
            "x (µm)",
            fontsize=7,
        )

        ax.set_ylabel(
            "y (µm)",
            fontsize=7,
        )

        ax.tick_params(
            axis="both",
            labelsize=6,
        )

    else:
        ax.set_xticks([])
        ax.set_yticks([])


# ------------------------------------------------------------
# TURN OFF UNUSED PANELS
# ------------------------------------------------------------

for ax in axes_flat[n_plot_top:]:
    ax.axis("off")


# ------------------------------------------------------------
# OVERALL TITLE
# ------------------------------------------------------------

fig.suptitle(
    (
        f"Top {n_plot_top} quality cells\n"
        f"{recording_name} | {channel_to_plot} | "
        f"{rf_title_name}"
    ),
    fontsize=12,
)


plt.tight_layout(rect=[0, 0, 1, 0.96])


# ------------------------------------------------------------
# OPTIONAL SAVE
# ------------------------------------------------------------

if top_quality_save:
    fig.savefig(
        top_quality_output_path,
        dpi=save_dpi,
        bbox_inches="tight",
        transparent=True,
    )

    print(
        "Saved figure to:",
        top_quality_output_path,
    )


plt.show()

# %% ============================================================
# CELL 5 — PLOT ONE CELL ACROSS ALL CHANNELS SIDE BY SIDE
# ============================================================
# Run CELL 1 and CELL 3 at least once first so that the alignment,
# RF scale correction, coordinate transforms, etc. already exist.
#
# This automatically:
#   - finds however many channels exist
#   - selects one cell ID
#   - plots that cell for every channel side by side
#   - applies the same left-right RF flip as the main plot
#   - uses the same cone/RF alignment
#   - gives every panel exactly the same physical x/y limits
#   - uses one small shared colourbar outside the plots


# ------------------------------------------------------------
# OPTIONS
# ------------------------------------------------------------

# Cell to plot
cell_id_to_plot = 224

# Half-width of the displayed region around the RFs.
#
# 200 = each panel shows a 400 × 400 µm region
multi_channel_zoom_half_width_um = 200


# How to determine the centre of the SHARED crop.
#
# "mean_peak"
#     Find the strongest RF pixel in every channel and use
#     the mean x/y position as the centre of all panels.
#
# "first_channel_peak"
#     Use the strongest RF pixel from the first channel.
#
# "stimulus_centre"
#     Centre every panel on x=0, y=0.
#
multi_channel_shared_centre_mode = "mean_peak"


# Draw an X on the RF peak in each channel
multi_channel_draw_peak = False


# Show x/y coordinate axes
multi_channel_show_axes = True


# Save figure
multi_channel_save = False

multi_channel_output_path = Path(
    f"cell_{cell_id_to_plot}_all_channels_{rf_dataset_to_plot}.png"
)


# ------------------------------------------------------------
# CHOOSE RF DATASET
# ------------------------------------------------------------

if rf_dataset_to_plot == "rms":
    rf_variable_name_multi = "rms"
    rf_cmap_multi = "Reds"
    rf_is_signed_multi = False
    rf_title_multi = "RMS"

elif rf_dataset_to_plot == "cm_most_important":
    rf_variable_name_multi = "cm_most_important"
    rf_cmap_multi = "coolwarm"
    rf_is_signed_multi = True
    rf_title_multi = "covariance"

else:
    raise ValueError("rf_dataset_to_plot must be " "'rms' or 'cm_most_important'")


# ------------------------------------------------------------
# GET ALL CHANNELS
# ------------------------------------------------------------

if "channel" not in dataset.coords:
    raise ValueError("Dataset does not contain a 'channel' coordinate.")


channels_to_plot = list(dataset["channel"].values)

n_channels = len(channels_to_plot)


if n_channels == 0:
    raise ValueError("No channels were found in dataset.")


print("")
print("------------------------------------------------------------")
print("SINGLE CELL — ALL CHANNELS")
print("------------------------------------------------------------")
print("Cell:", cell_id_to_plot)
print("RF dataset:", rf_dataset_to_plot)
print("Number of channels:", n_channels)

print("")

for i, channel in enumerate(
    channels_to_plot,
    start=1,
):
    print(f"{i}: {channel}")


# ------------------------------------------------------------
# CHECK CELL EXISTS
# ------------------------------------------------------------

rf_all_channels = dataset[rf_variable_name_multi]


if "cell_index" not in rf_all_channels.dims:
    raise ValueError(
        f"dataset['{rf_variable_name_multi}'] " "does not have a cell_index dimension."
    )


available_cells_multi = rf_all_channels["cell_index"].values


if cell_id_to_plot not in available_cells_multi:
    raise ValueError(
        f"Cell {cell_id_to_plot} was not found.\n"
        f"Example available cells: "
        f"{available_cells_multi[:20]}"
    )


# ------------------------------------------------------------
# GET QUALITY VALUES
# ------------------------------------------------------------

quality_multi = None


if quality_variable_name in dataset:
    quality_multi = dataset[quality_variable_name]

    if "cell_index" not in quality_multi.dims:
        quality_multi = None


# ------------------------------------------------------------
# PREPARE RF FOR EACH CHANNEL
# ------------------------------------------------------------

prepared_rfs = {}

channel_peaks_projected_um = {}


for channel in channels_to_plot:
    # --------------------------------------------------------
    # Select RF
    # --------------------------------------------------------

    rf_cell = dataset[rf_variable_name_multi].sel(
        channel=channel,
        cell_index=cell_id_to_plot,
    )

    # --------------------------------------------------------
    # SAME LEFT-RIGHT RF FLIP AS MAIN SCRIPT
    # --------------------------------------------------------

    rf_cell = flip_rf_left_right(rf_cell)

    # --------------------------------------------------------
    # PER-CELL NORMALISATION
    # --------------------------------------------------------

    if normalise_each_cell:
        if rf_is_signed_multi:
            rf_scale = np.abs(rf_cell).max(
                dim=["x", "y"],
                skipna=True,
            )

        else:
            rf_scale = rf_cell.max(
                dim=["x", "y"],
                skipna=True,
            )

        rf_scale_value = float(rf_scale.values)

        if np.isfinite(rf_scale_value) and rf_scale_value != 0:
            rf_cell = rf_cell / rf_scale_value

        rf_cell = rf_cell.fillna(0)

    # --------------------------------------------------------
    # ENSURE y,x ORDER
    # --------------------------------------------------------

    rf_cell = rf_cell.transpose(
        "y",
        "x",
    )

    rf_values = rf_cell.values.copy()

    # --------------------------------------------------------
    # SAME DISPLAYED-STIMULUS MASK
    # --------------------------------------------------------

    if clip_rf_to_displayed_stimulus:
        rf_values = np.where(
            valid_displayed_mask_yx,
            rf_values,
            np.nan,
        )

    # --------------------------------------------------------
    # RMS FINAL NORMALISATION
    # --------------------------------------------------------

    if not rf_is_signed_multi and normalise_final_projection_to_0_1:
        finite_values = rf_values[np.isfinite(rf_values)]

        if finite_values.size > 0:
            final_max = np.nanmax(finite_values)

            if np.isfinite(final_max) and final_max > 0:
                rf_values = rf_values / final_max

    # Save RF
    prepared_rfs[channel] = rf_values

    # --------------------------------------------------------
    # FIND STRONGEST ACTUAL PLOTTED RF PIXEL
    # --------------------------------------------------------

    if rf_is_signed_multi:
        rf_for_peak = np.abs(rf_values)

    else:
        rf_for_peak = rf_values

    finite_peak = np.isfinite(rf_for_peak)

    if np.any(finite_peak):
        peak_search = np.where(
            finite_peak,
            rf_for_peak,
            -np.inf,
        )

        peak_y_i, peak_x_i = np.unravel_index(
            np.argmax(peak_search),
            peak_search.shape,
        )

        # ----------------------------------------------------
        # Convert RF pixel index into corrected RF coordinate
        # ----------------------------------------------------

        peak_x_um = 0.5 * (
            rf_x_edges_corrected[peak_x_i] + rf_x_edges_corrected[peak_x_i + 1]
        )

        peak_y_um = 0.5 * (
            rf_y_edges_corrected[peak_y_i] + rf_y_edges_corrected[peak_y_i + 1]
        )

        # ----------------------------------------------------
        # Convert to projected-stimulus coordinates
        # ----------------------------------------------------

        peak_projected_um = transform_points(
            A_raw_rf_to_projected_um,
            np.array(
                [
                    [
                        peak_x_um,
                        peak_y_um,
                    ]
                ],
                dtype=float,
            ),
        )[0]

        channel_peaks_projected_um[channel] = peak_projected_um

    else:
        channel_peaks_projected_um[channel] = np.array(
            [
                np.nan,
                np.nan,
            ]
        )


# ------------------------------------------------------------
# PRINT PEAK POSITIONS
# ------------------------------------------------------------

print("")
print("------------------------------------------------------------")
print("RF PEAK POSITIONS")
print("------------------------------------------------------------")


for channel in channels_to_plot:
    peak = channel_peaks_projected_um[channel]

    print(f"{channel}: " f"x={peak[0]:.2f} µm, " f"y={peak[1]:.2f} µm")


# ------------------------------------------------------------
# FIND ONE SHARED CROP CENTRE
# ------------------------------------------------------------

all_valid_peaks = np.array(
    [
        channel_peaks_projected_um[channel]
        for channel in channels_to_plot
        if np.all(np.isfinite(channel_peaks_projected_um[channel]))
    ]
)


if multi_channel_shared_centre_mode == "mean_peak":
    if len(all_valid_peaks) == 0:
        shared_centre_um = np.array(
            [
                0.0,
                0.0,
            ]
        )

    else:
        shared_centre_um = np.mean(
            all_valid_peaks,
            axis=0,
        )


elif multi_channel_shared_centre_mode == "first_channel_peak":
    first_channel = channels_to_plot[0]

    shared_centre_um = channel_peaks_projected_um[first_channel]

    if not np.all(np.isfinite(shared_centre_um)):
        shared_centre_um = np.array(
            [
                0.0,
                0.0,
            ]
        )


elif multi_channel_shared_centre_mode == "stimulus_centre":
    shared_centre_um = np.array(
        [
            0.0,
            0.0,
        ]
    )


else:
    raise ValueError(
        "multi_channel_shared_centre_mode must be "
        "'mean_peak', "
        "'first_channel_peak', "
        "or 'stimulus_centre'."
    )


shared_x_centre_um = float(shared_centre_um[0])

shared_y_centre_um = float(shared_centre_um[1])


print("")
print("------------------------------------------------------------")
print("SHARED DISPLAY CENTRE")
print("------------------------------------------------------------")

print(f"x = {shared_x_centre_um:.2f} µm")

print(f"y = {shared_y_centre_um:.2f} µm")


# ------------------------------------------------------------
# COMMON COLOUR SCALE FOR ALL CHANNELS
# ------------------------------------------------------------

if rf_is_signed_multi:
    if normalise_each_cell:
        common_vmin = -1.0
        common_vmax = 1.0

    else:
        finite_arrays = [
            values[np.isfinite(values)]
            for values in prepared_rfs.values()
            if np.any(np.isfinite(values))
        ]

        if len(finite_arrays) > 0:
            all_finite_values = np.concatenate(finite_arrays)

            common_vmax = float(
                np.nanpercentile(
                    np.abs(all_finite_values),
                    rf_vmax_percentile_if_not_normalised,
                )
            )

        else:
            common_vmax = 1.0

        if not np.isfinite(common_vmax) or common_vmax <= 0:
            common_vmax = 1.0

        common_vmin = -common_vmax


else:
    common_vmin = 0.0

    if normalise_each_cell or normalise_final_projection_to_0_1:
        common_vmax = 1.0

    else:
        finite_arrays = [
            values[np.isfinite(values)]
            for values in prepared_rfs.values()
            if np.any(np.isfinite(values))
        ]

        if len(finite_arrays) > 0:
            all_finite_values = np.concatenate(finite_arrays)

            common_vmax = robust_vmax(
                all_finite_values,
                percentile=(rf_vmax_percentile_if_not_normalised),
                fallback=1.0,
            )

        else:
            common_vmax = 1.0


common_norm = Normalize(
    vmin=common_vmin,
    vmax=common_vmax,
)


# ------------------------------------------------------------
# CREATE FIGURE
# ------------------------------------------------------------
#
# Slightly wider figure to leave space for the colourbar.
# ------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=1,
    ncols=n_channels,
    figsize=(
        4 * n_channels,
        4,
    ),
    dpi=display_dpi,
    squeeze=False,
)


axes = axes.flatten()


# ------------------------------------------------------------
# PLOT EACH CHANNEL
# ------------------------------------------------------------

for ax, channel in zip(
    axes,
    channels_to_plot,
):
    rf_values = prepared_rfs[channel]

    # --------------------------------------------------------
    # CONE MOSAIC
    # --------------------------------------------------------

    ax.imshow(
        cone_display,
        alpha=cone_alpha,
        extent=[
            0,
            w1,
            h1,
            0,
        ],
        origin="upper",
        transform=(mpl_image1_to_projected_um + ax.transData),
        zorder=2,
    )

    # --------------------------------------------------------
    # RF OVERLAY
    # --------------------------------------------------------

    rf_im_multi = ax.imshow(
        rf_values,
        cmap=rf_cmap_multi,
        norm=common_norm,
        alpha=rf_alpha,
        origin="upper",
        extent=[
            rf_x_min_edge,
            rf_x_max_edge,
            rf_y_max_edge,
            rf_y_min_edge,
        ],
        transform=(mpl_rf_to_projected_um + ax.transData),
        interpolation="nearest",
        zorder=10,
    )

    # --------------------------------------------------------
    # OPTIONAL RF PEAK MARKER
    # --------------------------------------------------------

    peak = channel_peaks_projected_um[channel]

    if multi_channel_draw_peak and np.all(np.isfinite(peak)):
        ax.scatter(
            peak[0],
            peak[1],
            marker="x",
            s=60,
            linewidths=1.7,
            color="white",
            zorder=20,
        )

    # --------------------------------------------------------
    # SAME PHYSICAL LIMITS FOR ALL CHANNELS
    # --------------------------------------------------------

    ax.set_xlim(
        shared_x_centre_um - multi_channel_zoom_half_width_um,
        shared_x_centre_um + multi_channel_zoom_half_width_um,
    )

    ax.set_ylim(
        shared_y_centre_um + multi_channel_zoom_half_width_um,
        shared_y_centre_um - multi_channel_zoom_half_width_um,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    # --------------------------------------------------------
    # QUALITY VALUE FOR THIS CHANNEL
    # --------------------------------------------------------

    quality_string = ""

    if quality_multi is not None:
        try:
            q = quality_multi.sel(cell_index=cell_id_to_plot)

            if "channel" in q.dims:
                q = q.sel(channel=channel)

            q_value = float(q.values)

            quality_string = f"\nQI = {q_value:.2f}"

        except Exception:
            quality_string = ""

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ax.set_title(
        f"{channel}" f"{quality_string}",
        fontsize=9,
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    if multi_channel_show_axes:
        ax.set_xlabel(
            "x (µm)",
            fontsize=8,
        )

        ax.set_ylabel(
            "y (µm)",
            fontsize=8,
        )

        ax.tick_params(
            axis="both",
            labelsize=7,
        )

    else:
        ax.set_xticks([])
        ax.set_yticks([])


# ------------------------------------------------------------
# OVERALL TITLE
# ------------------------------------------------------------

fig.suptitle(
    (f"Cell {cell_id_to_plot} — all channels\n" f"{recording_name} | {rf_title_multi}"),
    fontsize=12,
)


# ------------------------------------------------------------
# MANUALLY POSITION SUBPLOTS
# ------------------------------------------------------------
#
# right=0.90 leaves a dedicated empty strip for the colourbar,
# preventing it from overlapping the final RF panel.
# ------------------------------------------------------------

fig.subplots_adjust(
    left=0.06,
    right=0.90,
    bottom=0.14,
    top=0.78,
    wspace=0.22,
)


# ------------------------------------------------------------
# SMALL SHARED COLOURBAR
# ------------------------------------------------------------
#
# fig.add_axes takes:
#
#     [left, bottom, width, height]
#
# These are fractions of the whole figure.
#
# Therefore:
#     0.012 = thin colourbar
#     0.40  = relatively short colourbar
# ------------------------------------------------------------

cbar_ax = fig.add_axes(
    [
        0.92,  # left
        0.29,  # bottom
        0.010,  # width
        0.36,  # height
    ]
)


cbar = fig.colorbar(
    rf_im_multi,
    cax=cbar_ax,
)


cbar.ax.tick_params(labelsize=7)


if rf_is_signed_multi:
    cbar.set_label(
        "Covariance",
        fontsize=8,
    )

else:
    cbar.set_label(
        "RMS",
        fontsize=8,
    )


# ------------------------------------------------------------
# OPTIONAL SAVE
# ------------------------------------------------------------

if multi_channel_save:
    fig.savefig(
        multi_channel_output_path,
        dpi=save_dpi,
        bbox_inches="tight",
        transparent=True,
    )

    print(
        "Saved figure to:",
        multi_channel_output_path,
    )


plt.show()
