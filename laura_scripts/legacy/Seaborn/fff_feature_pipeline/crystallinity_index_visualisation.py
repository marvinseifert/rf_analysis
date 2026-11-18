# %% Visualise cone mosaics for RFs spanning cone crystallinity
#
# Loads multiple recordings, pools all cells, then selects:
#
#     least crystalline
#     + 5 cells evenly spaced through the sorted population
#     + most crystalline
#
# For each selected cell:
#
#     1. Finds its saved RF centre in projected µm coordinates
#     2. Uses that recording's alignment cache
#     3. Maps the RF centre onto the cone mosaic
#     4. Extracts an exact physical cone-mosaic crop
#     5. Displays the crop with its crystallinity index
#
# A second figure shows where the selected RFs lie on each
# full cone mosaic.


from pathlib import Path
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio.v3 as iio
import xarray as xr

from scipy.ndimage import map_coordinates


# %% ------------------------------------------------------------
# Recordings
# ------------------------------------------------------------
#
# Define your four recordings here.
#
# Each recording needs:
#
#     recording_id
#     FFF_DATA_PATH
#     CONE_IMAGE_PATH
#     ALIGNMENT_CACHE_PATH
#
# The FFF dataset must contain:
#
#     cone_regularity_index
#     cone_rf_center_x_um
#     cone_rf_center_y_um


RECORDINGS = [
    {
        "recording_id": "zebrafish_14_08_2026",
        "FFF_DATA_PATH": Path(
            r"F:\Laura\zebrafish_14_08_2026" r"\Phase_00\fff_analysis\fff_data.nc"
        ),
        "CONE_IMAGE_PATH": Path(
            r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish\Experiments\Imaging\20260814_dragonfly_microscope\14_08_2026_cones.tif"
        ),
        "ALIGNMENT_CACHE_PATH": Path(
            r"F:\Laura\zebrafish_14_08_2026\alignment_videos\zebrafish_14_08_2026_phase_00_alignment_cache.npz"
        ),
    },
    {
        "recording_id": "zebrafish_22_07_2026",
        "FFF_DATA_PATH": Path(
            r"F:\Laura\zebrafish_22_07_2026" r"\Phase_00\fff_analysis\fff_data.nc"
        ),
        "CONE_IMAGE_PATH": Path(
            r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish\Experiments\Imaging\20260722_dragonfly_microscope\22_07_2026_cones.jpg"
        ),
        "ALIGNMENT_CACHE_PATH": Path(
            r"F:\Laura\zebrafish_22_07_2026\alignment_videos\zebrafish_22_07_2026_phase_00_alignment_cache.npz"
        ),
    },
    {
        "recording_id": "zebrafish_15_05_2026",
        "FFF_DATA_PATH": Path(
            r"F:\Laura\zebrafish_15_05_2026" r"\Phase_00\fff_analysis\fff_data.nc"
        ),
        "CONE_IMAGE_PATH": Path(
            r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish\Experiments\Imaging\20260515_dragonfly_microscope\15_05_2026_cones.jpg"
        ),
        "ALIGNMENT_CACHE_PATH": Path(
            r"F:\Laura\zebrafish_15_05_2026\alignment_videos\zebrafish_15_05_2026_phase_00_alignment_cache.npz"
        ),
    },
    {
        "recording_id": "zebrafish_14_05_2026",
        "FFF_DATA_PATH": Path(
            r"F:\Laura\zebrafish_14_05_2026" r"\Phase_00\fff_analysis\fff_data.nc"
        ),
        "CONE_IMAGE_PATH": Path(
            r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish\Experiments\Imaging\20260514_dragonfly_microscope\14_05_2026_cones.jpg"
        ),
        "ALIGNMENT_CACHE_PATH": Path(
            r"F:\Laura\zebrafish_14_05_2026\alignment_videos\zebrafish_14_05_2026_phase_00_alignment_cache.npz"
        ),
    },
]


# %% ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

CRYSTALLINITY_VARIABLE = "cone_regularity_index"

RF_X_VARIABLE = "cone_rf_center_x_um"

RF_Y_VARIABLE = "cone_rf_center_y_um"


# Size of cone mosaic shown around each RF
#
# 200 means:
#
#     200 × 200 µm

CROP_SIZE_UM = 200.0


# Output resolution of displayed crop
#
# 1.0 means one output pixel = 1 µm

OUTPUT_UM_PER_PX = 1.0


# Number of intermediate cells

N_INTERMEDIATE = 6


# Show the RF centre as a cross

SHOW_RF_CENTRE = True


# Length of scale bar

SCALE_BAR_UM = 50


# %% ------------------------------------------------------------
# Convert cone image to intensity image
# ------------------------------------------------------------


def make_cone_intensity_image(
    image,
):
    """
    Convert the cone mosaic TIFF into a 2D intensity image.

    This follows the same basic convention used for the cone
    autocorrelation analysis:

        intensity = mean(red, blue)

    for an RGB image.

    Grayscale images are left as grayscale.
    """

    image = np.asarray(image)

    image = np.squeeze(image)

    # --------------------------------------------------------
    # Already grayscale
    # --------------------------------------------------------

    if image.ndim == 2:
        intensity = image.astype(float)

    # --------------------------------------------------------
    # RGB / RGBA with channels last
    # --------------------------------------------------------

    elif image.ndim == 3 and image.shape[-1] in [3, 4]:
        image = image.astype(float)

        red = image[..., 0]

        blue = image[..., 2]

        intensity = (red + blue) / 2

    # --------------------------------------------------------
    # Channels first
    # --------------------------------------------------------

    elif image.ndim == 3 and image.shape[0] in [3, 4]:
        image = np.moveaxis(
            image,
            0,
            -1,
        )

        image = image.astype(float)

        red = image[..., 0]

        blue = image[..., 2]

        intensity = (red + blue) / 2

    else:
        raise ValueError("Unexpected cone image shape: " f"{image.shape}")

    return intensity


# %% ------------------------------------------------------------
# Normalise image for display
# ------------------------------------------------------------


def normalise_for_display(
    image,
):
    image = np.asarray(
        image,
        dtype=float,
    ).copy()

    valid = np.isfinite(image)

    if not np.any(valid):
        return image

    low = np.nanpercentile(
        image,
        1,
    )

    high = np.nanpercentile(
        image,
        99.5,
    )

    if high <= low:
        return image

    image = (image - low) / (high - low)

    image = np.clip(
        image,
        0,
        1,
    )

    return image


# %% ------------------------------------------------------------
# Build projected-µm -> cone-image-pixel transformation
# ------------------------------------------------------------


def load_projected_um_to_cone_transform(
    alignment_cache_path,
):
    """
    Reconstruct the same coordinate chain used by the alignment
    pipeline.

    Alignment cache contains:

        Image 2 microscope pixels -> Image 3 pixels

    Image 1 (cone mosaic) shares the microscope coordinate system
    with Image 2.

    We construct:

        projected stimulus µm
                ->
        Image 3 pixels
                ->
        cone mosaic pixels
    """

    with np.load(
        alignment_cache_path,
        allow_pickle=True,
    ) as cache:
        tform_2_to_3 = np.asarray(
            cache["tform_2_to_3_params"],
            dtype=float,
        )

        um_per_px_image3 = float(np.asarray(cache["um_per_px_image3"]).squeeze())

        stim_edge_top_px = np.asarray(
            cache["stim_edge_top_px"],
            dtype=float,
        ).reshape(2)

        stim_edge_bottom_px = np.asarray(
            cache["stim_edge_bottom_px"],
            dtype=float,
        ).reshape(2)

        visual_stim_center_px_image3 = np.asarray(
            cache["visual_stim_center_px_image3"],
            dtype=float,
        ).reshape(2)

    # --------------------------------------------------------
    # Direction of stimulus DOWN in Image 3
    # --------------------------------------------------------

    edge_vector = stim_edge_bottom_px - stim_edge_top_px

    edge_length = np.linalg.norm(edge_vector)

    if edge_length == 0:
        raise ValueError("Clicked stimulus edge has zero length.")

    stim_down_unit_px = edge_vector / edge_length

    # --------------------------------------------------------
    # Stimulus RIGHT direction
    # --------------------------------------------------------

    stim_right_unit_px = np.array(
        [
            stim_down_unit_px[1],
            -stim_down_unit_px[0],
        ]
    )

    # --------------------------------------------------------
    # Pixels per µm in Image 3
    # --------------------------------------------------------

    px_per_um_image3 = 1.0 / um_per_px_image3

    # --------------------------------------------------------
    # projected µm -> Image 3 pixels
    # --------------------------------------------------------

    A_projected_um_to_image3 = np.eye(
        3,
        dtype=float,
    )

    A_projected_um_to_image3[:2, 0] = stim_right_unit_px * px_per_um_image3

    A_projected_um_to_image3[:2, 1] = stim_down_unit_px * px_per_um_image3

    A_projected_um_to_image3[:2, 2] = visual_stim_center_px_image3

    # --------------------------------------------------------
    # Image 3 -> cone image
    # --------------------------------------------------------
    #
    # tform_2_to_3:
    #
    #     cone/microscope pixels
    #                ->
    #          Image 3 pixels
    #
    # so invert it.

    A_image3_to_cone = np.linalg.inv(tform_2_to_3)

    # --------------------------------------------------------
    # projected µm -> cone image pixels
    # --------------------------------------------------------

    A_projected_um_to_cone = A_image3_to_cone @ A_projected_um_to_image3

    return A_projected_um_to_cone


# %% ------------------------------------------------------------
# Transform one projected coordinate to cone pixels
# ------------------------------------------------------------


def projected_um_to_cone_px(
    x_um,
    y_um,
    transform,
):
    point = np.array(
        [
            x_um,
            y_um,
            1.0,
        ],
        dtype=float,
    )

    transformed = transform @ point

    x_px = transformed[0] / transformed[2]

    y_px = transformed[1] / transformed[2]

    return (
        x_px,
        y_px,
    )


# %% ------------------------------------------------------------
# Extract exact physical cone crop
# ------------------------------------------------------------


def extract_cone_crop(
    cone_image,
    rf_x_um,
    rf_y_um,
    transform,
    crop_size_um,
    output_um_per_px,
):
    """
    Resample a square in projected physical coordinates.

    This means the displayed crop is exactly:

        crop_size_um × crop_size_um

    around the RF centre, even if the microscope image is rotated.
    """

    n_pixels = int(round(crop_size_um / output_um_per_px))

    if n_pixels < 2:
        raise ValueError("Crop is too small.")

    # --------------------------------------------------------
    # Relative physical coordinates
    # --------------------------------------------------------

    relative_positions = (
        np.arange(
            n_pixels,
            dtype=float,
        )
        - (n_pixels - 1) / 2
    ) * output_um_per_px

    x_um = rf_x_um + relative_positions

    y_um = rf_y_um + relative_positions

    grid_x_um, grid_y_um = np.meshgrid(
        x_um,
        y_um,
    )

    # --------------------------------------------------------
    # Convert every projected point to cone pixels
    # --------------------------------------------------------

    ones = np.ones_like(grid_x_um)

    projected_points = np.stack(
        [
            grid_x_um,
            grid_y_um,
            ones,
        ],
        axis=0,
    )

    projected_points = projected_points.reshape(
        3,
        -1,
    )

    cone_points = transform @ projected_points

    cone_x_px = cone_points[0] / cone_points[2]

    cone_y_px = cone_points[1] / cone_points[2]

    # --------------------------------------------------------
    # Sample microscope image
    # --------------------------------------------------------

    crop = map_coordinates(
        cone_image,
        [
            cone_y_px,
            cone_x_px,
        ],
        order=1,
        mode="constant",
        cval=np.nan,
    )

    crop = crop.reshape(
        n_pixels,
        n_pixels,
    )

    return crop


# %% ------------------------------------------------------------
# Load FFF data from all recordings
# ------------------------------------------------------------

all_cells = []


for recording in RECORDINGS:
    print("\n" + "=" * 70)

    print(f"Loading {recording['recording_id']}")

    print("=" * 70)

    fff_dataset = xr.load_dataset(recording["FFF_DATA_PATH"])

    # --------------------------------------------------------
    # Make sure required variables exist
    # --------------------------------------------------------

    required_variables = [
        CRYSTALLINITY_VARIABLE,
        RF_X_VARIABLE,
        RF_Y_VARIABLE,
    ]

    for variable in required_variables:
        if variable not in fff_dataset:
            raise KeyError(
                f"\n{variable!r} not found in:\n"
                f"{recording['FFF_DATA_PATH']}\n\n"
                f"Available variables:\n"
                f"{list(fff_dataset.data_vars)}"
            )

    # --------------------------------------------------------
    # Convert to dataframe
    # --------------------------------------------------------

    df = fff_dataset[required_variables].to_dataframe().reset_index()

    df = df.drop_duplicates(subset="cell_index").copy()

    # --------------------------------------------------------
    # Add recording information
    # --------------------------------------------------------

    df["recording_id"] = recording["recording_id"]

    df["cone_image_path"] = str(recording["CONE_IMAGE_PATH"])

    df["alignment_cache_path"] = str(recording["ALIGNMENT_CACHE_PATH"])

    df["global_cell_id"] = (
        df["recording_id"].astype(str) + "_cell_" + df["cell_index"].astype(str)
    )

    all_cells.append(df)

    fff_dataset.close()


# %% ------------------------------------------------------------
# Combine recordings
# ------------------------------------------------------------

combined_df = pd.concat(
    all_cells,
    ignore_index=True,
)


# ------------------------------------------------------------
# Require valid crystallinity and RF position
# ------------------------------------------------------------

combined_df = combined_df.dropna(
    subset=[
        CRYSTALLINITY_VARIABLE,
        RF_X_VARIABLE,
        RF_Y_VARIABLE,
    ]
).copy()


print(f"\nTotal valid cells across recordings: " f"{len(combined_df)}")


# %% ------------------------------------------------------------
# Sort cells from least -> most crystalline
# ------------------------------------------------------------

sorted_df = combined_df.sort_values(
    CRYSTALLINITY_VARIABLE,
    ascending=True,
).reset_index(drop=True)


# %% ------------------------------------------------------------
# Select least + 5 intermediate + most
# ------------------------------------------------------------

n_to_select = N_INTERMEDIATE + 2


if len(sorted_df) < n_to_select:
    raise ValueError(
        f"Only {len(sorted_df)} valid cells are available, "
        f"but {n_to_select} are required."
    )


# ------------------------------------------------------------
# Evenly spaced through sorted population
# ------------------------------------------------------------
# %% ------------------------------------------------------------
# Select least + 5 intermediate + most
# based on crystallinity VALUE
# ------------------------------------------------------------

n_to_select = N_INTERMEDIATE + 2

min_crystallinity = sorted_df[CRYSTALLINITY_VARIABLE].min()

max_crystallinity = sorted_df[CRYSTALLINITY_VARIABLE].max()


# Target crystallinity values evenly spaced
# between minimum and maximum

target_values = np.linspace(
    min_crystallinity,
    max_crystallinity,
    n_to_select,
)


selected_rows = []


for target in target_values:
    # Find cell whose crystallinity is closest
    # to the target value

    distance = np.abs(sorted_df[CRYSTALLINITY_VARIABLE] - target)

    closest_index = distance.idxmin()

    selected_rows.append(sorted_df.loc[closest_index])


selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

# %% ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

selection_labels = [
    "Least",
]


for i in range(N_INTERMEDIATE):
    selection_labels.append(f"Intermediate {i + 1}")


selection_labels.append("Most")


selected_df["selection"] = selection_labels


# %% ------------------------------------------------------------
# Print selected cells
# ------------------------------------------------------------

print("\n" + "=" * 90)

print("SELECTED CELLS")

print("=" * 90)


print(
    selected_df[
        [
            "selection",
            "recording_id",
            "cell_index",
            CRYSTALLINITY_VARIABLE,
            RF_X_VARIABLE,
            RF_Y_VARIABLE,
        ]
    ].to_string(index=False)
)


# %% ------------------------------------------------------------
# Load cone images and transforms
# ------------------------------------------------------------

cone_images = {}

alignment_transforms = {}


for recording in RECORDINGS:
    recording_id = recording["recording_id"]

    print(f"Loading cone image: {recording_id}")

    # --------------------------------------------------------
    # Load cone image
    # --------------------------------------------------------

    raw_image = iio.imread(recording["CONE_IMAGE_PATH"])

    cone_images[recording_id] = make_cone_intensity_image(raw_image)

    # --------------------------------------------------------
    # Load alignment transformation
    # --------------------------------------------------------

    alignment_transforms[recording_id] = load_projected_um_to_cone_transform(
        recording["ALIGNMENT_CACHE_PATH"]
    )
# %% ------------------------------------------------------------
# FIGURE 1
#
# Local cone mosaic surrounding each selected RF
# ------------------------------------------------------------

n_cells = len(selected_df)


fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(12, 9),
    constrained_layout=True,
)

axes = axes.flatten()

half_crop = CROP_SIZE_UM / 2


for ax, (_, row) in zip(
    axes,
    selected_df.iterrows(),
):
    recording_id = row["recording_id"]

    cell_index = int(row["cell_index"])

    crystallinity = float(row[CRYSTALLINITY_VARIABLE])

    rf_x_um = float(row[RF_X_VARIABLE])

    rf_y_um = float(row[RF_Y_VARIABLE])

    # --------------------------------------------------------
    # Extract cone mosaic around RF
    # --------------------------------------------------------

    crop = extract_cone_crop(
        cone_image=(cone_images[recording_id]),
        rf_x_um=rf_x_um,
        rf_y_um=rf_y_um,
        transform=(alignment_transforms[recording_id]),
        crop_size_um=(CROP_SIZE_UM),
        output_um_per_px=(OUTPUT_UM_PER_PX),
    )

    crop_display = normalise_for_display(crop)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    ax.imshow(
        crop_display,
        cmap="gray",
        origin="upper",
        extent=[
            -half_crop,
            half_crop,
            half_crop,
            -half_crop,
        ],
    )

    # --------------------------------------------------------
    # RF centre
    # --------------------------------------------------------

    if SHOW_RF_CENTRE:
        ax.scatter(
            0,
            0,
            marker="+",
            s=90,
            linewidths=2,
        )

    # --------------------------------------------------------
    # Scale bar
    # --------------------------------------------------------

    scale_y = half_crop - 12

    scale_x_start = -half_crop + 12

    scale_x_end = scale_x_start + SCALE_BAR_UM

    ax.plot(
        [
            scale_x_start,
            scale_x_end,
        ],
        [
            scale_y,
            scale_y,
        ],
        linewidth=3,
        color="red",
    )

    ax.text(
        (scale_x_start + scale_x_end) / 2,
        scale_y - 10,
        f"{SCALE_BAR_UM} µm",
        ha="center",
        va="top",
        fontsize=9,
        color="red",
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        f"{row['selection']}\n"
        f"{recording_id} | cell {cell_index}\n"
        f"crystallinity = {crystallinity:.3f}",
        fontsize=11,
    )

    ax.set_xlim(
        -half_crop,
        half_crop,
    )

    ax.set_ylim(
        half_crop,
        -half_crop,
    )

    ax.set_xticks([])
    ax.set_yticks([])


fig.suptitle(
    "Cone mosaics from least to most crystalline RF location",
    fontsize=16,
)


plt.show()


# %% ------------------------------------------------------------
# FIGURE 2
#
# Show where the selected RFs lie on each whole cone mosaic
# ------------------------------------------------------------

n_recordings = len(RECORDINGS)


fig, axes = plt.subplots(
    nrows=1,
    ncols=n_recordings,
    figsize=(
        6 * n_recordings,
        6,
    ),
    constrained_layout=True,
)


if n_recordings == 1:
    axes = [axes]


for ax, recording in zip(
    axes,
    RECORDINGS,
):
    recording_id = recording["recording_id"]

    cone_image = cone_images[recording_id]

    transform = alignment_transforms[recording_id]

    ax.imshow(
        normalise_for_display(cone_image),
        cmap="gray",
        origin="upper",
    )

    # --------------------------------------------------------
    # Selected cells from this recording
    # --------------------------------------------------------

    recording_selected = selected_df[selected_df["recording_id"] == recording_id]

    for _, row in recording_selected.iterrows():
        cone_x_px, cone_y_px = projected_um_to_cone_px(
            x_um=float(row[RF_X_VARIABLE]),
            y_um=float(row[RF_Y_VARIABLE]),
            transform=transform,
        )
        # --------------------------------------------------------
        # RF centre
        # --------------------------------------------------------

        rf_x_um = float(row[RF_X_VARIABLE])

        rf_y_um = float(row[RF_Y_VARIABLE])

        # --------------------------------------------------------
        # Define the four corners of the physical cone region
        # used to calculate crystallinity
        # --------------------------------------------------------

        half_crop = CROP_SIZE_UM / 2

        corners_um = [
            (
                rf_x_um - half_crop,
                rf_y_um - half_crop,
            ),
            (
                rf_x_um + half_crop,
                rf_y_um - half_crop,
            ),
            (
                rf_x_um + half_crop,
                rf_y_um + half_crop,
            ),
            (
                rf_x_um - half_crop,
                rf_y_um + half_crop,
            ),
        ]

        # --------------------------------------------------------
        # Transform corners onto cone image
        # --------------------------------------------------------

        corners_px = []

        for corner_x_um, corner_y_um in corners_um:
            corner_x_px, corner_y_px = projected_um_to_cone_px(
                x_um=corner_x_um,
                y_um=corner_y_um,
                transform=transform,
            )

            corners_px.append(
                [
                    corner_x_px,
                    corner_y_px,
                ]
            )

        corners_px = np.asarray(corners_px)

        # --------------------------------------------------------
        # Draw blue box around analysis region
        # --------------------------------------------------------

        box = Polygon(
            corners_px,
            closed=True,
            fill=False,
            edgecolor="blue",
            linewidth=3,
        )

        ax.add_patch(box)

        # --------------------------------------------------------
        # Mark exact RF centre
        # --------------------------------------------------------

        ax.scatter(
            cone_x_px,
            cone_y_px,
            s=100,
            marker="+",
            color="blue",
            linewidths=2.5,
        )

        # --------------------------------------------------------
        # Label
        # --------------------------------------------------------

        ax.text(
            cone_x_px + 15,
            cone_y_px,
            (f"cell {int(row['cell_index'])}\n" f"{row[CRYSTALLINITY_VARIABLE]:.2f}"),
            color="blue",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(recording_id)

    ax.set_xticks([])
    ax.set_yticks([])


fig.suptitle(
    "Location of selected RFs on each complete cone mosaic",
    fontsize=16,
)


plt.show()
