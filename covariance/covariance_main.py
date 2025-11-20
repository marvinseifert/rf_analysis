# %%
"""
This script performs a covariance analysis on the STA of a single neuron. The following steps are performed:
1. Load the STA data.
2. Smooth the STA data.
3. Mask the STA data to only include a subset of pixels. This is done to reduce the computational complexity and RAM usage.
4. Perform a covariance analysis on the masked STA data.
5. Identify the most important pixel (in the covariance matrix) and plot the sums of the covariance matrix for this pixel.

"""

# %% Imports
import numpy as np
from pathlib import Path
from plotting.plots import plot_2d
from smoothing.gaussian import smooth_ker
from location.channel_handling import masking_square
from graph.networkx_graph import create_graph_from_covariance, nx_plot
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import datashader as ds
import datashader.transfer_functions as tf
import xarray as xr
from colorcet import coolwarm
from sklearn.decomposition import PCA
from PIL import Image
from scipy.signal import sosfiltfilt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorcet as cc
from scipy.signal import hilbert, butter
import matplotlib as mpl
from normalization.normalize_sta import zscore_sta as zscore
from covariance.filtering import cov_filtering_sum
from video.export_video import array_to_uncompressed_video

# %% Global variables
cell_id = 191
do_plot: bool = True  # Whether to plot the results.
do_video: bool = True  # Whether to create a video of the STA.
do_graph: bool = False  # Whether to plot the graph of the covariance matrix.
pixel_save: bool = False  # Whether to save the important pixels as numpy array.
cut: int = 150  # The size of the cutout around the centre of the STA in pixels.
half_cut: int = int(cut // 2)
pixel_size: int = (
    4  # The size of a pixel in micrometers. This is used for the scalebar.
)

# %% Load the data
project_root = Path(
    rf"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_13_11_2025/Phase_00/4px_20Hz_shuffle_led_535_idx_2"
)

data_path = project_root / rf"cell_{cell_id}/kernel.npy"
# data = np.load(
#     r"C:\Users\Marvin\Downloads\2025-5-14_1_4_SWN_2deg_RGBU_for_marvin_ROIs_0_10_52_56_59_64_101.npy"
# )
# sta_data = data[3, :, :, :]
sta_data = np.load(data_path)

ker_sm = smooth_ker(sta_data)

# sta_data = bandpass_filter(sta_data, 0.1, 40, 1000)
# sta_data = np.reshape(sta_data.T, (110, 16, 16))
sta_var = np.var(sta_data, axis=0)
original_sta_shape = sta_data.shape
# sta_data = sta_data[:, ::4, ::4]
data_flat = sta_data.reshape(
    original_sta_shape[0], original_sta_shape[1] * original_sta_shape[2]
)

sum_img = np.sum(data_flat, axis=0) / 100
median_sum = np.median(sum_img)
correction = sum_img - median_sum
for frame in range(data_flat.shape[0]):
    data_flat[frame, :] = data_flat[frame, :] - correction

sta_data = np.reshape(data_flat, original_sta_shape)

# %%

sta_extended = (
    np.full(
        (
            original_sta_shape[0],
            original_sta_shape[1] + cut,
            original_sta_shape[2] + cut,
        ), np.median(sta_data), dtype=float
    )
)

ker_sm_extended = (
    np.full(
        (
            ker_sm.shape[0],
            ker_sm.shape[1] + cut,
            ker_sm.shape[2] + cut,
        ), np.median(ker_sm), dtype=float
    )
)

sta_extended[:, half_cut:-half_cut, half_cut:-half_cut] = sta_data
ker_sm_extended[:, half_cut:-half_cut, half_cut:-half_cut] = ker_sm
sta_data = sta_extended
ker_sm = ker_sm_extended
del sta_extended, ker_sm_extended
original_sta_shape = sta_data.shape
sta_data = sta_data.astype(float)
var_extended = np.var(sta_data, axis=0)

# sta_data = smooth_ker(sta_data, axes=(0))


# %%
_, cY, cX = np.unravel_index(np.argmax(np.var(ker_sm, axis=0)), ker_sm.shape)
# The centre of the STA will be shifted in case the window is too close to the edge.
if cY > original_sta_shape[1] - int(cut / 2):
    cY = original_sta_shape[1] - int(cut / 2)
if cX > original_sta_shape[2] - int(cut / 2):
    cX = original_sta_shape[2] - int(cut / 2)
if cY < int(cut / 2):
    cY = int(cut / 2)
if cX < int(cut / 2):
    cX = int(cut / 2)
#

mask = masking_square(original_sta_shape[1], original_sta_shape[2], (cX, cY), cut, cut)
# mask[:] = True

# %% Plot the RFs centre

if do_plot:
    fig, ax = plot_2d(var_extended, pixel_size, title="Centre of STA variance")
    ax.plot(cX, cY, "ro")
    fig.show()

# %% Plot the smoothed version of the STA
if do_plot:
    fig, ax = plot_2d(
        np.var(ker_sm, axis=0), pixel_size, title="Centre of STA variance, smoothed"
    )
    # ax.plot(cX, cY, "ro")
    ax.imshow(mask, alpha=0.1)
    fig.show()

# %% Masking, taking subset and calculating the covariance matrix
subset_flat = sta_data[:, mask]  # sta_data.reshape(
#     (sta_data.shape[0], sta_data.shape[1] * sta_data.shape[2])
# )  #
# subset_flat_norm = np.zeros_like(subset_flat)
# for channel in range(subset_flat.shape[1]):
#     subset_flat_norm[:, channel] = zscore_sta(subset_flat[:, channel])
# subset_flat = subset_flat_norm

subset_flat[subset_flat == 0] = np.median(subset_flat)
subset = np.reshape(subset_flat, (original_sta_shape[0], cut, cut))
subset_var = np.var(subset, axis=0)
if do_plot:
    fig, ax = plot_2d(
        np.var(subset, axis=0), pixel_size, title="Centre of STA variance"
    )
    fig.show()
try:
    del cm
except NameError:
    pass
important_pixels, cm = cov_filtering_sum(subset, (cut, cut))
mean_cm = np.mean(cm)
std_cm = np.std(cm)

# %% Get the important pixel and calculate the covariance with all other pixels for this pixel
important_kernel = subset[
    :,
    np.unravel_index(np.argmax(important_pixels), subset.shape)[1],
    np.unravel_index(np.argmax(important_pixels), subset.shape)[2],
]

sta_data_flat = np.reshape(sta_data, (original_sta_shape[0], -1))

covariances = np.zeros(sta_data_flat.shape[1])
for pixel in range(sta_data_flat.shape[1]):
    covariances[pixel] = np.cov(sta_data_flat[:, pixel], important_kernel)[0, 1]

covariances_re = np.reshape(covariances, sta_data.shape[1:])

cov_subset_pos = np.quantile(covariances, 0.95)
cov_subset_neg = np.quantile(covariances, 0.05)
cov_for_plot_pos = np.copy(covariances_re)
cov_for_plot_neg = np.copy(covariances_re)
cov_for_plot_pos[covariances_re < cov_subset_pos] = np.NaN
cov_for_plot_neg[covariances_re > cov_subset_neg] = np.NaN
max_abs_cov = np.max(np.abs(covariances_re))

if do_plot:
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    im = ax.imshow(cov_for_plot_pos, cmap="Reds", vmin=0, vmax=max_abs_cov)
    plt.colorbar(im, ax=ax, shrink=0.5, label="Covariance")
    scalebar = ScaleBar(pixel_size, "um", fixed_value=100)
    ax.add_artist(scalebar)
    im = ax.imshow(cov_for_plot_neg, cmap="Blues_r", vmax=0, vmin=-max_abs_cov)
    plt.colorbar(im, ax=ax, shrink=0.5, label="Covariance")
    scalebar = ScaleBar(pixel_size, "um", fixed_value=100)
    ax.add_artist(scalebar)
    ax.set_title(
        "Covariance with the most important pixel, pixels in the 95 percentile"
    )
    fig.show()

# %% plot only the masked area
masked_cov_for_pos = cov_for_plot_pos[mask]
masked_cov_for_neg = cov_for_plot_neg[mask]
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
im = ax.imshow(
    masked_cov_for_pos.reshape((cut, cut)), cmap="Reds", vmin=0, vmax=max_abs_cov, alpha=1
)
plt.colorbar(im, ax=ax, shrink=0.5, label="Covariance")
im = ax.imshow(
    masked_cov_for_neg.reshape((cut, cut)), cmap="Blues_r", vmax=0, vmin=-max_abs_cov, alpha=1
)
plt.colorbar(im, ax=ax, shrink=0.5, label="Covariance")
scalebar = ScaleBar(pixel_size, "um", fixed_value=100)
ax.add_artist(scalebar)
fig.show()

# %%

# %% get the subset of the original data according to the mask
subset_cov_mask = np.isnan(masked_cov_for_pos) & np.isnan(masked_cov_for_neg)
# flip the mask to get the pixels which are not in the subset
subset_cov_mask = np.logical_not(subset_cov_mask)
sta_data_subset = subset_flat[:, subset_cov_mask]
mask_indices = np.where(subset_cov_mask.flatten())[0]

# %% Get the important pixels of the covariance matrix
cm_threshold: int = np.quantile(np.abs(important_pixels), 0.90)
# cm_threshold = int(
#     5e6
# )  # The threshold for when to consider a pixel important. This value is the maximal observed value for
# Get the x and y positions of important_sums and important_mins
important_x_pos, important_y_pos = np.where(important_pixels > cm_threshold)
important_pos_cm_idx = np.where(important_pixels.flatten() > cm_threshold)[0]
important_pos_cm = important_pixels.flatten()[important_pixels.flatten() > cm_threshold]
important_pos_cm_idx = important_pos_cm_idx[np.argsort(important_pos_cm)[::-1]]
important_pos_values = important_pixels[important_pixels > cm_threshold]
important_x_neg, important_y_neg = np.where(important_pixels < -cm_threshold / 2)
important_neg_cm_idx = np.where(important_pixels.flatten() < -cm_threshold / 2)[0]
important_neg_cm = important_pixels.flatten()[
    important_pixels.flatten() < -cm_threshold / 2
    ]
important_neg_cm_idx = important_neg_cm_idx[np.argsort(important_neg_cm)[::-1]]
important_neg_values = important_pixels[important_pixels < -cm_threshold / 2]

important_x_pos = important_x_pos[np.argsort(important_pos_values)[::-1]]
important_y_pos = important_y_pos[np.argsort(important_pos_values)[::-1]]
important_x_neg = important_x_neg[np.argsort(important_neg_values)[::-1]]
important_y_neg = important_y_neg[np.argsort(important_neg_values)[::-1]]

most_important_pixel = np.unravel_index(np.argmax(cm), cm.shape)
# %% plot histogram of the important pixels, serparate positive and negative
if do_plot:
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].hist(
        important_pos_cm,
        bins=100,
        color="purple",
        alpha=0.5,
        label="Positive covariance",
    )
    ax[1].hist(
        important_neg_cm,
        bins=100,
        color="blue",
        alpha=0.5,
        label="Negative covariance",
    )
    ax[0].set_title("Histogram of important pixels")
    ax[0].set_xlabel("Covariance")
    ax[0].set_ylabel("Count")
    ax[0].legend()
    fig.show()

# %% Plot the sum of covariance matrix for the important pixels
cm_subset = cm[most_important_pixel[0], :]
cm_most_important = cm_subset.reshape((cut, cut))
# plot the mean as image
if do_plot:
    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    x = y = np.arange(0, cut)
    mean_cm_xarr = xr.DataArray(cm_most_important, coords=[("x", x), ("y", y)])
    max_cm = np.max(np.abs(cm_most_important))
    # tf.shade(cvs.raster(da2))
    cvs = ds.Canvas(plot_width=600 - 1, plot_height=600 - 1)
    shade = tf.shade(
        cvs.raster(mean_cm_xarr, interpolate="nearest", agg="max"),
        how="linear",
        cmap=coolwarm,
        span=(-max_cm, max_cm),
    )
    img_array_linear = shade.to_pil()
    artist = axs[0].imshow(img_array_linear, origin="lower")
    # add colorbar with 0 in the middle
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # recreate the normalization & colormap you used in tf.shade
    norm = mpl.colors.Normalize(vmin=-max_cm, vmax=max_cm)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    sm.set_array([])  # dummy; needed for colorbar machinery

    cbar = fig.colorbar(sm, cax=cax,
                        orientation="vertical",
                        ticks=[-max_cm, 0, +max_cm])
    axs[0].set_title("Most important pixel in linear scale")
    # Plot the most important pixel in log scale.

    x = y = np.arange(0, cut)
    mean_cm_xarr = xr.DataArray(cm_most_important, coords=[("x", x), ("y", y)])
    max_cm = np.max(np.abs(cm_most_important))
    # tf.shade(cvs.raster(da2))
    cvs = ds.Canvas(plot_width=600 - 1, plot_height=600 - 1)
    shade = tf.shade(
        cvs.raster(-mean_cm_xarr, interpolate="nearest"),
        how="log",
        cmap="black",
    )
    img_array_log = shade.to_pil()
    im = axs[1].imshow(img_array_log, origin="lower", cmap="Greys")
    axs[1].set_title("Most important pixel in log scale")
    # Add colorbar with 0 in the middle
    # add scalebar

    scalebar = ScaleBar(pixel_size, "um", fixed_value=100)
    axs[0].add_artist(scalebar)
    fig.show()
# %% Old code
#     fig, ax = plt.subplots(1, 1, figsize=(20, 10))
#     im = ax.imshow(mean_cm_subset, cmap="seismic", norm=colors.CenteredNorm(0))
#     plt.colorbar(im, ax=ax, shrink=0.5, label="Sum of covariance")
#     scalebar = ScaleBar(pixel_size, "um", fixed_value=100)
#     ax.add_artist(scalebar)
#
#     # Add colorbar with 0 in the middle
#
#     fig.show()
#
# # %% Plot only the 95 percentile
# cm_subset_pos = np.quantile(mean_cm_subset, 0.95)
# cm_subset_neg = np.quantile(mean_cm_subset, 0.05)
# mean_cm_subset_pos = np.copy(mean_cm_subset)
# mean_cm_subset_neg = np.copy(mean_cm_subset)
# mean_cm_subset_pos[mean_cm_subset < cm_subset_pos] = np.NaN
# mean_cm_subset_neg[mean_cm_subset > cm_subset_neg] = np.NaN
#
# if do_plot:
#     fig, ax = plt.subplots(1, 1, figsize=(20, 10))
#     im = ax.imshow(mean_cm_subset_pos, cmap="Reds", norm=colors.CenteredNorm(0))
#     plt.colorbar(im, ax=ax, shrink=0.5, label="Sum of covariance")
#     scalebar = ScaleBar(pixel_size, "um", fixed_value=100)
#     ax.add_artist(scalebar)
#     im = ax.imshow(mean_cm_subset_neg, cmap="Blues_r", norm=colors.CenteredNorm(0))
#     plt.colorbar(im, ax=ax, shrink=0.5, label="Sum of covariance")
#     scalebar = ScaleBar(pixel_size, "um", fixed_value=100)
#     ax.add_artist(scalebar)
#
#     # Add colorbar with 0 in the middle
#
#     fig.show()


# %% Network x graph
# The covariance matrix can be represented as a graph. Here, we will plot the graph.
if do_graph:
    important_pixel_cm = np.concatenate(
        [important_pos_cm_idx[:100], important_neg_cm_idx[:100]]
    )
    important_x_graph = np.concatenate([important_x_pos[:100], important_x_neg[:100]])
    important_y_graph = np.concatenate([important_y_pos[:100], important_y_neg[:100]])
    cm_subset = cm[:, important_pixel_cm]
    cm_subset = cm_subset[important_pixel_cm, :]

    G = create_graph_from_covariance(cm_subset)
    graph_ds = nx_plot(G, "Graph", x=important_x_graph, y=important_y_graph)

    ds_img = tf.Image(graph_ds[2]).to_pil()
    ds_img = ds_img.transpose(Image.ROTATE_270)

    fig, axs = plt.subplots(1, 2, figsize=(40, 20))
    axs[0].imshow(img_array_linear)
    # Flip the y axis
    axs[0].invert_yaxis()
    axs[1].imshow(ds_img)
    fig.show()

# %% Plot the important pixels
# Here, we can look at the subset in more detail.
important_pixel_threshold = 1000
if do_plot:
    fig, ax = plot_2d(subset_var, pixel_size, title="Important pixels", plot_scalebar=False)

    # Calculate marker size based on plotting area and number of pixels
    # Get the axes size in points (default fig size is in inches, need to convert)
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_inches, height_inches = bbox.width, bbox.height
    # Convert to points (1 inch = 72 points)
    width_points = width_inches * 72
    height_points = height_inches * 72
    # Calculate area per pixel
    area_per_pixel = (width_points * height_points) / (cut * cut)
    # Marker size s is in points^2, so use the calculated area
    marker_size = area_per_pixel * 0.05  # Use 0.8 to leave small gaps between markers

    im1 = ax.scatter(
        important_y_pos[:important_pixel_threshold],
        important_x_pos[:important_pixel_threshold],
        c=important_pixels[
            important_x_pos[:important_pixel_threshold],
            important_y_pos[:important_pixel_threshold],
        ],
        cmap="Reds",
        vmin=0,
        vmax=np.max(important_pixels[:important_pixel_threshold]),
        marker="s",
        s=marker_size,
        alpha=0.8,
        edgecolors=None

    )

    im2 = ax.scatter(
        important_y_neg[:important_pixel_threshold],
        important_x_neg[:important_pixel_threshold],
        c=important_pixels[
            important_x_neg[:important_pixel_threshold],
            important_y_neg[:important_pixel_threshold],
        ],
        cmap="Blues_r",
        vmin=np.min(important_pixels[:important_pixel_threshold]),
        vmax=0,
        marker="s",
        s=marker_size,
        alpha=0.8,
        edgecolors=None
    )

    # Make the colorbar smaller
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im1, cax=cax1)
    cax2 = divider.append_axes("right", size="5%", pad=0.5)
    cbar2 = plt.colorbar(im2, cax=cax2, label="Covariance")

    fig.show()

# %% PLot only the important pixels as scatter plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(
    important_y_pos[:important_pixel_threshold],
    important_x_pos[:important_pixel_threshold],
    c=important_pixels[
        important_x_pos[:important_pixel_threshold],
        important_y_pos[:important_pixel_threshold],
    ],
    cmap="Reds",
    vmin=0,
    vmax=np.max(important_pixels[:important_pixel_threshold]),
    marker="s",
    s=10,
)
# flip the y axis
ax.invert_yaxis()
# ax.scatter(
#     important_y_neg[:important_pixel_threshold],
#     important_x_neg[:important_pixel_threshold],
#     c=important_pixels[
#         important_x_neg[:important_pixel_threshold],
#         important_y_neg[:important_pixel_threshold],
#     ],
#     cmap="Blues_r",
#     vmin=np.min(important_pixels[:important_pixel_threshold]),
#     vmax=0,
#     marker="s",
#     s=10,
# )
# Make the colorbar smaller
fig.show()

# # %%
# # save important pixels as numpy array
# combined_important_pixels = np.stack(
#     [
#         important_x_pos[:important_pixel_threshold],
#         important_y_pos[:important_pixel_threshold],
#     ]
# )
# np.savetxt(
#     project_root / rf"cell_{cell_id}\all_important_pixels.txt",
#     combined_important_pixels,
#     fmt="%d",
#     delimiter=",",
# )
# %% Plot the traces
# create matrix of indices equal to the size of the subset
subset_true = np.ones((subset.shape[1], subset.shape[2]), dtype=bool)
subset_true[important_x_pos, important_y_pos] = False
if do_plot:

    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    for i in range(important_x_pos[:important_pixel_threshold].shape[0]):
        ax[0].plot(subset[:, important_x_pos[i], important_y_pos[i]], color="red")
    for i in range(important_x_neg[:important_pixel_threshold].shape[0]):
        ax[0].plot(subset[:, important_x_neg[i], important_y_neg[i]], color="blue")
    # Plot the mean in the second subplot
    if not important_x_pos.size == 0:
        ax[1].plot(
            zscore(
                np.mean(
                    subset[
                        :,
                        important_x_pos[:important_pixel_threshold],
                        important_y_pos[:important_pixel_threshold],
                    ],
                    axis=1,
                )
            ),
            color="red",
        )

    if not important_x_neg.size == 0:
        ax[1].plot(
            zscore(
                np.mean(
                    subset[
                        :,
                        important_x_neg[:important_pixel_threshold],
                        important_y_neg[:important_pixel_threshold],
                    ],
                    axis=1,
                )
            ),
            color="blue",
        )
        # plot the pixel which are not important
    ax[1].plot(
        zscore(
            np.mean(
                subset[
                    :,
                    subset_true,
                ],
                axis=1,
            )
        ),
        color="black",
    )
    # Add axes
    ax[1].set_xlabel("Time in ms")
    ax[1].set_ylabel("Amplitude (z-score)")
    # ax[1].vlines(50, -1, 1, color="black", linestyle="--", label="Spike")
    # Remove top y box and right x box
    for i in range(2):
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
    # Remove ticks of the first subplot
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # Set the ticks of the second subplot
    # ax[1].set_xticks(
    #     np.array([0, 10, 20, 30, 40, 50]), ["-500", "-400", "-300", "-200", "-100", "0"]
    # )
    fig.show()

# %% Export sta mp4
# The sta can be exported as a video here, this might take a while.
if do_video:
    array_to_uncompressed_video(subset, out_path=str(data_path.parent / "sta_video.mkv"), fps=10)
