import numpy as np
from pathlib import Path
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import skimage.io

# %%
line_colours = ["red", "green"]
root = Path(rf"/media/mawa/52DE7A8FDE7A6B5D/noise_analysis/chicken_09_05_2024/kilosort4")
channels = ["4px_20Hz_shuffle_610nm_idx_10",
            "4px_20Hz_shuffle_535nm_idx_12"]
noise_names = ["610nm", "535nm"]
cell_idx = 486

rf_dict = {}
for channel in channels:
    rf_dict[channel] = {}
    save_root = root / channel / f"cell_{cell_idx}"
    rf_dict[channel]["s_nmf_contours"] = np.load(save_root / "s_nmf_contours.npy", allow_pickle=True)
    rf_dict[channel]["s_nmf_subunits"] = np.load(save_root / "s_nmf_subunits.npy", allow_pickle=True)
    rf_dict[channel]["snippet_mse"] = np.load(save_root / "snippet_mse.npy", allow_pickle=True)

# %%
red_mean = np.mean(rf_dict[channels[0]]["s_nmf_subunits"], axis=0)
green_mean = np.mean(rf_dict[channels[1]]["s_nmf_subunits"], axis=0)
# %%
correlated_channels = correlate2d(rf_dict[channels[0]]["snippet_mse"],
                                  rf_dict[channels[1]]["snippet_mse"],
                                  mode="full")
max_corr = np.array(np.unravel_index(np.argmax(correlated_channels), correlated_channels.shape)) - 60

# %%
red_mean_new = np.zeros((rf_dict[channel]["snippet_mse"].shape[0] + np.abs(max_corr[0]),
                         rf_dict[channel]["snippet_mse"].shape[1] + np.abs(max_corr[1])))
green_mean_new = np.zeros((rf_dict[channel]["snippet_mse"].shape[0] + np.abs(max_corr[0]),
                           rf_dict[channel]["snippet_mse"].shape[1] + np.abs(max_corr[1])))
# Recompute shift from correlation peak (assumes both means are same size)
h, w = red_mean.shape
y_peak, x_peak = np.unravel_index(np.argmax(correlated_channels), correlated_channels.shape)
shift_y = y_peak - (h - 1)
shift_x = x_peak - (w - 1)

# Determine padding needed so both fit after shifting red relative to green
pad_top = max(0, shift_y)
pad_left = max(0, shift_x)
pad_bottom = max(0, -shift_y)
pad_right = max(0, -shift_x)

H_new = h + pad_top + pad_bottom
W_new = w + pad_left + pad_right

red_mean_new = np.zeros((H_new, W_new), dtype=red_mean.dtype)
green_mean_new = np.zeros((H_new, W_new), dtype=green_mean.dtype)

# Place red (shift applied) and green (reference)
red_y0 = pad_top
red_x0 = pad_left
green_y0 = pad_bottom
green_x0 = pad_right

red_mean_new[red_y0:red_y0 + h, red_x0:red_x0 + w] = red_mean
green_mean_new[green_y0:green_y0 + h, green_x0:green_x0 + w] = green_mean

new_size = red_mean_new.shape
# %%
# Build RGB image: red region -> R+B, green region -> G+B, overlap (both strong) -> dark (near black)
eps = 1e-9
red_norm = red_mean_new / (np.max(red_mean_new) + eps)
green_norm = green_mean_new / (np.max(green_mean_new) + eps)

# Invert so higher activity is darker (for black overlap)
red_inv = 255 - (green_norm * 255)
green_inv = 255 - (red_norm * 255)
blue_inv = 255 - (0.5 * (red_norm + green_norm) * 255)

image = np.zeros((new_size[0], new_size[1], 3), dtype=np.uint8)
image[..., 0] = red_inv.astype(np.uint8)  # Red channel
image[..., 1] = green_inv.astype(np.uint8)  # Green channel
image[..., 2] = blue_inv.astype(np.uint8)  # Shared blue tint

overlay_metric = (red_mean_new - green_mean_new)
max_abs = np.max(np.abs(overlay_metric)) or 1.0
norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
cmap_overlay = LinearSegmentedColormap.from_list("magenta_black_green", ["#FF00FF", "#000000", "#00FF00"])
overlay_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_overlay)
# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(red_mean_new.reshape(new_size[0], new_size[1]), cmap="Reds_r")
ax[1].imshow(green_mean_new.reshape(new_size[0], new_size[1]), cmap="Greens_r")
ax[2].imshow(image)

fig.show()

# %%
upsampling_factor = 28
# import xarray as xr
# import datashader as ds
# import datashader.transfer_functions as tf
# from matplotlib import cm
#
# x = np.arange(0, 60)
# y = np.flipud(x)
#
# images_us = []
# for idx, ch in enumerate(channels):
#     arr = np.flipud(rf_dict[ch]["snippet_mse"])
#     mean_cm_xarr = xr.DataArray(arr, coords=[("y", y), ("x", x)])
#     max_cm = np.max(np.abs(arr))
#     cvs = ds.Canvas(plot_width=60 * upsampling_factor, plot_height=60 * upsampling_factor)
#     shade = tf.shade(
#         cvs.raster(mean_cm_xarr, interpolate="nearest", agg="max"),
#         how="linear",
#         cmap="black",
#         span=(-max_cm, max_cm),
#     )
#     images_us.append(shade.to_pil())

image_path = '/home/mawa/Downloads/central OD-1.tif'
from skimage.transform import rescale
from scipy.ndimage import zoom

cone_image = skimage.io.imread(image_path)
# interpolate up by a foctor of 1/12
# from scipy.ndimage import zoom
#
# cone_image = zoom(cone_image, (1 / 12.5, 1 / 12.5, 1), order=1)
#
# cone_image = rescale(cone_image, 1 / 12, channel_axis=2, preserve_range=True, anti_aliasing=True)
# cone_image = cone_image[:60, :60, :].astype(np.uint8)

# %%
# cone_image = zoom(cone_image, (8, 8, 1), order=1)
# cone_image = skimage.io.imread(image_path)[:-1, :-1, :]
# # %% fit the image into a 60*28 by 60*28 box
# image_complete = np.zeros((60*upsampling_factor, 60*upsampling_factor, 4), dtype=np.uint8)
# half = image_complete.shape[0] // 2
# image_complete[half-cone_image.shape[0]//2:half+cone_image.shape[0]//2,
#                half-cone_image.shape[1]//2:half+cone_image.shape[1]//2, :] = cone_image
# cone_image = image_complete
# %% contours
fig, ax = plt.subplots(figsize=(30, 10), nrows=2, ncols=6, dpi=300)
# ax[1, 0].imshow(cone_image, alpha=0.5)
for c_idx, channel in enumerate(channels):
    ax[0, c_idx].imshow(rf_dict[channel]["snippet_mse"], cmap="gray", vmin=0,
                        vmax=np.max(rf_dict[channel]["snippet_mse"]))
    for contour in rf_dict[channel]["s_nmf_contours"]:
        ax[0, c_idx].plot(contour[:, 1], contour[:, 0], linewidth=2, color=line_colours[c_idx], alpha=0.5)
        # ax[1, c_idx].plot((contour[:, 1] - 30) * upsampling_factor + cone_image.shape[1] // 2,
        #                   (contour[:, 0] - 30) * upsampling_factor + cone_image.shape[0] // 2, linewidth=2,
        #                   color=line_colours[c_idx], alpha=0.5)
    ax[0, c_idx].set_title(f"Subunits over STA, {noise_names[c_idx]}")
    # ax[c_idx].set_xlim(0, 60)
    # ax[c_idx].set_ylim(60, 0)
for contour in rf_dict[channels[0]]["s_nmf_contours"]:
    ax[0, 2].fill(contour[:, 1], contour[:, 0], linewidth=2, color=line_colours[0], alpha=0.5)
for contour in rf_dict[channels[1]]["s_nmf_contours"]:
    ax[0, 2].fill(contour[:, 1], contour[:, 0], linewidth=2, color=line_colours[1], alpha=0.5)
# flip y axis
ax[0, 2].set_ylim(60, 0)
ax[0, 2].set_xlim(0, 60)
ax[0, 2].set_title(f"Subunits overlayed")
ax[0, 3].set_title(f"Red nmf components")
ax[0, 4].set_title(f"Green nmf components")
ax[0, 5].set_title(f"Overlayed mean components")
ax[1, 0].set_title(f"Red subunits over cone mosaic")
ax[1, 1].set_title(f"Green subunits over cone mosaic")
ax[1, 0].set_ylim(0, cone_image.shape[0])
ax[1, 0].set_xlim(0, cone_image.shape[1])
ax[1, 1].set_ylim(0, cone_image.shape[0])
ax[1, 1].set_xlim(0, cone_image.shape[1])

ax[0, 3].imshow(red_mean_new.reshape(new_size[0], new_size[1]), cmap="Reds")
ax[0, 4].imshow(green_mean_new.reshape(new_size[0], new_size[1]), cmap="Greens")
ax[0, 5].imshow(image)

# ax[0, 5].imshow(red_mean_new, cmap="Reds")
# ax[0, 5].imshow(green_mean_new, cmap="Greens", alpha=0.5)

# erase unused subplots
for i in range(4):
    ax[1, i + 2].axis('off')

# add scale bar to the first subplot
scalebar = ScaleBar(3.0, "um", fixed_value=10)
ax[0, 0].add_artist(scalebar)

# remove all white backgrounds
for a in ax.flat: a.set_facecolor('none')
fig.patch.set_facecolor('none')

fig.show()
fig.savefig(fr"/media/mawa/52DE7A8FDE7A6B5D/noise_analysis/chicken_09_05_2024/subunit_overlay/cell_{cell_idx}.svg")

# %%
import plotly.graph_objs as go

fig = go.Figure()
fig.add_trace(go.Image(z=cone_image))
fig.update_layout(width=800, height=800)
fig.show(renderer="browser")
