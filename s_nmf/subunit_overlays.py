import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import einops
from IPython.core.pylabtools import figsize
from scipy.spatial.distance import cdist
from scipy.signal import correlate2d
from scipy.ndimage import shift
from skimage.measure import find_contours

# %%
root = Path(rf"/media/mawa/52DE7A8FDE7A6B5D/noise_analysis/chicken_09_05_2024/kilosort4")
channels = ["4px_20Hz_shuffle_610nm_idx_10",
            "4px_20Hz_shuffle_535nm_idx_12"]
cell_idx = 486
n_components = 30
# %%
rf_dict = {}
for channel in channels:
    rf_dict[channel] = {}
    rf_dict[channel]["reshaped_W"] = np.load(root / f"{channel}/cell_{cell_idx}/reshaped_W.npy")
    rf_dict[channel]["labels_2d"] = np.load(root / f"{channel}/cell_{cell_idx}/labels2d.npy")
    rf_dict[channel]["subset_dev_img"] = np.load(root / f"{channel}/cell_{cell_idx}/subset_dev_img.npy")
    labels_2d = einops.rearrange(rf_dict[channel]["labels_2d"], " r c -> (r c) ", r=20, c=n_components)
    rf_dict[channel]["clusters"] = np.zeros((np.max(labels_2d) + 1, 3600))
    rf_dict[channel]["all_clusters"] = np.zeros(3600)
    for cluster in range(np.max(labels_2d) + 1):
        mask = labels_2d == cluster
        temp = np.mean(
            rf_dict[channel]["reshaped_W"][mask, :], axis=0
        )
        rf_dict[channel]["clusters"][cluster] = temp / np.max(temp)
        temp = np.mean(rf_dict[channel]["reshaped_W"][labels_2d >= 0, :], axis=0)
        temp = temp / np.max(temp)
        temp[temp < 0.25] = 0
        rf_dict[channel]["all_clusters"] += temp

correlated_channels = correlate2d(rf_dict[channels[0]]["all_clusters"].reshape(60, 60),
                                  rf_dict[channels[1]]["all_clusters"].reshape(60, 60),
                                  mode="full")
max_corr = np.unravel_index(np.argmax(correlated_channels), correlated_channels.shape)
shift_y = max_corr[0] - (correlated_channels.shape[0] // 2)
shift_x = max_corr[1] - (correlated_channels.shape[1] // 2)

for channel in channels:
    rf_dict[channel]["contours"] = []
    for cluster in range(rf_dict[channel]["clusters"].shape[0]):
        temp = rf_dict[channel]["clusters"][cluster]
        temp = shift(temp.reshape(60, 60), [shift_y, shift_x], mode='constant', cval=0).reshape(3600)
        rf_dict[channel]["clusters"][cluster] = temp
        contours = find_contours(temp.reshape(60, 60) / np.max(temp), 0.4)
        if type(contours) is list and len(contours) > 0:
            contours = contours[0]
        rf_dict[channel]["contours"].append(contours)

    temp = shift(rf_dict[channel]["all_clusters"].reshape(60, 60), [shift_y, shift_x], mode='constant', cval=0).reshape(
        3600)
    rf_dict[channel]["all_clusters"] = temp

# %%

max_cluster = np.argsort([len(rf_dict[channels[0]]["clusters"]), len(rf_dict[channels[1]]["clusters"])])
cluster_distances = cdist(rf_dict[channels[max_cluster[1]]]["clusters"],
                          rf_dict[channels[max_cluster[0]]]["clusters"],
                          metric="cosine")

# %%
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.imshow(cluster_distances, cmap="viridis")
ax.set_xticks(np.arange(len(rf_dict[channels[0]]["clusters"])))
ax.set_yticks(np.arange(len(rf_dict[channels[1]]["clusters"])))
fig.show()

# %% find minimum distance between clusters
min_distance = np.argmin(cluster_distances, axis=1)

# %%
channel_cmaps = np.array(["Reds", "Greens"])
line_colours = np.array(["red", "green"])
line_colours = line_colours[max_cluster[::-1]]
channel_cmaps = [channel_cmaps[i] for i in max_cluster[::-1]]
contours_dev_img_red = find_contours(rf_dict[channels[0]]["subset_dev_img"][20:-20, 20:-20] / np.max(
    rf_dict[channels[0]]["subset_dev_img"]), 0.4)[0]
contours_dev_img_green = find_contours(rf_dict[channels[1]]["subset_dev_img"][20:-20, 20:-20] / np.max(
    rf_dict[channels[1]]["subset_dev_img"]), 0.4)[0]

fig, ax = plt.subplots(ncols=cluster_distances.shape[0], nrows=4, figsize=(20, 7), dpi=300, sharex=True, sharey=True)
# plot each cluster and the closest cluster from the other channel
for i in range(cluster_distances.shape[0]):
    ax[0, i].imshow(
        rf_dict[channels[max_cluster[1]]]["clusters"][i].reshape(60, 60),
        cmap=channel_cmaps[0],
        vmin=0,
        vmax=np.max(rf_dict[channels[max_cluster[1]]]["clusters"]),
    )

    closest_cluster = min_distance[i]
    ax[0, i].imshow(
        rf_dict[channels[max_cluster[0]]]["clusters"][closest_cluster].reshape(60, 60),
        cmap=channel_cmaps[1],
        vmin=0,
        vmax=np.max(rf_dict[channels[max_cluster[0]]]["clusters"]),
        alpha=0.5,
    )
    ax[0, i].imshow(
        rf_dict[channels[0]]["subset_dev_img"][20:-20, 20:-20],
        cmap="gray",
        alpha=0.2,
    )

    ax[1, i].imshow(
        rf_dict[channels[max_cluster[1]]]["clusters"][i].reshape(60, 60),
        cmap=channel_cmaps[0],
        vmin=0,
        vmax=np.max(rf_dict[channels[max_cluster[1]]]["clusters"]),
    )

    closest_cluster = min_distance[i]
    ax[1, i].imshow(
        rf_dict[channels[max_cluster[0]]]["clusters"][closest_cluster].reshape(60, 60),
        cmap=channel_cmaps[1],
        vmin=0,
        vmax=np.max(rf_dict[channels[max_cluster[0]]]["clusters"]),
        alpha=0.5,
    )
    ax[1, i].imshow(
        rf_dict[channels[1]]["subset_dev_img"][20:-20, 20:-20],
        cmap="gray",
        alpha=0.2,
    )
    ax[2, i].imshow(
        rf_dict[channels[max_cluster[1]]]["clusters"][i].reshape(60, 60),
        cmap=channel_cmaps[0],
        vmin=0,
        vmax=np.max(rf_dict[channels[max_cluster[1]]]["clusters"]),
    )

    closest_cluster = min_distance[i]
    ax[2, i].imshow(
        rf_dict[channels[max_cluster[0]]]["clusters"][closest_cluster].reshape(60, 60),
        cmap=channel_cmaps[1],
        vmin=0,
        vmax=np.max(rf_dict[channels[max_cluster[0]]]["clusters"]),
        alpha=0.5,
    )

ax[3, 0].imshow(
    rf_dict[channels[0]]["subset_dev_img"][20:-20, 20:-20],
    cmap=channel_cmaps[0],
    alpha=0.5,
)
ax[3, 1].imshow(
    rf_dict[channels[1]]["subset_dev_img"][20:-20, 20:-20],
    cmap=channel_cmaps[1],
    alpha=0.5,
)
# overlay the clusters on the image
ax[3, 2].imshow(
    rf_dict[channels[0]]["subset_dev_img"][20:-20, 20:-20],
    cmap=channel_cmaps[0],
    alpha=0.5,
)
ax[3, 2].imshow(
    rf_dict[channels[1]]["subset_dev_img"][20:-20, 20:-20],
    cmap=channel_cmaps[1],
    alpha=0.5,
)
ax[3, 3].imshow(
    rf_dict[channels[0]]["all_clusters"].reshape(60, 60),
    cmap=channel_cmaps[0],
    alpha=0.5,
)
ax[3, 3].imshow(
    rf_dict[channels[1]]["all_clusters"].reshape(60, 60),
    cmap=channel_cmaps[1],
    alpha=0.5,
)

# for c_idx, channel in enumerate(channels):
#     # draw contours
#     for cluster in range(rf_dict[channel]["clusters"].shape[0]):
#         contour = rf_dict[channel]["contours"][cluster]
#         try:
#             ax[3, 3].fill(contour[:, 1], contour[:, 0], color=line_colours[c_idx], alpha=0.5, linewidth=0)
#         except TypeError:
#             continue
# ax[3, 3].set_aspect('equal')

ax[3, 4].imshow(
    rf_dict[channels[0]]["subset_dev_img"][20:-20, 20:-20],
    cmap=channel_cmaps[0],
    alpha=0.5,
)
ax[3, 4].imshow(
    rf_dict[channels[1]]["subset_dev_img"][20:-20, 20:-20],
    cmap=channel_cmaps[1],
    alpha=0.5,
)
for c_idx, channel in enumerate(channels):
    # draw contours
    for cluster in range(rf_dict[channel]["clusters"].shape[0]):
        contour = rf_dict[channel]["contours"][cluster]
        try:
            ax[3, 4].fill(contour[:, 1], contour[:, 0], color=line_colours[c_idx], alpha=0.5, linewidth=0)
        except TypeError:
            continue
for ax in ax.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
# remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)
fig.show()
# %%
fig, ax = plt.subplots(figsize=(25, 5), ncols=5, dpi=300, sharey=True, sharex=True)
for c_idx, channel in enumerate(channels):
    # draw contours
    for cluster in range(rf_dict[channel]["clusters"].shape[0]):
        contour = rf_dict[channel]["contours"][cluster]
        try:
            ax[c_idx].fill(contour[:, 1], contour[:, 0], color=line_colours[c_idx], alpha=0.5, linewidth=0)
            if c_idx == 0:
                ax[2].fill(contour[:, 1], contour[:, 0], color=line_colours[c_idx], alpha=0.5, linewidth=0)
            else:
                ax[2].plot(contour[:, 1], contour[:, 0], color=line_colours[c_idx], alpha=1, linewidth=1)
        except TypeError:
            continue
arr0 = rf_dict[channels[0]]["all_clusters"].reshape(60, 60)
arr1 = rf_dict[channels[1]]["all_clusters"].reshape(60, 60)
arr0 = arr0 - np.min(arr0)
arr1 = arr1 - np.min(arr1)
arr0_norm = 1 - (arr0 / np.max(arr0) if np.max(arr0) != 0 else arr0)
arr1_norm = 1 - (arr1 / np.max(arr1) if np.max(arr1) != 0 else arr1)

composite = np.ones((60, 60, 3)) * np.mean((arr1_norm + arr0_norm) / 2)  # light gray background
composite[..., [0, 2]] = arr0_norm[:, :, np.newaxis]  # red where only channel 0
composite[..., 1] = arr1_norm  # green where only channel 1
# overlap (both high) remains black (0,0,0)

ax[3].imshow(composite)
ax[4].imshow(rf_dict[channels[0]]["subset_dev_img"][20:-20, 20:-20], cmap=channel_cmaps[0], alpha=1)
ax[4].imshow(rf_dict[channels[1]]["subset_dev_img"][20:-20, 20:-20], cmap=channel_cmaps[1], alpha=0.5)
fig.show()

# %%
fig, ax = plt.subplots(figsize=(10, 10), dpi=300, ncols=2)
ax[0].imshow(rf_dict[channels[0]]["subset_dev_img"][20:-20, 20:-20], cmap=channel_cmaps[0], alpha=1)
ax[1].imshow(rf_dict[channels[1]]["subset_dev_img"][20:-20, 20:-20], cmap=channel_cmaps[1], alpha=0.5)
for c_idx, channel in enumerate(channels):
    # draw contours
    for cluster in range(rf_dict[channel]["clusters"].shape[0]):
        contour = rf_dict[channel]["contours"][cluster]

        ax[c_idx].plot(contour[:, 1], contour[:, 0], color=line_colours[c_idx], alpha=1, linewidth=1)

fig.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.imshow(
    rf_dict[channels[0]]["all_clusters"].reshape(60, 60) / np.max(rf_dict[channels[0]]["all_clusters"]),
    cmap=channel_cmaps[0],
    alpha=0.5,
)
ax.imshow(
    rf_dict[channels[1]]["all_clusters"].reshape(60, 60) / np.max(rf_dict[channels[1]]["all_clusters"]),
    cmap=channel_cmaps[1],
    alpha=0.5,
)
fig.show()
# %%

test_cluster = rf_dict[channels[0]]["clusters"][2]
values, bins = np.histogram(test_cluster / np.max(test_cluster), bins=np.arange(0, 1.01, 0.01))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(bins[:-1], values, width=0.01)
fig.show()

# %% fit exponential
from scipy.optimize import curve_fit


def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


x_data = bins[:-1]
y_data = values
popt, pcov = curve_fit(exp_func, x_data, y_data, p0=(1, -1, 1))
# %% plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(bins[:-1], values, width=0.01, label="Data")
ax.plot(x_data, exp_func(x_data, *popt), color='red', label="Fitted curve")
ax.legend()
fig.show()
# %% estimate a threshold at which the exponential is lower than the data
threshold = x_data[np.where(exp_func(x_data, *popt) < values)[0]]
# %%
temp = test_cluster.copy()
temp[temp < threshold[1]] = 0
fig, ax = plt.subplots(figsize=(10, 10), ncols=2, dpi=300, )
ax[0].imshow(test_cluster.reshape(60, 60), cmap="Greys")
ax[1].imshow(temp.reshape(60, 60), cmap="Greys")
fig.show()

# %%
test_cluster = rf_dict[channels[0]]["clusters"][0].reshape(60, 60)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(test_cluster, cmap="Greys")
fig.show()
# %%
correlated = correlate2d(test_cluster, test_cluster, mode='full')
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(correlated, cmap="Greys")
fig.show()
# %%
from skimage.feature import match_template

normalized_corr = correlated / np.max(correlated)

# For normalized correlation that peaks at 1
normalized_corr = match_template(test_cluster, test_cluster, pad_input=True)
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(normalized_corr, cmap='viridis')
fig.colorbar(cax)
fig.show()
# %%
mean_corr = (np.max(normalized_corr, axis=0) + np.max(normalized_corr, axis=1)) / 2

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mean_corr)
fig.show()
