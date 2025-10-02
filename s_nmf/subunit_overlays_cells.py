import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import einops
from IPython.core.pylabtools import figsize
from scipy.spatial.distance import cdist
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from rf_torch.parameters import Cell_Params, Snippet_Params
from skimage.measure import find_contours

# %%
root = Path(rf"/media/mawa/52DE7A8FDE7A6B5D/noise_analysis/chicken_09_05_2024/kilosort4")
channels = ["4px_20Hz_shuffle_610nm_idx_10", "4px_20Hz_shuffle_535nm_idx_12"]  # "4px_20Hz_shuffle_535nm_idx_12"
cell_ids = [506, 573]  # [424, 419]  # 458
cut_size = 60
noise_shape = (600, 600)
rf_dict = {}
for channel in channels:
    rf_dict[channel] = {}
    quality = np.load(root / f"{channel}/quality.npy")
    for cell_idx in cell_ids:
        cell_data = {}
        cell_data["quality"] = quality[cell_idx, 0]
        cell_data["positions"] = quality[cell_idx, 2:4]
        x_start = max(0, cell_data["positions"][0] - cut_size // 2)
        y_start = max(0, cell_data["positions"][1] - cut_size // 2)
        x_end = min(noise_shape[0], x_start + cut_size)
        y_end = min(noise_shape[1], y_start + cut_size)
        cell_data["reshaped_W"] = np.load(root / f"{channel}/cell_{cell_idx}/reshaped_W.npy")
        cell_data["labels_2d"] = np.load(root / f"{channel}/cell_{cell_idx}/labels2d.npy")
        cell_data["subset_dev_img"] = np.load(root / f"{channel}/cell_{cell_idx}/subset_dev_img.npy")
        labels_2d = einops.rearrange(cell_data["labels_2d"], " r c -> (r c) ", r=20, c=10)
        n_time = cell_data["reshaped_W"].shape[1]
        cell_data["clusters"] = np.zeros((np.max(labels_2d) + 1, n_time))
        cell_data["contours"] = []
        cell_data["all_clusters"] = np.zeros((cut_size, cut_size))
        for cluster in range(np.max(labels_2d) + 1):
            mask = labels_2d == cluster
            cell_data["clusters"][cluster] = np.mean(cell_data["reshaped_W"][mask, :], axis=0)
            temp = np.mean(cell_data["reshaped_W"][mask, :], axis=0)
            temp = temp / np.max(temp)
            contours = find_contours(temp.reshape(cut_size, cut_size), 0.3)[0]
            contours[:, 0] = contours[:, 0] + x_start
            contours[:, 1] = contours[:, 1] + y_start

            cell_data["contours"].append(contours)
            cell_data["all_clusters"] += temp.reshape(cut_size, cut_size)

        rf_dict[channel][cell_idx] = cell_data

# %%

line_colours = ["red", "green", "blue"]
cmaps = ["Reds_r", "Greens_r", "Blues_r"]
fig = make_subplots(cols=len(channels), rows=1, shared_xaxes=True, shared_yaxes=True, )
images = np.zeros((len(channel), noise_shape[0], noise_shape[1], 3), dtype=np.uint8)
for channel_index, channel in enumerate(channels):
    for c_idx, cell_idx in enumerate(cell_ids):
        x_position_begin = int(rf_dict[channel][cell_idx]["positions"][1] - cut_size // 2)
        y_position_begin = int(rf_dict[channel][cell_idx]["positions"][0] - cut_size // 2)
        x_position_end = x_position_begin + cut_size
        y_position_end = y_position_begin + cut_size

        images[channel_index, y_position_begin:y_position_end, x_position_begin:x_position_end, c_idx + 1] = \
            rf_dict[channel][cell_idx]["all_clusters"] / np.max(
                rf_dict[channel][cell_idx]["all_clusters"]) * 255
        if cell_idx == 0:
            opacity = 1
        else:
            opacity = 1 / len(cell_ids)
        # fig.add_trace(go.Heatmap(
        #     z=temp,
        #     x=np.arange(x_position_begin, x_position_end),
        #     y=np.arange(y_position_begin, y_position_end),
        #     colorscale=cmaps[c_idx],
        #     zmin=0,
        #     zmax=1,
        #     showscale=False,
        #     hoverinfo='skip',
        #     opacity=opacity,
        #     name=f"Cell {cell_idx}",
        #     showlegend=True
        # ), row=1, col=channel_index + 1
        # )
        # for contour in rf_dict[channel][cell_idx]["contours"]:
        #     fig.add_trace(go.Scattergl(
        #         x=contour[:, 1],
        #         y=contour[:, 0],
        #         mode='lines',
        #         line=dict(color=line_colours[c_idx], width=2),
        #         showlegend=False
        #     ), row=1, col=channel_index + 1)
    fig.add_trace(go.Image(z=images[channel_index, :, :, :], opacity=1), row=1, col=channel_index + 1)
fig.update_xaxes(matches='x', range=(0, noise_shape[0]))
fig.update_yaxes(matches='y', autorange='reversed', range=(0, noise_shape[1]))

# change background color to black


# Enforce square pixels per subplot by anchoring each y-axis to its own x-axis
for i in range(1, len(channels) + 1):
    fig.update_yaxes(scaleanchor=f"x{i}", scaleratio=1, row=1, col=i)

fig.update_layout(width=1500, height=1500, title="Subunit Overlays for Different Cells", template="plotly_dark")
fig.show(renderer="browser")

# %%
fig = make_subplots(cols=2, rows=1, shared_xaxes=True, shared_yaxes=True, )
for channel_index, channel in enumerate(channels):
    for c_idx, cell_idx in enumerate(cell_ids):
        for contour in rf_dict[channel][cell_idx]["contours"]:
            fig.add_trace(go.Scattergl(
                x=contour[:, 1] * 2,
                y=contour[:, 0] * 2,
                mode='lines',
                line=dict(color="black", width=2),
                showlegend=False
            ), row=1, col=channel_index + 1)
fig.update_layout(width=1500, height=1000)
fig.show(renderer="browser")
# %%
for cluster in range(np.max(labels_2d) + 1):
    mask = labels_2d == cluster
    cell_data["clusters"][cluster] = np.mean(cell_data["reshaped_W"][mask, :], axis=0)
    temp = np.mean(cell_data["reshaped_W"][mask, :], axis=0)
    temp = temp / np.max(temp)
    contours = find_contours(temp.reshape(cut_size, cut_size), 0.3)[0]
    contours[:, 0] = contours[:, 0] + x_start
    contours[:, 1] = contours[:, 1] + y_start

    cell_data["contours"].append(contours)
    cell_data["all_clusters"] += temp.reshape(cut_size, cut_size)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(temp.reshape(60, 60))
ax.plot(contours[:, 1] - y_start, contours[:, 0] - x_start, color='black', linewidth=2)
fig.show()

# %% calculate 2d fourier transform of the STA
from scipy.fft import fft2, fftshift

test_image = rf_dict[channels[1]][cell_ids[0]]["all_clusters"]
f_transform = fftshift(fft2(test_image))
magnitude_spectrum = np.log(np.abs(f_transform) + 1)  # Log
phase_spectrum = np.angle(f_transform)
fig, ax = plt.subplots(1, 3, figsize=(20, 10))
ax[0].imshow(test_image, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(magnitude_spectrum, cmap='gray')
ax[1].set_title('Magnitude Spectrum (Log Scale)')
ax[2].imshow(phase_spectrum, cmap='gray')
ax[2].set_title('Phase Spectrum')
fig.show()
# %% plot only the central part of the magnitude spectrum
center = magnitude_spectrum.shape[0] // 2
trace = magnitude_spectrum[center, :]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(trace)
fig.show()
