import h5py
import hdf5plugin
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import rotate, gaussian_filter
from sympy.printing.pretty.pretty_symbology import line_width
from skimage.feature import (
    match_descriptors,
    ORB,
    plot_matched_features,
    SIFT,
    match_template,
)
from skimage.registration import phase_cross_correlation
from scipy.signal import find_peaks


# %% Function to convert video to numpy array
def video_to_numpy(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Optional: Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return np.array(frames)


# %% Import video of MEA array under visual stimuli projection and generate and save mean image from video
# import video and convert to numpy array using function
video_file = rf"F:\alignment\WIN_20260414_16_46_04_Pro.mp4"
video_data = video_to_numpy(video_file)
video_size = video_data.shape  # (number of frames, height, width, channels)

# generate mean image
mean_image = np.mean(video_data, axis=0)
mean_image = np.max(
    mean_image, axis=2
)  # Convert to grayscale by taking the max across color channels
mean_image = (mean_image / np.max(mean_image) * 255).astype(np.uint8)

# write mean image back to file
cv2.imwrite(
    rf"F:\alignment\mean_image.png",
    mean_image,
)
# display mean image
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(255 - mean_image, cmap="gray")
fig.show()

# %% Proof of concept for autocorrelation pipeline, using reference image of electrode array
# Load a reference electrode array image called channel.png
channel_file = rf"F:\alignment\New folder\channel.png"
channel_image = cv2.imread(channel_file, cv2.IMREAD_GRAYSCALE)
channel_image = (channel_image / np.max(channel_image) * 255).astype(np.uint8)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    rotate(channel_image, 10), cmap="gray"
)  # rotation just done for visual check, not saved
fig.show()

# # %% Create 100 scaling factors for matching the mean image to the reference image
# w = 248.708  # base width (check what this number is?)
# w_min = 520  # (check what this number is?)
# w_max = 570  # (check what this number is?)
#
# scaling = np.linspace(w_min / w, w_max / w, num=100)  # scaling is not used later on...

# %% 2d gaussian convolution (for smoothing of data? but actual guassian filtering commented out)
mean_image_gauss = mean_image
# mean_image_guass= gaussian_filter(mean_image, sigma=2)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mean_image_gauss, cmap="gray")
fig.show()
# %% Generate 2D autocorrelation of channel_image using 2D Fourier transform
image_product = np.fft.fft2(channel_image) * np.fft.fft2(channel_image).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))

# show autocorrelation
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(cc_image.real, cmap="gray")
fig.show()
# %% collapse to 1D horizontal signal and plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(np.mean(cc_image.real, axis=0))
fig.show()
# %%
# Get the 1D signal
signal = np.mean(cc_image.real, axis=0)

# Find peaks.
# You might need to adjust 'distance' based on a rough guess of your spacing
# and 'height' to ignore the flat baseline.
peaks, _ = find_peaks(signal, distance=5, height=np.mean(signal))

# Calculate the differences between adjacent peak locations
spacings = np.diff(peaks)

# The average spacing in pixels
average_spacing = np.mean(spacings)

# %% plot peaks into the signal
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(signal, label="Cross-Correlation Signal")
ax.plot(peaks, signal[peaks], "x", label="Detected Peaks")
ax.axhline(np.mean(signal), color="red", linestyle="--", label="Mean Signal")
ax.legend()
fig.show()
# %% print average spacing
print(f"Average spacing between peaks: {average_spacing:.2f} pixels")
# %%
from scipy.ndimage import rotate

mean_image_flipped = rotate(255 - mean_image_gauss, 36.5)
mean_image_flipped = mean_image_flipped[400:800, 550:950]
image_product = np.fft.fft2(mean_image_flipped) * np.fft.fft2(mean_image_flipped).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
fig, ax = plt.subplots(figsize=(20, 10), ncols=2)
ax[0].imshow(mean_image_flipped, cmap="gray")
ax[0].vlines(
    np.arange(0, mean_image_flipped.shape[1], average_spacing),
    0,
    mean_image_flipped.shape[0],
    color="red",
    linestyle="--",
)
ax[1].imshow(cc_image.real, cmap="viridis")

fig.show()
# %%

signal = np.mean(cc_image.real, axis=0)

# Find peaks.
# You might need to adjust 'distance' based on a rough guess of your spacing
# and 'height' to ignore the flat baseline.
peaks, _ = find_peaks(signal, distance=5)

# Calculate the differences between adjacent peak locations
spacings = np.diff(peaks)

# The average spacing in pixels
average_spacing = np.median(spacings)
print(f"Average spacing between peaks: {average_spacing:.2f} pixels")
# %% plot peaks into the signal
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(signal, label="Cross-Correlation Signal")
ax.plot(peaks, signal[peaks], "x", label="Detected Peaks")
ax.axhline(np.mean(signal), color="red", linestyle="--", label="Mean Signal")
ax.legend()
fig.show()

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mean_image_flipped, cmap="gray")
ax.vlines(peaks, 0, mean_image_flipped.shape[0], color="red", linestyle="--")
fig.show()
# %%
from skimage import feature
from skimage.feature import shape_index
from skimage.measure import find_contours

complete_figure_flipped = rotate(255 - mean_image_gauss, 36.5)
# mask all 0 values
mask = complete_figure_flipped != 0
edges = feature.canny(255 - complete_figure_flipped, sigma=3, mask=mask)
(np.max(complete_figure_flipped) + np.min(complete_figure_flipped)) / 2
contours = find_contours(complete_figure_flipped, mask=mask, level=150)

# %%
all_contours = np.vstack(contours[:2])
# %% perform 4 RANSAC fit iteratively. First fit, then remove inliers, then fit again, etc.
from sklearn.linear_model import RANSACRegressor


inliner_mask = np.ones(all_contours.shape[0], dtype=bool)
x_min = 0
x_max = np.max(complete_figure_flipped.shape)
models = []
line_points = []
orientation = []

for i in range(4):
    reg = RANSACRegressor(residual_threshold=10, max_trials=1000)
    X = all_contours[inliner_mask, 1].reshape(-1, 1)  # x values
    y = all_contours[inliner_mask, 0]  # y values
    reg.fit(X, y)
    if abs(reg.estimator_.coef_[0]) >= 1.0:  # slope too steep, refit swapped
        X = all_contours[inliner_mask, 0].reshape(-1, 1)  # x values
        y = all_contours[inliner_mask, 1]  # y values
        reg = RANSACRegressor(residual_threshold=10, max_trials=1000)
        reg.fit(X, y)
        orientation.append("vertical")
    else:
        orientation.append("horizontal")

    line = np.arange(x_min, x_max)[:, np.newaxis]
    line_points.append(line)
    inlier_mask = reg.inlier_mask_

    models.append(reg)
    inliner_mask[inliner_mask] = np.logical_not(
        inlier_mask, inliner_mask[inliner_mask]
    )  # remove inliers for next iteration
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(complete_figure_flipped, cmap="gray")
colors = ["red", "blue", "green", "yellow"]
for i, reg in enumerate(models):
    if orientation[i] == "horizontal":
        line = line_points[i]
        ax.plot(line, reg.predict(line), color=colors[i], label=f"Line {i+1}")
    else:
        line = line_points[i]
        ax.plot(reg.predict(line), line, color=colors[i], label=f"Line {i+1}")
ax.legend()
fig.show()


# %% calculate the angles between the lines
def get_intersection(m1, b1, m2, b2):
    # row = m1*col + b1 and row = m2*col + b2
    col = (b2 - b1) / (m1 - m2)
    row = m1 * col + b1
    return np.array([col, row])


def normalise_model(model, ori):
    """Always return (m, b) as row = m*col + b"""
    m = model.estimator_.coef_[0]
    b = model.estimator_.intercept_
    if ori == "horizontal":
        return m, b
    else:  # vertical was fit as col = m*row + b, so invert
        return 1 / m, -b / m


h_models = [
    (normalise_model(m, o)) for m, o in zip(models, orientation) if o == "horizontal"
]
v_models = [
    (normalise_model(m, o)) for m, o in zip(models, orientation) if o == "vertical"
]

corners = []
for hm, hb in h_models:
    for vm, vb in v_models:
        corners.append(get_intersection(hm, hb, vm, vb))

corners = np.array(corners)
center = corners.mean(axis=0)
angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
corners = corners[np.argsort(angles)]
plot_pts = np.vstack([corners, corners[0]])

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(complete_figure_flipped, cmap="gray")
ax.plot(plot_pts[:, 0], plot_pts[:, 1], "r--", linewidth=1.5)
ax.scatter(corners[:, 0], corners[:, 1], color="cyan", s=100)
fig.show()
# %%
line_angles = np.rad2deg([np.arctan(m) for m, b in h_models + v_models])
print("Line angles (deg):", line_angles)

# Angle between each h and v line pair
for hm, hb in h_models:
    for vm, vb in v_models:
        diff = abs(np.rad2deg(np.arctan(hm)) - np.rad2deg(np.arctan(vm)))
        interior_angle = min(diff, 180 - diff)
        print(f"Angle between lines: {interior_angle:.2f}°")
# %% calculate area using shoelace formula
x = corners[:, 0]
y = corners[:, 1]
area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
print(f"Area of the quadrilateral formed by the lines: {area:.2f} pixels^2")
# %%
# calculate euclidean distance between corners
from scipy.spatial.distance import pdist, squareform

distances = pdist(corners)
distance_matrix = squareform(distances)
print("Distance matrix between corners:\n", distance_matrix)
# %% these can be put into two categories: adjacent corners (should be close) and opposite corners (should be far). We can check if the average distance of adjacent corners is smaller than that of opposite corners to validate the corner detection.
adjacent_distances = [distance_matrix[i, (i + 1) % 4] for i in range(4)]
opposite_distances = [distance_matrix[i, (i + 2) % 4] for i in range(4)]
print(
    f"Average distance between adjacent corners: {np.round(np.mean(adjacent_distances)):.2f} pixels"
)
print(
    f"Average distance between opposite corners: {np.round(np.mean(opposite_distances)):.2f} pixels"
)

# %% load the original stimulus
with h5py.File(
    rf"F:\alignment\test_measure.h5",
    "r",
) as f:
    stimulus_size = f["Noise"][:].shape[1]
# %%
opposite_distances_mean = np.mean(opposite_distances)
a = np.sqrt(opposite_distances_mean**2 - np.mean(adjacent_distances) ** 2)
# %% Bringing it all together. The stimulus is stimulus_size pixels in size, which corresponds to the distance between
# adjacent corners, so dividing the distance between adjacent corners by stimulus_size gives us the size in pixels.
# We know the size between electrodes and the corresponding size in pixels measured.
electrode_spacing = (
    30  # 4 microns for the actual electrode, 30 microns center to center
)
pixel_size = electrode_spacing / average_spacing
square_size_micron = np.min(adjacent_distances) * pixel_size
micron_per_pixel = square_size_micron / stimulus_size


# %% Summary plotting

fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(complete_figure_flipped, cmap="gray")
ax.plot(plot_pts[:, 0], plot_pts[:, 1], "r--", linewidth=1.5)
ax.scatter(corners[:, 0], corners[:, 1], color="cyan", s=100)
# annotate distances
for i in range(4):
    x1, y1 = corners[i]
    x2, y2 = corners[(i + 1) % 4]
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    distance = adjacent_distances[i]
    ax.text(
        mid_x,
        mid_y,
        f"{distance:.1f}px\n{distance*micron_per_pixel:.1f}µm",
        color="yellow",
        fontsize=8,
        ha="center",
        va="center",
        bbox=dict(facecolor="black", alpha=0.5, boxstyle="round"),
    )
electrode_positions = np.arange(
    np.mean(corners[:, 0]) - (average_spacing * 8),
    np.mean(corners[:, 0]) + (average_spacing * 8),
    16,
)
ax.vlines(
    peaks + 550,
    np.mean(corners[:, 1]),
    np.mean(corners[:, 1]) - 300,
    color="red",
    linestyle="--",
)

ax.set_title(f"Pixel size: {micron_per_pixel:.2f} µm/px", color="white")
ax.set_ylim((300, 1200))
ax.set_xlim((300, 1200))
fig.show()
