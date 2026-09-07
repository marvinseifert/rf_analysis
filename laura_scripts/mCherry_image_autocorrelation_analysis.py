# mCherry cone autocorrelation analysis

# %% Summary:
# I want to find out the spacing (in um) of the zebrafish cone mosaic from the dragonfly microscope images, of an sws1mCherry retina (blue cones marked), imaged following an MEA recording
# I have already developed code to extract the MEA electrode positions from the microscope images and overlay them onto the fluorescence (mCherry) images (code called "image_analysis").
# Now, I can import that overlaid image and extract dimensions in um from it, since we know the distance between the centre of one electrode to the next is 100um.
# Then want to select a ROI of the image, and measure what the dominant repeating distance is in the cone mosaic pattern.

# %% Import dependencies
import numpy as np
import tifffile
import matplotlib
import matplotlib.pyplot as plt
from skimage import exposure, filters, feature
from scipy.optimize import least_squares
import os
import json
from pathlib import Path
import pickle

matplotlib.use("TkAgg")

# %% Set parameters
folder_path = Path(
    r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish MEA + imaging\20260514_dragonfly_microscope"
)

# image_path = folder_path / "electrodes_tiff.tif"
image_path = folder_path / "tissue_with_electrodes_tiff.tif"
expected_mea_pitch_um = 100.0  # adjacent MEA electrode center-to-center spacing

# Autocorrelation parameters
ac_peak_min_distance = 10  # Two autocorrelation peaks must be at least X pixels apart to count as separate peaks
centre_exclusion_radius_px = 20  # Autocorrelation will always have a large central peak at zero shift as it reflects the image matching itself with no shift. So setting this at X means ignore everything within X pixels of the centre of the autocorrelation image.
max_num_ac_peaks = (
    80  # Limits how many peaks are generated in the autocorrelation image.
)
first_shell_tol = 0.25  # tol (tolerance) determines how much variance is still considered the same shell (the first shell is the first ring of shifted image positions where the image matches itself again i.e. 1 lattice/mosaic step away).

# %% Functions list:
# 1) load_image
# 2) fit_circle_to_points
# 3) click_ring_and_fit_center_zoomed
# 4) manual_um_per_pixel_from_ring_fits
# 5) plot_calibration_zoom
# 6) manual_roi
# 7) make_retina_intensity_image
# 8) crop_image
# 9) preprocess_for_lattice
# 10) compute_autocorrelation
# 11) detect_autocorr_peaks
# 12) first_shell_mask


# %%
# Function to load image, using the image path defined above and normalise intensity
# (useful as different retina images may have different intensities, and we are interested in relative intensities anyway)
def load_image(path):
    img = tifffile.imread(path).astype(
        np.float32
    )  # read tiff file and convert to float

    # Normalize to [0, 1] (i.e. brightest channel in brightest pixel becomes 1)
    img /= img.max()
    return img


# %% Calibration (pixel to um): how big (in um) is one pixel?
# Uses known distance between MEA electrodes (100um)
# Following functions allows you to zoom into two electrodes, then you close the window.
# Then select 4 points around the circumference of the first electrode, and close the window.
# Then select 4 points around the circumference of the second electrode and close the window.


# Function to fit circle from 4 points: where input is 4 sets of coordinates ("points_xy") around the circumference of an electrode and fits a circle to those points
def fit_circle_to_points(points_xy):
    points_xy = np.asarray(
        points_xy, dtype=float
    )  # ensure all values are in a numpy array and are floats
    x = points_xy[:, 0]  # extract x coordinates
    y = points_xy[:, 1]  # extract y coordinates

    # Initial estimation of the circle's centre and radius
    x0 = x.mean()  # averages x coordinates
    y0 = y.mean()  # averages y coordinates
    r0 = np.mean(
        np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    )  # calculate distance between each coordinate to estimated centre coordinates, and averages those to estimate radius.

    # Inner function: how wrong is our estimate above?
    def residuals(params):
        (
            cx,
            cy,
            r,
        ) = params  # extract parameters (central coordinates (xy) and radius estimate)
        return (
            np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
        )  # calculate residuals (res = 0, means point lies exactly on the circle, res >0 = point outside circle, res < 0 = point inside circle)

    result = least_squares(
        residuals, x0=[x0, y0, r0]
    )  # minimise value of all residuals, so that all points lie on circle in estimated circle
    cx, cy, r = result.x  # result.x is best-fit parameters
    return cx, cy, r  # output best fit central coordinates (cx,cy) and radius value (r)


# Function to click on X points on circumference of electrode
# Uses function above (fit_circle_to_pts) to fit circle to clicked points
def click_ring_and_fit_center_zoomed(
    image,
    xlim,
    ylim,
    n_points=4,
):  # takes the original image, the zoomed in coordinates, and number of points to click
    fig, ax = plt.subplots(
        figsize=(10, 10)
    )  # displays the zoomed in version of the original image
    ax.imshow(image)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(
        f"click {n_points} points on the green ring, then close window"  # instructions
    )

    pts = plt.ginput(
        n_points, timeout=0
    )  # wait indefinitely for user to click n points
    plt.close(fig)

    pts = np.array(
        pts, dtype=float
    )  # convert clicked points into numpy array of coordinates
    cx, cy, r = fit_circle_to_points(
        pts
    )  # run coordinates through fit_circle_to_points function to fit circle

    # Show fitted circle result for visual confirmation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_xlim(xlim)  # same zoomed region
    ax.set_ylim(ylim)
    ax.scatter(pts[:, 0], pts[:, 1], c="yellow", s=40, label="Clicked points")
    ax.scatter([cx], [cy], c="red", s=50, label="Fitted center")

    theta = np.linspace(
        0, 2 * np.pi, 300
    )  # create 300 angles around the circle (0 to 2pi)
    xs = cx + r * np.cos(theta)
    ys = cy + r * np.sin(theta)
    ax.plot(xs, ys, color="cyan", linewidth=1.5, label="Fitted circle")  # plot circle

    ax.legend()
    ax.set_title(f"fitted circle")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return (
        np.array([cx, cy]),
        r,
        pts,
    )  # return circle centres, radius, and points clicked


def manual_um_per_pixel_from_ring_fits(image, pitch_um=100.0, n_points=4):
    # Step 1: explore image and set zoom
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title("Zoom into two electrodes. Then close window")
    plt.show(block=True)  # force blocking
    # Save the final zoomed view limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Step 2: re-load image at same zoom and fit first electrode circle
    print(
        "Select first electrode by clicking 4 points on its circumference. Then close window. "
    )
    centre1, r1, pts1 = click_ring_and_fit_center_zoomed(
        image,
        xlim,
        ylim,
        n_points=n_points,
    )

    # Step 3: re-load image at same zoom and fit second electrode circle
    print(
        "Select second neighbouring electrode by clicking 4 points on its circumference. Then close window. "
    )
    centre2, r2, pts2 = click_ring_and_fit_center_zoomed(
        image,
        xlim,
        ylim,
        n_points=n_points,
    )

    # Compute distance between centre of one electrode to centre of neighbouring electrode (in pixels)
    dist_px = np.sqrt(
        np.sum((centre2 - centre1) ** 2)
    )  # compute distance between two clicked coordinates - effectively using a^2 + b^2 = c^2 (pythagoras): (first calculate difference between x1 and x2, and y1 and y2 - generating a "difference vector" e.g. (dx,dy), then square each of those and sum them together, and finally square root.
    um_per_px = (
        pitch_um / dist_px
    )  # convert distance in pixels coordinates into um, using pitch_um set in parameters

    cal_pts = np.vstack(
        [centre1, centre2]
    )  # create new "calibration points" array, stacked with centre 1 coordinates and centre 2 coordinates

    cal_info = {  # summarise information
        "xlim": xlim,
        "ylim": ylim,
        "centre1": centre1,
        "centre2": centre2,
        "r1": r1,
        "r2": r2,
        "pts1": pts1,
        "pts2": pts2,
    }

    return um_per_px, dist_px, cal_pts, cal_info


# Function to plot calibration image in final figure
# Will plot zoomed in area of array, with two electrodes, with circles fitted and line between them.
# With labels o
def plot_calibration_zoom(
    ax, image, cal_info
):  # take input of subplot axes, original image and calibration information
    xlim = cal_info["xlim"]  # extract information from cal_info
    ylim = cal_info["ylim"]
    centre1 = cal_info["centre1"]
    centre2 = cal_info["centre2"]
    r1 = cal_info["r1"]
    r2 = cal_info["r2"]
    pts1 = cal_info["pts1"]
    pts2 = cal_info["pts2"]

    ax.imshow(image)  # plot image
    ax.set_xlim(xlim)  # at zoomed in scale
    ax.set_ylim(ylim)

    # show ring clicks in yellow for each electrode
    ax.scatter(pts1[:, 0], pts1[:, 1], c="yellow", s=25, label="Ring clicks")
    ax.scatter(pts2[:, 0], pts2[:, 1], c="yellow", s=25)

    # show calculated electrode centres in red
    ax.scatter(
        [centre1[0], centre2[0]],
        [centre1[1], centre2[1]],
        c="red",
        s=40,
        label="Fitted centres",
    )

    # show centre to centre line in blue
    ax.plot(
        [centre1[0], centre2[0]],
        [centre1[1], centre2[1]],
        color="cyan",
        linewidth=1.5,
        label="Centre distance",
    )
    # Add label to centre to centre line
    # compute midpoint
    mid_x = (centre1[0] + centre2[0]) / 2
    mid_y = (centre1[1] + centre2[1]) / 2

    # add label at midpoint
    ax.text(
        mid_x - 80,
        mid_y + 60,
        f"{expected_mea_pitch_um} um\n (1 pixel = {um_per_px:.2f}um)",
        color="white",
        fontsize=10,
    )

    # show fitted circles around each electrode in green
    theta = np.linspace(0, 2 * np.pi, 300)

    xs1 = centre1[0] + r1 * np.cos(theta)
    ys1 = centre1[1] + r1 * np.sin(theta)
    ax.plot(xs1, ys1, color="lime", linewidth=1.5)

    xs2 = centre2[0] + r2 * np.cos(theta)
    ys2 = centre2[1] + r2 * np.sin(theta)
    ax.plot(xs2, ys2, color="lime", linewidth=1.5, label="Fitted circles")

    ax.set_title("Calibration: fitted electrode circles")
    ax.axis("off")


# %% Manual ROI selection:
# Function to select ROI in microscope image (so that you can avoid doing autocorrelation on entire image if there are dark patches etc).
# CAVEAT: spacing may vary / does vary across retina - so perhaps update code to include multiple ROIs which are then averaged?
def manual_roi(image):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)  # display image
    ax.set_title(
        "Click two corners of a retinal ROI (top-left then bottom-right)"  # provide instructions
    )
    pts = plt.ginput(2, timeout=0)  # wait (indefinitely) for two clicks
    plt.close(fig)  # close figure

    # Similar to above, load clicked coordinates into a np array
    pts = np.array(pts, dtype=float)
    c0, r0 = pts[0]  # set first click coordinates as column 0 and row 0
    c1, r1 = pts[1]  # set second click coordinates as column 1 and row 1

    # Round each column and row value and convert to integer (as click may have fallen along a pixel)
    # Sort the rows and columns so that r0 is smaller than r1, and c0 smaller than c1 - so that they are the right way round
    r0, r1 = sorted([int(round(r0)), int(round(r1))])
    c0, c1 = sorted([int(round(c0)), int(round(c1))])

    return (r0, r1, c0, c1), pts  # outputs of ROI


# %% Preprocessing: convert the microscopy image into format where autocorrelation analysis can occur


# Function to remove green MEA electrode overlay from image, so that only the retina is present
# Could load non electrode array overlaid image instead? Necessary?
def make_retina_intensity_image(image):
    rb = image[..., [0, 2]].mean(axis=-1)  # take mean of red and blue channel only
    rb = exposure.rescale_intensity(
        rb, in_range="image", out_range=(0, 1)
    )  # now re-scale new image so that pixel intensity (grayscale) is between 0 and 1, using function from skimage. In range defines what you are rescaling, and outrange is the bounds between which you want to rescale it to.
    return rb


# Function to crop image to the roi that you selected
def crop_image(img, roi):
    r0, r1, c0, c1 = roi
    return img[r0:r1, c0:c1]


# Function to pre-process image for autocorrelation analysis
def preprocess_for_lattice(intensity):
    low = filters.gaussian(
        intensity, sigma=13
    )  # apply a very broad guassian blur - i.e. what does my image look like if I ignore all fine detail?
    hp = (
        intensity - low
    )  # high pass filter to remove background gradients in intensity - i.e. keep only what changes quickly in intensity across space i.e. cones.

    # create edge window (2D Hanning window) which is ~1 in the centre and smoothly goes to 0 at the edges
    # this is because our ROI is a rectangle with sharp edges which will create fake structure in FFT or autocorrelation, so we need a smooth fade at edges
    wy = np.hanning(hp.shape[0])  # create vertical fade
    wx = np.hanning(hp.shape[1])  # create horizontal fade
    window = np.outer(wy, wx)  # create window

    return hp * window  # apply window (called apodization)


# %% Autocorrelation functions


# Function to compute autocorrelation
# Note - autocorrelation is the inverse FFT of the power spectrum
def compute_autocorrelation(img):
    F = np.fft.fft2(
        img
    )  # Step 1: Compute 2D Fourier transform of the image (convert into frequency space). FFT gives complex numbers because each frequency has an amplitude and a phase.
    F_conj = np.conj(
        F
    )  # Step 2: Compute complex conjugate of the Fourier transform (this flips the sign of the imaginary part of a complex number)
    power_spectrum = (
        F * F_conj
    )  # Step 3: Multiply F by its conjugate to get power spectrum (removes phase part, just keeps amplitude)
    ac_unshifted = np.fft.ifft2(
        power_spectrum
    )  # Step 4: Inverse Fourier transform to get autocorrelation (in unshifted form)
    ac_real = (
        ac_unshifted.real
    )  # Step 5: Take real part (remove tiny imaginary numerical errors)
    ac = np.fft.fftshift(
        ac_real
    )  # Step 6: Shift so zero-lag (no shift) is at the center of the image
    ac = ac / np.max(ac)  # Step 7: Normalize so the central peak (zero shift) = 1
    return ac
    # ac is a 2D numpy array of real numbers, in the same shape as the input image.
    # Each value in ac = an autocorrelation value (i.e. how similar the image is to itself when shifted by that amount).
    # Higher values / bright spots away from the centre mean that if the image is shifted by that amount, then the image still matches itself well - shows repeating patterns.


# Function to detect the peaks in the autocorrelation array
def detect_autocorr_peaks(ac, min_distance=8, exclude_radius=20, num_peaks=80):
    cy, cx = (
        np.array(ac.shape) // 2
    )  # find the coordinates of the central pixel (this is the zero shift peak)

    yy, xx = np.indices(
        ac.shape
    )  # identify the row index (yy) and column index (xx) of every pixel/autocorrelation value
    rr = np.sqrt(
        (yy - cy) ** 2 + (xx - cx) ** 2
    )  # calculate how far away every pixel is from the centre of the autocorrelation array

    work = ac.copy()  # make a copy of the autocorrelation array
    work[
        rr < exclude_radius
    ] = 0  # zero out every value that is within the exclude radius (set in the function) of the central pixel, since a zero shift is not informative and the zero peak can be big

    coords = feature.peak_local_max(  # find pixels that are local maxima in the new autocorrelation array made in the step above
        work,
        min_distance=min_distance,  # two detected peaks must be this far apart (in pixels/values). This value should be smaller than cone spacing, but larger than autocorrelation peak width.
        threshold_rel=0.08,  # remove noise peaks (only include peaks >0.08/1 (1 being maximum value in autocorrelation matrix)
        num_peaks=num_peaks,  # set maximum number of peaks
        exclude_border=False,  # allow peaks near the edge of the image
    )
    # returns an array of pixel coordinates that are local maxima (coords)

    if (
        len(coords) == 0
    ):  # if no peaks are found, return arrays in the expected format still
        return coords, np.empty((0, 2)), np.array([]), np.array([])

    vals = ac[
        coords[:, 0], coords[:, 1]
    ]  # extract the autocorrelation values from the coordinates of local maxima
    order = np.argsort(vals)[
        ::-1
    ]  # sort the autocorrelation peaks from largest to smallest
    coords = coords[
        order
    ]  # reorder the coords array so biggest autocorrelation peaks come first

    vectors = np.column_stack(
        [coords[:, 1] - cx, coords[:, 0] - cy]
    )  # this computes each coordinate as a vector from the autocorrelation centre, rather than raw coordinates
    radii = np.sqrt(
        np.sum(vectors**2, axis=1)
    )  # computes the length of each vector (i.e. the distance of each local peak from the centre of the autocorrelation)
    angles = np.degrees(
        np.arctan2(vectors[:, 1], vectors[:, 0])
    )  # computes the angle of each vector (0 degrees = right, 90 degrees = left/right)

    return coords, vectors, radii, angles  # outputs


# Function to select only the first shell (layer) of peaks (cones)
def first_shell_mask(radii, tol=first_shell_tol):
    if len(radii) == 0:  # if no peaks found, return empty mask
        return np.array([], dtype=bool)

    r0 = np.min(
        radii
    )  # find closest peak to the centre (i.e. the vector with smallest length)
    mask = (
        np.abs(radii - r0) <= tol * r0
    )  # create a boolean mask (i.e. True, False) to keep other peaks within tol (set to 0.25 at start, so 25%) of closest peak to centre

    # expand tol slightly if too few peaks
    if (
        mask.sum() < 4
    ):  # if less than 4 peaks are selected than widen tol by 1.5x (blue cone spacing should in theory be hexagonal)
        print("Warning: <4 peaks in first shell — tol increased by 1.5x")
        mask = np.abs(radii - r0) <= (tol * 1.5) * r0

    return mask


# %% Main script

# Load image
image = load_image(image_path)

# 1) Calibration
# Check if there is a calibration file saved for this image
calibration_path = folder_path / "calibration.pkl"  # build calibration path
# If yes, load calibration information
if calibration_path.exists():
    print("Loading existing calibration...")

    with open(calibration_path.with_suffix(".pkl"), "rb") as f:
        data = pickle.load(f)

    um_per_px = data["um_per_px"]
    cal_pts = data["cal_pts"]
    cal_info = data["cal_info"]
    pitch_px = data["pitch_px"]

# If not, run manual calibration and save output
else:
    print("No calibration found. Running manual calibration...")

    um_per_px, pitch_px, cal_pts, cal_info = manual_um_per_pixel_from_ring_fits(
        image, pitch_um=expected_mea_pitch_um
    )

    # Save calibration
    with open(calibration_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(
            {
                "um_per_px": um_per_px,
                "cal_pts": cal_pts,
                "cal_info": cal_info,
                "pitch_px": pitch_px,
            },
            f,
        )
    print(f"Calibration saved to {calibration_path}")

# 2) choose ROI
roi, roi_pts = manual_roi(image)

# 3) make grayscale retina image with green overlay suppressed
retina = make_retina_intensity_image(image)
retina_crop = crop_image(retina, roi)

# 4) preprocess
proc = preprocess_for_lattice(retina_crop)

# 5) Autocorrelation
ac = compute_autocorrelation(proc)

peak_coords, peak_vecs, peak_radii_px, peak_angles_deg = detect_autocorr_peaks(
    ac,
    min_distance=ac_peak_min_distance,
    exclude_radius=centre_exclusion_radius_px,
    num_peaks=max_num_ac_peaks,
)

# 6) Detect first shell
shell_mask = first_shell_mask(peak_radii_px, tol=first_shell_tol)

# first shell metrics
shell_r_px = peak_radii_px[
    shell_mask
]  # distances of first shell peaks from centre in pixels
shell_r_um = shell_r_px * um_per_px  # distances of first shell peaks from centre in um
shell_angles = np.mod(
    peak_angles_deg[shell_mask], 180.0
)  # angles of first shell peaks from centre (0 to 180)
shell_vecs = peak_vecs[
    shell_mask
]  # displacement vector (dx,dy) of first shell peaks (i.e. distance and angle away from centre) (distance=sqrt(dx2+dy2), angle =arctan2(dy,dx)


# %% Plotting
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
gs = fig.add_gridspec(
    2, 3, wspace=0.05, hspace=0.08  # horizontal spacing  # vertical spacing
)

# original image with calibration electrodes + ROI labelled
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(image, aspect="equal")
ax0.set_anchor("C")  # centres the image in the axes

ax0.plot(
    cal_pts[:, 0], cal_pts[:, 1], "yo-", linewidth=1.5, markersize=6
)  # mea electrodes selected in yellow
r0, r1, c0, c1 = roi
ax0.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0], "c-", linewidth=1.5)  # ROI in blue
ax0.set_title("Original image: calibration electrodes + ROI")
ax0.axis("off")

# zoomed calibration region with fitted circles
electrode_image_path = folder_path / "electrodes_tiff.tif"
electrode_image = load_image(electrode_image_path)

ax1 = fig.add_subplot(gs[0, 1])
plot_calibration_zoom(ax1, electrode_image, cal_info)
ax1.set_anchor("C")  # centres the image in the axes

# cropped retina
ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(retina_crop, cmap="gray", aspect="equal")
ax2.set_anchor("C")  # centres the image in the axes
ax2.set_title("Retina crop used for analysis")
ax2.axis("off")

# autocorrelation
ax3 = fig.add_subplot(gs[1, 1])
ax3.imshow(ac, cmap="viridis", aspect="equal")
ax3.set_anchor("C")  # centres the image in the axes
cy, cx = np.array(ac.shape) // 2
ax3.scatter([cx], [cy], c="red", s=20, label="center")
ax3.scatter(
    peak_coords[:, 1],
    peak_coords[:, 0],
    s=20,
    facecolors="none",
    edgecolors="white",
    linewidths=0.7,
    label="all peaks",
)
ax3.scatter(
    peak_coords[shell_mask, 1],
    peak_coords[shell_mask, 0],
    s=45,
    facecolors="none",
    edgecolors="yellow",
    linewidths=1.3,
    label="first shell",
)
for vec in shell_vecs:
    ax3.plot([cx, cx + vec[0]], [cy, cy + vec[1]], color="yellow", linewidth=0.9)
ax3.legend(loc="lower right", fontsize=8)
ax3.set_title(f"2D autocorrelation\nmean spacing = {np.mean(shell_r_um):.2f} µm")
ax3.axis("off")

# summary text panel
summary_text = (
    "=== Calibration ===\n"
    f"Electrode spacing: {pitch_px:.2f} px = {expected_mea_pitch_um:.1f} µm\n"
    f"Scale: {um_per_px:.4f} µm/px\n\n"
    "=== ROI ===\n"
    f"ROI coordinates: {roi}\n"
    f"Size: {retina_crop.shape[1]} x {retina_crop.shape[0]} px\n"
    f"Size: {retina_crop.shape[1]*um_per_px:.1f} x {retina_crop.shape[0]*um_per_px:.1f} µm\n\n"
    "=== Autocorrelation - first shell peaks ===\n"
    f"Number of peaks: {shell_mask.sum()}\n"
    f"Distance between centre and peaks: {', '.join(f'{r:.2f}' for r in shell_r_um)} µm\n"
    f"Median distance of peaks: {np.median(shell_r_um):.2f} µm\n"
    f"Mean distance of peaks:   {np.mean(shell_r_um):.2f} µm\n"
    f"Std of peaks:    {np.std(shell_r_um):.2f} µm\n"
    f"Angle of peaks: {', '.join(f'{a:.1f}' for a in shell_angles)} deg"  # 0 degrees would be a horizontal line.
)

ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
ax5.text(
    0,
    1,
    summary_text,
    fontsize=10,
    family="monospace",
    color="black",
    va="top",
)

save_path = folder_path / "cone_mosaic_autocorrelation.jpg"

fig.savefig(
    save_path,
    dpi=300,  # high resolution
    bbox_inches="tight",  # trims extra whitespace
    facecolor="white",
)

print(f"Figure saved to:\n{save_path}")

plt.show()


# # %% Parameter checks
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def analyze_autocorr_once(
#     retina_crop,
#     um_per_px,
#     preprocess_sigma=12,
#     min_distance=8,
#     exclude_radius=20,
#     num_peaks=80,
#     first_shell_tol=0.20,
# ):
#     """
#     Run preprocessing + autocorrelation + peak detection once,
#     returning all useful intermediate results.
#     """
#
#     # --- preprocessing (inline version so sigma can be changed) ---
#     low = filters.gaussian(retina_crop, sigma=preprocess_sigma)
#     hp = retina_crop - low
#
#     hp = hp - np.median(hp)
#     sd = np.std(hp)
#     if sd > 0:
#         hp = hp / sd
#
#     wy = np.hanning(hp.shape[0])
#     wx = np.hanning(hp.shape[1])
#     window = np.outer(wy, wx)
#     proc = hp * window
#
#     # --- autocorrelation ---
#     ac = compute_autocorrelation(proc)
#
#     # --- peak detection ---
#     peak_coords, peak_vecs, peak_radii_px, peak_angles_deg = detect_autocorr_peaks(
#         ac,
#         min_distance=min_distance,
#         exclude_radius=exclude_radius,
#         num_peaks=num_peaks,
#     )
#
#     shell_mask = first_shell_mask(peak_radii_px, tol=first_shell_tol)
#
#     if shell_mask.sum() > 0:
#         shell_r_px = peak_radii_px[shell_mask]
#         shell_r_um = shell_r_px * um_per_px
#         spacing_um = float(np.median(shell_r_um))
#     else:
#         shell_r_px = np.array([])
#         shell_r_um = np.array([])
#         spacing_um = np.nan
#
#     return {
#         "proc": proc,
#         "ac": ac,
#         "peak_coords": peak_coords,
#         "peak_vecs": peak_vecs,
#         "peak_radii_px": peak_radii_px,
#         "peak_angles_deg": peak_angles_deg,
#         "shell_mask": shell_mask,
#         "shell_r_px": shell_r_px,
#         "shell_r_um": shell_r_um,
#         "spacing_um": spacing_um,
#         "preprocess_sigma": preprocess_sigma,
#         "min_distance": min_distance,
#         "exclude_radius": exclude_radius,
#         "num_peaks": num_peaks,
#         "first_shell_tol": first_shell_tol,
#     }
#
#
# def plot_autocorr_parameter_sweep(
#     retina_crop,
#     um_per_px,
#     parameter_name,
#     parameter_values,
#     preprocess_sigma=12,
#     min_distance=8,
#     exclude_radius=20,
#     num_peaks=80,
#     first_shell_tol=0.20,
#     ncols=4,
# ):
#     """
#     Plot autocorrelation results for a range of one parameter.
#     """
#
#     results = []
#
#     for val in parameter_values:
#         kwargs = {
#             "preprocess_sigma": preprocess_sigma,
#             "min_distance": min_distance,
#             "exclude_radius": exclude_radius,
#             "num_peaks": num_peaks,
#             "first_shell_tol": first_shell_tol,
#         }
#         kwargs[parameter_name] = val
#
#         result = analyze_autocorr_once(
#             retina_crop=retina_crop,
#             um_per_px=um_per_px,
#             **kwargs,
#         )
#         results.append((val, result))
#
#     n = len(results)
#     nrows = int(np.ceil(n / ncols))
#
#     fig, axes = plt.subplots(
#         nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), squeeze=False
#     )
#
#     for ax in axes.ravel():
#         ax.axis("off")
#
#     for ax, (val, result) in zip(axes.ravel(), results):
#         ac = result["ac"]
#         peak_coords = result["peak_coords"]
#         shell_mask = result["shell_mask"]
#         spacing_um = result["spacing_um"]
#
#         cy, cx = np.array(ac.shape) // 2
#
#         ax.imshow(ac, cmap="viridis")
#         ax.scatter([cx], [cy], c="red", s=18)
#
#         if len(peak_coords) > 0:
#             ax.scatter(
#                 peak_coords[:, 1],
#                 peak_coords[:, 0],
#                 s=18,
#                 facecolors="none",
#                 edgecolors="white",
#                 linewidths=0.7,
#             )
#
#         if shell_mask.sum() > 0:
#             ax.scatter(
#                 peak_coords[shell_mask, 1],
#                 peak_coords[shell_mask, 0],
#                 s=36,
#                 facecolors="none",
#                 edgecolors="yellow",
#                 linewidths=1.2,
#             )
#
#         title = f"{parameter_name} = {val}\nmedian spacing = {spacing_um:.2f} µm"
#         ax.set_title(title, fontsize=10)
#         ax.axis("off")
#
#     fig.suptitle(f"Autocorrelation sweep: {parameter_name}", fontsize=14, y=0.995)
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.show()
#
#     # Print compact summary
#     print(f"\n=== Sweep: {parameter_name} ===")
#     for val, result in results:
#         print(f"{parameter_name} = {val}: spacing = {result['spacing_um']:.3f} µm")
#
#     return results
#
#
# def plot_spacing_summary(results, parameter_name):
#     x = [val for val, _ in results]
#     y = [res["spacing_um"] for _, res in results]
#
#     plt.figure(figsize=(6, 4))
#     plt.plot(x, y, "o-")
#     plt.xlabel(parameter_name)
#     plt.ylabel("Median first-shell spacing (µm)")
#     plt.title(f"Spacing vs {parameter_name}")
#     plt.tight_layout()
#     plt.show()
#
#
# sigma_results = plot_autocorr_parameter_sweep(
#     retina_crop=retina_crop,
#     um_per_px=um_per_px,
#     parameter_name="preprocess_sigma",
#     parameter_values=[4, 6, 8, 10, 12, 14, 16],
#     min_distance=11,
#     exclude_radius=25,
#     first_shell_tol=0.20,
# )
#
# min_distance_results = plot_autocorr_parameter_sweep(
#     retina_crop=retina_crop,
#     um_per_px=um_per_px,
#     parameter_name="min_distance",
#     parameter_values=[6, 8, 10, 12, 14, 16],
#     preprocess_sigma=12,
#     exclude_radius=25,
#     first_shell_tol=0.20,
# )
#
# exclude_radius_results = plot_autocorr_parameter_sweep(
#     retina_crop=retina_crop,
#     um_per_px=um_per_px,
#     parameter_name="exclude_radius",
#     parameter_values=[10, 15, 20, 25, 30, 35],
#     preprocess_sigma=12,
#     min_distance=11,
#     first_shell_tol=0.20,
# )
#
# tol_results = plot_autocorr_parameter_sweep(
#     retina_crop=retina_crop,
#     um_per_px=um_per_px,
#     parameter_name="first_shell_tol",
#     parameter_values=[0.10, 0.15, 0.20, 0.25, 0.30],
#     preprocess_sigma=12,
#     min_distance=11,
#     exclude_radius=25,
# )
#
# plot_spacing_summary(sigma_results, "preprocess_sigma")
# plot_spacing_summary(min_distance_results, "min_distance")
# plot_spacing_summary(exclude_radius_results, "exclude_radius")
# plot_spacing_summary(tol_results, "first_shell_tol")
