# Code for generating synthetic MEA electrode grid and overlaying it over the retinal tissue image
# Need to change image input parameters

# %% Set dependencies, output directory and helper functions
# Import dependencies
import cv2
import numpy as np
import os

# Set output directory (may need to change this)
out_dir = r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish MEA + imaging\20260414_dragonfly_microscope\overlays"
# Or create folder if it doesn't exist
os.makedirs(out_dir, exist_ok=True)


# Helper function to convert image to dtype uint8 (might be uint16) - needed to display image
# uint = unsigned integer. The 8 or 16 is how many bits are used to store the pixel.
# unit8 = 0-255 values per pixel. uint16 = 0–65535 values per pixel.
def to_uint8(img):
    if img.dtype == np.uint8:  # if image is already uint8, do nothing
        return img
    img = img.astype(np.float32)  # convert image to float for calculations
    # Robust contrast stretch (better than simple min/max)
    lo, hi = np.percentile(
        img, (1, 99)
    )  # extract the 1st and 99th percentile pixels (rather than single min/max pixels as these may be VERY bright/dim)
    if hi <= lo:
        lo, hi = img.min(), img.max()
    img = np.clip(
        (img - lo) / (hi - lo + 1e-12), 0, 1
    )  # values above hi are clipped to 1, values below lo are clipped to 0
    return (img * 255).astype(np.uint8)  # returns uint8 image


# Helper function to ensure image is 3-channel BGR uint8 for blending (openCV uses BGR by default rather than RGB)
def to_bgr(img):
    if img is None:
        raise ValueError("Image is None (failed to load).")
    if img.ndim == 2:  # checks if it is grayscale (2 channels only)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


# Helper function: blend an RGBA overlay onto a BGR base image
# Alpha is an extra channel which controls transparency. Needed for overlay images.
def alpha_blend_bgr_with_rgba(base_bgr, overlay_rgba):
    if base_bgr.shape[:2] != overlay_rgba.shape[:2]:
        raise ValueError("Base and overlay must have same height/width.")
    base = base_bgr.astype(np.float32)

    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0  # (H,W,1)

    out = base * (1.0 - alpha) + overlay_rgb * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# %% Input image parameters

# Dimensions of electrode grid in image (in this case, there is a subset of 7 columns and 9 rows of electrodes - entire array is 16 x 16)
cols, rows = (
    16,
    16,
)
# Position of reference electrode from top right electrode (e.g. if it is the 16th column across (i.e. 15 in 0 indexing) and the 11th row down (i.e. 10 in 0 indexing), do 15, 10)
i_ref, j_ref = (
    0,
    15,
)
# Select a reference electrode from FIJI electrode array image, and input pixel coordinates of the centre of the electrode (x,y)
ref_electrode = np.array([9516, 9513], float)

image_width = (
    2276.49  # Width of entire tissue/electrode image in um (extract from FIJI)
)
image_width_px = (
    4723  # Width of entire tissue/electrode image in pixels (extract from FIJI)
)
um_per_px = image_width / image_width_px  # size of one pixel in um
pitch_um = 100  # um from centre of one electrode to the next electrode
pitch_px = (
    pitch_um / um_per_px
)  # the number of pixels from centre of one electrode to the next

# Generate number of pixels needed to move between electrodes
vx = np.array([pitch_px, 0.0])  # one column to the right
vy = np.array([0.0, pitch_px])  # one row down

# Electrode drawing size
electrode_diam_um = 30.0  # electrode diameter (um)
radius_px = int(
    round((electrode_diam_um / 2) / um_per_px)
)  # radius of electrodes in um

# %% Load images
# Electrode image
electrode_img = cv2.imread(
    r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish MEA + imaging\20260421_dragonfly_microscope\SUM_brightfield.tif",
    cv2.IMREAD_UNCHANGED,
)
print(electrode_img.dtype)  # check dtype, should be uint8
# electrode_img = to_uint8(electrode_img)  # convert to uint8 if needed

# Tissue image
tissue_img = cv2.imread(
    r"C:\Users\Laura Steel\Box\SUSSEX\Experiments\Zebrafish MEA + imaging\20260421_dragonfly_microscope\MAX_confocal.tif",
    cv2.IMREAD_UNCHANGED,
)
h, w = electrode_img.shape[
    :2
]  # extract size from one of the two images (electrode or tissue - IF THESE ARE NOT THE SAME SIZE, THIS WILL CAUSE ISSUES)
print(tissue_img.dtype)  # check dtype, should be uint8
tissue_img = to_uint8(tissue_img)  # convert to uint8 if needed

# %% Generate synthetic electrode grid
overlay = np.zeros((h, w, 4), dtype=np.uint8)  # create new array for synthetic grid

for j in range(rows):  # for the number of rows in grid
    for i in range(cols):  # for number of cols in grid
        p = (
            ref_electrode + (i - i_ref) * vx + (j - j_ref) * vy
        )  # from ref_electrode centre coordinates, move e.g. row 0 - row position of ref electrode (3), so move -3 rows up (with rows being measured by vx). Same for columns.
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), radius_px, (0, 255, 0, 180), 2)

# Save electrode_overlay image to output directory
cv2.imwrite("electrode_overlay.png", overlay)
print("Saved electrode_overlay.png to:", out_dir)

# %% Create overlay images

# Convert electrode and tissue images using to_bgr functions
electrode_bgr = to_bgr(electrode_img)
tissue_bgr = to_bgr(tissue_img)

# Create overlay images
electrode_with_grid = alpha_blend_bgr_with_rgba(electrode_bgr, overlay)
tissue_with_grid = alpha_blend_bgr_with_rgba(tissue_bgr, overlay)

# Save overlay images to output directory as .tif files
cv2.imwrite(os.path.join(out_dir, "electrode_with_grid.tif"), electrode_with_grid)
cv2.imwrite(os.path.join(out_dir, "tissue_with_grid.tif"), tissue_with_grid)
cv2.imwrite(os.path.join(out_dir, "grid_overlay_electrode.tif"), overlay)
print("Saved overlay images (as TIFF files) to:", out_dir)

# %%
