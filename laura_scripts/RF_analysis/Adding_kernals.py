# Script to add kernels from two channels (versions of noise) together

# Import dependencies
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Load folder paths for each channel
path_a = Path(
    r"F:\Laura\zebrafish_15_05_2026\Phase_00\12px_20Hz_25mins_shuffle_x12_white_idx_15"
)
path_b = Path(
    r"F:\Laura\zebrafish_15_05_2026\Phase_00\12px_20Hz_25mins_shuffle_x12_repeat_white_idx_19"
)

# Create output path
output_path = Path(r"F:\Laura\zebrafish_15_05_2026\Phase_00\combined_kernels")
output_path.mkdir(
    parents=True, exist_ok=True
)  # make new path and folder if not already there

# Find cells in for channel A
cells_a = {}  # create new dictionary that will store cell folder paths
for folder in path_a.iterdir():  # loop through everything inside path_a
    if folder.is_dir() and folder.name.startswith(
        "cell_"
    ):  # keep only folders (not files) beginning with "Cell"
        cells_a[
            folder.name
        ] = folder  # Assign the folder name as the dictionary key (and the value will be the full path to that folder) - so later can call as "cells_a["cell_25"]"

# Find cells for channel B
cells_b = {}  # create new dictionary that will store cell folder paths
for folder in path_b.iterdir():  # loop through everything inside path_a
    if folder.is_dir() and folder.name.startswith(
        "cell_"
    ):  # keep only folders (not files) beginning with "Cell"
        cells_b[
            folder.name
        ] = folder  # Assign the folder name as the dictionary key (and the value will be the full path to that folder) - so later can call as "cells_a["cell_25"]"

# Only use cells that exist in both folders
common_cells = list(set(cells_a) & set(cells_b))
print(f"{len(common_cells)} common cells found")

# %%
# Loop over cells
# For each cell, load the kernel from each of the channels
# Add the kernels together
# Save the kernel in new folder

for cell in tqdm(common_cells):  # progress bar
    kernel_a = np.load(cells_a[cell] / "kernel.npy")
    kernel_b = np.load(cells_b[cell] / "kernel.npy")
    combined_kernel = kernel_a + kernel_b
    save_path = output_path / cell / "combined_kernel.npy"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, combined_kernel)
