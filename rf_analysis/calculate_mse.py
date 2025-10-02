import pathlib
import tqdm
from polarspike import Overview
import numpy as np
from rf_analysis.channel_handling import masking_square
from rf_torch.parameters import Noise_Params, Cell_Params

"""
This script calculates the maximum mean squared error (MSE) for each cell's spike-triggered average (STA)
Usually, the MSE is calculated for a cutout around the peak of the STA to save computation time."""
# %% Parameters
cut = 100  # Size of the square cutout
quality_threshold = 10.0  # Only cells with a quality above this threshold are processed
new_cell_params = True  # If True, a new cell_params.json is created for each cell, otherwise the existing one is updated
# Define the path, you may remove or add multiple paths to different noise root folders
root_path = pathlib.Path(r"/home/mawa/nas_a/Marvin/chicken_13_05_2025/Phase_00")
# Define the root paths for the different noise analyses.
# This script assumes that the calculate_quality.py script has been run for each of these folders and that each
# folder contains the same nr of cells in the same order.
noise_roots = [
    root_path / "4px_20Hz_shuffle_460nm_idx_2",
    root_path / "4px_20Hz_shuffle_535nm_idx_7",
    root_path / "4px_20Hz_shuffle_610nm_idx_3"
]
# %%
quality_paths = [
    np.load(path / "quality.npy", mmap_mode="r")
    for path in noise_roots
]
# load noise parameters
noise_parameters = [Noise_Params.load_from_root_json(path) for path in noise_roots]
# %%
half_cut = int(cut // 2)
recording = Overview.Recording.load(root_path / "overview")
nr_cells = recording.nr_cells
# %%
for noise_idx, noise_root in enumerate(noise_roots):
    print("Processing folder:", noise_root)
    quality = np.load(noise_root / "quality.npy")
    # Load the number of spikes for the relevant stimulus
    nr_of_spikes = recording.spikes_df.query(f"stimulus_index=={noise_parameters[noise_idx]['noise_stim_index']}")[
        "nr_of_spikes"].values
    for cell_idx in tqdm.tqdm(range(nr_cells), smoothing=0):
        folder = noise_root / f"cell_{cell_idx}"
        quality_cell = quality[cell_idx][0]
        if quality_cell < quality_threshold:
            np.save(folder / "subset_dev_img.npy", np.zeros((cut, cut)))
            continue

        # Find most likely center position
        max_quality_channel = np.argmax([q[cell_idx][0] for q in quality_paths])
        position = (
            quality_paths[max_quality_channel][cell_idx][2].astype(int),
            quality_paths[max_quality_channel][cell_idx][3].astype(int),
        )
        sta_data = np.load(folder / "kernel.npy", mmap_mode="r")
        original_sta_shape = sta_data.shape

        mask = masking_square(original_sta_shape[1], original_sta_shape[2], (position[0], position[1]), cut, cut)
        if np.sum(mask) != cut * cut:
            mask_shape = (np.max(np.sum(mask, axis=0)), np.max(np.sum(mask, axis=1)))

        subset_flat = sta_data[:, mask].astype(float)
        # Calculate MSE for the cutout
        if np.sum(mask) != cut * cut:  # if the mask is smaller than the cut size, e.g. if the cell is close to the edge
            subset_dev = np.max((subset_flat / nr_of_spikes[cell_idx] - 0.5) ** 2, axis=0).reshape(mask_shape)
            # fill with zeros to get the right shape
            subset_dev_full = np.zeros((cut, cut))
            subset_dev_full[:mask_shape[0], :mask_shape[1]] = subset_dev
            subset_dev = subset_dev_full
        else:
            subset_dev = np.max((subset_flat / nr_of_spikes[cell_idx] - 0.5) ** 2, axis=0).reshape(cut, cut)
        np.save(folder / "subset_dev_img.npy", subset_dev)
        # update the cell parameters
        # check if cell_params.json exists
        if (folder / "cell_params.json").exists() and not new_cell_params:
            cell_params = Cell_Params.load_from_root_json(folder)
        else:
            cell_params = Cell_Params(cell_idx=cell_idx, recording=recording.name,
                                      output_folder=folder,
                                      noise_params=noise_parameters[noise_idx])
        cell_params["quality_analysis"]["max_quality_channel"] = int(max_quality_channel)
        cell_params["quality_analysis"]["position"] = (int(position[0]), int(position[1]))
        cell_params.save_json()
