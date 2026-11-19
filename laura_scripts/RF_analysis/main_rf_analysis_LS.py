# main_rf_analysis_LS.py — v2 version for v2 data (2026-11-18). The original (v1) is in laura_scripts/legacy/RF_analysis/main_rf_analysis_LS.py.
#   * reads noise_data_v2.nc (RF maps in stimulus coordinates, corrected sizes/angles)
# Receptive field analysis

# %% Import dependencies
from pathlib import Path
import xarray as xr
from loading.noise_data import load_noise_data  # v2 RF dataset, checked
import pandas as pd

# Load dataset(s)
path_to_data = Path(
    r"/mnt/mea_nas/Laura/zebrafish_15_05_2026/Phase_00/noise_analysis/noise_data_v2.nc"
)
dataset = load_noise_data(path_to_data)

# %%
# # Combine datasets if wanted
#
# # List of datasets
# datasets = [dataset_05_11_25, dataset_02_12_25, dataset_15_01_26]
#
# # Name datasets
# names = ["05_11_25", "02_12_25", "15_01_26"]
#
# # Concatenate along a new dimension called "date"
# combined = xr.concat(datasets, dim=pd.Index(names, name="date"))

# %% Set dataset, channel and qi_limit
channel = "12px_20Hz_25mins_shuffle_x12_white"
qi_limit = 20
# # Load quality file
quality = dataset["quality"]

# # Select cell indices for each channel individually which have qi > 20
top_cells_dict = {
    ch: quality.sel(channel=ch).cell_index.values[
        quality.sel(channel=ch).values > qi_limit
    ]
    for ch in quality.channel.values
}
# Can now index the top cells from a single channel:
top_cells = top_cells_dict[channel]
print(len(top_cells))
# %% Load and run functions
from laura_scripts.RF_analysis.receptive_field_functions import (
    plot_mse_rf,
    plot_centre_surrounds,
    plot_sta,
    plot_rf_statistics,
)


plot_mse_rf(dataset, channel, qi_limit)
# plot_centre_surrounds(dataset, channel, qi_limit)
# plot_sta(dataset, channel, qi_limit)
# plot_rf_statistics(dataset, channel, qi_limit, bin_size=20)
