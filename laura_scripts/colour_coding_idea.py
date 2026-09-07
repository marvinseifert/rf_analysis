# Load top qi (for rf) cells spike responses to scf
# Bin spikes into three - short or long.
# If there are considerably more short spikes than long, assign cell as short.
# If there are considerably more long spikes than short, assign cell as short.
# If there are approximately the same, assign cell as white.
# (decide on threshold for "considerably more")
# Return cell id with extra column of short, white, long

import numpy as np
from pathlib import Path
from polarspike import Overview

# %% Loading receptive fields

# Load dataset
root = Path(r"F:\Laura\zebrafish_02_12_2025\Phase_01\4px_20Hz_40mins_shuffle_idx_4")
# Load quality file
quality = np.load(root / "quality.npy")

# Find top cells with quality index > 20
qi_limit = 20
valid_rows_quality = ~np.isnan(quality[:, 0])
top_cells_idx = np.where(quality[valid_rows_quality, 0] > qi_limit)
top_cells_idx = np.where(valid_rows_quality)[0][top_cells_idx]
top_cells = list(quality[top_cells_idx, 1].astype(int))

# %% Loading spikes
# Import dataset as recording
recording = Overview.Recording.load(r"F:\Laura\zebrafish_02_12_2025\Phase_01\overview")
# Set scf stimulus number
stimulus = 1
# Import all top qi cells for stimulus
spikes = recording.get_spikes_triggered([{"cell_index": top_cells, "stimulus_index": stimulus}])

# %% Classifying cells by scf spike patterns
# Into short, long, white


# For each value in cell_index in good_cells_spikes, and for each value in the repeat column, find the number of rows where the times_triggered column represents each LED (e.g. led1 = >0 & <2, led2 = >2 & <4 etc)
# The number of rows will represent the number of spikes per led wavelength flash, for each repeat, for each cell
# Calculate the mean across the repeats, per cell
# Calculate the mean across cells
# Plot summed spikes across wavelength, against background of opsin spectral sensitivity

# Information
num_of_leds = 8
led_times = [(0,2), (4,6), (8,10), (12,14), (16,18), (20,22), (24,26), (28,30)]
num_of_repeats = 5

# Firstly, create a dictionary called 'leds', with 8 empty arrays to input data into, one for each led wavelength type (led1-8)
leds = {}
for i in range(num_of_leds):
    leds[f'led{i}'] = np.zeros((len(top_cells), num_of_repeats))

led_mean_repeats = {}
for i in range(num_of_leds):
    led_mean_repeats[f'led{i}_mean_repeat'] = np.zeros((len(top_cells), num_of_repeats))

for i in range (len(top_cells)): #number of cells
    for j in range(num_of_repeats): #number of repeats
        cell_num = spikes['cell_index'] == top_cells[i] # For each cell in good_cell_spikes
        repeat_num = spikes['repeat'] == j # For each repeat

        for k in range(num_of_leds):
            leds[f'led{k}'][i,j] = len(
                spikes[cell_num & repeat_num
                                  & (spikes['times_triggered'] >= led_times[k][0])
                                  & (spikes['times_triggered'] < led_times[k][1])
                                  ].index)
for l in range(num_of_leds):
    led_mean_repeats[f'led{l}_mean_repeat'] = np.mean(leds[f'led{l}'], axis =1)

summary_table = np.column_stack([
    led_mean_repeats[f'led{h}_mean_repeat']for h in range(num_of_leds)])

# %% Classify cells
# Sum first four and last four columns/led spikes for each cell
# Calculate ratio
# Set ratio threshold and classify:
# If ratio is 0.9 to 1.1, assign as white
# If ratio is < 0.9 assign as short
# If ratio is > 1.1 assign as long
# cells which do not spike in either category will be "white"

short_long = np.zeros((len(top_cells), 3))
cell_labels = np.empty((len(top_cells), 2), dtype=object)

for i, cell in enumerate(top_cells):
    short_long[i,0] = np.sum(summary_table[i,0:4]) # number of long spikes
    short_long[i,1] = np.sum(summary_table[i,4:8]) # number of short spikes
    short_long[i,2] = short_long[i,0]/short_long[i,1] # ratio of short to long (>1 means more long than short spikes)
    if short_long[i,2] < 0.9:
        cell_labels[i,1] = "short"
    elif short_long[i,2] > 1.1:
        cell_labels[i,1] = "long"
    else:
        cell_labels[i] = "white"
    cell_labels[i,0] = cell

np.save(root / "cell_labels.npy", cell_labels)