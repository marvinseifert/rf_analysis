# Script to process and analyse raw MEA array recording to align visual stimulus projection and MEA electrode array
# Stimulus played:"moving_bar_electrodes" x100 white, with additional 12mm aperture in (8 pixel (32um) wide bar, fps = 60, time = 1.33min)
# MEA: 256MEA100/30iR-ITO-pr (30um wide electrodes, 100um spaced)

# N/B: This is the structure of our .h5 (HDF5) file:
# Data
# └── Recording_0                ← one recording session
#     └── AnalogStream
#         ├── Stream_0           ← trigger input
#         │   ├── ChannelData
#         │   └── InfoChannel
#         │
#         └── Stream_1           ← 252 MEA electrodes
#             ├── ChannelData
#             └── InfoChannel

# Import dependencies
from pathlib import Path
import numpy as np
import h5py as h5py
import scipy.signal as sg
import matplotlib.pyplot as plt

# %% Load .h5 recording file (exported using MCS Data Manager)
recording_file = Path(
    r"F:\Laura\MEA_electrode_alignment\2026-03-06T14-43-35McsRecording.h5"
)

# %% Electrode selection:

# Create a list of electrode IDs, with corresponding electrode numbers e.g. electrode 0 is electrode G13 (the electrodes may not be stored in seemingly logical order)
with h5py.File(recording_file, "r") as f:
    info = f["Data"]["Recording_0"]["AnalogStream"]["Stream_1"]["InfoChannel"]

    labels = [x.decode("utf-8") for x in info["Label"]]

# Choose which electrodes from the MEA array to analyse data from e.g. 4 central rows or columns
channels = []

for i, label in enumerate(labels):
    if (
        ("H" in label)
        or ("J" in label)
        or ("K" in label)
        or ("G" in label)
        or ("8" in label)
        or ("9" in label)
        or ("7" in label)
        or ("10" in label)
    ):
        channels.append(i)

# %% Load the electrode data for specified electrode channels

# Select data from specified electrode channels, from the electrode dataset (Stream 1) and convert to numpy array
with h5py.File(recording_file, "r") as f:
    electrode_stream = np.asarray(
        f["Data"]["Recording_0"]["AnalogStream"]["Stream_1"]["ChannelData"][
            channels, :
        ],
        dtype=np.int32,
    ).astype(np.int16)

# %% Load the trigger channel

with h5py.File(recording_file, "r") as f:
    stream = np.asarray(
        f["Data"]["Recording_0"]["AnalogStream"]["Stream_0"]["ChannelData"][:],
        dtype=np.int32,
    )
stream = np.squeeze(stream)
stream[stream < 0] = 0

# %% Process trigger channel (convert to binary signal, by detecting peaks)
# Normalise the trigger to between 0 and 255
trigger_normalized = (
    (stream - stream.min()) / (stream.max() - stream.min()) * 255
).astype(np.uint8)
# Threshold the signal into ON or OFF
trigger_normalized[trigger_normalized <= 126] = 0
trigger_normalized[trigger_normalized > 126] = 1
# Convert signal to boolean (i.e. on = true, off = false)
trigger_normalized = trigger_normalized.astype(bool)
# Detect peaks (i.e. the start of each trigger)
trigger_peaks = sg.find_peaks(trigger_normalized, height=1, plateau_size=2)
# Calculate time between each trigger signal
peak_diffs = np.diff(trigger_peaks[1]["left_edges"])

# %% Assessing the trigger channel
# Visualise the peak_diff of the triggers (the time between each trigger) to check number of triggers is correct
plt.figure(figsize=(12, 4))
plt.plot(peak_diffs)  # plot portion of trigger channel
plt.title("Trigger channel")
plt.tight_layout()
plt.show()

# Can see that there are a few values over 30. This is indicative of dropped frames.
# The MEA was set to record at 1000Hz and the stimulus was presented at 60Hz.
# So that's a trigger every 16.66 recording samples.
# So the peak diffs should fluctuate between 16 and 17 (assuming everything worked correctly).
# Dropped frame would = ~32 (16*2). Looking at the plot above, we could consider anything >28 a dropped frame.

# %% Correcting the trigger channel
# At the locations of dropped frames, add in an extra trigger signal to trigger_peaks

# Identify location of >28 values in peak_diff array
frame_drop_indices = np.where(peak_diffs > 28)[0]
values = peak_diffs[frame_drop_indices]
print(f"location of frame drops: {frame_drop_indices}")
print(f"value of frame drops: {values}")

# Still 1 trigger short, so manually adding in an extra frame drop at 25799 (visually decided)
frame_drop_indices = np.append(frame_drop_indices, 25799)

# Add in new triggers in correct locations
new_triggers = []
trigger_onsets = trigger_peaks[1]["left_edges"]

for idx in frame_drop_indices:
    left = trigger_onsets[idx]
    right = trigger_onsets[idx + 1]

    # insert one trigger halfway between the two
    new_trigger = int(round((left + right) / 2))
    new_triggers.append(new_trigger)

# Add to trigger_onsets
trigger_onsets = np.sort(
    np.concatenate([trigger_onsets, np.array(new_triggers, dtype=int)])
)
trigger_onsets_diff = np.diff(trigger_onsets)

# Re-plot trigger_onset_diff to check trigger intervals:
plt.figure(figsize=(12, 4))
plt.plot(trigger_onsets_diff)  # plot portion of trigger channel
plt.title("Trigger channel")
plt.tight_layout()
plt.show()

# Make binary trigger channel for future alignment/visualisation
binary_trigger = np.zeros(len(stream), dtype=int)
binary_trigger[trigger_onsets] = 1

# Summary check of trigger number
print(f"number of triggers: {len(trigger_onsets)}")
print(f"number of triggers per repeat: {len(trigger_onsets)/100}")

# %% Electrode stream corrections
# %% 1.Baseline correction

# Define baseline window (time before first trigger)
baseline_start = 0
baseline_stop = trigger_onsets[0] - 1

# Convert electrode stream to float type, for calculations
electrode_stream_f = electrode_stream.astype(np.float64)

# Calculate baseline mean from pre-stimulus period for each channel
baseline_mean = electrode_stream_f[:, baseline_start:baseline_stop].mean(
    axis=1, keepdims=True
)

# Baseline correction for each channel
electrode_baseline_corrected = electrode_stream_f - baseline_mean

# %% 2. Common median subtraction (on top of baseline correction)
# Subtract the median across channels at every sample/timepoint, from each channel
electrode_common_median = electrode_baseline_corrected - np.median(
    electrode_baseline_corrected, axis=0, keepdims=True
)

# %% Downsample data by summing neighboring samples

# Choose downsampling factor
downsample_factor = 10  # i.e. 1000 to 100 Hz

# Use baseline corrected + common median corrected electrode data
data = electrode_common_median

# Extract number of electrode channels and number of samples from data
n_channels, n_samples = data.shape

# Reshape and sum neighbouring samples
electrode_downsampled = data.reshape(
    n_channels, n_samples // downsample_factor, downsample_factor
).sum(axis=2)

# Downsample trigger channel as well (use .max to just select trigger from binary trigger array)
binary_trigger_downsampled = binary_trigger.reshape(
    n_samples // downsample_factor, downsample_factor
).max(axis=1)
# trigger_onsets_ds = np.round(trigger_onsets / downsample_factor).astype(
#     int
# )  ## is this correct logic?? think.
trigger_onsets_ds = trigger_onsets // downsample_factor

# Check downsampled dimensions
print("Original electrode shape:", data.shape)
print("Downsampled electrode shape:", electrode_downsampled.shape)
print("Original binary trigger shape:", binary_trigger.shape)
print("Downsampled binary trigger shape:", binary_trigger_downsampled.shape)

# %% Visualisation checks so far

# Select electrode channel and repeat number of interest
electrode_ID = 1
repeat_number = 0
triggers_per_repeat = 4800
start_sample = trigger_onsets[repeat_number * triggers_per_repeat]
end_sample = trigger_onsets[
    (repeat_number * triggers_per_repeat) + triggers_per_repeat - 1
]

fig, axes = plt.subplots(6, 1, figsize=(12, 12))

axes[0].plot(binary_trigger[start_sample:end_sample])
axes[0].set_title("Raw (corrected) trigger")

axes[1].plot(
    binary_trigger_downsampled[
        start_sample // downsample_factor : end_sample // downsample_factor
    ]
)
axes[1].set_title("Downsampled trigger")

axes[2].plot(electrode_stream[electrode_ID, start_sample:end_sample])
axes[2].set_title("Raw electrode signal")

axes[3].plot(electrode_baseline_corrected[electrode_ID, start_sample:end_sample])
axes[3].set_title("Baseline corrected electrode signal")

axes[4].plot(electrode_common_median[electrode_ID, start_sample:end_sample])
axes[4].set_title("Baseline corrected + common median subtracted electrode signal")

axes[5].plot(
    electrode_downsampled[
        electrode_ID,
        start_sample // downsample_factor : end_sample // downsample_factor,
    ]
)
axes[5].set_title("Corrected + downsampled electrode")

plt.tight_layout()
plt.show()

# %% Visualise 1 electrode and 1 repeat, but cut into 8 different sections (corresponding to each moving bar position)
# Using downsampled data
# Select electrode and repeat of interest
electrode_ID = 10
repeat_number = 1
triggers_per_repeat = 4800
triggers_per_position = 600

repeat_start = repeat_number * triggers_per_repeat

fig, axes = plt.subplots(9, 1, figsize=(12, 12))

# First position trigger train
start_sample = trigger_onsets_ds[repeat_start]
end_sample = trigger_onsets_ds[repeat_start + triggers_per_position - 1]

# Plot downsampled trigger channel for the equivalent of 1 moving bar position duration
axes[0].plot(binary_trigger_downsampled[start_sample:end_sample])
axes[0].set_title("Downsampled trigger for one position")

# Plot 8 moving bar positions
for pos in range(8):
    start_idx = repeat_start + pos * triggers_per_position
    end_idx = repeat_start + (pos + 1) * triggers_per_position - 1

    start_sample = trigger_onsets_ds[start_idx]
    end_sample = trigger_onsets_ds[end_idx]

    axes[pos + 1].plot(electrode_downsampled[electrode_ID, start_sample:end_sample])
    axes[pos + 1].set_title(f"Moving bar position {pos + 1}")

fig.suptitle(f"Electrode: {electrode_ID}, repeat: {repeat_number}")
plt.tight_layout()
plt.show()

# %% Average downsampled and corrected data across repeats
