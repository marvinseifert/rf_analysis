# %% Seaborn comparison of RF and FFF traits - MULTIPLE RECORDINGS

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr


# %% ------------------------------------------------------------
# Seaborn appearance
# ------------------------------------------------------------

sns.set_theme(
    font_scale=1.4,
)


# %% ------------------------------------------------------------
# Recordings
# ------------------------------------------------------------
#
# Each recording can have its own RF channel name.
#
# recording_id:
#     label used for hue / legend
#
# root_path:
#     path to Phase_00
#
# rf_channel:
#     RF channel to use for that recording

RECORDINGS = [
    {
        "recording_id": "14_08_2026",
        "root_path": Path(r"F:\Laura\zebrafish_14_08_2026\Phase_00"),
        "rf_channel": "32px_15Hz_20mins_shuffle_x4",
    },
    {
        "recording_id": "22_07_2026",
        "root_path": Path(r"F:\Laura\zebrafish_22_07_2026\Phase_00"),
        "rf_channel": "32px_15Hz_20mins_shuffle_x4",
    },
    {
        "recording_id": "15_05_2026",
        "root_path": Path(r"F:\Laura\zebrafish_15_05_2026\Phase_00"),
        "rf_channel": "12px_20Hz_25mins_shuffle_white",
    },
    {
        "recording_id": "14_05_2026",
        "root_path": Path(r"F:\Laura\zebrafish_14_05_2026\Phase_00"),
        "rf_channel": "12px_20Hz_25mins_shuffle_x12",
    },
]


# %% ------------------------------------------------------------
# RF quality threshold
# ------------------------------------------------------------

RF_QUALITY_THRESHOLD = 20


# %% ------------------------------------------------------------
# RF variables to use
# ------------------------------------------------------------

RF_TRAITS = [
    "quality",
    "angle",
    "tilt",
    "center_size_mm2",
    "surround_size_mm2",
]


# %% ------------------------------------------------------------
# FFF variables to use
# ------------------------------------------------------------

FFF_TRAITS = [
    # Contrast steps
    "csteps_polarity_index",
    "csteps_compound_transience_index",
    # Single chromatic flash
    "scf_preferred_wavelength_nm",
    "scf_opponency_strength",
    # Chirp
    "chirp_power_max_frequency",
    "chirp_threshold_frequency",
    # Moving bar
    "moving_bar_OSI",
    "moving_bar_DSI",
    # Cone mosaic
    "cone_regularity_index",
    "cone_spacing_um",
]


# %% ------------------------------------------------------------
# Function to load one recording
# ------------------------------------------------------------


def load_recording(
    recording_id,
    root_path,
    rf_channel,
):
    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    rf_data_path = root_path / "noise_analysis" / "noise_data.nc"

    fff_data_path = root_path / "fff_analysis" / "fff_data.nc"

    print("\n" + "=" * 80)
    print(f"LOADING RECORDING: {recording_id}")
    print("=" * 80)

    print(f"RF channel: {rf_channel}")

    print("\nRF dataset:")
    print(rf_data_path)

    print("\nFFF dataset:")
    print(fff_data_path)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    rf_dataset = xr.load_dataset(rf_data_path)

    fff_dataset = xr.load_dataset(fff_data_path)

    # --------------------------------------------------------
    # Check requested RF channel exists
    # --------------------------------------------------------

    available_channels = rf_dataset["channel"].values.astype(str)

    if rf_channel not in available_channels:
        raise ValueError(
            f"\nRF channel '{rf_channel}' was not found "
            f"in recording '{recording_id}'.\n\n"
            f"Available channels:\n"
            f"{available_channels}"
        )

    # --------------------------------------------------------
    # RF dataframe
    # --------------------------------------------------------

    rf_df = rf_dataset[RF_TRAITS].sel(channel=rf_channel).to_dataframe().reset_index()

    print(f"\nRF cells before quality filtering: " f"{rf_df['cell_index'].nunique()}")

    # --------------------------------------------------------
    # Keep only RFs above quality threshold
    # --------------------------------------------------------

    rf_df = rf_df[rf_df["quality"] >= RF_QUALITY_THRESHOLD].copy()

    print(
        f"RF cells with quality >= "
        f"{RF_QUALITY_THRESHOLD}: "
        f"{rf_df['cell_index'].nunique()}"
    )

    # --------------------------------------------------------
    # FFF dataframe
    # --------------------------------------------------------

    fff_df = fff_dataset[FFF_TRAITS].to_dataframe().reset_index()

    # --------------------------------------------------------
    # One FFF feature set per cell
    # --------------------------------------------------------

    fff_df = fff_df.drop_duplicates(subset="cell_index").copy()

    print(f"FFF cells: " f"{fff_df['cell_index'].nunique()}")

    # --------------------------------------------------------
    # Merge RF + FFF
    # --------------------------------------------------------

    combined_df = rf_df.merge(
        fff_df,
        on="cell_index",
        how="inner",
    )

    # --------------------------------------------------------
    # Add recording information
    # --------------------------------------------------------

    combined_df["recording_id"] = recording_id

    combined_df["rf_channel"] = rf_channel

    # --------------------------------------------------------
    # Make globally unique cell ID
    # --------------------------------------------------------
    #
    # Cell 15 in recording A is different from
    # cell 15 in recording B.

    combined_df["global_cell_id"] = (
        combined_df["recording_id"].astype(str)
        + "_cell_"
        + combined_df["cell_index"].astype(str)
    )

    print(f"\nMerged cells: " f"{combined_df['cell_index'].nunique()}")

    print(f"Rows: " f"{len(combined_df)}")

    # --------------------------------------------------------
    # Close datasets
    # --------------------------------------------------------

    rf_dataset.close()
    fff_dataset.close()

    return combined_df


# %% ------------------------------------------------------------
# Load all recordings
# ------------------------------------------------------------

all_recordings = []


for recording in RECORDINGS:
    recording_df = load_recording(
        recording_id=recording["recording_id"],
        root_path=recording["root_path"],
        rf_channel=recording["rf_channel"],
    )

    all_recordings.append(recording_df)


# %% ------------------------------------------------------------
# Combine all recordings
# ------------------------------------------------------------

combined_df = pd.concat(
    all_recordings,
    ignore_index=True,
)


# %% ------------------------------------------------------------
# Print combined summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("COMBINED DATASET")
print("=" * 80)


print(
    combined_df[
        [
            "recording_id",
            "rf_channel",
            "cell_index",
            "global_cell_id",
        ]
    ].head()
)


print("\nCells per recording:")


print(combined_df.groupby("recording_id")["global_cell_id"].nunique())


print(f"\nTotal cells: " f"{combined_df['global_cell_id'].nunique()}")


print(f"Total rows: " f"{len(combined_df)}")


# %% ------------------------------------------------------------
# Rename variables for plotting
# ------------------------------------------------------------

plot_df = combined_df.rename(
    columns={
        # RF
        "quality": "RF quality",
        "angle": "RF angle",
        "tilt": "RF tilt",
        "center_size_mm2": "RF centre size (mm²)",
        "surround_size_mm2": "RF surround size (mm²)",
        # Contrast steps
        "csteps_polarity_index": "ON/OFF polarity",
        "csteps_compound_transience_index": "Transience",
        # SCF
        "scf_preferred_wavelength_nm": "Preferred wavelength (nm)",
        "scf_opponency_strength": "Colour opponency",
        # Chirp
        "chirp_power_max_frequency": "Preferred frequency (Hz)",
        "chirp_threshold_frequency": "Frequency threshold (Hz)",
        # Moving bar
        "moving_bar_OSI": "OSI",
        "moving_bar_DSI": "DSI",
        # Cone mosaic
        "cone_regularity_index": "Cone regularity",
        "cone_spacing_um": "Cone spacing (µm)",
    }
)


# %% ------------------------------------------------------------
# Traits to plot
# ------------------------------------------------------------

response_traits = [
    "ON/OFF polarity",
    "Transience",
    # "Preferred frequency (Hz)",
    "Frequency threshold (Hz)",
    "Preferred wavelength (nm)",
    # "Colour opponency",
    "OSI",
    # "DSI",
    "RF tilt",
    "RF centre size (mm²)",
    # "RF surround size (mm²)",
    "Cone regularity",
    # "Cone spacing (µm)",
]


# %% ------------------------------------------------------------
# Optional: custom colours for each recording
# ------------------------------------------------------------
#
# If you do NOT want custom colours,
# remove palette=RECORDING_PALETTE
# from sns.pairplot().
#
# Make sure every recording_id has a colour.

RECORDING_PALETTE = {
    "15_05_2026": "red",
    "14_05_2026": "blue",
    "14_08_2026": "green",
    "22_07_2026": "orange",
}


# %% ------------------------------------------------------------
# Pairplot
# Hue = recording
# ------------------------------------------------------------

g = sns.pairplot(
    data=plot_df,
    vars=response_traits,
    hue="recording_id",
    palette=RECORDING_PALETTE,
    corner=True,
    diag_kind="hist",
    plot_kws={
        "alpha": 0.6,
        "s": 25,
    },
    diag_kws={
        "alpha": 0.5,
    },
)


plt.show()
