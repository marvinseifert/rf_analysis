# seaborn_plots.py — v2 version for v2 data (2026-11-18). The original (v1) is in laura_scripts/legacy/Seaborn/fff_feature_pipeline/seaborn_plots.py.
#   * reads noise_data_v2.nc (RF maps in stimulus coordinates, corrected sizes/angles)
#   * angle is now in the stimulus frame (= 90° - old); sizes use the corrected values
#   * reads fff_data_v2.nc (corrected alignment + cone regularity)
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
    # All recordings with a v2 cone alignment (same set and RF channels as
    # RF_cone_row_angle_prediction/rf_cone_row_angle.py; 30/04 left out, 4px too fine).
    {
        "recording_id": "07_05_2026",
        "root_path": Path(r"/mnt/mea_nas/Laura/zebrafish_07_05_2026/Phase_00"),
        "rf_channel": "12px_20Hz_20mins_shuffle_x12_white",
    },
    {
        "recording_id": "14_05_2026",
        "root_path": Path(r"/mnt/mea_nas/Laura/zebrafish_14_05_2026/Phase_00"),
        "rf_channel": "12px_20Hz_25mins_shuffle_x12",
    },
    {
        "recording_id": "15_05_2026",
        "root_path": Path(r"/mnt/mea_nas/Laura/zebrafish_15_05_2026/Phase_00"),
        "rf_channel": "12px_20Hz_25mins_shuffle_white",
    },
    {
        "recording_id": "22_07_2026",
        "root_path": Path(r"/mnt/mea_nas/Laura/zebrafish_22_07_2026/Phase_00"),
        "rf_channel": "32px_15Hz_20mins_shuffle_x4",
    },
    {
        "recording_id": "07_08_2026",
        "root_path": Path(r"/mnt/mea_nas/Laura/zebrafish_07_08_2026/Phase_00"),
        "rf_channel": "32px_15Hz_20mins_shuffle_x4",
    },
    {
        "recording_id": "14_08_2026",
        "root_path": Path(r"/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00"),
        "rf_channel": "32px_15Hz_20mins_shuffle_x4",
    },
    {
        "recording_id": "15_10_2026",
        "root_path": Path(r"/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00"),
        "rf_channel": "32px_15Hz_20mins_shuffle_x4",
    },
]


# %% ------------------------------------------------------------
# RF quality threshold
# ------------------------------------------------------------

# One pairplot per threshold (cells with RF quality > threshold), so they can be compared.
# RF data come from noise_analysis_QI10 (RF maps/outlines for every cell with quality >= 10),
# so thresholds below 10 are not possible.
RF_QUALITY_THRESHOLDS = [10, 20]
RF_QUALITY_THRESHOLD = min(RF_QUALITY_THRESHOLDS)  # used when loading


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
    noise_folder="noise_analysis_QI10",
):
    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    rf_data_path = root_path / noise_folder / "noise_data_v2.nc"

    fff_data_path = root_path / "fff_analysis" / "fff_data_v2.nc"

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

    rf_df = rf_df[rf_df["quality"] > RF_QUALITY_THRESHOLD].copy()

    print(
        f"RF cells with quality > "
        f"{RF_QUALITY_THRESHOLD}: "
        f"{rf_df['cell_index'].nunique()}"
    )

    # --------------------------------------------------------
    # FFF dataframe
    # --------------------------------------------------------

    # Some recordings lack a stimulus (07/05: no moving bar; 07/08: only the
    # weighted 60 fps bar, not extracted): those traits are left as NaN.
    missing_traits = [t for t in FFF_TRAITS if t not in fff_dataset]
    if missing_traits:
        print(f"Missing FFF traits (set to NaN): {missing_traits}")
    fff_df = (
        fff_dataset[[t for t in FFF_TRAITS if t in fff_dataset]]
        .to_dataframe()
        .reindex(columns=FFF_TRAITS)
        .reset_index()
    )

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
        noise_folder=recording.get("noise_folder", "noise_analysis_QI10"),
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
    "07_05_2026": "purple",
    "14_05_2026": "blue",
    "15_05_2026": "red",
    "22_07_2026": "orange",
    "07_08_2026": "brown",
    "14_08_2026": "green",
    "15_10_2026": "black",
}


# %% ------------------------------------------------------------
# Pairplot
# Hue = recording
# ------------------------------------------------------------

for threshold in RF_QUALITY_THRESHOLDS:
    subset = plot_df[plot_df["RF quality"] > threshold]
    n_cells = subset.groupby("recording_id")["global_cell_id"].nunique()
    print(f"\nRF quality > {threshold}: {len(subset)} cells\n{n_cells.to_string()}")

    g = sns.pairplot(
        data=subset,
        vars=response_traits,
        hue="recording_id",
        hue_order=list(RECORDING_PALETTE),
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
    g.figure.suptitle(
        f"RF quality > {threshold} (noise_analysis_QI10, v2): {len(subset)} cells, "
        f"{subset['recording_id'].nunique()} recordings",
        y=1.01,
    )

    output_figure = Path(
        r"/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/Seaborn"
        rf"/seaborn_pairplot_v2_RFquality_above_{threshold}_all_aligned_recordings.png"
    )
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_figure, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_figure}")

plt.show()
