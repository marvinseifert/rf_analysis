# %% Build / update one per-cell FFF + cone-mosaic feature dataset
#
# This version can UPDATE an existing fff_data.nc / fff_features.csv
# without recalculating everything.

# This loads the existing FFF table, preserves all old columns,
# adds/replaces only the requested feature blocks,
# and saves the updated table again.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from extract_chirp_features import calculate_chirp_features
from extract_cone_regularity import calculate_cone_regularity_features
from extract_csteps_features import calculate_csteps_features
from extract_moving_bar_features import calculate_moving_bar_features
from extract_moving_edge_features import (
    MovingEdgeConditionConfig,
    calculate_moving_edge_features,
)
from extract_scf_features import calculate_scf_features
from fff_common import all_cell_indices


# =============================================================================
# 1. RECORDING PATHS
# =============================================================================

RECORDING_ROOT = Path(r"F:\Laura\zebrafish_14_08_2026\Phase_00")

OVERVIEW_PATH = RECORDING_ROOT / "overview"

NOISE_DATA_PATH = RECORDING_ROOT / "noise_analysis" / "noise_data.nc"

OUTPUT_DIR = RECORDING_ROOT / "fff_analysis"


# =============================================================================
# 2. CHOOSE WHICH FEATURE BLOCKS TO RUN
# =============================================================================
#
# False:
#     DO NOT calculate that feature again.
#     If it already exists in fff_data.nc / fff_features.csv,
#     the existing columns are preserved.
#
# True:
#     Calculate that feature now.
#     If columns from that extractor already exist,
#     they are replaced by the newly calculated values.
#
#
# YOUR CURRENT SETTINGS:
#
# You have already extracted:
#     csteps
#     SCF
#     chirp
#     moving bar
#
# So leave those False.
#
# Only cone regularity will be added.

RUN_CSTEPS = False
RUN_SCF = False
RUN_CHIRP = False
RUN_MOVING_BAR = False
RUN_MOVING_EDGE = False

RUN_CONE_REGULARITY = True


# =============================================================================
# 3. STIMULUS SETTINGS
# =============================================================================
SCF_STIMULUS_ID = 0
CSTEPS_STIMULUS_ID = 1
CHIRP_STIMULUS_ID = 2
MOVING_BAR_STIMULUS_ID = 3


CSTEPS_PARAMETERS = dict(
    duration_s=40.0,
    expected_repeats=3,
    step_span_s=2.0,
    bin_size_s=0.01,
)


SCF_PARAMETERS = dict(
    wavelengths_nm=(
        660,
        610,
        560,
        535,
        500,
        460,
        420,
        365,
    ),
    expected_repeats=5,
    stimulus_span_s=2.0,
    bin_size_s=0.01,
    response_window_s=1.0,
)


# Leave empty to preserve defaults
# from the existing chirp extractor.

CHIRP_PARAMETERS = dict()


MOVING_BAR_PARAMETERS = dict(
    frames_per_direction=300,
    seconds_per_frame=1 / 60,
    expected_repeats=10,
    time_unit="seconds",
    bin_size_s=0.10,
    response_metric="peak_response",
)


# =============================================================================
# MOVING EDGE SETTINGS
# =============================================================================
#
# Moving edges are independent from moving bars.
#
# Leave this empty if you are not currently extracting moving-edge features.
#
MOVING_EDGE_CONDITIONS = (
    # Example - replace IDs and H5 paths with this recording's moving-edge stimuli:
    MovingEdgeConditionConfig(
        output_name="on_13fps",
        stimulus_index=15,
        h5_path=Path(
            r"F:\Laura\stimuli\stimuli_July_2026\moving_edge_weighted_13fps_ON_800px_wide_800px_high_20px_thick_30.0px_ramp_1.0px_per_frame.h5"
        ),
        expected_repeats=3,
        expected_polarity="ON",
        expected_frame_rate_hz=13,
        time_unit="seconds",
        bin_size_s=0.25,
        response_metric="peak_response",
    ),
    MovingEdgeConditionConfig(
        output_name="off_13fps",
        stimulus_index=16,
        h5_path=Path(
            r"F:\Laura\stimuli\stimuli_July_2026\moving_edge_weighted_13fps_OFF_800px_wide_800px_high_20px_thick_30.0px_ramp_1.0px_per_frame.h5"
        ),
        expected_repeats=3,
        expected_polarity="OFF",
        expected_frame_rate_hz=13,
        time_unit="seconds",
        bin_size_s=0.25,
        response_metric="peak_response",
    ),
)

# =============================================================================
# 4. CONE-MOSAIC REGULARITY SETTINGS
# =============================================================================
#
# For each cell:
#
# 1. Find RF centre from one selected RF channel.
#
# 2. Use the SAME RF left-right flip and alignment logic
#    as the RF/cone alignment script.
#
# 3. Convert RF centre into the aligned cone-image coordinate system.
#
# 4. Take a square cone-mosaic crop around that RF centre.
#
# 5. Run autocorrelation.
#
# 6. Detect the first shell of autocorrelation peaks.
#
# 7. Calculate:
#
#       cone_regularity_index
#       cone_spacing_um
#       cone_autocorr_strength
#       cone_spacing_consistency
#       etc.
#
#
# CONE_CROP_SIZE_UM is the SIDE LENGTH of the crop.
#
# Example:
#
#     CONE_CROP_SIZE_UM = 200
#
# means:
#
#     200 µm × 200 µm
#
# centred on the RF.


CONE_IMAGE_PATH = Path(
    r"C:\Users\Laura Steel\Box\SUSSEX\Experiments"
    r"\Zebrafish\Experiments\Imaging"
    r"\20260814_dragonfly_microscope"
    r"\14_08_2026_cones.jpg"
)


ALIGNMENT_CACHE_PATH = Path(
    r"F:\Laura\zebrafish_14_08_2026"
    r"\alignment_videos"
    r"\zebrafish_14_08_2026_phase_00_alignment_cache.npz"
)


# -------------------------------------------------------------
# Which RF channel should define each cell's RF centre?
# -------------------------------------------------------------

CONE_RF_CHANNEL = "32px_15Hz_20mins_shuffle_x4"


# -------------------------------------------------------------
# Which RF map should be used to locate the RF peak?
# -------------------------------------------------------------

CONE_RF_VARIABLE = "cm_most_important"


# -------------------------------------------------------------
# Physical size of cone crop
# -------------------------------------------------------------

CONE_CROP_SIZE_UM = 200.0


# -------------------------------------------------------------
# Cone regularity / autocorrelation parameters
# -------------------------------------------------------------

CONE_REGULARITY_PARAMETERS = dict(
    rf_variable=CONE_RF_VARIABLE,
    crop_size_um=CONE_CROP_SIZE_UM,
    # ---------------------------------------------------------
    # RF coordinate scale
    # ---------------------------------------------------------
    #
    # MUST match the alignment script.
    dataset_assumed_um_per_px=2.0,
    displayed_stim_height_px=800,
    # ---------------------------------------------------------
    # Autocorrelation preprocessing
    # ---------------------------------------------------------
    #
    # These reproduce the settings from your original
    # cone autocorrelation script.
    preprocess_sigma_px=13.0,
    ac_peak_min_distance_px=10,
    centre_exclusion_radius_px=20,
    max_num_ac_peaks=80,
    ac_peak_threshold_rel=0.08,
    first_shell_tol=0.25,
    # ---------------------------------------------------------
    # QC
    # ---------------------------------------------------------
    min_first_shell_peaks=4,
    min_valid_crop_fraction=0.95,
    # ---------------------------------------------------------
    # Image resolution
    # ---------------------------------------------------------
    #
    # None means preserve approximately native
    # physical resolution of the cone image.
    resample_um_per_px=None,
)


# =============================================================================
# 5. HELPERS FOR LOADING EXISTING FEATURES
# =============================================================================


def _load_existing_feature_table(
    cells: pd.DataFrame,
    csv_path: Path,
    nc_path: Path,
) -> pd.DataFrame:
    """
    Load the existing FFF feature dataset.

    Priority:

        1. fff_features.csv
        2. fff_data.nc

    If neither exists, start a new dataframe.

    The current recording's cell list is always used as the master list.
    """

    if csv_path.exists():
        print(f"\nLoading existing feature table:\n" f"{csv_path}")

        existing = pd.read_csv(csv_path)

    elif nc_path.exists():
        print(f"\nLoading existing feature dataset:\n" f"{nc_path}")

        existing_ds = xr.load_dataset(nc_path)

        existing = existing_ds.to_dataframe().reset_index()

        existing_ds.close()

    else:
        print("\nNo existing FFF dataset found." "\nStarting a new feature table.")

        return cells.copy()

    existing["cell_index"] = existing["cell_index"].astype(int)

    existing = existing.drop_duplicates(subset="cell_index")

    # ---------------------------------------------------------
    # Preserve existing features but use the current
    # recording cell list as the master list.
    # ---------------------------------------------------------

    return cells.merge(
        existing,
        on="cell_index",
        how="left",
        validate="one_to_one",
    )


# =============================================================================
# 6. REPLACE ONLY ONE FEATURE BLOCK
# =============================================================================


def _replace_feature_columns(
    master: pd.DataFrame,
    feature: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """
    Add / replace the columns from ONE extractor.

    All unrelated existing columns remain untouched.
    """

    if feature is None or feature.empty:
        print(
            f"{name}: no features returned." "\nExisting columns were left unchanged."
        )

        return master

    if "cell_index" not in feature.columns:
        raise KeyError(f"{name} feature table has no " f"'cell_index' column.")

    feature = feature.copy()

    feature["cell_index"] = feature["cell_index"].astype(int)

    # ---------------------------------------------------------
    # Check one row per cell
    # ---------------------------------------------------------

    if feature["cell_index"].duplicated().any():
        duplicated = feature.loc[
            feature["cell_index"].duplicated(),
            "cell_index",
        ].unique()

        raise ValueError(f"{name} returned duplicate cells: " f"{duplicated}")

    # ---------------------------------------------------------
    # Find feature columns
    # ---------------------------------------------------------

    new_columns = [column for column in feature.columns if column != "cell_index"]

    # ---------------------------------------------------------
    # Remove OLD versions of those exact columns
    # ---------------------------------------------------------

    old_matching_columns = [
        column for column in new_columns if column in master.columns
    ]

    if old_matching_columns:
        print(
            f"{name}: replacing "
            f"{len(old_matching_columns)} "
            f"existing feature columns."
        )

        master = master.drop(columns=old_matching_columns)

    print(f"{name}: " f"{len(feature)} cells, " f"{len(new_columns)} feature columns")

    # ---------------------------------------------------------
    # Add newly calculated columns
    # ---------------------------------------------------------

    return master.merge(
        feature,
        on="cell_index",
        how="left",
        validate="one_to_one",
    )


# =============================================================================
# 7. SAVE DATASET
# =============================================================================


def save_fff_dataset(
    dataframe: pd.DataFrame,
    csv_path: Path,
    nc_path: Path,
):
    dataframe = dataframe.sort_values("cell_index").reset_index(drop=True)

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    # ---------------------------------------------------------
    # NetCDF
    # ---------------------------------------------------------

    dataset = xr.Dataset.from_dataframe(dataframe.set_index("cell_index"))

    dataset.attrs.update(
        {
            "description": (
                "Per-cell RF-linked functional " "and cone-mosaic feature dataset"
            ),
            "overview_path": str(OVERVIEW_PATH),
            "noise_data_path": str(NOISE_DATA_PATH),
            "cone_rf_channel": str(CONE_RF_CHANNEL),
            "cone_crop_size_um": float(CONE_CROP_SIZE_UM),
        }
    )

    dataset.to_netcdf(nc_path)

    return dataset


# =============================================================================
# 8. BUILD / UPDATE FEATURES
# =============================================================================


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Create output directory
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CSV_PATH = OUTPUT_DIR / "fff_features.csv"

    NC_PATH = OUTPUT_DIR / "fff_data.nc"

    # ---------------------------------------------------------
    # Master list of every cell in recording
    # ---------------------------------------------------------

    cells = all_cell_indices(OVERVIEW_PATH)

    cells_df = pd.DataFrame({"cell_index": cells.astype(int)})

    print(f"\nRecording contains " f"{len(cells_df)} cells.")

    # ---------------------------------------------------------
    # Load EXISTING FFF dataset
    # ---------------------------------------------------------
    #
    # This is the important part.
    #
    # Existing csteps / SCF / chirp / moving-bar
    # features are loaded FIRST.
    #
    # Therefore setting:
    #
    #     RUN_CSTEPS = False
    #
    # means:
    #
    #     preserve existing csteps columns
    #
    # NOT:
    #
    #     delete csteps columns
    # ---------------------------------------------------------

    fff_df = _load_existing_feature_table(
        cells=cells_df,
        csv_path=CSV_PATH,
        nc_path=NC_PATH,
    )

    # =============================================================================
    # CONTRAST STEPS
    # =============================================================================

    if RUN_CSTEPS:
        print(
            "\n============================="
            "\nExtracting contrast steps"
            "\n============================="
        )

        csteps_df = calculate_csteps_features(
            overview_path=OVERVIEW_PATH,
            stimulus_id=CSTEPS_STIMULUS_ID,
            **CSTEPS_PARAMETERS,
        )

        fff_df = _replace_feature_columns(
            fff_df,
            csteps_df,
            "Contrast steps",
        )

    else:
        print("\nContrast steps: SKIPPED")

    # =============================================================================
    # SINGLE CHROMATIC FLASH
    # =============================================================================

    if RUN_SCF:
        print(
            "\n============================="
            "\nExtracting SCF"
            "\n============================="
        )

        scf_df = calculate_scf_features(
            overview_path=OVERVIEW_PATH,
            stimulus_id=SCF_STIMULUS_ID,
            **SCF_PARAMETERS,
        )

        fff_df = _replace_feature_columns(
            fff_df,
            scf_df,
            "SCF",
        )

    else:
        print("SCF: SKIPPED")

    # =============================================================================
    # CHIRP
    # =============================================================================

    if RUN_CHIRP:
        print(
            "\n============================="
            "\nExtracting chirp"
            "\n============================="
        )

        chirp_df = calculate_chirp_features(
            overview_path=OVERVIEW_PATH,
            stimulus_id=CHIRP_STIMULUS_ID,
            **CHIRP_PARAMETERS,
        )

        fff_df = _replace_feature_columns(
            fff_df,
            chirp_df,
            "Chirp",
        )

    else:
        print("Chirp: SKIPPED")

    # =============================================================================
    # MOVING BAR
    # =============================================================================

    if RUN_MOVING_BAR:
        print(
            "\n============================="
            "\nExtracting moving bar"
            "\n============================="
        )

        moving_bar_df = calculate_moving_bar_features(
            overview_path=OVERVIEW_PATH,
            stimulus_id=MOVING_BAR_STIMULUS_ID,
            **MOVING_BAR_PARAMETERS,
        )

        fff_df = _replace_feature_columns(
            fff_df,
            moving_bar_df,
            "Moving bar",
        )

    else:
        print("Moving bar: SKIPPED")

    # =============================================================================
    # MOVING EDGE
    # =============================================================================

    if RUN_MOVING_EDGE:
        print(
            "\n============================="
            "\nExtracting moving edge"
            "\n============================="
        )

        moving_edge_df = calculate_moving_edge_features(
            overview_path=OVERVIEW_PATH,
            conditions=MOVING_EDGE_CONDITIONS,
        )

        fff_df = _replace_feature_columns(
            fff_df,
            moving_edge_df,
            "Moving edge",
        )

    else:
        print("Moving edge: SKIPPED")

    # =============================================================================
    # CONE-MOSAIC REGULARITY
    # =============================================================================

    if RUN_CONE_REGULARITY:
        print(
            "\n============================="
            "\nExtracting cone regularity"
            "\n============================="
        )

        cone_df = calculate_cone_regularity_features(
            noise_data_path=NOISE_DATA_PATH,
            cone_image_path=CONE_IMAGE_PATH,
            alignment_cache_path=ALIGNMENT_CACHE_PATH,
            rf_channel=CONE_RF_CHANNEL,
            **CONE_REGULARITY_PARAMETERS,
        )

        fff_df = _replace_feature_columns(
            fff_df,
            cone_df,
            "Cone mosaic",
        )

    else:
        print("Cone mosaic: SKIPPED")

    # =============================================================================
    # SUMMARY
    # =============================================================================

    print(
        "\n===================================="
        "\nFINAL FEATURE DATAFRAME"
        "\n===================================="
    )

    print(fff_df)

    print(f"\nShape: " f"{fff_df.shape}")

    print(f"Feature columns: " f"{len(fff_df.columns) - 1}")

    # =============================================================================
    # SAVE UPDATED DATASET
    # =============================================================================

    fff_dataset = save_fff_dataset(
        dataframe=fff_df,
        csv_path=CSV_PATH,
        nc_path=NC_PATH,
    )

    print(
        "\n===================================="
        "\nSAVED UPDATED FILES"
        "\n===================================="
    )

    print(CSV_PATH)

    print(NC_PATH)

    print("\nXarray dataset:")

    print(fff_dataset)
