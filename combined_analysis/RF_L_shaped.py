from pathlib import Path

from organize.configs import (
    Recording_Config,
    Analysis_Pipeline,
    Circular_Reduction_Config,
    Collapse_2d_Config,
)

from location.x_array import x_y_and_scale
from combined_analysis.reduce_all import sta_2d_cov_collapse, circular_reduction
from rf_analysis.calculate_quality_L_shape import calculate_rf_quality_mask_safe
from stats.summary import calculate_stats


# ---------------------------------------------------------------------
# Wrapper so the pipeline still sees the task name as "calculate_rf_quality"
# This matters because sta_2d_cov_collapse depends on calculate_rf_quality.
# ---------------------------------------------------------------------
def calculate_rf_quality(
    recording_config: Recording_Config,
    analysis_folder: str = "noise_analysis",
    cpus: int = 4,
):
    return calculate_rf_quality_mask_safe(
        recording_config=recording_config,
        analysis_folder=analysis_folder,
        cpus=cpus,
        ignore_zero_variance_pixels=True,
    )


if __name__ == "__main__":
    # %% Creating Recording_Config object

    rec_object = Recording_Config(
        root_path=Path(r"F:\Laura\zebrafish_14_08_2026\Phase_01"),
    )

    rec_object.add_channel(
        stimulus_id=13,
        name="32px_15Hz_20mins_L_shaped_shuffle",
        colour="red",
    )

    # %% Create a NEW pipeline folder
    # Do not use old "noise_analysis" for this test

    pipeline = Analysis_Pipeline(
        "noise_analysis_L_shape_safe_quality_400um",
        recording_config=rec_object,
        other_configs=[],
    )

    # %% Define collapse settings

    collapse_2d_config = Collapse_2d_Config(
        recording_config=pipeline.configs[Recording_Config],
        cut_size_um=x_y_and_scale(
            400,
            400,
        ),
        threshold=1.5,  # IMPORTANT: lower threshold for L-shaped masked stimulus
    )

    circular_reduction_config = Circular_Reduction_Config(
        degree_bins=10,
    )

    # %% Add configs to pipeline

    pipeline.add_config(collapse_2d_config)
    pipeline.add_config(circular_reduction_config)

    # %% Schedule tasks

    pipeline.schedule([calculate_rf_quality])
    pipeline.schedule([sta_2d_cov_collapse])
    pipeline.schedule([circular_reduction])
    pipeline.schedule([calculate_stats])

    # %% Run

    pipeline.run()
    pipeline.save()
