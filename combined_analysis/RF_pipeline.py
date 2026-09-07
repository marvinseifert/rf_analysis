from pathlib import Path
from organize.configs import (
    Recording_Config,
    Analysis_Pipeline,
    Circular_Reduction_Config,
    Collapse_2d_Config,
)
from location.x_array import x_y_and_scale
from combined_analysis.reduce_all import sta_2d_cov_collapse, circular_reduction
from rf_analysis.calculate_quality import calculate_rf_quality
from stats.summary import calculate_stats

if __name__ == "__main__":
    # %% Creating Recording_Config object

    rec_object = Recording_Config(
        root_path=Path(r"F:\Laura\zebrafish_14_08_2026\Phase_00"),
    )
    rec_object.add_channel(
        stimulus_id=6, name="32px_15Hz_20mins_shuffle_x4", colour="red"
    )
    rec_object.add_channel(
        stimulus_id=4, name="8px_15Hz_35mins_shuffle_x8", colour="green"
    )
    rec_object.add_channel(
        stimulus_id=10, name="6px_15Hz_30mins_shuffle_x6", colour="blue"
    )

    # %% create pipeline
    pipeline = Analysis_Pipeline(
        "noise_analysis",
        recording_config=rec_object,
        other_configs=[],
    )
    pipeline.save()
    # %% define some more parameters
    collapse_2d_config = Collapse_2d_Config(
        recording_config=pipeline.configs[Recording_Config],
        cut_size_um=x_y_and_scale(
            600, 600
        ),  # cut size in micrometers!!! # this will determine edge filling dimensions and dataframe size
    )
    circular_reduction_config = Circular_Reduction_Config(
        degree_bins=10,
    )

    # %% # If you want to re-run something just take it out of the finished tasks list (i.e. if it is in the list, it won't re-run)
    # pipeline._finished_tasks = ["calculate_rf_quality"]
    # %% add configs to pipeline
    pipeline.add_config(collapse_2d_config)
    pipeline.add_config(circular_reduction_config)

    # %%
    pipeline.schedule([calculate_rf_quality])
    pipeline.schedule([sta_2d_cov_collapse])
    pipeline.schedule([circular_reduction])
    pipeline.schedule([calculate_stats])
    pipeline.run()
    pipeline.save()
    # %%
