from pathlib import Path
from organize.configs import (
    Recording_Config,
    Analysis_Pipeline,
    Circular_Reduction_Config,
    Collapse_2d_Config,
)
from stats.summary import calculate_stats
from location.x_array import x_y_and_scale
from combined_analysis.reduce_all import sta_2d_cov_collapse, circular_reduction
from rf_analysis.calculate_quality import calculate_rf_quality

rec_object = Recording_Config(
    root_path=Path(
        "/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Laura/zebrafish_02_12_2025/Phase_01"
    ),
)
rec_object.add_channel(4, "white_4px", colour="white")
rec_object.add_channel(stimulus_id=8, name="white_12px", colour="white")

# # # %%
# pipeline = Analysis_Pipeline(
#     "noise_analysis_test",
#     recording_config=rec_object,
#     other_configs=[],
# )
# # # #
# # # # # %%
# pipeline.save()

# %%
pipeline = Analysis_Pipeline.load(
    Path(
        "/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Laura/zebrafish_02_12_2025/Phase_01/noise_analysis_test/tasks_configs.json"
    )
)
pipeline._finished_tasks = [
    "calculate_rf_quality",
    "sta_2d_cov_collapse",
]

# # %%
collapse_2d_config = Collapse_2d_Config(
    recording_config=pipeline.configs[Recording_Config],
    cut_size_um=x_y_and_scale(800, 800),
)
circular_reduction_config = Circular_Reduction_Config(
    degree_bins=10,
)
pipeline.add_config(collapse_2d_config)
pipeline.add_config(circular_reduction_config)
pipeline.schedule([calculate_rf_quality])
pipeline.schedule([sta_2d_cov_collapse])
pipeline.schedule([circular_reduction])
pipeline.schedule([calculate_stats])
pipeline.run()
#
pipeline.save()
# pipeline.add_config(circular_reduction_config)
# # %%
# pipeline.schedule([sta_2d_cov_collapse])
# %%
# pipeline.schedule([calculate_stats])
