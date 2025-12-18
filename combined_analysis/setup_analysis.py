from pathlib import Path
from organize.configs import (
    Recording_Config,
    Analysis_Pipeline,
    Circular_Reduction_Config,
    Collapse_2d_Config,
)
from location.x_array import x_y_and_scale

rec_object = Recording_Config(
    root_path=Path("/home/mawa/nas_a/Marvin/chicken_13_05_2025/Phase_00"),
)
rec_object.add_channel(2, "460_nm", colour="blue")
rec_object.add_channel(stimulus_id=7, name="535_nm", colour="green")
rec_object.add_channel(stimulus_id=3, name="610_nm", colour="red")
rec_object.add_channel(stimulus_id=11, name="white", colour="grey")

# %%
# pipeline = Analysis_Pipeline(
#     "noise_analysis_test",
#     recording_config=rec_object,
#     other_configs=[],
# )
# # # # #
# # # # # # %%
# pipeline.save()

# %%
pipeline = Analysis_Pipeline.load(
    Path(
        "/home/mawa/nas_a/Marvin/chicken_13_05_2025/Phase_00/noise_analysis_test/tasks_configs.json"
    )
)


# # %%
collapse_2d_config = Collapse_2d_Config(
    recording_config=pipeline.configs[Recording_Config],
    cut_size_um=x_y_and_scale(800, 800),
)
circular_reduction_config = Circular_Reduction_Config(
    degree_bins=10,
)
# pipeline.add_config(collapse_2d_config)
# pipeline.add_config(circular_reduction_config)
# pipeline.schedule([calculate_rf_quality])
# pipeline.schedule([sta_2d_cov_collapse])
# pipeline.schedule([circular_reduction])
# pipeline.schedule([calculate_stats])
pipeline._finished_tasks = [
    "calculate_rf_quality",
]
pipeline.run()
#
pipeline.save()
# pipeline.add_config(circular_reduction_config)
# # %%
# pipeline.schedule([sta_2d_cov_collapse])
# %%
# pipeline.schedule([calculate_stats])
