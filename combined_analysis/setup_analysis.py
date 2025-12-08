from pathlib import Path
from organize.configs import (
    Recording_Config,
    Analysis_Pipeline,
    Circular_Reduction_Config,
)
from rf_analysis import calculate_quality
from combined_analysis.reduce_all import perform_circular_reduction
from location.x_array import x_y_and_scale

# rec_object = Recording_Config(
#     root_path=Path(
#         "/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_13_11_2025/Phase_00"
#     ),
# )
# rec_object.add_channel(1, "610_nm", colour="red")
# rec_object.add_channel(2, "535_nm", colour="green")
# # %%
# pipeline = Analysis_Pipeline("noise_analysis", rec_object)
#
# # %%
# pipeline.save()


# %%
pipeline = Analysis_Pipeline.load(
    Path(
        "/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/chicken_13_11_2025/Phase_00/noise_analysis/tasks_configs.json"
    )
)

# %%
circular_reduction_config = Circular_Reduction_Config(
    recording_config=pipeline.configs[Recording_Config],
    cut_size_um=x_y_and_scale(800, 800),
    degree_bins=10,
)
pipeline.add_config(circular_reduction_config)

# %%
pipeline.schedule([perform_circular_reduction])
# %%
pipeline.run()
