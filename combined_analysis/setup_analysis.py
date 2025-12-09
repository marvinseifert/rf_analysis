from pathlib import Path
from organize.configs import (
    Recording_Config,
    Analysis_Pipeline,
    Circular_Reduction_Config,
    Collapse_2d_Config,
)
from location.x_array import x_y_and_scale
from combined_analysis.reduce_all import sta_2d_cov_collapse, circular_reduction

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
        "/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Marvin/"
        "chicken_13_11_2025/Phase_00/noise_analysis/tasks_configs.json"
    )
)

# # %%
# collapse_2d_config = Collapse_2d_Config(
#     recording_config=pipeline.configs[Recording_Config],
#     cut_size_um=x_y_and_scale(800, 800),
# )
# pipeline.add_config(collapse_2d_config)
#
# circular_reduction_config = Circular_Reduction_Config(
#     degree_bins=10,
# )
# pipeline.add_config(circular_reduction_config)
# # %%
# pipeline.schedule([sta_2d_cov_collapse])
# %%
pipeline.schedule([circular_reduction])
pipeline.run()
