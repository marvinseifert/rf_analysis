"""
RF_pipeline_colour_sequence.py — RF pipeline (v2) for interleaved colour noise: one noise movie
with the LED colour switching every N frames (e.g. 15/10/2026 P0 stim 5, 22/10/2026 P1 stim 4).

Each colour is analysed as its own channel, so the output (noise_data_v2.nc in
<root>/<analysis_folder>) has one channel per colour, like
zebrafish_15_10_2026/Phase_00/noise_analysis_colour_QI10 and zebrafish_22_10_2026/Phase_01/noise_analysis.

Steps:
  1. rf_torch/main_sta_colour_sequence.py (rf_torch repo; run with this repo's poetry env, which has
     torch + CUDA): per-colour STAs in <stimulus>_white_idx_<i> / <stimulus>_led_<nm>_idx_<i>.
  2. This script: quality, covariance collapse, polar outlines, stats for every colour channel.

Run from the rf_analysis repo root (needs the __main__ guard for multiprocessing):
    poetry run python combined_analysis/RF_pipeline_colour_sequence.py
Runtime: about as long as the normal RF pipeline per channel (~10-20 min per colour for ~280 cells).
"""
from pathlib import Path

from combined_analysis.reduce_all import circular_reduction, sta_2d_cov_collapse
from location.x_array import x_y_and_scale
from organize.configs import (
    Analysis_Pipeline,
    Circular_Reduction_Config,
    Collapse_2d_Config,
    Recording_Config,
)
from rf_analysis.calculate_quality import calculate_rf_quality
from stats.summary import calculate_stats

# ---------------------------------------------------------------- settings
ROOT = Path("/mnt/mea_nas/Laura/zebrafish_22_10_2026/Phase_01")
STIMULUS_ID = 4  # stimulus_index of the colour noise in the overview
COLOURS = ["420", "500", "610"]  # colours in the sequence ("white" or nm); order = channel order
ANALYSIS_FOLDER = "noise_analysis_colour_QI10"
QUALITY_THRESHOLD = 10.0  # outlines/stats only for channels with quality >= this (15/10 used 10)
CUT_SIZE_UM = 600  # cutout around each RF (both earlier colour analyses used 600 µm)
# 15/10/2026 P0 was: ROOT .../zebrafish_15_10_2026/Phase_00, STIMULUS_ID 5,
# COLOURS ["white", "610", "500", "420"].

if __name__ == "__main__":
    rec_object = Recording_Config(root_path=ROOT)
    rec_object.add_colour_sequence_channels(stimulus_id=STIMULUS_ID, colours=COLOURS)
    for name in rec_object.channel_names:
        print(name, "->", rec_object.channel_configs[name].root_path)

    pipeline = Analysis_Pipeline(
        ANALYSIS_FOLDER,
        recording_config=rec_object,
        other_configs=[],
    )
    pipeline.save()
    pipeline.add_config(
        Collapse_2d_Config(
            recording_config=pipeline.configs[Recording_Config],
            cut_size_um=x_y_and_scale(CUT_SIZE_UM, CUT_SIZE_UM),
            threshold=QUALITY_THRESHOLD,
        )
    )
    pipeline.add_config(Circular_Reduction_Config(degree_bins=10))

    # To re-run a step, take it out of pipeline._finished_tasks.
    pipeline.schedule([calculate_rf_quality])
    pipeline.schedule([sta_2d_cov_collapse])
    pipeline.schedule([circular_reduction])
    pipeline.schedule([calculate_stats])
    pipeline.run()
    pipeline.save()
