# legacy/: v1 RF-pipeline code (before 18 November 2026)

These are the original versions of the shared RF-pipeline files. They were replaced by the v2 versions under the same names. They're kept as a backup and for reproducing old results. **Use the files outside `legacy/` for new work.** The changes are explained in `laura_scripts/Alignment_pipeline/V2_CHANGES.md` and `ORIENTATION.md`.

| v1 (here) | Replaced by (v2) |
|---|---|
| `legacy/combined_analysis/reduce_all.py` | `combined_analysis/reduce_all.py` |
| `legacy/combined_analysis/RF_pipeline.py` | `combined_analysis/RF_pipeline.py` |
| `legacy/loading/load_sta.py` | `loading/load_sta.py` |
| `legacy/stats/summary.py` | `stats/summary.py` |

What's different about v1:
- It writes `noise_data.nc` with the RF maps' x and y swapped.
- Its outlines are 1.41× too large, and `stats/summary.py` would make RF areas pixel_size² too large.

The only edits made to these files when they were moved: imports point at the other `legacy/` files (so v1 keeps using v1 code), and each has a header saying it is legacy.

Note: `organize/configs.py` imports the **v2** functions, so `Analysis_Pipeline.load()` always runs v2. To run v1, run `legacy/combined_analysis/RF_pipeline.py` top to bottom.

Laura's v1 analysis scripts are in `laura_scripts/legacy/`. Marvin's `combined_analysis/combined_plotting.py` and `clustering_pipeline.py` are unchanged (v1 data); their v2 copies are `combined_plotting_v2.py` and `clustering_pipeline_v2.py`.

Full history of every file: `git log --follow <file>`.
