# legacy/: the v1 versions (before 18 November 2026)

These are the original scripts that were replaced by the v2 pipeline. They're kept for reference and for reproducing old figures. **Use the scripts outside `legacy/` for new work.** The changes are explained in `laura_scripts/Alignment_pipeline/V2_CHANGES.md` and `ORIENTATION.md`.

Each file here has the same name and subfolder as its replacement, e.g. `legacy/OSI/OS_analysis.py` was replaced by `OSI/OS_analysis.py`.

What's different about v1:
- It reads the old `noise_data.nc`, whose RF maps have x and y swapped.
- The alignment flips RFs left-right only, so they are upside down relative to the cones.
- It uses the old (hard-coded) stimulus sizes.
- The cone regularity reports a spurious ~4.9 µm spacing.

The only edits made to these files when they were moved are import paths, so they keep using v1 code:
- `Alignment_pipeline/run_single_cell_cone_rf_connectivity_plot_cell.py` imports `laura_scripts.legacy.Alignment_pipeline…`
- `RF_analysis/main_rf_analysis_LS.py` imports `laura_scripts.legacy.RF_analysis.receptive_field_functions`
- `Seaborn/fff_feature_pipeline/build_fff_dataset.py` also looks in the main `fff_feature_pipeline` folder for the unchanged extractors (chirp, csteps, SCF, moving bar/edge, `fff_common`).

Their full history is in git (`git log --follow <file>`).
