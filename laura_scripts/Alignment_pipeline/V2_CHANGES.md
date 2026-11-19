# v2 pipeline: what changed and why

*Written 18 November 2026. Covers the RF pipeline, the RF/cone alignment and the analyses that use them. For every rotation and flip in detail, see `ORIENTATION.md` next to this file.*

## 1. Summary

Starting from the old cone-alignment work, we found that the RF maps in every `noise_data.nc` were stored with x and y swapped. The alignment script had been correcting this with a left-right flip that was only right in an xarray plot (which draws y pointing up). In the alignment figures, which draw y pointing down, the RFs were therefore upside down relative to the cones. We also found two size errors, a stimulus-centre error for 14/08, a cone-spacing artefact, and a stimulus pixel size of 2.5 µm (now 2.0 µm) for 14/05 and 15/05.

**Where the code is:** in `laura_scripts`, the v2 scripts now carry the original names, and the original (v1) versions are in `laura_scripts/legacy/` (same names and subfolders; see `legacy/README.md`). Since 18/11/2026 the same applies to the shared RF-pipeline files: `combined_analysis/reduce_all.py`, `combined_analysis/RF_pipeline.py`, `loading/load_sta.py` and `stats/summary.py` are the v2 versions, and the v1 versions are in the top-level `legacy/` folder (same subfolders; see `legacy/README.md`). Every script opens RF data through `loading/noise_data.py` (`load_noise_data`), which always opens `noise_data_v2.nc` and refuses v1 data. Marvin's `combined_plotting.py` and `clustering_pipeline.py` are unchanged; their v2 copies are `combined_plotting_v2.py` and `clustering_pipeline_v2.py`. On the NAS, new `_v2` data files sit next to the originals. The only original data files edited are the 14/05 and 15/05 pixel-size settings, each with a backup.

## 2. Problems found and how they were fixed

| # | Problem | Effect | Fix |
|---|---|---|---|
| 1 | `reduce_all.py` pastes each RF map into the full grid with x and y swapped (lines 566–583). | `rms` / `cm_most_important` swapped relative to the stimulus, `kernel.npy` and `positions`. Outlines, `tilt` and `angle` computed in the swapped frame. | `reduce_all.py` (v2) saves maps the right way round. `convert_noise_data_v2.py` fixes existing files (exact, no recomputation). |
| 2 | The alignment applied a left-right flip to the stored maps. The flip came from an L test plotted with xarray, which draws y **up**. | In the alignment and cone figures (y down), RFs were upside down relative to the cones. The single-cell marker didn't match the RF (up to ~1 mm apart). | One transform per recording: stimulus (x, y) → projected (−y, −x), i.e. the OpenGL up-down flip plus the projector's 90° anticlockwise rotation (confirmed with the L-shaped noise); for recordings from 8 Oct 2026 see #11. Data are never flipped. |
| 3 | Cell 1 of `RF_alignment.py` hard-coded the displayed stimulus as 800 × 1000 when calculating the stimulus centre from the clicked edge. | For square (800 × 800) displays the centre was off sideways: **14/08 by 156 µm** (RFs in the dark strip). 07/08 and 22/07 were cached correctly. 14/05 and 15/05 are 800 × 1000 and were correct. | The centre is always calculated from the clicked edge plus the recording's own displayed size. |
| 4 | Polar outlines: 212 radius bins spread over 150 px (0.71 px per bin) but converted to µm as 1 px per bin. | Outline radii, `tir` and `center_shift` 1.41× too large; `center_size_mm2` / `surround_size_mm2` ~2× too large. | 1-px radius bins in `reduce_all.py` (v2); existing files recalculated by the converter. |
| 5 | `stats/summary.py` multiplies RF areas by pixel size² although the outlines are already in µm. | Would make areas 4× too large (2 µm pixels). **Your existing files did not have this**; your Windows version had already fixed it. | `stats/summary.py` (v2). The v1 file, with an added warning, is `legacy/stats/summary.py`. |
| 6 | Cone regularity: a fixed 20 px (4.9 µm) centre exclusion in the autocorrelation. | Elongated cones made the central peak stick out of the excluded disc; its tips were taken as the "first shell". Spacing ≈ 4.9 µm for every cell; regularity ≈ 0 (old `fff_data.nc` included). | `extract_cone_regularity.py` excludes the central peak by its shape and rejects peaks touching the excluded region. Median spacing now 12.8–15.7 µm; most good cells have a full first shell. |
| 7 | Stimulus sizes and pixel size typed into several scripts (some with different values). | Easy to get out of step: the cause of #3. | Pixel size and raw stimulus size from `noise_params.json`; displayed size stored once per recording in `…_alignment_cache_v2.npz`. Cell 1 shows a red box on the projection photo to check the displayed size by eye. |
| 8 | `noise_pixel_size` was 2.5 µm for 14/05 and 15/05, but the stimulus measured on the MEA is ~1.9 µm per pixel in every recording. | RF sizes for those two recordings ~25% too large (areas ~1.6×) compared with the later recordings. | Set to 2.0 µm (your choice). See section 4. |
| 9 | `merged_stimuli_RFs.py` labels its own combined maps' x and y the wrong way round. | Its figures showed maps transposed. | Labels corrected in `merged_stimuli_RFs.py`. |
| 10 | `circular_reduction`: on a shape mismatch it printed a warning, then reused the previous cell's data. | Silent wrong outline for that cell (rare). | Skips the cell in v2. |
| 11 | The stimulus software changed on 8 October 2026 (`fps_py`): it no longer adds the OpenGL up-down flip, so only the projector's 90° anticlockwise rotation remains. | Recordings from 8 Oct 2026 (15/10, 22/10, 28/10) would come out mirrored left-right on the cones if the old transform were used. | The transform is chosen by recording date: before 8 Oct 2026 (x, y) → (−y, −x); from 8 Oct 2026 (x, y) → (y, −x). All five aligned recordings (May–August) are unchanged. |

## 3. Files

### v2 files

| Stage | File | What it is |
|---|---|---|
| RF pipeline | `combined_analysis/RF_pipeline.py` | Driver for new recordings (v1: `legacy/combined_analysis/RF_pipeline.py`) |
| | `combined_analysis/reduce_all.py` | RF cutouts, covariance/RMS maps, outlines (fixes 1, 4, 10) (v1: `legacy/…`) |
| | `loading/load_sta.py` | STA loading (axis labels corrected) (v1: `legacy/…`) |
| | `stats/summary.py` | RF size, tilt, angle, TIR, centre shift (fix 5) (v1: `legacy/…`) |
| | `loading/noise_data.py` | `load_noise_data()`: the one way to open RF data; always `noise_data_v2.nc`, checked |
| | `organize/configs.py` | Imports the v2 functions, so a re-loaded saved pipeline also runs v2 |
| | `combined_analysis/convert_noise_data_v2.py`, `convert_all_noise_data_v2.sh` | Convert an existing `noise_data.nc` → `noise_data_v2.nc` (`--pixel-size 2.0` rescales a 2.5 µm recording) |
| | `combined_analysis/relabel_pixel_size_settings.py` | Change a recording's stimulus pixel size in its settings files (e.g. 2.5 → 2.0 µm), backing each file up as `…_orig_2.5um.json` first |
| Marvin's plots | `combined_analysis/combined_plotting_v2.py`, `clustering_pipeline_v2.py` | v2 copies: read v2 data, maps drawn +y down, polar plots clockwise (originals unchanged) |
| Alignment | `laura_scripts/Alignment_pipeline/alignment_geometry.py` | **Shared rules**: the one RF rotation, the geometry loader, stimulus sizes from `noise_params.json` / the v2 cache, the displayed-size check box, the v2 file check |
| | `…/Alignment_pipeline/RF_alignment.py`, `alignment_recordings.py` | Interactive alignment (clicks, overlays); per-recording settings |
| | `…/Alignment_pipeline/plot_population_overlays_all_recordings.py`, `plot_top_quality_cells.py` | Population and top-N single-cell RF overlays on the cone mosaic, all recordings with a v2 cache |
| | `…/Alignment_pipeline/cone_rf_connectivity_functions.py`, `run_single_cell_cone_rf_connectivity_plot_cell.py` | Single-cell cone connectivity |
| | `…/Alignment_pipeline/ORIENTATION.md`, `V2_CHANGES.md` | Documentation |
| Cone features | `laura_scripts/Seaborn/fff_feature_pipeline/extract_cone_regularity.py` | Cone spacing / regularity (fixes 2, 3, 6) |
| | `…/build_fff_dataset.py` | Writes `fff_features_v2.csv` / `fff_data_v2.nc` |
| | `…/crystallinity_index_visualisation.py`, `seaborn_plots.py` | Cone-regularity and RF-vs-FFF figures |
| Figures and analyses | `laura_scripts/Summary_figures/summary_plot_single_cell.py` | Per-cell summary (writes to `separate_cell_plots_v2`) |
| | OSI: `OS_analysis.py`, `OS_analysis_stimulus_filtered_cells.py`, `moving_edge_analysis.py` | Read `noise_data_v2.nc` (only `tilt` used) |
| | Elongated RFs: `Angle_histogram.py`, `chirp_plots_by_tilt.py`, `ff_plots_by_tilt.py`, `elongation_cells_poster.py` | Read v2 data; maps drawn y-down |
| | `RF_analysis/`: `receptive_field_functions.py`, `main_rf_analysis_LS.py`, `merged_stimuli_RFs.py`, `weighted_STA.py`; `Summary_figures/poster_plot_12panel.py`; `Scratch/`: `test.py`, `test_rms_cm_single_plot.py`, `test_datashader_plots.py` | Read v2 data; maps drawn y-down (stimulus-file layout) |

`laura_scripts` paths are the v2 versions under their original names (the v1 versions are in `laura_scripts/legacy/`). The RF-pipeline files keep `_v2` names because they sit next to shared originals.

### Folder layout of `laura_scripts` (18 November 2026)
`Alignment_pipeline/` (v2 alignment; `electrode_tools/` for electrode location and image scripts), `RF_analysis/`, `Subunits/`, `Cone_imaging/`, `OSI/`, `Elongated receptive fields/`, `Seaborn/fff_feature_pipeline/`, `Summary_figures/`, `Scratch/` (tests and quick explorations), and `legacy/` (the v1 versions, in the same layout). No scripts are left at the top level.

### Original files changed
- `stats/summary.py`: warning comment and runtime warning only; the calculation is unchanged.
- `laura_scripts/legacy/`: the v1 scripts were moved here (with git history). Three had their import paths changed so they keep using v1 code (listed in `legacy/README.md`).
- `s_nmf/gollisch_sparsity_examples.py`: one import path updated (`laura_scripts.Subunits.subunit_sparsity_plots`) after `subunit_sparsity_plots.py` moved to `Subunits/`.

### Not changed on purpose
- `combined_analysis/clustering_pipeline.py` and `combined_plotting.py` (Marvin's) still read the old `noise_data.nc`.

## 4. Data on the NAS

| File(s) | Where | Status |
|---|---|---|
| `noise_data_v2.nc` | Next to **every** `noise_data.nc` on the NAS (all `noise_analysis*` folders of all 25 recordings, 05/11/2025 to 22/10/2026): 56 files, ~367 GB. | Done (last batch 18/11/2026, 19:41–20:48, no errors). The first 11 files (07/08, 14/05, 15/05, 22/07, 14/08) were checked in detail (frame marker, `quality`/`positions`/STA unchanged, size-verified); the rest were made by the same converter. |
| `…_alignment_cache_v2.npz` | `alignment_videos/` of 07/08, 14/05, 15/05, 22/07, 14/08 | Same electrode and stimulus-edge clicks as the original cache, plus raw size, displayed size and stimulus pixel size. Originals untouched. |
| `fff_features_v2.csv`, `fff_data_v2.nc` | `fff_analysis/` of 14/05, 15/05, 22/07, 14/08 (07/08 has no `fff_analysis`) | Other features copied from the old tables; cone features recalculated with fixes 2, 3 and 6. |
| Pixel size 2.5 → 2.0 µm | Rule: every recording from 20/04/2026 uses 2.0 µm. Relabelled: 21/04, 22/04, 23/04, 24/04, 30/04, 07/05, 13/05, 14/05, 15/05 (2026). Files: `noise_params.json` (each noise channel folder), `recording_config.json` and `collapse_2d_config.json` (`noise_analysis` and `QI10`), v2 caches | Done. Every edited file has an `…_orig_2.5um` backup next to it. In `collapse_2d_config.json` only `cut_size_um` changed (× 0.8, e.g. 800 → 640); the cutout in pixels is unchanged. 12/08, 13/08, 15/10 and 22/10 were already at 2.0 µm. Recordings before 20/04/2026 keep 4 µm. |
| `noise_data_v2.nc` rescaled to 2.0 µm (the nine recordings above) | x/y grid and outline radii, `tir` and centre shift × 0.8; areas × 0.64; maps, `positions`, `quality`, `tilt`, `angle` unchanged. Marked with `stimulus_pixel_size_um = 2.0`. | Done for all nine (`noise_analysis` and `QI10`); 14/05 and 15/05 checked and size-verified. |
| Old files | `noise_data.nc`, `fff_data.nc`, old caches, old figures | Untouched. The old `noise_data.nc` of the nine relabelled recordings are still labelled at 2.5 µm. |

## 5. How to use v2

**Existing recordings** (already converted): point scripts at `noise_data_v2.nc` and `…_alignment_cache_v2.npz`. Stimulus sizes and pixel size are read automatically.

**A recording analysed with the wrong pixel size** (e.g. 2.5 µm instead of 2.0): `poetry run python -m combined_analysis.relabel_pixel_size_settings zebrafish_DD_MM_YYYY --old 2.5 --new 2.0`, then `combined_analysis/convert_all_noise_data_v2.sh zebrafish_DD_MM_YYYY 2.0`. Add `--dry-run` to the first command to see what it would change.

**A new recording:**
1. Run `combined_analysis/RF_pipeline.py` (v2). It writes `noise_analysis/noise_data_v2.nc`, never `noise_data.nc`, so it can't overwrite an original even if run on an existing recording.
2. Run Cell 1 of `Alignment_pipeline/RF_alignment.py` with `alignment_cache_path` set to a new `…_alignment_cache_v2.npz`. Set `displayed_stim_width_px` / `displayed_stim_height_px` for this recording's projector window (800 × 800 or 800 × 1000). Do the electrode and stimulus-edge clicks, check that the red box sits on the lit stimulus, and the sizes are saved in the cache.
3. Run the downstream v2 scripts. Nothing else needs typing.

**Reaching the NAS from Linux:** use the CIFS mount at `/mnt/mea_nas` (`/mnt/mea_nas/Laura/…`). It reads at ~110 MB/s, against 18–65 MB/s through KDE's `kio-fuse` (the Dolphin `smb://` path). Mount it in a terminal with `sudo mount -t cifs //mea_nas_25.local/root /mnt/mea_nas -o username=MEA_Computer,uid=$(id -u),gid=$(id -g),vers=3.0`; it lasts until the PC restarts.

## 6. Checks done

- Stored maps vs raw `kernel.npy` (several cells, two recordings): old maps swapped, v2 maps match.
- L-shaped noise (14/08 Phase_01): v2 gives the bottom-right L predicted by your optics diagram; the old alignment gave a top-right L.
- Single-cell marker vs plotted RF peak: 10–20 µm in v2 (was up to ~1 mm).
- The same RF positions from every v2 script that places RFs (alignment, summary figure, cone regularity, crystallinity), to within rounding.
- Converted files: maps match the raw kernels; `quality`, `positions` and STAs identical to the originals; sizes changed by the predicted factors.
- Displayed size: the red check box matches the lit stimulus in all five recordings.
- Electrode clicks: fit errors 3.7–5.9 µm on average (worst < 10 µm) against a 100 µm grid; no mirroring; rotation ≤ 3.6°.

## 7. Known open issues

1. **15/05 projection-image scale ~10% too small.** The cache sets µm per pixel from the first two clicked electrodes only. This doesn't affect where RFs sit on the cones, but 15/05's absolute µm values from the alignment (cone spacing, crop size, scale bars) are ~10% off. Fix: set the scale from all 12 clicks plus the microscope calibration (0.243 µm/px).
2. **Electrode matching can only be checked by eye.** The grid repeats, so a whole-grid shift or rotation fits equally well. Check that non-repeating features (array edge, tissue outline) line up.
3. **`organize/configs.py`:** the step-order check in `Analysis_Pipeline.run()` never runs (typo in `getattr`). (Re-loading a saved pipeline now runs v2, since `configs.py` imports the v2 functions.)
4. **`RF_pipeline.py` (v2) not yet run end to end** (needs Windows). Compare its first output with that recording's `noise_data_v2.nc`.
5. **polarspike extractors** may skip or duplicate `cell_index` values (`linspace`); check if RF and spike cell numbers ever disagree.
6. **Moving-bar directions** use their own compass convention, not the projected frame. Convert before comparing with RF `angle` or elongation. **Fixed 19/11/2026:** the direction table now follows the recording date, like the RFs. Before 8/10/2026 it is unchanged; from 8/10/2026 (fps_py, no up-down flip) the vertical bars and the diagonals are mirrored left↔right (e.g. `bottom_to_top` → left, not right). One rule, `alignment_geometry.moving_bar_direction_table()` / `moving_bar_direction_names()`, is computed from the transform matrices and used by `moving_edge_analysis.py`, `OS_analysis.py`, `OS_analysis_stimulus_filtered_cells.py`, `extract_moving_bar_features.py`, `extract_moving_edge_features.py` and `summary_plot_single_cell.py`. Affects 15/10 Phase_00 and 22/10 Phase_00/01 (none had been processed). Checked in the stimulus files (`stimuli_Oct_2026`): `moving_bar_bigger.h5` (no labels; 8 × 300 frames) shows its blocks in the order left→right, right→left, bottom→top, top→bottom, then the diagonals bottom-left→top-right, bottom-right→top-left, top-left→bottom-right, top-right→bottom-left, the order the block-order scripts assume; and in the labelled moving-bar/edge files every `Directions` label matches the bar's motion in the stored frames (top = row 0). **RF angle in the same frame (19/11/2026):** `alignment_geometry.rf_angle_to_polar_orientation_deg(angle, recording)` converts the stored RF `angle` (stimulus frame, 0° = right, towards down) to the polar-plot / aligned-cone-image frame (0° = up, clockwise; before 8/10/2026 = (180 − angle) mod 180, from 8/10/2026 = angle). The OS_analysis tables, `moving_edge_analysis.tuning_table()` and `summary_plot_single_cell.tuning_table()` now give `rf_orientation_polar_deg` and `rf_vs_bar_orientation_diff_deg` (0° = RF long axis parallel to the preferred motion axis, 90° = perpendicular) next to tilt, OSI, DSI and preferred direction/orientation, and every moving-bar polar plot draws the RF long axis as a solid black line.
7. **Limits:** square RF cutouts only; one pixel size per recording; `center_std` / `surround_std` are placeholders (0); cone-regularity settings are in pixels (right for the current microscope, ~0.24 µm/px).
8. **07/08 cone image:** only an unprocessed MAX projection in Box; you're checking whether a door-view version exists.
9. **30/04 alignment:** v2 clicks done 19/11/2026 (`zebrafish_30_04_2026_phase_00_alignment_cache_v2.npz`; 800 × 1000 window, right edge). Electrodes are only visible on the right, so the 12 clicks span ~140 × 650 µm and the affine fit is only well defined near that strip (12% stretch; a rotation + uniform-scale fit would differ by ~75 µm on average across the stimulus). Left out of the figures: its only noise channel is 4px (8 µm pixels) and its RFs (18 cells with quality ≥ 10) are mostly single-pixel peaks, several at the stimulus edge.
