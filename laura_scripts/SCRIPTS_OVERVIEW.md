# Scripts overview

**Covers:** `combined_analysis/`, `laura_scripts/` and the top-level `legacy/`.
**Written:** 19/11/2026. Everything runs on Linux.

**"v2"** means the fixed RF pipeline: RF maps the right way round, correct outline sizes, and files named `noise_data_v2.nc`. Full details of the v2 changes are in [`Alignment_pipeline/V2_CHANGES.md`](Alignment_pipeline/V2_CHANGES.md).

---

## 1. `combined_analysis/`

The RF analysis pipeline (shared with Marvin) and its conversion tools.

| Script | What it does |
|---|---|
| `RF_pipeline.py` | Runs the full RF analysis for a new recording (v2): STAs → quality → RF maps → outlines → stats. Writes `noise_analysis/noise_data_v2.nc`. |
| `reduce_all.py` | The RF-map functions `RF_pipeline.py` uses (covariance/RMS maps, outlines), v2 version. |
| `convert_noise_data_v2.py` | Converts an old `noise_data.nc` into `noise_data_v2.nc` without recomputing anything. |
| `convert_all_noise_data_v2.sh` | Runs the converter on every `noise_analysis*` folder of one recording. |
| `relabel_pixel_size_settings.py` | Changes a recording's stimulus pixel size in its settings (e.g. 2.5 → 2.0 µm). Keeps `_orig_2.5um` backups. |
| `RF_L_shaped.py` | One-off: RF analysis of the L-shaped noise (14/08/2026, Phase_01). |
| `clustering_pipeline.py` | Marvin's script: clusters cells by their RF and response features. Reads v1 data. Do not edit. |
| `clustering_pipeline_v2.py` | Copy of the above for v2 data, with Linux paths. *Not yet run.* |
| `combined_plotting.py` | Marvin's script: example plots of all analyses for one recording. Reads v1 data. Do not edit. |
| `combined_plotting_v2.py` | Copy of the above for v2 data, with Linux paths. *Not yet run.* |

---

## 2. `laura_scripts/`

Laura's analysis scripts. The v2 versions keep the normal script names; the v1 versions are in `laura_scripts/legacy/`.

### `Alignment_pipeline/`: aligning RFs to the cone mosaic

The alignment uses the projection photo and clicks on the electrodes.

| Script | What it does |
|---|---|
| `RF_alignment.py` | Main interactive alignment: electrode and stimulus-edge clicks (CELL 1), population overlays (CELL 3), top-N single cells (CELL 4). |
| `alignment_recordings.py` | Per-recording settings for `RF_alignment.py` (paths, window size, channel), plus path helpers for the plotting scripts. |
| `alignment_geometry.py` | Shared rules: the stimulus → projection transform by date (it changed on 8/10/2026), the moving-bar direction tables, and the conversion of RF angles to the polar-plot frame. |
| `plot_population_overlays_all_recordings.py` | One figure showing the population covariance RFs on the cone mosaic for every recording. |
| `plot_top_quality_cells.py` | The top-N quality cells of one recording/channel on the cone mosaic. |
| `cone_rf_connectivity_functions.py` | Functions for single-cell RF ↔ cone-mosaic connectivity. |
| `run_single_cell_cone_rf_connectivity_plot_cell.py` | Runs those functions: loads the data once, then `plot_cell(cell_index)`. |
| `ORIENTATION.md` | Explains every rotation and flip between the stimulus, the projection and the cone image. |
| `V2_CHANGES.md` / `.odt` | What was wrong in v1, what was fixed, the status of the data, and open issues. |

#### `Alignment_pipeline/electrode_tools/`

Early electrode-alignment prototypes from March/April 2026. Most of their data is gone.

| Script | What it does |
|---|---|
| `MEA_electrode_alignment.py` | Finds the electrodes from an MEA recording of a moving bar passing over them. |
| `MEA_electrode_image_analysis_MS.py` | Makes a mean image from the projection video and tests electrode autocorrelation. |
| `electrode_overlay_analysis.py` | Draws a synthetic 100 µm electrode grid over the tissue image. |

### `6px_subunit_exploration/`: looking for cone / subunit input in single RFs (20 Nov 2026)

Started on 14/08/2026, 6px noise (12 µm pixels shuffled in 2 µm steps). Figures go to Box `Figures/RF_cone_alignment_v2/`. Run with `MPLBACKEND=Agg`.

| Script | What it does |
|---|---|
| `plot_cells_with_stas.py` | Top-N quality cells (covariance and RMS) and the most "subunit-like" cells on the cone mosaic, each with its STAs: single pixel and the top 3 % weighted STA from `RF_analysis/weighted_STA.py`. All panels are centred on the pipeline RF centre. |
| `subunit_tools.py` | Shared "subunit" score: counts separate peaks in the RF centre that rise clearly above the map's noise. |
| `compare_6px_8px_subunit_cells.py` | Checks whether the small spots in each RF repeat between the 6px and 8px noise, against a rotated-map chance level. |
| `check_duplicate_units.py` | Checks whether two cells are one neuron counted twice: shared spikes, firing together at 0 lag, peak electrode, template and STA similarity. |
| `plot_neighbour_rf_overlap.py` | A cell plus its 1, 2 and 3 nearest neighbours: the actual RF maps overlaid in colour, plus a figure with the RFs in greyscale and overlapping pixels coloured (also shown on the cones). |

### `SNMF/`: testing sNMF / subunits against fair noise controls (20 Nov 2026)

14/08/2026 6px shuffled noise, using the per-cell `snippets.h5` from rf_torch `main_sta_snippets_all.py`. Result: RF shapes reproduce in independent halves (reliable to ~19 µm detail in cell 150), but no structure smaller than the RF beats chance. Two traps found: the shuffle makes even pure-noise sNMF modules smooth (Moran's I cut-off useless), and the 30-min sequence is played 3× identically (any spike split shares frames: split by half of the noise *sequence* instead). Figures go to Box `Figures/SNMF/`. Run with `MPLBACKEND=Agg`, one at a time.

| Script | What it does |
|---|---|
| `random_control_snippets.py` | Control snippets for chosen cells: random times (`--mode random`) or the cell's real spike train shifted in time (`--mode shifted`, keeps bursts; the fair control). `--seed` for independent sets. Saves under new names, never touches the real snippets. |
| `snmf_real_vs_random.py` | sNMF (as `s_nmf/gollisch_pipeline.py`) on real spikes vs random times; Moran's I of every module. Crops zero-filled edge columns. |
| `snmf_one_time_course.py` | Same, but one time course for all pixels (removes the RF imprint) and two ways of judging modules: (a) shape vs random modules, (b) use before spikes. |
| `snmf_consistency.py` | Split-half reproducibility of sNMF modules for several r, real vs random (note: random halves share frames because of the 3 repeats). |
| `stc_minus_random.py` | Spike-triggered covariance minus control covariance (cancels the shuffle correlation); `--control shifted`, `--crop-um`. Chance from two control sets. |
| `split_half_blobs.py` | RF map from two halves of the spikes (recording halves, and independent noise-sequence halves) and whether the blobs come back; shifted spike train as chance. |
| `rf_resolution_test.py` | Fourier ring correlation of two independent half-maps: the finest detail size that is reliable. |
| `plot_bar_stimulus_design.py` | Design sketch (no data): random-bar stimuli (single bar, few bars, 1D flickering bars across / along the cone rows) on a schematic mosaic. |

### `Cone_imaging/`: cone mosaic spacing

Uses the dragonfly (sws1 mCherry) images.

| Script | What it does |
|---|---|
| `mCherry_image_autocorrelation_analysis.py` | Cone spacing (µm) from the image autocorrelation. |
| `mCherry_image_analysis_mulitple_ROIs.py` | The same for several chosen ROIs, plus a nearest-neighbour check. |

### `Elongated receptive fields/`: RF elongation (tilt) and orientation

| Script | What it does |
|---|---|
| `Angle_histogram.py` | Histograms of RF long-axis angles (and tilt) across recordings. |
| `chirp_plots_by_tilt.py` | Chirp responses grouped by RF elongation. |
| `ff_plots_by_tilt.py` | Full-field flash (SCF) responses grouped by RF elongation. |
| `elongation_cells_poster.py` | Poster panel of the most elongated RFs, pooled across recordings. |

#### `Elongated receptive fields/RF_cone_row_angle_prediction/`: RF orientation vs predicted cone rows (19 Nov 2026)

Cone rows are assumed to radiate from the optic nerve head (ONH); each RF's long axis is compared with the line ONH -> RF centre. Final figure: Box `Figures/RF_cone_row_angle_prediction/FINAL/`.

| Script | What it does |
|---|---|
| `click_optic_nerve_head.py` | Click the ONH on a recording's cone image (`--recording DD_MM_YYYY`); saves to `optic_nerve_head_positions.json`. |
| `rf_cone_row_angle.py` | Shared loading/geometry (RF centres and axes, ONH and row direction in the aligned cone-image frame, edge flag); run on its own it writes the per-cell table `rf_vs_cone_row_angle_all_recordings_cells.csv`. |
| `summary_figure.py` | The final summary figure (landscape): cone images + covariance RFs, RF axes vs radiating rows, histograms, across-recording test, pooled elongated/round histograms. |
| `supporting_analyses/rf_orientation_field.py` | Smoothed RF orientation field vs circles around the ONH, with a shuffle test (source of 14/05 p = 0.027). |
| `supporting_analyses/rf_position_arc_test.py` | Do RF centres line up in arcs around the ONH (neighbour-pair test, calibrated on synthetic data)? No consistent effect. |
| `supporting_analyses/check_rf_angle_reliability.py` | Stored RF angle vs an independent map measurement, and same cell across noise stimuli vs different cells. |

### `OSI/`: moving-bar direction and orientation selectivity

The bar directions are chosen by recording date. In some recordings the moving bar may not have covered the whole array.

| Script | What it does |
|---|---|
| `OS_analysis.py` | OSI/DSI for all cells of one or more recordings: tables, a tilt-vs-OSI scatter plot, and polar plots showing the RF long axis. |
| `OS_analysis_stimulus_filtered_cells.py` | The same, with an optional list of cells to keep for each recording. |
| `moving_edge_analysis.py` | Per-cell summary figures for 1–4 weighted moving bar/edge stimuli (H5 files). |

### `RF_analysis/`: general RF plotting and STA tools

| Script | What it does |
|---|---|
| `receptive_field_functions.py` | RF plotting functions (multi-panel RF maps etc.) used by other scripts. |
| `receptive_field_functions_old.py` | An older version of the above. Kept but not used. |
| `main_rf_analysis_LS.py` | Loads a recording's RF data and plots the RFs with those functions. |
| `merged_stimuli_RFs.py` | Combines two 12 px white-noise runs (15/05) into one RF per cell. |
| `Adding_kernals.py` | One-off: adds the kernels of two noise channels (15/05 → combined_kernels). |
| `weighted_STA.py` | Recomputes STAs and covariance maps for chosen cells (weighted STA). |

### `Scratch/`: tests and quick ideas

None of these are part of a pipeline.

| Script | What it does |
|---|---|
| `test.py` | RMS maps from individual kernels compared with combined kernels (15/05). |
| `test_rms_cm_single_plot.py` | The RMS and covariance maps of single cells side by side. |
| `test_datashader_plots.py` | Grids of top-quality RFs drawn with datashader. Uses the 19/02 recording, whose data is now missing. |
| `colour_coding_idea.py` | An idea, notes only: label cells short/long/white from their SCF spikes. |

### `Seaborn/fff_feature_pipeline/`: one table of per-cell features

The table combines RF, full-field, moving-stimulus and cone features.

| Script | What it does |
|---|---|
| `README.txt` | How to run the pipeline. |
| `build_fff_dataset.py` | Runs all the extractors for one recording and writes `fff_features_v2.csv` and `fff_data_v2.nc`. |
| `fff_common.py` | Shared helpers for the extractors (spike loading, PSTHs). |
| `extract_scf_features.py` | Single chromatic flash (SCF) features. |
| `extract_csteps_features.py` | Contrast-step features. |
| `extract_chirp_features.py` | Chirp features (frequency and amplitude responses). |
| `extract_moving_bar_features.py` | Moving-bar OSI, DSI and preferred direction (direction table chosen by date). |
| `extract_moving_edge_features.py` | The same for weighted moving bars/edges (direction labels from the H5 files). |
| `extract_cone_regularity.py` | Cone-mosaic regularity (spacing, crystallinity) around each RF centre. |
| `crystallinity_index_visualisation.py` | Shows cone mosaics for RFs from low to high crystallinity. |
| `seaborn_plots.py` | Seaborn plots comparing RF and full-field traits across recordings. |
| `__init__.py` | Makes the folder importable. |

### `Subunits/`: RF subunits with Gollisch semi-NMF

| Script | What it does |
|---|---|
| `subunit_sparsity_plots.py` | The subunits of one cell for several sparsity values (used by the s_nmf examples). |
| `subunit_analysis_drafts.py` | Drafts of the subunit analysis. Its orientation differs from the RF scripts. |
| `subunit_functions_old.py` | Older subunit plotting functions. Kept but not used. |

### `Summary_figures/`

| Script | What it does |
|---|---|
| `summary_plot_single_cell.py` | Per-cell summary: RFs on the cones, STA, SCF, contrast steps, chirp and moving bar. |
| `poster_plot_12panel.py` | 12-panel poster figure. |

### `legacy/`: Laura's v1 scripts (backup only)

These are the scripts from before 18/11/2026, in the same folder layout. They read the old `noise_data.nc`.

| Folder | Contents |
|---|---|
| `README.md` | What is in this folder and which imports were changed. |
| `Alignment_pipeline/` | v1 `RF_alignment.py`, `cone_rf_connectivity_functions.py`, `run_single_cell_cone_rf_connectivity_plot_cell.py` |
| `Elongated receptive fields/` | v1 `Angle_histogram.py`, `chirp_plots_by_tilt.py`, `elongation_cells_poster.py`, `ff_plots_by_tilt.py` |
| `OSI/` | v1 `OS_analysis.py`, `OS_analysis_stimulus_filtered_cells.py`, `moving_edge_analysis.py` |
| `RF_analysis/` | v1 `main_rf_analysis_LS.py`, `merged_stimuli_RFs.py`, `receptive_field_functions.py`, `weighted_STA.py` |
| `Scratch/` | v1 `test.py`, `test_datashader_plots.py`, `test_rms_cm_single_plot.py` |
| `Seaborn/fff_feature_pipeline/` | v1 `build_fff_dataset.py`, `crystallinity_index_visualisation.py`, `extract_cone_regularity.py`, `seaborn_plots.py` |
| `Summary_figures/` | v1 `poster_plot_12panel.py`, `summary_plot_single_cell.py` |

`SCRIPTS_OVERVIEW.md` is this file.

---

## 3. `legacy/` (top level): v1 shared RF pipeline files (backup only)

| File | What it does |
|---|---|
| `README.md` | What is in this folder and why. |
| `combined_analysis/RF_pipeline.py` | The v1 pipeline driver. Writes the old `noise_data.nc`. |
| `combined_analysis/reduce_all.py` | The v1 RF-map functions. Their maps have x and y swapped, and the outlines are 1.41× too large. |
| `loading/load_sta.py` | v1 STA loading. |
| `stats/summary.py` | v1 RF stats. Do not use: it would multiply areas by pixel size². |
