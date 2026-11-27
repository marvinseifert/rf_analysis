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
| `RF_pipeline_colour_sequence.py` | RF pipeline for interleaved colour noise (LED switching every N frames): every colour becomes its own channel. Run `rf_torch/main_sta_colour_sequence.py` first for the per-colour STAs (25 Nov 2026). |
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

#### Added 23 Nov 2026: can 6px noise see cone / bipolar input at all?

Results: the 6px checker (11.7 µm at 1.9476 µm/px) has ~2 % stimulus power at 10-15 µm periods and a zero at 11.7 µm. Simulations: single-cone-type RFs are detectable, dense all-cone pooling mostly not; 4px resolves UV/blue cones, 2px everything (lower SNR). Real cells: no reproducible structure < 10 µm beyond RF shape (shape-twin control). At 15-50 µm, 12/45 cells beat their shape twin (p = 1.5e-6), but the spots are point-like (single cone or small bipolar, not separable at 6px); only cell 282's central hot spot clearly reproduces across independent halves (not seen in 8px, where the cell is weak). Figures: Box `Figures/RF_cone_alignment_v2/` (colour checks: `Figures/colour_noise_checks/`).

| Script | What it does |
|---|---|
| `stimulus_power_spectrum.py` | Spatial power spectrum of the actual 6px frames vs the sinc² prediction; power in the 10-15 µm cone band; grid-shift histogram. |
| `cone_bump_power_and_split_half.py` | (1) Half-split of fine structure (map - 6 µm blur) in all quality >= 10 cells, independent sequence halves, shifted-spike and elliptical "smooth twin" controls. (2) Cone-bump power simulation (8 cells x 4 RF types x 20 placements) with a position test. Old schematic mosaic: see the note in the docstring. Its JSON feeds the scripts below. ~50 min. |
| `shape_twin_control.py` | Stricter control: twin = the cell's own map smoothed 4 µm (same RF shape, no fine detail). Only 2/13 cells pass, by tiny margins. |
| `cone_bump_checker_size.py` | Same cell simulated with synthetic shuffled 2/4/6 px noise, Laura's row mosaic (UV-B / R-G / G-R / B-UV rows), RFs on all cones or one cone type; maps, position-test power and SNR; mosaic schematic in the figure. |
| `colour_noise_checks.py` | 15/10 and 22/10 interleaved colour noise (32px): single colour vs white quality, chromatic offset of RF centres (shift and magnification), adaptation after a colour switch. |
| `bipolar_scale_split_half.py` | Half-split at bipolar scale (15-50 µm band, Gaussian 3 - 8 µm) against shape twins (own map smoothed 10 µm), all cells. |
| `bipolar_spots_on_cones.py` | The reproducible 15-50 µm spots of chosen cells on the aligned cone image, their width vs single-cone / bipolar references through the same blur, and the covariance map overlaid. |
| `band_controls_on_cones.py` | Same band display for a cell and its controls on the cone image: covariance map, STA map, halves A/B, shape twin, shifted-spike halves (+ chance r from 20 shifts). |
| `compare_6px_8px_hotspot.py` | Does a cell's 6px hot spot appear in the independent 8px noise (maps, bands, halves, chance)? |

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

### `chromatic_RFs/`: RFs of interleaved colour noise

#### `chromatic_RFs/spatial_offsets/` (25 Nov 2026)

| Script | What it does |
|---|---|
| `plot_chromatic_rfs.py` | Per-cell RF maps, outlines + overlay, RGB composites and STA kernels for every colour (15/10 P0, 22/10 P1), grouped by quality. Optional rows: STA frames (`--sta-frames`) and every scf / csteps PSTH (`--fff`). Rewrite of the lost script behind `Figures/RF_colour_offsets.pptx`. |

#### `chromatic_RFs/frequency_offsets/`: speed / timing per colour (27 Nov 2026)

Results in `frequency_offsets/results/`. Figure: Box `Figures/noise_rate_comparison/colour_noise_comparison.png`.

| Script | What it does |
|---|---|
| `colour_noise_comparison.py` | Own 8-px-block STAs per colour for 22/10 P1 (10 Hz; 413/500/610 nm) and 15/10 P0 (15 Hz; white/610/500/420 nm, plus white alone): RF above chance per colour (spikes shifted by whole colour cycles), signal ÷ noise, half width, integration time, peak time. ~6.5 min, GPU. |
| `colour_timing_detail.py` | Finer timing in the 15/10 colour noise (interpolated peak, half-height crossings, centre of mass), same cells across colours; polarity per colour (colour opponency). |
| `plot_colour_noise_comparison.py` | The colour figure. |

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

**Added 23 Nov 2026 (per-cell tests, measured rows, mosaic regularity).** The across-recording test in `summary_figure.py` (p = 0.007) compares MEAN row directions, which is not meaningful where rows fan out from a nearby ONH (4 of 7 recordings); use the per-cell within-recording shuffle instead (p = 0.009, all cells). Results: RFs are under-represented ALONG their own row (about 10 % vs 13-17 % chance), not "along or across"; the bias is present only over regular (crystalline) mosaic (score +0.29, p < 0.001) and absent over disrupted mosaic and the larval patch (< 200 µm from the ONH); in 15/05 with traced rows, regular-patch RFs lie squarely ACROSS the real rows (35 % at 75-90°, 3 % along), and disrupted-patch RFs follow the rows predicted from the surrounding intact mosaic better than the imaged ones (suggesting mostly post-recording distortion). The 60-75° peak against ONH-radial rows is mostly the rows' twist away from radial (about 10-15° in 15/05), not the lattice diagonal. Moving bars/edges add little (few cells orientation-tuned; mixed relation to rows). No cell-type or clear colour specificity. Figures in the same Box folder.

| Script | What it does |
|---|---|
| `summary_figure_per_cell.py` | Corrected summary figure in the same style as `summary_figure.py` (unchanged): per-cell tests (within-recording shuffle, fake ONH positions), per-recording % across own row with row spread. |
| `supporting_analyses/rf_row_strict_tests.py` | The two per-cell tests on their own, all and elongated cells. |
| `supporting_analyses/rf_row_angle_distribution.py` | Along / oblique / across fractions and the "along OR across" (cos 4θ) vs "across > along" (cos 2θ) tests; stimulus-grid control. |
| `supporting_analyses/rf_vs_measured_rows_15_05.py` | 15/05: RF axes vs the rows traced on the cone image (`Imaging/20260515_dragonfly_microscope/15_05_2026_cones_rows.csv`); traced vs ONH-radial rows. |
| `supporting_analyses/cone_lattice_angles_15_05.py` | 15/05: labelled-cone lattice directions relative to the traced rows (same-row neighbours ~12 µm, adjacent-row diagonals at ~66-69° folded). |
| `supporting_analyses/rf_row_bias_by_mosaic_regularity.py` | **Key figure.** Crystalline vs disrupted (cone_regularity_index median split) vs larval patch (4 recordings), RF-size-matched check and adjusted regression, and 15/05 traced rows: imaged vs predicted pre-distortion rows. |
| `supporting_analyses/rf_row_bias_by_cell_type_and_mosaic.py` | Cell-type groupings within crystalline vs disrupted mosaic: every type shows the bias over crystalline mosaic, none over disrupted; no type differences. |
| `supporting_analyses/rf_alignment_vs_properties_crystalline.py` | Crystalline cells lying across their row vs oblique ones: 19 response / RF properties, within-recording ranks, FDR; nothing survives correction. |
| `supporting_analyses/expansion_vs_contraction.py` | Preferred motion DIRECTION vs outward-from-ONH (bars: DSI >= 0.15; edges: DSI shuffle p < 0.05), by region. Crystalline bar cells prefer INWARD motion (8 % outward, p = 0.001 vs a shuffle that is itself inward-biased); edges: too few cells. Not looming-like. |
| `supporting_analyses/expansion_vs_contraction_central_onh.py` | Same test in the recordings with the ONH near the array centre (14/05, 14/08), where a shared preferred direction cannot fake inward/outward; maps of preferred-direction arrows. 14/05: 12 of 13 DS bar cells prefer INWARD (contracting) motion (mean cos -0.70 vs shuffle -0.11, p < 0.001); 14/08: -0.26 (p = 0.16). Small n. |
| `supporting_analyses/direction_preference_summary.py` | Summary figure (Box Figures/RF_cone_row_angle_prediction/direction_preference_summary.png): A moving-bar DS cells by mosaic region vs within-recording shuffle baselines (crystalline -0.62, p = 0.001; disrupted -0.27, p = 0.19; larval 3 cells; crystalline - disrupted -0.35, p = 0.005, region labels shuffled within recording); B/C central-ONH maps 14/05 and 14/08 with preferred-direction arrows; caveats and interpretation box. |
| `supporting_analyses/schematic_rf_row_alignment_ideas.py` | Schematic summary figure (no data): the alignment (A-C), computational ideas (D-I) and tentative conclusions. |
| `supporting_analyses/computational_hypotheses.py` | Five computational hypotheses with existing data + summary figure: lattice anti-aliasing (right direction, wrong scale), ring vs spoke population sensitivity (+5.6 %, crystalline only), moving bars prefer motion along the rows in crystalline cells (p = 0.002), RF ~5 % wider across rows, colour-sampling advantage of a regular mosaic only 1-3 % at RF scale. |
| `supporting_analyses/rf_row_bias_by_cell_type.py` | Bias by RF polarity, RF size, contrast-step polarity, transience, chirp frequency; 15° bins with shuffle chance. |
| `supporting_analyses/rf_row_bias_by_colour.py` | Bias by colour preference and strict colour opponency (single chromatic flashes). |
| `supporting_analyses/bar_orientation_vs_cone_rows.py` | Moving-bar preferred motion axis vs own row and RF axis (peak response; per-cell within-recording shuffle). |
| `supporting_analyses/plot_bar_rows_schematic.py` | Explainer figure: row mosaic, RF and bar alternatives, whole-retina schematic, real data in the row frame. |

### `OSI/`: moving-bar direction and orientation selectivity

The bar directions are chosen by recording date. In some recordings the moving bar may not have covered the whole array.

| Script | What it does |
|---|---|
| `OS_analysis.py` | OSI/DSI for all cells of one or more recordings: tables, a tilt-vs-OSI scatter plot, and polar plots showing the RF long axis. |
| `OS_analysis_stimulus_filtered_cells.py` | The same, with an optional list of cells to keep for each recording. |
| `moving_edge_analysis.py` | Per-cell summary figures for 1–4 weighted moving bar/edge stimuli (H5 files). |
| `edge_tuning_table.py` | (23/11/2026) Moving-edge OSI/DSI for every cell and condition (22/07 ON/OFF 13 and 60 fps, 14/08 ON/OFF 13 fps; 07/08 skipped: triggers do not fit whole repeats), with a direction-label shuffle test; reuses `moving_edge_analysis.py` functions. Output: Box `Figures/OSI/edge_tuning_table.csv`. |
| `edge_orientation_vs_cone_rows.py` | (23/11/2026) Significantly orientation-tuned edge cells vs their own cone row and RF axis (within-recording shuffle); only 3-10 % of cells tuned (14/08 at chance). |

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

### `stimulus_frequency_exploration/`: choosing the noise frame rate (27 Nov 2026)

Which noise rate (frame length) is best for RFs: signal ÷ noise vs temporal blur. Mainly 15/10/2026 (32px white noise at 5/10/15/20 Hz, 20 min each) plus the 20 Hz clustering recordings. All outputs in `stimulus_frequency_exploration/results/`; figures in Box `Figures/noise_rate_comparison/`. Agreed conclusions: memory note / conclusion box in `frame_rate_concept.png`.

| Folder / script | What it does |
|---|---|
| `rate_comparison_15_10/quality_null_by_rate.py` | Per cell and rate: RF quality from own STAs (8-px blocks) and chance quality from time-shifted spikes. Basis of the chance correction. ~3.5 min, GPU. |
| `rate_comparison_15_10/retina_health_bracket_stimuli.py` | Response reliability (QI) and firing of scf/csteps/chirp before (6–8) and after (13–15) the noise → per-cell health change. |
| `rate_comparison_15_10/plot_noise_rate_summary.py` | Chance- and health-corrected RF quality per rate linked to chirp frequencies (`noise_rate_summary.png`); also writes `noise_rate_corrected_quality.csv` (best rate, chirp frequencies, ON/OFF). |
| `rate_comparison_15_10/rf_timecourse_by_noise_rate.py` | RF-centre time courses per rate: peak time, half width, frames per width; same 27 cells (`rf_timecourse_by_noise_rate_same_cells.png`). |
| `rate_comparison_15_10/signal_noise_by_rate.py` | Signal per spike vs noise per rate (same 27 cells): noise follows spike count, not frames. |
| `chirp/chirp_max_following_frequency.py` | Replacement chirp measure: highest / strongest followed frequency from phase locking vs time-shifted chance (the fff `chirp_threshold_frequency` is broken). |
| `response_width_integration/timecourse_measures.py` | Shared functions: half-height width + peak time, integration time (main lobe; both lobes). |
| `response_width_integration/width_and_integration.py` | Width and integration time for the 20 Hz recordings (clustering selection + 15/05) and 15/10 per rate. |
| `repeat_number/repeats_vs_rf.py` | How much the 2nd and 3rd identical 20-min repeats of 32px_15Hz_20mins_shuffle_x4 add (14/08, 07/08): RF yield, signal ÷ noise, time course, centre size, surround, map agreement for 20 / 40 / 60 min. ~2.5 min, GPU. |
| `summary_figures/plot_frame_rate_concept.py` | The summary figure `frame_rate_concept.png` (schematic, measured numbers, per-recording widths, conclusion). |
| `summary_figures/plot_sta_blur_schematic.py` | Schematic of how frames blur the STA time course (`sta_blur_schematic.png`). |

Run order if regenerating: `quality_null_by_rate` → `retina_health_bracket_stimuli` → `chirp_max_following_frequency` → `plot_noise_rate_summary` → `rf_timecourse_by_noise_rate` → `signal_noise_by_rate` → `width_and_integration` → `plot_frame_rate_concept`.

Old version kept in `legacy/stimulus_frequency_exploration/`.

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
