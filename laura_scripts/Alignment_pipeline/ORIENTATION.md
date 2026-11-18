# RF / cone alignment: every rotation and flip in the v2 pipeline

Search the code for `ORIENTATION [` to find each place below. The tags mean:

| Tag | Meaning |
|---|---|
| `[RF ROTATION]` | Changes where RFs land. There is only **one** such rotation in v2. |
| `[DATA FIX]` | Undoes the old `reduce_all.py` x/y swap so the stored maps are correct. |
| `[LABELS only]` / `[AXIS ORDER only]` | Names or reorders axes. The picture does not move. |
| `[FRAME DEFINITION]` / `[CONE IMAGE, fitted]` | How the projected frame and cone image are set up from your clicks. |
| `[DISPLAY only]` | How a plot is drawn (y pointing down). The data are not rotated. |
| `[NOT RF]` | Raster or time-axis flips, unrelated to RF orientation. |

## Coordinate frames

- **Stimulus frame**: the noise stimulus file as numpy stores it. x = column (+ right), y = row (+ down), origin = stimulus centre. `kernel.npy`, `quality.nc`, `positions` and (v2) the `rms` / `cm_most_important` maps are all in this frame.
- **Projected frame**: the stimulus as it lands on the MEA, viewed facing the MEA from the door. x = right, y = down, origin = displayed-stimulus centre. The projection image (Image 3) and the cone image are placed in this frame.

## The steps, in order

1. **RF maps are saved in the stimulus frame.** `[DATA FIX]` in `combined_analysis/reduce_all.py` (v2; new runs) and `combined_analysis/convert_noise_data_v2.py` (existing files → `noise_data_v2.nc`). The old `reduce_all.py` stored the maps with x and y swapped.
2. **The projected frame is built from your clicks.** `[FRAME DEFINITION]` in `alignment_geometry.py`: +y runs down along the clicked stimulus edge, +x is 90° clockwise of it, and the centre comes from the edge plus this recording's displayed width and height (saved in `…_alignment_cache_v2.npz`).
3. **The cone image goes into the projected frame using the affine fitted to your 12 electrode clicks.** `[CONE IMAGE, fitted]` No fixed flip is added. The cone image must already be in the door view (Fiji: 90° clockwise, then up-down flip).
4. **The one RF rotation, chosen by recording date.** `[RF ROTATION]` It depends on the stimulus software:
   - **Recordings before 8 October 2026:** stimulus (x, y) → projected (−y, −x). That's the OpenGL up-down flip followed by the projector's 90° anticlockwise rotation, viewed from the door. Confirmed with the L-shaped noise (14/08/2026).
   - **Recordings from 8 October 2026 (`fps_py`):** stimulus (x, y) → projected (y, −x). That's the projector's 90° anticlockwise rotation only; the new software shows the stimulus in the numpy layout, with no OpenGL flip.

   `stimulus_to_projection_for_recording()` in `alignment_geometry.py` picks the right one from the recording name (`DD_MM_YYYY`). It's used by `RF_alignment.py`, the cone-connectivity functions, `extract_cone_regularity.py` and `summary_plot_single_cell.py`. The RF data are never flipped; they are placed through this transform.

That's all. Every other `ORIENTATION` tag is labels, axis order, display, or not about RFs.

## Plots of RF maps on their own (no cones)

The v2 analysis scripts draw maps with **y pointing down** (`yincrease=False` / `origin="upper"`), so a map looks exactly like the stimulus file. Note that xarray's `.plot.imshow()` draws y pointing **up** by default, which is how the original L test ended up with a left-right flip instead of a 180° rotation.

## Angles

`angle` in `noise_data_v2.nc` is in the stimulus frame: 0° = +x (right), measured towards +y (down), so clockwise on screen. It equals 90° − the old value. Differences in angle between cells are unchanged.
