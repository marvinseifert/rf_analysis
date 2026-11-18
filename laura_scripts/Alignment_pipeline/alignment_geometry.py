"""
Shared geometry rules for the v2 RF / cone alignment. Every script that places RFs
on the projection or cone images should import these instead of keeping its own copy.

Coordinate frames
-----------------
STIMULUS frame   numpy 'upper' layout of the noise stimulus file: x = column (+ right),
                 y = row (+ down), origin = stimulus centre. kernel.npy, quality.nc,
                 dataset["positions"] and (v2) the rms / cm_most_important maps are
                 all in this frame.
PROJECTED frame  the stimulus as it lands on the MEA, viewed facing the MEA from the
                 door: x = right, y = down, origin = displayed stimulus centre.
                 Image 3 (visual_stim_on_MEA.png) and the cone image are placed here.

Stimulus -> projected depends on the stimulus software, i.e. on the recording date:
  * recordings before 8 October 2026: OpenGL flips the image up-down, then the
    projector rotates it 90° anticlockwise: (x, y) -> (-y, -x). Confirmed with the
    L-shaped noise (zebrafish_14_08_2026 Phase_01).
  * recordings from 8 October 2026 (new fps_py software): only the 90° anticlockwise
    rotation, no up-down flip: (x, y) -> (y, -x).
stimulus_to_projection_for_recording() picks the right one from the recording name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

# ORIENTATION [RF ROTATION] The rotation/flip applied to RFs in v2 (one per recording,
# chosen by date, see stimulus_to_projection_for_recording):
#
# Recordings BEFORE 8 October 2026:
#   stimulus (x, y) -> projected (-y, -x)
#   = up-down flip (OpenGL) then 90° anticlockwise (projector), viewed from the door.
#   Confirmed with the L-shaped noise (14_08_2026 Phase_01).
STIMULUS_TO_PROJECTION_BEFORE_2026_10_08 = np.array(
    [
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
# Recordings FROM 8 October 2026 (fps_py; the GUI shows the numpy layout, no OpenGL flip):
#   stimulus (x, y) -> projected (y, -x)
#   = 90° anticlockwise (projector) only, viewed from the door.
STIMULUS_TO_PROJECTION_FROM_2026_10_08 = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
PROJECTION_CHANGE_DATE = "2026-10-08"  # first day of the new transform
# Kept for older code: the pre-October transform.
STIMULUS_TO_PROJECTION = STIMULUS_TO_PROJECTION_BEFORE_2026_10_08


def recording_date_from_name(name) -> "datetime.date":
    """Date of a recording from any name/path containing DD_MM_YYYY, e.g.
    'zebrafish_15_10_2026_phase_00_alignment_cache_v2.npz' -> 2026-10-15."""
    import datetime
    import re

    m = re.search(r"(\d{2})_(\d{2})_(\d{4})", str(name))
    if m is None:
        raise ValueError(
            f"Cannot find a recording date (DD_MM_YYYY) in {name!r}; it is needed to choose "
            "the stimulus -> projection transform (it changed on 8 October 2026)."
        )
    day, month, year = (int(v) for v in m.groups())
    return datetime.date(year, month, day)


def stimulus_to_projection_for_recording(name) -> tuple[np.ndarray, str]:
    """
    (3x3 transform, description) for the recording named in `name` (a recording name,
    cache file name or path containing DD_MM_YYYY).
    ORIENTATION [RF ROTATION]: before 8 Oct 2026 -> (x, y) -> (-y, -x);
    from 8 Oct 2026 -> (x, y) -> (y, -x).
    """
    import datetime

    date = recording_date_from_name(name)
    if date >= datetime.date.fromisoformat(PROJECTION_CHANGE_DATE):
        return (
            STIMULUS_TO_PROJECTION_FROM_2026_10_08.copy(),
            f"{date}: from 8 Oct 2026 (fps_py): 90° anticlockwise only, (x, y) -> (y, -x)",
        )
    return (
        STIMULUS_TO_PROJECTION_BEFORE_2026_10_08.copy(),
        f"{date}: before 8 Oct 2026: up-down flip + 90° anticlockwise, (x, y) -> (-y, -x)",
    )


# Value of noise_data.nc attrs["rf_map_frame"] written by combined_analysis/reduce_all.py (v2) and
# combined_analysis/convert_noise_data_v2.py.
RF_MAP_FRAME_V2 = "stimulus_v2"


def read_noise_params(noise_analysis_folder) -> list[dict]:
    """
    The noise_params.json of every RF channel of a recording.

    The RF pipeline saves recording_config.json in the noise_analysis folder; it
    names each channel's stimulus folder (…/<stimulus>_idx_<n>). noise_params.json
    is read from that folder next to noise_analysis (Windows paths are not needed).
    """
    import json

    folder = Path(noise_analysis_folder)
    config = json.loads((folder / "recording_config.json").read_text())
    params = []
    for name, channel in config["channel_configs"].items():
        stim_folder = folder.parent / Path(str(channel["root_path"]).replace("\\", "/")).name
        params_file = stim_folder / "noise_params.json"
        if not params_file.exists():
            raise FileNotFoundError(f"noise_params.json not found for channel {name}: {params_file}")
        params.append({"channel": name, **json.loads(params_file.read_text())})
    return params


def dataset_um_per_px(noise_analysis_folder, dataset: xr.Dataset | None = None) -> float:
    """
    Stimulus pixel size (µm) the RF file was built with: noise_pixel_size from the
    channels' noise_params.json (the smallest, if channels differ, as the RF pipeline
    uses). If the dataset is given, it is checked against the spacing of its x grid.
    """
    sizes = {float(p["noise_pixel_size"]) for p in read_noise_params(noise_analysis_folder)}
    size = min(sizes)
    if dataset is not None:
        step = float(np.median(np.abs(np.diff(np.asarray(dataset["x"].values, dtype=float)))))
        if not np.isclose(step, size, rtol=1e-3):
            raise ValueError(
                f"noise_params.json says {size} µm/px but the RF file's x grid is {step} µm/px."
            )
    return size


def raw_stimulus_size_px(noise_analysis_folder) -> tuple[int, int]:
    """(x_len, y_len) of the noise stimulus array in stimulus pixels, from noise_params.json
    (largest over channels)."""
    params = read_noise_params(noise_analysis_folder)
    return max(int(p["x_len"]) for p in params), max(int(p["y_len"]) for p in params)


def displayed_stimulus_size_px(cache: dict, width=None, height=None) -> tuple[int, int]:
    """
    Displayed stimulus (width, height) in stimulus pixels. This is a projector
    setting, so it cannot be calculated from the data: it is stored once per
    recording in the v2 alignment cache. Values given here override the cache
    (only needed for a NEW recording; check them against the lit area in Image 3).
    """
    if width is not None and height is not None:
        return int(width), int(height)
    if "displayed_stim_width_px" in cache and "displayed_stim_height_px" in cache:
        return (
            int(np.asarray(cache["displayed_stim_width_px"]).item()),
            int(np.asarray(cache["displayed_stim_height_px"]).item()),
        )
    raise ValueError(
        "The displayed stimulus size is not in this recording's alignment cache yet.\n"
        "Set displayed_stim_width_px / displayed_stim_height_px (stimulus pixels,\n"
        "e.g. 800 x 800 or 800 x 1000) for this NEW recording. Check: width/height\n"
        "should match the lit area of the stimulus in Image 3."
    )


def displayed_box_corners_px(
    edge_top_px, edge_bottom_px, edge_side: str, displayed_width_px: int, displayed_height_px: int
) -> np.ndarray:
    """
    Corners (5 x 2, closed) of the displayed stimulus in Image 3 implied by the
    clicked edge and the displayed size. Drawn on Image 3 as a visual CHECK of the
    displayed size: the box should sit exactly on the lit stimulus.
    """
    top, bottom = np.asarray(edge_top_px, float), np.asarray(edge_bottom_px, float)
    height = float(np.linalg.norm(bottom - top))
    down = (bottom - top) / height
    right = np.array([down[1], -down[0]])
    across = right if str(edge_side).lower() == "left" else -right
    width = height * displayed_width_px / displayed_height_px
    far_top, far_bottom = top + across * width, bottom + across * width
    return np.array([top, bottom, far_bottom, far_top, top])


def require_v2_rf_frame(dataset: xr.Dataset, source: str = "dataset") -> None:
    """Stop with a clear message if the RF maps are not in the v2 stimulus frame."""
    frame = dataset.attrs.get("rf_map_frame")
    if frame != RF_MAP_FRAME_V2:
        raise ValueError(
            f"{source} has rf_map_frame={frame!r}, expected {RF_MAP_FRAME_V2!r}.\n"
            "Its RF maps were written by the old reduce_all.py (x and y swapped). "
            "Convert it first:\n"
            "    python -m combined_analysis.convert_noise_data_v2 <noise_analysis folder>\n"
            "and point the script at the resulting noise_data_v2.nc."
        )


def stimulus_centre_from_clicked_edge(
    edge_top_px,
    edge_bottom_px,
    edge_side: str,
    displayed_width_px: int,
    displayed_height_px: int,
) -> np.ndarray:
    """
    Centre of the displayed stimulus in Image 3 from the clicked vertical edge.

    The displayed width/height (stimulus px) must be this recording's real
    displayed size (e.g. 800 x 800 or 800 x 1000): the centre lies half a
    displayed width away from the clicked edge.
    """
    edge_top_px = np.asarray(edge_top_px, dtype=float)
    edge_bottom_px = np.asarray(edge_bottom_px, dtype=float)
    down_vec = edge_bottom_px - edge_top_px
    height_in_image_px = float(np.linalg.norm(down_vec))
    down_unit = down_vec / height_in_image_px
    right_unit = np.array([down_unit[1], -down_unit[0]], dtype=float)
    half_width = height_in_image_px * (displayed_width_px / displayed_height_px) / 2.0
    edge_mid = 0.5 * (edge_top_px + edge_bottom_px)
    side = str(edge_side).lower()
    if side == "left":
        return edge_mid + right_unit * half_width
    if side == "right":
        return edge_mid - right_unit * half_width
    raise ValueError(f"edge_side must be 'left' or 'right', got {edge_side!r}")


@dataclass
class AlignmentGeometry:
    """Transforms and scale factors copied from the final alignment logic."""

    tform_image1_to_image3: np.ndarray
    um_per_px_image3: float
    corrected_stim_um_per_px: float
    rf_scale_factor: float
    A_projected_um_to_image3: np.ndarray
    A_image3_to_projected_um: np.ndarray
    A_image1_to_projected_um: np.ndarray
    A_raw_rf_to_projected_um: np.ndarray
    visual_stim_center_px_image3: np.ndarray
    stim_top_left_px: np.ndarray
    stim_bottom_left_px: np.ndarray



def load_alignment_geometry(
    alignment_cache_path: str | Path,
    *,
    dataset_assumed_um_per_px: float | None = None,
    displayed_stim_width_px: int | None = None,
    displayed_stim_height_px: int | None = None,
) -> AlignmentGeometry:
    """
    THE alignment geometry of one recording, from its alignment cache (.npz).

    The stimulus sizes and dataset pixel size are read from the cache (written by
    Alignment_pipeline/RF_alignment.py). Arguments, if given, override them;
    they are required for older caches that do not contain them yet.

    v2: A_raw_rf_to_projected_um is chosen by recording date (read from the cache
    file name): before 8 Oct 2026 (x, y) -> (-y, -x); from 8 Oct 2026 (x, y) -> (y, -x).
    The displayed-stimulus centre is recomputed from the clicked edge with the
    recording's displayed width/height. The scale calculation is unchanged. The
    cache stores the clicked stimulus edge as stim_edge_top_px /
    stim_edge_bottom_px, irrespective of whether LEFT or RIGHT was clicked.
    """
    cache_path = Path(alignment_cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Alignment cache not found: {cache_path}\n"
            "Run the RF alignment script first so its .npz cache exists."
        )

    loaded = np.load(cache_path, allow_pickle=True)

    def from_cache(name, value, cast):
        if value is not None:
            return cast(value)
        if name in loaded.files:
            return cast(np.asarray(loaded[name]).item())
        raise KeyError(
            f"{name} is not in {cache_path.name} (older cache). Pass it as an "
            "argument, or re-run CELL 1 of Alignment_pipeline/RF_alignment.py "
            "with the recording's settings to store it."
        )

    dataset_assumed_um_per_px = from_cache(
        "dataset_assumed_um_per_px", dataset_assumed_um_per_px, float
    )
    displayed_stim_width_px = from_cache("displayed_stim_width_px", displayed_stim_width_px, int)
    displayed_stim_height_px = from_cache(
        "displayed_stim_height_px", displayed_stim_height_px, int
    )
    required = {
        "tform_2_to_3_params",
        "um_per_px_image3",
        "stim_edge_top_px",
        "stim_edge_bottom_px",
        "visual_stim_center_px_image3",
        "stimulus_edge_side",
    }
    missing = sorted(required.difference(loaded.files))
    if missing:
        raise KeyError(
            "Alignment cache is not in the format produced by the current RF "
            f"alignment script. Missing: {missing}"
        )

    tform_2_to_3_params = np.asarray(loaded["tform_2_to_3_params"], dtype=float)
    um_per_px_image3 = float(loaded["um_per_px_image3"])
    px_per_um_image3 = 1.0 / um_per_px_image3

    stim_edge_top_px = np.asarray(loaded["stim_edge_top_px"], dtype=float)
    stim_edge_bottom_px = np.asarray(loaded["stim_edge_bottom_px"], dtype=float)
    visual_stim_center_px_image3 = np.asarray(
        loaded["visual_stim_center_px_image3"], dtype=float
    )
    stimulus_edge_side = str(np.asarray(loaded["stimulus_edge_side"]).item()).lower()

    if stimulus_edge_side not in {"left", "right"}:
        raise ValueError(
            "Cached stimulus_edge_side must be 'left' or 'right'; "
            f"got {stimulus_edge_side!r}."
        )

    # v2: recompute the displayed-stimulus centre from the clicked edge and this
    # recording's displayed width/height (the cached centre was made with a
    # hard-coded 800 x 1000, which is wrong for square stimuli).
    cached_centre = visual_stim_center_px_image3
    visual_stim_center_px_image3 = stimulus_centre_from_clicked_edge(
        stim_edge_top_px,
        stim_edge_bottom_px,
        stimulus_edge_side,
        displayed_stim_width_px,
        displayed_stim_height_px,
    )
    print(
        "v2 stimulus centre: "
        f"({visual_stim_center_px_image3[0]:.2f}, {visual_stim_center_px_image3[1]:.2f}) px; "
        f"cached ({cached_centre[0]:.2f}, {cached_centre[1]:.2f}); shift "
        f"{np.linalg.norm(visual_stim_center_px_image3 - cached_centre) * um_per_px_image3:.0f} µm"
    )

    # ------------------------------------------------------------
    # EXACT RF ALIGNMENT SCALE CALCULATION
    # ------------------------------------------------------------
    clicked_edge_vec_px = stim_edge_bottom_px - stim_edge_top_px
    clicked_edge_length_px = float(np.linalg.norm(clicked_edge_vec_px))

    if clicked_edge_length_px == 0:
        raise ValueError("stim_edge_top_px and stim_edge_bottom_px are identical.")

    clicked_edge_length_um = clicked_edge_length_px * um_per_px_image3
    corrected_stim_um_per_px = clicked_edge_length_um / displayed_stim_height_px
    rf_scale_factor = corrected_stim_um_per_px / dataset_assumed_um_per_px

    print("------------------------------------------------------------")
    print("SCALE CHECK — MATCHING RF ALIGNMENT SCRIPT")
    print("------------------------------------------------------------")
    print(
        f"Clicked {stimulus_edge_side.upper()} vertical edge length: "
        f"{clicked_edge_length_px:.2f} Image 3 px"
    )
    print(
        f"Clicked {stimulus_edge_side.upper()} vertical edge length: "
        f"{clicked_edge_length_um:.2f} µm"
    )
    print(f"Displayed stimulus height: {displayed_stim_height_px} stimulus px")
    print(f"Corrected stimulus pixel size: {corrected_stim_um_per_px:.4f} µm/px")
    print(f"RF dataset assumed pixel size: {dataset_assumed_um_per_px:.4f} µm/px")
    print(f"RF coordinate scale factor: {rf_scale_factor:.4f}")

    # ------------------------------------------------------------
    # EXACT PROJECTED-STIMULUS COORDINATE SYSTEM
    # ------------------------------------------------------------
    # +x = right across displayed stimulus
    # +y = down along clicked vertical edge
    # origin = displayed stimulus centre
    stim_down_unit_px = clicked_edge_vec_px / clicked_edge_length_px
    stim_right_unit_px = np.array(
        [stim_down_unit_px[1], -stim_down_unit_px[0]],
        dtype=float,
    )

    M_projected_um_to_image3 = np.column_stack(
        [
            stim_right_unit_px * px_per_um_image3,
            stim_down_unit_px * px_per_um_image3,
        ]
    )

    A_projected_um_to_image3 = np.eye(3, dtype=float)
    # ORIENTATION [FRAME DEFINITION, no flip] Projected frame in Image 3: +y runs DOWN along the clicked
    # stimulus edge, +x is 90° clockwise of it (right), origin = displayed-stimulus centre.
    A_projected_um_to_image3[:2, :2] = M_projected_um_to_image3
    A_projected_um_to_image3[:2, 2] = visual_stim_center_px_image3

    A_image3_to_projected_um = np.linalg.inv(A_projected_um_to_image3)
    # ORIENTATION [CONE IMAGE, fitted] Cone image -> projected uses the affine fitted to your 12 electrode
    # clicks (Image 2 -> Image 3). Whatever rotation/flip the image needs comes from those clicks;
    # no fixed flip is added here. The cone image must already be in the door view (Fiji step).
    A_image1_to_projected_um = A_image3_to_projected_um @ tform_2_to_3_params

    # v2: RF maps are in stimulus coordinates; this places them in the projection.
    # ORIENTATION [RF ROTATION] chosen by recording date (from the cache file name):
    #   before 8 Oct 2026: (x, y) -> (-y, -x);  from 8 Oct 2026: (x, y) -> (y, -x).
    A_raw_rf_to_projected_um, transform_description = stimulus_to_projection_for_recording(
        cache_path.name
    )
    print(f"Stimulus -> projection: {transform_description}")

    origin_check = A_projected_um_to_image3 @ np.array([0.0, 0.0, 1.0])

    print("------------------------------------------------------------")
    print("TRANSFORM CHECK — MATCHING RF ALIGNMENT SCRIPT")
    print("------------------------------------------------------------")
    print("Raw RF [0,0] maps to Image 3 / displayed stimulus centre:")
    print(f"x = {origin_check[0]:.2f}, y = {origin_check[1]:.2f}")
    print(
        "Clicked displayed stimulus centre: "
        f"x = {visual_stim_center_px_image3[0]:.2f}, "
        f"y = {visual_stim_center_px_image3[1]:.2f}"
    )
    print(f"Raw (stimulus) -> projected transform: {transform_description}")
    print(f"Stimulus edge used for clicks: {stimulus_edge_side.upper()}")

    # AlignmentGeometry still uses the historical field names stim_top_left_px /
    # stim_bottom_left_px. They are not used for the maths below; store the actual
    # clicked top/bottom edge points there only for compatibility with the helper
    # dataclass and output routines.
    return AlignmentGeometry(
        tform_image1_to_image3=tform_2_to_3_params,
        um_per_px_image3=um_per_px_image3,
        corrected_stim_um_per_px=corrected_stim_um_per_px,
        rf_scale_factor=rf_scale_factor,
        A_projected_um_to_image3=A_projected_um_to_image3,
        A_image3_to_projected_um=A_image3_to_projected_um,
        A_image1_to_projected_um=A_image1_to_projected_um,
        A_raw_rf_to_projected_um=A_raw_rf_to_projected_um,
        visual_stim_center_px_image3=visual_stim_center_px_image3,
        stim_top_left_px=stim_edge_top_px,
        stim_bottom_left_px=stim_edge_bottom_px,
    )
