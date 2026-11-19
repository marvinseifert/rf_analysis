"""
Per-recording settings for the v2 alignment (RF_alignment.py reads these).

Add one entry per recording. RF_alignment.py only needs `recording = "<key>"`.
Paths are for this Linux PC: the NAS mounted at /mnt/mea_nas (CIFS) and Box at
/home/laura/Box. On Windows, change NAS_LAURA / BOX_IMAGING only.

Fields
------
phase                   "Phase_00", ...
noise_folder            noise_analysis folder (its noise_data_v2.nc is used)
cone_image              door-view (flipped) cone image, Box
electrode_image         door-view (flipped) electrode image, same pixel grid as cone_image
alignment_folder        folder holding the projection photo and the v2 alignment cache
video                   projection video (makes visual_stim_on_MEA.png if that is missing)
video_crop              (y1, y2, x1, x2) around the lit stimulus in the video
displayed_size          (width, height) of the displayed stimulus in the door view,
                        stimulus px. Projector setting; check with the red box in CELL 1.
channel_to_plot         RF channel shown in CELL 3
stimulus_edge_to_click  "left" or "right" vertical edge of the lit stimulus
The stimulus pixel size and raw stimulus size come from noise_params.json; the
stimulus -> projection transform comes from the recording date (see
alignment_geometry.stimulus_to_projection_for_recording).
"""

from pathlib import Path

NAS_LAURA = Path("/mnt/mea_nas/Laura")
BOX_IMAGING = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Experiments/Imaging")

RECORDINGS = {
    # 30/04: v2 clicks done 19/11/2026, but left out of the figures: its only noise
    # channel is 4px (8 µm pixels), too fine to show RFs on the cone mosaic this way.
    # The clicks only cover a strip on the right (no electrodes visible elsewhere).
    "30_04_2026": dict(
        phase="Phase_00",
        noise_folder=NAS_LAURA / "zebrafish_30_04_2026/Phase_00/noise_analysis",
        cone_image=BOX_IMAGING / "20260430_dragonfly_microscope/30_04_2026_p0_cones_flipped.jpg",
        electrode_image=BOX_IMAGING / "20260430_dragonfly_microscope/30_04_2026_p0_electrodes_flipped.jpg",
        alignment_folder=NAS_LAURA / "zebrafish_30_04_2026/30_04_2026_alignment_videos",
        # visual_stim_on_MEA.png already exists (made 8 Oct 2026), so the video is not
        # used; the crop is only needed if that image is ever remade.
        video="WIN_20260430_09_26_58_Pro.mp4",
        video_crop=None,
        # Lit area on visual_stim_on_MEA.png is ~0.82 as wide as tall -> 800 x 1000
        # (same window as 07/05, 14/05, 15/05). Stimulus file 1000 x 1000.
        displayed_size=(800, 1000),
        channel_to_plot="4px_20Hz_15mins_shuffle_white",  # stimulus index 4
        # The v1 clicks used the right edge (the left edge of the lit area is blurred).
        stimulus_edge_to_click="right",
    ),
    "07_05_2026": dict(
        phase="Phase_00",
        noise_folder=NAS_LAURA / "zebrafish_07_05_2026/Phase_00/noise_analysis",
        cone_image=BOX_IMAGING / "20260507_dragonfly_microscope/07_05_2026_p0_cones_flipped.jpg",
        electrode_image=BOX_IMAGING / "20260507_dragonfly_microscope/07_05_2026_p0_electrodes_flipped.jpg",
        alignment_folder=NAS_LAURA / "zebrafish_07_05_2026/07_05_2026_alignment_videos",
        # Two videos that day; 09:40 shows the electrodes across the whole stimulus.
        video="WIN_20260507_09_40_43_Pro.mp4",
        video_crop=(154, 494, 271, 546),
        displayed_size=(800, 1000),  # same window as 14/05 and 15/05
        channel_to_plot="12px_20Hz_20mins_shuffle_x12_white",
        stimulus_edge_to_click="left",
    ),
    "15_10_2026": dict(
        phase="Phase_00",
        # Folder renamed on the NAS on 19/11/2026: noise_analysis -> noise_analysis_QI10.
        noise_folder=NAS_LAURA / "zebrafish_15_10_2026/Phase_00/noise_analysis_QI10",
        cone_image=BOX_IMAGING / "20261015_dragonfly_microscope/15_10_2026_p0_cones_flipped.jpg",
        electrode_image=BOX_IMAGING / "20261015_dragonfly_microscope/15_10_2026_p0_electrodes_flipped.jpg",
        alignment_folder=NAS_LAURA / "zebrafish_15_10_2026/15_10_2026_alignment_videos",
        video="WIN_20261015_11_18_23_Pro.mp4",
        video_crop=(218, 658, 340, 716),
        # Projector window 1000 x 800 in the stimulus software = 800 wide x 1000 tall
        # in the door view (stimulus file 1024 x 800).
        displayed_size=(800, 1000),
        channel_to_plot="32px_15Hz_20mins_shuffle_x4",  # stimulus index 11
        stimulus_edge_to_click="left",
    ),
}


def settings_for(recording: str) -> dict:
    """Settings for one recording, plus derived paths."""
    if recording not in RECORDINGS:
        raise KeyError(f"No alignment settings for {recording!r}. Known: {sorted(RECORDINGS)}")
    s = dict(RECORDINGS[recording])
    s["recording_name"] = f"zebrafish_{recording}_{s['phase'].lower()}"
    s["path_to_data"] = Path(s["noise_folder"]) / "noise_data_v2.nc"
    s["image_3_visual_stim_path"] = Path(s["alignment_folder"]) / "visual_stim_on_MEA.png"
    s["video_path"] = Path(s["alignment_folder"]) / s["video"]
    s["alignment_cache_path"] = Path(s["alignment_folder"]) / f"{s['recording_name']}_alignment_cache_v2.npz"
    return s


# ------------------------------------------------------------
# Path helpers for any recording with a v2 alignment cache (used by
# plot_population_overlays_all_recordings.py and plot_top_quality_cells.py)
# ------------------------------------------------------------
# noise_analysis folders to try, lowest map threshold first (the main folder only
# keeps maps for quality >= 20; QI10 / QI0 keep them down to 10 / 0).
NOISE_FOLDERS_IN_ORDER = ["noise_analysis_QI10", "noise_analysis_QI0", "noise_analysis"]


def noise_data_v2_file(recording: str, phase: str = "Phase_00") -> Path:
    phase_folder = NAS_LAURA / f"zebrafish_{recording}" / phase
    for folder in NOISE_FOLDERS_IN_ORDER:
        f = phase_folder / folder / "noise_data_v2.nc"
        if f.exists():
            return f
    raise FileNotFoundError(f"No noise_data_v2.nc for {recording} in {phase_folder}")


def alignment_cache_v2_file(recording: str) -> Path:
    caches = sorted((NAS_LAURA / f"zebrafish_{recording}").glob("*_alignment_videos/*_alignment_cache_v2.npz"))
    if not caches:
        raise FileNotFoundError(f"No v2 alignment cache for {recording}; run RF_alignment.py first")
    return caches[0]


def cone_image_file(recording: str) -> Path:
    """Door-view (flipped) cone image on Box: Imaging/YYYYMMDD_dragonfly_microscope/DD_MM_YYYY_p0_cones_flipped.jpg"""
    dd, mm, yyyy = recording.split("_")
    return BOX_IMAGING / f"{yyyy}{mm}{dd}_dragonfly_microscope" / f"{recording}_p0_cones_flipped.jpg"
