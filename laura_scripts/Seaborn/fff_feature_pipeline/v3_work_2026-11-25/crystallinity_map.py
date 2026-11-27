"""Crystallinity index on a regular grid over a cone image (same calculation and settings as
build_fff_dataset.py), drawn over the cone image with the hand-marked regions."""
import contextlib, io, json, sys, time
from pathlib import Path
import numpy as np
REPO = Path("/home/laura/PycharmProjects/rf_analysis")
PIPE = REPO / "laura_scripts/Seaborn/fff_feature_pipeline"
TOOL = REPO / "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction"
sys.path[:0] = [str(REPO), str(PIPE), str(TOOL)]
import imageio.v3 as iio
import extract_cone_regularity as ecr
import mark_mosaic_regions as m
import cone_lattice_v3 as v3
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file

rec, step_um = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
out = Path(sys.argv[3])
PARAMS = dict(crop_size_um=200.0, preprocess_sigma_px=13.0, ac_peak_min_distance_px=10, centre_exclusion_radius_px=20,
              max_num_ac_peaks=80, ac_peak_threshold_rel=0.08, first_shell_tol=0.25, min_first_shell_peaks=4,
              min_valid_crop_fraction=0.95, resample_um_per_px=None)  # = build_fff_dataset.py
cfg = ecr.ConeRegularityConfig(noise_data_path=Path("unused"), cone_image_path=cone_image_file(rec),
                               alignment_cache_path=alignment_cache_v2_file(rec), rf_channel="unused", **PARAMS)
with contextlib.redirect_stdout(io.StringIO()):
    al = ecr._load_alignment_geometry(cfg)
cone = ecr._prepare_cone_intensity(iio.imread(cfg.cone_image_path))
H, W = cone.shape[:2]
# grid over the cone image, in projected µm
corners = ecr._transform_points(al["A_image1_to_projected_um"], np.array([[0, 0], [W, 0], [W, H], [0, H]], float))
xs = np.arange(corners[:, 0].min(), corners[:, 0].max() + step_um, step_um)
ys = np.arange(corners[:, 1].min(), corners[:, 1].max() + step_um, step_um)
res = np.full((len(ys), len(xs)), np.nan)
res3 = np.full((len(ys), len(xs)), np.nan)
t = time.time()
for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        crop, upp, valid = ecr._sample_aligned_square_crop(cone_intensity=cone, centre_projected_um=(x, y),
                              A_projected_um_to_image1=al["A_projected_um_to_image1"], crop_size_um=cfg.crop_size_um,
                              requested_um_per_px=al["native_cone_um_per_px"])
        if valid >= cfg.min_valid_crop_fraction:
            res[i, j] = ecr.analyse_cone_crop(crop=crop, um_per_px=upp, config=cfg)["cone_regularity_index"]
            res3[i, j] = v3.analyse_cone_crop_v3(crop, upp, cfg)["lattice_regularity"]
    if i == 0:
        print(f"{len(xs)} x {len(ys)} grid; first row {time.time() - t:.1f} s -> ~{(time.time() - t) * len(ys) / 60:.1f} min", flush=True)
np.savez(out.with_suffix(".npz"), xs=xs, ys=ys, res=res, res3=res3)
print(f"done in {(time.time() - t) / 60:.1f} min", flush=True)
