import contextlib, io, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path("/home/laura/PycharmProjects/rf_analysis")
sys.path[:0] = [str(REPO), str(REPO / "laura_scripts/Seaborn/fff_feature_pipeline"), str(REPO / "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction")]
import imageio.v3 as iio
import extract_cone_regularity as ecr, cone_lattice_v3 as v3, mark_mosaic_regions as m
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file
rec = sys.argv[1]; pts = [tuple(map(float, p.split(":"))) for p in sys.argv[3].split(",")]  # µm from ONH (x:y)
cfg = ecr.ConeRegularityConfig(noise_data_path=Path("x"), cone_image_path=cone_image_file(rec), alignment_cache_path=alignment_cache_v2_file(rec), rf_channel="x")
with contextlib.redirect_stdout(io.StringIO()):
    al = ecr._load_alignment_geometry(cfg)
raw = iio.imread(cfg.cone_image_path); print("raw image", raw.shape, raw.dtype, "max", raw.max())
cone = ecr._prepare_cone_intensity(raw)
onh = json.load(open(m.ONH_FILE))[rec]; upp = al["native_cone_um_per_px"]
fig, axs = plt.subplots(3, len(pts), figsize=(5.5 * len(pts), 16), squeeze=False)
for k, (dx, dy) in enumerate(pts):
    px = np.array([onh["x_px"] + dx / upp, onh["y_px"] + dy / upp, 1.0])
    pu = al["A_image1_to_projected_um"] @ px; c = (pu[0] / pu[2], pu[1] / pu[2])
    crop, u, valid = ecr._sample_aligned_square_crop(cone_intensity=cone, centre_projected_um=c, A_projected_um_to_image1=al["A_projected_um_to_image1"], crop_size_um=200.0, requested_um_per_px=upp)
    out, det = v3.analyse_cone_crop_v3(crop, u, cfg, return_details=True)
    proc = ecr.preprocess_for_lattice(np.where(np.isfinite(crop), crop, np.nanmedian(crop)), sigma_px=cfg.preprocess_sigma_px)
    print(f"({dx:.0f},{dy:.0f}) valid {valid:.2f} v3 {out['lattice_regularity']:.3f} a {out['lattice_spacing_a_um']:.1f} b {out['lattice_spacing_b_um']:.1f} found {out['lattice_points_found']}, sampled", np.round(det.get('values', []), 2))
    lo, hi = np.nanpercentile(crop, [1, 99.5]); axs[0, k].imshow(crop, cmap="gray", vmin=lo, vmax=hi); axs[0, k].set_title(f"({dx:.0f}, {dy:.0f}) µm from ONH: v3 {out['lattice_regularity']:.2f}")
    axs[1, k].imshow(proc, cmap="gray"); axs[1, k].set_title("after high-pass + window")
    ac = det["ac"]; cy, cx = np.array(ac.shape) // 2; h = int(30 / u)
    axs[2, k].imshow(ac[cy - h:cy + h, cx - h:cx + h], cmap="magma", vmin=-.2, vmax=.6, extent=[-30, 30, 30, -30])
    if det.get("predicted") is not None:
        p = det["predicted"] * u; axs[2, k].scatter(p[:, 0], p[:, 1], marker="x", color="lime", s=60)
    axs[2, k].set_title("autocorrelation ±30 µm (x = predicted lattice points)")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=65)
