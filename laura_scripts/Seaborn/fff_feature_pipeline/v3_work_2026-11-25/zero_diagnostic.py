import contextlib, io, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path("/home/laura/PycharmProjects/rf_analysis")
sys.path[:0] = [str(REPO), str(REPO / "laura_scripts/Seaborn/fff_feature_pipeline"), str(REPO / "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction")]
import imageio.v3 as iio
import extract_cone_regularity as ecr
import mark_mosaic_regions as m
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file
rec = "15_05_2026"
f = pd.read_csv(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
f = f[f.cone_regularity_index.notna()].copy()
f["marked"] = [str(x) for x in m.region_of_projected_um(rec, f.cone_rf_center_x_um, f.cone_rf_center_y_um)]
z = f[(f.marked == "crystalline") & (f.cone_regularity_index == 0)]
print("15/05 crystalline cells at 0:", len(z), "| first-shell peaks:", z.cone_first_shell_n_peaks.value_counts().to_dict())
hi = f[f.marked == "crystalline"].sort_values("cone_regularity_index").iloc[-10]
pick = [z.iloc[0], z.iloc[len(z) // 2], hi]
cfg = ecr.ConeRegularityConfig(noise_data_path=Path("x"), cone_image_path=cone_image_file(rec), alignment_cache_path=alignment_cache_v2_file(rec), rf_channel="x")
with contextlib.redirect_stdout(io.StringIO()):
    al = ecr._load_alignment_geometry(cfg)
cone = ecr._prepare_cone_intensity(iio.imread(cfg.cone_image_path))
fig, axs = plt.subplots(2, 3, figsize=(16, 11))
for k, r in enumerate(pick):
    crop, upp, _ = ecr._sample_aligned_square_crop(cone_intensity=cone, centre_projected_um=(r.cone_rf_center_x_um, r.cone_rf_center_y_um), A_projected_um_to_image1=al["A_projected_um_to_image1"], crop_size_um=200.0, requested_um_per_px=al["native_cone_um_per_px"])
    proc = ecr.preprocess_for_lattice(np.where(np.isfinite(crop), crop, np.nanmedian(crop)), sigma_px=cfg.preprocess_sigma_px)
    ac = ecr.compute_autocorrelation(proc)
    ex = ecr.central_peak_mask(ac, cfg.centre_exclusion_radius_px, cfg.central_peak_threshold)
    coords, vec, radii, _ = ecr.detect_autocorr_peaks(ac, cfg.ac_peak_min_distance_px, cfg.centre_exclusion_radius_px, cfg.max_num_ac_peaks, cfg.ac_peak_threshold_rel, exclude_mask=ex)
    shell = ecr.first_shell_mask(radii, cfg.first_shell_tol)
    lo, hi_ = np.nanpercentile(crop, [1, 99.5])
    axs[0, k].imshow(crop, cmap="gray", vmin=lo, vmax=hi_, extent=[0, 200, 200, 0])
    axs[0, k].set_title(f"cell {int(r.cell_index)}: crystallinity {r.cone_regularity_index:.2f}\n(marked crystalline) 200 µm crop", fontsize=11)
    cy, cx = np.array(ac.shape) // 2; half = 25 / upp  # show +-25 um of shifts
    axs[1, k].imshow(ac, cmap="magma", extent=[(-cx) * upp, (ac.shape[1] - cx) * upp, (ac.shape[0] - cy) * upp, (-cy) * upp], vmin=-0.2, vmax=0.6)
    axs[1, k].contour((np.arange(ac.shape[1]) - cx) * upp, (np.arange(ac.shape[0]) - cy) * upp, ex, levels=[0.5], colors="white", linewidths=1)
    v = vec * upp
    axs[1, k].scatter(v[~shell, 0], v[~shell, 1], s=40, facecolor="none", edgecolor="cyan", label="other peaks")
    axs[1, k].scatter(v[shell, 0], v[shell, 1], s=90, marker="x", color="lime", lw=2.5, label=f"'first shell' ({shell.sum()} peaks; need >= 4)")
    r0 = radii.min() * upp if len(radii) else np.nan
    for t in (1 - 0.375, 1 + 0.375):
        axs[1, k].add_patch(plt.Circle((0, 0), r0 * t, fill=False, ls=":", color="lime"))
    axs[1, k].set_xlim(-25, 25); axs[1, k].set_ylim(25, -25)
    srt = np.sort(radii * upp)[:8]
    axs[1, k].set_title("autocorrelation (shift in µm), nearest peaks at\n" + ", ".join(f"{x:.1f}" for x in srt) + " µm", fontsize=10)
    axs[1, k].legend(loc="lower left", fontsize=8)
fig.suptitle("Why clearly crystalline mosaic can score 0: only peaks within ±25-37 % of the NEAREST peak distance count as the 'first shell'", fontsize=13)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=75)
