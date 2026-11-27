"""Example 200 um cone crops for each 0.1 step of the crystallinity index (same crops and
settings as build_fff_dataset.py). Only already-marked recordings, no positions shown."""
import contextlib, io, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path("/home/laura/PycharmProjects/rf_analysis")
sys.path[:0] = [str(REPO), str(REPO / "laura_scripts/Seaborn/fff_feature_pipeline"),
                str(REPO / "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction")]
import imageio.v3 as iio
import extract_cone_regularity as ecr
import mark_mosaic_regions as m
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file

RECS = sys.argv[2].split(",")
N = 6
BINS = [("0 (no lattice)", 0, 0)] + [(f"{a:.1f}-{a + .1:.1f}", a, a + .1) for a in np.arange(0, .5, .1)] + [("≥ 0.5", .5, 9)]
cache = {}
d = pd.read_csv(Path(sys.argv[1]).parent / "v3_eval_cells.csv")
d = d[d.recording.isin(RECS)].rename(columns={"x_um": "cone_rf_center_x_um", "y_um": "cone_rf_center_y_um"})
d["cone_regularity_index"] = d.lattice_regularity
cfg_of = {}
def crop_of(r):
    rec = r.recording
    if rec not in cache:
        cfg = ecr.ConeRegularityConfig(noise_data_path=Path("x"), cone_image_path=cone_image_file(rec),
                                       alignment_cache_path=alignment_cache_v2_file(rec), rf_channel="x")
        with contextlib.redirect_stdout(io.StringIO()):
            al = ecr._load_alignment_geometry(cfg)
        cache[rec] = (ecr._prepare_cone_intensity(iio.imread(cfg.cone_image_path)), al)
    cone, al = cache[rec]
    crop, upp, _ = ecr._sample_aligned_square_crop(cone_intensity=cone, centre_projected_um=(r.cone_rf_center_x_um, r.cone_rf_center_y_um),
                                                   A_projected_um_to_image1=al["A_projected_um_to_image1"], crop_size_um=200.0,
                                                   requested_um_per_px=al["native_cone_um_per_px"])
    return crop
fig, axs = plt.subplots(len(BINS), N, figsize=(N * 2.6, len(BINS) * 2.75))
for i, (name, lo, hi) in enumerate(BINS):
    g = d[d.cone_regularity_index == 0] if hi == 0 else d[(d.cone_regularity_index > lo) & (d.cone_regularity_index <= hi) if lo == 0 else (d.cone_regularity_index > lo) & (d.cone_regularity_index <= hi)]
    g = g.sort_values("cone_regularity_index")
    pick = g.iloc[np.unique(np.linspace(0, len(g) - 1, N).round().astype(int))] if len(g) else g
    axs[i, 0].set_ylabel(f"{name}\n(n = {len(g)})", fontsize=12, rotation=0, ha="right", va="center", labelpad=10)
    for j in range(N):
        ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if j >= len(pick): continue
        r = pick.iloc[j]; c = crop_of(r)
        lo_, hi_ = np.nanpercentile(c, [1, 99.5])
        ax.imshow(c, cmap="gray", vmin=lo_, vmax=hi_)
        col = m.COLOURS.get(r.marked, "white")
        ax.set_title(f"{r.cone_regularity_index:.2f} (v2 {r.v2:.2f}) {r.recording[:5]}  {r.marked if r.marked != 'None' else 'unmarked'}",
                     fontsize=8, color="black", backgroundcolor=(col if r.marked != "None" else "white"), alpha=1)
fig.suptitle("Lattice regularity v3: example 200 × 200 µm cone crops per 0.1 step (spread across each step)\n"
             f"recordings {', '.join(RECS)}; label colour = your marked region", fontsize=13)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=80)
print(d.groupby(pd.cut(d.cone_regularity_index, [-1e-9, 1e-9, .1, .2, .3, .4, .5, 1])).marked.value_counts().unstack(fill_value=0).to_string())
