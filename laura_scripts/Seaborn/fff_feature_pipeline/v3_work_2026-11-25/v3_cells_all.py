import contextlib, io, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/laura/PycharmProjects/rf_analysis")
sys.path[:0] = [str(REPO), str(REPO / "laura_scripts/Seaborn/fff_feature_pipeline"), str(REPO / "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction")]
import imageio.v3 as iio
import extract_cone_regularity as ecr, cone_lattice_v3 as v3, mark_mosaic_regions as m
from laura_scripts.Alignment_pipeline.alignment_recordings import alignment_cache_v2_file, cone_image_file
S = Path(sys.argv[1]); rec = sys.argv[2]
cfg = ecr.ConeRegularityConfig(noise_data_path=Path("x"), cone_image_path=cone_image_file(rec), alignment_cache_path=alignment_cache_v2_file(rec), rf_channel="x")
with contextlib.redirect_stdout(io.StringIO()):
    al = ecr._load_alignment_geometry(cfg)
cone = ecr._prepare_cone_intensity(iio.imread(cfg.cone_image_path))
f = pd.read_csv(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
f = f[f.cone_rf_center_x_um.notna()]
rows = []
for r in f.itertuples():
    crop, u, valid = ecr._sample_aligned_square_crop(cone_intensity=cone, centre_projected_um=(r.cone_rf_center_x_um, r.cone_rf_center_y_um), A_projected_um_to_image1=al["A_projected_um_to_image1"], crop_size_um=200.0, requested_um_per_px=al["native_cone_um_per_px"])
    o = v3.analyse_cone_crop_v3(crop, u, cfg) if valid >= cfg.min_valid_crop_fraction else {"lattice_regularity": np.nan}
    rows.append(dict(recording=rec, cell_index=r.cell_index, x_um=r.cone_rf_center_x_um, y_um=r.cone_rf_center_y_um, valid=valid, **o))
d = pd.DataFrame(rows)
d["marked"] = [str(x) for x in m.region_of_projected_um(rec, d.x_um, d.y_um)]
d.to_csv(S / f"v3cells_{rec}.csv", index=False); print(rec, "done", len(d))
