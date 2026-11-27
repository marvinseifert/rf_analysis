"""Run build_fff_dataset.py for one recording with per-recording config overrides (repo file untouched)."""
import re, sys, os, time
from pathlib import Path
PIPE = Path("/home/laura/PycharmProjects/rf_analysis/laura_scripts/Seaborn/fff_feature_pipeline")
NAS = "/mnt/mea_nas/Laura"
BOX = "/home/laura/Box/SUSSEX/Experiments/Zebrafish/Experiments/Imaging"
CFG = {
    "14_08_2026": dict(ch="32px_15Hz_20mins_shuffle_x4", noise="noise_analysis_QI10", new=False),
    "22_07_2026": dict(ch="32px_15Hz_20mins_shuffle_x4", noise="noise_analysis_QI10", new=False),
    "15_05_2026": dict(ch="12px_20Hz_25mins_shuffle_white", noise="noise_analysis_QI10", new=False),
    "14_05_2026": dict(ch="12px_20Hz_25mins_shuffle_x12", noise="noise_analysis_QI10", new=False),
    "07_05_2026": dict(ch="12px_20Hz_20mins_shuffle_x12_white", noise="noise_analysis_QI10", new=False, mb=None),
    "07_08_2026": dict(ch="32px_15Hz_20mins_shuffle_x4", noise="noise_analysis_QI10", new=False, mb=None),
    "15_10_2026": dict(ch="32px_15Hz_20mins_shuffle_x4", noise="noise_analysis_QI10", new=False, mb=3),
}
rec = sys.argv[1]; c = CFG[rec]
dd, mm, yyyy = rec.split("_")
src = (PIPE / "build_fff_dataset.py").read_text()
over = []
def sub(name, value):
    over.append(f"{name} = {value}")
sub("RECORDING_ROOT", f'Path(r"{NAS}/zebrafish_{rec}/Phase_00")')
sub("OVERVIEW_PATH", "RECORDING_ROOT / \"overview\"")
sub("OUTPUT_DIR", "RECORDING_ROOT / \"fff_analysis\"")
sub("NOISE_DATA_PATH", f'RECORDING_ROOT / "{c["noise"]}" / "noise_data_v2.nc"')
sub("CONE_IMAGE_PATH", f'Path(r"{BOX}/{yyyy}{mm}{dd}_dragonfly_microscope/{rec}_p0_cones_flipped.jpg")')
sub("ALIGNMENT_CACHE_PATH", f'Path(r"{NAS}/zebrafish_{rec}/{rec}_alignment_videos/zebrafish_{rec}_phase_00_alignment_cache_v2.npz")')
sub("CONE_RF_CHANNEL", f'"{c["ch"]}"')
sub("RUN_CONE_REGULARITY", "True")
sub("RUN_MOVING_EDGE", "False")
for k in ("RUN_CSTEPS", "RUN_SCF", "RUN_CHIRP"):
    sub(k, str(c["new"]))
if c["new"]:
    sub("RUN_MOVING_BAR", str(c["mb"] is not None))
    if c["mb"] is not None:
        sub("MOVING_BAR_STIMULUS_ID", str(c["mb"]))
else:
    sub("RUN_MOVING_BAR", "False")
marker = "# =============================================================================\n# 5. HELPERS"
assert src.count(marker) == 1
src = src.replace(marker, "# --- per-recording overrides ---\n" + "\n".join(over) + "\n\n" + marker)
for p in [f"{NAS}/zebrafish_{rec}/Phase_00/{c['noise']}/noise_data_v2.nc",
          f"{BOX}/{yyyy}{mm}{dd}_dragonfly_microscope/{rec}_p0_cones_flipped.jpg"]:
    assert Path(p).exists(), p
os.chdir(PIPE); sys.path.insert(0, str(PIPE))
t = time.time()
exec(compile(src, str(PIPE / "build_fff_dataset.py"), "exec"), {"__name__": "__main__", "__file__": str(PIPE / "build_fff_dataset.py")})
print(f"\nDONE {rec} in {(time.time()-t)/60:.1f} min")
