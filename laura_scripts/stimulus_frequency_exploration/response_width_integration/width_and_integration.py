"""Half-height width and integration time of RF time courses (27 Nov 2026; replaces
response_width_20hz_clustering.py, integration_time.py and integration_vs_width_15_05.py).

A) 20 Hz recordings (12px, 60 µm squares): the five 20 Hz recordings of
   RGC_clustering/clustering_recordings_and_stimuli.txt + 15/05/2026 (12px 20 Hz white, stim 15).
   Cells as in the clustering selection: flash QI > 0.5 in any of the first scf / csteps / chirp, and
   RF quality > 10. Time course = temporal_sta in retina_master.sqlite (sta_single_pixel of
   noise_data_v2; 0.5 = no response) -> results/width_integration_20hz.csv
B) 15/10/2026 at 5/10/15/20 Hz, the 27 cells with an RF above chance at all rates (RF-centre time
   courses from rf_timecourse_by_noise_rate.npz) -> results/width_integration_15_10.csv (integration
   times only; half widths for 15/10 are in rf_timecourse_by_noise_rate.csv)
Measured values only: frames still smear both measures (20 Hz least).
"""
import io, sqlite3, sys, zlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timecourse_measures import LAGS, integration, width_and_peak  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
DB = "/mnt/mea_nas/Laura/RGC_clustering/master_data_extraction/retina_master.sqlite"
# recording: (noise stimulus index, [scf, csteps, chirp] of the first flash group)
REC = {"05_11_2025": (6, [0, 2, 1]), "02_12_2025": (8, [0, 1, 2]), "15_01_2026": (16, [0, 1, 2]),
       "26_02_2026": (12, [0, 1, 2]), "05_03_2026": (11, [0, 1, 2]), "15_05_2026": (15, [0, 1, 2])}


def measure(tc):
    tc = tc - np.median(tc[:30])
    w, pk, _ = width_and_peak(tc)
    _, m, o = integration(tc)
    return dict(polarity="ON" if tc[np.argmax(np.abs(tc) * (LAGS <= 0))] > 0 else "OFF",
                peak_ms=pk, width_ms=w, int_main_ms=m, int_both_ms=m + o)


# ---------------- A) 20 Hz recordings
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = []
for rec, (nidx, flashes) in REC.items():
    q = dict(con.execute("select cell_index, value from features where recording=? and context like ? and name='rf_quality'",
                         (rec, f"rf:{nidx}:%")).fetchall())
    fq = pd.DataFrame(con.execute(f"select cell_index, qi from observations where recording=? and stimulus_index in ({','.join('?' * len(flashes))})",
                                  (rec, *flashes)).fetchall(), columns=["cell", "qi"]).groupby("cell").qi.max()
    for cell, blob in con.execute("select cell_index, values_blob from features where recording=? and context like ? and name='temporal_sta'",
                                  (rec, f"rf:{nidx}:%")):
        if (q.get(cell) or 0) <= 10 or fq.get(cell, 0) <= 0.5:
            continue
        tc = np.load(io.BytesIO(zlib.decompress(blob))).astype(float)
        if not np.all(np.isfinite(tc)) or np.ptp(tc) == 0:
            continue
        rows.append(dict(recording=rec, cell=cell, rf_quality=q[cell], flash_qi=fq[cell], **measure(tc)))
A = pd.DataFrame(rows)
A.to_csv(RESULTS / "width_integration_20hz.csv", index=False)
print(f"A) 20 Hz recordings, {len(A)} cells — medians (ms):")
print(A.groupby("recording")[["peak_ms", "width_ms", "int_main_ms", "int_both_ms"]].median().round(0).to_string())
print(A.groupby("polarity")[["peak_ms", "width_ms", "int_main_ms", "int_both_ms"]].median().round(0).to_string())

# ---------------- B) 15/10, same 27 cells at each rate
z = np.load(RESULTS / "rf_timecourse_by_noise_rate.npz")
tcm = pd.read_csv(RESULTS / "rf_timecourse_by_noise_rate.csv")
good = tcm[tcm.rf]
cells = set(good.pivot(index="cell", columns="rate", values="t_peak").dropna().index)
rows = [dict(cell=c, rate=hz, **{k: v for k, v in measure(tc).items() if k in ("int_main_ms", "int_both_ms")})
        for c, hz, tc in zip(z["cells"], z["rates"], z["tc"]) if c in cells]
Bd = pd.DataFrame(rows)   # widths / peaks for 15/10: rf_timecourse_by_noise_rate.csv (fwhm, t_peak)
Bd.to_csv(RESULTS / "width_integration_15_10.csv", index=False)
print(f"\nB) 15/10, same {len(cells)} cells — medians (ms) per noise rate:")
print(Bd.groupby("rate")[["int_main_ms", "int_both_ms"]].median().round(0).to_string())
