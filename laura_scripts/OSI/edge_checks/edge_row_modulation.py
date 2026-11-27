"""Do slow moving edges (22/07, 12 fps, 24 µm/s) produce repeat-locked firing modulation at the cone-row
crossing rate (~1.5-6 Hz)? Reliable power = mean cross-spectrum between repeats (only stimulus-locked
structure survives); chance = same with each repeat shifted by a random offset within the sweep."""
import pickle, h5py, hdf5plugin, numpy as np, pandas as pd
from pathlib import Path
RES = Path(__file__).resolve().parent / "results"
rng = np.random.default_rng(0)
ROOT = "/mnt/mea_nas/Laura/zebrafish_22_07_2026/Phase_00"; STIM = "/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/"
o = pickle.load(open(ROOT + "/overview", "rb")); sd = o.stimulus_df; sp = pd.read_parquet(ROOT + "/alldata.parquet")
BIN = 0.04
out = []
for idx, fn in [(6, "moving_edge_weighted_13fps_OFF_800px_wide_800px_high__5px_thick_30.0px_ramp_1.0px_per_frame.h5"),
                (4, "moving_bar_weighted_13fps_ON_800px_wide_800px_high_5px_thick_30.0px_ramp_1.0px_per_frame.h5")]:
    rel = pd.read_csv(RES / f"edge_rel_22_07_2026_{idx}.csv"); thr = np.nanpercentile(rel.null, 99)
    cells = rel[rel.real > thr].cell.values
    h = h5py.File(STIM + fn, "r"); st, cnt = h["Direction_Start_Frame"][()], h["Direction_Frame_Count"][()]; n_per = h["Noise"].shape[0]
    r = sd[sd.stimulus_index == idx].iloc[0]; fs = r.sampling_freq; trig = np.asarray(r.trigger_fr_relative) + r.begin_fr; R = int(r.nr_repeats)
    for c in cells:
        t = sp[sp.cell_index == c].times.values
        for k in range(len(st)):
            segs = []
            for rep in range(R):
                a = trig[rep * n_per + st[k]]; b = trig[rep * n_per + st[k] + cnt[k]]
                segs.append(((t[(t >= a) & (t < b)] - a) / fs, (b - a) / fs))
            L = min(s[1] for s in segs); nb = int(L / BIN)
            def spec(shift):
                X = []
                for (x, _), sh in zip(segs, shift):
                    hst = np.histogram((x + sh) % L, np.arange(nb + 1) * BIN)[0].astype(float)
                    hst -= hst.mean(); X.append(np.fft.rfft(hst * np.hanning(nb)))
                cp = np.mean([np.real(X[i] * np.conj(X[j])) for i in range(R) for j in range(i + 1, R)], axis=0)
                return cp
            f = np.fft.rfftfreq(nb, BIN)
            band, env = (f >= 1.5) & (f <= 6), (f > 0.05) & (f < 0.5)
            real = spec([0] * R)
            nulls = np.array([spec(rng.uniform(0, L, R))[band].sum() for _ in range(30)])
            out.append(dict(stim=idx, cell=c, direction=k, band_real=real[band].sum(), env_real=real[env].sum(),
                            band_null_mean=nulls.mean(), band_null_sd=nulls.std(),
                            z=(real[band].sum() - nulls.mean()) / (nulls.std() + 1e-12)))
d = pd.DataFrame(out); d.to_csv(RES / "edge_row_modulation_22_07.csv", index=False)
for idx, g in d.groupby("stim"):
    print(f"stim {idx}: {g.cell.nunique()} reliable cells x 8 directions; row-band (1.5-6 Hz) repeat-locked power z: "
          f"median {g.z.median():.2f}, z>3 in {(g.z > 3).sum()}/{len(g)} cell-directions (chance ~{0.0013*len(g):.1f}); "
          f"band/envelope power ratio median {np.median(g.band_real / g.env_real):.3f}")
    best = g.sort_values("z", ascending=False).head(5)[["cell", "direction", "z"]]
    print(best.to_string(index=False))
