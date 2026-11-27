"""Is the RF quality index biased by noise frame rate? (15/10/2026, 32px 5/10/15/20 Hz)

For every cell and rate: STA from real spikes and from time-shifted spikes (chance level),
both at 8-px block resolution (stimulus is exactly uniform in 8x8 blocks), 10 ms bins,
-800..+190 ms, then the same quality formula as rf_analysis/calculate_quality.py
(gaussian smooth, max/median of per-pixel temporal variance).
"""
import pickle, h5py, hdf5plugin, numpy as np, pandas as pd, torch
from scipy.ndimage import gaussian_filter

ROOT = "/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00"
STIM = "/mnt/mea_nas/Laura/stimuli/stimuli_Oct_2026/32px_{}Hz_20mins_shuffle_x4.h5"
RATES = {5: 9, 10: 10, 15: 11, 20: 12}
B = 8                      # block size in stimulus px
FS = 20000
LAGS_MS = -800 + 10 * np.arange(100)
N_SHIFTS = 3
rng = np.random.default_rng(0)
dev = "cuda"

o = pickle.load(open(f"{ROOT}/overview", "rb"))
sp = pd.read_parquet(f"{ROOT}/alldata.parquet")
ncell = 281


def quality(k):
    """k: (time, y, x). Same as calculate_quality, spatial sigma scaled to 8-px blocks."""
    ks = gaussian_filter(k.astype(float), (5, 5 / B, 5 / B), mode="constant", cval=np.median(k))
    v = ks.var(axis=0)
    return v.max() / np.median(v)


out = []
for hz, idx in RATES.items():
    r = o.stimulus_df[o.stimulus_df.stimulus_index == idx].iloc[0]
    trig = np.asarray(r.trigger_fr_relative)
    with h5py.File(STIM.format(hz), "r") as f:
        S = f["Noise"][:]
    nf = S.shape[0]
    S = (S[:, ::B, ::B] == 255).astype(np.float32)          # (nf, 100, 128)
    hb, wb = S.shape[1:]
    S_t = torch.from_numpy(S.reshape(nf, -1)).to(dev)
    del S
    end = trig[nf]                                          # end of last frame
    m = (sp.times >= r.begin_fr) & (sp.times < r.begin_fr + end)
    cell_spikes = {c: g.times.values - r.begin_fr for c, g in sp[m].groupby("cell_index")}
    lag_samp = (LAGS_MS * FS // 1000).astype(np.int64)

    def sta(t):
        tt = t[:, None] + lag_samp[None, :]                 # (nspk, 100)
        fi = np.searchsorted(trig[: nf + 1], tt, side="right") - 1
        ok = (fi >= 0) & (fi < nf)
        W = np.zeros((100, nf), np.float32)
        for li in range(100):
            W[li] = np.bincount(fi[ok[:, li], li], minlength=nf)
        return (torch.from_numpy(W).to(dev) @ S_t).cpu().numpy().reshape(100, hb, wb)

    for c in range(ncell):
        t = cell_spikes.get(c)
        if t is None or len(t) < 20:
            continue
        row = dict(rate=hz, cell=c, n_spikes=len(t), q_real=quality(sta(t)))
        for s in range(N_SHIFTS):
            off = int(rng.uniform(0.1, 0.9) * end)
            row[f"q_null{s}"] = quality(sta((t + off) % end))
        out.append(row)
    print(hz, "Hz done", flush=True)
    del S_t; torch.cuda.empty_cache()

df = pd.DataFrame(out)
df.to_csv(Path(__file__).resolve().parents[1] / "results" / "quality_null_by_rate.csv", index=False)
print(df.groupby("rate").size())
