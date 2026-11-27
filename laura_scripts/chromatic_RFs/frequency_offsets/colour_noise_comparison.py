"""Colour noise comparison (27 Nov 2026): RF signal, blur, half width and integration time per colour.

Recordings (32px, 64 µm squares, shuffled):
  - 22/10/2026 Phase_01, stim 4: 10 Hz, colours 413 ('420') / 500 / 610 nm switching every 500 frames,
    20-min sequence x 9. Each colour always gets the same 1/3 of the frames (4000 unique frames x 9).
    Note: LED power file of 21/10 says the 610 nm LED was replaced by a 622 nm LED (3.3 nW; others 4.5 nW).
  - 15/10/2026 Phase_00, stim 5: 15 Hz, white / 610 / 500 / 420 nm switching every 900 frames,
    x4 (4500 unique frames per colour x 4).
  - 15/10/2026 Phase_00, stim 11: 15 Hz white noise alone (18000 unique frames), for reference.

Per colour, same methods as the rate comparison (8-px blocks = 16 µm, 10 ms bins, -800..+190 ms):
  - spikes kept only if their whole STA window lies inside one block of that colour (as the
    colour-sequence pipeline does)
  - chance: spike train shifted by whole colour cycles (same colour, different frames, never a whole
    number of repeats) -> 3 shifts per cell; RF 'above chance' = quality > highest chance quality
  - quality = max/median of temporal variance (gaussian smooth 5, 5/8, 5/8)
  - signal / noise = RF-centre (3x3 blocks) peak / SD of 3x3-block averages > 300 µm away at the peak lag
  - time course of the RF centre: width at half height, integration time (main lobe; both lobes)
Writes colour_noise_comparison.csv (chromatic_RFs/frequency_offsets/results/).
"""
import io, pickle
from pathlib import Path
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent / "results"   # chromatic_RFs/frequency_offsets/results
SFE = Path(__file__).resolve().parents[2] / "stimulus_frequency_exploration" / "response_width_integration"
B, FS = 8, 20000
LAGS = -800 + 10 * np.arange(100)
dev = "cuda"
rng = np.random.default_rng(0)
STIMS = Path("/mnt/mea_nas/Laura/stimuli/stimuli_Oct_2026")
REC = [
    dict(name="22/10 (10 Hz)", root="/mnt/mea_nas/Laura/zebrafish_22_10_2026/Phase_01", idx=4,
         file="32px_10Hz_20mins_shuffle_x4.h5", order=["413", "500", "610"], fpb=500, per_rep=12000),
    dict(name="15/10 (15 Hz)", root="/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00", idx=5,
         file="32px_15Hz_20mins_shuffle_x4.h5", order=["white", "610", "500", "420"], fpb=900, per_rep=18000),
    dict(name="15/10 (15 Hz)", root="/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00", idx=11,
         file="32px_15Hz_20mins_shuffle_x4.h5", order=["white alone"], fpb=18000, per_rep=18000),
]

import sys
sys.path.insert(0, str(SFE))
from timecourse_measures import integration, width_and_peak  # noqa: E402


def quality(sta):
    ks = ndimage.gaussian_filter(sta.astype(float), (5, 5 / B, 5 / B), mode="constant", cval=np.median(sta))
    v = ks.var(axis=0)
    return v.max() / np.median(v)


rows, tcs = [], []
for R in REC:
    o = pickle.load(open(f"{R['root']}/overview", "rb"))
    r = o.stimulus_df[o.stimulus_df.stimulus_index == R["idx"]].iloc[0]
    trig = np.asarray(r.trigger_fr_relative).astype(np.int64)
    n_tot = len(trig) - 1
    with h5py.File(STIMS / R["file"], "r") as f:
        S = f["Noise"][:]
    S = (S[:, ::B, ::B] == 255).astype(np.float32)
    nf, hb, wb = S.shape
    S_t = torch.from_numpy(S.reshape(nf, -1)).to(dev)
    S_t -= S_t.mean(0, keepdim=True)
    del S
    g = np.arange(n_tot)
    colour_of = np.asarray(R["order"])[((g % R["per_rep"]) // R["fpb"]) % len(R["order"])]
    block_id = g // R["fpb"]
    file_frame = g % nf
    rate = FS / np.median(np.diff(trig))
    lag_fr = LAGS / 1000 * rate                                   # lags in frames (fractional)
    cycle = len(R["order"]) * R["fpb"]
    k_cycles = R["per_rep"] // cycle
    sp = pd.read_parquet(f"{R['root']}/alldata.parquet")
    m = (sp.times >= r.begin_fr) & (sp.times < r.begin_fr + trig[-1])
    spikes = {c: gg.times.values - r.begin_fr for c, gg in sp[m].groupby("cell_index")}

    def positions(t):
        i = np.clip(np.searchsorted(trig, t, side="right") - 1, 0, n_tot - 1)
        return i + (t - trig[i]) / np.maximum(trig[i + 1] - trig[i], 1)

    def sta_for(p, colour):
        """p: spike positions in frames. Keep spikes whose whole window is in one block of this colour."""
        lo = np.floor(p + lag_fr[0]).astype(int)
        hi = np.floor(p + lag_fr[-1]).astype(int)
        ok = (lo >= 0) & (hi < n_tot)
        lo, hi, p = lo[ok], hi[ok], p[ok]
        keep = (block_id[lo] == block_id[hi]) & (colour_of[lo] == colour)
        p = p[keep]
        if len(p) < 50:
            return None, len(p)
        fi = file_frame[np.floor(p[:, None] + lag_fr[None, :]).astype(int)]
        W = np.zeros((100, nf), np.float32)
        for li in range(100):
            W[li] = np.bincount(fi[:, li], minlength=nf)
        return (torch.from_numpy(W).to(dev) @ S_t).cpu().numpy().reshape(100, hb, wb) / len(p), len(p)

    for colour in R["order"]:
        for cell, t in spikes.items():
            p = positions(t)
            sta, n = sta_for(p, colour)
            if sta is None:
                continue
            q = quality(sta)
            nulls = []
            for _ in range(3):
                mshift = rng.integers(1, k_cycles) * cycle if k_cycles > 1 else rng.integers(1, R["fpb"] // 2)
                mshift += rng.integers(0, n_tot // R["per_rep"]) * R["per_rep"]
                s2, _ = sta_for(np.sort((p + mshift) % n_tot), colour)
                if s2 is not None:
                    nulls.append(quality(s2))
            v = ndimage.gaussian_filter(sta, (2, 1, 1)).var(axis=0)
            cy, cx = np.unravel_index(v.argmax(), v.shape)
            tc = sta[:, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].mean(axis=(1, 2))
            j = np.argmax(np.abs(tc) * (LAGS <= 0))
            yy, xx = np.mgrid[:hb, :wb]
            far = np.hypot(yy - cy, xx - cx) * 16 > 300
            noise = ndimage.uniform_filter(sta[j], 3)[far].std()
            tcb = tc - np.median(tc[:30])
            w, pk, _ = width_and_peak(tcb)
            _, im, io_ = integration(tcb, LAGS, 10.0)
            tcs.append(tcb.astype(np.float32))
            rows.append(dict(recording=R["name"], stim=R["idx"], colour=colour, cell=cell, n_spikes=n, q=q,
                             q_null_max=max(nulls) if nulls else np.nan, snr=abs(tc[j]) / noise,
                             polarity="ON" if tc[j] > 0 else "OFF", peak_ms=pk, width_ms=w,
                             int_main_ms=im, int_both_ms=im + io_, frame_ms=1000 / rate))
        print(R["name"], R["idx"], colour, "done", flush=True)
    del S_t; torch.cuda.empty_cache()

df = pd.DataFrame(rows)
chance = df.groupby(["recording", "stim", "colour"]).q_null_max.max().rename("chance_max")
df = df.join(chance, on=["recording", "stim", "colour"])
df["rf"] = df.q > df.chance_max
df.to_csv(HERE / "colour_noise_comparison.csv", index=False)
np.save(HERE / "colour_noise_comparison_timecourses.npy", np.array(tcs))   # row order = csv rows
summ = df.groupby(["recording", "stim", "colour"]).apply(lambda g: pd.Series(dict(
    cells=len(g), chance_max=g.chance_max.iloc[0], rf_cells=int(g.rf.sum()),
    snr_rf=g[g.rf].snr.median(), width=g[g.rf].width_ms.median(), int_main=g[g.rf].int_main_ms.median(),
    int_both=g[g.rf].int_both_ms.median(), peak=g[g.rf].peak_ms.median(),
    on=int((g.rf & (g.polarity == "ON")).sum()), off=int((g.rf & (g.polarity == "OFF")).sum()))), include_groups=False)
print(summ.round(1).to_string())
