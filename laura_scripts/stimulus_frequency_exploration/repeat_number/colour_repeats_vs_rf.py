"""Colour-sequence noise: are more repeats worth it, or would new noise versions be better? (27 Nov 2026)

In the colour sequences each colour always gets the SAME frames every repeat (15/10: white/610/500/420 nm,
900-frame blocks, 4500 unique frames per colour x 4 repeats, 15 Hz; 22/10 P1: 413/500/610 nm, 500-frame
blocks, 4000 unique frames per colour x 9 repeats, 10 Hz). Per colour and cell: STAs from the first
k repeats (spikes whose whole STA window lies inside one block of that colour, as the pipeline does).
Chance: spikes shifted by whole colour cycles within their repeat (same colour, different frames).

Key test: STA noise x sqrt(spikes). If noise only came from the number of spikes, this stays constant
as repeats are added; if it rises, the fixed set of frames limits the RF ('frame-limited') and new,
unique noise versions would help more than repeats.
Also per k: RF yield vs chance, signal / noise, map and time-course agreement with all repeats,
centre size. 8-px blocks (16 µm), 10 ms bins, -800..+190 ms. Writes results/colour_repeats_vs_rf.csv.
"""
import pickle
from pathlib import Path
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

RESULTS = Path(__file__).resolve().parents[1] / "results"
STIMS = Path("/mnt/mea_nas/Laura/stimuli/stimuli_Oct_2026")
B, FS = 8, 20000
LAGS = -800 + 10 * np.arange(100)
HALF = 25
dev = "cuda"
rng = np.random.default_rng(0)
REC = [
    dict(name="15/10 (15 Hz)", root="/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00", idx=5,
         file="32px_15Hz_20mins_shuffle_x4.h5", order=["white", "610", "500", "420"], fpb=900, per_rep=18000,
         ks=[1, 2, 4]),
    dict(name="22/10 (10 Hz)", root="/mnt/mea_nas/Laura/zebrafish_22_10_2026/Phase_01", idx=4,
         file="32px_10Hz_20mins_shuffle_x4.h5", order=["413", "500", "610"], fpb=500, per_rep=12000,
         ks=[1, 2, 3, 5, 9]),
]


def quality(sta):
    ks = ndimage.gaussian_filter(sta.astype(float), (5, 5 / B, 5 / B), mode="constant", cval=np.median(sta))
    v = ks.var(axis=0)
    return v.max() / np.median(v)


def window(a, cy, cx):
    out = np.zeros((2 * HALF, 2 * HALF))
    H, W = a.shape
    y0, x0 = cy - HALF, cx - HALF
    ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + 2 * HALF, H), min(x0 + 2 * HALF, W)
    out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return out


rows = []
for R in REC:
    o = pickle.load(open(f"{R['root']}/overview", "rb"))
    r = o.stimulus_df[o.stimulus_df.stimulus_index == R["idx"]].iloc[0]
    assert r.sampling_freq == FS
    trig = np.asarray(r.trigger_fr_relative).astype(np.int64)
    n_tot = len(trig) - 1
    n_rep = n_tot // R["per_rep"]
    with h5py.File(STIMS / R["file"], "r") as f:
        S = f["Noise"][:]
    S = (S[:, ::B, ::B] == 255).astype(np.float32)
    nf, hb, wb = S.shape
    S_t = torch.from_numpy(S.reshape(nf, -1)).to(dev)
    S_t -= S_t.mean(0, keepdim=True)
    del S
    g = np.arange(n_tot)
    colour_of = np.asarray(R["order"])[((g % R["per_rep"]) // R["fpb"]) % len(R["order"])]
    block_id, file_frame, rep_of = g // R["fpb"], g % nf, g // R["per_rep"]
    rate = FS / np.median(np.diff(trig))
    lag_fr = LAGS / 1000 * rate
    cycle = len(R["order"]) * R["fpb"]
    k_cyc = R["per_rep"] // cycle
    sp = pd.read_parquet(f"{R['root']}/alldata.parquet")
    m = (sp.times >= r.begin_fr) & (sp.times < r.begin_fr + trig[-1])
    spikes = {c: gg.times.values - r.begin_fr for c, gg in sp[m].groupby("cell_index")}
    yy, xx = np.mgrid[:hb, :wb]
    wy, wx = np.mgrid[:2 * HALF, :2 * HALF]

    def positions(t):
        i = np.clip(np.searchsorted(trig, t, side="right") - 1, 0, n_tot - 1)
        return i + (t - trig[i]) / np.maximum(trig[i + 1] - trig[i], 1)

    def sta_per_repeat(p, colour):
        """Per repeat: (STA sum, n spikes) for spikes whose window lies inside one block of this colour."""
        lo = np.floor(p + lag_fr[0]).astype(int)
        hi = np.floor(p + lag_fr[-1]).astype(int)
        ok = (lo >= 0) & (hi < n_tot)
        lo, hi, p = lo[ok], hi[ok], p[ok]
        keep = (block_id[lo] == block_id[hi]) & (colour_of[lo] == colour)
        p, lo = p[keep], lo[keep]
        out = []
        for k in range(n_rep):
            pk = p[rep_of[lo] == k]
            if len(pk) == 0:
                out.append((np.zeros((100, hb, wb)), 0)); continue
            fi = file_frame[np.floor(pk[:, None] + lag_fr[None, :]).astype(int)]
            W = np.zeros((100, nf), np.float32)
            for li in range(100):
                W[li] = np.bincount(fi[:, li], minlength=nf)
            out.append(((torch.from_numpy(W).to(dev) @ S_t).cpu().numpy().reshape(100, hb, wb), len(pk)))
        return out

    def shifted(p):
        """Shift by whole colour cycles within each repeat (same colour, different frames)."""
        rep = (p // R["per_rep"]).astype(int)
        within = p - rep * R["per_rep"]
        sh = rng.integers(1, k_cyc) * cycle
        return np.sort(rep * R["per_rep"] + (within + sh) % R["per_rep"])

    for colour in R["order"]:
        for cell, t in spikes.items():
            p = positions(t)
            real = sta_per_repeat(p, colour)
            if sum(n for _, n in real) < 200:
                continue
            null = sta_per_repeat(shifted(p), colour)
            full_n = sum(n for _, n in real)
            ref = sum(s for s, _ in real) / full_n
            v = ndimage.gaussian_filter(ref, (2, 1, 1)).var(axis=0)
            cy, cx = np.unravel_index(v.argmax(), v.shape)
            tc_ref = ref[:, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].mean(axis=(1, 2))
            jref = np.argmax(np.abs(tc_ref) * (LAGS <= 0))
            map_ref = window(ndimage.gaussian_filter(ref[jref], 1), cy, cx)
            far = np.hypot(yy - cy, xx - cx) * 16 > 300
            for k in R["ks"]:
                n = sum(real[i][1] for i in range(k))
                nn = sum(null[i][1] for i in range(k))
                if n < 50 or nn < 50:
                    continue
                sta = sum(real[i][0] for i in range(k)) / n
                nsta = sum(null[i][0] for i in range(k)) / nn
                tc = sta[:, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].mean(axis=(1, 2))
                j = np.argmax(np.abs(tc) * (LAGS <= 0))
                noise = ndimage.uniform_filter(sta[j], 3)[far].std()
                mp = window(ndimage.gaussian_filter(sta[j], 1), cy, cx)
                sgn = np.sign(mp[HALF, HALF]) or 1.0
                lab, _ = ndimage.label(sgn * mp >= 0.5 * sgn * mp[HALF, HALF])
                rows.append(dict(recording=R["name"], colour=colour, cell=cell, repeats=k, n_repeats_total=n_rep,
                                 n_spikes=n, q=quality(sta), q_null=quality(nsta), snr=abs(tc[j]) / noise,
                                 noise_x_sqrt_spikes=noise * np.sqrt(n),
                                 centre_um2=(lab == lab[HALF, HALF]).sum() * 256.0,
                                 map_r_vs_all=np.corrcoef(mp.ravel(), map_ref.ravel())[0, 1],
                                 tc_r_vs_all=np.corrcoef(tc, tc_ref)[0, 1]))
        print(R["name"], colour, "done", flush=True)
    del S_t; torch.cuda.empty_cache()

df = pd.DataFrame(rows)
chance = df.groupby(["recording", "colour", "repeats"]).q_null.max().rename("chance_max")
df = df.join(chance, on=["recording", "colour", "repeats"])
df["rf"] = df.q > df.chance_max
df.to_csv(RESULTS / "colour_repeats_vs_rf.csv", index=False)
print(df.groupby(["recording", "colour", "repeats"]).agg(rf=("rf", "sum"), chance=("chance_max", "first")).to_string())
