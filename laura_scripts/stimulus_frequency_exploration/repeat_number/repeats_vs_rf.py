"""How much do 2nd and 3rd repeats of the 20-min 32px_15Hz_20mins_shuffle_x4 noise add? (27 Nov 2026)

The noise sequence (18000 frames, 15 Hz, 20 min) is played 3x IDENTICALLY, so extra repeats add spikes
but no new frames. Recordings: 14/08/2026 Phase_00 stim 6 and 07/08/2026 Phase_00 stim 3 (both white x3, 15 Hz, 60 min);
stimulus = the July 2026 file (800 x 800 frames) these recordings were processed with.

Per cell: STAs from repeat 1, 2, 3 alone (20 min each); 1+2 (40 min) and 1+2+3 (60 min) are spike-count
weighted sums of those. 8-px blocks (16 µm), 10 ms bins, -800..+190 ms (as the frame-rate analyses).
Chance: spikes shifted circularly WITHIN each repeat (10-90 % of a repeat), same combinations.
Per cell and condition: quality (max/median temporal variance) and chance max (over cells), signal ÷ noise
(RF centre 3x3 blocks / far blocks > 300 µm), RF-centre time course -> half width, integration time
(timecourse_measures.py), centre area (smoothed map at the peak lag, connected region >= 50 % of the
centre value), surround index (mean of a ring 1.5-3x the centre radius, opposite sign = positive,
relative to the centre value), map correlation with the 60-min map, time-course correlation with 60 min.
Writes results/repeats_vs_rf.csv.
"""
import pickle, sys
from pathlib import Path
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "response_width_integration"))
from timecourse_measures import LAGS, integration, width_and_peak  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
STIM = "/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/32px_15Hz_20mins_shuffle_x4.h5"
RECS = {"14_08_2026": ("/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00", 6),
        "07_08_2026": ("/mnt/mea_nas/Laura/zebrafish_07_08_2026/Phase_00", 3)}
# (22/07 stim 14 was left out: its triggers are 33.35 ms apart = 30 Hz, 30 min in total, although named 15 Hz)
B, FS, PER_REP, N_REP = 8, 20000, 18000, 3
HALF = 25                                   # 50 x 50 blocks = 800 µm window
dev = "cuda"
rng = np.random.default_rng(0)


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
for rec, (root, idx) in RECS.items():
    o = pickle.load(open(f"{root}/overview", "rb"))
    r = o.stimulus_df[o.stimulus_df.stimulus_index == idx].iloc[0]
    trig = np.asarray(r.trigger_fr_relative).astype(np.int64)
    assert len(trig) - 1 == N_REP * PER_REP, len(trig)
    with h5py.File(STIM, "r") as f:
        S = f["Noise"][:]
    S = (S[:, ::B, ::B] == 255).astype(np.float32)
    nf, hb, wb = S.shape
    assert nf == PER_REP
    S_t = torch.from_numpy(S.reshape(nf, -1)).to(dev)
    S_t -= S_t.mean(0, keepdim=True)
    del S
    lag_samp = (LAGS * FS // 1000).astype(np.int64)
    rep_start = trig[np.arange(N_REP) * PER_REP]
    rep_end = trig[(np.arange(N_REP) + 1) * PER_REP]
    sp = pd.read_parquet(f"{root}/alldata.parquet")
    m = (sp.times >= r.begin_fr) & (sp.times < r.begin_fr + trig[-1])
    spikes = {c: g.times.values - r.begin_fr for c, g in sp[m].groupby("cell_index")}

    def sta_rep(t, k):
        """STA (sum, not divided) of the spikes t that lie in repeat k; frames looked up in that repeat."""
        t = t[(t >= rep_start[k]) & (t < rep_end[k])]
        if len(t) == 0:
            return np.zeros((100, hb, wb)), 0
        fi = np.searchsorted(trig, t[:, None] + lag_samp[None, :], side="right") - 1
        ok = (fi >= k * PER_REP) & (fi < (k + 1) * PER_REP)
        W = np.zeros((100, nf), np.float32)
        for li in range(100):
            W[li] = np.bincount(fi[ok[:, li], li] - k * PER_REP, minlength=nf)
        return (torch.from_numpy(W).to(dev) @ S_t).cpu().numpy().reshape(100, hb, wb), len(t)

    def shift_within(t):
        out = []
        for k in range(N_REP):
            tk = t[(t >= rep_start[k]) & (t < rep_end[k])] - rep_start[k]
            L = rep_end[k] - rep_start[k]
            out.append((tk + int(rng.uniform(0.1, 0.9) * L)) % L + rep_start[k])
        return np.sort(np.concatenate(out))

    CONDS = {"rep1": [0], "rep2": [1], "rep3": [2], "rep1+2": [0, 1], "rep1+2+3": [0, 1, 2]}
    for cell, t in spikes.items():
        if len(t) < 300:
            continue
        real = [sta_rep(t, k) for k in range(N_REP)]
        ts = shift_within(t)
        null = [sta_rep(ts, k) for k in range(N_REP)]
        comb = {}
        for name, ks in CONDS.items():
            n = sum(real[k][1] for k in ks)
            if n < 50:
                continue
            comb[name] = (sum(real[k][0] for k in ks) / n,
                          sum(null[k][0] for k in ks) / max(sum(null[k][1] for k in ks), 1), n)
        if "rep1+2+3" not in comb:
            continue
        ref = comb["rep1+2+3"][0]
        v = ndimage.gaussian_filter(ref, (2, 1, 1)).var(axis=0)
        cy, cx = np.unravel_index(v.argmax(), v.shape)        # one centre for all conditions (60 min)
        tc_ref = ref[:, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].mean(axis=(1, 2))
        jref = np.argmax(np.abs(tc_ref) * (LAGS <= 0))
        map_ref = window(ndimage.gaussian_filter(ref[jref], 1), cy, cx)
        yy, xx = np.mgrid[:hb, :wb]
        far = np.hypot(yy - cy, xx - cx) * 16 > 300
        wy, wx = np.mgrid[:2 * HALF, :2 * HALF]
        dist = np.hypot(wy - HALF, wx - HALF)
        for name, (sta, nsta, n) in comb.items():
            tc = sta[:, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2].mean(axis=(1, 2))
            j = np.argmax(np.abs(tc) * (LAGS <= 0))
            noise = ndimage.uniform_filter(sta[j], 3)[far].std()
            tcb = tc - np.median(tc[:30])
            w, pk, _ = width_and_peak(tcb)
            _, im, io_ = integration(tcb)
            mp = window(ndimage.gaussian_filter(sta[j], 1), cy, cx)
            sgn = np.sign(mp[HALF, HALF]) or 1.0
            lab, _ = ndimage.label(sgn * mp >= 0.5 * sgn * mp[HALF, HALF])
            centre = lab == lab[HALF, HALF]
            rad = np.sqrt(centre.sum() / np.pi)
            ring = (dist >= 1.5 * rad) & (dist <= 3 * rad) & ~centre
            surround = (-sgn * mp[ring].mean()) / abs(mp[HALF, HALF]) if ring.any() else np.nan
            rows.append(dict(recording=rec, cell=cell, cond=name, n_spikes=n, q=quality(sta), q_null=quality(nsta),
                             snr=abs(tc[j]) / noise, peak_ms=pk, width_ms=w, int_both_ms=im + io_,
                             centre_um2=centre.sum() * 256.0, surround_index=surround,
                             map_r_vs_60=np.corrcoef(mp.ravel(), map_ref.ravel())[0, 1],
                             tc_r_vs_60=np.corrcoef(tc, tc_ref)[0, 1],
                             polarity="ON" if tc[j] > 0 else "OFF"))
    print(rec, "done", flush=True)
    del S_t; torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df["cond_minutes"] = df.cond.map({"rep1": 20, "rep2": 20, "rep3": 20, "rep1+2": 40, "rep1+2+3": 60})
chance = df.groupby(["recording", "cond_minutes"]).q_null.max().rename("chance_max")
df = df.join(chance, on=["recording", "cond_minutes"])
df["rf"] = df.q > df.chance_max
df.to_csv(RESULTS / "repeats_vs_rf.csv", index=False)
print(df.groupby(["recording", "cond"]).agg(cells_rf=("rf", "sum"), chance=("chance_max", "first")).to_string())
