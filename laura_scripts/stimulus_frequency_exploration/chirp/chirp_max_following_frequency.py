"""Highest chirp frequency each cell can follow (15/10/2026), and how this relates to which
noise rate (5/10/15/20 Hz) gives it an RF.

Chirp (from extract_chirp_features.generate_chirp_data): full-field sine, 1 -> 30 Hz,
exponential sweep over 30 s, starts at trigger 1 of each repeat (3 s grey before), 3 repeats.
Per repeat the measured chirp duration (~30.017 s) is used to rescale time, so the phase stays
correct up to 30 Hz.

Following = spikes locked to the phase of the flicker. The chirp is cut into 15 pieces of 2 s
(each covers ~25 % frequency range). For each piece: phase-locking amplitude |sum exp(i*phase)|
over all spikes of the 3 repeats. Chance = same spikes circularly shifted by 5-25 s inside the
chirp (keeps bursts, breaks the link to the phase), 300 shifts; response delay 0-300 ms
searched per piece (same search for the shifts). A piece counts as followed when
the real value beats 99 % of the shifts. Highest followed frequency = top piece of the highest
run of >= 2 neighbouring followed pieces (single lucky pieces ignored).
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00")
CHIRPS = [2, 8, 15]                 # 8 is just before the four noise rates
F0, F1, DUR = 1.0, 30.0, 30.0
BETA = np.log(F1 / F0) / DUR
N_BANDS, N_SHIFT = 15, 300
EDGES_T = np.linspace(0, DUR, N_BANDS + 1)
EDGES_F = F0 * np.exp(BETA * EDGES_T)
BAND_F = np.sqrt(EDGES_F[:-1] * EDGES_F[1:])
FS = 20000
OUT = Path(__file__).resolve().parents[1] / "results" / "chirp_max_following_frequency.csv"
rng = np.random.default_rng(1)


def phase(tau):
    return 2 * np.pi * (F0 / BETA) * (np.exp(BETA * tau) - 1)


DELAYS = np.arange(0, 0.31, 0.01)   # response delay 0-300 ms


def locking(tau_list):
    """tau_list: chirp-time spike arrays (one per repeat). Returns |Z| per band, best delay.
    The stimulus phase is read at (spike time - delay); a delay error would make the phase
    drift inside high-frequency pieces and cancel real locking, so the best delay (0-300 ms)
    is used per piece. Null shifts get the same search, so chance is treated equally."""
    tau = np.concatenate(tau_list)
    b = np.clip(np.searchsorted(EDGES_T, tau, side="right") - 1, 0, N_BANDS - 1)
    z = np.exp(1j * phase(np.clip(tau[None, :] - DELAYS[:, None], 0, None)))   # (delay, spike)
    out = np.zeros((len(DELAYS), N_BANDS))
    for k in range(len(DELAYS)):
        out[k] = np.abs(np.bincount(b, weights=z[k].real, minlength=N_BANDS)
                        + 1j * np.bincount(b, weights=z[k].imag, minlength=N_BANDS))
    return out.max(axis=0)


def highest_followed(sig):
    best = np.nan
    for i in range(1, N_BANDS):
        if sig[i] and sig[i - 1]:
            best = EDGES_F[i + 1]           # top edge of the followed piece
    return best


o = pickle.load(open(ROOT / "overview", "rb"))
sd = o.stimulus_df
sp = pd.read_parquet(ROOT / "alldata.parquet")
rows = []
for idx in CHIRPS:
    r = sd[sd.stimulus_index == idx].iloc[0]
    trig = np.asarray(r.trigger_fr_relative) + r.begin_fr
    starts, ends = trig[1::2], trig[2::2]            # chirp start, next trigger
    m = (sp.times >= starts[0]) & (sp.times < ends[-1])
    for cell, g in sp[m].groupby("cell_index"):
        t = g.times.values
        reps = []
        for s, e in zip(starts, ends):
            tt = t[(t >= s) & (t < e)]
            reps.append((tt - s) / (e - s) * DUR)     # rescale to nominal 30 s
        n = sum(len(x) for x in reps)
        real = locking(reps)
        null = np.empty((N_SHIFT, N_BANDS))
        for k in range(N_SHIFT):
            null[k] = locking([(x + rng.uniform(5, 25)) % DUR for x in reps])
        p99 = np.percentile(null, 99, axis=0)
        sig = real > p99
        amp = real / (len(reps) * (DUR / N_BANDS))   # modulation, spikes/s
        row = dict(chirp=idx, cell=cell, n_spikes=n, n_followed=int(sig.sum()),
                   max_followed_hz=highest_followed(sig),
                   best_hz=BAND_F[np.argmax(real / np.maximum(np.median(null, axis=0), 1e-9))])
        row.update({f"amp_{f:.1f}": a for f, a in zip(BAND_F, amp)})
        row.update({f"sig_{f:.1f}": s for f, s in zip(BAND_F, sig)})
        rows.append(row)
    print("chirp", idx, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("band frequencies (Hz):", np.round(EDGES_F, 1))
for idx, g in df.groupby("chirp"):
    f = g.max_followed_hz
    print(f"chirp {idx}: cells {len(g)}, follow chirp {f.notna().sum()}, "
          f"median highest {f.median():.1f} Hz, follow to the end (30 Hz) {np.sum(f >= 29.9)}")
