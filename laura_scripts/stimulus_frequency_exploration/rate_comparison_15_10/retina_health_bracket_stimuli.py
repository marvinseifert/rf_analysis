"""Retina health across the 4 noise rates (15/10/2026): compare responses to the same full-field
stimuli played before the noise (scf 6, csteps 7, chirp 8) and after it (scf 13, csteps 14,
chirp 15); the first presentations (0, 1, 2, 4) show the longer trend.

Per cell and stimulus presentation:
  response quality = variance over time of the repeat-averaged PSTH / mean over repeats of
  the variance over time of single repeats (Baden et al. 2016 quality index; 0..1, 1 = every
  repeat identical) - 50 ms bins.
  firing rate = mean spikes/s during the stimulus.
Writes retina_health_bracket_stimuli.csv (one row per cell x presentation).
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00")
PRESENTATIONS = {  # stimulus index: (name, when)
    0: ("scf", "start"), 1: ("csteps", "start"), 2: ("chirp", "start"), 4: ("csteps", "start2"),
    6: ("scf", "before noise"), 7: ("csteps", "before noise"), 8: ("chirp", "before noise"),
    13: ("scf", "after noise"), 14: ("csteps", "after noise"), 15: ("chirp", "after noise"),
}
BIN_S = 0.05
FS = 20000
OUT = Path(__file__).resolve().parents[1] / "results" / "retina_health_bracket_stimuli.csv"

o = pickle.load(open(ROOT / "overview", "rb"))
sd = o.stimulus_df
sp = pd.read_parquet(ROOT / "alldata.parquet")
ncell = int(sp.cell_index.max()) + 1
rows = []
for idx, (name, when) in PRESENTATIONS.items():
    r = sd[sd.stimulus_index == idx].iloc[0]
    trig = np.asarray(r.trigger_fr_relative) + r.begin_fr
    starts = trig[:-1][:: int(r.stimulus_repeat_logic)][: int(r.nr_repeats)]
    rep_len = int(np.median(np.diff(trig[:: int(r.stimulus_repeat_logic)])))
    nb = int(rep_len / FS / BIN_S)
    m = (sp.times >= starts[0]) & (sp.times < starts[-1] + rep_len)
    sub = sp[m]
    t_mid_h = (starts[0] + starts[-1] + rep_len) / 2 / FS / 3600
    for cell, g in sub.groupby("cell_index"):
        t = g.times.values
        psth = np.zeros((len(starts), nb))
        for k, s in enumerate(starts):
            rel = (t[(t >= s) & (t < s + rep_len)] - s) / FS
            psth[k] = np.histogram(rel, bins=nb, range=(0, nb * BIN_S))[0]
        denom = psth.var(axis=1).mean()
        qi = psth.mean(axis=0).var() / denom if denom > 0 else np.nan
        rows.append(dict(stim=idx, name=name, when=when, time_h=t_mid_h, cell=cell,
                         qi=qi, rate_hz=len(t) / (len(starts) * rep_len / FS)))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.groupby(["stim", "name", "when"]).agg(time_h=("time_h", "first"), cells=("cell", "size"),
      median_qi=("qi", "median"), median_rate=("rate_hz", "median")).round(2).to_string())
