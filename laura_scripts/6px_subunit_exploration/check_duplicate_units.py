"""
Are two "cells" with overlapping RFs really one neuron counted twice by spike sorting? (20/11/2026)

For each pair prints:
  * spike counts
  * fraction of the smaller cell's spikes with a partner spike within ±0.5 / ±1 ms
    (a duplicate = same spikes found twice -> most spikes shared)
  * cross-correlogram at 0 lag (±1 ms) / its flanks (±30-50 ms): ~1 = independent,
    > 1 = fire together more than chance (two neurons: one neuron can't fire twice within ~1 ms,
    so a unit split in two would show a dip here, < 1)
  * electrode with the biggest template amplitude, template correlation (same waveform?)
  * correlation of the STA time courses (sta_single_pixel)
Spike times: <phase>/alldata.parquet (cell_index, times in samples, 20 kHz); its cell_index is the
template index in the UNMERGED alldata.templates.hdf5 (both have 289 units for 14/08/2026).

Result 14/08/2026 (6px STAs): 150/173 and 172/151 are NOT duplicates (different electrodes,
template r ~0, 1-2 % shared spikes like unrelated pairs). 150/151 share electrode 40 and look alike
(template r 0.71) but fire together more than chance at 0 lag -> two neurons.

Run from the repo root:
    poetry run python laura_scripts/6px_subunit_exploration/check_duplicate_units.py \
        --pairs 150-173 172-151 150-151 150-252
"""

import argparse

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp

from loading.noise_data import load_noise_data  # v2 RF dataset, checked
from laura_scripts.Alignment_pipeline.alignment_recordings import NAS_LAURA, noise_data_v2_file

parser = argparse.ArgumentParser()
parser.add_argument("--recording", default="14_08_2026")
parser.add_argument("--channel", default="6px_15Hz_30mins_shuffle_x6")
parser.add_argument("--pairs", nargs="+", default=["150-173", "172-151", "150-172", "150-151", "150-252", "151-154"])
parser.add_argument("--sampling-rate", type=float, default=20000)
args = parser.parse_args()
FS = args.sampling_rate
phase = NAS_LAURA / f"zebrafish_{args.recording}" / "Phase_00"

spikes_df = pl.read_parquet(phase / "alldata.parquet")
with h5py.File(phase / "alldata.templates.hdf5", "r") as f:
    n_elec, n_t, n2 = f["temp_shape"][:]
    templates = sp.csc_matrix((f["temp_data"][:], (f["temp_x"][:], f["temp_y"][:])), shape=(n_elec * n_t, n2))
with load_noise_data(noise_data_v2_file(args.recording), lazy=True) as ds:
    sta = ds.sta_single_pixel.sel(channel=args.channel).load()


def spikes(c):
    return np.sort(spikes_df.filter(pl.col("cell_index") == c)["times"].to_numpy())


def template(c):
    return templates[:, c].toarray().reshape(n_elec, n_t)


def shared_fraction(a, b, window_samples):
    i = np.searchsorted(b, a)
    lo, hi = np.clip(i - 1, 0, len(b) - 1), np.clip(i, 0, len(b) - 1)
    return float((np.minimum(abs(a - b[lo]), abs(a - b[hi])) <= window_samples).mean())


def zero_lag_ratio(a, b):
    i0, i1 = np.searchsorted(b, a - 0.05 * FS), np.searchsorted(b, a + 0.05 * FS)
    lags_ms = np.concatenate([b[s:e] - t for t, s, e in zip(a, i0, i1)]) / FS * 1000
    h, _ = np.histogram(lags_ms, bins=np.arange(-50, 50.5, 0.5))
    centre = h[98:102].mean()  # -1 .. +1 ms
    flank = np.r_[h[:40], h[-40:]].mean()  # 30-50 ms either side
    return centre / flank if flank > 0 else np.nan


for pair in args.pairs:
    a, b = (int(v) for v in pair.split("-"))
    sa, sb = spikes(a), spikes(b)
    small, big = (sa, sb) if len(sa) < len(sb) else (sb, sa)
    ta, tb = template(a), template(b)
    ea, eb = np.abs(ta).max(1).argmax(), np.abs(tb).max(1).argmax()
    print(f"{a}-{b}: spikes {len(sa)}/{len(sb)} | shared ±0.5 ms {shared_fraction(small, big, 0.0005 * FS):.1%}, "
          f"±1 ms {shared_fraction(small, big, 0.001 * FS):.1%} | 0-lag / flank {zero_lag_ratio(sa, sb):.1f} | "
          f"electrode {ea}/{eb} | template r {np.corrcoef(ta.ravel(), tb.ravel())[0, 1]:.2f} | "
          f"STA r {np.corrcoef(sta.sel(cell_index=a).values, sta.sel(cell_index=b).values)[0, 1]:.2f}")
