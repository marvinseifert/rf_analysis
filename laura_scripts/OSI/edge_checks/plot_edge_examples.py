"""Example cells responding to the moving edges, 22/07/2026 (10 kHz recording), timed from the real triggers."""
import pickle
import h5py, hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/mnt/mea_nas/Laura/zebrafish_22_07_2026/Phase_00"
STIM = "/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/"
CONDS = [(7, "ON edge, 60 fps", "moving_bar_weighted_60fps_ON_800px_wide_800px_high_5px_thick_30.0px_ramp_1.0px_per_frame.h5"),
         (8, "OFF edge, 60 fps", "moving_edge_weighted_60fps_OFF_800px_wide_800px_high__5px_thick_30.0px_ramp_1.0px_per_frame.h5"),
         (6, "OFF edge, 13 fps file (played 12 fps)", "moving_edge_weighted_13fps_OFF_800px_wide_800px_high__5px_thick_30.0px_ramp_1.0px_per_frame.h5")]
CELLS = [148, 28, 232]
SHORT = {"left_to_right": "L→R", "right_to_left": "R→L", "bottom_to_top": "B→T", "top_to_bottom": "T→B",
         "bottom_left_to_top_right": "BL→TR", "bottom_right_to_top_left": "BR→TL",
         "top_left_to_bottom_right": "TL→BR", "top_right_to_bottom_left": "TR→BL"}
OUT = "/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/noise_rate_comparison/moving_edge_example_cells_22_07.png"

o = pickle.load(open(ROOT + "/overview", "rb")); sd = o.stimulus_df
sp = pd.read_parquet(ROOT + "/alldata.parquet")
fig, axes = plt.subplots(len(CELLS), len(CONDS), figsize=(16, 4.2 * len(CELLS)), squeeze=False)
for ci, (idx, title, fn) in enumerate(CONDS):
    h = h5py.File(STIM + fn, "r")
    st, cnt = h["Direction_Start_Frame"][()], h["Direction_Frame_Count"][()]
    names = [x.decode() for x in h["Directions"][()]]
    n_per = h["Noise"].shape[0]
    r = sd[sd.stimulus_index == idx].iloc[0]
    fs = r.sampling_freq
    trig = np.asarray(r.trigger_fr_relative) + r.begin_fr
    R = int(r.nr_repeats)
    for ri, cell in enumerate(CELLS):
        ax = axes[ri, ci]
        t = sp[(sp.cell_index == cell)].times.values
        y0, yt, yl = 0, [], []
        for k, nm in enumerate(names):
            dur = (trig[st[k] + cnt[k]] - trig[st[k]]) / fs
            allx = []
            for rep in range(R):
                a = trig[rep * n_per + st[k]]; b = trig[rep * n_per + st[k] + cnt[k]]
                x = (t[(t >= a) & (t < b)] - a) / fs
                ax.plot(x, np.full(len(x), y0 + rep), "|", color="0.15", ms=3, mew=0.7)
                allx.append(x)
            # PSTH (0.25 s) scaled into the band
            bins = np.arange(0, dur + 0.25, 0.25)
            hist = np.histogram(np.concatenate(allx), bins)[0] / R / 0.25
            if hist.max() > 0:
                ax.fill_between(bins[:-1], y0 - 0.5, y0 - 0.5 + hist / max(hist.max(), 1) * (R - 0.2) * 0.9,
                                step="post", color="#2171b5", alpha=0.25, lw=0)
            yt.append(y0 + (R - 1) / 2); yl.append(SHORT[nm])
            y0 += R + 1.5
            ax.axhline(y0 - 1.25, color="0.85", lw=0.5)
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8)
        ax.set_ylim(-1, y0 - 1); ax.invert_yaxis()
        ax.set_xlabel("time from start of this direction (s)")
        ax.spines[["top", "right"]].set_visible(False)
        if ri == 0:
            ax.set_title(f"{title}\n({R} repeats; ticks = spikes, shading = PSTH)", fontsize=10, loc="left")
        if ci == 0:
            ax.text(-0.18, 0.5, f"cell {cell}", transform=ax.transAxes, rotation=90, va="center", fontsize=11, fontweight="bold")
fig.suptitle("22/07/2026: example cells responding to the moving edges (5 px bar + 30 px ramps on grey; directions as in the "
             "stimulus file, row 0 = top; timed from real triggers)", fontsize=11, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(OUT)
