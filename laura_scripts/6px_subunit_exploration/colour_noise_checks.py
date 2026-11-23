"""
Checks on the interleaved colour noise (32px shuffled) for planning 4px colour noise (23/11/2026).

Recordings:
  15/10/2026 P0, stim 5: 32px 15 Hz, colours white, 610, 500, 420 nm switching every 900 frames (1 min),
                         20-min sequence x 4.  RF maps: noise_analysis_colour_QI10.
  22/10/2026 P1, stim 4: 32px 10 Hz, colours 420(413), 500, 610 nm switching every 500 frames (50 s),
                         20-min sequence x 9.   RF maps: noise_analysis.

1. Single colour vs white (15/10): quality per cell, white vs each LED.
2. Chromatic offset: RF centre (centroid of the smoothed covariance map above 50 % of its peak) in each colour,
   for cells with quality >= --min-q in both colours; offset relative to 500 nm. A projector shift would be the
   same for every cell (mean offset != 0); a magnification difference would grow with distance from the
   stimulus centre (offset parallel to position). Maps are in stimulus coordinates (µm, v2 frame).
3. Adaptation after a colour switch: for good cells, (a) firing rate vs time since the colour switched, and
   (b) STA amplitude vs time since the switch: frame STA (8 x 8 px blocks = the shuffle step, so exact) collapsed
   with the cell's time course, projected onto the full-data map of that colour; spikes binned by time since
   block start. A transient = early bins differ from late bins.
   The colour of frame f within a repeat is ORDER[(f // BLOCK) % n] (blocks divide the 20-min sequence exactly);
   checked by correlating our per-colour STA map with the stored covariance map of the same colour.

Run from the repo root (~30-40 min, loads both stimulus files block-averaged):
    MPLBACKEND=Agg poetry run python laura_scripts/6px_subunit_exploration/colour_noise_checks.py
"""
import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from polarspike import Overview
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("--min-q", type=float, default=10.0)
parser.add_argument("--n-lags", type=int, default=8)
args = parser.parse_args()

OUT = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/colour_noise_checks")
OUT.mkdir(parents=True, exist_ok=True)
REC = {
    "15_10_2026": dict(root=Path("/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00"), stim_index=5,
                       nd="noise_analysis_colour_QI10",
                       stim=Path("/mnt/mea_nas/Laura/stimuli/stimuli_Oct_2026/32px_15Hz_20mins_shuffle_x4.h5"),
                       order=["white", "610", "500", "420"], block=900, fps=15),
    "22_10_2026": dict(root=Path("/mnt/mea_nas/Laura/zebrafish_22_10_2026/Phase_01"), stim_index=4,
                       nd="noise_analysis",
                       stim=Path("/mnt/mea_nas/Laura/stimuli/stimuli_Oct_2026/32px_10Hz_20mins_shuffle_x4.h5"),
                       order=["420", "500", "610"], block=500, fps=10),
}
COL = {"white": "0.3", "610": "tab:red", "500": "tab:green", "420": "tab:blue"}
B8 = 8  # shuffle step of the 32px x4 stimuli: every 8 x 8 block is constant
L = args.n_lags
summary = {}


def label(ch):
    return ch.split("_")[-1]


def centroid(m, x, y):
    sm = gaussian_filter(np.nan_to_num(m), 5)  # 10 µm on the 2 µm grid
    s = sm if sm.max() >= -sm.min() else -sm
    w = np.where(s >= 0.5 * s.max(), s, 0)
    return np.array([(w.sum(0) * x).sum(), (w.sum(1) * y).sum()]) / w.sum()


data = {}
for name, R in REC.items():
    ds = xr.open_dataset(R["root"] / R["nd"] / "noise_data_v2.nc")
    labs = [label(c) for c in ds.channel.values]
    q = np.nan_to_num(ds.quality.values)
    cells = ds.cell_index.values
    data[name] = dict(ds=ds, labs=labs, q=q, cells=cells)

# ------------------------------------------------------------------ 1. single colour vs white (15/10)
d = data["15_10_2026"]
iw = d["labs"].index("white")
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
res1 = {}
for j, lab in enumerate([l for l in d["labs"] if l != "white"]):
    ic = d["labs"].index(lab)
    ax = axs[j]
    qw, qc = d["q"][:, iw], d["q"][:, ic]
    either = (qw >= args.min_q) | (qc >= args.min_q)
    ax.scatter(qw[~either], qc[~either], s=5, color="0.8")
    ax.scatter(qw[either], qc[either], s=12, color=COL[lab])
    mx = max(qw.max(), qc.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.8)
    ax.axhline(args.min_q, color="0.6", lw=0.5); ax.axvline(args.min_q, color="0.6", lw=0.5)
    both = (qw >= args.min_q) & (qc >= args.min_q)
    ratio = np.median(qc[qw >= args.min_q] / qw[qw >= args.min_q]) if (qw >= args.min_q).any() else np.nan
    res1[lab] = dict(n_white=int((qw >= args.min_q).sum()), n_colour=int((qc >= args.min_q).sum()),
                     n_both=int(both.sum()), median_ratio_on_white_good=float(ratio))
    ax.set_xlabel("quality, white"); ax.set_ylabel(f"quality, {lab} nm")
    ax.set_title(f"{lab} nm vs white: good in white {res1[lab]['n_white']}, in {lab} {res1[lab]['n_colour']}, "
                 f"both {res1[lab]['n_both']}\nmedian {lab}/white quality (white-good cells) {ratio:.2f}",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
ax = axs[3]
for lab in [l for l in d["labs"] if l != "white"]:
    ic = d["labs"].index(lab)
    good = d["q"][:, iw] >= args.min_q
    r = d["q"][good, ic] / d["q"][good, iw]
    ax.hist(r, bins=np.linspace(0, 2, 21), histtype="step", lw=2, color=COL[lab], label=f"{lab} nm")
ax.axvline(1, color="k", ls="--", lw=0.8)
ax.set_xlabel("quality ratio colour / white\n(cells good in white)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"15/10/2026 P0, 32px interleaved colour noise (1-min blocks; each colour ~20 min): "
             f"how much is lost with one LED instead of white? (quality >= {args.min_q:.0f} lines)",
             fontsize=10, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "1_colour_vs_white_15_10_2026.png", dpi=150)
print("1.", json.dumps(res1, indent=1))
summary["1_colour_vs_white"] = res1

# ------------------------------------------------------------------ 2. chromatic offset
res2 = {}
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for r_i, (name, d) in enumerate(data.items()):
    ds, labs, q = d["ds"], d["labs"], d["q"]
    x, y = ds.x.values, ds.y.values
    ref = labs.index("500")
    others = [l for l in labs if l != "500"]
    cent = {}
    for i_c in range(len(d["cells"])):
        if q[i_c, ref] < args.min_q:
            continue
        cent[i_c] = {"500": centroid(ds.cm_most_important.isel(cell_index=i_c, channel=ref).values, x, y)}
        for lab in others:
            j = labs.index(lab)
            if q[i_c, j] >= args.min_q:
                cent[i_c][lab] = centroid(ds.cm_most_important.isel(cell_index=i_c, channel=j).values, x, y)
    for k, lab in enumerate(others[:3]):
        ax = axs[r_i, k]
        pairs = [(c["500"], c[lab]) for c in cent.values() if lab in c]
        if len(pairs) < 3:
            ax.set_title(f"{name}: {lab} vs 500 nm: only {len(pairs)} cells", fontsize=9, loc="left")
            ax.axis("off")
            continue
        P = np.array([p[0] for p in pairs]); D = np.array([p[1] - p[0] for p in pairs])
        mean, sem = D.mean(0), D.std(0, ddof=1) / np.sqrt(len(D))
        # magnification: offset = m * position (+ shift); least squares on both axes together
        A = np.r_[np.c_[P[:, 0], np.ones(len(P)), np.zeros(len(P))], np.c_[P[:, 1], np.zeros(len(P)), np.ones(len(P))]]
        coef, *_ = np.linalg.lstsq(A, np.r_[D[:, 0], D[:, 1]], rcond=None)
        resid = np.r_[D[:, 0], D[:, 1]] - A @ coef
        cov = np.linalg.inv(A.T @ A) * resid.var(ddof=3)
        mag_se = np.sqrt(cov[0, 0])
        res2[f"{name}|{lab}"] = dict(n=len(D), mean_shift_um=mean.tolist(), sem_um=sem.tolist(),
                                     median_abs_offset_um=float(np.median(np.hypot(*D.T))),
                                     magnification_diff=float(coef[0]), magnification_se=float(mag_se),
                                     edge_effect_um_at_600um=float(coef[0] * 600))
        q_ = ax.quiver(P[:, 0], P[:, 1], D[:, 0], D[:, 1], angles="xy", scale_units="xy", scale=0.1,
                       color=COL[lab], width=0.004)
        ax.scatter(P[:, 0], P[:, 1], s=6, color="k")
        ax.quiverkey(q_, 0.8, 1.03, 20, "20 µm offset (drawn x10)", labelpos="E", fontproperties={"size": 7})
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("x (µm, stimulus frame)"); ax.set_ylabel("y (µm)")
        ax.set_title(f"{name}: {lab} vs 500 nm RF centre, n = {len(D)}\nmean shift ({mean[0]:+.1f} ± {sem[0]:.1f}, "
                     f"{mean[1]:+.1f} ± {sem[1]:.1f}) µm; median |offset| {np.median(np.hypot(*D.T)):.1f} µm\n"
                     f"magnification diff {coef[0] * 100:+.2f} ± {mag_se * 100:.2f} % "
                     f"(= {coef[0] * 600:+.1f} µm at 600 µm from centre)", fontsize=8, loc="left")
    for k in range(len(others), 3):
        axs[r_i, k].axis("off")
fig.suptitle("Chromatic offset: arrows = RF centre in each colour minus RF centre at 500 nm (x10), at the cell's "
             "position. Same direction everywhere = projector shift; pointing outwards growing with distance = "
             "magnification difference.", fontsize=10, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "2_chromatic_offset.png", dpi=150)
print("2.", json.dumps(res2, indent=1))
summary["2_chromatic_offset"] = res2
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))

# ------------------------------------------------------------------ 3. adaptation after a colour switch
res3 = {}
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
for r_i, (name, R) in enumerate(REC.items()):
    d = data[name]
    ds, labs, q, cells = d["ds"], d["labs"], d["q"], d["cells"]
    rec = Overview.Recording.load(R["root"] / "overview")
    rec.dataframes["noise_analysis"] = rec.dataframes["spikes_df"].query("stimulus_index == @R['stim_index']")
    trig = np.asarray(rec.dataframes["stimulus_df"].query("stimulus_index == @R['stim_index']")
                      ["trigger_fr_relative"].to_numpy()[0])
    print(f"3. {name}: loading stimulus (8 x 8 block means) ...", flush=True)
    with h5py.File(R["stim"], "r") as f:
        n_seq, H, Wd = f["Noise"].shape
        S = np.empty((n_seq, H // B8, Wd // B8), np.int8)
        for s0 in range(0, n_seq, 1000):
            a = f["Noise"][s0:s0 + 1000, ::B8, ::B8]
            S[s0:s0 + 1000] = (a > 127).astype(np.int8) * 2 - 1
    Hb, Wb = S.shape[1:]
    Sf = S.reshape(n_seq, -1).astype(np.float32)
    del S
    n_col = len(R["order"])
    fi = np.arange(n_seq)
    colour_of = np.array(R["order"])[(fi // R["block"]) % n_col]
    t_in_block = (fi % R["block"]) / R["fps"]  # s since the colour switched
    edges = np.array([0, 5, 10, 20, 30, R["block"] / R["fps"] + 1e-6])
    edges = edges[edges <= R["block"] / R["fps"] + 1e-6]
    cen = 0.5 * (edges[1:] + edges[:-1])
    rate_curves = {lab: [] for lab in R["order"]}
    amp_curves = {lab: [] for lab in R["order"]}
    check = []
    for lab in R["order"]:
        j = labs.index(lab)
        good = np.where(q[:, j] >= max(args.min_q, 15))[0]
        good = good[np.argsort(-q[good, j])][:25]
        for i_c in good:
            cell = int(cells[i_c])
            sp = rec.get_spikes_triggered([{"stimulus_index": R["stim_index"], "cell_index": cell}],
                                          time="frames", pandas=False)
            t = np.sort(sp["times_relative"].to_numpy().astype(np.int64).ravel())
            fr = np.searchsorted(trig, t, side="right") - 1
            fr = fr[(fr >= 0) & (fr < len(trig) - 1)] % n_seq
            mine = fr[(colour_of[fr] == lab) & ((fr % R["block"]) >= L)]  # lags stay inside the block
            if len(mine) < 500:
                continue
            # rate vs time in block (spikes per frame, all repeats)
            n_rep = (len(trig) - 1) // n_seq
            frames_in_bin = np.histogram(t_in_block[colour_of == lab], edges)[0] * n_rep
            spk_in_bin = np.histogram(t_in_block[mine], edges)[0]
            rate = spk_in_bin / frames_in_bin * R["fps"]
            # STA from spike counts per frame on the whole 8 px grid; RF window around its own peak
            def sta_from(frames_sel):
                c = np.bincount(frames_sel, minlength=n_seq).astype(np.float32)
                return np.stack([np.roll(c, -lag) @ Sf for lag in range(L)]) / max(len(frames_sel), 1)
            sta_all = sta_from(mine)
            strength = (sta_all ** 2).max(0).reshape(Hb, Wb)
            cy, cx = np.unravel_index(np.argmax(gaussian_filter(strength, 1)), strength.shape)
            hw = 12  # 25 x 25 blocks = 400 µm
            box = np.zeros((Hb, Wb), bool)
            box[max(cy - hw, 0):cy + hw + 1, max(cx - hw, 0):cx + hw + 1] = True
            box = box.ravel()
            sta = sta_all[:, box]
            st = (sta ** 2).max(0)
            k = sta[:, st >= 0.5 * st.max()].mean(1)
            k /= np.linalg.norm(k)
            full = k @ sta
            amps = []
            for b0, b1 in zip(edges[:-1], edges[1:]):
                sel = mine[(t_in_block[mine] >= b0) & (t_in_block[mine] < b1)]
                if len(sel) < 50:
                    amps.append(np.nan); continue
                m = k @ sta_from(sel)[:, box]
                amps.append(float((m * full).sum() / (full * full).sum()))
            rate_curves[lab].append(rate / rate[-1])
            amp_curves[lab].append(np.array(amps) / amps[-1])
            # colour check: the same spikes assigned with blocks shifted by one (wrong colour)
            wrong = fr[(colour_of[(fr + R["block"]) % n_seq] == lab) & ((fr % R["block"]) >= L)]
            mw = k @ sta_from(wrong)[:, box]
            check.append((lab, float(np.abs(full).max() / (np.abs(mw).max() + 1e-9))))
    for lab in R["order"]:
        if not rate_curves[lab]:
            continue
        rc, ac = np.array(rate_curves[lab]), np.array(amp_curves[lab])
        axs[r_i, 0].errorbar(cen, np.nanmedian(rc, 0), yerr=np.nanstd(rc, 0) / np.sqrt(len(rc)), marker="o",
                             color=COL[lab], label=f"{lab} (n={len(rc)})", capsize=2)
        axs[r_i, 1].errorbar(cen, np.nanmedian(ac, 0), yerr=np.nanstd(ac, 0) / np.sqrt(len(ac)), marker="o",
                             color=COL[lab], label=f"{lab} (n={len(ac)})", capsize=2)
        res3[f"{name}|{lab}"] = dict(n=len(rc), bins_s=edges.tolist(), rate_rel=np.nanmedian(rc, 0).tolist(),
                                     sta_amp_rel=np.nanmedian(ac, 0).tolist())
    ratio = {lab: float(np.median([r for l_, r in check if l_ == lab])) for lab in R["order"]
             if any(l_ == lab for l_, _ in check)}
    res3[f"{name}|colour_assignment_check"] = ratio
    print(f"   colour check (map peak, right / shifted-by-one-block assignment): {ratio}")
    for c_i, ttl in enumerate(["firing rate", "STA amplitude"]):
        ax = axs[r_i, c_i]
        ax.axhline(1, color="k", ls="--", lw=0.8)
        ax.set_xlabel("time since the colour switched (s)")
        ax.set_ylabel(f"{ttl} relative to the last bin")
        ax.set_title(f"{name}: {ttl} vs time in block ({R['block'] / R['fps']:.0f}-s blocks; median over "
                     f"the best cells)", fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Adaptation after a colour switch: is the start of each colour block different? "
             "(1 = same as the end of the block)", fontsize=10, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "3_adaptation_after_switch.png", dpi=150)
print("3.", json.dumps(res3, indent=1))
summary["3_adaptation"] = res3
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print("saved", OUT)
