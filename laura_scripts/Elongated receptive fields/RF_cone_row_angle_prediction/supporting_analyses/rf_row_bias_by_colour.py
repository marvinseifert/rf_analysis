"""
Does the RF-vs-cone-row bias depend on a cell's colour preference or colour opponency? (23/11/2026)

Spectral responses: single chromatic flashes (scf: 660, 610, 560, 535, 500, 460, 420, 365 nm; 2 s on / 2 s off)
from fff_features_v2.csv (fff_feature_pipeline/extract_scf_features.py), recordings 14/08, 22/07, 14/05, 15/05.
Per wavelength: on/off response (Hz), signed tuning (ON - OFF), amplitude = max(ON, OFF).

  responsive    : max amplitude - min amplitude across wavelengths >= --min-response-hz (spontaneous firing
                  is the same for all wavelengths, so the range removes it)
  colour index  : (short - long) / (short + long), short = mean of (amplitude - min) at 365-460 nm, long = at
                  560-660 nm; +1 = only UV/blue, -1 = only green/red. Split into thirds within each recording
                  ("long-pref.", "mid", "short-pref."). NB the LEDs differ in power, so the index mixes cell
                  preference with LED strength; comparisons between cells are still fair.
  strict opponent: signed tuning (ON - OFF) averaged over the short and over the long wavelengths has opposite
                  signs, each at least 20 % of the cell's largest |signed tuning| (the existing scf_colour_opponent
                  flag marks 70-85 % of cells, too lenient for noisy flash responses).
RF/row data and tests as rf_row_bias_by_cell_type.py: angle = |RF long axis - own ONH-radial row|; across > along
score = mean -cos(2 x angle); one-sided p from RF axes shuffled within recordings; group difference p from labels
shuffled within recordings (2-sided); six 15° bins with the shuffle's median and 95 % range.

Run from the repo root (~2 min):
    MPLBACKEND=Agg poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/rf_row_bias_by_colour.py"
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--min-response-hz", type=float, default=3.0)
args = parser.parse_args()

FIG = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/rf_analysis/Elongated receptive fields/RF_cone_row_angle_prediction")
N_PERM = 5000
WL = [660, 610, 560, 535, 500, 460, 420, 365]
SHORT, LONG = [460, 420, 365], [660, 610, 560]
FINE_BINS = np.arange(0, 91, 15)
rng = np.random.default_rng(21)

df = pd.read_csv(FIG / "rf_vs_cone_row_angle_all_recordings_cells.csv")
df = df[(df.quality >= 10) & df.inside_displayed_stimulus.astype(bool) & ~df.near_edge.astype(bool)].copy()
df["angle"] = df.rf_vs_row_difference_deg
feat = []
for rec in df.recording.unique():
    f = Path(f"/mnt/mea_nas/Laura/zebrafish_{rec}/Phase_00/fff_analysis/fff_features_v2.csv")
    if not f.exists():
        continue
    cols = ["cell_index"] + [f"scf_{w}nm_amplitude_hz" for w in WL] + [f"scf_{w}nm_signed_tuning_hz" for w in WL]
    x = pd.read_csv(f, usecols=cols)
    x["recording"] = rec
    feat.append(x)
feat = pd.concat(feat)
amp = feat[[f"scf_{w}nm_amplitude_hz" for w in WL]].values
amp0 = amp - np.nanmin(amp, 1, keepdims=True)
feat["scf_range_hz"] = np.nanmax(amp, 1) - np.nanmin(amp, 1)
s = amp0[:, [WL.index(w) for w in SHORT]].mean(1)
l = amp0[:, [WL.index(w) for w in LONG]].mean(1)
with np.errstate(invalid="ignore", divide="ignore"):
    feat["colour_index"] = (s - l) / (s + l)
tun = feat[[f"scf_{w}nm_signed_tuning_hz" for w in WL]].values
ts = tun[:, [WL.index(w) for w in SHORT]].mean(1)
tl = tun[:, [WL.index(w) for w in LONG]].mean(1)
mx = np.nanmax(np.abs(tun), 1)
feat["strict_opponent"] = np.where((np.sign(ts) != np.sign(tl)) & (np.abs(ts) >= 0.2 * mx) & (np.abs(tl) >= 0.2 * mx),
                                   "opponent", "non-opponent")
d = df.merge(feat, on=["recording", "cell_index"], how="inner")
d = d[d.scf_range_hz >= args.min_response_hz].copy()
d["colour_group"] = d.groupby("recording").colour_index.transform(
    lambda x: pd.qcut(x.rank(method="first"), 3, labels=["long-pref.", "mid", "short-pref."]).astype(str))
print(f"{len(d)} responsive cells with RF/row data (range >= {args.min_response_hz} Hz) in {d.recording.nunique()} "
      f"recordings; strict opponent: {(d.strict_opponent == 'opponent').sum()} "
      f"({(d.strict_opponent == 'opponent').mean() * 100:.0f} %)")
print("colour index by recording (median):", d.groupby("recording").colour_index.median().round(2).to_dict())


def diff(a, b):
    x = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(x, 180 - x)


def score(a):
    return float(-np.cos(2 * np.deg2rad(a)).mean())


def bin_pct(a):
    return np.histogram(a, bins=FINE_BINS)[0] / len(a) * 100


def shuffle(sel):
    real = score(sel.angle)
    groups = [g for _, g in sel.groupby("recording")]
    null, nb = [], []
    for _ in range(N_PERM):
        a = np.concatenate([diff(rng.permutation(g.rf_axis_projected_deg.values), g.cone_row_axis_projected_deg.values)
                            for g in groups])
        null.append(score(a)); nb.append(bin_pct(a))
    nb = np.array(nb)
    rb = bin_pct(sel.angle.values)
    return real, float(np.mean(np.array(null) >= real)), dict(real=rb, med=np.median(nb, 0), lo=np.percentile(nb, 2.5, 0),
                                                              hi=np.percentile(nb, 97.5, 0), p_hi=(nb >= rb).mean(0))


def group_diff(sel, col, levels):
    sel = sel[sel[col].isin(levels)]
    def stat(lab):
        return np.ptp([score(sel.angle[lab == lv]) for lv in levels if (lab == lv).sum()])
    real = stat(sel[col].values)
    null = [stat(sel.groupby("recording")[col].transform(lambda x: rng.permutation(x.values)).values)
            for _ in range(1000)]
    return float(np.mean(np.array(null) >= real))


GROUPINGS = [("colour_group", "Colour preference (scf)", ["long-pref.", "mid", "short-pref."]),
             ("strict_opponent", "Colour opponency (strict)", ["non-opponent", "opponent"])]
rows, fine = [], {}
s_all, p_all, fine["all"] = shuffle(d)
print(f"all responsive cells (n = {len(d)}): score {s_all:+.3f}, p = {p_all:.4f}; 60-75° bin "
      f"{fine['all']['real'][4]:.1f} % vs shuffle {fine['all']['med'][4]:.1f} % (p = {fine['all']['p_hi'][4]:.3f})")
for col, title, levels in GROUPINGS:
    pdiff = group_diff(d, col, levels)
    for lv in levels:
        g = d[d[col] == lv]
        if len(g) < 5:
            continue
        sc, p, fine[(title, lv)] = shuffle(g)
        a = g.angle.values
        rows.append(dict(grouping=title, group=lv, n=len(g), score=sc, p=p, along=(a < 15).mean() * 100,
                         bin_60_75=fine[(title, lv)]["real"][4], bin_60_75_chance=fine[(title, lv)]["med"][4],
                         p_60_75=fine[(title, lv)]["p_hi"][4], across=(a >= 75).mean() * 100, p_difference=pdiff,
                         median_colour_index=g.colour_index.median()))
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
res.to_csv(FIG / "rf_row_bias_by_colour.csv", index=False)

# ------------------------------------------------------------------ figure
fig, axs = plt.subplots(2, 5, figsize=(23, 9))
ax = axs[0, 0]
for rec, g in d.groupby("recording"):
    ax.hist(g.colour_index, bins=np.linspace(-1, 1, 21), histtype="step", lw=1.5, label=f"{rec} (n = {len(g)})")
ax.set_xlabel("colour index (+1 = UV/blue only, -1 = green/red only)")
ax.set_ylabel("cells")
ax.legend(fontsize=7, frameon=False)
ax.set_title("Colour preference of responsive cells\n(depends on LED powers too)", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax = axs[1, 0]
xc = FINE_BINS[:-1] + 7.5
fb = fine["all"]
ax.fill_between(xc, fb["lo"], fb["hi"], color="0.85", label="shuffle 95 % range")
ax.plot(xc, fb["med"], "k--", lw=1)
ax.plot(xc, fb["real"], "o-", color="#1e6fd9", lw=2, label=f"all responsive (n = {len(d)})")
ax.set_xticks(FINE_BINS); ax.set_xlim(0, 90); ax.set_ylim(0, 35)
ax.set_xlabel("RF vs own row (°)"); ax.set_ylabel("% of cells")
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.set_title(f"All: score {s_all:+.2f}, p = {p_all:.3f}", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
for j, (col, title, levels) in enumerate(GROUPINGS):
    r = res[res.grouping == title]
    ax = axs[0, 1 + 2 * j]
    x = np.arange(len(r))
    ax.bar(x, r.score, color="#1e6fd9", edgecolor="k")
    for xi, (_, rr) in zip(x, r.iterrows()):
        ax.text(xi, max(rr.score, 0) + 0.01, f"p = {rr.p:.3f}\nn = {rr.n}", ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, r.group); ax.set_ylim(-0.25, 0.5)
    ax.set_ylabel("across > along score")
    ax.set_title(f"{title}\ngroups differ: p = {r.p_difference.iloc[0]:.3f}", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axs[0, 2 + 2 * j]
    ax.bar(x, r.bin_60_75, color="#3bb273", edgecolor="k", label="observed")
    ax.scatter(x, r.bin_60_75_chance, marker="_", s=900, color="k", lw=2, zorder=3, label="shuffle chance")
    for xi, (_, rr) in zip(x, r.iterrows()):
        ax.text(xi, rr.bin_60_75 + 0.8, f"p = {rr.p_60_75:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, r.group); ax.set_ylim(0, 40)
    ax.set_ylabel("% of cells at 60-75° from own row")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("The 60-75° excess per group", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1, 1 + 2 * j]
    colsr = plt.cm.viridis(np.linspace(0.15, 0.85, len(levels)))
    for lv, cc in zip(levels, colsr):
        if (title, lv) not in fine:
            continue
        fb = fine[(title, lv)]
        ax.fill_between(xc, fb["lo"], fb["hi"], color=cc, alpha=0.12)
        ax.plot(xc, fb["real"], "o-", color=cc, lw=1.8, label=lv)
    ax.set_xticks(FINE_BINS); ax.set_xlim(0, 90); ax.set_ylim(0, 40)
    ax.set_xlabel("RF vs own row (°), 15° bins")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"{title}: fine bins", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    axs[1, 2 + 2 * j].axis("off")
fig.suptitle(f"Does the RF-vs-cone-row bias depend on colour preference or opponency? Single chromatic flashes "
             f"(4 recordings), responsive cells only (response range >= {args.min_response_hz:g} Hz).",
             fontsize=11, x=0.01, ha="left")
fig.tight_layout()
for fmt in ("png", "pdf"):
    fig.savefig(FIG / f"rf_row_bias_by_colour.{fmt}", dpi=150 if fmt == "png" else None, bbox_inches="tight")
print("saved", FIG / "rf_row_bias_by_colour.png")
