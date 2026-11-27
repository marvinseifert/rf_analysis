"""
Chromatic RFs and spatial offsets: per-cell receptive fields and STA kernels of interleaved
colour noise, one column per cell (rewritten 25/11/2026; the original script was lost, its
figures are in Box .../Figures/RF_colour_offsets.pptx).

Rows per cell (top to bottom):
  one row per colour channel : covariance map (cm_most_important, red = positive) with the RF
                               centre outline (polar outline from the RF pipeline, drawn around the
                               channel's cutout centre "+") if quality >= --min-q
  overlay                    : all centre outlines + cutout centres
  RGB RMS / RGB |cm|         : 610 nm -> red, 500 nm -> green, 420 nm -> blue, each scaled to its
                               own max (channels with quality < --min-q and white are left out)
  kernels                    : sta_single_pixel of every channel (dashed if quality < --min-q)
optional:
  --sta-frames               : spatial STA frames at --sta-lags (ms before the spike) per colour,
                               fraction of spikes on bright pixels minus 0.5 (one grey scale per cell,
                               so colours can be compared). Reads kernel.npy (memory-mapped).
  --fff                      : PSTH of every scf and csteps presentation in the recording (one row
                               each, same y scale within a cell; scf background = LED colours, csteps
                               background = contrast)
Column titles: cell, moving-bar DSI (peak vs opposite) and OSI from the peak PSTH rate.

Maps are drawn as projected on the retina by default (--frame projected; same layout as the
lost October figures), or in v2 stimulus coordinates (--frame stimulus); both +y down, µm.

Cells are split into groups (GROUPS below; a cell is only shown in the first group it fits),
sorted by summed quality, --per-page cells per figure.

Run from the rf_analysis repo root, e.g.
    MPLBACKEND=Agg poetry run python laura_scripts/chromatic_RFs/spatial_offsets/plot_chromatic_rfs.py \
        --recording 22_10_2026 --sta-frames --fff
    ... --recording 15_10_2026 --cells 14 272
Runtime: ~1 min to open the data, then ~10-30 s per cell (longer with --sta-frames).
"""
import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO, _REPO / "laura_scripts" / "Seaborn" / "fff_feature_pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from polarspike import Overview  # noqa: E402

# ------------------------------------------------------------------ recordings
# groups: (description, {channel label: min quality}); "all" = every channel, "any" = at least one. A cell must have
# quality > the value in each listed channel.
RECORDINGS = {
    "15_10_2026": dict(
        root=Path("/mnt/mea_nas/Laura/zebrafish_15_10_2026/Phase_00"),
        analysis="noise_analysis_colour_QI10",
        moving_bar_index=3,
        groups=[
            ("QI > 10 in all channels", {"all": 10}),
            ("QI > 20 in 500 nm and 420 nm", {"500": 20, "420": 20}),
        ],
    ),
    "22_10_2026": dict(
        root=Path("/mnt/mea_nas/Laura/zebrafish_22_10_2026/Phase_01"),
        analysis="noise_analysis",
        moving_bar_index=3,
        groups=[
            ("QI > 10 in all channels", {"all": 10}),
            ("QI > 20 in 500 nm and 420 nm", {"500": 20, "420": 20}),
        ],
    ),
}
CH_COLOURS = {"white": "black", "610": "red", "500": "green", "420": "blue"}
RGB = ["610", "500", "420"]
SCF_COLOURS = ("#fd1717", "#000000", "#fe7c7c", "#000000", "#fafe7c", "#000000", "#dffc7e", "#000000",
               "#8afe7c", "#000000", "#7cfcfe", "#000000", "#7c86fe", "#000000", "#fe7cfe", "#000000")
SCF_LABELS = ("660", "610", "560", "535", "500", "460", "420", "365")  # every 4 s (2 s on, 2 s off)
CSTEPS_CONTRASTS = (100, -100, 90, -90, 80, -80, 70, -70, 60, -60, 50, -50, 40, -40, 30, -30, 20, -20, 10, -10)
FFF_DURATION_S = {"scf": 32.0, "csteps": 40.0}

parser = argparse.ArgumentParser()
parser.add_argument("--recording", default="22_10_2026", choices=list(RECORDINGS))
parser.add_argument("--cells", type=int, nargs="*", default=None, help="plot these cells (skips the groups)")
parser.add_argument("--any-q", type=float, default=None,
                    help="instead of GROUPS: group 1 = quality > this in all channels, group 2 = every "
                         "other cell with quality > this in at least one channel")
parser.add_argument("--split-opponent", action="store_true",
                    help="split the selected cells into spectrally opponent / not opponent / one colour only "
                         "(see spectral_class); writes spectral_opponency_<recording>.csv")
parser.add_argument("--opp-window-ms", type=float, default=400, help="main lobe searched in the last N ms")
parser.add_argument("--opp-max-dt-ms", type=float, default=100, help="opposite lobes at most this far apart")
parser.add_argument("--opp-noise-sd", type=float, default=4, help="lobe must be > this x baseline SD")
parser.add_argument("--min-q", type=float, default=10.0, help="quality needed for outlines / RGB")
parser.add_argument("--crop-um", type=float, default=700.0)
parser.add_argument("--per-page", type=int, default=6)
parser.add_argument("--max-pages", type=int, default=None, help="per group")
parser.add_argument("--sta-frames", action="store_true")
parser.add_argument("--sta-lags", type=float, nargs="*", default=[-400, -300, -200, -150, -100, -50])
parser.add_argument("--sta-window-um", type=float, default=400.0)
parser.add_argument("--fff", action="store_true")
parser.add_argument("--frame", default="projected", choices=["stimulus", "projected"],
                    help="stimulus: v2 stimulus coordinates; projected: as projected on the retina "
                         "(alignment_geometry, rule changed 8 Oct 2026); both drawn +y down")
parser.add_argument("--fff-bin-s", type=float, default=0.1)
parser.add_argument("--out", type=Path,
                    default=Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/chromatic_RFs_spatial_offset"))
parser.add_argument("--dpi", type=int, default=130)
args = parser.parse_args()

R = RECORDINGS[args.recording]
args.out.mkdir(parents=True, exist_ok=True)
analysis_dir = R["root"] / R["analysis"]


if args.frame == "projected":
    from laura_scripts.Alignment_pipeline.alignment_geometry import stimulus_to_projection_for_recording

    M, frame_desc = stimulus_to_projection_for_recording(args.recording)
    M = np.asarray(M)[:2, :2]
    print("projected frame:", frame_desc)
else:
    M, frame_desc = np.eye(2), "stimulus coordinates"
SWAP = M[0, 1] != 0
SX, SY = (M[0, 1], M[1, 0]) if SWAP else (M[0, 0], M[1, 1])


def tf_img(a):
    """(y, x[, c]) image in stimulus coordinates -> frame image, rows = ascending frame y."""
    if SWAP:
        a = np.swapaxes(a, 0, 1)
    if SY < 0:
        a = a[::-1]
    if SX < 0:
        a = a[:, ::-1]
    return a


def tf_pt(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return M[0, 0] * x + M[0, 1] * y, M[1, 0] * x + M[1, 1] * y


def tf_ext(ext):
    """[x0, x1, y1, y0] (origin upper) in stimulus coordinates -> frame extent."""
    px, py = tf_pt([ext[0], ext[1], ext[0], ext[1]], [ext[2], ext[2], ext[3], ext[3]])
    return [px.min(), px.max(), py.max(), py.min()]


def label(ch):
    return str(ch).split("_")[-1]


# ------------------------------------------------------------------ data
ds = xr.open_dataset(analysis_dir / "noise_data_v2.nc")
assert ds.attrs.get("rf_map_frame") == "stimulus_v2", "needs a v2 noise_data file"
channels = [str(c) for c in ds.channel.values]
labs = [label(c) for c in channels]
quality = pd.DataFrame(np.nan_to_num(ds.quality.values), index=ds.cell_index.values, columns=labs)
xs, ys = ds.x.values, ds.y.values
dx = float(ds.attrs.get("stimulus_pixel_size_um", np.median(np.diff(xs))))
t_ms = ds.time_max.values.astype(float)
print(f"{args.recording}: {len(quality)} cells, channels {labs}")

rec_cfg = json.loads((analysis_dir / "recording_config.json").read_text())
stim_index = None
kernel_dirs, spike_counts = {}, {}
for ch in channels:
    folder = re.split(r"[\\/]", rec_cfg["channel_configs"][ch]["root_path"])[-1]
    kernel_dirs[ch] = R["root"] / folder
    stim_index = int(folder.rsplit("_idx_", 1)[1])
    sc = kernel_dirs[ch] / "spike_counts.npy"
    spike_counts[ch] = dict(np.load(sc)) if sc.exists() else {}

recording = Overview.Recording.load(R["root"] / "overview")
stim_df = recording.dataframes["stimulus_df"]

# moving-bar DSI / OSI (cached)
mb_cache = args.out / f"moving_bar_features_{args.recording}.csv"
if mb_cache.exists():
    mb = pd.read_csv(mb_cache).set_index("cell_index")
else:
    try:
        from extract_moving_bar_features import calculate_moving_bar_features

        mb = calculate_moving_bar_features(R["root"] / "overview", R["moving_bar_index"])
        mb.to_csv(mb_cache, index=False)
        mb = mb.set_index("cell_index")
    except Exception as e:  # moving bar missing or not analysable: titles without DSI/OSI
        print("moving bar features failed:", repr(e))
        mb = pd.DataFrame()

fff_stims = []
if args.fff:
    for _, row in stim_df.iterrows():
        name = str(row["stimulus_name"]).strip()
        if name in FFF_DURATION_S:
            fff_stims.append((name, int(row["stimulus_index"]), int(row["nr_repeats"])))
    print("fff presentations:", fff_stims)


# ------------------------------------------------------------------ spectral opponency
def spectral_class(cell):
    """
    Opponency from the STA kernels (sta_single_pixel) of the LED channels with quality >= --min-q
    (white is left out). Main lobe of a channel = largest deflection from 0.5 within the last
    --opp-window-ms before the spike; it counts only if > --opp-noise-sd x the SD of the kernel
    800-500 ms before the spike. Opponent = two counted lobes of opposite sign whose peaks are at most
    --opp-max-dt-ms apart. Returns (class, description).
    """
    t = t_ms
    win, base = (t >= -args.opp_window_ms) & (t <= 0), (t >= -800) & (t <= -500)
    lobes = []
    for lab, ch in zip(labs, channels):
        if lab == "white" or quality.loc[cell, lab] < args.min_q:
            continue
        k = ds.sta_single_pixel.sel(cell_index=cell, channel=ch).values - 0.5
        if not np.isfinite(k).all():
            continue
        m = np.argmax(np.abs(k[win]))
        amp, lat = k[win][m], t[win][m]
        if abs(amp) > args.opp_noise_sd * np.std(k[base]):
            lobes.append((lab, amp, lat))
    desc = ", ".join(f"{l} {'ON' if a > 0 else 'OFF'} {lat:.0f} ms" for l, a, lat in lobes)
    if len(lobes) < 2:
        return "one colour", desc
    for i in range(len(lobes)):
        for j in range(i + 1, len(lobes)):
            if np.sign(lobes[i][1]) != np.sign(lobes[j][1]) and abs(lobes[i][2] - lobes[j][2]) <= args.opp_max_dt_ms:
                return "opponent", desc
    return "not opponent", desc


# ------------------------------------------------------------------ cell selection
SPECTRAL = {}


def select_groups():
    if args.cells:
        return [("selected cells", list(args.cells))]
    out, used = [], set()
    groups = R["groups"]
    if args.any_q is not None:
        groups = [(f"QI > {args.any_q:g} in all channels", {"all": args.any_q}),
                  (f"QI > {args.any_q:g} in at least one channel", {"any": args.any_q})]
    for desc, spec in groups:
        cols = labs if ("all" in spec or "any" in spec) else [c for c in spec if c in labs]
        thr = {c: spec.get("all", spec.get("any", spec.get(c))) for c in cols}
        test = np.any if "any" in spec else np.all
        ok = test([quality[c] > thr[c] for c in cols], axis=0)
        cells = [c for c in quality.index[ok] if c not in used]
        cells = sorted(cells, key=lambda c: -quality.loc[c, cols].sum())
        used.update(cells)
        out.append((desc, cells))
        print(f"{desc}: {len(cells)} cells")
    if args.split_opponent:
        cells = [c for _, cs in out for c in cs]
        cls = {c: spectral_class(c) for c in cells}
        pd.DataFrame([dict(cell_index=c, spectral_class=v[0], lobes=v[1],
                           **{f"quality_{l}": quality.loc[c, l] for l in labs}) for c, v in cls.items()]
                     ).to_csv(args.out / f"spectral_opponency_{args.recording}.csv", index=False)
        sel = "; ".join(d for d, _ in out)
        out = []
        for name in ("opponent", "not opponent", "one colour"):
            cs = sorted([c for c in cells if cls[c][0] == name], key=lambda c: -quality.loc[c].sum())
            desc = {"opponent": "spectrally opponent STAs", "not opponent": "not spectrally opponent",
                    "one colour": "only one colour with a clear STA (opponency not testable)"}[name]
            out.append((f"{desc} [{sel}]", cs))
            print(f"{name}: {len(cs)} cells")
        SPECTRAL.update(cls)
    return out


# ------------------------------------------------------------------ per-cell helpers
def cutout_centre(m):
    """Centre of the cutout (non-NaN area) in µm: where the pipeline's polar outline is centred."""
    ok = np.isfinite(m)
    rows, cols = np.where(ok.any(1))[0], np.where(ok.any(0))[0]
    if len(rows) == 0:
        return None
    return xs[cols[0] + len(cols) // 2], ys[rows[0] + len(rows) // 2]


def outline_xy(cell, ch, centre):
    r = ds.center_outline_um.sel(cell_index=cell, channel=ch).values
    if centre is None or not np.any(r > 0):
        return None
    th = np.deg2rad(ds.degree.values)
    x, y = centre[0] + r * np.cos(th), centre[1] + r * np.sin(th)  # stimulus frame, +y down
    return tf_pt(np.append(x, x[0]), np.append(y, y[0]))


def crop_slices(cx, cy, half):
    xi = np.where((xs >= cx - half) & (xs <= cx + half))[0]
    yi = np.where((ys >= cy - half) & (ys <= cy + half))[0]
    return slice(yi[0], yi[-1] + 1), slice(xi[0], xi[-1] + 1)


def rgb_image(maps, qs, sl):
    img = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start, 3))
    for k, lab in enumerate(RGB):
        if lab not in maps or qs[lab] < args.min_q:
            continue
        v = maps[lab][sl]
        if not np.isfinite(v).any():
            continue
        base, top = np.nanmedian(maps[lab]), np.nanmax(v)
        img[..., k] = np.nan_to_num(np.clip((v - base) / (top - base + 1e-12), 0, 1))
    return img


def sta_frames(cell, ch, centre):
    """(n_lags, h, w) STA frames around centre (µm), as fraction bright - 0.5; None if missing."""
    f = kernel_dirs[ch] / f"cell_{cell}" / "kernel.npy"
    if not f.exists() or centre is None:
        return None
    k = np.load(f, mmap_mode="r")
    T, H, W = k.shape
    half = int(round(args.sta_window_um / 2 / dx))
    cx, cy = int(round(centre[0] / dx + W / 2)), int(round(centre[1] / dx + H / 2))
    y0, x0 = cy - half, cx - half
    out = np.full((len(args.sta_lags), 2 * half, 2 * half), np.nan)
    ti = [int(np.argmin(np.abs(t_ms - lag))) for lag in args.sta_lags]
    sy0, sy1, sx0, sx1 = max(0, y0), min(H, y0 + 2 * half), max(0, x0), min(W, x0 + 2 * half)
    if sy1 <= sy0 or sx1 <= sx0:
        return None
    win = np.asarray(k[ti, sy0:sy1, sx0:sx1], dtype=float)
    n = spike_counts[ch].get(cell)
    if n:
        win = win / n - 0.5
    else:  # no spike count: z-score against the earliest bins
        base = np.asarray(k[:10, sy0:sy1, sx0:sx1], dtype=float)
        win = (win - base.mean()) / (base.std() + 1e-12)
    out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = win
    return out


def fff_psth(cell, name, idx, n_rep):
    d = recording.get_spikes_triggered([{"stimulus_index": idx, "cell_index": int(cell)}])
    d = d.to_pandas() if hasattr(d, "to_pandas") else d
    dur = FFF_DURATION_S[name]
    bins = np.arange(0, dur + args.fff_bin_s / 2, args.fff_bin_s)
    t = d["times_triggered"].to_numpy(float) if len(d) else np.array([])
    counts, _ = np.histogram(t[(t >= 0) & (t <= dur)], bins=bins)
    return bins[:-1] + args.fff_bin_s / 2, counts / (args.fff_bin_s * max(n_rep, 1))


def fff_background(ax, name):
    if name == "scf":
        for i, c in enumerate(SCF_COLOURS):
            ax.axvspan(2 * i, 2 * i + 2, color=c, alpha=0.18 if c != "#000000" else 0.06, lw=0)
    else:
        for i, c in enumerate(CSTEPS_CONTRASTS):
            ax.axvspan(2 * i, 2 * i + 2, color=str(0.5 * (1 + c / 100)), alpha=0.35, lw=0)


# ------------------------------------------------------------------ figure
def style(ax, small=True):
    ax.tick_params(labelsize=6 if small else 8)
    ax.set_xlabel("x (µm)", fontsize=6, labelpad=1)
    ax.set_ylabel("y (µm)", fontsize=6, labelpad=1)


def plot_page(cells, title, fname):
    n_ch = len(channels)
    rows = [("map", lab) for lab in labs] + [("overlay", None), ("rgb_rms", None), ("rgb_cm", None),
                                             ("kernels", None)]
    if args.sta_frames:
        rows += [("sta", lab) for lab in labs]
    if args.fff:
        rows += [("fff", s) for s in fff_stims]
    heights = [1.0 if k not in ("sta", "fff") else (0.55 if k == "sta" else 0.5) for k, _ in rows]
    col_w, row_h = 2.9, 2.2
    fig = plt.figure(figsize=(1.15 + col_w * len(cells), 1.75 + row_h * sum(heights)))
    gs = fig.add_gridspec(len(rows), len(cells), height_ratios=heights, hspace=0.55, wspace=0.42,
                          left=1.15 / fig.get_figwidth(), right=0.99,
                          top=1 - 1.55 / fig.get_figheight(), bottom=0.35 / fig.get_figheight())

    for j, cell in enumerate(cells):
        cell = int(cell)
        sel = ds.sel(cell_index=cell)
        cm = {lab: sel.cm_most_important.sel(channel=ch).values for lab, ch in zip(labs, channels)}
        rms = {lab: sel.rms.sel(channel=ch).values for lab, ch in zip(labs, channels)}
        qs = quality.loc[cell].to_dict()
        centres = {lab: cutout_centre(cm[lab]) for lab in labs}
        good = [centres[lab] for lab in labs if centres[lab] is not None and qs[lab] >= args.min_q]
        anyc = good or [c for c in centres.values() if c is not None]
        if not anyc:
            print(f"cell {cell}: no maps, skipped")
            continue
        cx, cy = np.mean([c[0] for c in anyc]), np.mean([c[1] for c in anyc])
        sl = crop_slices(cx, cy, args.crop_um / 2)
        ext = [xs[sl[1]][0] - dx / 2, xs[sl[1]][-1] + dx / 2, ys[sl[0]][-1] + dx / 2, ys[sl[0]][0] - dx / 2]
        ext = tf_ext(ext)
        outlines = {lab: outline_xy(cell, ch, centres[lab]) if qs[lab] >= args.min_q else None
                    for lab, ch in zip(labs, channels)}

        title_txt = f"Cell {cell}"
        if cell in getattr(mb, "index", []):
            title_txt += f"\nDSI {mb.loc[cell, 'moving_bar_DSI']:.2f}, OSI {mb.loc[cell, 'moving_bar_OSI']:.2f}"
        if cell in SPECTRAL:
            title_txt += f"\n{SPECTRAL[cell][1] or 'no clear lobe'}"
        sta = None
        if args.sta_frames:
            sta = {lab: sta_frames(cell, ch, centres[lab] or (cx, cy)) for lab, ch in zip(labs, channels)}
            vmax_sta = np.nanmax([np.nanmax(np.abs(v)) for v in sta.values() if v is not None] or [1])
        fff = [fff_psth(cell, *s) for s in fff_stims] if args.fff else []
        fff_max = max([p.max() for _, p in fff] + [1]) * 1.05

        fff_i = 0
        for i, (kind, what) in enumerate(rows):
            if kind == "sta":
                sub = gs[i, j].subgridspec(1, len(args.sta_lags), wspace=0.05)
                frames = sta[what] if sta else None
                for li, lag in enumerate(args.sta_lags):
                    ax = fig.add_subplot(sub[0, li])
                    ax.set_xticks([]); ax.set_yticks([])
                    if frames is not None:
                        ax.imshow(tf_img(frames[li]), cmap="gray", vmin=-vmax_sta, vmax=vmax_sta)
                    for s in ax.spines.values():
                        s.set_color(CH_COLOURS.get(what, "0.5"))
                    if li == 0 and i == rows.index(("sta", labs[0])):
                        ax.set_title(f"{lag:.0f} ms", fontsize=5, pad=1)
                    elif i == rows.index(("sta", labs[0])):
                        ax.set_title(f"{lag:.0f}", fontsize=5, pad=1)
                    if li == 0 and j == 0:
                        ax.set_ylabel(f"STA {what}", fontsize=8, fontweight="bold")
                continue
            ax = fig.add_subplot(gs[i, j])
            if kind == "map":
                m = cm[what][sl]
                if np.isfinite(m).any():
                    v = np.nanmax(np.abs(m))
                    cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("white")
                    ax.imshow(tf_img(m), extent=ext, cmap=cmap, vmin=-v, vmax=v, interpolation="nearest")
                    if outlines[what] is not None:
                        ax.plot(*outlines[what], color=CH_COLOURS.get(what, "k"), lw=1.3)
                        ax.plot(*tf_pt(*centres[what]), "+", color=CH_COLOURS.get(what, "k"), ms=6)
                else:
                    ax.text(0.5, 0.5, "no map", transform=ax.transAxes, ha="center", fontsize=7)
                ax.set_xlim(ext[:2]); ax.set_ylim(ext[2], ext[3])
                ax.set_title(f"{what}{'' if what == 'white' else ' nm'} (QI {qs[what]:.0f})", fontsize=7, pad=2)
                style(ax)
            elif kind == "overlay":
                for lab in labs:
                    if outlines[lab] is not None:
                        ax.plot(*outlines[lab], color=CH_COLOURS.get(lab, "k"), lw=1.2)
                        ax.plot(*tf_pt(*centres[lab]), "+", color=CH_COLOURS.get(lab, "k"), ms=6)
                ax.set_xlim(ext[:2]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
                ax.set_title("overlay", fontsize=7, pad=2)
                style(ax)
            elif kind in ("rgb_rms", "rgb_cm"):
                maps = rms if kind == "rgb_rms" else {k: np.abs(v) for k, v in cm.items()}
                ax.imshow(tf_img(rgb_image(maps, qs, sl)), extent=ext, interpolation="nearest")
                ax.set_title("RGB RMS" if kind == "rgb_rms" else "RGB |cm_most_important|", fontsize=7, pad=2)
                style(ax)
            elif kind == "kernels":
                for lab, ch in zip(labs, channels):
                    k = sel.sta_single_pixel.sel(channel=ch).values
                    if np.isfinite(k).any():
                        ax.plot(t_ms, k, color=CH_COLOURS.get(lab, "k"), lw=1,
                                ls="-" if qs[lab] >= args.min_q else "--")
                ax.axvline(0, color="0.7", lw=0.6); ax.axhline(0.5, color="0.7", lw=0.6)
                ax.set_title("spectral kernels", fontsize=7, pad=2)
                ax.set_xlabel("time (ms)", fontsize=6, labelpad=1); ax.set_ylabel("STA", fontsize=6, labelpad=1)
                ax.tick_params(labelsize=6)
            elif kind == "fff":
                name, idx, n_rep = what
                t, p = fff[fff_i]; fff_i += 1
                fff_background(ax, name)
                ax.plot(t, p, color="black", lw=0.8)
                ax.set_xlim(0, FFF_DURATION_S[name]); ax.set_ylim(0, fff_max)
                ax.tick_params(labelsize=5)
                ax.set_title(f"{name} (stim {idx})", fontsize=6, pad=1)
                if name == "scf" and fff_i == 1 + [s[0] for s in fff_stims].index("scf"):
                    for k2, lab2 in enumerate(SCF_LABELS):
                        ax.text(4 * k2 + 1, 0.97, lab2, transform=ax.get_xaxis_transform(), fontsize=4,
                                ha="center", va="top")
                if i == len(rows) - 1:
                    ax.set_xlabel("time (s)", fontsize=6, labelpad=1)
                if j == 0:
                    ax.set_ylabel("Hz", fontsize=6, labelpad=1)
            if i == 0:
                ax.set_title(title_txt + "\n" + ax.get_title(), fontsize=7)
            if j == 0 and kind not in ("sta",):
                row_lab = {"map": f"{what}{'' if what == 'white' else ' nm'}", "overlay": "overlay",
                           "rgb_rms": "RGB RMS", "rgb_cm": "RGB |cm|", "kernels": "kernels",
                           "fff": None}[kind]
                if row_lab:
                    ax.annotate(row_lab, xy=(-0.42, 0.5), xycoords="axes fraction", rotation=90,
                                ha="center", va="center", fontsize=9, fontweight="bold")

    handles = [Line2D([], [], color=CH_COLOURS.get(l, "k"), label=f"{l}{'' if l == 'white' else ' nm'}")
               for l in labs] + [Line2D([], [], color="0.4", marker="+", ls="", label="cutout centre")]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), fontsize=8, frameon=True,
               bbox_to_anchor=(0.005, 1 - 0.45 / fig.get_figheight()))
    fig.suptitle(f"{args.recording} {R['root'].name} stim {stim_index}, {title}.\nRF centre outlines "
                 f"(QI ≥ {args.min_q:.0f}) on covariance maps, {args.crop_um:.0f} µm crops, {frame_desc} "
                 f"(+y down).\nKernels dashed: QI < {args.min_q:.0f}. DSI/OSI: moving bar "
                 f"(stim {R['moving_bar_index']}), peak PSTH rate.", x=0.005, ha="left", fontsize=8,
                 y=1 - 0.12 / fig.get_figheight())
    fig.savefig(fname, dpi=args.dpi)
    plt.close(fig)
    print("saved", fname)


for g, (desc, cells) in enumerate(select_groups(), start=1):
    pages = [cells[k:k + args.per_page] for k in range(0, len(cells), args.per_page)]
    for p, page in enumerate(pages[: args.max_pages] if args.max_pages else pages, start=1):
        extra = ("_stimframe" if args.frame == "stimulus" else "") + ("_sta" if args.sta_frames else "") + ("_fff" if args.fff else "")
        name = "cells" if args.cells else f"group{g}"
        if args.split_opponent:
            name = ["opponent", "not_opponent", "one_colour"][g - 1]
        plot_page(page, f"Group {g}: {desc}, sorted by summed QI" if not args.cells else desc,
                  args.out / f"{args.recording}_{name}_page{p}{extra}.png")
