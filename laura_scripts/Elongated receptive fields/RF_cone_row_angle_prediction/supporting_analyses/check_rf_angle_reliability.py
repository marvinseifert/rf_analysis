"""
How reliable is the RF long-axis `angle` in noise_data_v2.nc? (checked 19/11/2026)

`angle` (and `tilt`) come from a PCA of each cell's centre outline (stats/summary.py).
Two checks, in the stimulus frame, for every channel of the summary-plot recordings:

  1. Independent measurement: the long axis of the central blob of the cell's covariance
     map (cm_most_important; pixels >= 50% of the peak, connected to it, weighted second
     moments) vs the stored angle, for the cells used in the figures.
     19/11/2026: median difference 11° (6° for tilt < 0.75); chance 45°.
  2. Same cell, different noise stimulus (repeats, other check sizes, other rates): the
     stored angle of the same cell in two channels vs the angle of DIFFERENT cells
     (chance level, from shuffling which cells are paired).
     19/11/2026: same cell 9-23° apart (4-14° for elongated cells), different cells 23-46°;
     also between 32 px and 8 / 6 px checks, so not a pixel-grid artefact.

Cells: quality >= QUALITY_MIN in the channel(s); cells within EDGE_EXCLUSION_UM of the
stimulus edge left out (needs rf_vs_cone_row_angle_all_recordings_cells.csv from
rf_cone_row_angle.py, for the edge flag).

Run from the repo root (loads one recording at a time; ~1-2 min each):
    poetry run python "laura_scripts/Elongated receptive fields/RF_cone_row_angle_prediction/supporting_analyses/check_rf_angle_reliability.py"
Writes per-recording tables to OUTPUT_FOLDER/angle_reliability/ and prints the summaries.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rf_cone_row_angle.py is one folder up
import rf_cone_row_angle as rca  # noqa: E402
from laura_scripts.Alignment_pipeline.alignment_recordings import noise_data_v2_file  # noqa: E402
from loading.noise_data import load_noise_data  # noqa: E402

QUALITY_MIN = rca.QUALITY_MIN
WINDOW_PX = 150  # half-size of the map window around each cell (maps are only stored near the cell)
OUT_DIR = rca.OUTPUT_FOLDER / "angle_reliability"
N_SHUFFLES = 2000


def axis_diff(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180
    return np.minimum(d, 180 - d)


def moment_angle(m):
    """Long-axis angle (0-180°, 0 = +x towards +y) and minor/major of the central blob."""
    m = np.nan_to_num(m)
    a = np.abs(m)
    if a.max() == 0:
        return np.nan, np.nan
    py, px = np.unravel_index(a.argmax(), a.shape)
    sign = np.sign(m[py, px])
    lab, _ = ndimage.label((m * sign) >= 0.5 * a.max())
    yy, xx = np.nonzero(lab == lab[py, px])
    if len(yy) < 3:
        return np.nan, np.nan
    w = (m * sign)[yy, xx]
    mx, my = np.average(xx, weights=w), np.average(yy, weights=w)
    ev, evec = np.linalg.eigh(np.cov(np.vstack([xx - mx, yy - my]), aweights=w))
    v = evec[:, 1]
    return np.degrees(np.arctan2(v[1], v[0])) % 180, np.sqrt(max(ev[0], 0) / ev[1])


def measure_recording(rec):
    rows = []
    with load_noise_data(noise_data_v2_file(rec), lazy=True) as ds:
        for ch in ds.channel.values:
            q = ds.quality.sel(channel=ch).values
            cells = ds.cell_index.values[q >= QUALITY_MIN]
            for c in cells:
                pos = ds.positions.sel(channel=ch, cell_index=c).values  # (x, y) indices
                x0, y0 = int(pos[0]), int(pos[1])
                win = (ds.cm_most_important.sel(channel=ch, cell_index=c)
                       .isel(x=slice(max(x0 - WINDOW_PX, 0), x0 + WINDOW_PX),
                             y=slice(max(y0 - WINDOW_PX, 0), y0 + WINDOW_PX))
                       .transpose("y", "x").values)
                ma, mr = moment_angle(win)
                rows.append(dict(recording=rec, channel=str(ch), cell_index=int(c),
                                 angle=float(ds.angle.sel(channel=ch, cell_index=c)),
                                 tilt=float(ds.tilt.sel(channel=ch, cell_index=c)),
                                 moment_angle=ma, moment_ratio=mr))
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_cells = pd.read_csv(rca.OUTPUT_FOLDER / "rf_vs_cone_row_angle_all_recordings_cells.csv")
    near_edge = main_cells.set_index(["recording", "cell_index"]).near_edge
    frames = []
    for rec, _ in rca.RECORDINGS:  # one at a time (large files)
        out = OUT_DIR / f"{rec}_angles.csv"
        d = pd.read_csv(out) if out.exists() else measure_recording(rec)
        d.to_csv(out, index=False)
        frames.append(d)
        print(rec, "channels:", sorted(d.channel.unique()))
    d = pd.concat(frames, ignore_index=True)
    d = d[[not near_edge.get((r, c), False) for r, c in zip(d.recording, d.cell_index)]]

    print("\n1. Stored angle vs independent moment angle (figure channel, edge cells left out)")
    m = d.merge(main_cells[["recording", "channel", "cell_index"]], on=["recording", "channel", "cell_index"])
    m["diff"] = axis_diff(m.angle, m.moment_angle)
    for lab, s in [("all", m), ("tilt < 0.75", m[m.tilt < 0.75]), ("tilt >= 0.85", m[m.tilt >= 0.85])]:
        print(f"  {lab:12s} n={len(s):3d}  median {s['diff'].median():4.0f}°  "
              f"within 20°: {np.mean(s['diff'] < 20) * 100:3.0f}%   (chance: 45°, 22%)")

    print("\n2. Same cell in two channels vs different cells (chance)")
    rng = np.random.default_rng(0)
    for rec, g in d.groupby("recording", sort=False):
        chans = list(dict.fromkeys(g.channel))
        for c1, c2 in itertools.combinations(chans, 2):
            a = g[g.channel == c1].set_index("cell_index")
            b = g[g.channel == c2].set_index("cell_index")
            common = a.index.intersection(b.index)
            if len(common) < 5:
                continue
            A, B = a.loc[common, "angle"].values, b.loc[common, "angle"].values
            chance = np.median([np.median(axis_diff(A, rng.permutation(B))) for _ in range(N_SHUFFLES)])
            el = np.maximum(a.loc[common, "tilt"].values, b.loc[common, "tilt"].values) < 0.85
            line = (f"  {rec} {c1[:22]:22s} vs {c2[:22]:22s} n={len(common):3d}  same cell "
                    f"{np.median(axis_diff(A, B)):3.0f}°  different cells {chance:3.0f}°")
            if el.sum() >= 5:
                line += f"  | elongated (tilt < 0.85) n={el.sum()}: {np.median(axis_diff(A[el], B[el])):3.0f}°"
            print(line)


if __name__ == "__main__":
    main()
