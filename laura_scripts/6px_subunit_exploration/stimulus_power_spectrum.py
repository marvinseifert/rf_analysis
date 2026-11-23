"""
Spatial power spectrum of the 6px shuffled checkerboard, computed from the frames in the stimulus file.

Question: does a ~12 µm checker drive spatial periods of 10-15 µm (the cone spacing)?
A binary checker of side s has, per axis, power ~ sinc^2(f * s), which is zero at f = 1/s.
For s = 6 px * 1.9476 µm = 11.7 µm the first zero sits at a period of 11.7 µm, inside the cone band.
The grid shift ("shuffle") only changes the phase of each frame's Fourier transform, so it does
not change the power spectrum; it only spreads the grid edges over the shift positions.
For a linear cell and white-in-time noise, STA = RF convolved with the stimulus spatial
autocorrelation, i.e. STA spectrum = RF spectrum x this power spectrum. So wherever the power
is near zero, the STA cannot show that spatial frequency of the RF.

Steps:
  1. read N_BLOCKS blocks of frames spread through the file (blocks = HDF5 chunks of 422 frames)
  2. contrast per frame: 0/255 -> -1/+1, frame mean removed, Hann window
  3. average |FFT|^2 over frames -> 2D power spectrum (cycles/µm on the axes)
  4. compare with the sinc^2 x sinc^2 prediction for a 6 px checker; cuts along x, y, diagonal,
     and a radial average; the 10-15 µm cone-period band is shaded
  5. read the grid shift of each frame (positions of the checker edges, modulo 6)

Pixel size: corrected_stim_um_per_px for 14/08/2026 (1.9476 µm/px, from the alignment cache);
the file grid is 798 x 798 px.

Run from the repo root:
    poetry run python laura_scripts/6px_subunit_exploration/stimulus_power_spectrum.py
"""
import argparse
from pathlib import Path

import hdf5plugin  # noqa: F401  registers the blosc filter used by the stimulus file
import h5py
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FOLDER = Path("/home/laura/Box/SUSSEX/Experiments/Zebrafish/Figures/RF_cone_alignment_v2")
STIM_FILE = Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/6px_15Hz_30mins_shuffle_x6.h5")

parser = argparse.ArgumentParser()
parser.add_argument("--stim-file", type=Path, default=STIM_FILE)
parser.add_argument("--um-per-px", type=float, default=1.9476, help="corrected_stim_um_per_px, 14/08/2026")
parser.add_argument("--n-blocks", type=int, default=8, help="blocks of 422 frames, spread through the file")
parser.add_argument("--cone-band-um", type=float, nargs=2, default=(10.0, 15.0))
args = parser.parse_args()

# ------------------------------------------------------------ frames
with h5py.File(args.stim_file, "r") as f:
    noise = f["Noise"]
    s_px = int(f["Checkerboard_Size"][()])
    n_frames, ny, nx = noise.shape
    block = noise.chunks[0]
    starts = np.linspace(0, n_frames - block, args.n_blocks).astype(int) // block * block
    print(f"{args.stim_file.name}: {n_frames} frames, {ny}x{nx} px, checker {s_px} px, "
          f"frame rate {f['Frame_Rate'][()]} Hz, shuffle {bool(f['Shuffle'][()])}")
    print(f"using {args.n_blocks} blocks of {block} frames starting at {starts.tolist()}")

    um = args.um_per_px
    win = np.outer(np.hanning(ny), np.hanning(nx)).astype(np.float32)
    win_norm = (win ** 2).sum()
    psd = np.zeros((ny, nx), np.float64)
    offsets = []
    n_used = 0
    for st in starts:
        frames = noise[st:st + block].astype(np.float32) / 127.5 - 1.0  # -1 / +1
        # grid shift: columns/rows where the frame changes from the previous one, modulo s
        dx = np.any(frames[:, :, 1:] != frames[:, :, :-1], axis=1)  # (frames, nx-1)
        dy = np.any(frames[:, 1:, :] != frames[:, :-1, :], axis=2)
        for ex, ey in zip(dx, dy):
            jx = np.flatnonzero(ex) + 1
            jy = np.flatnonzero(ey) + 1
            offsets.append((np.bincount(jx % s_px, minlength=s_px).argmax(),
                            np.bincount(jy % s_px, minlength=s_px).argmax()))
        frames -= frames.mean(axis=(1, 2), keepdims=True)
        F = np.fft.fft2(frames * win)
        psd += (np.abs(F) ** 2).sum(axis=0)
        n_used += len(frames)
psd /= n_used * win_norm  # per-pixel variance units: sum over all bins = mean frame variance
psd = np.fft.fftshift(psd)
fx = np.fft.fftshift(np.fft.fftfreq(nx, d=um))  # cycles / µm
fy = np.fft.fftshift(np.fft.fftfreq(ny, d=um))
FX, FY = np.meshgrid(fx, fy)
FR = np.hypot(FX, FY)

# ------------------------------------------------------------ prediction and normalisation
s_um = s_px * um
pred = np.sinc(FX * s_um) ** 2 * np.sinc(FY * s_um) ** 2  # np.sinc(x) = sin(pi x)/(pi x)
low = (FR > 0) & (FR < 0.1 / s_um)  # plateau, well below the first zero
norm = psd[low].mean()
psd_n = psd / norm  # 1 = power at the lowest spatial frequencies

f_lo, f_hi = 1 / args.cone_band_um[1], 1 / args.cone_band_um[0]
band = (FR >= f_lo) & (FR <= f_hi)
nyq = 0.5 / um
print(f"\nchecker side {s_um:.2f} µm -> first zero along x/y at a period of {s_um:.2f} µm "
      f"({1 / s_um:.4f} cyc/µm); Nyquist of the file grid {nyq:.3f} cyc/µm ({1 / nyq:.2f} µm)")
print(f"cone band {args.cone_band_um[0]:.0f}-{args.cone_band_um[1]:.0f} µm period "
      f"= {f_lo:.4f}-{f_hi:.4f} cyc/µm")
print(f"mean power in the cone band (all orientations), relative to the low-frequency plateau: "
      f"measured {psd_n[band].mean():.3f}, sinc^2 prediction {pred[band].mean():.3f}")

# fraction of the total stimulus variance carried at each radial frequency (cumulative)
order = np.argsort(FR.ravel())
cum = np.cumsum(psd.ravel()[order]) / psd.sum()
fr_sorted = FR.ravel()[order]
for p in (args.cone_band_um[1], s_um, args.cone_band_um[0]):
    print(f"  variance at periods longer than {p:5.1f} µm: {np.interp(1 / p, fr_sorted, cum) * 100:5.1f} %")
print(f"  variance inside the cone band: "
      f"{(np.interp(f_hi, fr_sorted, cum) - np.interp(f_lo, fr_sorted, cum)) * 100:.1f} %")

# cuts
cx, cy = nx // 2, ny // 2
cut_x = psd_n[cy, cx:]
f_cut = fx[cx:]
diag_n = min(nx, ny) // 2
cut_d = psd_n[cy + np.arange(diag_n), cx + np.arange(diag_n)]
f_diag = np.hypot(fx[cx:cx + diag_n], fy[cy:cy + diag_n])
rbins = np.linspace(0, nyq, 120)
idx = np.digitize(FR.ravel(), rbins)
rad = np.array([psd_n.ravel()[idx == i].mean() if np.any(idx == i) else np.nan for i in range(1, len(rbins))])
rad_pred = np.array([pred.ravel()[idx == i].mean() if np.any(idx == i) else np.nan for i in range(1, len(rbins))])
rc = 0.5 * (rbins[1:] + rbins[:-1])

for name, ff, cc in (("x axis", f_cut, cut_x), ("diagonal", f_diag, cut_d), ("radial", rc, rad)):
    vals = [np.interp(1 / p, ff, cc) for p in (15.0, 12.5, s_um, 10.0)]
    print(f"{name:9s} power at 15 / 12.5 / {s_um:.1f} / 10 µm period: " + " / ".join(f"{v:.3f}" for v in vals))

off = np.array(offsets)
hist = np.zeros((s_px, s_px), int)
np.add.at(hist, (off[:, 1], off[:, 0]), 1)
print(f"\ngrid shift (x, y) in file px, counts over {len(off)} frames:\n{hist}")

# ------------------------------------------------------------ figure
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
ax = axs[0]
lim = 0.25
sel = (np.abs(fx) <= lim)
ext = [fx[sel][0], fx[sel][-1], fy[np.abs(fy) <= lim][0], fy[np.abs(fy) <= lim][-1]]
im = ax.imshow(10 * np.log10(psd_n[np.ix_(np.abs(fy) <= lim, sel)] + 1e-6), origin="lower", extent=ext,
               cmap="magma", vmin=-40, vmax=3)
th = np.linspace(0, 2 * np.pi, 300)
for r in (f_lo, f_hi):
    ax.plot(r * np.cos(th), r * np.sin(th), color="cyan", lw=0.8)
ax.set_xlabel("fx (cycles/µm)")
ax.set_ylabel("fy (cycles/µm)")
ax.set_title("2D power (dB re low-frequency plateau)\ncyan: 10-15 µm period band")
fig.colorbar(im, ax=ax, shrink=0.8)

for ax, xs in ((axs[1], "lin"), (axs[2], "log")):
    ax.axvspan(f_lo, f_hi, color="cyan", alpha=0.2, label="cone band 10-15 µm")
    ax.plot(f_cut, cut_x, color="tab:blue", label="measured, along x")
    ax.plot(f_cut, np.sinc(f_cut * s_um) ** 2, color="tab:blue", ls=":", label="sinc² prediction, along x")
    ax.plot(f_diag, cut_d, color="tab:green", label="measured, diagonal")
    ax.plot(rc, rad, color="k", lw=2, label="measured, radial average")
    ax.plot(rc, rad_pred, color="k", ls=":", label="prediction, radial average")
    for k in (1, 2):
        ax.axvline(k / s_um, color="grey", lw=0.6, ls="--")
    ax.set_xlim(0, nyq)
    ax.set_xlabel("spatial frequency (cycles/µm)")
    ax.set_ylabel("power re low-frequency plateau")
    sec = ax.secondary_xaxis("top", functions=(lambda f: 1 / np.maximum(f, 1e-3), lambda p: 1 / np.maximum(p, 1e-3)))
    sec.set_xticks([50, 20, 15, 12, 10, 8, 6, 5])
    sec.set_xlabel("period (µm)")
    if xs == "log":
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2)
    else:
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7, loc="upper right")
fig.suptitle(f"{args.stim_file.stem}: {n_used} frames, checker {s_px} px = {s_um:.1f} µm "
             f"({um:.4f} µm/px); dashed grey = zeros at 1/s, 2/s")
fig.tight_layout()

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
stem = f"stimulus_power_spectrum_{args.stim_file.stem}"
for fmt in ("png", "pdf"):
    out = OUTPUT_FOLDER / f"{stem}.{fmt}"
    fig.savefig(out, dpi=200 if fmt == "png" else None, bbox_inches="tight")
    print(f"saved {out}")
