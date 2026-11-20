"""
Random-time control snippets for sNMF (20/11/2026).

Copy of rf_torch/rf_torch/main_sta_snippets_all.py with random_control switched ON, for chosen cells only.
Instead of cutting the noise around the cell's real spikes, it cuts it around the SAME NUMBER of random time
points (drawn from the whole noise stimulus, seed 0, as in rf_torch). Everything else is identical to the real
run: same 60x60 window around the cell, same 100 bins (79 before, 1, 20 after), same 10 ms grid.

Why: the shuffled noise makes neighbouring snippet pixels correlated, so sNMF on PURE NOISE may still give
smooth, blob-shaped "subunits". Running sNMF on these random snippets shows what noise subunits look like
(see snmf_real_vs_random.py).

Output (in each cell folder, NEW names - the real snippets.h5 / params.json / spike_times_snippets.npy are
never touched):
    cell_N/snippets_random_control.h5      (dataset "snippets", same layout as snippets.h5)
    cell_N/params_random_control.json
    (--seed N other than 0: ..._seed<N>.h5 / params_..._seed<N>.json - a second, independent random set)
Refuses to overwrite an existing snippets_random_control.h5.

Run from the repo root (GPU needed, ~30 s per cell):
    poetry run python laura_scripts/SNMF/random_control_snippets.py --cells 252
"""

import argparse
import json
import shutil
from pathlib import Path

import einops
import h5py
import hdf5plugin  # noqa: F401  (needed to read the Blosc-compressed stimulus / snippets)
import numpy as np
from polarspike import Overview
from rf_torch import sta_gpu
from rf_torch.parameters import Noise_Params

parser = argparse.ArgumentParser()
parser.add_argument("--cells", type=int, nargs="+", required=True)
parser.add_argument("--mode", choices=["random", "shifted"], default="random",
                    help="random: random times (rf_torch random_control). shifted: the cell's REAL spike train moved "
                         "by one random offset (wrapping round), keeps bursts and spike count, breaks the link "
                         "to the stimulus")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (0 = as rf_torch). Other seeds save as snippets_random_control_seed<N>.h5")
parser.add_argument("--project-root", type=Path, default=Path("/mnt/mea_nas/Laura/zebrafish_14_08_2026/Phase_00"))
parser.add_argument("--noise-stim-index", type=int, default=10)
parser.add_argument("--noise-stim-name", default="6px_15Hz_30mins_shuffle_x6")
parser.add_argument("--noise-file", type=Path,
                    default=Path("/mnt/mea_nas/Laura/stimuli/stimuli_July_2026/6px_15Hz_30mins_shuffle_x6.h5"))
args = parser.parse_args()

# Same snippet settings as main_sta_snippets_all.py
MAX_SPIKES = 50000
CUT_HALF = 30
BORDER_SIZE = 0
CHUNK_LEN = 30
if args.mode == "random":
    OUTPUT_NAME = "snippets_random_control" + ("" if args.seed == 0 else f"_seed{args.seed}")
else:
    OUTPUT_NAME = f"snippets_shifted_seed{args.seed}"
TMP_DIR = Path.home() / "snippets_tmp"
TMP_DIR.mkdir(exist_ok=True)

stim_folder = args.project_root / f"{args.noise_stim_name}_idx_{args.noise_stim_index}"
noise_params = Noise_Params.load_json(stim_folder / "noise_params.json")
# noise_params.json still holds the old Windows paths: replace the two that are used
noise_params.noise_file = args.noise_file
noise_params.output_folder = stim_folder

recording = Overview.Recording.load(args.project_root / "overview")
recording.dataframes["noise_analysis"] = recording.dataframes["spikes_df"].query(
    "stimulus_index == @args.noise_stim_index"
)
triggers = (
    recording.dataframes["stimulus_df"]
    .query("stimulus_index == @args.noise_stim_index")["trigger_fr_relative"]
    .to_numpy()
)[0]
quality = np.load(stim_folder / "quality.npy")  # rows = cells: (quality, cell_index, centre_x, centre_y)

print(f"Loading stimulus {args.noise_file}")
with h5py.File(args.noise_file, "r") as f:
    stim = einops.rearrange(np.asarray(f[noise_params.noise_key]), "t h w -> h w t")
    stim = (stim == 255).astype(np.uint8)


def cut_window(location):
    """Same as main_sta_snippets_all.cut_window: 60x60 window around (x, y), zero-padded outside the stimulus."""
    size = 2 * CUT_HALF
    origin_y = location[1] - CUT_HALF - BORDER_SIZE
    origin_x = location[0] - CUT_HALF - BORDER_SIZE
    y0, y1 = max(origin_y, 0), min(origin_y + size, stim.shape[0])
    x0, x1 = max(origin_x, 0), min(origin_x + size, stim.shape[1])
    window = np.zeros((size, size, stim.shape[2]), dtype=stim.dtype)
    window[y0 - origin_y:y1 - origin_y, x0 - origin_x:x1 - origin_x] = stim[y0:y1, x0:x1]
    attrs = dict(window_origin_y=origin_y, window_origin_x=origin_x,
                 cut_begin_y=y0, cut_end_y=y1, cut_begin_x=x0, cut_end_x=x1)
    return window, attrs


for cell in args.cells:
    cell_dir = stim_folder / f"cell_{cell}"
    final_file = cell_dir / f"{OUTPUT_NAME}.h5"
    if final_file.exists():
        print(f"cell {cell}: {final_file.name} already exists - skipped (delete it to redo)")
        continue

    spikes = recording.get_spikes_triggered(
        [{"stimulus_index": args.noise_stim_index, "cell_index": cell}], time="frames", pandas=False
    )
    n = min(len(spikes), MAX_SPIKES)  # same number of snippets as the real run
    if n == 0:
        print(f"cell {cell}: no spikes - skipped")
        continue

    if args.mode == "random":
        # RANDOM CONTROL: identical to the random_control branch of main_sta_snippets_all.py
        possible_times = np.arange(0, triggers.max(), dtype=int)
        np.random.seed(args.seed)
        random_idxs = np.sort(np.random.choice(possible_times, size=n, replace=False))
        shift_info = {}
    else:
        # SHIFTED SPIKE TRAIN: same spikes as the real run (same 50,000 cap and seed), all moved by one offset,
        # wrapping round at the end of the noise. The noise sequence is played stim_repeats times, so an
        # offset close to a whole repeat would put the spikes back on the SAME stimulus: the offset is kept
        # between 10 % and 90 % of one repeat (plus any whole number of repeats).
        if len(spikes) > MAX_SPIKES:
            spikes = spikes.sample(n=MAX_SPIKES, seed=0)
        times = spikes["times_relative"].to_numpy().astype(np.int64).ravel()
        total = int(triggers.max())
        repeat_len = total / noise_params.stim_repeats
        rng = np.random.default_rng(args.seed)
        offset = int(rng.integers(0, noise_params.stim_repeats) * repeat_len + rng.uniform(0.1, 0.9) * repeat_len)
        random_idxs = np.sort((times + offset) % total)
        shift_info = dict(offset_samples=offset, offset_s=offset / noise_params.sample_freq,
                          repeat_len_s=repeat_len / noise_params.sample_freq)
        print(f"cell {cell}: shift {offset / noise_params.sample_freq:.1f} s "
              f"(one noise repeat = {repeat_len / noise_params.sample_freq:.1f} s)")

    window, attrs = cut_window(quality[cell, 2:4].astype(int))
    attrs["random_control"] = 1 if args.mode == "random" else 0
    attrs["shifted_spikes"] = 1 if args.mode == "shifted" else 0
    tmp_file = TMP_DIR / f"{recording.name}_{args.noise_stim_name}_cell_{cell}_{OUTPUT_NAME}.h5"
    try:
        sta_gpu.kernel_gpu_snippets_buffered(
            window, random_idxs, triggers,
            total_len=noise_params.total_sta_len,
            post_spike_len=noise_params.post_spike_len,
            downsample=noise_params.downsample,
            h5_path=tmp_file,
            stim_repeats=noise_params.stim_repeats,
            chunk_len=CHUNK_LEN,
            attrs=attrs,
        )
        cell_dir.mkdir(exist_ok=True)
        part = final_file.with_name(final_file.name + ".part")
        shutil.copyfile(tmp_file, part)
        if part.stat().st_size != tmp_file.stat().st_size:
            raise IOError(f"size mismatch copying {tmp_file}")
        part.replace(final_file)
    finally:
        tmp_file.unlink(missing_ok=True)

    with open(cell_dir / f"params_{OUTPUT_NAME}.json", "w") as f:
        json.dump(dict(cell_idx=cell, n_random_times=int(n), n_real_spikes=int(len(spikes)), seed=args.seed, mode=args.mode, **shift_info,
                       max_spikes=MAX_SPIKES, cut_half=CUT_HALF, noise_file=str(args.noise_file),
                       total_sta_len=noise_params.total_sta_len, post_spike_len=noise_params.post_spike_len,
                       downsample=noise_params.downsample, stim_repeats=noise_params.stim_repeats,
                       source="laura_scripts/SNMF/random_control_snippets.py"), f, indent=4)
    print(f"cell {cell}: {n} random-time snippets -> {final_file}")
