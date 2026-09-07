from __future__ import annotations

from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count, get_start_method
from typing import List

import sys
import numpy as np
import xarray as xr
import tqdm

from smoothing.gaussian import smooth_ker
from loading.load_sta import _create_sta_dataarray


def calculate_safe_quality_from_var_map(
    var_map: xr.DataArray,
    ignore_zero_variance_pixels: bool = True,
) -> float:
    """
    Calculate a safer RF quality value from a temporal variance map.

    Original quality was:

        max variance / median variance

    That fails for masked stimuli, because large static grey regions can make
    the median variance exactly zero, giving quality = inf.

    This version optionally ignores zero-variance pixels when calculating the
    denominator.
    """

    var_values = var_map.values.ravel()
    var_values = var_values[np.isfinite(var_values)]

    if var_values.size == 0:
        return 0.0

    max_var = np.nanmax(var_values)

    if not np.isfinite(max_var) or max_var <= 0:
        return 0.0

    if ignore_zero_variance_pixels:
        denominator_values = var_values[var_values > 0]
    else:
        denominator_values = var_values

    if denominator_values.size == 0:
        return 0.0

    median_var = np.nanmedian(denominator_values)

    if not np.isfinite(median_var) or median_var <= 0:
        return 0.0

    q_index = max_var / median_var

    if not np.isfinite(q_index):
        return 0.0

    return float(q_index)


def quality_on_cells_mask_safe(
    folder: Path,
    cell_ids: List[int],
    dt_ms: float = 1.0,
    t_zero_index: int = 0,
    ignore_zero_variance_pixels: bool = True,
):
    """
    Calculate quality and RF centre for a batch of cells.

    Saves the same metric layout as the original function:

        quality
        cell_id
        center_x
        center_y

    but avoids inf quality values caused by zero median variance.
    """

    quality = np.zeros((len(cell_ids), 4), dtype=float)

    quality = xr.DataArray(
        data=quality,
        dims=["cell_index", "metrics"],
        coords={
            "cell_index": cell_ids,
            "metrics": ["quality", "cell_id", "center_x", "center_y"],
        },
    )

    for idx, cell_id in enumerate(cell_ids):
        data_path = Path(folder / rf"cell_{cell_id}" / "kernel.npy")

        try:
            sta_data = np.load(data_path)

        except FileNotFoundError:
            quality[idx, 0] = 0
            quality[idx, 1] = cell_id
            quality[idx, 2] = 0
            quality[idx, 3] = 0
            continue

        except Exception as e:
            print("\n" + "=" * 80, flush=True)
            print("FAILED TO LOAD STA", flush=True)
            print("cell_id:", cell_id, flush=True)
            print("data_path:", data_path, flush=True)
            print("error:", repr(e), flush=True)
            print("=" * 80 + "\n", flush=True)
            raise

        ker_sm = _create_sta_dataarray(
            smooth_ker(sta_data),
            dt_ms,
            t_zero_index,
        )

        var_map = ker_sm.var(dim="time")

        center = var_map.argmax(dim=["x", "y"])

        q_index = calculate_safe_quality_from_var_map(
            var_map,
            ignore_zero_variance_pixels=ignore_zero_variance_pixels,
        )

        quality[idx, 0] = q_index
        quality[idx, 1] = cell_id
        quality[idx].loc["center_x"] = center["x"].item()
        quality[idx].loc["center_y"] = center["y"].item()

    return quality


def calculate_rf_quality_mask_safe(
    recording_config: "Recording_Config",
    cpus: int = 4,
    analysis_folder: str = "rf_analysis",
    ignore_zero_variance_pixels: bool = True,
):
    """
    Replacement for calculate_rf_quality that avoids infinite quality values.

    Use this for masked/asymmetric stimuli, such as the L-shaped noise stimulus,
    where a large static grey region can cause median variance = 0.

    This writes the same output filenames as the original function:

        quality.nc
        quality.npy

    inside each channel folder.
    """

    if cpus is None:
        cpus = cpu_count()

    current_name = __name__

    if current_name != "__main__":
        default_start_method = get_start_method(allow_none=True)

        if default_start_method in ["spawn", "forkserver"] and "ipython" in sys.modules:
            raise RuntimeError(
                "DANGER: Cannot run multiprocessing in this environment "
                f"('{current_name}') using the '{default_start_method}' start method. "
                "You must wrap the call inside an if __name__ == '__main__' block, "
                "run the script directly from the terminal, or use single-threaded execution."
            )

    for channel in recording_config.channel_names:
        folder = recording_config.channel_configs[channel].root_path

        cell_ids = np.arange(
            0,
            recording_config.overview.spikes_df["cell_index"].max() + 1,
        )

        chunks = np.array_split(cell_ids, cpus)

        func = partial(
            quality_on_cells_mask_safe,
            folder,
            dt_ms=recording_config.channel_configs[channel].dt_ms,
            t_zero_index=(
                recording_config.channel_configs[channel].total_sta_len
                - recording_config.channel_configs[channel].post_spike_bins
            ),
            ignore_zero_variance_pixels=ignore_zero_variance_pixels,
        )

        pool = Pool(cpus)

        results = list(
            tqdm.tqdm(
                pool.imap(func, chunks),
                total=len(chunks),
                desc=f"Processing {channel} using {cpus} CPUs",
            )
        )

        pool.close()
        pool.join()

        results = xr.concat(results, dim="cell_index")

        q = results.sel(metrics="quality").values

        print("\nQuality summary for", channel)
        print("min:", np.nanmin(q))
        print("max:", np.nanmax(q))
        print("finite:", np.isfinite(q).sum(), "of", q.size)
        print("inf:", np.isinf(q).sum())
        print("nan:", np.isnan(q).sum())
        print("nonzero:", np.sum(q > 0))

        results.to_netcdf(folder / "quality.nc")
        np.save(folder / "quality.npy", results.values)

        print("\nSaved:")
        print(folder / "quality.nc")
        print(folder / "quality.npy")
