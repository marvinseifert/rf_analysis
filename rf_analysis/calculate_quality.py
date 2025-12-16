from __future__ import annotations
import numpy as np
from pathlib import Path
from smoothing.gaussian import smooth_ker
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm
import sys
from multiprocessing import get_start_method
import xarray as xr
from loading.load_sta import _create_sta_dataarray


def quality_on_cells(
        folder: Path, cell_ids: List, dt_ms: float = 1.0, t_zero_index: int = 0
):
    quality = np.zeros((len(cell_ids), 4))
    quality = xr.DataArray(
        data=quality,
        dims=["cell_index", "metrics"],
        coords={
            "cell_index": cell_ids,
            "metrics": ["quality", "cell_id", "center_x", "center_y"],
        },
    )

    for idx, cell_id in enumerate(cell_ids):
        data_path = Path(folder / rf"cell_{cell_id}//kernel.npy")

        # data_path = Path(rf"C:\Users\Marvin\Downloads\2023-5-5_MP_0_1_SWN_R_25_STRFs.npy")

        # data_path = Path(rf"D:\zebrafish_26_10_23\BW_Noise\\cell_{cell_id}\\ker.npy")
        try:
            sta_data = np.load(data_path)
        except FileNotFoundError:
            quality[idx, 0] = 0
            quality[idx, 1] = cell_id
            quality[idx, 2] = 0
            quality[idx, 3] = 0

            continue
        # sta_data = sta_data[cell_id, :, 20:-20, 20:-20]

        # sta_data = sta_data[:, ::3, ::3] #decimate_ker(sta_data, 3)
        ker_sm = _create_sta_dataarray(smooth_ker(sta_data), dt_ms, t_zero_index)

        center = ker_sm.var(dim="time").argmax(dim=["x", "y"])
        q_index = ker_sm.var(dim="time").max() / ker_sm.var(dim="time").median()

        quality[idx, 0] = q_index
        quality[idx, 1] = cell_id
        quality[idx].loc["center_x"] = center["x"].item()
        quality[idx].loc["center_y"] = center["y"].item()

    return quality


# def quality_parallel(folder, cell_ids):
#     quality = np.zeros((len(cell_ids), 4))
#     for idx, cell_id in enumerate(cell_ids):
#
#         data_path = Path(folder / rf"cell_{cell_id}//kernel.npy")
#
#         # data_path = Path(rf"C:\Users\Marvin\Downloads\2023-5-5_MP_0_1_SWN_R_25_STRFs.npy")
#
#         # data_path = Path(rf"D:\zebrafish_26_10_23\BW_Noise\\cell_{cell_id}\\ker.npy")
#         try:
#             sta_data = np.load(data_path)
#         except FileNotFoundError:
#             quality[idx, 0] = 0
#             quality[idx, 1] = cell_id
#             quality[idx, 2] = 0
#             quality[idx, 3] = 0
#
#             continue
#         # sta_data = sta_data[cell_id, :, 20:-20, 20:-20]
#
#         # sta_data = sta_data[:, ::3, ::3] #decimate_ker(sta_data, 3)
#         ker_sm = smooth_ker(sta_data)
#         _, cY, cX = np.unravel_index(np.argmax(np.var(ker_sm, axis=0)), ker_sm.shape)
#         quality[idx, 0] = np.max(np.var(ker_sm, axis=0)) / np.median(
#             np.var(ker_sm, axis=0)
#         )
#         quality[idx, 1] = cell_id
#         quality[idx, 2] = cX
#         quality[idx, 3] = cY
#
#     return quality


# %%


def calculate_rf_quality(
        recording_config: "Recording_Config",
        cpus: int = None,
        analysis_folder: str = "rf_analysis",
):
    # Check if parallel processing  is possible:
    if cpus is None:
        cpus = cpu_count()

    # --- The Safety Guard ---
    current_name = __name__

    if current_name != "__main__":
        # Check 2: Are we using a start method that requires the main guard (e.g., 'spawn')?
        # Note: We check the OS's default method to be safe.
        default_start_method = get_start_method(allow_none=True)

        # If the OS uses 'spawn' by default, we MUST raise a warning/error.
        # 'spawn' is the default on Windows and macOS.
        if default_start_method in ["spawn", "forkserver"] and "ipython" in sys.modules:
            # Raise a clear, actionable error for the user
            raise RuntimeError(
                "DANGER: Cannot run multiprocessing in this environment "
                f"('{current_name}') using the '{default_start_method}' start method. "
                "You must wrap the call to 'run_analysis_parallel(...)' "
                "inside an 'if __name__ == \"__main__\":' block in your calling script "
                "or run the script directly from the terminal or set the start method to 'fork' or"
                "use only single-threaded execution."
            )

    # Start the parallel processing
    # Check number of folders in that folder
    for channel in recording_config.channel_names:
        folder = recording_config.channel_configs[channel].root_path
        nr_folders = len([f for f in folder.iterdir() if f.is_dir()])
        cpus = cpu_count()
        cell_ids = recording_config.overview.spikes_df["cell_index"].unique()
        chunk_size = len(cell_ids) // cpus
        chunks = np.array_split(cell_ids, cpus)

        # Create partial function
        func = partial(quality_on_cells, folder)

        # start pool
        pool = Pool(cpus)
        results = list(
            tqdm.tqdm(
                pool.imap(func, chunks),
                total=len(chunks),
                desc=f"Processing using {cpus} CPUs",
            )
        )
        pool.close()
        pool.join()
        results = xr.concat(results, dim="cell_index")
        results.to_netcdf(folder / "quality.nc")
