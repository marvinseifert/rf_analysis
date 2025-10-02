import numpy as np
from pathlib import Path
from analysis_tools import smooth_ker
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm


def quality_parallel(folder, cell_ids):
    quality = np.zeros((len(cell_ids), 4))
    for idx, cell_id in enumerate(cell_ids):

        data_path = Path(folder / rf"cell_{cell_id}//kernel_test.npy")

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
        ker_sm = smooth_ker(sta_data)
        _, cY, cX = np.unravel_index(np.argmax(np.var(ker_sm, axis=0)), ker_sm.shape)
        quality[idx, 0] = np.max(np.var(ker_sm, axis=0)) / np.median(
            np.var(ker_sm, axis=0)
        )
        quality[idx, 1] = cell_id
        quality[idx, 2] = cX
        quality[idx, 3] = cY

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
if __name__ == "__main__":
    # Start the parallel processing
    folder = Path(
        rf"/home/mawa/nas_a/Marvin/chicken_13_05_2025/kilosort4/4px_20Hz_shuffle_460nm_idx_2"
    )
    # Check number of folders in that folder
    nr_folders = len([f for f in folder.iterdir() if f.is_dir()])
    cpus = cpu_count()
    cell_ids = np.arange(nr_folders)
    chunk_size = len(cell_ids) // cpus
    chunks = np.array_split(cell_ids, cpus)

    # Create partial function
    func = partial(quality_parallel, folder)

    # start pool
    pool = Pool(cpus)
    results = list(tqdm.tqdm(pool.imap(func, chunks), total=len(chunks), desc="Processing"))
    pool.close()
    pool.join()
    results = np.vstack(results)
    np.save(
        folder / "quality.npy",
        results,
    )
