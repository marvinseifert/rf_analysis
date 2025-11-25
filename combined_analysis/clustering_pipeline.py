from clustering import rf_clustering
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import hdbscan
from normalization.normalize_sta import zscore_sta
from pickle import load as load_pickle
from importlib import reload

# %%
root_paths = [Path(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Laura/zebrafish_05_11_2025/Phase_01/noise_analysis")]

with open(root_paths[0] / "settings.pkl", "rb") as f:
    settings_dict = load_pickle(f)
# %%
reload(rf_clustering)
rfc = rf_clustering.RFClustering(root_paths)
rfc.load_rf_data()
# %%
rfc.dfs = rfc.dfs.filter(pl.col("quality") > 15)
# %%
rfc.prepare_data(include_external_data=False)
