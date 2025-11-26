from clustering import rf_clustering
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
from pickle import load as load_pickle
from importlib import reload
import numpy as np

# %%
root_paths = [Path(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Laura/zebrafish_05_11_2025/Phase_01/noise_analysis")]

with open(root_paths[0] / "settings.pkl", "rb") as f:
    settings_dict = load_pickle(f)

reload(rf_clustering)
rfc = rf_clustering.RFClustering(root_paths)
rfc.channel_colours = ["black", "green"]
rfc.load_rf_data()

rfc.prepare_data(include_external_data=False)

fig, ax = plt.subplots()
ax.plot(rfc.all_data[40])
fig.show()

# %%

rfc.run_pca()

fig, ax = rfc.plot_pca()
fig.show()

fig, ax = rfc.plot_distance_matrix()
fig.show()

rfc.cluster_data(method="agglomerative", n_clusters=15)

fig, ax = rfc.plot_clusters_pca()
fig.show()

# %%
figs = rfc.plot_cm_most_important(
    save_path=r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Laura/zebrafish_05_11_2025/Phase_01/noise_analysis/cluster_plots",
    cmap="coolwarm")
# %%
fig, ax = rfc.plot_stas()
for a in ax.flatten():
    # set x ticks to -800 to +200 in steps of 100
    a.set_xticks(np.arange(0, 100, 10))
    a.set_xticklabels(np.arange(-800, 200, 100))
    a.axvline(80, color="black", linestyle="--")
    a.set_xlabel("Time (ms)")
    a.set_ylabel("Z-scored amplitude")

fig.savefig(
    r"/run/user/1000/gvfs/smb-share:server=mea_nas_25.local,share=root/Laura/zebrafish_05_11_2025/Phase_01/noise_analysis/cluster_plots/stas.svg")
