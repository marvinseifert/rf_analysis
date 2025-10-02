import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
import tqdm
import hdbscan

import einops
from scipy.spatial.distance import cdist
from pathlib import Path
from s_nmf.factorization import semi_nmf_hals
from rf_torch.parameters import Cell_Params

# %%
############# Semi NMF parameters #############
n_components = 20
n_runs = 30
sparsity = 1e-2
topk_per_feature = 1
min_cluster_size = 10
n_cluster_samples = 5
# %%
data_root = Path(
    r"/media/mawa/52DE7A8FDE7A6B5D/noise_analysis/chicken_09_05_2024/kilosort4/4px_20Hz_shuffle_535nm_idx_12")
cell_idx = 466
save_root = data_root / f"cell_{cell_idx}"
cell_params = Cell_Params.load_from_root_json(save_root)
cell_params["s_nmf_analysis"]["n_components"] = n_components
cell_params["s_nmf_analysis"]["n_runs"] = n_runs
cell_params["s_nmf_analysis"]["sparsity"] = sparsity
cell_params["s_nmf_analysis"]["top_k_per_feature"] = 1
cell_params["s_nmf_analysis"]["min_cluster_size"] = min_cluster_size
cell_params["s_nmf_analysis"]["n_cluster_samples"] = n_cluster_samples

# %%
with h5py.File(data_root / f"cell_{cell_idx}/snippets.h5",
               "r") as f:
    print(f["snippets"].shape)
    # calculate size in GB
    size_gb = f["snippets"].size * f["snippets"].dtype.itemsize / (1024 ** 3)
    print(f"Size of snippets: {size_gb:.2f} GB")
    snippets = f["snippets"][:]

# %%
snippets = snippets[100:]  # Crop the snippets to remove borders
# random_indices = np.random.choice(snippets.shape[0], size=min(10000, snippets.shape[0]), replace=False)
# snippets = snippets[random_indices]  # Subsample snippets if there are too many
# %%
sta = np.mean(snippets, axis=0)  # Calculate the mean of all snippets
mse_snippets = np.max((np.mean(snippets, axis=0) - 0.5) ** 2, axis=0)  # Calculate MSE for each snippet
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mse_snippets, cmap='gray')
# flip the y-axis
ax.set_ylim(ax.get_ylim()[::-1])
fig.show()
# %%


T, H, W = sta.shape
N = snippets.shape[0]

# ---- Signed projection (keeps ON/OFF polarity) ----
W_signed = (sta - 0.5).astype(np.float32)  # (T,H,W)
projected_snippets = np.zeros((N, H, W), np.float32)

for t in tqdm.tqdm(range(T)):
    projected_snippets += (snippets[:, t, :, :].astype(np.float32) - 0.5) * W_signed[t]

# %%
sparsity = sparsity * np.median(np.abs(projected_snippets))
H_all = np.zeros((n_runs, n_components, projected_snippets.shape[1] * projected_snippets.shape[2]), dtype=np.float32)
W_all = np.zeros((n_runs, projected_snippets.shape[0], n_components), dtype=np.float32)
one_percent = int(0.1 * (projected_snippets.shape[0]))
for run in range(n_runs):
    W, H, info = semi_nmf_hals(projected_snippets.reshape(projected_snippets.shape[0], -1), n_components, max_iter=20,
                               subsample_svd=one_percent, random_state=run, l1_h=sparsity,
                               topk_per_feature=topk_per_feature)  # or 2 power_sharpen=1, ortho_lambda=1e-3, )

    W_all[run] = W
    H_all[run] = H

# %%For debugging:

W, H, info = semi_nmf_hals(projected_snippets.reshape(projected_snippets.shape[0], -1), 30, max_iter=1000,
                           subsample_svd=snippets.shape[0], random_state=run, l1_h=sparsity,
                           topk_per_feature=topk_per_feature)  # or 2 power_sharpen=1, ortho_lambda=1e-3, )

# %%
np.save(save_root / "H_single.npy", H)
# %%
fig, ax = plt.subplots(figsize=(20, 10), ncols=30)
for component in range(30):
    ax[component].imshow(
        H[component, :].reshape(snippets.shape[2], snippets.shape[3]),
        cmap='Greys')
    # ax[component].imshow(
    #     mse_snippets,
    #     cmap='Reds', alpha=0.3)
    ax[component].set_ylim(ax[component].get_ylim()[::-1])
    # remove ticks
    ax[component].set_xticks([])
    ax[component].set_yticks([])
fig.show()

# %% draw contounr for each component
from skimage.measure import find_contours

contours = []
for component in range(H.shape[0]):
    temp = H[component, :]
    temp = temp / np.max(temp)
    contour = find_contours(temp.reshape(60, 60), 0.3)
    if len(contour) > 0:
        contours.append(contour[0])
    else:
        contours.append(np.array([]))
# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mse_snippets, cmap='gray')
for contour in contours:
    if contour.size > 0:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=4, alpha=0.1, c="red")
fig.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10), ncols=n_components, nrows=n_runs)
for run in range(n_runs):
    sorted_W = H_all[run, :, :]
    for component in range(n_components):
        ax[run, component].imshow(
            sorted_W[component].reshape(snippets.shape[2], snippets.shape[3]),
            cmap='Greys')
        ax[run, component].set_ylim(ax[run, component].get_ylim()[::-1])
        # remove ticks
        ax[run, component].set_xticks([])
        ax[run, component].set_yticks([])

fig.show()

# %%
import EntropyHub as EH

images = H_all.reshape(H_all.shape[0], H_all.shape[1], 60, 60)
# Normalize all images at once
images_norm = images / np.max(images, axis=(2, 3), keepdims=True) * 255
images_uint8 = images_norm.astype(np.uint8)

entropies = np.zeros((H_all.shape[0], H_all.shape[1]))
# Still need loop for EntropyHub, but with pre-processed data
for r in tqdm.tqdm(range(H_all.shape[0])):
    for c in range(H_all.shape[1]):
        SE2D, Phi1 = EH.SampEn2D(images_uint8[r, c])
        entropies[r, c] = SE2D

indices = np.where(entropies > 0.1)
H_all[indices[0], indices[1], :] = 0
# %%
positive_indices = np.where(entropies <= 0.1)
H_subset = H_all[positive_indices[0], positive_indices[1], :]
# %%

reshaped_W = einops.rearrange(H_all, "r c p -> (r c) p")  # shape: (n_runs*n_components, H*W)

# %% draw contours
from skimage.measure import find_contours

contours = []
for component in range(reshaped_W.shape[0]):
    temp = reshaped_W[component, :]
    temp = temp / np.max(temp)
    contour = find_contours(temp.reshape(60, 60), 0.3)
    if len(contour) > 0:
        contours.append(contour[0])
    else:
        contours.append(np.array([]))

# %%
fig, ax = plt.subplots(figsize=(10, 10))
for contour in contours:
    if contour.size > 0:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=4, alpha=0.1, c="black")
fig.show()

# %%
from scipy.spatial import cKDTree


def _resample_by_arclength(P, n=400, closed=False):
    P = np.asarray(P, float)
    if closed and np.any(P[0] != P[-1]):
        P = np.vstack([P, P[0]])  # close loop
    seg = np.sqrt(((np.diff(P, axis=0)) ** 2).sum(1))
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] == 0:
        return np.repeat(P[:1], n, axis=0)
    t = np.linspace(0, s[-1], n)
    out = np.empty((n, 2))
    j = 0
    for i, ti in enumerate(t):
        while j < len(s) - 2 and s[j + 1] < ti:
            j += 1
        denom = max(s[j + 1] - s[j], 1e-12)
        a = (ti - s[j]) / denom
        out[i] = (1 - a) * P[j] + a * P[j + 1]
    return out


def _resample(P, n=600, closed=False):
    return _resample_by_arclength(P, n=n, closed=closed)


def _mean_symmetric_nn(P1, P2):
    """Mean symmetric nearest-neighbor distance between two point sets."""
    t2 = cKDTree(P2);
    d1, _ = t2.query(P1, k=1)
    t1 = cKDTree(P1);
    d2, _ = t1.query(P2, k=1)
    return 0.5 * (d1.mean() + d2.mean())


def _hausdorff_approx(P1, P2):
    t2 = cKDTree(P2);
    d1, _ = t2.query(P1, k=1)
    t1 = cKDTree(P1);
    d2, _ = t1.query(P2, k=1)
    return float(max(d1.max(), d2.max()))


def cdist_contours(contours, metric="mean_symmetric", n_samples=600, closed=False):
    """
    contours: list of (Ni,2) arrays
    metric: 'mean_symmetric' or 'hausdorff'
    returns: (K x K) matrix of curve-to-curve distances
    """
    # exclude empty contours

    P = [_resample(C, n_samples, closed=closed) for C in contours]
    K = len(P)
    D = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(i + 1, K):
            if metric == "mean_symmetric":
                d = _mean_symmetric_nn(P[i], P[j])
            elif metric == "hausdorff":
                d = _hausdorff_approx(P[i], P[j])
            else:
                raise ValueError("metric must be 'mean_symmetric' or 'hausdorff'")
            D[i, j] = D[j, i] = d
    return D


# %%
contours_nonempty = [c for c in contours if c.size > 0]
dist_matrix = cdist_contours(contours_nonempty, closed=True, metric="hausdorff")

# %%
from scipy.spatial.distance import directed_hausdorff

# compute pairwise directed hausdorff distance between all contours
n_contours = len(contours_nonempty)
dist_matrix_directed = np.zeros((n_contours, n_contours))
for i in range(n_contours):
    for j in range(n_contours):
        if i != j:
            dist_matrix_directed[i, j] = directed_hausdorff(contours_nonempty[i], contours_nonempty[j])[0]

# %%
all_contours = np.empty((n_runs, n_components), dtype=object)
for run in range(n_runs):
    for component in range(n_components):
        temp = H_all[run, component, :]
        if np.all(temp == 0):
            contour = np.array([])
        temp = temp / np.max(temp)
        contour = find_contours(temp.reshape(60, 60), 0.3)
        if len(contour) > 0:
            contour = contour[0]
        all_contours[run, component] = contour

# %%
established_contours = [all_contours[0, i] for i in range(n_components) if len(all_contours[0, i]) > 0]
contour_indices = np.zeros((n_runs, n_components), dtype=int) - 1
valid_first_contours = [i for i in range(n_components) if len(all_contours[0, i]) > 0]
contour_indices[0, valid_first_contours] = np.arange(len(valid_first_contours))
threshold = 2.2
for run in range(1, n_runs):
    for component in range(n_components):
        current_contour = all_contours[run, component]
        if len(current_contour) == 0:
            continue
        dists = [directed_hausdorff(current_contour, ec)[0] for ec in established_contours]
        if min(dists) > threshold:  # threshold to consider a new contour
            established_contours.append(current_contour)
            contour_indices[run, component] = len(established_contours) - 1
        else:
            contour_indices[run, component] = np.argmin(dists)

# %%
clusters, counts = np.unique(contour_indices, return_counts=True)
# %%
nrows, ncols = len(clusters) // 5 + 1, min(len(clusters), 5)
fig, ax = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
for i, cluster in enumerate(clusters):
    r, c = divmod(i, 5)
    if cluster == -1:
        continue
    for run in range(n_runs):
        for component in range(n_components):
            if contour_indices[run, component] == cluster:
                contour = all_contours[run, component]
                if contour.size > 0:
                    ax[r, c].plot(contour[:, 1], contour[:, 0], c="black", linewidth=2, alpha=0.1)
    ax[r, c].set_title(f'Cluster {i + 1} ({counts[i]} contours)')
    ax[r, c].set_xlim(0, 60)
    ax[r, c].set_ylim(60, 0)
    ax[r, c].axis('off')
fig.show()

# %% plot contour mean for each cluster in a single plot
cluster_colours = plt.cm.get_cmap('hsv', len(clusters))
cluster_colours = [cluster_colours(i) for i in range(len(clusters))]
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(mse_snippets, cmap='gray')
for i, cluster in enumerate(clusters):
    if cluster == -1:
        continue
    if counts[i] < 5:
        continue
    all_points = []
    for run in range(n_runs):
        for component in range(n_components):
            if contour_indices[run, component] == cluster:
                contour = all_contours[run, component]
                if contour.size > 0:
                    all_points.append(contour)
                ax.plot(contour[:, 1], contour[:, 0], c=cluster_colours[i], linewidth=2, alpha=0.05)

    if all_points:
        # Resample all contours to have the same number of points before averaging
        resampled_points = [_resample(contour, n=600, closed=True) for contour in all_points]
        all_points_concat = np.stack(resampled_points, axis=0)
        mean_contour = np.mean(all_points_concat, axis=0)
        ax.plot(mean_contour[:, 1], mean_contour[:, 0], c=cluster_colours[i], linewidth=2, label=f'Cluster {i + 1}')
fig.show()
# %% start at contour 0 and search for all contours with distance less than 2. Exclude those from the next search,
# continnue at the next valid contour
visited = set()
clusters = []
threshold = 2
for i in range(n_contours):
    if i not in visited:
        cluster = [i]
        visited.add(i)
        to_visit = [i]
        while to_visit:
            current = to_visit.pop()
            neighbors = np.where(dist_matrix_directed[current] < threshold)[0]
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    cluster.append(neighbor)
                    to_visit.append(neighbor)
        clusters.append(cluster)

# %%
from shapely.geometry import Polygon, LineString
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# pick Polygon if your contours form closed loops; else LineString + .buffer(eps)
eps = 0.1  # “thicken” lines a bit to count near-touching as overlap
geoms = []
for c in contours_nonempty:
    g = Polygon(c) if np.allclose(c[0], c[-1]) else LineString(c).buffer(eps)
    geoms.append(g)

n = len(geoms)
# binary distance: 0 if overlaps/intersects, 1 otherwise
Dbin = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(i + 1, n):
        overlap = geoms[i].intersects(geoms[j])
        d = 0.0 if overlap else 1.0
        Dbin[i, j] = Dbin[j, i] = d

# complete-linkage with t < 1 forces pairwise overlap inside clusters
Z = linkage(squareform(Dbin), method='complete')
labels = fcluster(Z, t=0.5, criterion='distance')  # 0/1 distances → t<1
clusters = [np.where(labels == k)[0].tolist() for k in np.unique(labels)]
# %% plot all clusters in different subplots
rows, cols = len(clusters) // 5 + 1, min(len(clusters), 5)
fig, ax = plt.subplots(rows, cols, figsize=(15, 3 * rows))
for i, cluster in enumerate(clusters):
    r, c = divmod(i, 5)
    for idx in cluster:
        ax[r, c].plot(contours_nonempty[idx][:, 1], contours_nonempty[idx][:, 0], c="black", linewidth=2, alpha=0.1)
    ax[r, c].set_title(f'Cluster {i + 1} ({len(cluster)} contours)')
    ax[r, c].set_xlim(0, 60)
    ax[r, c].set_ylim(60, 0)
    ax[r, c].axis('off')
fig.tight_layout()
fig.show()
# %%
dist_matrix = cdist(H_subset, H_subset, metric='cosine')  # cosine distance matrix
dist_matrix[np.isnan(dist_matrix)] = 1.0  # Replace NaNs with neutral distance
# Force max distance between components from the same run (prevents within-run clustering)
run_ids = positive_indices[0]  # run index for each row in H_subset
for r in np.unique(run_ids):
    idx = np.where(run_ids == r)[0]
    if idx.size > 1:
        dist_matrix[np.ix_(idx, idx)] = 2.0  # max cosine distance (avoid np.inf for HDBSCAN stability)
        np.fill_diagonal(dist_matrix[np.ix_(idx, idx)], 0.0)  # keep self-distance zero
# %%
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(dist_matrix, cmap='viridis')
fig.colorbar(cax)
fig.show()
# %%
cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,  # smallest subunit you care about
                     min_samples=n_cluster_samples,  # raise to be stricter about noise
                     metric='precomputed', )
labels = cl.fit_predict(dist_matrix)  # -1 are outliers
proba = cl.probabilities_  # membership strength
o = cl.outlier_scores_
print(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")

# %%
labels_2d = np.ones((n_runs, n_components), dtype=int) * -1
labels_2d[positive_indices[0], positive_indices[1]] = labels
labels = labels_2d.flatten()
new_probabilities = np.zeros_like(labels_2d, dtype=float)
new_probabilities[positive_indices[0], positive_indices[1]] = cl.probabilities_
new_probabilities = new_probabilities.flatten()

# %%
W_all_re = einops.rearrange(W_all, "r p c -> r c p")  # shape: (n_runs*n_components, n_snippets)

# # %% compress image
# from skimage.filters.rank import entropy
#
# example_entroy = 6.0573047270603695
# target_total = 1 / 36000
# c = 8
# control_image = np.random.permutation(H_all[1, c, :]).reshape(60, 60)
# control_image = control_image / np.max(control_image) * 255
# control_image = control_image.astype(np.uint8)
# test_image = H_all[1, c, :].reshape(60, 60)
# test_image = test_image / np.max(test_image) * 255
# test_image = test_image.astype(np.uint8)
# entropy_image = entropy(test_image, np.ones((5, 5)))
# entropy_shuffle = entropy(control_image, np.ones((5, 5)))
#
# fig, ax = plt.subplots(figsize=(15, 15), nrows=2, ncols=2)
# ax[0, 0].imshow(entropy_image, cmap="Greys")
# ax[0, 1].imshow(test_image, cmap="Greys")
# ax[1, 0].imshow(entropy_shuffle, cmap="Greys")
# ax[1, 1].imshow(control_image, cmap="Greys")
#
# fig.show()
# print(np.sum(entropy_image))
# print(np.sum(entropy_shuffle))


# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(entropies, cmap="viridis", aspect="auto")
fig.show()

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(entropies.flatten(), bins=50)
fig.show()
# %%
fig, ax = plt.subplots(figsize=(20, 10))
cl.condensed_tree_.plot(axis=ax, cmap='viridis', colorbar=True, select_clusters=True)
fig.show()

# %%
fig, ax = plt.subplots(figsize=(20, 10))
cl.single_linkage_tree_.plot(ax, cmap='viridis', colorbar=True)
fig.show()
# %%
fig, ax = plt.subplots(figsize=(10, 10), ncols=np.max(labels) + 3, nrows=1)
for i in range(np.max(labels) + 1):
    cluster_mask = labels == i
    ax[i].imshow(
        np.average(reshaped_W[cluster_mask], axis=0, weights=new_probabilities[cluster_mask]).reshape(snippets.shape[2],
                                                                                                      snippets.shape[
                                                                                                          3]),
        cmap='Greys')
    ax[i].set_title(f'Cluster {i + 1}')
    ax[i].set_ylim(ax[i].get_ylim()[::-1])
cluster_mask = labels >= 0
ax[-2].imshow(np.max(reshaped_W[cluster_mask], axis=0).reshape(snippets.shape[2], snippets.shape[3]), cmap='Greys')
ax[-1].imshow(mse_snippets, cmap="Greys")
ax[-2].set_ylim(ax[-2].get_ylim()[::-1])
ax[-1].set_ylim(ax[-1].get_ylim()[::-1])
fig.show()
# # %%
# test_array = np.mean(reshaped_W[labels == 2], axis=0)
# # %% 2d correlation
# from scipy.signal import correlate2d
# from skimage.feature import match_template
#
# image = test_array.reshape(sta.shape[1], sta.shape[2])
# correlated = correlate2d(image, image, mode='full')
# normalized_corr = correlated / np.max(correlated)
#
# # For normalized correlation that peaks at 1
# normalized_corr = match_template(image, image, pad_input=True)
# fig, ax = plt.subplots(figsize=(10, 10))
# cax = ax.imshow(normalized_corr, cmap='viridis')
# fig.colorbar(cax)
# fig.show()
# print(np.sum(normalized_corr) / (60 * 60))
#
# # %%
# mean_corr = (np.max(normalized_corr, axis=0) + np.max(normalized_corr, axis=1)) / 2
#
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(mean_corr)
# fig.show()
# from scipy.signal import find_peaks
#
# peaks, properties = find_peaks(mean_corr, height=0.9, distance=5, width=1)
# np.sum(
#     mean_corr[np.round(properties["left_ips"].astype(int))[0]:np.round(properties["right_ips"].astype(int))[
#         0]]) / np.sum(mean_corr)
#
# # %%
#
# single_prominence = np.zeros(np.max(labels) + 1)
#
# for cluster in range(np.max(labels) + 1):
#     test_array = np.mean(reshaped_W[labels == cluster], axis=0)
#     image = test_array.reshape(sta.shape[1], sta.shape[2])
#     correlated = correlate2d(image, image, mode='full')
#     normalized_corr = correlated / np.max(correlated)
#
#     # For normalized correlation that peaks at 1
#     normalized_corr = match_template(image, image, pad_input=True)
#     mean_corr = (np.max(normalized_corr, axis=0) + np.max(normalized_corr, axis=1)) / 2
#     peaks, properties = find_peaks(mean_corr, height=0.9, distance=5, width=1)
#     single_prominence[cluster] = np.sum(
#         mean_corr[np.round(properties["left_ips"].astype(int))[0]:np.round(properties["right_ips"].astype(int))[
#             0]]) / np.sum(mean_corr)

# # %%  only keep labels with prominence >= 0.3
# labels[single_prominence[labels] < 0.3] = -1
# # now we have x clusters but the cluster numbers are not consecutive
# unique_labels = np.unique(labels)
# label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
# labels = np.array([label_mapping[label] for label in labels])
# labels_2d = einops.rearrange(labels, "(r c) -> r c", r=n_runs, c=n_components)
# # %%
# fig, ax = plt.subplots(figsize=(20, 10), ncols=np.max(labels) + 2, nrows=1)
# for i in range(np.max(labels)):
#     cluster_mask = labels == i + 1
#     ax[i].imshow(np.mean(reshaped_W[cluster_mask], axis=0).reshape(snippets.shape[2], snippets.shape[3]), cmap='Greys')
#     ax[i].set_title(f'Subunit {i + 1}')
#     ax[i].set_ylim(ax[i].get_ylim()[::-1])
# cluster_mask = labels >= 0
# mean_all = np.zeros((sta.shape[1], sta.shape[2]))
# for i in range(np.max(labels)):
#     cluster_mask = labels == i + 1
#     temp = np.mean(reshaped_W[cluster_mask], axis=0).reshape(snippets.shape[2], snippets.shape[3])
#     mean_all += temp / np.max(temp)
# ax[-2].imshow(mean_all, cmap='Greys')
# ax[-2].set_title(f'All Subunits')
# ax[-1].imshow(mse_snippets, cmap="Greys")
# ax[-2].set_ylim(ax[-2].get_ylim()[::-1])
# ax[-1].set_ylim(ax[-1].get_ylim()[::-1])
# ax[-1].set_title(f'STA MSE')
# fig.show()

# # %% extract Hs
# W_all_re = einops.rearrange(W_all, "r p c -> (r c) p")  # shape: (n_runs*n_components, n_snippets)
# cluster_W = np.zeros((np.max(labels) + 1, W_all_re.shape[1]), dtype=np.float32)
# cluster_W_var = np.zeros_like(cluster_W)
# for cluster in range(np.max(labels) + 1):
#     cluster_mask = labels == cluster
#     cluster_W[cluster] = np.mean(W_all_re[cluster_mask, :], axis=0)
#     cluster_W_var[cluster] = np.var(W_all_re[cluster_mask, :], axis=0)
#
# # %%
# snippets_cluster = np.average(snippets, axis=0, weights=cluster_W[0, :].astype(np.float16))
# %% save data
np.save(save_root / "reshaped_W.npy", reshaped_W)
np.save(save_root / "labels2d.npy", labels_2d)

# %% save cell_params
cell_params.save_json()
