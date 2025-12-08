#!/usr/bin/env python3
"""
RF Clustering Module

This module provides functions to cluster receptive field data loaded from polars dataframes.
It supports different clustering methods and visualization of the clustering results.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib_scalebar.scalebar import ScaleBar
from polarspike import colour_template
from pickle import load as load_pickle
from normalization.normalize_sta import zscore_sta
from organize.dataframe import Noise_store, RF_plotting
from scipy.spatial.distance import pdist, squareform


class RFClustering:
    """
    A class for clustering receptive field data.
    """

    def __init__(self, paths: list = None):
        """
        Initialize the RFClustering object.

        Parameters
        ----------
        settings : dict, optional
            Settings dictionary with parameters for clustering and visualization.
        """
        self.paths = paths
        channel_dict_temp = {}
        for path in paths:
            with open(path / "settings.pkl", "rb") as f:
                settings_dict = load_pickle(f)
            channel_dict_temp[path.parts[-3]] = settings_dict["data_dict"]["channels"]

        # print channel_dict_temp in a readable way
        print("Channels found in each path:")
        for key, value in channel_dict_temp.items():
            print(f"{key}: {value}")

        self.columns_for_clustering = ["sta_single", "center_outline", "surround_outline", "in_out_outline"]

        # Check how many channels are in each path

        self.scaler = StandardScaler()
        self.threshold = 20

        # Initialize data containers
        self.dfs = None
        self.all_data = None
        self.all_data_scaled = None
        self.cells_recordings = None
        self.data_sizes = None
        self.data_labels = None
        self.transformed = None
        self.labels = None
        self.external_data = None
        self.bins = None
        self.distance_matrix = None
        self.good_cells = None
        self.channel_colours = None

    def load_rf_data(self):
        """
        Load receptive field data from the specified paths.

        Parameters
        ----------
        paths : list
            List of paths to the noise analysis folders.
        
        Returns
        -------
        self
        """
        dfs = []
        for path in self.paths:
            noise_path = Path(path)
            noise_store = Noise_store.load(noise_path)
            noise_store.df = noise_store.df.with_columns(
                recording=pl.lit(f"{path.parts[-3]}_p0")
            )
            dfs.append(noise_store.df)

        self.dfs = pl.concat(dfs)

    def load_external_data(self, psth_path, bins_path, cells_path):
        """
        Load external PSTH data to be combined with RF data for clustering.

        Parameters
        ----------
        psth_path : str
            Path to the PSTH data file (.npy)
        bins_path : str
            Path to the bins data file (.npy)
        cells_path : str
            Path to the cells data file (.npy)
        
        Returns
        -------
        self
        """
        psths = np.load(psth_path, allow_pickle=True)
        self.bins = np.load(bins_path, allow_pickle=True)
        psths_cells = np.load(cells_path, allow_pickle=True)

        # Save in dataframe
        multiindex = pd.MultiIndex.from_arrays(
            (psths_cells[:, 0], psths_cells[:, 1].astype(int)),
            names=["recording", "cell_index"],
        )
        self.external_data = pd.DataFrame(data=psths, index=multiindex)

        return self

    def prepare_data(self, include_external_data: bool = False, scale: bool = True):
        """

        """
        extracted_data = {}
        group_cols = ["recording", "cell_index"]
        for col_idx, col in enumerate(self.columns_for_clustering):
            df_temp = (self.dfs.explode(pl.col(col))
                       .group_by(group_cols, maintain_order=True)
                       .agg(pl.col(col).alias("export")))
            extracted_data[col] = np.vstack(df_temp["export"].to_numpy())
        # save cells_recordings
        self.cells_recordings = np.vstack((
            (df_temp["recording"].to_numpy()),
            (df_temp["cell_index"].to_numpy())
        )).T

        # apply normalization to in_out_outline and surround_outline
        for col in ["center_outline", "in_out_outline", "surround_outline"]:
            try:
                extracted_data[col] = extracted_data[col] / np.max(np.abs(extracted_data[col]), axis=1, keepdims=True)
            except ValueError:
                pass
        # normalize sta_single
        # the original array needs to be split in half along axis 1
        sta_data = extracted_data["sta_single"]
        half_point = sta_data.shape[1] // 2
        sta_first_half = sta_data[:, :half_point]
        sta_second_half = sta_data[:, half_point:]
        sta_first_half_norm = zscore_sta(sta_first_half)
        sta_second_half_norm = zscore_sta(sta_second_half)
        extracted_data["sta_single"] = np.hstack((sta_first_half_norm, sta_second_half_norm))

        self.all_data = np.hstack(
            [extracted_data[col] for col in self.columns_for_clustering])
        # fill nan with 0
        self.all_data = np.nan_to_num(self.all_data)
        self.all_data_scaled = self.scaler.fit_transform(self.all_data)
        self.good_cells = (
                self.dfs.group_by(["recording", "cell_index"], maintain_order=True).agg(pl.col("quality").max())[
                    "quality"].to_numpy() > self.threshold)

    def run_pca(self, n_components=10):
        """
        Perform PCA on the data.

        Parameters
        ----------
        n_components : int, optional
            Number of PCA components to compute.
            
        Returns
        -------
        self
        """
        pca = PCA(n_components=n_components, whiten=True)
        pca.fit(self.all_data_scaled[self.good_cells])
        self.transformed = pca.transform(self.all_data_scaled[self.good_cells])

        # Print variance explained per component
        print("Variance explained by PCA components:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"PC {i + 1}: {var:.4f}")

    def plot_pca(self):
        """
        Plot the first two PCA components.
        
        Returns
        -------
        tuple
            Figure and axis objects.
        """
        if self.transformed is None:
            self.run_pca()

        fig, ax = plt.subplots(figsize=(10, 8))
        unique_rec = np.unique(self.cells_recordings[:, 0], return_inverse=True)[1][self.good_cells]
        scatter = ax.scatter(self.transformed[:, 0], self.transformed[:, 1], c=unique_rec, cmap='tab10')

        # Add legend for recordings
        unique_recordings = np.unique(self.cells_recordings[:, 0][self.good_cells])
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      label=rec, markerfacecolor=plt.cm.tab10(i % 10),
                                      markersize=8)
                           for i, rec in enumerate(unique_recordings)]
        ax.legend(handles=legend_elements, title="Recordings")

        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("PCA of RF data")

        return fig, ax

    def calculate_distance_matrix(self, distance_metric: str = 'cosine'):
        """
        Calculate the distance matrix for the scaled data.

        Returns
        -------
        np.ndarray
            Distance matrix.
        """
        if self.all_data_scaled is None:
            raise ValueError("Data not prepared. Run prepare_data first.")

        # Calculate distance matrix
        self.distance_matrix = squareform(pdist(self.all_data_scaled[self.good_cells], metric=distance_metric))

    def plot_distance_matrix(self, **kwargs):
        """
        Plot the distance matrix as a heatmap.

        Parameters
        ----------
        distance_matrix : np.ndarray, optional
            Precomputed distance matrix. If None, it will be calculated.

        Returns
        -------
        tuple
            Figure and axis objects.
        """
        if self.distance_matrix is None:
            self.calculate_distance_matrix(**kwargs)

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(self.distance_matrix, cmap='viridis')
        fig.colorbar(cax)

        ax.set_title("Distance Matrix Heatmap")
        ax.set_xlabel("Cells")
        ax.set_ylabel("Cells")

        return fig, ax

    def cluster_data(self, method='affinity', n_clusters=10, distance_metric: str = 'cosine'):
        """
        Cluster the data.

        Parameters
        ----------
        method : str, optional
            Clustering method to use ('affinity' or 'agglomerative')
        n_clusters : int, optional
            Number of clusters for agglomerative clustering.
        distance_metric : str, optional
            Distance metric to use for clustering.
            
        Returns
        -------
        self
        """
        # Calculate distance matrix
        self.calculate_distance_matrix(distance_metric=distance_metric)

        if method.lower() == 'affinity':
            # Use similarity matrix for affinity propagation
            similarity_matrix = -(self.distance_matrix ** 2)
            clustering = AffinityPropagation(random_state=5, max_iter=10000).fit(similarity_matrix)
            self.labels = clustering.labels_
            print(f"Affinity Propagation found {np.max(self.labels) + 1} clusters")

        elif method.lower() == 'agglomerative':
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric='precomputed', linkage='average'
            )
            self.labels = clustering.fit_predict(self.distance_matrix)
            print(f"Agglomerative clustering with {n_clusters} clusters completed")

        else:
            raise ValueError("Method must be 'affinity' or 'agglomerative'")

    def plot_clusters_pca(self):
        """
        Plot clusters in PC space.
        
        Returns
        -------
        tuple
            Figure and axis objects.
        """
        if self.labels is None:
            raise ValueError("No clustering results available. Run cluster_data first.")

        if self.transformed is None:
            self.run_pca()

        # Define colors based on number of clusters
        colors = plt.cm.get_cmap('tab10', np.max(self.labels) + 2)

        fig, ax = plt.subplots(figsize=(10, 8))
        for cluster in np.unique(self.labels):
            if cluster == -1:  # Noise points in some clustering methods
                ax.scatter(
                    self.transformed[self.labels == cluster, 0],
                    self.transformed[self.labels == cluster, 1],
                    label=f"Noise",
                    c='gray',
                    marker='x'
                )
            else:
                c_color = cluster + 1 if cluster >= 0 else 0
                ax.scatter(
                    self.transformed[self.labels == cluster, 0],
                    self.transformed[self.labels == cluster, 1],
                    label=f"Cluster {cluster}",
                    c=[colors(c_color)],
                )

        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("Clusters in PC space")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        return fig, ax

    def plot_cm_most_important(self, max_cells_per_cluster=10, save_path=None, cmap="viridis"):
        """
        Plot cm_most_important for example cells from each cluster.
        
        Parameters
        ----------
        max_cells_per_cluster : int, optional
            Maximum number of cells to show per cluster
        save_path : str, optional
            Path to save the figures
            
        Returns
        -------
        list
            List of figures, one per cluster
        """
        if self.labels is None:
            raise ValueError("No clustering results available. Run cluster_data first.")

        clusters = np.unique(self.labels)
        figs = []

        for cluster in clusters:
            if cluster == -1 or np.sum(self.labels == cluster) < 2:  # Skip noise or small clusters
                continue

            cell_list = self.cells_recordings[self.good_cells][self.labels == cluster, 1]
            recording_list = self.cells_recordings[self.good_cells][self.labels == cluster, 0]

            cluster_df = self.dfs.filter(
                pl.col("cell_index").is_in(cell_list.astype(int))
                & pl.col("recording").is_in(recording_list)
            )
            # Select the cells with highest quality (unique by cell_index and recording)
            quality_df = (cluster_df
                          .group_by(["recording", "cell_index"])
                          .agg(pl.col("quality").max())
                          .sort("quality", descending=True)
                          .filter(pl.col("quality").is_not_nan())
                          .head(max_cells_per_cluster))

            # get cell indices and recording names for the top cells in this cluster
            cell_list = quality_df["cell_index"].to_numpy().astype(int)
            recording_list = quality_df["recording"].to_numpy()

            fig, ax = plt.subplots(
                nrows=len(self.dfs["channel"].unique()),
                ncols=len(cell_list),
                figsize=(20, 10)
            )

            # Ensure ax is 2D array even when there's only one channel
            ax = np.atleast_2d(ax)

            # # Add scalebar
            # scalebar = ScaleBar(self.pixel_size, "um", fixed_value=100)

            # Determine common vmin/vmax for color scaling
            random_df = self.dfs.filter(
                (pl.col("cell_index").is_in(cell_list))
                & (pl.col("recording").is_in(recording_list))
            )
            vmax = random_df.with_columns(
                pl.col("cm_most_important").arr.max().alias("max_val"),
                pl.col("cm_most_important").arr.min().alias("min_val")
            ).with_columns(
                pl.max("max_val").alias("global_max"),
                pl.min("min_val").alias("global_min")
            ).select(pl.max("global_max")).to_numpy()[0, 0]
            vmin = 0.0

            for idx, cell_recording in enumerate(zip(cell_list, recording_list)):
                # Filter data for this cell and recording
                df_sub = self.dfs.filter(
                    (
                            (pl.col("cell_index") == int(cell_recording[0]))
                            & (pl.col("recording") == cell_recording[1])
                    )
                )

                for channel_idx, channel in enumerate(self.dfs["channel"].unique()):
                    try:
                        # Get receptive field data for this cell and channel
                        cm_most_important = df_sub.filter(pl.col("channel") == channel)[
                            "cm_most_important"
                        ].to_numpy()[0]

                        cm_most_important = np.reshape(cm_most_important, (200, 200)) / np.max(cm_most_important)
                        median_val = np.median(cm_most_important)
                        centered = cm_most_important - median_val
                        scale = np.max(np.abs(centered))
                        cm_most_important = np.zeros_like(centered) if scale == 0 else centered / scale

                        # Plot receptive field
                        cax = ax[channel_idx, idx].imshow(
                            cm_most_important,
                            cmap=cmap,
                            label=channel,
                            vmin=-1,
                            vmax=1
                        )
                        # add colorbar to last column
                        if idx == len(cell_list) - 1:
                            fig.colorbar(cax, ax=ax[channel_idx, idx], fraction=0.046, pad=0.04)

                        # Add row and column labels
                        if idx == 0:
                            ax[channel_idx, idx].set_ylabel(channel)
                        if channel_idx == 0:
                            ax[channel_idx, idx].set_title(f"Cell {cell_recording[0]}")
                    except (IndexError, KeyError):
                        print(
                            f"Could not plot cell {cell_recording[0]} from recording {cell_recording[0]} for channel {channel}")

            # Add scalebar to first plot
            # ax[0, 0].add_artist(scalebar)

            plt.suptitle(f"Cluster {cluster} - Receptive Fields", fontsize=16)
            plt.tight_layout()

            if save_path:
                fig.savefig(Path(save_path) / f"cluster_{cluster}_rfs.svg")

            figs.append(fig)

        return figs

    def plot_stas(self, save_path=None):
        """
        Plot STAs for each cluster.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        tuple
            Figure and axis objects
        """
        if self.labels is None:
            raise ValueError("No clustering results available. Run cluster_data first.")

        clusters = np.unique(self.labels)

        fig, ax = plt.subplots(
            nrows=len(self.dfs["channel"].unique()),
            ncols=len(clusters),
            figsize=(10 * len(clusters), 10)
        )

        # Ensure ax is 2D array even when there's only one cluster or channel
        ax = np.atleast_2d(ax)

        for cluster_idx, cluster in enumerate(clusters):
            if cluster == -1:  # Skip noise points
                continue

            cl_cells = self.cells_recordings[self.good_cells][self.labels == cluster, 1]
            recording_list = self.cells_recordings[self.good_cells][self.labels == cluster, 0]

            sub_df = self.dfs.filter(
                pl.col("cell_index").is_in(cl_cells.astype(int))
                & pl.col("recording").is_in(recording_list)
            )

            for channel_idx, channel in enumerate(self.dfs["channel"].unique()):
                print(channel_idx)
                try:
                    stas = np.vstack(sub_df.filter(pl.col("channel") == channel)["sta_single"])
                    stas_mean = np.mean(stas, axis=0)
                    sta_scaled = zscore(stas_mean, axis=0)

                    # Plot mean STA
                    ax[channel_idx, cluster_idx].plot(sta_scaled, c=self.channel_colours[channel_idx], linewidth=2)

                    # Plot individual STAs
                    for sta in stas:
                        ax[channel_idx, cluster_idx].plot(
                            zscore(sta), c=self.channel_colours[channel_idx], alpha=0.1, linewidth=0.5
                        )

                    ax[channel_idx, cluster_idx].set_title(f"Cluster {cluster}")
                except (IndexError, ValueError):
                    print(f"Could not plot STA for cluster {cluster}, channel {channel}")

        ax[0, 0].set_ylabel("z-score")
        plt.tight_layout()

        if save_path:
            fig.savefig(Path(save_path) / "cluster_stas.png")

        return fig, ax

    def plot_center_outlines(self, save_path=None):
        """
        Plot center outlines for each cluster.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        tuple
            Figure and axis objects
        """
        if self.labels is None:
            raise ValueError("No clustering results available. Run cluster_data first.")

        clusters = np.unique(self.labels)

        fig, ax = plt.subplots(
            nrows=len(self.dfs["channel"].unique()),
            ncols=len(clusters),
            figsize=(max(5 * len(clusters), 10), 10),
            subplot_kw={"projection": "polar"}
        )

        # Ensure ax is 2D array even when there's only one cluster or channel
        ax = np.atleast_2d(ax)

        # Set up degree angles for polar plot
        degrees = np.arange(0, 360, 10)  # Assuming 15 degree steps
        degrees = np.concatenate([degrees, [degrees[0]]])

        for cluster_idx, cluster in enumerate(clusters):
            if cluster == -1:  # Skip noise points
                continue

            cl_cells = self.cells_recordings[self.good_cells][self.labels == cluster, 1]
            recording_list = self.cells_recordings[self.good_cells][self.labels == cluster, 0]

            sub_df = self.dfs.filter(
                pl.col("cell_index").is_in(cl_cells.astype(int))
                & pl.col("recording").is_in(recording_list)
            )

            for channel_idx, channel in enumerate(self.dfs["channel"].unique()):
                try:
                    # Get center outlines for this cluster and channel
                    cell_outlines = np.vstack(sub_df.filter(pl.col("channel") == channel)["center_outline"])

                    # Plot individual outlines
                    for outline in cell_outlines:
                        if np.sum(outline) == 0 or np.any(np.isnan(outline)):
                            continue

                        # Complete the circle by adding first point at the end
                        outline_complete = np.concatenate([outline, [outline[0]]])
                        ax[channel_idx, cluster_idx].plot(
                            np.radians(degrees),
                            outline_complete,
                            alpha=0.3,
                            c=self.channel_colours[channel_idx]
                        )

                    # Plot mean outline
                    mean_outline = np.nanmean(cell_outlines, axis=0)
                    mean_outline_complete = np.concatenate([mean_outline, [mean_outline[0]]])
                    ax[channel_idx, cluster_idx].plot(
                        np.radians(degrees),
                        mean_outline_complete,
                        linewidth=2,
                        c=self.channel_colours[channel_idx]
                    )

                    # Configure polar plot
                    ax[channel_idx, cluster_idx].set_theta_zero_location("E")
                    ax[channel_idx, cluster_idx].set_theta_direction(-1)
                    ax[channel_idx, cluster_idx].set_title(f"Cluster {cluster}")
                except (IndexError, ValueError):
                    print(f"Could not plot center outlines for cluster {cluster}, channel {channel}")

        plt.tight_layout()

        if save_path:
            fig.savefig(Path(save_path) / "cluster_center_outlines.png")

        return fig, ax

    def plot_psth(self, cluster, save_path=None):
        """
        Plot PSTH for a specific cluster.
        
        Parameters
        ----------
        cluster : int
            Cluster index to plot
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        tuple
            Figure and axis objects
        """
        if self.labels is None:
            raise ValueError("No clustering results available. Run cluster_data first.")

        if self.external_data is None or self.bins is None:
            raise ValueError("External PSTH data not loaded. Use load_external_data first.")

        # Create a color template for stimulus visualization
        CT = colour_template.Colour_template()
        CT.pick_stimulus("FFF_6_MC")  # Adjust based on your stimulus

        # Create figure with one big row and one small row for stimulus
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(20, 10),
            gridspec_kw={"height_ratios": [10, 1]},
        )

        # Get PSTH data for this cluster
        new_mulitindex = pd.MultiIndex.from_arrays(
            (self.cells_recordings[:, 0], self.cells_recordings[:, 1].astype(int)),
            names=["recording", "cell_index"],
        )

        try:
            psth_data = self.external_data.loc[new_mulitindex].to_numpy()
            psth_sub = psth_data[self.labels == cluster]
            psth_mean = np.mean(psth_sub, axis=0)

            # Plot mean PSTH
            axs[0].plot(self.bins[:-1], psth_mean, c="k", linewidth=2)

            # Plot individual PSTHs
            for psth in psth_sub:
                axs[0].plot(self.bins[:-1], psth, c="k", alpha=0.2)

            # Add stimulus visualization
            fig = CT.add_stimulus_to_plot(fig, [2] * 12)  # Adjust based on your stimulus pattern

            # Add labels
            axs[0].set_title(f"Cluster {cluster} - PSTH", fontsize=16)
            axs[0].set_ylabel("Firing rate (Hz)")
            axs[1].set_xlabel("Time (s)")

            if save_path:
                fig.savefig(Path(save_path) / f"cluster_{cluster}_psth.png")

            return fig, axs

        except (KeyError, ValueError):
            print(f"Could not plot PSTH for cluster {cluster}. Check if external data is correctly loaded.")
            return None, None
