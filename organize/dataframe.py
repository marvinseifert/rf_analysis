import polars as pl
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


class Noise_store:
    def __init__(self, settings_dict, df=None, nr_cells=None):
        self.settings_dict = settings_dict
        if df is not None:
            self.df = df
            self.nr_cells = df["cell_index"].n_unique()
            self.schema = self.df.schema
        elif nr_cells is not None:
            self.nr_cells = nr_cells
            self.schema = {
                "cell_index": pl.UInt32,
                "channel": pl.Categorical,
                "sta_path": pl.Utf8,
            }
            self.df = self.create_df(settings_dict["data_dict"]["channels"])
        else:
            exit("Either a dataframe or the number of cells must be provided")

    def create_df(self, channels):
        cat_dtype = pl.Enum(channels)

        channels_complete = []
        cell_indices = []
        paths = []
        for channel_idx, channel in enumerate(channels):
            channels_complete.extend([channel] * self.nr_cells)
            cell_indices.extend(np.arange(self.nr_cells))
            paths.extend(
                str(
                    self.settings_dict["data_dict"]["project_root"]
                    / self.settings_dict["data_dict"]["channel_paths"][channel_idx]
                    / f"cell_{idx}"
                    / self.settings_dict["data_dict"]["sta_file"]
                )
                for idx in range(self.nr_cells)
            )
        data_dict = {
            "channel": pl.Series(
                channels_complete,
                dtype=cat_dtype,
            ),
            "cell_index": pl.Series(cell_indices, dtype=pl.UInt32),
            "sta_path": pl.Series(paths, dtype=pl.Utf8),
        }

        df = pl.DataFrame(data=data_dict, schema=self.schema)
        return df

    def save(self):
        self.df.write_parquet(
            self.settings_dict["data_dict"]["project_root"]
            / self.settings_dict["data_dict"]["output_folder"]
            / "noise_df"
        )
        with open(
                self.settings_dict["data_dict"]["project_root"]
                / self.settings_dict["data_dict"]["output_folder"]
                / "settings.pkl",
                "wb",
        ) as f:
            pickle.dump(self.settings_dict, f)

    @classmethod
    def load(cls, analysis_folder):
        with open(analysis_folder / "settings.pkl", "rb") as f:
            settings_dict = pickle.load(f)
        df = pl.read_parquet(analysis_folder / "noise_df")
        obj = cls(settings_dict, df)
        return obj


class RF_plotting:

    def __init__(self, settings_dict, df=None):
        self.settings_dict = settings_dict
        if df is not None:
            self.df = df
            self.nr_cells = len(df.select(pl.col("cell_index").unique()).collect())
            self.schema = self.df.schema
        else:
            exit("Either a dataframe or the number of cells must be provided")

    def most_important(self, cells: list, channels: list, channel_cmaps=None):
        if channels[0] == "all":
            channels = self.settings_dict["data_dict"]["channels"]
            if not channel_cmaps:
                channel_cmaps = ["seismic"] * len(channels)
        nr_cells = len(cells)
        fig, ax = plt.subplots(
            nrows=nr_cells, ncols=len(channels), figsize=(10, 2 * nr_cells)
        )
        ax = np.atleast_2d(
            ax,
        )
        if len(channels) == 1:
            ax = ax.T
        cells_df = self.df.filter(pl.col("channel").is_in(channels)).filter(
            pl.col("cell_index").is_in(cells)
        )

        for cell_idx, cell in enumerate(cells):
            cell_df = cells_df.filter(pl.col("cell_index") == cell).collect()
            for channel_idx, channel in enumerate(channels):
                cm_most_important = np.reshape(
                    cell_df.filter(pl.col("channel") == channel)["cm_most_important"],
                    (
                        self.settings_dict["noise"]["cut_size"],
                        self.settings_dict["noise"]["cut_size"],
                    ),
                )
                vmax = np.max(np.abs(cm_most_important))
                ax[cell_idx, channel_idx].imshow(
                    cm_most_important,
                    cmap=channel_cmaps[channel_idx],
                    label=channel,
                    vmin=-vmax,
                    vmax=vmax,
                )
        for channel_idx, channel in enumerate(channels):
            ax[0, channel_idx].title.set_text(f"{channel} nm")
        scalebar = ScaleBar(
            self.settings_dict["noise"]["pixel_size"], "um", fixed_value=100
        )
        ax[0, 0].add_artist(scalebar)

        return fig, ax

    def center_outline(self, cells: list, channels: list, thresholding=True):
        if channels[0] == "all":
            channels = self.settings_dict["data_dict"]["channels"]
        nr_cells = len(cells)
        degrees = np.arange(
            0, 360, self.settings_dict["circular_reduction"]["degree_step"]
        )
        degrees = np.concatenate([degrees, [degrees[0]]])
        fig, ax = plt.subplots(
            nrows=nr_cells,
            ncols=len(channels),
            subplot_kw={"projection": "polar"},
            sharex=True,
            sharey=True,
            figsize=(10, 2 * nr_cells),
        )
        ax = np.atleast_2d(ax)
        cells_df = self.df.filter(pl.col("channel").is_in(channels)).filter(
            pl.col("cell_index").is_in(cells)
        )

        for cell_idx, cell in enumerate(cells):
            cell_df = cells_df.filter(pl.col("cell_index") == cell).collect()
            for channel_idx, channel in enumerate(channels):
                center_outline = (
                    cell_df.filter(pl.col("channel") == channel)[
                        "center_outline"
                    ].to_numpy(),
                )[0].flatten()
                if (
                        cell_df.filter(pl.col("channel") == channel)["quality"]
                        < self.settings_dict["thresholding"]["threshold"]
                )[0] and thresholding:
                    continue

                center_outline = np.concatenate([center_outline, [center_outline[0]]])
                ax[cell_idx, channel_idx].plot(np.radians(degrees), center_outline)
                ax[cell_idx, channel_idx].set_theta_zero_location("E")
                ax[cell_idx, channel_idx].set_theta_direction(-1)

        for channel_idx, channel in enumerate(channels):
            ax[0, channel_idx].title.set_text(f"{channel} nm")

        return fig, ax

    def stas(self, cells: list, channels: list, thresholding=True):
        if channels[0] == "all":
            channels = self.settings_dict["data_dict"]["channels"]
        nr_cells = len(cells)
        fig, ax = plt.subplots(
            nrows=nr_cells, ncols=len(channels), figsize=(10, 2 * nr_cells)
        )
        ax = np.atleast_2d(ax)
        cells_df = self.df.filter(pl.col("channel").is_in(channels)).filter(
            pl.col("cell_index").is_in(cells)
        )

        for cell_idx, cell in enumerate(cells):
            cell_df = cells_df.filter(pl.col("cell_index") == cell).collect()
            for channel_idx, channel in enumerate(channels):
                sta_single = (
                    cell_df.filter(pl.col("channel") == channel)["sta_single"]
                ).to_numpy()
                if (
                        cell_df.filter(pl.col("channel") == channel)["quality"]
                        < self.settings_dict["thresholding"]["threshold"]
                )[0] and thresholding:
                    continue

                for sta_idx, sta in enumerate(sta_single):
                    ax[cell_idx, channel_idx].plot(sta)

        for channel_idx, channel in enumerate(channels):
            ax[0, channel_idx].title.set_text(f"{channel} nm")

        return fig, ax

    @classmethod
    def load(cls, analysis_folder):
        with open(analysis_folder / "settings.pkl", "rb") as f:
            settings_dict = pickle.load(f)
        df = pl.scan_parquet(analysis_folder / "noise_df")
        obj = cls(settings_dict, df)
        return obj
