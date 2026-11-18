"""
Convert an existing noise_data.nc (made with reduce_all.py) to the v2 format,
without recomputing the STAs, covariance or RMS maps.

What it does
------------
1. rms and cm_most_important: swap x and y back. reduce_all.py pasted every map
   with x and y exchanged; on the square grids used here, swapping the whole grid
   back gives exactly what reduce_all_v2.py produces.
2. Recompute the polar outlines (reduce_all_v2.compute_circular_outlines, 1-px radius
   steps) and the RF stats (summary_v2.compute_rf_stats, areas in true mm²).
3. Mark the file with attrs["rf_map_frame"] = "stimulus_v2".

Everything else (quality, positions, sta_path, sta_single_pixel) is copied as is.
The original file is never modified: the result is written as noise_data_v2.nc
next to it (or to --output).

Settings (channel names, pixel sizes, quality threshold, cut size, degree bins)
are read from the JSON configs the pipeline saved in the same folder, so the
recording's overview/Windows paths are not needed.

Usage
-----
    python -m combined_analysis.convert_noise_data_v2 <noise_analysis folder> [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from combined_analysis.reduce_all_v2 import RF_MAP_FRAME, compute_circular_outlines
from stats.summary_v2 import compute_rf_stats

MAP_VARIABLES = ("rms", "cm_most_important")
OUTLINE_VARIABLES = (
    "center_outline_um",
    "surround_outline_um",
    "in_out_outline_um",
    "center_std",
    "surround_std",
)


def read_settings(folder: Path) -> dict:
    recording = json.loads((folder / "recording_config.json").read_text())
    collapse = json.loads((folder / "collapse_2d_config.json").read_text())
    circular = json.loads((folder / "circular_reduction_config.json").read_text())

    def xy(size_values: dict) -> tuple[int, int]:
        labels = size_values["coords"]["dimension"]["data"]
        values = dict(zip(labels, size_values["data"]))
        return int(values["x"]), int(values["y"])

    cut_sizes = [xy(v) for v in collapse["cut_size_px"].values()]
    return dict(
        channel_names=list(recording["channel_names"]),
        min_pixel_size=min(
            float(c["pixel_size"]) for c in recording["channel_configs"].values()
        ),
        threshold=float(collapse["threshold"]),
        x_cut_size=max(c[0] for c in cut_sizes),
        y_cut_size=max(c[1] for c in cut_sizes),
        degree_bins=int(circular["degree_bins"]),
    )


def swap_map_axes(ds: xr.Dataset) -> xr.Dataset:
    """Undo the reduce_all.py x/y swap of the stored RF maps."""
    if ds.sizes["x"] != ds.sizes["y"] or not np.allclose(ds["x"].values, ds["y"].values):
        raise ValueError(
            "Expected a square grid with identical x and y coordinates; got "
            f"x={ds.sizes['x']}, y={ds.sizes['y']}. Rerun the pipeline with "
            "reduce_all_v2.py instead."
        )
    for name in MAP_VARIABLES:
        if name not in ds:
            continue
        da = ds[name].transpose("cell_index", "channel", "y", "x")
        # ORIENTATION [DATA FIX] swap x and y back (undo the old reduce_all.py swap). Afterwards the maps are
        #   in stimulus coordinates, like kernel.npy and positions. No other rotation.
        ds[name] = (da.dims, np.swapaxes(da.values, -1, -2), da.attrs)
    return ds


def convert(folder: Path, output: Path | None = None, pixel_size_um: float | None = None) -> Path:
    """
    pixel_size_um: stimulus pixel size (µm) for the v2 file. Default: the
    recording's pixel size from recording_config.json. If it differs from the
    pixel size the original file was built with (read from its x grid), all µm
    values are relabelled exactly (x/y grid, outlines, sizes); the maps
    themselves are in stimulus pixels and do not change.
    """
    folder = Path(folder)
    source = folder / "noise_data.nc"
    output = Path(output) if output else folder / "noise_data_v2.nc"
    if output.resolve() == source.resolve():
        raise ValueError("Refusing to overwrite the original noise_data.nc.")

    settings = read_settings(folder)
    print(f"Loading {source}")
    ds = xr.load_dataset(source)
    if ds.attrs.get("rf_map_frame") == RF_MAP_FRAME:
        raise ValueError(f"{source} is already in the v2 frame.")

    ds = swap_map_axes(ds)

    # Stimulus pixel size: the one the file was built with (its x grid spacing) and
    # the one wanted for v2. A difference is an exact relabelling of µm values.
    built_with = float(np.round(np.median(np.abs(np.diff(ds["x"].values))), 6))
    wanted = float(pixel_size_um) if pixel_size_um is not None else settings["min_pixel_size"]
    if not np.isclose(built_with, wanted):
        factor = wanted / built_with
        print(f"Relabelling stimulus pixel size {built_with} -> {wanted} µm (x {factor:.4f})")
        ds = ds.assign_coords(x=ds["x"] * factor, y=ds["y"] * factor)
        ds.attrs["rescaled_pixel_size_from_um"] = built_with
    ds.attrs["stimulus_pixel_size_um"] = wanted

    ds = ds.drop_vars([v for v in OUTLINE_VARIABLES if v in ds] + ["radius"], errors="ignore")
    ds = compute_circular_outlines(
        ds,
        channel_names=settings["channel_names"],
        threshold=settings["threshold"],
        x_cut_size=settings["x_cut_size"],
        y_cut_size=settings["y_cut_size"],
        degree_bins=settings["degree_bins"],
        min_pixel_size=wanted,
    )
    ds = compute_rf_stats(ds, settings["channel_names"])
    ds.attrs["rf_map_frame"] = RF_MAP_FRAME
    ds.attrs["converted_from"] = str(source)

    print(f"Saving {output}")
    ds.to_netcdf(output, engine="netcdf4")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="noise_analysis folder")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="stimulus pixel size in µm for the v2 file (default: recording_config.json)")
    args = parser.parse_args()
    convert(args.folder, args.output, args.pixel_size)
