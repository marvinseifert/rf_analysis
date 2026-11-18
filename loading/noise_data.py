"""
loading/noise_data.py — the one way to open a recording's RF dataset (v2).

    from loading.noise_data import load_noise_data
    ds = load_noise_data(".../Phase_00/noise_analysis")            # folder
    ds = load_noise_data(".../noise_analysis/noise_data_v2.nc")      # v2 file
    ds = load_noise_data(".../noise_analysis/noise_data.nc")         # old name -> uses noise_data_v2.nc

Always returns noise_data_v2.nc and checks that its RF maps are in the v2 stimulus
frame (attrs["rf_map_frame"] == "stimulus_v2"). It never opens the old v1
noise_data.nc (x and y swapped, outlines 1.41x too large); if only that exists it
stops and prints the command that makes the v2 copy.
"""
from __future__ import annotations

from pathlib import Path

import xarray as xr

NOISE_DATA_V2_NAME = "noise_data_v2.nc"
NOISE_DATA_V1_NAME = "noise_data.nc"
RF_MAP_FRAME_V2 = "stimulus_v2"  # same value as combined_analysis.reduce_all.RF_MAP_FRAME


def noise_data_v2_path(path: str | Path) -> Path:
    """noise_data_v2.nc for a noise_analysis folder or any noise_data*.nc path in it."""
    path = Path(path)
    if path.suffix == ".nc":
        if path.name != NOISE_DATA_V2_NAME:
            print(f"load_noise_data: {path.name} -> {NOISE_DATA_V2_NAME} (v2)")
        path = path.parent / NOISE_DATA_V2_NAME
    else:
        path = path / NOISE_DATA_V2_NAME
    if not path.exists():
        hint = ""
        if (path.parent / NOISE_DATA_V1_NAME).exists():
            hint = (f"\nOnly the old v1 {NOISE_DATA_V1_NAME} is there. Make the v2 copy with:\n"
                    f"    poetry run python -m combined_analysis.convert_noise_data_v2 \"{path.parent}\"")
        raise FileNotFoundError(f"No {NOISE_DATA_V2_NAME} in {path.parent}.{hint}")
    return path


def require_v2(ds: xr.Dataset, source: str = "dataset") -> xr.Dataset:
    """Stop with a clear message if the RF maps are not in the v2 stimulus frame."""
    frame = ds.attrs.get("rf_map_frame")
    if frame != RF_MAP_FRAME_V2:
        raise ValueError(
            f"{source} has rf_map_frame={frame!r}, expected {RF_MAP_FRAME_V2!r}: its RF maps "
            "were written by the v1 pipeline (x and y swapped). Convert it with "
            "combined_analysis/convert_noise_data_v2.py."
        )
    return ds


def load_noise_data(path: str | Path, lazy: bool = False, **open_kwargs) -> xr.Dataset:
    """Open a recording's noise_data_v2.nc (checked to be v2).

    lazy=False reads everything into memory (like xr.load_dataset);
    lazy=True reads on demand (like xr.open_dataset) — use for big files.
    Extra keyword arguments (e.g. chunks=...) go to xarray.
    """
    f = noise_data_v2_path(path)
    ds = xr.open_dataset(f, **open_kwargs) if lazy else xr.load_dataset(f, **open_kwargs)
    return require_v2(ds, str(f))
