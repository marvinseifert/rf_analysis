from __future__ import annotations
from typing import List, Dict
import xarray as xr
import numpy as np
from organize.decorators import depends_on
import polars as pl
import math
from circle_fit import standardLSQ
from sklearn.decomposition import PCA
from itertools import product
from tqdm import tqdm


def calculate_radial_area(radii: List[float]) -> float:
    """
    Calculates the approximate area of a 2D shape defined by radial distances
    from a central point, assuming uniform angular spacing.

    The shape is approximated as a polygon formed by triangles, where each
    triangle is defined by the center and two adjacent radii.
    Parameters
    ----------
    radii : List[float]
        A list of radial distances from the center to the boundary of the shape.

    """
    n = len(radii)

    if n < 3:
        raise ValueError("A polygon must have at least 3 radii/sides.")

    # 1. Calculate the uniform angular separation (in radians)
    delta_theta = (2 * math.pi) / n

    # 2. Pre-calculate the constant part of the triangle area formula
    # Area_triangle = 0.5 * r_i * r_{i+1} * sin(delta_theta)
    constant_factor = 0.5 * math.sin(delta_theta)

    total_area = 0.0

    # 3. Sum the areas of n triangles
    for i in range(n):
        # Current radius: r_i
        r_i = radii[i]

        # Next radius: r_{i+1}. Use the modulus operator (%) to wrap around
        # to the first element (r_0) when i+1 == n (r_n = r_0)
        r_next = radii[(i + 1) % n]

        # Calculate the area of the current triangle
        triangle_area = constant_factor * r_i * r_next

        total_area += triangle_area

    return total_area


def project_radii_to_xy(radii: List[float]) -> np.ndarray:
    n = len(radii)
    delta_theta = (2 * math.pi) / n
    x_coords = []
    y_coords = []
    for i in range(n):
        theta = i * delta_theta
        r = radii[i]
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        x_coords.append(x)
        y_coords.append(y)
    return np.column_stack((x_coords, y_coords))


@depends_on("calculate_rf_quality")
@depends_on("sta_2d_cov_collapse")
@depends_on("circular_reduction")
def calculate_stats(
        recording_config: "Recording_Config",
        collapse_2d_config: "Collapse_2d_Config",
        circular_reduction_config: "Circular_Reduction_Config",
        analysis_folder: str,
):
    ds = xr.load_dataset(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_data.nc",
    )
    # % Predefine outputs
    template = xr.DataArray(
        np.zeros((ds.cell_index.shape[0], len(recording_config.channel_names))),
        dims=["cell_index", "channel"],
        coords={
            "cell_index": ds.cell_index,
            "channel": recording_config.channel_names,
        },
    )
    # Create all arrays using the template
    center_sizes_mm2 = xr.zeros_like(template)
    surround_sizes_mm2 = xr.zeros_like(template)
    tirs = xr.zeros_like(template)
    center_shifts = xr.zeros_like(template)
    tilts = xr.zeros_like(template)
    angles = xr.zeros_like(template)

    # loop over cells and channels
    channel_names = recording_config.channel_names
    with np.errstate(divide="ignore", invalid="ignore"):
        for (cell_position, cell_index), channel in tqdm(
                product(enumerate(ds.cell_index), channel_names),
                total=ds.cell_index.shape[0] * len(channel_names),
                desc="Calculating RF stats",
        ):
            # %% Calculate RF size
            center_size = [
                calculate_radial_area(
                    ds.sel(cell_index=cell_index, channel=channel)[
                        "center_outline_um"
                    ].values.tolist()
                )
            ]
            surround_size = [
                calculate_radial_area(
                    ds.sel(cell_index=cell_index, channel=channel)[
                        "surround_outline_um"
                    ].values.tolist()
                )
            ]

            center_sizes_mm2.loc[dict(cell_index=cell_index, channel=channel)] = (
                    np.asarray(center_size)
                    * np.asarray(recording_config.channel_configs[channel].pixel_size) ** 2
                    * 1e-6
            )[0]
            surround_sizes_mm2.loc[dict(cell_index=cell_index, channel=channel)] = (
                    np.asarray(surround_size)
                    * np.asarray(recording_config.channel_configs[channel].pixel_size) ** 2
                    * 1e-6
            )[0]

            positions = project_radii_to_xy(
                ds.sel(cell_index=cell_index, channel=channel)[
                    "center_outline_um"
                ].values.tolist()
            )

            circle_params = standardLSQ(positions)
            # 1. Fit the best-fit circle: returns (xc, yc, R, algebraic_residuals)
            xc, yc, r, sigma = circle_params
            # 2. Calculate the true radial distance (ri) from every point to the fitted center
            # ri = sqrt((x_i - xc)^2 + (y_i - yc)^2)
            radial_distances = np.sqrt(
                (positions[:, 0] - xc) ** 2 + (positions[:, 1] - yc) ** 2
            )
            # 3. Calculate the deviation (radial error) for each point
            # Deviation (Delta_i) = Actual Radial Distance - Best-Fit Radius
            deviations = radial_distances - r

            # 4. Calculate the Total Indicated Runout (TIR)
            # TIR = Max Deviation - Min Deviation
            tir_value = np.max(deviations) - np.min(deviations)
            tirs.loc[dict(cell_index=cell_index, channel=channel)] = tir_value

            nominal_x, nominal_y = (0, 0)

            # Distance formula: sqrt((xc - nominal_x)^2 + (yc - nominal_y)^2)
            center_shift = np.sqrt((xc - nominal_x) ** 2 + (yc - nominal_y) ** 2)
            center_shifts.loc[
                dict(cell_index=cell_index, channel=channel)
            ] = center_shift
            pca = PCA(n_components=2)
            pca.fit(positions)
            major_axis_vector = pca.components_[0]
            minor_axis_vector = pca.components_[1]
            major_axis_length = 2 * np.sqrt(pca.explained_variance_[0])
            minor_axis_length = 2 * np.sqrt(pca.explained_variance_[1])
            tilt = minor_axis_length / major_axis_length
            # calculate tilt
            angle_rad = np.arctan2(major_axis_vector[1], major_axis_vector[0])
            angle_deg = np.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 180
            elif angle_deg > 180:
                angle_deg -= 180

            tilts.loc[dict(cell_index=cell_index, channel=channel)] = tilt
            angles.loc[dict(cell_index=cell_index, channel=channel)] = angle_deg

    print("Finished, saving stats to dataset")
    ds["center_size_mm2"] = center_sizes_mm2
    ds["surround_size_mm2"] = surround_sizes_mm2
    ds["tir"] = tirs
    ds["center_shift"] = center_shifts
    ds["tilt"] = tilts
    ds["angle"] = angles
    ds.to_netcdf(
        recording_config.root_path
        / recording_config.output_folder
        / analysis_folder
        / "noise_data.nc",
        engine="netcdf4",
        encoding={var: {"zlib": True, "complevel": 9} for var in ds.data_vars},
    )
