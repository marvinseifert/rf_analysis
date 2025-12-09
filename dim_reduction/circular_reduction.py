import numpy as np
from pathlib import Path
from covariance.filtering import cov_filtering_sum
from smoothing.gaussian import smooth_ker

from location.channel_handling import masking_square
from dataclasses import dataclass, field
import scipy.ndimage as ndimage
import pickle


# %% Functions
def polar_transform(
        array: np.ndarray, center: tuple = (100, 100), max_radius: int = 70
):
    # Define the shape of the new array
    new_shape = (360, max_radius)
    # Define the center of the original array
    center_y, center_x = center

    # Create arrays for the polar coordinates
    theta = np.linspace(
        0, 2 * np.pi, new_shape[0], endpoint=False
    )  # Angles from 0 to 2*pi
    radius = np.linspace(0, np.min(center), new_shape[1])  # Radii from 0 to 100

    # Create a grid of polar coordinates
    theta_grid, radius_grid = np.meshgrid(theta, radius, indexing="ij")

    # Convert polar coordinates to Cartesian coordinates
    x_cartesian = center_x + radius_grid * np.cos(theta_grid)
    y_cartesian = center_y + radius_grid * np.sin(theta_grid)

    # Interpolation: map the coordinates to the original array
    coordinates = np.array([y_cartesian.ravel(), x_cartesian.ravel()])

    # Use map_coordinates to interpolate values from the original array
    polar_array = ndimage.map_coordinates(
        array, coordinates, order=1, mode="reflect"
    ).reshape(new_shape)

    return polar_array


def cartesian_to_polar(x, y, center):
    """
    Convert Cartesian coordinates to polar coordinates.
    The angle is given in degrees.
    """
    x_centered = x - center[0]
    y_centered = y - center[1]

    distance = np.sqrt(x_centered ** 2 + y_centered ** 2)
    angle = np.degrees(np.arctan2(y_centered, x_centered))

    # Normalize angle to be in the range [0, 360)
    angle = (angle + 360) % 360

    return angle, distance


def create_polar_indexed_array(array):
    """
    Create an array indexed by angle and distance from the center.
    """
    size = array.shape[0]
    center = (size // 2, size // 2)

    # Create a grid of coordinates
    y, x = np.indices((size, size))

    # Convert the coordinates to polar coordinates
    angles, distances = cartesian_to_polar(x, y, center)

    # Define the maximum distance (which will be the radius of the array)
    max_distance = np.max(distances)

    # Create a new array to hold the polar indexed values
    polar_array = np.zeros((360, int(np.ceil(max_distance)) + 1))

    # Fill the polar array with values from the original array
    for i in range(size):
        for j in range(size):
            angle = int(angles[i, j])
            distance = int(distances[i, j])
            polar_array[angle, distance] = array[i, j]

    return polar_array


def polar_to_cartesian(angle, distances, center):
    """
    Convert polar coordinates (angle in degrees, distances) to Cartesian coordinates.
    """
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Compute the Cartesian coordinates
    x = center[0] + distances * np.cos(angle_rad)
    y = center[1] + distances * np.sin(angle_rad)

    return np.round(x).astype(int), np.round(y).astype(int)


def extract_entries_at_angle(array, angle):
    """
    Extract entries from the array at a given angle.
    """
    size = array.shape[1]
    center = (size // 2, size // 2)

    # Define the range of distances from the center
    max_distance = int(np.ceil(np.sqrt(center[0] ** 2 + center[1] ** 2)))
    distances = np.arange(max_distance + 1)

    # Convert the angle and distances to Cartesian coordinates
    x_coords, y_coords = polar_to_cartesian(angle, distances, center)

    # Ensure coordinates are within the array bounds
    valid_mask = (
            (x_coords >= 0) & (x_coords < size) & (y_coords >= 0) & (y_coords < size)
    )
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]

    # Extract the entries from the array
    entries = array[y_coords, x_coords]

    return entries


# %%
@dataclass
class CircularReduction:
    """
    Class to perform the circular reduction of the STAs. Circular reduction is a method to map the 2D receptive field
    onto a circle and then perform PCA on the resulting array in two ways.
    First, the PCA is performed on the circle itself, which gives the spatial principal components around the circle.
    Second, the PCA is performed on the transposed array, which gives the temporal principal components. Temporal here
    does not refer to the time per se, but to the temporal structure of the receptive field, so from inside to outside.

    The class is initialized with the folder containing the STAs, the cell indices to be processed, the size of the cutout
    around the centre of the STA, and the pixel size of the STA.

    The class has a reduce method that performs the circular reduction and stores the results in the class attributes.

    :param folder: Path to the folder containing the STAs.
    :param cells: Array of cell indices to be processed.
    :param cut: Size of the cutout around the centre of the STA in pixels.
    :param pixel_size: Pixel size of the STA.
    :param noise_size: Tuple with the size of the noise STAs.

    """

    folder: Path
    all_cells: int
    good_cells: np.ndarray
    cut: int
    pixel_size: float
    noise_size: tuple = field(init=True, default=(600, 600))
    noise_bins: int = field(init=True, default=55)
    shuffle_repeat: int = field(init=True, default=1)
    cm_most_important: np.ndarray = field(init=False)
    polar_cm: np.ndarray = field(init=False)
    center_outline: np.ndarray = field(init=False)
    surround_outline: np.ndarray = field(init=False)
    in_out_outline: np.ndarray = field(init=False)
    sta_single: np.ndarray = field(init=False)

    def __post_init__(self):
        self.center_outline = np.zeros((self.all_cells, 36))
        self.surround_outline = np.zeros((self.all_cells, 36))
        self.in_out_outline = np.zeros((self.all_cells, 70))
        self.cm_most_important = np.zeros((self.all_cells, self.cut, self.cut))
        self.sta_single = np.zeros((self.all_cells, self.noise_bins))
        self.half_cut = int(self.cut // 2)
        self.border_cut = int(self.half_cut * 0.5)
        self.positions = np.zeros((self.all_cells, 2))

    def add_positions(self, positions):
        self.positions = (
                positions + self.border_cut
        )  # Needs to be added due to the extension of the STAs in the reduce method

    def reduce(self, output_folder=None, use_precomputed_cm=False, use_positions=False):
        if output_folder is None:
            output_folder = self.folder / "circular_reduction"
        output_folder.mkdir(
            exist_ok=True
        )  # Create the output folder if it does not exist

        for cell in self.good_cells:
            if not use_precomputed_cm:
                data_path = self.folder / rf"cell_{cell}\kernel.npy"
                sta_data = np.load(data_path)
                # Downsample
                sta_data = sta_data[:, :: self.shuffle_repeat, :: self.shuffle_repeat]
                original_sta_shape = sta_data.shape
                sta_extended = (
                        np.zeros(
                            (
                                original_sta_shape[0],
                                original_sta_shape[1] + int(self.cut * 0.5),
                                original_sta_shape[2] + int(self.cut * 0.5),
                            )
                        )
                        * 0.5
                )
                sta_extended[
                    :,
                    self.border_cut: -self.border_cut,
                    self.border_cut: -self.border_cut,
                ] = sta_data
                sta_data = sta_extended
                del sta_extended
                original_sta_shape = sta_data.shape
                sta_data = sta_data.astype(float)

                if not use_positions:
                    ker_sm = smooth_ker(sta_data)
                    _, cY, cX = np.unravel_index(
                        np.argmax(np.var(ker_sm, axis=0)), ker_sm.shape
                    )
                    self.positions[cell, :] = (cX, cY)
                else:
                    cX, cY = self.positions[cell, :]
                # If the center is still too close to any of the edges, skip the cell
                if cY > original_sta_shape[1] - int(self.cut / 2):
                    continue
                if cX > original_sta_shape[2] - int(self.cut / 2):
                    continue
                if cY < int(self.cut / 2):
                    continue
                if cX < int(self.cut / 2):
                    continue

                mask = masking_square(
                    original_sta_shape[1],
                    original_sta_shape[2],
                    (cX, cY),
                    self.cut,
                    self.cut,
                )

                subset_flat = sta_data[:, mask]
                subset = np.reshape(
                    subset_flat, (original_sta_shape[0], self.cut, self.cut)
                )
                try:
                    del cm
                except NameError:
                    pass
                important_pixels, cm = cov_filtering_sum(subset, (self.cut, self.cut))
                most_important_pixel = np.unravel_index(np.argmax(cm), cm.shape)
                x, y = np.unravel_index(most_important_pixel, (self.cut, self.cut))
                sta_single = subset[:, x, y]
                sta_single = sta_single[:, 0]
                self.sta_single[cell, :] = sta_single
                cm_subset = cm[most_important_pixel[0], :]
                mean_cm_subset = cm_subset.reshape((self.cut, self.cut))
                mean_cm_subset[x, y] = np.var(subset, axis=0)[x, y]
                self.cm_most_important[cell, :, :] = mean_cm_subset

            polar_cm = polar_transform(
                self.cm_most_important[cell, :, :], (self.half_cut, self.half_cut)
            )
            entries = []
            for deg_10 in range(0, 360, 10):
                entries.append(np.mean(polar_cm[deg_10: deg_10 + 10, :], axis=0))
            entries = np.asarray(entries)
            entries_pos = entries.copy()
            entries_pos[entries_pos < 0] = 0
            entries_mean = (
                    np.mean(entries_pos / np.max(entries_pos), axis=1)
                    * entries_pos.shape[1]
            )

            entries_neg = entries.copy()
            entries_neg[entries_neg > 0] = 0
            entries_mean_neg = (
                    np.mean(entries_neg / np.min(entries_neg), axis=1)
                    * entries_neg.shape[1]
            )

            self.center_outline[cell, :] = entries_mean
            self.surround_outline[cell, :] = entries_mean_neg
            self.in_out_outline[cell, :] = np.mean(entries, axis=0)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
