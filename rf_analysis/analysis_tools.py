import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from matplotlib_scalebar.scalebar import ScaleBar
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import decimate
import networkx as nx
from sklearn.decomposition import PCA


# %% Functions
def smooth_ker(ker: np.ndarray, axes: tuple | None = None) -> np.ndarray:
    """
    Smooth the kernel using a Gaussian filter.
    :param ker: The sta_kernel to be smoothed.
    :return: The smoothed kernel.
    """

    return gaussian_filter(ker.astype(float), 0.5, mode="constant", cval=np.median(ker), axes=axes)


def decimate_ker(ker_sm: np.ndarray, factor: int = 5) -> np.ndarray:
    """
    Decimate the kernel by a factor in x and y direction.
    :param ker_sm: The smoothed kernel.
    :param factor: The decimation factor.
    :return: The decimated kernel.
    """
    ker_ds = decimate(
        decimate(ker_sm, factor, ftype="fir", axis=(1)), factor, ftype="fir", axis=2
    )
    ker_ds = ker_ds[:, 2 * factor: -2 * factor, 2 * factor: -2 * factor]
    return ker_ds


def identify_center(sta_std: np.ndarray) -> tuple[int, int]:
    """
    Identify the center of the STA by finding the largest contour in the image.

    :param sta_std: The STA_STD image.
    :return: The x and y coordinates of the center of the STA.
    """
    # Convert the image to 8 bit 3 channel image
    sta_std = sta_std / np.max(sta_std) * 255
    sta_std = sta_std.astype(np.uint8)
    # Extend the image to the 3 channel format
    sta_std = cv2.merge([sta_std, sta_std, sta_std])
    # Convert the image to grayscale
    sta_std = cv2.cvtColor(sta_std, cv2.COLOR_BGR2GRAY)
    # Apply the thresholding
    _, sta_std = cv2.threshold(sta_std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find the contours
    contours, _ = cv2.findContours(sta_std, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour
    c = max(contours, key=cv2.contourArea)
    # Find the centroid of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def zscore(array: np.ndarray) -> np.ndarray:
    """
    Normalize the array to have a mean of 0 and a standard deviation of 1.
    :param array: A numpy array.
    :return: The z-score normalized array.
    """
    return (array - np.mean(array)) / np.std(array)


def cov_filtering(
        ker: np.ndarray, shape: tuple = (100, 100)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the covariance matrix of the kernel and filter the pixels with the highest number of connections.
    :param ker: The sta_kernel to be filtered as flattened array.
    :param shape: The shape of the kernel in 2D.
    :return: The location of the important pixels and the covariance matrix.
    """
    ker_z = zscore(ker)

    ker_shape = ker_z.shape
    ker_z = ker_z.reshape(ker_shape[0], -1)
    cm = np.cov(ker_z.T)

    # Postprocessing
    diag = np.where(np.diag(np.diag(cm)))[0]
    cm[diag, diag] = 0
    # cm_max = np.max(cm, axis=0)
    cm_argmax = np.argmax(np.triu(cm), axis=0)
    pix, px_nr = np.unique(cm_argmax, return_counts=True)

    important_pixels = np.zeros(shape[0] * shape[1])
    important_pixels[pix] = px_nr
    important_pixels = important_pixels.reshape(shape)
    return important_pixels, cm


def cov_filtering_sum(
        ker: np.ndarray, shape: tuple = (100, 100)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the covariance matrix of the kernel and filter the pixels with the highest sum of connections.
    :param ker: The sta_kernel to be filtered as flattened array.
    :param shape: The shape of the kernel in 2D.
    :return: The location of the important pixels and the covariance matrix.
    """

    # ker_z = zscore(ker)
    ker_z = ker.copy()

    ker_shape = ker_z.shape
    ker_z = ker_z.reshape(ker_shape[0], -1)
    cm = np.cov(ker_z.T)

    # Postprocessing
    diag = np.where(np.diag(np.diag(cm)))[0]
    cm[diag, diag] = 0
    cm_sum = np.sum(cm, axis=0)
    # Get pixels with higest sum of connections
    important_sums = np.where(cm_sum > 0)[0]
    important_mins = np.where(cm_sum < 0)[0]

    important_pixels = np.zeros(shape[0] * shape[1])
    important_pixels[important_sums] = cm_sum[important_sums]
    important_pixels[important_mins] = cm_sum[important_mins]
    important_pixels = important_pixels.reshape(shape)
    return important_pixels, cm


def add_scalebar(ax: plt.Axes, pixel_size: int = 1) -> None:
    """
    Add a scalebar to the plot based on the size of a single pixel.
    :param ax: The axis to add the scalebar to.
    :param pixel_size: The size of a single pixel in micrometers.
    :return: None
    """
    scalebar = ScaleBar(pixel_size, "um", fixed_value=100)
    ax.add_artist(scalebar)


def check_border_entries_false(array: np.ndarray) -> bool:
    """
    Check if the first and last row and column of the array are all False.

    :param array: A 2D numpy array.
    :return: True if the first and last row and column are all False, False otherwise.
    """
    # Check if the first row and the last row are all False
    first_last_row_check = np.all(array[0, :] == False) and np.all(
        array[-1, :] == False
    )

    # Check if the first column and the last column are all False,
    # excluding the first and last element since they were checked above.
    first_last_column_check = np.all(array[1:-1, 0] == False) and np.all(
        array[1:-1, -1] == False
    )

    return first_last_row_check and first_last_column_check


def merge_clusters(similarity_matrix: np.ndarray) -> list[set[int]]:
    """
    Merge clusters based on a similarity matrix.
    :param similarity_matrix: A similarity matrix between clusters.
    :return: A list of sets of clusters to be merged.
    """
    # Step 1: Create a graph
    G = nx.Graph()
    n_clusters = similarity_matrix.shape[0]

    # Add nodes for each cluster
    G.add_nodes_from(range(n_clusters))

    # Add edges for each pair of clusters with similarity 1
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):  # Avoid duplicates and self-loops
            if similarity_matrix[i, j] == 1:
                G.add_edge(i, j)

    # Step 2: Find connected components
    connected_components = list(nx.connected_components(G))

    # Step 3: Optionally, merge clusters
    # This step depends on how you want to represent the merged clusters.
    # For demonstration, I'll return the list of clusters to be merged.
    return connected_components


#
# def isolate_central_island(matrix):
#     rows = len(matrix)
#     columns = len(matrix[0])
#
#     # Step 1: Process all four borders to mark non-central True values as False
#     # Top and bottom
#     for i in range(columns):
#         if matrix[0][i]: flood_fill(matrix, 0, i, True, False)
#         if matrix[rows - 1][i]: flood_fill(matrix, rows - 1, i, True, False)
#     # Left and right
#     for i in range(rows):
#         if matrix[i][0]: flood_fill(matrix, i, 0, True, False)
#         if matrix[i][columns - 1]: flood_fill(matrix, i, columns - 1, True, False)
#
#     # At this point, all `True` values outside the central island have been set to `False`
#     return matrix


# %%
def array_to_graph(matrix: np.ndarray) -> nx.Graph:
    """
    Convert a binary matrix to a graph where True values are nodes and adjacent True values are connected by edges.
    :param matrix: A binary matrix.
    :return: A graph where True values are nodes and adjacent True values are connected by edges.
    """
    rows, cols = len(matrix), len(matrix[0])
    G = nx.Graph()

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j]:  # Add node if True
                G.add_node((i, j))
                # Add edges to adjacent nodes if True
                if i > 0 and matrix[i - 1][j]:
                    G.add_edge((i, j), (i - 1, j))
                if i < rows - 1 and matrix[i + 1][j]:
                    G.add_edge((i, j), (i + 1, j))
                if j > 0 and matrix[i][j - 1]:
                    G.add_edge((i, j), (i, j - 1))
                if j < cols - 1 and matrix[i][j + 1]:
                    G.add_edge((i, j), (i, j + 1))
    return G


def isolate_central_island(matrix: np.ndarray) -> np.ndarray:
    """
    Isolate the central island in a binary matrix by marking all non-central islands as False.
    :param matrix: A binary matrix.
    :return: The matrix with only the central island marked as True.
    """
    G = array_to_graph(matrix)
    largest_component = max(nx.connected_components(G), key=len)

    # Mark cells not in the largest component as False
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (i, j) not in largest_component:
                matrix[i][j] = False
            else:
                matrix[i][
                    j
                ] = True  # Ensure all nodes in the largest component are True
    return matrix


def video(
        sta,
        important_pixels=None,
        important_y_pos=None,
        important_x_pos=None,
        important_x_neg=None,
        important_y_neg=None,
        cmap="Greys",
):
    """Creates a video from a 3d array where the first dimension is time and the second and third dimensions are
    space.
    :param data_re:
    :return: html object containing the video
    """
    matplotlib.rcParams["animation.embed_limit"] = 2 ** 128
    fig, ax = plt.subplots(figsize=(10, 10), dpi=500)
    ims = []
    # ax.scatter(
    #     important_y_pos,
    #     important_x_pos,
    #     c=important_pixels[important_x_pos, important_y_pos],
    #     cmap="Reds",
    #     marker="x",
    #     vmin=0,
    #     vmax=np.max(important_pixels),
    #     s=1,
    #     alpha=1,
    # )
    # ax.scatter(
    #     important_y_neg,
    #     important_x_neg,
    #     c=important_pixels[important_x_neg, important_y_neg],
    #     cmap="Blues_r",
    #     marker="x",
    #     vmin=np.min(important_pixels),
    #     vmax=0,
    #     s=1,
    #     alpha=1,
    # )

    for i in range(sta.shape[0]):
        # Add the important pixels

        im = ax.imshow(
            sta[i, :, :],
            animated=True,
            cmap=cmap,
            vmin=np.min(sta),
            vmax=np.max(sta),
        )

        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return ani


class GroupPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pcas = []
        self.final_pca = PCA(n_components=n_components)

    def fit(self, Xs):
        # Step 1: Perform PCA on each view
        transformed_views = []
        for X in Xs:
            pca = PCA(n_components=self.n_components)
            transformed = pca.fit_transform(X)
            self.pcas.append(pca)
            transformed_views.append(transformed)

        # Step 2: Concatenate the transformed data
        concatenated = np.hstack(transformed_views)

        # Step 3: Perform PCA on the concatenated data
        self.final_pca.fit(concatenated)

    def transform(self, Xs):
        # Step 1: Transform each view using the fitted PCAs
        transformed_views = [pca.transform(X) for pca, X in zip(self.pcas, Xs)]

        # Step 2: Concatenate the transformed data
        concatenated = np.hstack(transformed_views)

        # Step 3: Transform the concatenated data using the final PCA
        return self.final_pca.transform(concatenated)

    def fit_transform(self, Xs):
        self.fit(Xs)
        return self.transform(Xs)
