import numpy as np
import cv2
import networkx as nx
from graph.build_graph import array_to_graph


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
