import numpy as np
import networkx as nx


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
