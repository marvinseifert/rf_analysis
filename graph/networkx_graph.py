from datashader.layout import random_layout, circular_layout, forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle
from colorcet import coolwarm
from datashader import transfer_functions as tf
import pandas as pd
import numpy as np
import networkx as nx
import datashader as ds

# Global
cvsopts = dict(plot_height=600, plot_width=600)


def create_graph_from_covariance(cov_matrix, threshold=0.1):
    # Using a threshold to filter edges with very low covariance
    labels = [f"Var{i+1}" for i in range(cov_matrix.shape[0])]
    G = nx.Graph()
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if (
                i != j and abs(cov_matrix[i, j]) > threshold
            ):  # Only consider significant covariances
                G.add_edge(label1, label2, weight=cov_matrix[i, j])
    return G


def edgesplot(edges, name=None, canvas=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    return tf.shade(
        canvas.line(edges, "x", "y", agg=ds.count(), line_width=2),
        name=name,
        how="log",
        cmap=coolwarm[::-1],
    )


def graphplot(nodes, edges, name="", canvas=None, cat=None):
    if canvas is None:
        xr = nodes.x.min(), nodes.x.max()
        yr = nodes.y.min(), nodes.y.max()
        canvas = ds.Canvas(x_range=xr, y_range=yr, **cvsopts)

    np = nodesplot(nodes, name + " nodes", canvas, cat)
    ep = edgesplot(edges, name + " edges", canvas)
    return tf.stack(ep, np, how="over", name=name)


def nodesplot(nodes, name=None, canvas=None, cat=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    aggregator = None if cat is None else ds.count_cat(cat)
    agg = canvas.points(nodes, "x", "y", aggregator)
    return tf.spread(
        tf.shade(agg, color_key=[coolwarm[0], coolwarm[-1]]),
        px=5,
        name=name,
    )


def nx_layout(graph, x=None, y=None):
    layout = nx.random_layout(graph)
    data = [[node] + layout[node].tolist() for node in graph.nodes]

    nodes = pd.DataFrame(data, columns=["id", "x", "y"])
    nodes.set_index("id", inplace=True)

    # Categeoize the nodes by the sum of the weights of their edges. If above 0, it is positive, otherwise negative
    # Extract the edges weights for a given node, sum them and categorize the node as positive or negative
    cat = {
        node: str(
            np.sign(
                sum([graph[node][neigh]["weight"] for neigh in graph.neighbors(node)])
            )
        )
        for node in graph.nodes
    }
    nodes["cat"] = [cat[node] for node in nodes.index]
    nodes["cat"] = nodes["cat"].astype("category")
    if x is not None and y is not None:
        nodes["x"] = x
        nodes["y"] = y

    edges = pd.DataFrame(list(graph.edges), columns=["source", "target"])
    return nodes, edges


def nx_plot(graph, name="", x=None, y=None):
    print(graph.name, len(graph.edges))
    nodes, edges = nx_layout(graph, x, y)
    if x is None and y is None:
        fd = forceatlas2_layout(nodes, edges)
        direct = connect_edges(fd, edges)
        bundled_bw005 = hammer_bundle(fd, edges)
        bundled_bw030 = hammer_bundle(fd, edges, initial_bandwidth=0.30)

        return [
            graphplot(fd, direct, graph.name, cat="cat"),
            graphplot(fd, bundled_bw005, "Bundled bw=0.05", cat="cat"),
            graphplot(fd, bundled_bw030, "Bundled bw=0.30", cat="cat"),
        ]
    else:
        direct = connect_edges(nodes, edges)
        bundled_bw005 = hammer_bundle(nodes, edges)
        bundled_bw030 = hammer_bundle(nodes, edges, initial_bandwidth=0.30)
        return [
            graphplot(nodes, direct, graph.name, cat="cat"),
            graphplot(nodes, bundled_bw005, "Bundled bw=0.05", cat="cat"),
            graphplot(nodes, bundled_bw030, "Bundled bw=0.30", cat="cat"),
        ]
