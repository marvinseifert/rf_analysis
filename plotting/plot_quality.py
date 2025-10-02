import plotly.graph_objs as go
import numpy as np
from pathlib import Path

# %%
root = Path(rf"/media/mawa/52DE7A8FDE7A6B5D/noise_analysis/chicken_09_05_2024/kilosort4")
channels = ["4px_20Hz_shuffle_610nm_idx_10", "4px_20Hz_shuffle_535nm_idx_12",
            "12px_20Hz_shuffle_white_idx_15"]  # "4px_20Hz_shuffle_535nm_idx_12"
colours = ["Reds", "Greens", "Greys"]
fig = go.Figure()
for channel in channels:
    quality = np.load(root / f"{channel}/quality.npy")

    fig.add_trace(go.Scatter(
        x=quality[:, 2],
        y=quality[:, 3],
        mode='markers',
        marker=dict(
            size=5,
            color=quality[:, 0],  # set color to quality
            colorscale=colours[channels.index(channel)],
            colorbar=dict(title='Quality'),
            showscale=False
        ),
        text=[f'Cell {i}' for i in range(len(quality))],  # hover text
        name=f'Channel {channel}'
    ))
    fig.update_layout(
        title=f'Cell Positions and Quality - {channel}',
        xaxis_title='X Position',
        yaxis_title='Y Position',
        width=1200,
        height=1200
    )
    fig.show(renderer="browser")
