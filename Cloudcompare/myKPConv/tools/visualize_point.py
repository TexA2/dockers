import numpy as np
from typing import List, Dict, Tuple, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_point_cloud(points: np.ndarray, colors: np.ndarray) -> go.Figure:
    """
    In :
       points : np.ndarray - Облако точек (x,y,z) [N,3]
       colors : np.ndarray - Цвета точек [N,3] или список цветов
    Out :
       fig : go.Figure - график в плотли
    """
    
    # Преобразуем цвета в RGB строки для plotly
    if isinstance(colors, list) and len(colors) > 0 and isinstance(colors[0], tuple):
        colors_rgb = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
    else:
        colors_rgb = colors

    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors_rgb,
            opacity=0.8
        )
    )

    layout = go.Layout(
        title="3D Point Cloud Visualization",
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y", 
            zaxis_title="Z",
            aspectmode='data'
        )
    )

    fig = go.Figure(data=[trace], layout=layout)
    return fig