import numpy as np
import plotly.graph_objects as go
import utils
from math import pi, sin, cos
from common import edge_iterator
from drawing_tools import semi_circle, quadratic_bezier_curve, kamada_kawai

HEIGHT = 1000

def arc(adjacency_list):
    N = len(adjacency_list)
    vcoords = {ID:(idx/(N-1), 0) for idx, ID in enumerate(adjacency_list.keys())}
    fig = go.Figure()
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        if vertex < neighbour:
            scx, scy = semi_circle(vcoords[vertex][0], vcoords[neighbour][0])
            fig.add_trace(go.Scattergl(
                x=scx,
                y=scy,
                mode='markers',
                #mode='lines',
                #line_width=1,
                #line_color='red',
                marker_color=np.arange(len(scx)),
                showlegend=False
            ))
    fig.add_trace(go.Scattergl(
        x=np.arange(N)/(N-1),
        y=np.zeros(N),
        mode='markers',
        marker_size=10,
        marker_color='blue',
        showlegend=False,
        text='a'
    ))
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20)
    fig.show()
    return

def circular(adjacency_list, arcs=True):
    N = len(adjacency_list)
    edge_width, vertex_size = 1, 10
    vcoords = {ID:(sin(2*pi*idx/N), cos(2*pi*idx/N)) for idx, ID in enumerate(adjacency_list.keys())}
    fig = go.Figure()
    if arcs:
        for vertex, neighbour, attributes in edge_iterator(adjacency_list):
            if vertex < neighbour:
                rscx, rscy = quadratic_bezier_curve(vcoords[vertex], (0,0), vcoords[neighbour])
                fig.add_trace(go.Scattergl(
                    x=rscx,
                    y=rscy,
                    mode='lines',
                    line_width=1,
                    line_color='red',
                    showlegend=False
                ))
    else:
        for vertex, neighbour, attributes in edge_iterator(adjacency_list):
            if vertex < neighbour:
                (x_1, y_1), (x_2, y_2) = vcoords[vertex], vcoords[neighbour]
                fig.add_trace(go.Scattergl(
                    x=(x_1, x_2),
                    y=(y_1, y_2),
                    mode='lines',
                    line_width=1,
                    line_color='red',
                    showlegend=False
                ))
    fig.add_trace(go.Scattergl(
        x=np.sin(2*pi*np.arange(N)/N),
        y=np.cos(2*pi*np.arange(N)/N),
        mode='markers',
        marker_size=vertex_size,
        marker_color='blue',
        showlegend=False
    ))
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20)
    fig.show()
    return

def matrix(adjacency_list):
    adjacency_matrix = np.zeros((len(adjacency_list), len(adjacency_list)))
    rev_dict = {ID:idx for idx, ID in enumerate(adjacency_list.keys())}
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        adjacency_matrix[rev_dict[vertex]][rev_dict[neighbour]] = attributes.get('weight', 1)
    fig = go.Figure(data=go.Heatmap(z=adjacency_matrix[::-1], showscale=False))
    fig.update_yaxes(tickvals=[], scaleanchor='x')
    fig.update_xaxes(tickvals=[])
    fig.update_layout(width=HEIGHT-20, height=HEIGHT)
    fig.show()
    return

def pretty(adjacency_list, edge_list=None, z=None):
    if not utils.ispercolating(adjacency_list):
        print("Discjunct components!")
        return
    GD = np.array(utils.graph_theoretical_distance(adjacency_list))
    optres = kamada_kawai(adjacency_list, GD)
    N = len(adjacency_list)
    coords = optres.x.reshape((N,2))
    edge_list = utils.edge_list(adjacency_list) if edge_list is None else edge_list
    fig = go.Figure()
    for i, j in edge_list:
        fig.add_trace(go.Scattergl(
            x=coords[[i,j], 0],
            y=coords[[i,j], 1],
            mode='lines',
            line_width=1,
            line_color='red',
            showlegend=False
        ))
    fig.add_trace(go.Scattergl(
        x=coords[:,0],
        y=coords[:,1],
        mode='markers',
        marker_size=10,
        marker_color='blue' if z is None else z,
        showlegend=False
    ))
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20)
    fig.show()
    return
