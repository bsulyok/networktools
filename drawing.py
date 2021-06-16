import numpy as np
import plotly.graph_objects as go
import utils
import random
from math import pi, sin, cos
from common import edge_iterator
from drawing_tools import euclidean_line, semi_circle, poincare_line, edge_trace, kamada_kawai, hyperbolic_polar_line

HEIGHT = 1000

def easydraw(X, Y=None):
    fig = go.Figure()
    if Y is None:
        fig.add_trace(go.Scattergl(x=np.arange(len(X)), y=X, mode='lines'))
    else:
        fig.add_trace(go.Scattergl(x=X, y=Y, mode='lines'))
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False)
    fig.update_layout(height=HEIGHT, width=HEIGHT-20, showlegend=False)
    fig.show()
    return

def draw(adjacency_list, vertices=None, representation='euclidean', vertex_scale=None, edge_scale=None, default_vertex_color=1, default_edge_color='blue'):
    '''
    Draw the provided network.
    Parameters
    ----------
    adjacency_list : list of lists
        Adjacency list containing edge data.
    vertices : list of lists
        List containing vertex data.
    '''
    
    if vertices is None:
        vertices = {vertex : {'coord':{'r':random.random(), 'phi':2*np.pi*random.random()}} for vertex in adjacency_list}
    if vertex_scale is None:
        vertex_scale = 5
    if edge_scale is None:
        edge_scale = 1
    if representation == 'hyperbolic_polar':
        path_function = hyperbolic_polar_line
    elif representation == 'poincare':
        path_function = poincare_line
    elif representation == 'euclidean':
        path_function = euclidean_line
    
    fig = go.Figure()

    # draw the edges
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        if vertex < neighbour:
            coord_1, coord_2 = vertices[vertex]['coord'], vertices[neighbour]['coord']
            r1, phi1 = coord_1.get('r'), coord_1.get('phi')
            r2, phi2 = coord_2.get('r'), coord_2.get('phi')
            path = path_function( r1, phi1, r2, phi2)
            line_width = attributes.get('width', 1) * edge_scale
            line_color = attributes.get('color', default_edge_color)
            fig.add_trace(go.Scattergl(x=path[0], y=path[1], mode='lines', line_color=line_color, line_width=line_width))

    # draw the vertices
    N = len(adjacency_list)
    marker_x, marker_y, marker_size, marker_color = np.empty(N), np.empty(N), np.empty(N), np.empty(N)
    for idx, attributes in enumerate(vertices.values()):
        marker_x[idx] = attributes['coord'].get('r') * np.cos(attributes['coord'].get('phi'))
        marker_y[idx] = attributes['coord'].get('r') * np.sin(attributes['coord'].get('phi'))
        marker_color[idx] = attributes.get('color', default_vertex_color)
        marker_size[idx] = attributes.get('size', 1) * vertex_scale
    fig.add_trace(go.Scattergl(x=marker_x, y=marker_y, mode='markers', marker_size=marker_size, marker_color=marker_color, showlegend=False, marker_reversescale=True, marker_opacity=1))

    # figure settings
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20, showlegend=False)
    fig.show()
    return

def arc(adjacency_list):
    '''
    Draw the arc type representation of the provided network.
    Parameters
    ----------
    adjacency_list : list of lists
        Adjacency list containing edge data.
    '''
    edge_resolution, color_resolution = 100, 10
    N = len(adjacency_list)
    X = np.linspace(0, 1, N)
    vert_dict = {ID:idx for idx, ID in enumerate(adjacency_list.keys())}
    fig = go.Figure()

    # draw the edges
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        if vertex < neighbour:
            p1, p2 = X[vert_dict[vertex]], X[vert_dict[neighbour]]
            path = semi_circle(p1, p2)
            fig.add_trace(go.Scattergl(x=path.real, y=path.imag, mode='lines', line_color='red', line_width=1))

    # draw the vertices
    fig.add_trace(go.Scattergl(x=X, y=np.zeros(N), mode='markers', marker_size=10, marker_color='blue', showlegend=False, text='a'))

    # figure settings
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20, showlegend=False)
    fig.show()
    return

def circular(adjacency_list, lines='euclidean'):
    '''
    Draw the arc type representation of the provided network.
    Parameters
    ----------
    adjacency_list : list of lists
        Adjacency list containing edge data.
    '''
    edge_resolution, color_resolution = 100, 10
    N = len(adjacency_list)
    edge_width, vertex_size = 1, 10
    vert_dict = {ID:idx for idx, ID in enumerate(adjacency_list.keys())}
    angle = np.linspace(0, 2*np.pi, N, endpoint=False)
    fig = go.Figure()

    # draw the edges
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        if vertex < neighbour:
            i, j = vert_dict[vertex], vert_dict[neighbour]
            x1, y1 = np.cos(angle[i]), np.sin(angle[i])
            x2, y2 = np.cos(angle[j]), np.sin(angle[j])
            if not hyperbolic or abs((angle[i]-angle[j])%np.pi) < 1e-5:
                x_path, y_path = euclidean_line(x1, y1, x2, y2, color_resolution+1)
            else:
                x_path, y_path = poincare_line(x1, y1, x2, y2, edge_resolution)
            fig.add_traces(edge_trace(x_path, y_path, 1, color_resolution))

    # draw the vertices
    fig.add_trace(go.Scattergl(x=np.cos(angle), y=np.sin(angle), mode='markers', marker_size=vertex_size, marker_color='rgb(255,0,0)', showlegend=False))

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