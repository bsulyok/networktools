import numpy as np
from numpy.random import default_rng
import plotly.graph_objects as go
import utils
import random
from math import pi, sin, cos
from common import edge_iterator
from drawing_tools import euclidean_line, semi_circle, poincare_line, edge_trace, kamada_kawai, hyperbolic_polar_line

HEIGHT = 1000

representation_dict = {'hyperbolic_polar': hyperbolic_polar_line, 'poincare': poincare_line, 'euclidean':euclidean_line}

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

def draw(adjacency_list, vertices=None, **kwargs):

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
        vertices = {vertex : {'r':random.random(), 'phi':2*np.pi*random.random()} for vertex in adjacency_list}
    path_function = representation_dict.get(kwargs.get('representation', 'euclidean'), euclidean_line)
    vertex_scale = kwargs.get('vertex_scale', 3)
    default_vertex_color = kwargs.get('default_vertex_color', 1)
    default_vertex_size = kwargs.get('default_vertex_size', 5)
    vertex_color_attribute = kwargs.get('vertex_color_attribute', 'color')
    vertex_size_attribute = kwargs.get('vertex_size_attribute', 'size')
    edge_scale = kwargs.get('edge_scale', 1)
    default_edge_width = kwargs.get('default_vertex_size', 1) 
    edge_width_attribute = kwargs.get('edge_width_attribute', 'width')
    
    fig = go.Figure()

    # draw the edges
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        if neighbour < vertex:
            path = path_function( vertices[vertex]['r'], vertices[vertex]['phi'], vertices[neighbour]['r'], vertices[neighbour]['phi'])
            line_width = attributes.get(edge_width_attribute, default_edge_width) * edge_scale
            fig.add_trace(go.Scattergl(
                    x=path[0],
                    y=path[1],
                    mode='lines',
                    line_color='rgb(125,125,125)',
                    hoverinfo='skip',
                    line_width=line_width
            ))

    # draw the vertices
    N = len(adjacency_list)
    marker_x, marker_y, marker_size, marker_color, marker_name = np.empty(N), np.empty(N), np.empty(N), np.empty(N), []
    for idx, (vertex, attributes) in enumerate(vertices.items()):
        marker_x[idx] = attributes.get('r') * np.cos(attributes.get('phi'))
        marker_y[idx] = attributes.get('r') * np.sin(attributes.get('phi'))
        marker_color[idx] = attributes.get(vertex_color_attribute, default_vertex_color)
        marker_size[idx] = attributes.get(vertex_size_attribute, default_vertex_size) * vertex_scale
        marker_name.append(attributes.get('name', 'vertex_{}'.format(vertex)))
    fig.add_trace(go.Scattergl(
        x=marker_x,
        y=marker_y,
        mode='markers',
        hoverinfo='text+x+y',
        hovertext=marker_name,
        marker_cmin=0,
        marker_cmax=1,
        marker_colorscale='HSV_r',
        marker_size=marker_size,
        marker_color=marker_color,
        showlegend=False,
        marker_reversescale=True,
        marker_opacity=1
    ))

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
        if neighbour < vertex:
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
        if neighbour < vertex:
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