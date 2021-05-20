import numpy as np
import plotly.graph_objects as go
import utils
from math import pi, sin, cos
from common import edge_iterator
from drawing_tools import line, semi_circle, circular_arc, edge_trace, quadratic_bezier_curve, kamada_kawai

HEIGHT = 1000

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
    vert_dict = {ID:idx for idx, ID in enumerate(adjacency_list.keys())}
    fig = go.Figure()

    # draw the edges
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        if vertex < neighbour:
            x1, x2 = vert_dict[vertex]/N, vert_dict[neighbour]/N
            angle = np.linspace(0, np.pi, edge_resolution)
            x_path = (x1+x2)/2 + abs(x1-x2) / 2 * np.cos(angle)
            y_path = abs(x1-x2) / 2 * np.sin(angle)
            fig.add_traces(edge_trace(x_path, y_path, 1, color_resolution))

    # draw the vertices
    fig.add_trace(go.Scattergl(x=np.arange(N)/(N-1), y=np.zeros(N), mode='markers', marker_size=10, marker_color='blue', showlegend=False, text='a'))

    # figure settings
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20)
    fig.show()
    return

def hyperbolic(adjacency_list, vertices, euclidean=False):
    '''
    Draw the hyperbolic representation of the provided network on the Poincare disc.
    Parameters
    ----------
    adjacency_list : list of lists
        Adjacency list containing edge data.
    vertices : list of dicts
        List of vertex data. Must contain "r" and "angle" values.
    euclidean : boolean
        Whether to draw hyperbolic or euclidean lines. Default is False
    '''
    edge_resolution, color_resolution = 100, 10
    N = len(adjacency_list)
    edge_width, vertex_size = 1, 10
    r = np.array([attr['r'] for attr in vertices.values()])
    phi = np.array([attr['phi'] for attr in vertices.values()])
    r = r / r.max()
    vert = r * np.exp(1j * phi)
    #vert = 2 * vert / ( 1 + vert * vert.conjugate())
    vert_dict = {ID:idx for idx, ID in enumerate(vertices.keys())}
    fig = go.Figure()

    # draw the edges
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        if vertex < neighbour:
            i, j = vert_dict[vertex], vert_dict[neighbour]
            x1, y1 = r[i] * np.cos(phi[i]), r[i] * np.sin(phi[i])
            x2, y2 = r[j] * np.cos(phi[j]), r[j] * np.sin(phi[j])
            if euclidean or r[i] == 0 or r[j] == 0 or abs((phi[i]-phi[j])%np.pi) < 1e-5:
                path = line(vert[i], vert[j], color_resolution+1)
            else:
                path = circular_arc(vert[i], vert[j], edge_resolution)
            fig.add_traces(edge_trace(path.real, path.imag, 1, color_resolution))

    # draw the vertices
    fig.add_trace(go.Scattergl(x=vert.real, y=vert.imag, mode='markers', marker_size=vertex_size, marker_color='rgb(255,0,0)', showlegend=False))

    # figure settings
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20)
    fig.show()
    return

def circular(adjacency_list, euclidean=False):
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
            if euclidean or abs((angle[i]-angle[j])%np.pi) < 1e-5:
                x_path, y_path = line(x1, y1, x2, y2, color_resolution+1)
            else:
                x_path, y_path = circular_arc(x1, y1, x2, y2, edge_resolution)
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
