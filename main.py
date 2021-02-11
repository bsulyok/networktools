import numpy as np
import warnings
from itertools import combinations
import plotly.graph_objects as go
import plotly.io as pio


def erdos_renyi_random_graph(N, p=None, L=None, output='matrix'):
    '''
    Create an Erdos-Renyi random graph.
    Parameters
    ----------
    N : int
        Number of vertices. Must be larger than one.
    p : float
        Probability of links. Must be in [0,1].
    L : int
        Number of links. If "p" and "L" both provided the former takes precedence.
    output : str
        The kind of the generated network. It can be:
        * "matrix": create an adjacency matrix of the network.
        * "list": create an adjacency list of the network.
    '''
    if N < 2:
        raise TypeError('This model requires at least two vertices.')
    if p is None and L is None:
        raise TypeError('Missing edge probability "p" and/or edge number "L"')
    if p is not None and L is not None:
        warnings.warn('Both edge probability "p" and edge number "L" was provided, the former takes precedence!')
    if p is None and N*(N-1)/2 < L:
        raise TypeError('Edge number "L" is too large!')
    if output not in ['matrix', 'list']:
        raise TypeError('Wrong output format!')

    # adjacency matrix with edge probability
    if output == 'matrix' and p is not None:
        adjacency_triu_matrix = np.triu(np.random.random((N,N)) < p, k=0)
        return adjacency_triu_matrix + adjacency_triu_matrix.T

    # adjacency matrix with edge number
    elif output == 'matrix' and p is None:
        adjacency_triu_matrix = np.zeros((N,N), dtype=bool)
        while L > 0:
            v1, v2 = np.random.choice(N, size=2, replace=False)
            if v2 < v1:
                v1, v2 = v2, v1
            if not adjacency_triu_matrix[v1, v2]:
                adjacency_triu_matrix[v1, v2] = True
                L -= 1
        return adjacency_triu_matrix + adjacency_triu_matrix.T

    # adjacency list with edge probability
    elif output == 'list' and p is not None:
        adjacency_list = [[] for _ in range(N)]
        for i, j in combinations(range(N), r=2):
            if np.random.random() < p:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
        return adjacency_list

    # adjacency list with edge probability
    elif output == 'list' and p is None:
        adjacency_list = [[] for _ in range(N)]
        while L > 0:
            v1, v2 = np.random.choice(N, size=2, replace=False)
            if j not in adjacency_list[i]:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
                L -= 1
    print('Something went wrong!')
    return

def regular_ring_lattice(N, k=2, output='matrix'):
    '''
    Create a regular ring lattice.
    Parameters
    ----------
    N : int
        Number of vertices. Must be larger than one.
    k : int
        Coordination number or the number of connections to nearest neighbours.
    output : str
        The format of the generated lattice. It can be:
        * "matrix" (default): create an adjacency matrix of the network.
        * "list": create an adjacency list of the network.
    '''
    if N < 3:
        raise TypeError('This model requires at least three vertices.')
    if type(k) is not int or k%2 != 0 or k < 2:
        raise TypeError('Coordination number "k" must be a positive even integer!')
    if N-1 < k:
        raise TypeError('Coordination number "k" is too high.')
    if output not in ['matrix', 'list']:
        raise TypeError('Wrong output format!')


    if output == 'matrix':
        adjacency_matrix = np.zeros((N,N), dtype=bool)
        r = np.arange(N)
        for offset in range(1, k//2+1):
            adjacency_matrix[r, r-offset] = True
            adjacency_matrix[r-offset, r] = True
        return adjacency_matrix

    elif output == 'list':
        adjacency_list = []
        for i in range(N):
            adjacency_list.append([(i+offset)%N for offset in range(-k//2, k//2+1) if offset !=0])
        return adjacency_list
    print('Something went wrong!')
    return

def mat2list(adjacency_matrix):
    '''
    Produce the adjacency list of the given adjacency matrix.
    '''
    N = len(adjacency_matrix)
    adjacency_list = [[] for _ in range(N)]
    for i, j in np.argwhere(np.triu(adjacency_matrix, k=1)):
        adjacency_list[i].append(j)
        adjacency_list[j].append(i)
    return adjacency_list

def list2mat(adjacency_list):
    '''
    Produce the adjacency matrix of the given adjacency list.
    '''
    N = len(adjacency_list)
    adjacency_matrix = np.zeros((N,N), dtype=bool)
    for i in range(N):
        for j in adjacency_list[i]:
            adjacency_matrix[i,j] = True
    return adjacency_matrix

def edge_number(adjacency_object):
    '''
    Compute the total number of edges.
    '''
    if type(adjacency_object) == list:
        return sum([len(adjacency_object[i] for i in range(len(adjacency_object)))])//2
    elif type(adjacency_object) == np.ndarray:
        return adjacency_object.sum()//2

def degree(adjacency_object):
    '''
    Compute the edge degree of each vertex.
    '''
    if type(adjacency_object) == list:
        return [len(adjacency_object[i] for i in range(len(adjacency_object)))]
    elif type(adjacency_object) == np.ndarray:
        return adjacency_object.sum(0)

def semi_circle(x1, x2):
    angle = np.linspace(0, np.pi, int(abs(x2-x1) * 1000))
    x_coords = (x1+x2)/2 + abs(x2-x1)/2 * np.cos(angle)
    y_coords = abs(x2-x1)/2 * np.sin(angle)
    return x_coords, y_coords

def arc_diagram(adjacency_object):
    fig = go.Figure()
    N = len(adjacency_object)
    x_coords = np.linspace(0,1,N)

    if type(adjacency_object) == list:
        edges = [(i,j) for i, neigh in enumerate(A) for j in neigh if j < i]
    elif type(adjacency_object) == np.ndarray:
        edges = np.argwhere(np.triu(adjacency_object, k=1))

    for i, j in edges:
        sc = semi_circle(x_coords[i], x_coords[j])
        fig.add_trace(go.Scatter(
            x=sc[0],
            y=sc[1],
            mode='lines',
            line_width=1,
            line_color='red',
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=x_coords,
        y=np.zeros(N),
        mode='markers',
        marker_size=20,
        marker_color='blue',
        showlegend=False
    ))

    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x', scaleratio=1)
    fig.show()
    return fig

def quadratic_bezier_curve(p1, p2, p3):
    t = np.linspace(0,1,100)[:,None]
    return p2 + (1-t)**2 * (p1 - p2) + t**2 * (p3 - p2)

def radial_semi_circle(a1, a2):
    p1 = np.array([np.sin(a1), np.cos(a1)])
    p2 = np.array([0,0])
    p3 = np.array([np.sin(a2), np.cos(a2)])
    return quadratic_bezier_curve(p1, p2, p3).T

def radial_diagram(adjacency_object):
    fig = go.Figure()
    N = len(adjacency_object)
    angle = np.linspace(0,2*np.pi,N+1)[1:]

    if type(adjacency_object) == list:
        edges = [(i,j) for i, neigh in enumerate(A) for j in neigh if j < i]
    elif type(adjacency_object) == np.ndarray:
        edges = np.argwhere(np.triu(adjacency_object, k=1))

    for i, j in edges:
        rsc = radial_semi_circle(angle[i], angle[j])
        fig.add_trace(go.Scatter(
            x=rsc[0],
            y=rsc[1],
            mode='lines',
            line_width=1,
            line_color='red',
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=np.sin(angle),
        y=np.cos(angle),
        mode='markers',
        marker_size=20,
        marker_color='blue',
        showlegend=False
    ))

    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x', scaleratio=1)
    fig.show()
    return fig

A = erdos_renyi_random_graph(N=15, p=0.3, output='list')
#fig=radial_diagram(A)


class network:
    '''
    Create a random network with the given inputs.
    Parameters
    ----------
    type : string
        The generative algorithm to be used. The following methods apply with their respective parameters:
            * "Erdos-Renyi"
            * "SBM"
            * "Barabasi-Albert"
            * "Watts-Strogatz"
    '''

    def __init__(self, type='erdos-renyi', **args):
        pass




