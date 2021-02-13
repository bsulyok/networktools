import numpy as np
import warnings
from itertools import combinations
import random
import plotly.graph_objects as go
import plotly.io as pio
from simulationtools import annealing
from copy import deepcopy

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
    Returns
    -------
    adjacency_matrix : ndarray
        Adjacency matrix indicating edges between vertices.
    adjacency_list : list of lists
        Adjacency list containing edge indices in a concise form.

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
        adjacency_triu_matrix = np.triu(np.random.random((N,N)) < p, k=1)
        return adjacency_triu_matrix | adjacency_triu_matrix.T

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
        return adjacency_triu_matrix | adjacency_triu_matrix.T

    # adjacency list with edge probability
    elif output == 'list' and p is not None:
        adjacency_list = [[] for _ in range(N)]
        for i, j in combinations(range(N), r=2):
            if random.random() < p:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
        return adjacency_list

    # adjacency list with edge probability
    elif output == 'list' and p is None:
        adjacency_list = [[] for _ in range(N)]
        ran = list(range(N))
        while L > 0:
            v1, v2 = random.sample(ran, k=2)
            if v1 not in adjacency_list[v2]:
                adjacency_list[v1].append(v2)
                adjacency_list[v2].append(v1)
                L -= 1
        return adjacency_list
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
            adjacency_list.append({(i+offset)%N for offset in range(-k//2, k//2+1) if offset !=0})
        return adjacency_list
    print('Something went wrong!')
    return

def watts_strogatz_model(N, beta, k, output='matrix'):
    adjacency = regular_ring_lattice(N, k, output)

    if islist(adjacency):
        vertset = set(range(N))
        for i, neigh in enumerate(adjacency):
            for j in range(i+1, i+k//2+1):
                j = j%N
                if random.random() < beta:

                    avail = vertset.difference(neigh.union({i}))
                    r = random.sample(vertset.difference(neigh.union({i})), k=1)[0]
                    adjacency[i].remove(j)
                    adjacency[i].add(r)
                    adjacency[j].remove(i)
                    adjacency[r].add(i)
        return adjacency

    if ismat(adjacency):
        pass #TODO

def stochastic_block_model(P=None, z=None, s=None, output='matrix'):

    '''
    Create a random graph with predetermined community structure.
    Parameters
    ----------
    P : ndarray
        Probability of edges between groups. If not provided a generic sample will be used.
    z : array_like
        Group indices of each vertex.
    s : array_like
        Number of vertices in each group. This is an alternative parameter in place of z.
    output : str
        The kind of the generated network. It can be:
        * "matrix": create an adjacency matrix of the network.
        * "list": create an adjacency list of the network.
    Returns
    -------
    adjacency_matrix : ndarray
        Adjacency matrix indicating edges between vertices.
    adjacency_list : list of lists
        Adjacency list containing edge indices in a concise form.
    '''

    if z is None and s is None:
        raise TypeError('Missing group indices "z" and/or group sizes "s"')
    if z is not None and s is not None:
        warnings.warn('Both group indices "z" and group sizes "s" were provided, the former takes precedence!')
    if output not in ['matrix', 'list']:
        raise TypeError('Wrong output format!')

    if z is None:
        z = [idx for idx, i in enumerate(s) for j in range(i)]

    K = len(s)
    N = len(z)

    if P is None:
        P = 0.9 * np.eye(K) * np.random.random(K) + 0.1*np.random.random((K,K))

    if output == 'matrix':
        return np.random.random((N,N)) < P[z*np.ones((N,1), dtype=bool), np.array(z)[:,None]*np.ones((1,N),dtype=bool)]

    if output == 'list':
        pass #TODO

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

def islist(target):
    return type(target) is list

def ismat(target):
    return type(target) is np.ndarray

def degree(adjacency):
    '''
    Compute the edge degree of each vertex.
    '''
    if islist(adjacency):
        return [len(v) for v in adjacency]
    elif ismat(adjacency):
        return adjacency.sum(0)
    else:
        return 'Wrong input!'

def edge_number(adjacency):
    '''
    Compute the total number of edges.
    '''
    return sum(degree(adjacency))/2

def edge_list(adjacency):
    '''
    Returns an array or list of all unique edges in the graph.
    '''
    if islist(adjacency):
        return [(i,j) for i, neigh in enumerate(adjacency) for j in neigh if j < i]
    elif ismat(adjacency):
        return np.argwhere(np.triu(adjacency, k=1))
    else:
        return 'Wrong input!'

def rearrange(adjacency, perm):
    '''
    Rearrange nodes in an adjacency matrix or list for vanity purposes.
    '''
    N = len(perm)
    inv_perm = [j for i in range(N) for j in perm if perm[j]==i]
    if islist(adjacency):
        return [[perm[j] for j in adjacency[i]] for i in inv_perm]
    elif ismat(adjacency):
        new_adjacency = np.zeros_like(adjacency, dtype=bool)
        for v1, v2 in edge_list(adjacency):
            new_adjacency[perm[v1], perm[v2]] = True
        return new_adjacency | new_adjacency.T

def semi_circle(x1, x2):
    angle = np.linspace(0, np.pi, int(abs(x2-x1) * 1000))
    x_coords = (x1+x2)/2 + abs(x2-x1)/2 * np.cos(angle)
    y_coords = abs(x2-x1)/2 * np.sin(angle)
    return x_coords, y_coords

def diagram_arcs(adjacency):
    fig = go.Figure()
    N = len(adjacency)
    x_coords = np.linspace(0,1,N)
    for i, j in edge_list(adjacency):
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
    return

def quadratic_bezier_curve(p1, p2, p3):

    '''
    Compute the quadratic Bezier curve for the given points

    Parameters
    ----------
    p1, p2, p3 : array_like
        Points defining the Bezier curve.
    num : int
        Number of points on the curve.

    Returns
    ---------
    curve : ndarray
        Point of the generated curve.
    '''

    t = np.linspace(0,1,100)[:,None]
    return p2 + (1-t)**2 * (p1 - p2) + t**2 * (p3 - p2)

def radial_semi_circle(a1, a2):
    p1 = np.array([np.sin(a1), np.cos(a1)])
    p2 = np.array([0,0])
    p3 = np.array([np.sin(a2), np.cos(a2)])
    return quadratic_bezier_curve(p1, p2, p3).T

def diagram_radial(adjacency):
    fig = go.Figure()
    N = len(adjacency)
    angle = np.linspace(0,2*np.pi,N+1)[1:]

    for i, j in edge_list(adjacency):
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
    return

def optimize_arc(adjacency):
    '''
    Optimize vertex arrangement via simulated annealing for the arc type visualization. The elementary step is swapping two vertices, the energy is the sum of edge lengths.
    '''
    adjacency = deepcopy(adjacency)
    N = len(adjacency)

    if islist(adjacency):
        initial_perm = ran = list(range(N))
        def swap_nodes(perm):
            new_perm = deepcopy(perm)
            i, j = random.sample(ran, k=2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = 0
            dE += sum([abs(perm[j]-perm[r])for r in adjacency[i]])
            dE -= sum([abs(perm[i]-perm[r])for r in adjacency[i]])
            dE += sum([abs(perm[i]-perm[r])for r in adjacency[j]])
            dE -= sum([abs(perm[j]-perm[r])for r in adjacency[j]])
            if i in adjacency[j]:
                dE += 2 * abs(perm[i] - perm[j])
            return new_perm, dE

    elif ismat(adjacency):
        initial_perm = np.arange(N)
        def swap_nodes(perm):
            new_perm = deepcopy(perm)
            i, j = np.random.choice(N, size=2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = 0
            dE += sum(abs(perm[adjacency[i]] - perm[j]))
            dE -= sum(abs(perm[adjacency[i]] - perm[i]))
            dE += sum(abs(perm[adjacency[j]] - perm[i]))
            dE -= sum(abs(perm[adjacency[j]] - perm[j]))
            if adjacency[i,j]:
                dE += 2 * abs(perm[i] - perm[j])
            return new_perm, dE

    perm = annealing(swap_nodes, initial_perm, 1000)
    return perm

def optimize_radial(adjacency):
    '''
    Optimize vertex arrangement via simulated annealing for the radial type visualization. The elementary step is swapping two vertices, the energy is the sum of edge lengths.
    '''
    adjacency = deepcopy(adjacency)
    N = len(adjacency)

    if islist(adjacency):
        initial_perm = ran = list(range(N))
        def swap_nodes(perm):
            new_perm = deepcopy(perm)
            i, j = random.sample(ran, k=2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = 0
            dE += sum([abs((perm[j]-perm[r])%N) for r in adjacency[i]])
            dE -= sum(([abs(perm[i]-perm[r])%N) for r in adjacency[i]])
            dE += sum(([abs(perm[i]-perm[r])%N) for r in adjacency[j]])
            dE -= sum(([abs(perm[j]-perm[r])%N) for r in adjacency[j]])
            if i in adjacency[j]:
                dE += 2 * abs(perm[i] - perm[j])
            return new_perm, dE

    elif ismat(adjacency):
        initial_perm = np.arange(N)
        def swap_nodes(perm):
            new_perm = deepcopy(perm)
            i, j = np.random.choice(N, size=2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = 0
            dE += sum((abs(perm[adjacency[i]] - perm[j])%N))
            dE -= sum((abs(perm[adjacency[i]] - perm[i])%N))
            dE += sum((abs(perm[adjacency[j]] - perm[i])%N))
            dE -= sum((abs(perm[adjacency[j]] - perm[j])%N))
            if adjacency[i,j]:
                dE += 2 * abs(perm[i] - perm[j])
            return new_perm, dE

    perm = annealing(swap_nodes, initial_perm, 1000)
    return perm

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
