import numpy as np
import warnings
from itertools import combinations, combinations_with_replacement
import random
from classes import Graph

def empty_graph(N):
    '''
    Create an empty graph of given size.
    Parameters
    ----------
    N:int
        Number of vertices.
    Returns
    -------
    G : Graph
        Generated empty graph
    '''
    G = Graph()
    for i in range(N):
        G.add_vertex(i)
    return G

def erdos_renyi_graph(N, edge_param=None):
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
    Returns
    -------
    adjacency_list : list of lists
        Adjacency list containing edge indices in a concise form.
    '''
    if N < 2:
        raise TypeError('This model requires at least two vertices.')

    if edge_param is None:
        edge_param = 3/(N-1)

    G = empty_graph(N)

    # adjacency defined by edge probability
    if 0 < edge_param < 1:
        p = edge_param
        for i, j in combinations(range(N), r=2):
            if random.random() < p:
                G.add_edge(i, j)

    # adjacency defined by binomial vertex selection
    elif type(edge_param) is int and 0 < edge_param < N*(N-1)/2:
        L = edge_param
        ran = list(range(N))
        while L > 0:
            i, j = random.sample(ran, k=2)
            if (i,j) not in G:
                G.add_edge(i, j)
                L -= 1
    else:
        raise TypeError('Wrong edge parameter!')

    return G

def SBMP(s):
    K = len(s)
    N = sum(s)
    scale, exp_degree = 0.1, N**(1/2)
    P = [[0 for j in range(K)] for i in range(K)]
    for i, j in combinations_with_replacement(range(K), r=2):
        prob = random.normalvariate(exp_degree/((1-scale)*s[i]+scale*N), 0.2)
        if i == j:
            P[i][i] = prob
        else:
            prob2 = abs(random.normalvariate(prob, 0.05))
            P[i][j] = prob2
            P[j][i] = prob2
    return P

def stochastic_block_model(s, P=None):
    '''
    Create a random graph with predetermined community structure.
    Parameters
    ----------
    s : array_like
        Number of vertices in each group. This is an alternative parameter in place of z.
    P : ndarray
        Probability of edges between groups. If not provided a generic sample will be used.
    Returns
    -------
    G : Graph
        Generated SBM graph.
    '''
    K = len(s)
    N = sum(s)

    if P is None:
        P = [[0 for j in range(K)] for i in range(K)]
        for i, j in combinations_with_replacement(range(K), r=2):
            if i == j:
                P[i][j] = abs(random.normalvariate(3/(s[i]-1), 0.05))
            else:
                rnd = abs(random.normalvariate(0.25/(max(s[i], s[j])-1), 0.025))
                P[i][j] = rnd
                P[i][j] = rnd

    P = SBMP(s) if P is None else P
    z = [i for i,groupsize in enumerate(s) for _ in range(groupsize)]
    G = empty_graph(N)

    for i, j in combinations(range(N), r=2):
        if random.random() < P[z[i]][z[j]]:
            G.add_edge(i, j)
    return G

def barabasi_albert_graph(N, m):
    '''
    Create a Barabasi-Albert random graph. The initial clique is of size 2m.
    Parameters
    ----------
    N : int
        Number of vertices. Must be larger than one.
    m : int
        Number of edges brought in by new vertices.
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
    if type(N) is not int or N < 8:
        raise TypeError('N must be an integer larger than 8')
    if type(m) is not int or m < 1 or N < 2*m:
        raise TypeError('m must be a positive integer not larger than N/2')

    G = erdos_renyi_graph(2*m, m/(2*m-1))


    clique = 2*m

    adjacency_list = erdos_renyi_graph(clique, clique**2 // 4, output='list')
    stubs = np.concatenate(([len(i) for i in adjacency_list], np.zeros(N-clique)))
    stubs[:clique] += 1
    stubs = np.concatenate((np.ones(clique), np.zeros(N-clique)))
    stub_sum = sum(stubs)
    for new_vertex in range(clique, N):
        old_vertices = np.random.choice(new_vertex, size=m, replace=False, p=stubs[:new_vertex]/stub_sum)
        for old_vertex in old_vertices:
            adjacency_list[old_vertex].append(new_vertex)
        adjacency_list.append(list(old_vertices))
        stubs[old_vertices] += 1
        stubs[new_vertex] += m
        stub_sum += 2*m
    return Graph(adjacency_list)

def regular_ring_lattice(N, k):
    return [[j%N for j in range(i-k//2, i+k//2+1) if j!=i] for i in range(N)]

def watts_stogratz_graph(N, k=2, beta=0.5, output='graph'):

    '''
    Create a regular ring lattice.
    Parameters
    ----------
    N : int
        Number of vertices. Must be larger than one.
    k : int
        Coordination number or the number of connections to nearest neighbours.
    Returns
    -------
    adjacency_list : list of lists
        Adjacency list containing edge indices in a concise form.
    '''

    if N < 3:
        raise TypeError('This model requires at least three vertices.')
    if type(k) is not int or k%2 == 1 or k < 2 or N-2 < k:
        raise TypeError('Coordination number must be a positive even integer smaller than N-2!')
    if N-1 < k:
        raise TypeError('Coordination number "k" is too high.')

    V = list(range(N))
    adjacency_list = regular_ring_lattice(N, k)
    for i in range(N):
        for j in range(i+1, i+k//2+1):
            j = j%N
            if random.random() < beta and len(adjacency_list[i]) < N-1:
                l = j
                while l == i or l in adjacency_list[i]:
                    l = random.choice(V)
                rewire_edge(adjacency_list, (i,j), (i,l))
    return Graph(adjacency_list)
