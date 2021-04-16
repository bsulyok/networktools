import numpy as np
import warnings
from itertools import combinations, combinations_with_replacement
import random
from classes import Graph, DiGraph




def empty_graph(N, directed=False):
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
    G = Graph() if not directed else DiGraph()
    for i in range(N):
        G.add_vertex(i)
    return G

def erdos_renyi_graph(N, edge_param=None, directed=False):
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
        edge_param = 3/(N-1) if not directed else 3/2/(N-1)

    G = empty_graph(N, directed)

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

    clique = 2*m
    G = erdos_renyi_graph(clique, m/(clique-1))
    for i in range(clique, N):
        G.add_vertex(i)
    degree = np.array(list(G.degree().values()) + [0 for _ in range(clique, N)])
    degree_sum = sum(degree)
    for newcomer in range(clique, N):
        G.add_vertex(newcomer)
        old_vertices = np.random.choice(newcomer, size=m, replace=False, p=degree[:newcomer]/degree_sum)
        for old_vertex in old_vertices:
            G.add_edge(newcomer, old_vertex)
        degree[old_vertices] += 1
        degree[newcomer] = m
        degree_sum += 2*m
    return G

def regular_ring_lattice(N, k):
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
    G : Graph
    '''
    G = empty_graph(N)
    for vertex in range(N):
        for offset in range(k//2):
            neighbour = (vertex+offset+1)%N
            G.add_edge(vertex, neighbour)
    return G

def watts_stogratz_graph(N, k=2, beta=0.5, output='graph'):
    '''
    Create a random graph according to the Watts-Stograts model.
    Parameters
    ----------
    N : int
        Number of vertices. Must be larger than one.
    k : int
        Coordination number or the number of connections to nearest neighbours.
    Returns
    -------
    G : Graph
    '''
    if N < 3:
        raise TypeError('This model requires at least three vertices.')
    if type(k) is not int or k%2 == 1 or k < 2 or N-2 < k:
        raise TypeError('Coordination number must be a positive even integer smaller than N-2!')
    if N-1 < k:
        raise TypeError('Coordination number "k" is too high.')
    G = regular_ring_lattice(N, k)
    V = list(range(N))
    for vertex, neighbourhood in G.adjacency.items():
        for offset in range(k//2):
            neighbour = (vertex+offset+1)%N
            if random.random() < beta and len(neighbourhood) < N-1:
                new_neighbour = neighbour
                while new_neighbour == neighbour or new_neighbour == vertex or new_neighbour in neighbourhood:
                    new_neighbour = random.choice(V)
                G.remove_edge(vertex, neighbour)
                G.add_edge(vertex, new_neighbour)
    return G

def hyperbolic_distance(r_source, angle_source, r_target, angle_target, curv=1):
    angular_difference = np.pi - abs( np.pi - abs( angle_source - angle_target ) )
    comp1 =  np.cosh( curv * r_source ) * np.cosh( curv * r_target )
    comp2 =  np.sinh( curv * r_source ) * np.sinh( curv * r_target ) * np.cos(angular_difference)
    return np.arccosh( comp1 - comp2 ) / curv

def popularity_similarity_optimisation_model(N, m, beta=0.5, T=0.5, curv=1):

    radial_coordinate = 2/curv*np.log(np.arange(1,N+1))
    angular_coordinate = 2*np.pi*np.random.rand(N)

    if T != 0 and beta == 1:
        cutoff = radial_coordinate - 2 / curv * np.log( T / np.sin( T * np.pi ) * curv * radial_coordinate / m )
    elif T !=0 and beta != 1:
        cutoff = radial_coordinate - 2 / curv * np.log( T / np.sin( T * np.pi ) * ( 1 - np.exp( - curv / 2 * (1-beta) * radial_coordinate ) ) / m / (1-beta) )

    G = empty_graph(N)
    G = Graph()
    for i, (r, angle) in enumerate(zip(radial_coordinate, angular_coordinate)):
        G.add_vertex(i, r=r, angle=angle)

    for i in range(N):
        radial_coordinate[:i] = beta * radial_coordinate[:i] + (1-beta) * radial_coordinate[i]

        if i <= m:
            for j in range(i):
                G.add_edge(i, j)
            continue

        distance = hyperbolic_distance(radial_coordinate[i], angular_coordinate[i], radial_coordinate[:i], angular_coordinate[:i], curv)

        if T == 0:
            for j in distance.argsort()[:m]:
                G.add_edge(i, j)

        else:
            edge_probability = 1 / ( 1 + np.exp( curv / 2 / T * (distance - cutoff[i]) ) )
            for j in np.where(np.random.rand(i) < edge_probability)[0]:
                G.add_edge(i, j)

    return G

    def extended_popularity_similarity_optimisation_model(N, m, beta=0.5, T=0.5, curv=1):

    ar = np.arange(N)
    radial_coordinate = 2/curv*np.log(np.arange(1,N+1))
    angular_coordinate = 2*np.pi*np.random.rand(N)

    if T != 0 and beta == 1:
        cutoff = radial_coordinate - 2 / curv * np.log( T / np.sin( T * np.pi ) * curv * radial_coordinate / m )
    elif T !=0 and beta != 1:
        cutoff = radial_coordinate - 2 / curv * np.log( T / np.sin( T * np.pi ) * ( 1 - np.exp( - curv / 2 * (1-beta) * radial_coordinate ) ) / m / (1-beta) )

    G = empty_graph(N)
    G = Graph()
    for i, (r, angle) in enumerate(zip(radial_coordinate, angular_coordinate)):
        G.add_vertex(i, r=r, angle=angle)

    for i in range(N):
        radial_coordinate[:i] = beta * radial_coordinate[:i] + (1-beta) * radial_coordinate[i]

        if i <= m:
            for j in range(i):
                G.add_edge(i, j)
            continue

        distance = hyperbolic_distance(radial_coordinate[i], angular_coordinate[i], radial_coordinate[:i], angular_coordinate[:i], curv)

        if T == 0:
            for j in distance.argsort()[:m]:
                G.add_edge(i, j)

        else:
            edge_probability = 1 / ( 1 + np.exp( curv / 2 / T * (distance - cutoff[i]) ) )
            for j in np.where(np.random.rand(i) < edge_probability)[0]:
                G.add_edge(i, j)

    return G

ER = erdos_renyi_graph
SBM = stochastic_block_model
BA = barabasi_albert_graph
PSO = popularity_similarity_optimisation_model
EPSO = extended_popularity_similarity_optimisation_model