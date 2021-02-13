import numpy as np
import warnings
from itertools import combinations
import random

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

    if output == 'list':
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

    elif output == 'matrix':
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
