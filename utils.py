import numpy as np
from itertools import combinations
import random
from simulationtools import annealing
from copy import deepcopy
from timeit import default_timer as timer
import heapq

def add_edge(adjacency_list, edge):
    adjacency_list[edge[0]].append(edge[1])
    adjacency_list[edge[1]].append(edge[0])

def remove_edge(adjacency_list, edge):
    adjacency_list[edge[0]].remove(edge[1])
    adjacency_list[edge[1]].remove(edge[0])

def rewire_edge(adjacency_list, edge1, edge2):
    remove(adjacency_list, edge1)
    add(adjacency_list, edge2)

def unique_sample(seq, size):
    samples = {}
    while len(samples) < size:
        samples.add(RNG.choice(seq))
    return samples

def elapsed(func, reps=1):
    start = timer()
    for _ in range(reps):
        func
    dt = timer() - start
    print('Average time over {} rounds: {} s'.format(reps, dt))
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

def islist(target):
    return type(target) is list

def ismat(target):
    return type(target) is np.ndarray

def degree(adjacency):
    '''
    Compute the edge degree of each vertex.
    '''
    if islist(adjacency):
        return np.array([len(v) for v in adjacency])
    elif ismat(adjacency):
        return adjacency.sum(0)

def edge_number(adjacency):
    '''
    Compute the total number of edges.
    '''
    return sum(degree(adjacency))//2

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

def group_sort(adjacency, z, order='index'):
    '''
    Arrange vertices into groups. Order can be "index", "size".
    '''
    N = len(adjacency)
    if islist(adjacency):
        pass #TODO
    elif ismat(adjacency):
        if order == 'index':
            iperm = np.arange(N)[z.argsort()]
        elif order == 'size':
            u, f = np.unique(z, return_counts=True)
            nz = inverse_permutation(u[f.argsort()][::-1])[z]
            iperm = np.arange(N)[nz.argsort()]
        perm = inverse_permutation(iperm)
        return rearrange(adjacency, perm), z[iperm]

def degree_sort(adjacency, descending=True):
    '''
    Sort vertices by their degree. Descending order by default.
    '''
    N = len(adjacency)
    d = np.array(degree(adjacency))
    if descending:
        iperm = np.arange(N)[d.argsort()][::-1]
    else:
        iperm = np.arange(N)[d.argsort()]
    return rearrange(adjacency, inverse_permutation(iperm))

def inverse_permutation(perm):
    if islist(perm):
        return [i for i, j in sorted(enumerate(perm), key=lambda i_j: i_j[1])]
    elif ismat(perm):
        iperm = np.empty_like(perm)
        iperm[perm] = np.arange(len(iperm), dtype=iperm.dtype)
        return iperm

def rearrange(adjacency, perm):
    '''
    Rearrange nodes in an adjacency matrix or list for vanity purposes.
    '''
    N = len(perm)
    iperm = inverse_permutation(perm)
    if islist(adjacency):
        return [[perm[j] for j in adjacency[i]] for i in iperm]
    elif ismat(adjacency):
        new_adjacency = np.zeros_like(adjacency, dtype=bool)
        for v1, v2 in edge_list(adjacency):
            new_adjacency[perm[v1], perm[v2]] = True
        return new_adjacency | new_adjacency.T

def ispercolating(adjacency):
    return len(identify_components(adjacency)) == 1

def identify_components(adjacency):
    N = len(adjacency)

    if islist(adjacency):
        components = []
        identified = []
        for v1 in range(N):
            if v1 not in identified:
                comp = []
                stack = [v1]
                while 0 < len(stack):
                    v2 = stack.pop()
                    identified.append(v2)
                    comp.append(v2)
                    for v3 in adjacency[v2]:
                        if v3 not in identified and v3 not in stack:
                            stack.append(v3)
                components.append(comp)
        return components

    elif ismat(adjacency):
        components = identified = np.zeros(N, dtype=int)
        comp_id = 1
        for v1 in range(N):
            if components[v1] == 0:
                comp = np.zeros(N, bool)
                comp[v1] = True
                old_size, new_size = 0, 1
                while old_size != new_size:
                    comp = comp | adjacency @ comp
                    old_size = new_size
                    new_size = sum(comp)
                components[comp] = comp_id
                comp_id += 1
        return components - 1

def dijsktra(adjacency, source):
    N = len(adjacency)

    if islist(adjacency):
        Q = []
        visited = np.zeros(N, bool)
        distance = np.full(N, np.inf)
        parent = -np.ones(N, int)

        distance[source] = 0
        visited[source] = True
        heapq.heappush(Q, (0, source))

        while 0 < len(Q):
            vertex_dist, vertex = heapq.pop(Q)
            for neighbour in adjacency[vertex]:
                new_dist = vertex_dist + 1


                neighbour_dist = min(distance[neighbour], distance[vertex]+1)

    if ismat(adjacency):
        visited = inqueue = np.zeros(N, bool)
        distance = np.full(N, np.inf)
        vertices = np.arange(N)

        distance[source] = 0
        visited[source] = True
        inqueue[source] = True

        while np.any(inqueue):
            vertex = vertices[inqueue][distance[inqueue].argmin()]
            neighbours = adjacency[vertex]
            distance[neighbours] = np.minimum(distance[neighbours], distance[vertex]+1)
            unvisited_neighbours = neighbours & ~visited & ~inqueue
            inqueue = inqueue | unvisited_neighbours
            inqueue[vertex] = False
            visited[vertex] = True
        return distance





