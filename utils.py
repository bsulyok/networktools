import numpy as np
from itertools import combinations
import random
from copy import deepcopy
from timeit import default_timer as timer
from math import inf
from common import priority_queue
import heapq

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
    N = len(adjacency_list)
    adjacency_matrix = np.zeros((N,N), bool)
    for i, neigh in enumerate(adjacency_list):
        adjacency_matrix[i, neigh] = True
        adjacency_matrix[neigh, i] = True
    return adjacency_matrix

def edge_list(adjacency_list):
    return ((i,j) for i, neigh in enumerate(adjacency_list) for j in neigh if j < i)

def islist(target):
    return type(target) is list

def ismat(target):
    return type(target) is np.ndarray

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

def ispercolating(adjacency_list):
    return len(identify_components(adjacency_list)) == 1

def identify_components(adjacency_list):
    '''
    Identify disjunct components via a simple percolation algorithm.
    '''
    components = []
    identified = []
    for v1 in range(len(adjacency_list)):
        if v1 not in identified:
            comp = []
            stack = [v1]
            while 0 < len(stack):
                v2 = stack.pop()
                identified.append(v2)
                comp.append(v2)
                for v3 in adjacency_list[v2]:
                    if v3 not in identified and v3 not in stack:
                        stack.append(v3)
            components.append(comp)
    return components

def dijsktra(adjacency_list, source, target=None):
    '''
    Find the graph theoretical distance using the Dijsktra algorithm.
    If a target node is provided the program terminates when target is reached and only the distance between source and target is returned.
    '''
    N = len(adjacency_list)
    Q = priority_queue()
    visited = [False for i in range(N)]
    distance = [inf for i in range(N)]
    distance[source] = 0
    Q.push( (0, source) )
    while 0 < len(Q):
        vertex_dist, vertex = Q.pop()
        if visited[vertex]:
            continue
        visited[vertex] = True
        if target is not None and vertex == target:
            return vertex_dist
        for neighbour in adjacency_list[vertex]:
            if visited[neighbour]:
                continue
            new_dist = vertex_dist + 1
            if new_dist < distance[neighbour]:
                distance[neighbour] = new_dist
                Q.push( (new_dist, neighbour) )
    return distance

def graph_theoretical_distance(adjacency_list, source=None, target=None):
    '''
    Calculate the graph theoretical distance for all pairs of vertices.
    '''
    if source is None and target is None:
        return [dijsktra(adjacency_list, source) for source in range(len(adjacency_list))]
    elif source is not None and target is None:
        return dijsktra(adjacency_list, source)
    elif source is not None and target is not None:
        return dijsktra(adjacency_list, source, target)
    else:
        return 'Wrong input!'
