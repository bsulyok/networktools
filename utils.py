import numpy as np
from itertools import combinations
import random
from simulationtools import annealing
from copy import deepcopy

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
