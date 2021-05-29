import numpy as np
import random
from copy import deepcopy
from timeit import default_timer as timer
from math import inf, nan
from common import priority_queue, edge_iterator
import disjoint_set

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
    return max(identify_components(adjacency_list).values()) == 1

def identify_components(adjacency_list):
    '''
    Identify disjunct components via a simple percolation algorithm.
    '''
    component = dict.fromkeys(adjacency_list, None)
    counter = 0
    for source in adjacency_list:
        if component[source] is not None:
            continue
        component[source] = counter
        counter += 1
        stack = {source}
        while 0 < len(stack):
            vertex = stack.pop()
            component[vertex] = component[source]
            for neighbour in adjacency_list[vertex]:
                if component[neighbour] is None:
                    stack.add(neighbour)
    return component

def dijsktra(adjacency_list, source, target=None, weight_attribute='weight'):
    '''
    Find the graph theoretical distance using Dijsktra's algorithm.
    If a target vertex is provided the program terminates when the target is reached and only the distance between the source and the target is returned.
    '''
    Q = priority_queue()
    visited = dict.fromkeys(adjacency_list, False)
    dist = dict.fromkeys(adjacency_list, inf)
    dist[source] = 0
    Q.push( (0, source) )
    while 0 < len(Q):
        vertex_dist, vertex = Q.pop()
        if vertex == target:
            return vertex_dist
        if visited[vertex]:
            continue
        visited[vertex] = True
        for neighbour, attributes in adjacency_list[vertex].items():
            if visited[neighbour]:
                continue
            neighbour_dist = vertex_dist + attributes.get(weight_attribute, 1)
            if neighbour_dist < dist[neighbour]:
                dist[neighbour] = neighbour_dist
                Q.push( (neighbour_dist, neighbour) )
    return dist

def greedy_routing_dijsktra(adjacency_list, metric_distance, source):
    greedy_distance = dict.fromkeys(adjacency_list, nan)
    greedy_distance[source] = 0
    for vertex, distance in sorted(metric_distance.items(), key=lambda li: li[1]):
        if greedy_distance[vertex] is nan:
            greedy_distance[vertex] = inf
        for neighbour in adjacency_list[vertex]:
            if greedy_distance[neighbour] is nan and distance < metric_distance[neighbour]:
                greedy_distance[neighbour] = greedy_distance[vertex] + 1
    return greedy_distance

def minimal_depth_child_search(adjacency_list, root=0):
    '''
    Find the system of ascendancy in a minimal depth tree rooted at root of the provided graph.
    '''
    distance_from_root = dijsktra(adjacency_list, root)
    if max(distance_from_root.values()) is inf:
        raise TypeError('The graph is not connected!')
    children = {vertex:list() for vertex in adjacency_list}
    for vertex, neighbours in adjacency_list.items():
        if vertex is root:
            continue
        parent = None
        for neighbour in neighbours:
            if parent is None or distance_from_root[neighbour] < distance_from_root[parent]:
                parent = neighbour
        children[parent].append(vertex)
    return children

def complex_distance(p1, p2):
    return abs(p1 - p2)

def poincare_hyperbolic_distance(p1, p2):
    return np.arccosh( 1 + 2 * abs(p1-p2)**2 / (1 - abs(p1)**2) / (1 - abs(p2)**2) )

def greedy_routing_score(adjacency_list, coords, distance_function=complex_distance):
    N = len(coords)
    GR = 0
    for target in adjacency_list.keys():
        shortest_path = dijsktra(adjacency_list, target)
        metric_distance = {vertex : distance_function(coords[target], coord) for vertex, coord in coords.items()}
        greedy_path = greedy_routing_dijsktra(adjacency_list, metric_distance, target)
        for sp, gp in zip(shortest_path.values(), greedy_path.values()):
            if sp != 0:
                GR += sp/gp
    return GR / N / (N-1)

def minimum_weight_spanning_tree(adjacency_list):
    tree_adjacency_list = dict.fromkeys(adjacency_list, None)
    disjoint_vertices = disjoint_set.DisjointSet()
    #for vertex, neighbour, attributes in edge_iterator(adjacency_list):

    