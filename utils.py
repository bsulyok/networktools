import numpy as np
from copy import deepcopy
import random
import heapq
import queue
from math import degrees, inf, nan
from common import edge_iterator, disjoint_set, ienumerate

def merge_components(adjacency_list_1, vertices_1, adjacency_list_2, vertices_2):
    '''
    Merge the two provided adjacency lists and their respective lists of vertices.
    
    Parameters
    ----------
    adjacency_list_1 : dict of dicts
        Adjacency list containing edge data.
    vertices_1 : dict of dicts
        List containing vertex data.
    adjacency_list_2 : dict of dicts
        Adjacency list containing edge data.
    vertices_2 : dict of dicts
        List containing vertex data.

    Returns
    -------
    adjacency_list_1 : dict of dicts
        Adjacency list containing edge data.
    vertices_1 : dict of dicts
        List containing vertex data.
    '''
    if len(vertices_1) < len(vertices_2):
        adjacency_list_1, vertices_1, adjacency_list_2, vertices_2 = adjacency_list_2, vertices_2, adjacency_list_1, vertices_1
    adjacency_list_1, vertices_1 = defragment_indices(adjacency_list_1, vertices_1, start=0)
    adjacency_list_2, vertices_2 = defragment_indices(adjacency_list_2, vertices_2, start=len(vertices_1))
    adjacency_list_1.update(adjacency_list_2)
    vertices_1.update(vertices_2)
    return adjacency_list_1, vertices_1

def defragment_indices(adjacency_list, vertices, start=0):
    '''
    Reassign vertex identifiers as a list of consequtive integers starting from an arbitrary number.
    
    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    vertices : dict of dicts
        List containing vertex data.
    start : int
        Specifies the smallest index in the new labeling.

    Returns
    -------
    new_adjacency_list : dict of dicts
        Adjacency list containing edge data.
    new_vertices : dict of dicts
        List containing vertex data.
    '''
    if list(adjacency_list) == list(range(len(adjacency_list))):
        return adjacency_list, vertices
    relabel = dict(ienumerate(adjacency_list, start=start))
    new_adjcency_list = {relabel[vertex]:{relabel[neighbour]:attributes for neighbour, attributes in adjacency_list[vertex].items()} for vertex in adjacency_list}
    new_vertices = {relabel[vertex]:attributes for vertex, attributes in vertices.items()}
    return new_adjcency_list, new_vertices

def ispercolating(adjacency_list):
    '''
    Check whether the provided adjacency list describes a single component. The depth first search algorithm is started from the last vertex as the best case scenario results in faster running time with some models.
    
    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.

    Returns
    -------
    ispercolating : bool
        Whether the graph is percolating.
    '''
    visited = []
    def recursive_depth_first_search(vertex):
        visited.append(vertex)
        for neighbour in adjacency_list[vertex]:
            if neighbour not in visited:
                recursive_depth_first_search(neighbour)
    root = next(reversed(adjacency_list))
    recursive_depth_first_search(root)
    return len(visited) == len(adjacency_list)

def disjunct_components(adjacency_list, vertices=None):
    '''
    Break the graph into its disjunct components. Returns a generator object.
    
    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    vertices : dict of dicts
        List containing vertex data.

    Returns
    -------
    disjunct_component_generator : generator
        The generator yields disjunct components in increasing order of their smallest members.
    '''
    if vertices is None:
        vertices = {vertex:{} for vertex in adjacency_list}
    visited = []

    def recursive_depth_first_search(vertex):
        visited.append(vertex)
        component_adjacency_list[vertex] = adjacency_list[vertex]
        component_vertices[vertex] = vertices[vertex]
        for neighbour in adjacency_list[vertex]:
            if neighbour not in visited:
                recursive_depth_first_search(neighbour)

    for root in adjacency_list:
        if root not in visited:
            component_adjacency_list, component_vertices = {}, {}
            recursive_depth_first_search(root)
            yield component_adjacency_list, component_vertices
                
def breadth_first_distance(adjacency_list, source, target=None):
    '''
    Find the distance from source while performing a breadth first search.
    If a target vertex is provided the function terminates when the target is reached.
    
    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    source : int
        The source of the distance search.
    target : int
        The source of the distance search.

    Returns
    -------
    dist : dict of ints OR int
        If dict this contains the distance of all vertices w.r.t. the source; if int this is the distance between source and target.
    '''
    visited = dict.fromkeys(adjacency_list, False)
    visited[source] = True
    vertex_queue = queue.Queue()
    vertex_queue.put(source)
    dist = dict.fromkeys(adjacency_list, inf)
    dist[source] = 0
    while not vertex_queue.empty():
        vertex = vertex_queue.get()
        if vertex == target:
            return dist[vertex]
        for neighbour in adjacency_list[vertex].keys():
            if not visited[neighbour]:
                dist[neighbour] = dist[vertex] + 1
                vertex_queue.put(neighbour)
                visited[neighbour] = True
    return dist

def dijsktra(adjacency_list, source, target=None, weight_attribute='weight'):
    '''
    Find the graph theoretical distance using Dijsktra's algorithm.
    If a target vertex is provided the function terminates when the target is reached.
    
    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    source : int
        The source of the distance search.
    target : int
        The source of the distance search.
    weight_attribute : str
        The weight attribute key that will be checked when extracting edge data.

    Returns
    -------
    dist : dict of floats OR float
        If dict this contains the distance of all vertices w.r.t. the source; if float this is the distance between source and target.
    '''
    N = len(adjacency_list)
    visited = dict.fromkeys(adjacency_list, False)
    dist = dict.fromkeys(adjacency_list, inf)
    dist[source] = 0
    queue_size = 1
    vertex_queue = [ (0, source) ]
    heapq.heapify(vertex_queue)
    while vertex_queue:
        vertex_dist, vertex = heapq.heappop(vertex_queue)
        queue_size -= 1
        if vertex == target:
            return vertex_dist
        if not visited[vertex]:
            visited[vertex] = True
            for neighbour, attributes in adjacency_list[vertex].items():
                if not visited[neighbour]:
                    neighbour_dist = vertex_dist + attributes[weight_attribute]
                    if neighbour_dist < dist[neighbour]:
                        dist[neighbour] = neighbour_dist
                        heapq.heappush(vertex_queue, (neighbour_dist, neighbour))
                        queue_size += 1
    return dist

def distance(adjacency_list, source=None, target=None, weight_attribute=None):
    '''
    A wrapper for calling breadth first search (unweighted) and dijsktra (weighted) in the appropriate setting.
    If no source vertex is provided graph theoretical distances are calculated for all source-target pairs.

    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    source : int
        The source of the distance search.
    target : int
        The source of the distance search.
    weight_attribute : str
        The weight attribute key that will be checked when extracting edge data.

    Returns
    -------
    dist : dict of floats (ints) OR float (int)
        If dict this contains the distance of all vertices w.r.t. the source; if float this is the distance between source and target.
    '''
    if weight_attribute is None:
        if source is None:
            return {vertex:breadth_first_distance(adjacency_list, vertex) for vertex in adjacency_list}
        else:
            return breadth_first_distance(adjacency_list, source, target)
    else:
        if source is None:
            return {vertex:dijsktra(adjacency_list, vertex, None, weight_attribute) for vertex in adjacency_list}
        else:
            return dijsktra(adjacency_list, source, target, weight_attribute)
    
def minimum_depth_spanning_tree(adjacency_list, root=None, directed=False):
    '''
    Find the minimum depth spanning tree rooted at root of the provided graph.

    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    root : int
        The vertex at which the tree is rooted.
    directed : bool
        Whether to return a directed or undirected tree. In the undirected case edges originate from parents and point to children.

    Returns
    -------
    tree_adjacency_list : dict of dicts
        The adjacency list of the minimum depth spanning tree.
    '''
    N = len(adjacency_list)
    tree_adjacency_list = {vertex:{} for vertex in adjacency_list}
    vertex_queue = queue.Queue()
    vertex_queue.put(root)
    visited = dict.fromkeys(adjacency_list, False)
    visited[root] = True
    visited_total = 0
    queue_size = 1
    while 0 < queue_size and visited_total < N:
        vertex = vertex_queue.get()
        queue_size -= 1
        for neighbour in adjacency_list[vertex]:
            if not visited[neighbour]:
                tree_adjacency_list[vertex].update({neighbour:adjacency_list[vertex][neighbour]})
                if not directed:
                    tree_adjacency_list[neighbour].update({vertex:adjacency_list[neighbour][vertex]})
                vertex_queue.put(neighbour)
                queue_size += 1
                visited[neighbour] = True
                visited_total += 1
    return tree_adjacency_list

def minimum_weight_spanning_tree(adjacency_list, weight_attribute='weight'):
    '''
    Find the minimum weight spanning tree rooted at root of the provided graph.

    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    weight_attribute : str
        The weight attribute key that will be checked when extracting edge data.

    Returns
    -------
    tree_adjacency_list : dict of dicts
        The adjacency list of the minimum weight spanning tree.
    '''
    N = len(adjacency_list)
    tree_adjacency_list = {vertex:{} for vertex in adjacency_list}
    disjoint_vertices = disjoint_set(N)
    edge_queue = sorted(edge_iterator(adjacency_list), key=lambda li:li[2][weight_attribute])
    component_size = 1
    for vertex, neighbour, attributes in edge_queue:
        if not disjoint_vertices.is_connected(vertex, neighbour):
            disjoint_vertices.union(vertex, neighbour)
            tree_adjacency_list[vertex].update({neighbour:adjacency_list[vertex][neighbour]})
            tree_adjacency_list[neighbour].update({vertex:adjacency_list[neighbour][vertex]})
            component_size = max(component_size, disjoint_vertices.size(vertex), disjoint_vertices.size(neighbour))
        if component_size == N:
            break
    return tree_adjacency_list    

def euclidean_distance(r1, phi1, r2, phi2):
    '''
    Calculate the euclidean distance between two points characterized by their polar coordinates.

    Parameters
    ----------
    r1 : float
        Radial coordinate of the first point.
    phi1 : float
        Angular coordinate of the first point.
    r2 : float
        Radial coordinate of the second point.
    phi2 : float
        Angular coordinate of the second point.

    Returns
    -------
    dist : float
        The distance between the points.
    '''
    arg = r1**2 + r2**2 - 2 * r1 * r2 * np.cos(phi1 - phi2)
    if arg <= 0:
        return 0
    return np.sqrt( arg )

def poincare_distance(r1, phi1, r2, phi2):
    '''
    Calculate the hyperbolic distance between two points of the Poincare disk model characterized by their polar coordinates.

    Parameters
    ----------
    r1 : float
        Radial coordinate of the first point.
    phi1 : float
        Angular coordinate of the first point.
    r2 : float
        Radial coordinate of the second point.
    phi2 : float
        Angular coordinate of the second point.

    Returns
    -------
    dist : float
        The distance between the points.
    '''
    return np.arccosh( 1 + 2 * euclidean_distance(r1, phi1, r2, phi2) / (1-r1)**2 / (1-r2)**2 )

def hyperbolic_polar_distance(r1, phi1, r2, phi2):
    '''
    Calculate the euclidean distance between two points of the hyperbolic polar plane characterized by their polar coordinates.

    Parameters
    ----------
    r1 : float
        Radial coordinate of the first point.
    phi1 : float
        Angular coordinate of the first point.
    r2 : float
        Radial coordinate of the second point.
    phi2 : float
        Angular coordinate of the second point.

    Returns
    -------
    dist : float
        The distance between the points.
    '''
    arg = np.cosh(r1) * np.cosh(r2) - np.sinh(r1) * np.sinh(r2) * np.cos(np.pi - abs(np.pi - abs(phi1 - phi2)))
    if arg <= 1:
        return 0
    return np.arccosh( arg )

def greedy_routing_success_score(adjacency_list, vertices, representation='euclidean'):
    '''
    Compute the success rate of greedy routes in the provided graph.

    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    vertices : dict of dicts
        List containing vertex data. Must contain polar coordinates 'r' and 'phi'.
    representation : str
        The metric representation of the vertices. Possible values are 'euclidean', 'hyperbolic_polar', 'poincare'. 

    Returns
    -------
    GRS : float
        The greedy routing success rate.
    '''
    GRS = 0
    if representation == 'euclidean':
        distance_function = euclidean_distance
    elif representation == 'hyperbolic_polar':
        distance_function = hyperbolic_polar_distance
    elif representation == 'poincare':
        distance_function = poincare_distance
    for target in adjacency_list.keys():
        target_r, target_phi = vertices[target]['coord']['r'], vertices[target]['coord']['phi']
        metric_distance = {vertex : distance_function(target_r, target_phi, attributes['coord']['r'], attributes['coord']['phi']) for vertex, attributes in vertices.items()}
        greedy_success = dict.fromkeys(adjacency_list, None)
        greedy_success[target] = True
        for vertex in sorted(adjacency_list, key=lambda vert:metric_distance[vert]):
            if greedy_success[vertex] is None:
                greedy_success[vertex] = False
            for neighbour in adjacency_list[vertex]:
                if greedy_success[neighbour] is None:
                    greedy_success[neighbour] = greedy_success[vertex]
                    GRS += 1
    N = len(adjacency_list)
    return GRS / N / (N-1)

def greedy_routing_badness(adjacency_list, vertices, representation='euclidean'):
    '''
    Compute the greedy routing badness score for each vertex of the provided graph. Higher score corresponds to more greedy paths breaking at or near the vertex.

    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    vertices : dict of dicts
        List containing vertex data. Must contain polar coordinates 'r' and 'phi'.
    representation : str
        The metric representation of the vertices. Possible values are 'euclidean', 'hyperbolic_polar', 'poincare'. 

    Returns
    -------
    GRB : dict of floats
        The greedy routing badness score of each vertex.
    '''
    GRB = {vertex:0 for vertex in adjacency_list}
    if representation == 'euclidean':
        distance_function = euclidean_distance
    elif representation == 'hyperbolic_polar':
        distance_function = hyperbolic_polar_distance
    elif representation == 'poincare':
        distance_function = poincare_distance
    for target in adjacency_list.keys():
        target_r, target_phi = vertices[target]['coord']['r'], vertices[target]['coord']['phi']
        metric_distance = {vertex : distance_function(target_r, target_phi, attributes['coord']['r'], attributes['coord']['phi']) for vertex, attributes in vertices.items()}
        greedy_success = dict.fromkeys(adjacency_list, nan)
        greedy_success[target] = 0
        for vertex in sorted(adjacency_list, key=lambda vert:metric_distance[vert]):
            if greedy_success[vertex] is nan:
                greedy_success[vertex] = 1
                GRB[vertex] += 1
            for neighbour in adjacency_list[vertex]:
                if greedy_success[neighbour] is nan:
                    greedy_success[neighbour] = greedy_success[vertex] / 2
                    GRB[neighbour] += greedy_success[neighbour]
    return GRB

def greedy_routing_score(adjacency_list, vertices, representation='euclidean'):
    '''
    Compute the greedy routing score of the provided graph which is the sum of ratio of shortest paths and greedy paths normed by the number of directed pairs of vertices.

    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    vertices : dict of dicts
        List containing vertex data. Must contain polar coordinates 'r' and 'phi'.
    representation : str
        The metric representation of the vertices. Possible values are 'euclidean', 'hyperbolic_polar', 'poincare'. 

    Returns
    -------
    GR : float
        The greedy routing score.
    '''
    GR = 0
    if representation == 'euclidean':
        distance_function = euclidean_distance
    elif representation == 'hyperbolic_polar':
        distance_function = hyperbolic_polar_distance
    elif representation == 'poincare':
        distance_function = poincare_distance
    for target in adjacency_list.keys():
        shortest_path = distance(adjacency_list, target)
        target_r, target_phi = vertices[target]['coord']['r'], vertices[target]['coord']['phi']
        metric_distance = {vertex : distance_function(target_r, target_phi, attributes['coord']['r'], attributes['coord']['phi']) for vertex, attributes in vertices.items()}
        greedy_distance = dict.fromkeys(adjacency_list, nan)
        greedy_distance[target] = 0
        for vertex in sorted(adjacency_list, key=lambda vert:metric_distance[vert]):
            if greedy_distance[vertex] is nan:
                greedy_distance[vertex] = inf
            for neighbour in adjacency_list[vertex]:
                if greedy_distance[neighbour] is nan:
                    greedy_distance[neighbour] = greedy_distance[vertex] + 1
                    GR += shortest_path[neighbour] / greedy_distance[neighbour]
    N = len(adjacency_list)
    return GR / N / (N-1)