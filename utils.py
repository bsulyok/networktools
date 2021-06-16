import numpy as np
import random
import heapq
import queue
from math import degrees, inf, nan
from common import edge_iterator, disjoint_set

def merge_components(adjacency_list_1, vertices_1, adjacency_list_2, vertices_2):
    if len(vertices_1) < len(vertices_2):
        adjacency_list_1, vertices_1, adjacency_list_2, vertices_2 = adjacency_list_2, vertices_2, adjacency_list_1, vertices_1
    adjacency_list_1, vertices_1 = defragment_indices(adjacency_list_1, vertices_1, start=0)
    adjacency_list_2, vertices_2 = defragment_indices(adjacency_list_2, vertices_2, start=len(vertices_1))
    adjacency_list_1.update(adjacency_list_2)
    vertices_1.update(vertices_2)
    return adjacency_list_1, vertices_1

def defragment_indices(adjacency_list, vertices, start=0):
    N = len(vertices)
    if list(vertices) == list(range(N)):
        return adjacency_list, vertices
    new_adjacency_list, new_vertices = {vertex:{} for vertex in range(start, N+start)}, {vertex:{} for vertex in range(start, N+start)}
    relabel = {old:new for new, old in enumerate(vertices, start=start)}
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        new_adjacency_list[relabel[vertex]].update({relabel[neighbour]:attributes})
    for vertex, attributes in vertices.items():
        new_vertices[relabel[vertex]].update(attributes)
    return new_adjacency_list, new_vertices

def ispercolating(adjacency_list):
    '''
    Decide whether the provided adjacency list describes a large single component.
    '''
    root = list(adjacency_list)[-1]
    visited = {vertex:False for vertex in adjacency_list}
    visited[root] = True
    visited_counter = 1
    vertex_queue = queue.Queue()
    vertex_queue.put(root)
    while not vertex_queue.empty():
        vertex = vertex_queue.get()
        for neighbour in adjacency_list[vertex]:
            if not visited[neighbour]:
                vertex_queue.put(neighbour)
                visited[neighbour] = True
                visited_counter += 1
    return visited_counter == len(adjacency_list)

def identify_components(adjacency_list):
    '''
    Identify disjunct components via a simple percolation algorithm.
    '''
    component_id = dict.fromkeys(adjacency_list, None)
    component_counter = 0
    vertex_queue = queue.Queue()
    for source in adjacency_list:
        if component_id[source] is None:
            component_id[source] = component_counter
            vertex_queue.put(source)
            while not vertex_queue.empty():
                vertex = vertex_queue.get()
                for neighbour in adjacency_list[vertex]:
                    if component_id[neighbour] is None:
                        component_id[neighbour] = component_counter
                        vertex_queue.put(neighbour)
            component_counter += 1
    return component_id

def disjunct_components(adjacency_list, vertices=None):
    if vertices is None:
        vertices = {vertex:{} for vertex in adjacency_list}
    disjunct_comps, component_size = [], []
    component_id = dict.fromkeys(adjacency_list, None)
    component_counter = 0
    vertex_queue = queue.Queue()
    for root in adjacency_list:
        if component_id[root] is None:
            disjunct_comps.append([{root:adjacency_list[root]}, {root:vertices[root]}])
            component_size.append(1)
            component_id[root] = component_counter
            vertex_queue.put(root)
            while not vertex_queue.empty():
                vertex = vertex_queue.get()
                disjunct_comps[component_counter][0].update({vertex:adjacency_list[vertex]})
                disjunct_comps[component_counter][1].update({vertex:vertices[vertex]})
                for neighbour in adjacency_list[vertex]:
                    if component_id[neighbour] is None:
                        component_size[component_counter] += 1
                        component_id[neighbour] = component_counter
                        vertex_queue.put(neighbour)
            component_counter += 1
    return [disjunct_comps[idx] for idx in sorted(range(len(component_size)), key=lambda x:component_size[x], reverse=True)]

def breadth_first_distance(adjacency_list, source, target=None):
    '''
    Find the distance from source while performing a breadth first search.
    If a target vertex is provided the function terminates when the target is reached.
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

def minimum_depth_spanning_tree2(adjacency_list, root=None, directed=False):
    vertex_queue = queue.Queue()
    tree_adjacency_list = {vertex:{} for vertex in adjacency_list}
    degree, depth = {}, {}
    visited = 0
    for vertex, neighbourhood in adjacency_list.items():
        deg = len(neighbourhood)
        degree[vertex] = deg
        if deg == 1:
            vertex_queue.put(vertex)
            depth[vertex] = 0
        else:
            depth[vertex] = None

    while not vertex_queue.empty():
        visited += 1
        vertex = vertex_queue.get()
        for neighbour in adjacency_list[vertex]:
            if depth[neighbour] is None:
                depth[neighbour] = depth[vertex] + 1
                tree_adjacency_list[neighbour].update({vertex:adjacency_list[neighbour][vertex]})
                degree[neighbour] -= 1
                vertex_queue.put(neighbour)
                break
    print(visited)
    



def minimum_depth_spanning_tree(adjacency_list, root=None, directed=False):
    '''
    Find the minimum depth spanning tree rooted at root of the provided graph.
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
    arg = r1**2 + r2**2 - 2 * r1 * r2 * np.cos(phi1 - phi2)
    if arg <= 0:
        return 0
    return np.sqrt( arg )

def poincare_distance(r1, phi1, r2, phi2):
    return np.arccosh( 1 + 2 * euclidean_distance(r1, phi1, r2, phi2) / (1-r1)**2 / (1-r2)**2 )

def hyperbolic_polar_distance(r1, phi1, r2, phi2):
    arg = np.cosh(r1) * np.cosh(r2) - np.sinh(r1) * np.sinh(r2) * np.cos(np.pi - abs(np.pi - abs(phi1 - phi2)))
    if arg <= 1:
        return 0
    return np.arccosh( arg )

def greedy_routing_success_score(adjacency_list, vertices, representation='euclidean'):
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
    badness = {vertex:0 for vertex in adjacency_list}
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
                badness[vertex] += 1
            for neighbour in adjacency_list[vertex]:
                if greedy_success[neighbour] is nan:
                    greedy_success[neighbour] = greedy_success[vertex] / 2
                    badness[neighbour] += greedy_success[neighbour]
    return badness

def greedy_routing_score(adjacency_list, vertices, representation='euclidean'):
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