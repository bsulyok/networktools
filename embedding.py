import utils
import numpy as np
from common import edge_iterator
from scipy.special import zeta
from copy import deepcopy

def extract_moebius_transformation(z1, w1, z2, w2, z3, w3):
    a = np.linalg.det(np.array([[z1*w1, w1, 1], [z2*w2, w2, 1], [z3*w3, w3, 1]]))
    b = np.linalg.det(np.array([[z1*w1, z1, w1], [z2*w2, z2, w2], [z3*w3, z3, w3]]))
    c = np.linalg.det(np.array([[z1, w1, 1], [z2, w2, 1], [z3, w3, 1]]))
    d = np.linalg.det(np.array([[z1*w1, z1, 1], [z2*w2, z2, 1], [z3*w3, z3, 1]]))
    return np.array([[a, b], [c, d]])

def invert_mobius(M):
    return np.array([[M[1,1], -M[0,1]], [-M[1,0], M[0,0]]])

def apply_mobius_transformation(z, M):
    return (M[0,0] * z + M[0,1]) / (M[1,0] * z + M[1,1])

def calculate_CCDF(degree):
    min_degree, max_degree = min(degree), max(degree)
    degree_values = np.arange(min_degree, max_degree + 1)
    degree_distribution = np.zeros_like(degree_values)
    for deg in degree:
        degree_distribution[deg - min_degree] += 1
    return degree_values, np.cumsum(degree_distribution[::-1])[::-1]

def estimate_beta(degree, min_samples=50):
    degree_values = np.unique(degree)
    min_degree, max_degree = min(degree_values), max(degree_values)
    degree_axis = np.arange(min_degree, max_degree + 1)
    degree_distribution = np.zeros_like(degree_axis)
    for deg in degree:
        degree_distribution[deg - min_degree] += 1
    CCDF = np.cumsum(degree_distribution[::-1])[::-1]
    gamma, D = None, None
    for deg_min in degree_values:
        proper_degree_list = degree[deg_min <= degree]
        if min_samples <= len(proper_degree_list):
            new_gamma = 1 + 1 / np.mean(np.log( proper_degree_list / (deg_min - 1/2) ))
            deg_min_idx = np.where(degree_axis == deg_min)[0][0]
            degrees_in_tail = degree_axis[deg_min_idx:]
            distribution_in_tail = CCDF[deg_min_idx:]
            const = np.exp(np.mean(np.log(distribution_in_tail) + (new_gamma-1) * np.log(degrees_in_tail) )) * (new_gamma-1)
            Z = zeta(new_gamma, deg_min)
            new_D = np.max(np.abs( distribution_in_tail / const / Z - degrees_in_tail ** (1-new_gamma) / (new_gamma-1) / Z ))
            if D is None or new_D < D:
                gamma, D = new_gamma, new_D
        else:
            break
    return 1 / (gamma-1)

def pre_weighting(adjacency_list):
    degree = {vertex:len(neighbourhood) for vertex, neighbourhood in adjacency_list.items()}
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        common_neighbours = set(adjacency_list[vertex].keys()).intersection(set(adjacency_list[neighbour].keys()))
        weight = ( degree[vertex] + degree[neighbour] + degree[vertex] * degree[neighbour] ) / ( 1 + len(common_neighbours) )
        adjacency_list[vertex][neighbour].update({'weight' : weight})
    return adjacency_list

def circular_adjustment(angular_coordinates):
    return (angular_coordinates - min(angular_coordinates)) / (max(angular_coordinates) - min(angular_coordinates)) * 2 * np.pi

def equidistant_adjustment(angular_coordinates):
    d_phi = 2 * np.pi / len(angular_coordinates)
    phi = np.empty_like(angular_coordinates)
    for idx, vertex in enumerate(np.argsort(angular_coordinates)):
        phi[vertex] = idx * d_phi
    return phi

def greedy_embedding(adjacency_list, vertices=None, representation='hyperbolic_polar'):
    if vertices is None:
        vertices = {vertex:{} for vertex in adjacency_list}
    else:
        vertices = deepcopy(vertices)
    
    min_dist = {vertex:min(dist.values()) for vertex, dist in utils.distance(adjacency_list).items()}
    root = min(min_dist, key=lambda vertex:min_dist[vertex])

    children = utils.minimum_depth_spanning_tree(adjacency_list, root=root, directed=True)
    tree_degree = len(adjacency_list[root])
    
    # declare the relevant Mobius transformations
    last_root = np.exp(-2j*np.pi/tree_degree)
    last_half_root = np.exp(-2j*np.pi/2/tree_degree)
    sigma = extract_moebius_transformation(1, 1, last_root, -1, last_half_root, -1j)
    isigma = extract_moebius_transformation(1, 1, -1, last_root, -1j, last_half_root)
    A = np.array([[-1, 0], [0, 1]])
    B = [sigma @ np.array([[np.exp(2j*np.pi*k/tree_degree), 0], [0, 1]]) @ isigma for k in range(tree_degree)]

    # fix the origin
    u = apply_mobius_transformation(0j, sigma)
    v = u.conjugate()
    root_transform = sigma

    def iterative_coordination_search(vertex, transform):
        coord = apply_mobius_transformation(v, invert_mobius(transform))
        r = abs(coord)
        if representation == 'hyperbolic_polar':
            r = np.arccosh( 1 + 2 * r**2 / (1 - r**2) )
        vertices[vertex].update({'r':r, 'phi': np.arctan2(coord.imag, coord.real)})
        for child_id, child in enumerate(children[vertex], 1):
            child_transform = B[child_id] @ A @ transform
            iterative_coordination_search(child, child_transform)

    vertices[root].update({'r':0, 'phi':0})
    # iteratively find the coordinates of all vertices
    for child_id, child in enumerate(children[root]):
        child_transform = B[child_id] @ root_transform
        iterative_coordination_search(child, child_transform)
    return vertices

def ncMCE(adjacency_list, vertices=None, angular_adjustment=equidistant_adjustment, representation='hyperbolic_polar'):    
    if vertices is None:
        vertices = {vertex:{} for vertex in adjacency_list}
    else:
        vertices = deepcopy(vertices)

    weighted_adjacency_list = pre_weighting(adjacency_list)
    min_tree_adjacency_list = utils.minimum_weight_spanning_tree(weighted_adjacency_list)
    vertex_distance = utils.distance(min_tree_adjacency_list)
    D = np.array([[vertex_distance[vertex][neighbour] for neighbour in min_tree_adjacency_list] for vertex in min_tree_adjacency_list])
    U, S, VH = np.linalg.svd(D, full_matrices=False)
    S[2:] = 0
    coordinates = np.transpose(np.sqrt(np.diag(S)) @ VH)
    angular_coordinates = angular_adjustment(coordinates[:,1])
    degree = np.array([len(neighbourhood) for neighbourhood in adjacency_list.values()])
    beta = estimate_beta(degree)
    radial_coordinates = np.empty_like(angular_coordinates)
    radial_coordinates[np.argsort(degree)[::-1]] = (2 * ( beta * np.log(np.linspace(1, len(adjacency_list), len(adjacency_list))) + (1-beta)*np.log(len(adjacency_list))))
    if representation == 'poincare':
        radial_coordinates = np.tanh(radial_coordinates/2) ** 2
    for idx, vertex in enumerate(vertices):
        vertices[vertex].update({'r':radial_coordinates[idx], 'phi': angular_coordinates[idx] })
    return vertices

def mercator(adjacency_list, vertices=None, representation='hyperbolic_polar'):
    #TODO
    if vertices is None:
        vertices = {vertex:{} for vertex in adjacency_list}
    else:
        vertices = deepcopy(vertices)

    return vertices
    
def hypermap(adjacency_list, vertices=None, representation='hyperbolic_polar'):
    #TODO
    if vertices is None:
        vertices = {vertex:{} for vertex in adjacency_list}
    else:
        vertices = deepcopy(vertices)
    
    return vertices