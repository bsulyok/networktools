import utils
import numpy as np
from common import edge_iterator

def extract_moebius_transformation(z1, w1, z2, w2, z3, w3):
    a = np.linalg.det(np.array([[z1*w1, w1, 1], [z2*w2, w2, 1], [z3*w3, w3, 1]]))
    b = np.linalg.det(np.array([[z1*w1, z1, w1], [z2*w2, z2, w2], [z3*w3, z3, w3]]))
    c = np.linalg.det(np.array([[z1, w1, 1], [z2, w2, 1], [z3, w3, 1]]))
    d = np.linalg.det(np.array([[z1*w1, z1, 1], [z2*w2, z2, 1], [z3*w3, z3, 1]]))
    return np.array([[a, b], [c, d]])

def invert(M):
    return np.array([[M[1,1], -M[0,1]], [-M[1,0], M[0,0]]])

def apply_mobius_transformation(z, M):
    return (M[0,0] * z + M[0,1]) / (M[1,0] * z + M[1,1])

def greedy_embedding(adjacency_list):
    children = utils.minimal_depth_child_search(adjacency_list)
    coord = dict.fromkeys(adjacency_list, 0j)
    n = max([len(family) for family in children.values()])
    
    # declare the relevant Mobius transformations
    last_root = np.exp(-2j*np.pi/n)
    last_half_root = np.exp(-2j*np.pi/2/n)
    sigma = extract_moebius_transformation(1, 1, last_root, -1, last_half_root, -1j)
    isigma = extract_moebius_transformation(1, 1, -1, last_root, -1j, last_half_root)
    a = np.array([[-1, 0], [0, 1]])
    B = [sigma @ np.array([[np.exp(2j*np.pi*k/n), 0], [0, 1]]) @ isigma for k in range(n)]

    # fix the origin
    u = apply_mobius_transformation(0j, sigma)
    v = u.conjugate()
    root_transform = sigma

    def iterative_coordination_search(vertex, transform):
        coord[vertex] = apply_mobius_transformation(v, invert(transform))
        for child_id, child in enumerate(children[vertex], 1):
            child_transform = B[child_id] @ a @ transform
            iterative_coordination_search(child, child_transform)

    # iteratively find the coordinates of all vertices
    for child_id, child in enumerate(children[0]):
        child_transform = B[child_id] @ root_transform
        iterative_coordination_search(child, child_transform)
    return coord

def CCDF(adjacency_list):
    degree = [len(neighbourhood) for neighbourhood in adjacency_list.values()]
    min_degree, max_degree = min(degree), max(degree)
    degree_values = np.arange(min_degree, max_degree+1)
    degree_distribution = np.zeros_like(degree_values)
    for neighbourhood in adjacency_list.values():
        degree_distribution[len(neighbourhood)-min_degree] += 1
    complementary_cumulative_distribution = np.cumsum(degree_distribution[::-1])[::-1]
    return degree_values, complementary_cumulative_distribution

def pre_weighting(adjacency_list):
    degree = {vertex:len(neighbourhood) for vertex, neighbourhood in adjacency_list.items()}
    for vertex, neighbour, attributes in edge_iterator(adjacency_list):
        common_neighbours = set(adjacency_list[vertex].keys()).intersection(set(adjacency_list[neighbour].keys()))
        weight = ( degree[vertex] + degree[neighbour] + degree[vertex] * degree[neighbour] ) / ( 1 + len(common_neighbours) )
        adjacency_list[vertex][neighbour].update({'weight' : weight})
    return adjacency_list

def NCMCE(adjacency_list):
    weighted_adjacency_list = pre_weighting(adjacency_list)
    
