import utils
import numpy as np

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

def greedy_embedding(G):
    children = utils.minimal_depth_child_search(G.adj)
    coord = dict.fromkeys(G.adj, 0)
    n = max([len(family) for family in children.values()])
    
    # declare the relevant Mobius transformations
    last_root = np.exp(-2j*np.pi/n)
    last_half_root = np.exp(-2j*np.pi/2/n)
    sigma = extract_moebius_transformation(1, 1, last_root, -1, last_half_root, -1j)
    isigma = extract_moebius_transformation(1, 1, -1, last_root, -1j, last_half_root)
    a = np.array([[-1, 0], [0, 1]])
    B = [sigma @ np.array([[np.exp(2j*np.pi*k/n), 0], [0, 1]]) @ isigma for k in range(n)]

    # fix the origin
    u = apply_mobius_transformation(0, sigma)
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