import numpy as np
from itertools import combinations
from math import inf, pi, sin, cos

HEIGHT = 1000

def semi_circle_np(x1, x2):
    angle = np.linspace(0, np.pi, int(abs(x2-x1) * 1000))
    X = (x1+x2)/2 + abs(x2-x1)/2 * np.cos(angle)
    Y = abs(x2-x1)/2 * np.sin(angle)
    return X, Y

def semi_circle(x1, x2):
    X, Y = [], []
    N = int(abs(x2-x1)*1000)
    incr = pi/(N-1)
    center = (x1+x2)/2
    radius = abs(x1-x2)/2
    for asd in range(N):
        angle = asd*incr
        X.append(center+radius*cos(angle))
        Y.append(radius*sin(angle))
    return X, Y

def quadratic_bezier_curve(p1, p2, p3):
    '''
    Compute the quadratic Bezier curve for the given points
    Parameters
    ----------
    p1, p2, p3 : array_like
        Points defining the Bezier curve.
    num : int
        Number of points on the curve.
    Returns
    ---------
    curve : ndarray
        Point of the generated curve.
    '''
    N = 100
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    t = np.linspace(0,1,N)[:,None]
    return (p2 + (1-t)**2 * (p1 - p2) + t**2 * (p3 - p2)).T

def kamada_kawai(adjacency_list, graph_distance):
    N = len(adjacency_list)
    graph_distance = np.array(graph_distance)
    L0, K = 1.0, N
    diameter = graph_distance.max()
    if diameter == inf:
        return
    L = L0 / diameter
    des_length = L * graph_distance
    springs = K / (graph_distance + 1e-3*np.eye(N))**2
    np.fill_diagonal(springs, 0)
    angle = 2 * np.pi * np.linspace(0,1, N+1)[1:]
    pairs = combinations(range(N), r=2)
    invdist = 1 / (graph_distance + 1e-3 * np.eye(N))
    meanweight = 1e-3
    initial_position_array = np.hstack([np.sin(angle), np.cos(angle)]).ravel('F')

    def kamada_kawai_cost_function(position_array):
        coords = position_array.reshape(N, 2)
        coord_diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        vert_dist = np.sqrt(np.sum(coord_diff**2, axis=2))
        direction = coord_diff * (1 / (vert_dist + 1e-3 * np.eye(N)))[:,:,np.newaxis]
        offset = vert_dist * invdist - 1
        np.fill_diagonal(offset, 0)
        cost = 1/2 * np.sum(offset**2)
        something = (invdist * offset)[:,:,np.newaxis] * direction
        grad = np.sum(something, axis=1) - np.sum(something, axis=0)
        sum_pos = np.sum(coords, axis=0)
        cost += 1/2 * meanweight * np.sum(sum_pos ** 2)
        grad += meanweight * sum_pos
        return (cost, grad.ravel())

    def kamada_kawai_cost_function2(position_array):
        coords = position_array.reshape(N, 2)
        coord_diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        vert_dist = np.sqrt(np.sum(coord_diff**2, axis=2))
        direction = coord_diff * (1 / (vert_dist + 1e-3 * np.eye(N)))[:,:,np.newaxis]
        offset = vert_dist - des_length
        np.fill_diagonal(offset, 0)
        cost = np.sum(springs * offset ** 2) / 2
        something = (springs * offset)[:,:,np.newaxis] * direction
        grad = np.sum(something, axis=1) - np.sum(something, axis=0)
        sum_pos = np.sum(coords, axis=0)
        cost += 1/2 * meanweight * np.sum(sum_pos ** 2)
        grad += meanweight * sum_pos
        return (cost, grad.ravel())

    from scipy.optimize import minimize
    optres = minimize(
            kamada_kawai_cost_function,
            initial_position_array,
            method='L-BFGS-B',
            jac=True
            )
    return optres

