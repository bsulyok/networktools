from math import exp, inf
from random import random, seed
import numpy as np
from common import priority_queue
from itertools import combinations

def annealing(elementary_step, state, steps=1000):
    '''
    Minimalistic simulated annealing.
    '''
    seed(123456)
    for step in range(steps):
        new_state, energy_change = elementary_step(state)
        if energy_change < 0 or random() < exp(-energy_change * (1-step/steps)):
            state = new_state
    return state

def solve_kamada_kawai(adjacency_list, graph_distance):
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
