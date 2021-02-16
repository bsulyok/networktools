from math import exp
from random import random, seed
import numpy as np
from utils import priority_queue, alldistance

def annealing(elementary_step, state, max_step=1000):
    seed(123456)
    for step in range(max_step):
        new_state, energy_change = elementary_step(state)
        if energy_change < 0 or random() < exp(-energy_change * (1-step/max_step)):
            state = new_state
    return state

def kamada_kawai(adjacency_list):
    N = len(adjacency_list)
    graph_distance = np.array(alldistance(adjacency_list))
    L0, K = 1.0, 1.0
    diameter = graph_distance.max()
    if diameter == inf:
        return
    L = L0 / diameter
    l = L * graph_distance
    k = K / graph_distance**2
    np.fill_diagonal(k, 0)
    angle = 2 * np.pi * np.linspace(0,1, N+1)[1:]
    pairs = combinations(range(N), r=2)

    def energy(coords):
        xdiff = coords[:,0] - coords[:,0][:,None]
        ydiff = coords[:,1] - coords[:,1][:,None]
        return np.sum(k / 2 * ( np.sum([xdiff**2, ydiff**2], axis=0)**(1/2) - l )**2)

    def grad(coords, i):
        dcoords = np.delete(coords, i, axis=0) - coords[i]
        k_i = np.delete(k[i], i)[:,None]
        l_i = np.delete(l[i], i)[:,None]
        eucl_dist = np.sqrt(np.sum(dcoords**2, axis=1))[:,None]
        return np.sum( k_i * dcoords * (1 - l_i / eucl_dist), axis=0)

    def calc_delta(coords, i):
        return sum((grad(coords, i)/coords[i])**2) ** (1/2)

    def jacobi(coords, i):
        dcoords = np.delete(coords, i, axis=0) - coords[i]
        k_i = np.delete(k[i], i)[:,None]
        l_i = np.delete(l[i], i)[:,None]
        dist_prod = np.prod(dcoords, axis=1)[:, None]
        eucl_dist3 = (np.sum(dcoords**2, axis=1)**(1/3))[:,None]
        J = np.empty((2,2))
        J[[0,1],[1,0]] = np.sum( k_i * l_i * dist_prod / eucl_dist3)
        J[[0,1],[0,1]] = np.sum( k_i * ( 1 - l_i * dcoords**2 / eucl_dist3 ) )
        return J

    coords = np.stack([np.sin(angle), np.cos(angle)]).T / 2

    DELTA = priority_queue(reverse=True)
    for i in range(N):
        DELTA.push((calc_delta(coords,i), i))
    stepcount = 0
    delta_threshold = -max(DELTA.queue)[0]
    print(energy(coords))
    while delta_threshold < abs(min(DELTA.queue)[0]):
        delta, vertex = DELTA.pop()
        old_coord = coords[vertex].copy()
        while delta_threshold < delta:
            ijac = np.linalg.inv(jacobi(coords, vertex))
            coords[vertex] -= ijac @ grad(coords, vertex)
            delta = calc_delta(coords, vertex)
        print(energy(coords))
        # update
        for i in range(N):
            DELTA.push((calc_delta(coords,i), i))

    return coords
