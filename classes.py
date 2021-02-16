import numpy as np
from itertools import combinations
import random
from drawing import arc, radial, matrix

class Graph(object):
    def __init__(self, adjacency_list, gids=None):
        '''
        Initialize the graph defined by an adjacency list.
        '''
        self.N = len(adjacency_list)
        self.V = tuple(range(self.N))
        self.adjacency = adjacency_list
        self.update_edges()
        self.update_degree()
        self.update_edge_number()

        # not implemented yet
        self.gids = None if gids is None else gids
        self.directed = False

    def update(self, wrt_adjacency=True):
        if wrt_adjacency:
            self.update_edges()
        else:
            self.update_adjacency()
        self.update_degree()

    def update_edges(self):
        '''
        Update the edge list w.r.t. the adjacency list.
        '''
        self.edges = tuple((i,j) for i, neigh in enumerate(self.adjacency) for j in neigh if j < i)

    def update_adjacency(self):
        '''
        Update the adjacency list w.r.t. the edge list.
        '''
        adjacency = [[] for _ in self.V]
        for i, j in self.edges:
            adjacency[i].append(j)
            adjacency[j].append(i)
        self.adjacency = adjacency

    def update_degree(self):
        '''
        Update the degree w.r.t. the adjacency list.
        '''
        self.degree = tuple([len(neighbourhood) for neighbourhood in self.adjacency])

    def update_edge_number(self):
        '''
        Update edge number w.r.t. the edge list.
        '''
        self.edgenum = len(self.edges)

    def draw_arc(self):
        arc(self.adjacency, self.edges)

    def draw_radial(self):
        radial(self.adjacency, self.edges)

    def draw_matrix(self):
        matrix(self.adjacency)









