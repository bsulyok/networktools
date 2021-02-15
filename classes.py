import numpy as np
from itertools import combinations
import random

class Graph(object):
    def __init__(self, adjacency_list, gids=None):
        '''
        Initialize the graph defined by an adjacency list.
        '''
        self.N = len(adjacency_list)
        self.alist = adjacency_list
        self.degree = np.array([len(i) for i in self.alist])
        self.gids = None if gids is None else gids
        self.directed = False

    def list2mat(self):
        '''
        Produce the adjacency matrix of the graph.
        '''
        adjacency_matrix = np.zeros((self.N, self.N), bool)
        for i in range(self.N):
            for j in self.alist[i]:
                adjacency_matrix[i,j] = True
                adjacency_matrix[j,i] = True
        return adjacency_matrix
