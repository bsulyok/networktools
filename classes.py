import numpy as np
from itertools import combinations
import random
from drawing import arc, radial, matrix, kamada_kawai
import networkx as nx


class Graph2:
    def __init__(self, graph_data=None):
        if graph_data is None:
            # create empty graph
            self.vertices=dict()
            self.adjacency=dict()

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, vertex):
        return self.adjacency[vertex]

    def __contains__(self, vertex):
        if type(element) is tuple:
            source, target = element
            return target in self.adjacency[source]
        else:
            return vertex in self.vertices

    def add_vertex(self, vertex=None):
        if vertex in self:
            return "Vertex already exists!"
        if vertex is None:
            vertex = max(self.vertices) + 1
        self.vertices[vertex] = dict()
        self.adjacency[vertex] = dict()

    def remove_vertex(self, vertex):
        if vertex not in self:
            return 'Vertex does not exist!'
        for neighbour in self.adjacency[vertex]:
            del self.adjacency[neighbour][vertex]
        del self.adjacency[vertex]

    def add_edge(self, source, target):
        if source not in self or target not in self:
            return 'Source and/or target vertex does not exist!'
        elif target in self.adjacency[source]:
            return 'Edge already exists!'
        self.adjacency[source][target] = dict()
        self.adjacency[target][source] = dict()

    def remove_edge(self, source, target):
        if source not in self or target not in self:
            return 'Source and/or target vertex does not exist!'
        elif target in self.adjacency[source]:
            return 'Edge already exists!'
        del self.adjacency[source][target]
        del self.adjacency[target][source]

    def neighbours(self, vertex):
        if vertex not in self:
            return 'Vertex does not exist!'
        return iter(self.adjacency[vertex])

    def edge_list(self):
        elist = []
        for vertex, neighbourhood in self.adjacency.items():
            for neighbour in neighbourhood:
                if vertex < neighbour:
                    elist.append((vertex, neighbour))
        return elist




A = Graph2()

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

    def kamada_kawai(self):
        kamada_kawai(self.adjacency)







