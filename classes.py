import numpy as np
from itertools import combinations
import random
import drawing
import networkx as nx
from common import edge_iterator


class Graph:
    def __init__(self, graph_data=None):
        if graph_data is None:
            # create empty graph
            self.vertices = dict()
            self.adjacency = dict()
            self.attributes = set()

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, vertex):
        return self.adjacency[vertex]

    def __contains__(self, item):
        if type(item) is tuple:
            source, target = item
            return target in self.adjacency[source]
        else:
            return item in self.vertices

    def add_vertex(self, vertex, **attr):
        if vertex in self:
            return "Vertex already exists!"
        self.vertices[vertex] = dict(**attr)
        self.adjacency[vertex] = dict()

    def remove_vertex(self, vertex):
        if vertex not in self:
            return 'Vertex does not exist!'
        for neighbour in self.adjacency[vertex]:
            del self.adjacency[neighbour][vertex]
        del self.adjacency[vertex]

    def update_vertex(self, vertex, **attr):
        if vertex not in self:
            return 'Vertex does not exist!'
        self.vertices[vertex].update(**attr)

    def add_edge(self, source, target, **attr):
        if source not in self or target not in self:
            return 'Source and/or target vertex does not exist!'
        elif target in self.adjacency[source]:
            return 'Edge already exists!'
        self.adjacency[source][target] = dict(**attr)
        self.adjacency[target][source] = dict(**attr)

    def remove_edge(self, source, target):
        if source not in self or target not in self:
            return 'Source and/or target vertex does not exist!'
        elif (source, target) in self:
            return 'Edge already exists!'
        del self.adjacency[source][target]
        del self.adjacency[target][source]

    def update_edge(self, source, target, **attr):
        if source not in self or target not in self:
            return 'Source and/or target vertex does not exist!'
        elif (source, target) not in self:
            return 'Edge does not exist!'
        self.adjacency[source][target].update(**attr)
        self.adjacency[target][source].update(**attr)

    def neighbours(self, vertex):
        if vertex not in self:
            return 'Vertex does not exist!'
        return iter(self.adjacency[vertex])

    def write(self, filename):
        import csv
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #writer.writerow(['source', 'target', 'weight'])
            writer.writerow(['source', 'target'])
            for vertex, neighbourhood in self.adjacency.items():
                for neighbour, attributes in neighbourhood.items():
                    if vertex < neighbour:
                        writer.writerow([vertex, neighbour] + list(attributes.values()))

    def edge_list(self):
        return edge_iterator(self.adjacency)

    def draw_arc(self):
        drawing.arc(self.adjacency)

    def draw_radial(self, arcs=True):
        drawing.radial(self.adjacency, arcs)

    def draw_matrix(self):
        drawing.matrix(self.adjacency)

class Graph2(object):
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
