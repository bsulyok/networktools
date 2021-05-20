import drawing
import networkx as nx
from common import edge_iterator, priority_queue, ienumerate
from math import inf
from random import random
from utils import dijsktra, identify_components

##################
# DIRECTED GRAPH #
##################

class Graph:

    ###################
    # magic functions #
    ###################

    def __init__(self, graph_data=None):
        if graph_data is None:
            # create empty graph
            self._vertices = dict()
            self._adjacency = dict()
            self._successor = self._adjacency
            self._predecessor = self._adjacency

    def __len__(self):
        return len(self._vertices)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._vertices[item]
        elif isinstance(item, tuple):
            return self._successor[item[0]][item[1]]

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self._vertices
        elif isinstance(item, (tuple, list)):
            return item[1] in self._successor[item[0]]

    def __iter__(self):
        return edge_iterator(self._successor)

    def __radd__(self, other):
        if other == 0:
            return self
        elif isinstance(other, Graph):
            return self.__add__(other)


    #TODO
    def __add__(self, other):
        if not isinstance(other, Graph):
            return 'Both input must be Graph objects!'
        if len(self) == 0:
            return other
        elif len(other) == 0:
            return self
        if len(other) < len(self):
            other, self = self, other
        if len(self) != max(self.vertices) + 1:
            self.defragment()
        relabel = {old:new for new, old in enumerate(other.vertices, len(self))}
        for vertex in relabel.values():
            self.add_vertex(vertex)
        for source, target, attributes in iter(other):
            if source < target:
                continue
            self.add_edge(relabel[source], relabel[target])
        return self

    ##############
    # properties #
    ##############

    @property
    def adj(self):
        return self._adjacency

    @property
    def pred(self):
        return self._predecessor

    @property
    def succ(self):
        return self._successor

    @property
    def vert(self):
        return self._vertices

    #####################
    # vertex operations #
    #####################

    def add_vertex(self, vertex, **attr):
        if vertex in self._vertices:
            return "Vertex already exists!"
        self._vertices[vertex] = dict(**attr)
        self._adjacency[vertex] = dict()

    def update_vertex(self, vertex, **attr):
        if vertex not in self._vertices:
            return 'Vertex does not exist!'
        self._vertices[vertex].update(**attr)

    def remove_vertex(self, vertex):
        if vertex not in self._vertices:
            return 'Vertex does not exist!'
        for neighbour in self._adjacency[vertex]:
            del self._adjacency[neighbour][vertex]
        del self._adjacency[vertex]
        del self._vertices[vertex]

    ###################
    # edge operations #
    ###################

    def add_edge(self, source, target, **attr):
        if source not in self._vertices or target not in self._vertices or target in self._successor[source]:
            return 'Invalid source and/or target'
        self._adjacency[source][target] = dict(**attr)
        self._adjacency[target][source] = dict(**attr)

    def remove_edge(self, source, target):
        if source not in self._vertices or target not in self._vertices or target not in self._successor[source]:
            return 'Invalid source and/or target'
        del self._adjacency[source][target]
        del self._adjacency[target][source]

    ###################
    # writing to file #
    ###################

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

    ######################
    # topology iterators #
    ######################

    def edges(self):
        return edge_iterator(self.successor)

    def vertices(self):
        return iter(self.iterators)

    def neighbours(self, vertex):
        if vertex not in self:
            return 'Vertex does not exist!'
        return iter(self.adjacency[vertex].items())

    ###################
    # drawing methods #
    ###################

    def draw_arc(self):
        drawing.arc(self._successor)

    def draw_circular(self, euclidean=False):
        drawing.circular(self._successor, euclidean)

    def draw_matrix(self):
        drawing.matrix(self._successor)

    def draw_hyperbolic(self, euclidean=False):
        drawing.hyperbolic(self._successor, self._vertices, euclidean)


    ##############################
    # graph theoretical distance #
    ##############################

    def distance(self, source=None, target=None):
        if source is not None:
            return dijsktra(self.adjacency, source, target)
        else:
            return {source:dijsktra(self.adjacency, source) for source in self.adjacency}

    ########
    # misc #
    ########

    def degree(self):
        return {vertex:len(neighbourhood) for vertex, neighbourhood in self.adjacency.items()}

    def components(self):
        return identify_components(self.adjacency)

    def divide(self):
        comps = identify_components(self.adjacency)
        for i in range(max(comps.values())+1):
            True

    def defragment(self):
        relabel = dict(ienumerate(self.vertices))
        for vertex in list(self.vertices.keys()):
            if relabel[vertex] == vertex:
                continue
            self.vertices[relabel[vertex]] = self.vertices.pop(vertex)
            for neighbour in self.adjacency[vertex]:
                self.adjacency[neighbour][relabel[vertex]] = self.adjacency[neighbour].pop(vertex)
            self.adjacency[relabel[vertex]] = self.adjacency.pop(vertex)

    ##################################
    # functions for testing purposes #
    ##################################

    def genweight(self):
        for _, _, attributes in self.edges():
            attributes.update({'weight':random()})

    def fill_vertices(self, data):
        if len(data) != len(self.vertices):
            return 'Mismatch in the number or attributes.'

##################
# DIRECTED GRAPH #
##################

class DiGraph(Graph):

    ###################
    # magic functions #
    ###################
    def __init__(self):
        Graph.__init__(self)
        self.successor = dict()
        self.predecessor = dict()

    #####################
    # vertex operations #
    #####################

    def add_vertex(self, vertex, **attr):
        if vertex in self.vertices:
            return 'Vertex already exists!'
        self.vertices[vertex] = dict(**attr)
        self.adjacency[vertex] = dict()
        self.successor[vertex] = dict()
        self.predecessor[vertex] = dict()

    def remove_vertex(self, vertex):
        if vertex not in self.vertices:
            return 'Vertex does not exist!'
        for pred in self.predecessor[vertex]:
            del self.successor[pred][vertex]
        for succ in self.successor[vertex]:
            del self.predecessor[succ][vertex]
        for neighbour in self.adjacency[vertex]:
            del self.adjacency[neighbour][vertex]
        del self.adjacency[vertex]
        del self.predecessor[vertex]
        del self.successor[vertex]
        del self.vertices[vertex]

    ###################
    # edge operations #
    ###################

    def add_edge(self, source, target, **attr):
        if source not in self.vertices or target not in self.vertices or target in self.successor[source]:
            return 'Invalid source and/or target'
        self.successor[source][target] = dict(**attr)
        self.predecessor[target][source] = dict(**attr)
        if target not in self.adjacency[source]:
            self.adjacency[source][target] = dict(**attr)
            self.adjacency[target][source] = dict(**attr)

    def remove_edge(self, source, target):
        if source not in self.vertices or target not in self.vertices or target not in self.successor[source]:
            return 'Invalid source and/or target'
        del self.successor[source][target]
        del self.predecessor[target][source]
        del self.adjacency[source][target]
        del self.adjacency[target][source]
