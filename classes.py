import drawing
import networkx as nx
from common import edge_iterator, priority_queue, ienumerate
from math import inf
from random import random
from utils import dijsktra, identify_components

class Graph:

    ###################
    # magic functions #
    ###################

    def __init__(self, graph_data=None):
        if graph_data is None:
            # create empty graph
            self.vertices = dict()
            self.adjacency = dict()
            self.attributes = set()
            self.weighted = False

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.vertices[item]
        elif isinstance(item, tuple):
            return self.adjacency[item[0]][item[1]]

    def __contains__(self, item):
        if isinstance(item, int):
            return item in self.vertices
        elif isinstance(item, (tuple, list)):
            return item[1] in self.adjacency[item[0]]

    def __iter__(self):
        return edge_iterator(self.adjacency)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self._add__(other)

    def __add__(self, other):
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

    #####################
    # vertex operations #
    #####################

    def add_vertex(self, vertex, **attr):
        if vertex in self:
            return "Vertex already exists!"
        self.vertices[vertex] = dict(**attr)
        self.adjacency[vertex] = dict()

    def remove_vertex(self, vertex):
        if vertex not in self:
            return 'Vertex does not exist!'
        try:
            del self.vertices[vertex]
            for neighbour in self.adjacency[vertex]:
                del self.adjacency[neighbour][vertex]
            del self.adjacency[vertex]
        except:
            return 'Vertex not in graph!'

    def update_vertex(self, vertex, **attr):
        if vertex not in self:
            return 'Vertex does not exist!'
        self.vertices[vertex].update(**attr)

    ###################
    # edge operations #
    ###################

    def add_edge(self, source, target, **attr):
        #if source not in self or target not in self:
        #    return 'Source and/or target vertex does not exist!'
        #elif target in self.adjacency[source]:
        #    rIeturn 'Edge already exists!'
        if source in self and target in self and (source, target) not in self:
            self.adjacency[source][target] = dict(**attr)
            self.adjacency[target][source] = dict(**attr)
        else:
            return 'Invalid source and/or target'

    def remove_edge(self, source, target):
        try:
            del self.adjacency[source][target]
            del self.adjacency[target][source]
        except KeyError:
            return 'Edge does not exist!'

    def update_edge(self, source, target, **attr):
        if source in self and target in self and (source, target) in self:
            self.adjacency[source][target].update(**attr)
            self.adjacency[target][source].update(**attr)
        else:
            return 'Invalid source and/or target'

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
        return edge_iterator(self.adjacency)

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
        drawing.arc(self.adjacency)

    def draw_radial(self, arcs=True):
        drawing.radial(self.adjacency, arcs)

    def draw_matrix(self):
        drawing.matrix(self.adjacency)


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
        self.weighted = True
        for _, _, attributes in self.edges():
            attributes.update({'weight':random()})
