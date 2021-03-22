import drawing
import networkx as nx
from common import edge_iterator, priority_queue
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
        for neighbour in self.adjacency[vertex]:
            del self.adjacency[neighbour][vertex]
        del self.adjacency[vertex]

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

    ##################################
    # functions for testing purposes #
    ##################################

    def genweight(self):
        self.weighted = True
        for _, _, attributes in self.edges():
            attributes.update({'weight':random()})
