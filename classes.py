import drawing, embedding
from common import edge_iterator
from random import random
import clustering
import utils
import readwrite

####################
# UNDIRECTED GRAPH #
####################

class Graph:

    ###################
    # magic functions #
    ###################

    def __init__(self, adjacency_list=None, vertices=None):
        # create empty graph
        if adjacency_list is None:
            self._adjacency = {}
            self._vertices = {}
        else:
            self._adjacency = adjacency_list
            if vertices is None:
                self._vertices = {vertex:{} for vertex in adjacency_list}
            else:
                self._vertices = vertices
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
        if isinstance(other, Graph):
            self._adjacency, self._vertices = utils.merge_components(self._adjacency, self._vertices, other._adjacency, other._vertices)
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

    @property
    def degree(self):
        return {vertex:len(neighbourhood) for vertex, neighbourhood in self._adjacency.items()}

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
        #if source not in self._vertices or target not in self._vertices or target in self._successor[source]:
        #    return 'Invalid source and/or target'
        self._adjacency[source][target] = dict(**attr)
        self._adjacency[target][source] = dict(**attr)

    def remove_edge(self, source, target):
        #if source not in self._vertices or target not in self._vertices or target not in self._successor[source]:
        #    return 'Invalid source and/or target'
        del self._adjacency[source][target]
        del self._adjacency[target][source]

    ###################
    # writing to file #
    ###################

    def write(self, path, vertex_attributes=[], edge_attributes=[]):
        readwrite.write_graph(self._adjacency, self._vertices, path, vertex_attributes, edge_attributes)

    ######################
    # topology iterators #
    ######################

    def edges(self):
        return edge_iterator(self._successor)

    def vertices(self):
        return iter(self._vertices.items())

    ###################
    # drawing methods #
    ###################

    def draw(self, **kwargs):
        drawing.draw(self._adjacency, vertices=self._vertices, **kwargs)

    def draw_arc(self):
        drawing.arc(self._successor)

    def draw_circular(self, lines='euclidean'):
        drawing.circular(self._successor, lines=lines)

    def draw_matrix(self):
        drawing.matrix(self._successor)

    #####################
    # embedding methods #
    #####################

    def embed_greedy(self, representation='hyperbolic_polar'):
        self._vertices = embedding.greedy_embedding(self._adjacency, self._vertices, representation=representation)
        self.representation = representation

    def embed_ncMCE(self, representation='hyperbolic_polar', angular_adjustment=embedding.equidistant_adjustment):
        self._vertices = embedding.ncMCE(self._adjacency, self._vertices, angular_adjustment=angular_adjustment, representation=representation)
        self.representation = representation

    def embed_hypermap(self, representation='hyperbolic_polar'):
        self._vertices = embedding.hypermap(self._adjacency, self._vertices, representation=representation)
        self.representation = representation
        
    def embed_mercator(self, representation='hyperbolic_polar'):
        self._vertices = embedding.mercator(self._adjacency, self._vertices, representation=representation)
        self.representation = representation

    #########################
    # clustering algorithms #
    #########################

    def clustering_label_propagation(self):
        label = clustering.asynchronous_label_propagation(self._adjacency)
        cluster_count = max(label.values())+1
        if cluster_count != 1:
            for vertex, lab in label.items():
                self._vertices[vertex].update({'color':lab/cluster_count})

    ##############################
    # graph theoretical distance #
    ##############################  

    def distance(self, source=None, target=None, weight_attribute=None):
        return utils.distance(self._adjacency, source, target, weight_attribute)

    ########
    # misc #
    ########

    def ispercolating(self):
        return utils.ispercolating(self._adjacency)

    def divide(self):
        return [Graph(adjacency_list, vertices) for adjacency_list, vertices in utils.disjunct_components(self._adjacency, self._vertices)]

    def defragment_indices(self, start=0):
        self._adjacency, self._vertices = utils.defragment_indices(self._adjacency, self._vertices, start=start)

    def largest_component(self, defragment=True):
        adjacency_list, vertices, comp_size = {}, {}, 0
        for next_adjacency_list, next_vertices in utils.disjunct_components(self._adjacency, self._vertices):
            if comp_size < (next_comp_size:=len(next_adjacency_list)):
                adjacency_list, vertices, comp_size = next_adjacency_list, next_vertices, next_comp_size
        if defragment:
            adjacency_list, vertices = utils.defragment_indices(adjacency_list, vertices)
        return Graph(adjacency_list=adjacency_list, vertices=vertices)

    def greedy_routing_score(self, normalized=False):
        if normalized:
            return utils.greedy_routing_score(self._adjacency, self._vertices, self.representation)
        else:
            return utils.greedy_routing_success_score(self._adjacency, self._vertices, self.representation)
    
    def greedy_routing_badness(self):
        return utils.greedy_routing_badness(self._adjacency, self._vertices, self.representation)

    ##################################
    # functions for testing purposes #
    ##################################

    def fill_vertices(self, attribute_name='size', generator=random):
        for vertex, attributes in self.vertices():
            attributes.update({attribute_name: generator() })

    def fill_edges(self, attribute_name='weight', generator=random):
        for vertex, neighbour, attributes in self.edges():
            attributes.update({attribute_name: generator() })

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