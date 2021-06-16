import numpy as np
import random
import classes
import models
import drawing
import utils
import networkx as nx
import embedding
import common
import readwrite

def main():
    return

if __name__ == "__main__":
    main()

def test_badness():
    G = models.regular_tree(3, 4)
    G.embed_greedy()
    G.draw(representation='hyperbolic_polar', vertex_scale=20)
    badness = G.greedy_routing_badness()
    print('highest badness:', max(badness.values()), '(original)')
    G.remove_edge(1, 0)
    G.add_edge(1, 30)
    G.draw(representation='hyperbolic_polar', vertex_scale=20)
    badness = G.greedy_routing_badness()
    print('highest badness:', max(badness.values()), '(rewired)')
    max_badness = max(badness.values())
    for vertex, attributes in G.vert.items():
        attributes.update({'color':badness[vertex]/max_badness})
        attributes.update({'size':badness[vertex]/max_badness})
    G.draw(representation='hyperbolic_polar', vertex_scale=50)
    G.embed_greedy()
    G.draw(representation='hyperbolic_polar', vertex_scale=50)
    return G

def test_greedy_embetting():
    G = models.regular_tree(3, 8)
    G.embed_greedy()
    G.draw(representation='hyperbolic_polar')
    print('greedy routing success:', G.greedy_routing_score(), '(regular tree)')
    G = models.popularity_similarity_optimisation_model(766, 2, T=0)
    G.embed_greedy()
    G.draw()
    print('greedy routing success:', G.greedy_routing_score(), '(PSO graph)')