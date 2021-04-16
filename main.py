import numpy as np
import random
import classes
import models
import drawing
import utils
import networkx as nx

def main():
    G = models.popularity_similarity_optimisation_model(1000, 2, beta=0.9)
    G.draw_hyperbolic(euclidean=False)

if __name__ == "__main__":
    main()
