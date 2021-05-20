import numpy as np
import random
import classes
import models
import drawing
import utils
import networkx as nx
import embedding_tools

def main():
    G = models.regular_tree(3,3)
    G.draw_matrix()

if __name__ == "__main__":
    G = main()
