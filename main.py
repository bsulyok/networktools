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
import csv
import testing
from copy import deepcopy
import clustering
import plotly.graph_objects as go
import math
import time

def main():
    G = models.PSO(200,3,T=0.5)
    G = G.largest_component()
    return G

if __name__ == "__main__":
    G = main()