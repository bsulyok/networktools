import math
import numpy as np
from common import disjoint_set
import random


def random_walk_entropy(adjacency_list):
    '''
    Returns the random walk entropy of the provided unweighted undirected graph
    '''
    degree = np.array([len(neighbourhood) for neighbourhood in adjacency_list.values()])
    return sum(degree*np.log2(degree)/sum(degree))

def select_new_label(neighbour_labels):
    maximal_label, maximal_occurence = None, 0
    for label, occurence in neighbour_labels.items():
        if maximal_occurence < occurence:
            maximal_label, maximal_occurence = label, occurence
        elif maximal_occurence == occurence:
            try:
                maximal_label.append(label)
            except:
                maximal_label = [maximal_label, label]
    if isinstance(maximal_label, int):
        return maximal_label
    else:
        return random.choice(maximal_label)

def asynchronous_label_propagation(adjacency_list, defragment_labels=False):
    label = {vertex:vertex for vertex in adjacency_list}
    vertex_order = list(adjacency_list)
    random.shuffle(vertex_order)
    running = True
    t = 0
    while running:
        running = False
        for vertex in vertex_order:
            neighbour_labels = {}
            for neighbour in adjacency_list[vertex]:
                neighbour_label = label[neighbour]
                try:
                    neighbour_labels[neighbour_label] += 1
                except:
                    neighbour_labels[neighbour_label] = 1
            new_label = select_new_label(neighbour_labels)
            if new_label != label[vertex]:
                running = True
                label[vertex] = new_label
            
    if defragment_labels:
        label_dict, highest_label = {}, 0
        for vertex, lab in label.items():
            try:
                label[vertex] = label_dict[label[vertex]]
            except:
                highest_label += 1
                label_dict[label[vertex]] = label[vertex] = highest_label
    return label