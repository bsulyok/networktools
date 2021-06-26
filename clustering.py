import math
import numpy as np
from common import disjoint_set
import random
import queue

def asynchronous_label_propagation(adjacency_list):
    label = {vertex:vertex for vertex in adjacency_list}
    majority = {vertex:math.ceil((len(neighourhood)+1)/2) for vertex, neighourhood in adjacency_list.items()}
    vertex_order = list(adjacency_list)
    random.shuffle(vertex_order)

    def update_label(vertex):
        vert_maj, occurence = majority[vertex], {}
        maximal_label, maximal_occurence = None, 0
        old_label = label[vertex]
        for neighbour in adjacency_list[vertex]:
            neigh_lab = label[neighbour]
            lab_occ = occurence.get(neigh_lab, 0) + 1
            if vert_maj <= lab_occ:
                label[vertex] = neigh_lab
                return old_label != neigh_lab
            occurence[neigh_lab] = lab_occ
            if maximal_occurence == lab_occ:
                if isinstance(maximal_label, list):
                    maximal_label.append(neigh_lab)
                elif isinstance(maximal_label, int):
                    maximal_label = [maximal_label, neigh_lab]
            elif maximal_occurence < lab_occ:
                maximal_label, maximal_occurence = neigh_lab, lab_occ
        if isinstance(maximal_label, int):
            label[vertex] = maximal_label
            return old_label != maximal_label
        elif isinstance(maximal_label, list):
            label[vertex] = random.choice(maximal_label)
            return old_label not in maximal_label
             
    running = True
    while running:
        running = False
        for vertex in vertex_order:
            running = update_label(vertex)
            
    new_label, label_counter = {}, 0

    def recursive_depth_first_search(vertex):
        new_label[vertex] = label_counter
        for neighbour in adjacency_list[vertex]:
            if label[neighbour] == old_label and neighbour not in new_label:
                recursive_depth_first_search(neighbour)
    
    for root, old_label in label.items():
        if root not in new_label:
            recursive_depth_first_search(root)
            label_counter += 1

    return new_label

def infomap(adjacency_list):
    label = {vertex:vertex for vertex in adjacency_list}

    def core_algorithm():
        pass

    return label