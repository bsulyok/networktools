import math
import numpy as np
from common import disjoint_set
import random
import queue

def select_new_label(neighbour_labels, current_label):
    maximal_label, maximal_occurence = None, 0
    for label, occurence in neighbour_labels.items():
        if maximal_occurence < occurence:
            maximal_label, maximal_occurence = label, occurence
        elif maximal_occurence == occurence:
            if isinstance(maximal_label, list):
                maximal_label.append(label)
            elif isinstance(maximal_label, int):
                maximal_label = [maximal_label, label]
    if isinstance(maximal_label, int):
        return maximal_label
    elif isinstance(maximal_label, list):
        if current_label in maximal_label:
            return current_label
        return random.choice(maximal_label)

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
            if label[vertex] in maximal_label:
                return False
            else:
                new_label = random.choice(maximal_label)
                label[vertex] = new_label
                return new_label != old_label
             
    running = True
    while running:
        running = False
        for vertex in vertex_order:
            running = update_label(vertex)
            
    new_label, label_counter, vertex_queue = {}, 0, queue.Queue()
    for root, old_lab in label.items():
        if root not in new_label:
            vertex_queue.put(root)
            new_label[root] = label_counter
            while not vertex_queue.empty():
                vertex = vertex_queue.get()
                for neighbour in adjacency_list[vertex]:
                    if label[neighbour] == old_lab and neighbour not in new_label:
                        vertex_queue.put(neighbour)
                        new_label[neighbour] = label_counter           
            label_counter += 1
    return new_label