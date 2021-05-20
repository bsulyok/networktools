import numpy as np
from numpy.core.function_base import linspace
import plotly.graph_objects as go
import models
from drawing_tools import circular_arc, line
from common import edge_iterator
import utils
import math

def extract_moebius_transformation(z1, w1, z2, w2, z3, w3):
    a = np.linalg.det(np.array([[z1*w1, w1, 1], [z2*w2, w2, 1], [z3*w3, w3, 1]]))
    b = np.linalg.det(np.array([[z1*w1, z1, w1], [z2*w2, z2, w2], [z3*w3, z3, w3]]))
    c = np.linalg.det(np.array([[z1, w1, 1], [z2, w2, 1], [z3, w3, 1]]))
    d = np.linalg.det(np.array([[z1*w1, z1, 1], [z2*w2, z2, 1], [z3*w3, z3, 1]]))
    return np.array([[a, b], [c, d]])

def invert(M):
    return np.array([[M[1,1], -M[0,1]], [-M[1,0], M[0,0]]])

def apply_mobius_transformation(z, M):
    return (M[0,0] * z + M[0,1]) / (M[1,0] * z + M[1,1])

def minimal_depth_tree_update(G, root=0):
    distance_from_root = utils.dijsktra(G.adj, root)
    depth = max(distance_from_root.values())
    if depth is math.inf:
        print('The graph is not connected!')
        return
    parent = dict.fromkeys(G.adj, None)
    parent[root] = 0
    for vertex in G.vert:
        G.update_vertex(vertex, children=[])
    for vertex, neighbours in G.adj.items():
        if vertex is root:
            continue
        for neighbour in neighbours:
            if parent[vertex] is None or distance_from_root[neighbour] < distance_from_root[parent[vertex]]:
                parent[vertex] = neighbour
        G.vert[parent[vertex]]['children'].append(vertex)
    return G

def greedy_embedding():   
    G = models.erdos_renyi_graph(20, 0.3)
    G = minimal_depth_tree_update(G, 0)
    n = 1
    for attr in G.vert.values():
        if len(attr['children']) > n:
            n = len(attr['children'])
    
    # declare the relevant Mobius transformations
    last_root = np.exp(-2j*np.pi/n)
    last_half_root = np.exp(-2j*np.pi/2/n)
    sigma = extract_moebius_transformation(1, 1, last_root, -1, last_half_root, -1j)
    isigma = extract_moebius_transformation(1, 1, -1, last_root, -1j, last_half_root)
    a = np.array([[-1, 0], [0, 1]])
    B = [sigma @ np.array([[np.exp(2j*np.pi*k/n), 0], [0, 1]]) @ isigma for k in range(n)]

    # fix the origin
    u = apply_mobius_transformation(0, sigma)
    v = u.conjugate()
    G.update_vertex(0, coord=0.0)
    root_transform = sigma

    # iteratively find the coordinates of all vertices
    def iterative_coordination_search(vertex, transform):
        G.update_vertex(vertex, coord=apply_mobius_transformation(v, invert(transform)))
        children = G.vert[vertex].get('children', [])
        if len(children) is None:
            return
        for child_id, child in enumerate(children, 1):
            child_transform = B[child_id] @ a @ transform
            iterative_coordination_search(child, child_transform)

    for child_id, child in enumerate(G.vert[0].get('children', None)):
        child_transform = B[child_id] @ root_transform
        iterative_coordination_search(child, child_transform)
    
    # drawing happens here
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=np.cos(np.linspace(0,2*np.pi,1000)), y=np.sin(np.linspace(0,2*np.pi,1000)), line_color='black'))
    coords = np.array([attributes['coord'] for attributes in G.vert.values()])
    for vertex, neighbour, attributes in edge_iterator(G.adj):
        if vertex < neighbour:
            path = line(G.vert[vertex]['coord'], G.vert[neighbour]['coord'], 2)
            fig.add_trace(go.Scattergl(x=path.real, y=path.imag, mode='lines', line_color='blue'))
    fig.add_trace(go.Scattergl(x=coords.real, y=coords.imag, mode='markers', marker_size=10, marker_color='red'))
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20, showlegend=False)
    fig.show()

greedy_embedding()