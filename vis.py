import numpy as np
import random
import plotly.graph_objects as go
from simulationtools import annealing
from copy import deepcopy
from utils import *

HEIGHT = 1000

def semi_circle(x1, x2):
    angle = np.linspace(0, np.pi, int(abs(x2-x1) * 1000))
    x_coords = (x1+x2)/2 + abs(x2-x1)/2 * np.cos(angle)
    y_coords = abs(x2-x1)/2 * np.sin(angle)
    return x_coords, y_coords

def arcs(adjacency, z=None):
    fig = go.Figure()
    N = len(adjacency)
    x_coords = np.linspace(0,1,N)
    for i, j in edge_list(adjacency):
        sc = semi_circle(x_coords[i], x_coords[j])
        fig.add_trace(go.Scattergl(
            x=sc[0],
            y=sc[1],
            mode='lines',
            line_width=1,
            line_color='red',
            showlegend=False
        ))

    fig.add_trace(go.Scattergl(
        x=x_coords,
        y=np.zeros(N),
        mode='markers',
        marker_size=10,
        marker_color='blue' if z is None else z,
        showlegend=False
    ))

    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20)
    fig.show()
    return

def quadratic_bezier_curve(p1, p2, p3):

    '''
    Compute the quadratic Bezier curve for the given points

    Parameters
    ----------
    p1, p2, p3 : array_like
        Points defining the Bezier curve.
    num : int
        Number of points on the curve.

    Returns
    ---------
    curve : ndarray
        Point of the generated curve.
    '''

    t = np.linspace(0,1,100)[:,None]
    return p2 + (1-t)**2 * (p1 - p2) + t**2 * (p3 - p2)

def radial_semi_circle(a1, a2):
    p1 = np.array([np.sin(a1), np.cos(a1)])
    p2 = np.array([0,0])
    p3 = np.array([np.sin(a2), np.cos(a2)])
    return quadratic_bezier_curve(p1, p2, p3).T

def radial(adjacency, z=None):
    fig = go.Figure()
    N = len(adjacency)
    angle = np.linspace(0,2*np.pi,N+1)[1:]

    for i, j in edge_list(adjacency):
        rsc = radial_semi_circle(angle[i], angle[j])
        fig.add_trace(go.Scatter(
            x=rsc[0],
            y=rsc[1],
            mode='lines',
            line_width=1,
            line_color='red',
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        x=np.sin(angle),
        y=np.cos(angle),
        mode='markers',
        marker_size=10,
        marker_color='blue' if z is None else z,
        showlegend=False
    ))

    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False, scaleanchor='x')
    fig.update_layout(height=HEIGHT, width=HEIGHT-20)
    fig.show()
    return

def optimize_arc(adjacency):
    '''
    Optimize vertex arrangement via simulated annealing for the arc type visualization. The elementary step is swapping two vertices, the energy is the sum of edge lengths.
    '''
    adjacency = deepcopy(adjacency)
    N = len(adjacency)

    if islist(adjacency):
        initial_perm = ran = list(range(N))
        def swap_nodes(perm):
            new_perm = deepcopy(perm)
            i, j = random.sample(ran, k=2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = 0
            dE += sum([abs(perm[j]-perm[r])for r in adjacency[i]])
            dE -= sum([abs(perm[i]-perm[r])for r in adjacency[i]])
            dE += sum([abs(perm[i]-perm[r])for r in adjacency[j]])
            dE -= sum([abs(perm[j]-perm[r])for r in adjacency[j]])
            if i in adjacency[j]:
                dE += 2 * abs(perm[i] - perm[j])
            return new_perm, dE

    elif ismat(adjacency):
        initial_perm = np.arange(N)
        def swap_nodes(perm):
            new_perm = deepcopy(perm)
            i, j = np.random.choice(N, size=2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = 0
            dE += sum(abs(perm[adjacency[i]] - perm[j]))
            dE -= sum(abs(perm[adjacency[i]] - perm[i]))
            dE += sum(abs(perm[adjacency[j]] - perm[i]))
            dE -= sum(abs(perm[adjacency[j]] - perm[j]))
            if adjacency[i,j]:
                dE += 2 * abs(perm[i] - perm[j])
            return new_perm, dE

    perm = annealing(swap_nodes, initial_perm, 1000)
    return perm

def optimize_radial(adjacency):
    #TODO: the computations are incorrect!
    '''
    Optimize vertex arrangement via simulated annealing for the radial type visualization. The elementary step is swapping two vertices, the energy is the sum of edge lengths.
    '''
    adjacency = deepcopy(adjacency)
    N = len(adjacency)

    if islist(adjacency):
        initial_perm = ran = list(range(N))
        def swap_nodes(perm):
            new_perm = deepcopy(perm)
            i, j = random.sample(ran, k=2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = 0
            dE += sum([(abs(perm[j]-perm[r])%N) for r in adjacency[i]])
            dE -= sum([(abs(perm[i]-perm[r])%N) for r in adjacency[i]])
            dE += sum([(abs(perm[i]-perm[r])%N) for r in adjacency[j]])
            dE -= sum([(abs(perm[j]-perm[r])%N) for r in adjacency[j]])
            if i in adjacency[j]:
                dE += 2 * abs(perm[i] - perm[j])
            return new_perm, dE

    elif ismat(adjacency):
        initial_perm = np.arange(N)
        def swap_nodes(perm):
            new_perm = deepcopy(perm)
            i, j = np.random.choice(N, size=2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            dE = 0
            dE += sum((abs(perm[adjacency[i]] - perm[j])%N))
            dE -= sum((abs(perm[adjacency[i]] - perm[i])%N))
            dE += sum((abs(perm[adjacency[j]] - perm[i])%N))
            dE -= sum((abs(perm[adjacency[j]] - perm[j])%N))
            if adjacency[i,j]:
                dE += 2 * abs(perm[i] - perm[j])
            return new_perm, dE

    perm = annealing(swap_nodes, initial_perm, 1000)
    return perm

def matrix(adjacency):
    if islist(adjacency):
        adjacency = list2mat(adjacency)

    fig = go.Figure(data=go.Heatmap(
        z=adjacency[::-1].astype(int),
        showscale=False
        ))
    fig.update_yaxes(tickvals=[], scaleanchor='x')
    fig.update_xaxes(tickvals=[])
    fig.update_layout(width=HEIGHT-20, height=HEIGHT)

    fig.show()
    return

def force_directed(adjacency, z=None, initial_position=None):
    '''
    Force directed graph drawing.
    '''
    if initial_position is not None:
        coord = initial_position
    elif z is not None:
        K = len
        group_coord = 2 * np.random.random((K,2)) - 1
        coord = group_coord[z] + np.random.random((z,2)) / 5 - 1/10
    else:
        coord = 2 * np.random.random((z,2)) - 1





