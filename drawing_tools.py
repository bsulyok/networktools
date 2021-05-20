import numpy as np
from itertools import combinations
from math import inf, pi, sin, cos
import plotly.graph_objects as go

HEIGHT = 1000

def semi_circle_np(x1, x2):
    angle = np.linspace(0, np.pi, int(abs(x2-x1) * 1000))
    X = (x1+x2)/2 + abs(x2-x1)/2 * np.cos(angle)
    Y = abs(x2-x1)/2 * np.sin(angle)
    return X, Y

def semi_circle(x1, x2):
    X, Y = [], []
    N = int(abs(x2-x1)*1000)
    incr = pi/(N-1)
    center = (x1+x2)/2
    radius = abs(x1-x2)/2
    for asd in range(N):
        angle = asd*incr
        X.append(center+radius*cos(angle))
        Y.append(radius*sin(angle))
    return X, Y

def circular_arc(p1, p2, number_of_samples=100):
    center = ( p1 * (1 + p2*p2.conjugate()) - p2 * (1 + p1*p1.conjugate()) ) / ( p1 * p2.conjugate() - p1.conjugate() * p2 )
    radius = np.sqrt(center*center.conjugate() - 1)
    start_angle = np.angle(p1-center)
    end_angle = np.angle(p2-center)
    if np.pi < start_angle-end_angle:
        start_angle -= 2*np.pi
    elif np.pi < end_angle - start_angle:
        end_angle -= 2*np.pi
    phi = np.linspace(start_angle, end_angle, number_of_samples)
    arc = center + radius * np.exp(1j * phi)
    return arc

def circular_arc2(x1, y1, x2, y2, number_of_samples=100):
    x_center = ( y1*(x2*x2 + y2*y2 + 1) - y2*(x1*x1 + y1*y1 +1) ) / (x2*y1 - x1*y2) / 2
    y_center = ( x2*(x1*x1 + y1*y1 + 1) - x1*(x2*x2 + y2*y2 +1) ) / (x2*y1 - x1*y2) / 2
    radius = np.sqrt(x_center**2 + y_center**2 - 1)
    start_angle = np.arctan2(x1 - x_center, y1 - y_center)
    end_angle = np.arctan2(x2 - x_center, y2 - y_center)
    if np.pi < start_angle-end_angle:
        start_angle -= 2*np.pi
    elif np.pi < end_angle - start_angle:
        end_angle -= 2*np.pi
    phi = np.linspace(start_angle, end_angle, number_of_samples)
    return x_center + radius * np.sin(phi), y_center + radius * np.cos(phi)

def edge_trace(x_coords, y_coords, width=1, color='black'):
    if isinstance(color, str):
        return go.Scattergl(x=x_coords, y=y_coords, mode='lines', line_width=width, line_color=color, showlegend=False)
    elif isinstance(color, int) and color < len(x_coords):
        div = (np.arange(color+1)*(len(x_coords)/color)).astype(int)
        traces = []
        for i in range(color):
            curcolor='rgb(0,{},{})'.format(int(255*i/(color-1)), int(255*(1-i/(color-1))))
            traces.append(go.Scattergl(x=x_coords[div[i]:div[i+1]+1], y=y_coords[div[i]:div[i+1]+1], mode='lines', line_width=width, line_color=curcolor, showlegend=False))
        return traces

def line(p1, p2, number_of_samples=100):
    return np.linspace(p1, p2, number_of_samples)

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
    N = 100
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    t = np.linspace(0,1,N)[:,None]
    return (p2 + (1-t)**2 * (p1 - p2) + t**2 * (p3 - p2)).T

def kamada_kawai(adjacency_list, graph_distance):
    N = len(adjacency_list)
    graph_distance = np.array(graph_distance)
    L0, K = 1.0, N
    diameter = graph_distance.max()
    if diameter == inf:
        return
    L = L0 / diameter
    des_length = L * graph_distance
    springs = K / (graph_distance + 1e-3*np.eye(N))**2
    np.fill_diagonal(springs, 0)
    angle = 2 * np.pi * np.linspace(0,1, N+1)[1:]
    pairs = combinations(range(N), r=2)
    invdist = 1 / (graph_distance + 1e-3 * np.eye(N))
    meanweight = 1e-3
    initial_position_array = np.hstack([np.sin(angle), np.cos(angle)]).ravel('F')

    def kamada_kawai_cost_function(position_array):
        coords = position_array.reshape(N, 2)
        coord_diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        vert_dist = np.sqrt(np.sum(coord_diff**2, axis=2))
        direction = coord_diff * (1 / (vert_dist + 1e-3 * np.eye(N)))[:,:,np.newaxis]
        offset = vert_dist * invdist - 1
        np.fill_diagonal(offset, 0)
        cost = 1/2 * np.sum(offset**2)
        something = (invdist * offset)[:,:,np.newaxis] * direction
        grad = np.sum(something, axis=1) - np.sum(something, axis=0)
        sum_pos = np.sum(coords, axis=0)
        cost += 1/2 * meanweight * np.sum(sum_pos ** 2)
        grad += meanweight * sum_pos
        return (cost, grad.ravel())

    def kamada_kawai_cost_function2(position_array):
        coords = position_array.reshape(N, 2)
        coord_diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        vert_dist = np.sqrt(np.sum(coord_diff**2, axis=2))
        direction = coord_diff * (1 / (vert_dist + 1e-3 * np.eye(N)))[:,:,np.newaxis]
        offset = vert_dist - des_length
        np.fill_diagonal(offset, 0)
        cost = np.sum(springs * offset ** 2) / 2
        something = (springs * offset)[:,:,np.newaxis] * direction
        grad = np.sum(something, axis=1) - np.sum(something, axis=0)
        sum_pos = np.sum(coords, axis=0)
        cost += 1/2 * meanweight * np.sum(sum_pos ** 2)
        grad += meanweight * sum_pos
        return (cost, grad.ravel())

    from scipy.optimize import minimize
    optres = minimize(
            kamada_kawai_cost_function,
            initial_position_array,
            method='L-BFGS-B',
            jac=True
            )
    return optres

def get_ellipse_path(center=(0,0), start_angle=0, end_angle=2*np.pi, radius_x=1, radius_y=1):
    t = np.linspace(start_angle, end_angle, NUM_OF_SAMPLES)
    xs = center[0] + radius_x * np.cos(t)
    ys = center[1] + radius_y * np.sin(t)

    coords_as_str = (f'{x},{y}' for x,y in zip(xs, ys))
    path = ' L '.join(coords_as_str)

    return 'M ' + path
