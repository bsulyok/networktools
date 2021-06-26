import numpy as np
from itertools import combinations
from math import inf, floor
import plotly.graph_objects as go

def hue_to_rgb(H):
    X = 255 * ( 1 - abs( ( (6*H) % 2 ) - 1 ) )
    if 0 <= H < 1/6:
        return 'rgb(255,{},0)'.format(floor(X))
    elif 1/6 <= H < 2/6:
        return 'rgb({},255,0)'.format(floor(X))
    elif 2/6 <= H < 3/6:
        return 'rgb(0,255,{})'.format(floor(X))
    elif 3/6 <= H < 4/6:
        return 'rgb(0,{},255)'.format(floor(X))
    elif 4/6 <= H < 5/6:
        return 'rgb({},0,255)'.format(floor(X))
    elif 5/6 <= H < 1:
        return 'rgb(255,0,{})'.format(floor(X))

def hsv_to_rgb(H, S, V):
    C = S * 255 * V
    M = 255 * V - C 
    X = C * ( 1 - abs( ( (H / 60) % 2) -1 ) )
    if 0 <= H < 60:
        return [floor(C+M), floor(X+M), floor(M)]
    elif 60 <= H < 120:
        return [floor(X+M), floor(C+M), floor(M)]
    elif 120 <= H < 180:
        return [floor(M), floor(C+M), floor(X+M)]
    elif 180 <= H < 240:
        return [floor(M), floor(X+M), floor(C+M)]
    elif 240 <= H < 300:
        return [floor(X+M), floor(M), floor(C+M)]
    elif 300 <= H < 360:
        return [floor(C+M), floor(M), floor(X+M)]

def generate_distinct_colours(N, S=0.5, V=0.5):
    S, V = 0.5, 0.5
    return [hsv_to_rgb(360*i/N,S,V) for i in range(N)]

def semi_circle(p1, p2, number_of_samples=100):
    return (p1+p2)/2 + abs(p1-p2) / 2 * np.exp(1j * np.linspace(0, np.pi, number_of_samples))

def poincare_line(r1, phi1, r2, phi2, number_of_samples=100):
    if r1 < 1e-5 or r2 < 1e-5 or abs((phi1 - phi2) % np.pi ) < 1e-5:
        return euclidean_line(r1, phi1, r2, phi2, 2)
    c = (r1+1/r1) / (r2+1/r2)
    phi0 = np.arctan( (np.cos(phi1) - c * np.cos(phi2)) / (c * np.sin(phi2) - np.sin(phi1)) )
    r0 = (r1 + 1/r1) / 2 / np.cos(phi1-phi0)
    start_angle, end_angle = phi1, phi2
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle
    if np.pi < end_angle - start_angle:
        end_angle -= 2*np.pi
    phi = np.linspace(start_angle, end_angle, number_of_samples)
    rcos = r0 * np.cos(phi - phi0)
    r = rcos - np.sqrt( rcos**2 - 1 )
    return r*np.cos(phi), r*np.sin(phi)

def poincare_line2(p1, p2, number_of_samples=100):
    if abs(p1) < 1e-5 or abs(p2) < 1e-5 or abs((p1/p2).imag) < 1e-5:
        return euclidean_line(p1, p2, 2)
    center = ( p1 * (1 + p2*p2.conjugate()) - p2 * (1 + p1*p1.conjugate()) ) / ( p1 * p2.conjugate() - p1.conjugate() * p2 )
    radius = np.sqrt(center*center.conjugate() - 1)
    start_angle = np.angle(p1-center)
    end_angle = np.angle(p2-center)
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle
    if np.pi < end_angle - start_angle:
        end_angle -= 2*np.pi
    phi = np.linspace(start_angle, end_angle, number_of_samples)
    return center + radius * np.exp(1j * phi)

def hyperbolic_polar_line(r1, phi1, r2, phi2, number_of_samples=100):
    if r1 < 1e-5 or r2 < 1e-5 or abs((phi1 - phi2) % np.pi ) < 1e-5:
        return euclidean_line(r1, phi1, r2, phi2, 2)
    A_1, A_2 = np.tanh(r1), np.tanh(r2)
    phi0 = np.arctan( - (A_1 * np.cos(phi1) - A_2 * np.cos(phi2)) / (A_1 * np.sin(phi1) - A_2 * np.sin(phi2)) )
    B = A_1 * np.cos(phi1 - phi0)
    start_angle, end_angle = phi1, phi2
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle
    if np.pi < end_angle - start_angle:
        end_angle -= 2*np.pi
    phi = np.linspace(start_angle, end_angle, number_of_samples)
    r = np.arctanh( B / np.cos(phi - phi0) )
    return r*np.cos(phi), r*np.sin(phi)

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

def euclidean_line(r1, phi1, r2, phi2, number_of_samples=2):
    return np.linspace(r1*np.cos(phi1), r2*np.cos(phi2), number_of_samples), np.linspace(r1*np.sin(phi1), r2*np.sin(phi2), number_of_samples)

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
