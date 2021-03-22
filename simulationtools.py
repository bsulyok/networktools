from math import exp, inf
from random import random, seed
import numpy as np
from common import priority_queue
from itertools import combinations

def annealing(elementary_step, state, steps=1000):
    '''
    Minimalistic simulated annealing.
    '''
    seed(123456)
    for step in range(steps):
        new_state, energy_change = elementary_step(state)
        if energy_change < 0 or random() < exp(-energy_change * (1-step/steps)):
            state = new_state
    return state
