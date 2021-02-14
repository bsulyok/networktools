from math import exp
from random import random, seed

def annealing(elementary_step, state, max_step=1000):
    seed(123456)
    for step in range(max_step):
        new_state, energy_change = elementary_step(state)
        if energy_change < 0 or random() < exp(-energy_change * (1-step/max_step)):
            state = new_state
    return state

