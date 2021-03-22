from timeit import default_timer as timer
import heapq

class priority_queue:
    def __init__(self):
        self.queue = []
        heapq.heapify(self.queue)
    def pop(self):
        item = heapq.heappop(self.queue)
        return item
    def push(self, item):
        heapq.heappush(self.queue, item)
    def __len__(self):
        return len(self.queue)
    def min(self):
        return min(self.queue)[0]
    def max(self):
        return max(self.queue)[0]

def elapsed(func, reps=1, **kwargs):
    start = timer()
    for i in range(reps):
        _ = func(**kwargs)
    dt = timer() - start
    print('Average time over {} rounds: {} s'.format(reps,dt/reps))
    return

def inverse_permutation(perm):
    return [i for i, j in sorted(enumerate(perm), key=lambda i_j: i_j[1])]

def edge_iterator(adjacency_list):
    '''
    Generator object yielding edges from the provided adjacency list.
    Parameters
    ----------
    adjacency_list : dict of dicts
    Returns
    vertex : int
        Source vertex.
    neighbour : int
        Target vertex.
    attributes : dict
        Attributes of the given edge.
    '''
    for vertex, neighbourhood in adjacency_list.items():
        for neighbour, attributes in neighbourhood.items():
            yield vertex, neighbour, attributes

def ienumerate(iterable):
    for index, item in enumerate(iterable):
        yield item, index
