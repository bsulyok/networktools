from timeit import default_timer as timer
import heapq
import disjoint_set as ds

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

class disjoint_set():
    def __init__(self, N):
        self._data = {i:i for i in range(N)}
        self._size = {i:1 for i in range(N)}
    def __contains__(self, item):
        return item in self._data
    def __iter__(self):
        return iter(self._data.items())
    def find(self, item):
        while item != self._data[item]:
            self._data[item] = self._data[self._data[item]]
            item = self._data[item]
        return item
    def union(self, item_1, item_2):
        item_1, item_2 = self.find(item_1), self.find(item_2)
        if item_1 != item_2:
            size_1, size_2 = self._size[item_1], self._size[item_2]
            if size_1 < size_2:
                item_1, item_2 = item_2, item_1
            self._data[item_2] = item_1
            self._size[item_1] = size_1 + size_2
    def size(self, item):
        return self._size[self.find(item)]
    def is_connected(self, item_1, item_2):
        return self.find(item_1) == self.find(item_2)

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

def ienumerate(iterable, start=0):
    for index, item in enumerate(iterable, start=0):
        yield item, index