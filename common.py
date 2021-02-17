from timeit import default_timer as timer
import heapq

class priority_queue:
    def __init__(self, reverse=False):
        self.queue = []
        heapq.heapify(self.queue)
        self.reverse=reverse
    def pop(self):
        item = heapq.heappop(self.queue)
        if self.reverse:
            return (-item[0], item[1])
        else:
            return item
    def push(self, item):
        if self.reverse:
            heapq.heappush(self.queue, (-item[0], item[1]))
        else:
            heapq.heappush(self.queue, item)
    def __len__(self):
        return len(self.queue)
    def min(self):
        if self.reverse:
            return -max(self.queue)[0]
        else:
            return min(self.queue)[0]
    def max(self):
        if self.reverse:
            return -min(self.queue)[0]
        else:
            return max(self.queue)[0]

def elapsed(func, reps=1):
    start = timer()
    for _ in range(reps):
        func
    dt = timer() - start
    print('Average time over {} rounds: {} s'.format(reps, dt))
    return

def inverse_permutation(perm):
    return [i for i, j in sorted(enumerate(perm), key=lambda i_j: i_j[1])]
