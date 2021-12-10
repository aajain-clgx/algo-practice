from collections import heapq


class StreamMedian:
    def __init__(self):
        self.lower = []
        self.upper = []

    def add_number(self, num):
        heapq.heappush(self.lower, -1 * num)

    def rebalance(self):
        # if lower is greater than upper
        if self.lower and self.upper and self.lower[0] > self.upper[0]:
            val = -1 * heapq.heappop(self.lower[0])
            heapq.heappush(self.upper, val)

        # if the size is different
        if len(self.lower) > (len(self.upper) + 1):
            val = -1 * heapq.heappop(self.lower[0])
            heapq.heappush(self.upper, val)

        if len(self.upper) > len(self.lower) + 1:
            val = -1 * heapq.heappop(self.upper)
            heapq.heappush(self.lower, val)

    def median(self, arr):
        for n in arr:
            self.add_number(n)
        self.rebalance()

        if len(self.lower) == len(self.upper):
            return (self.lower + self.upper) / 2
        if len(self.lower) > len(self.upper):
            return self.lower[0]
        else:
            return self.upper[0]
