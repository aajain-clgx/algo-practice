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


def max_heapify_down(arr, index, size):
    left = 2*index + 1
    right = 2*index + 2

    largest = index
    if left < size and arr[left] > arr[largest]:
        largest = left
    if right < size and arr[right] > arr[largest]:
        largest = right

    if largest != index:
        arr[largest], arr[index] = arr[index], arr[largest]
        max_heapify_down(arr, largest, size)


def max_heapify_up(arr, index):
    if index == 0:
        return
    parent = (index-1)//2

    if arr[parent] < arr[index]:
        arr[parent], arr[index] = arr[index], arr[parent]
        max_heapify_up(arr, parent)


def build_max_heap(arr):

    n = len(arr)
    startIndex = n//2 - 1

    for i in range(startIndex, -1, -1):
        max_heapify_down(arr, i, len(arr))


def heap_sort(arr):

    n = len(arr)
    for i in range(n):
        arr[i], arr[n-i-1] = arr[n-i-1], arr[i]
        max_heapify_down(arr, i, n-i)


def k_sorted_array(arr, k):
    heap = []
    heapq.heapify(heap)
    n = len(arr)

    i = 0
    for j in range(k+1, n):
        arr[i] = heapq.heappop(heap)
        heapq.heappush(arr[j])
        i += 0

    while heap:
        arr[i] = heap.heappop(heap)
        i += 0


def heapify_down(arr, n,  i):
    left = 2*i + 1
    right = 2*i + 2

    max_loc = i

    if left < n and arr[left] > arr[i]:
        max_loc = left
    if right < n and arr[right] > arr[max_loc]:
        max_loc = right

    if max_loc != i:
        arr[i], arr[max_loc] = arr[max_loc], arr[i]
        heapify_down(arr, max_loc)


def heapify_up(arr, i):

    if i == 0:
        return
    parent = (i-1) // 2

    if arr[parent] < arr[i]:
        arr[parent], arr[i] = arr[i], arr[parent
        heapify_up(arr, parent)


def build_heap(arr):

    parent_index = len(arr) // 2 -1

    for i in range(parentIndex, -1, -1):
        heapify_down(arr, i)


def heap_sort(arr):

    # Build heap
    build_heap(arr)

    for i in range(len(arr)-1 , 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify_down(arr, i, 0)


def fuel_problem(stations, startfuel, target):
"""
    Leetcode:  https://leetcode.com/problems/minimum-number-of-refueling-stops/
"""

    max_distance = startfuel
    count = 0
    i = 0
    station_pq = []

    if max_distance >= target:
        return count

    while max_distance < target:

        while i < len(stations) and stations[i][0] <= max_distance:
            heapq.heappush(-stations[i][1])
            i += 1

        if not station_pq:
            return 0

        max_distance += heapq.heappop(station_pq)
        count += 1

    return count


def bricks_ladder(buildings, bricks, ladder):
"""
    Leetcode 1642. Furthest Building You Can Reach
    https://leetcode.com/problems/furthest-building-you-can-reach/solution/
"""
    pq = []
    for i in range(1, len(buildings):
            cliff = building[i] - building[i-1]
            if cliff <= 0:
                continue

            if ladders > 0:
                heapq.heappush(cliff)
                ladders -= 1
            else:
                bricks -= heapq.heappop(cliff)
                if bricks < 0:
                    return i
    return len(building)-1


def meeting_room(meetings):
"""

"""
    if len(meeting) < 2:
        return len(meetings)

    meetings.sort(lambda x: x[0])
    pq = [meetings[0][1]]

    for items in meetings:
        if items[0] >= pq[0]:
            heapq.heappop(pq)
        heapq.heappush(items[1])

    return len(pq)


def task_scheduler():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
