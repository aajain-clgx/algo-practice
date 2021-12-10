"""
    Monotonic Queue

    Useful for finding problems with window and next greater / smaller constraints
    Can be implemented as stack (if only need to insert at one end) or deque (both end)
"""

from collections import deque


def next_greater(arr):
    """
        Solved with monotonically decreasing queue
        If next inserted element is greater - triggers cleanup
    """
    n = len(arr)
    result = [-1] * n
    stack = []

    for i in range(n):
        # push new entry while cleaning
        # Simulate monotonically decreasing queue

        while stack and arr[i] >= stack[-1]:
            item = stack.pop()
            result[item] = i

        stack.append(i)

    return result


def next_smaller(arr):
    """
        Solved with monotonically increasing queue
        Next smaller element triggers cleanup
    """
    n = len(arr)
    result = [-1] * n
    stack = []

    for i in range(n):
        # Push new entry while cleaning
        while stack and arr[i] <= stack[-1]:
            item = stack.pop()
            result[item] = i

        stack.append(i)

    return result


def area_histogram(arr):
    """
        Max area is bounded by next smallest height
        Using a monotonically increasing queue and

        Smaller height item = right boundary
        We compute left boundary for items in the stack
        since it it monotonically increasing, it is the last element

        Finally for anything left in the stack, we pop and it should span
        the entire range of array
    """

    n = len(arr)
    if n == 0:
        return 0

    area = 0
    stack = []

    for i in range(n):
        # Clean up stack before inserting element (M. Increasing)
        while stack and arr[i] < stack[-1]:
            item = stack.pop()
            left_limit = -1 if not stack else stack[-1]
            width = i - left_limit - 1
            area = max(area, arr[item] * width)

        stack.append(i)

    while stack:
        item = stack.pop()
        area = max(arr[item] * (n - item - 1))

    return area


def max_rectangle_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    dp = [0] * cols
    area = 0

    for i in len(rows):
        for j in len(cols):
            if matrix[i][j]:
                dp[j] += 1
            else:
                dp[i] = 0

        area = max(area, area_histogram(dp))

    return area


def sliding_window_max(arr, k):

    n = len(arr)
    queue = deque()
    result = []

    for i in range(xxk,an):
        # Pop element from front that are outside window
        while queue and (i - queue[0] + 1) > k:
            queue.popleft()

        # Keep queue monotonically decreasing
        while queue and arr[i] > arr[queue[-1]]:
            queue.pop()

        queue.append(i)

        if i >= k-1:
            result.append(queue[0])

    return result


def minimum_size_subrarry_sum_greater_K(arr, k):
"""
    In subarray sum, try keeping prefix array to simplify

    P[0] = 0
    P[1] = a[0]
    P[i] = a[i-1] + P[i-1]

    P[j] - P[i] = range from i to j-1
    P[j] - P[i] >= K

    In kept in increasing order of P[i], as soon as you reach K sum,
    do not need to look further and start trimming
"""

    if not arr:
        return -1

    n = len(arr)
    prefix =[0] * n
    for i in range(1,n):
        prefix[i] += prefix[i-1] + arr[i-1]

    queue = deque()
    result = math.inf

    for i in range(n):
        # Clean the entry
        while queue and prefix[queue[-1]] > prefix[arr[i]]:
            queue.pop()

        # Check for sum and truncate
        while queue and prefix[queue[0]] - prefix[arr[i]] >= k:
            result = min(result, i - queue[0])
            queue.popleft()

        queue.append(i)

    return -1 if result is math.inf else result
