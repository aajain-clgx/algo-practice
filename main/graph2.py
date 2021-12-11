#!/usr/bin/env python3

from collections import defaultdict
from collections import deque
from collections import heapq


class Graph:

    def __init__(self):
        self.graph = defaultdict(list)

    def append(self, src, node):
        self.graph[src].append(node)

    def breadth_first_search(self, node):

        visited = set()
        queue = []
        queue.append(node)
        visited.append(node)

        while queue:
            n = queue.pop(0)
            for neighbors in self.graph[n]:
                if neighbors not in visited:
                    queue.append(neighbors)
                    visited.append(neighbors)

    def dfs(self, start):

        def dfs_util(start, visited):

            visited.add(start)

            for n in self.graph[start]:
                if n not in visited:
                    self.dfs(n, visited)

        visited = set()
        dfs_util(start, visited)


class UnionFind:

    def __init__(self, size):
        self.id = [i for i in range(size)]
        self.sz = [1] * size
        self.num_components = size

    def find(self, p):

        root = p
        while root != self.id[root]:
            root = self.id[root]

        # Path compression
        while p != root:
            next_id = self.id[p]
            self.id[p] = root
            p = next_id

        return root

    def union(self, p, q):
        root1 = self.find(p)
        root2 = self.find(q)

        if root1 == root2:
            return

        if self.sz[root1] < self.sz[root2]:
            self.sz[root2] += self.sz[root1]
            self.id[root1] = root2
            self.sz[root1] = 0
        else:
            self.sz[root1] += self.sz[root2]
            self.id[root2] = self.id[root1]
            self.sz[root2] = 0

        self.num_components -= 1


def dfs_cycle_directed(graph):

    def dfs_util(node, visited, path):

        visited[node] = True
        path[node] = True

        for i in graph[node]:
            if path[i]:
                return True
            if not visited[i]:
                if dfs_util(i, visited, path):
                    return True

        path[node] = False
        return False

    visited = [False] * len(graph)
    path = [False] * len(graph)

    for i in range(len(visited)):
        if not visited[i]:
            if dfs_util(i, visited, path):
                return True

    return False


def dfs_cycle_colors(graph):

    def dfs_util(node, visited):

        visited[node] = "GREY"

        for i in graph[node]:
            if visited[i] == "GREY":
                return True
            if visited[i] == "WHITE" and dfs_util(i, visited):
                return True

        visited[node] = "BLACK"
        return False

    visited = ["WHITE"] * len(graph)

    for i in graph:
        if visited[i] == "WHITE":
            if dfs_util(i, visited):
                return True

    return False


def dfs_cycle_undirected(graph):

    def dfs_util(node, parent):
        visited[node] = True

        for i in graph[node]:
            if not visited:
                if dfs_util(i, node):
                    return True
            elif i != parent:
                return True

        return False

    visited = [False] * len(graph)
    for i in range(len(visited)):
        if not visited[i]:
            if dfs_util(i, i):
                return True

    return False


def cycle_length_n_undirected(graph, n):
    """
        Use DFS for length N, cycle if ends at start node
        print path long the way

        For a cycle of length N, need to search n-1 starting from a node
        Overall only need to search vertex = V- (n-1) for a cycle of n-1

    """

    def cycle_util(node, n, start, path, count):

        if n == 0:

            if graph[node][start] == 1:
                count += 1
                cycle_path.append(node)
                return count
            else:
                return count

        for i in range(len(graph)):

            if not visited[i] and graph[node][i]:
                visited[i] = True
                path.append(i)

                count = cycle_util(i, n-1, start, path + [i], count)

                visited[i] = False
                path.pop()

        return count

    len_graph = len(graph)
    visited = [False] * len_graph
    cycle_path = []
    count = 0

    for i in range(len_graph - (n-1)):
        count += cycle_util(i, n-1, i, [i], count)
        visited[i] = True

    count //= 2
    print(count)
    print(cycle_path)


def dfs_topological_sort(graph):

    def dfs_util(node, visited, topo_list):

        visited[node] = True
        for i in graph[node]:
            if not visited[node]:
                dfs_util(i, visited, topo_list)

        topo_list.appendleft(node)

    visited = [False] * len(graph)
    topo_list = deque()

    for i in graph:
        if not visited[i]:
            dfs_util(i, visited, topo_list)

    return topo_list


def topological_sort_queue(graph):

    size = len(graph)
    in_degree = [0] * size
    queue = []

    for i in graph:
        for j in i:
            in_degree[j] += 1

    for i in range(size):
        if in_degree[i] == 0:
            queue.append(i)

    index = 0
    toporder = [0] * size

    while queue:
        node = queue.pop(0)
        toporder[index] = node
        index += 1

        for j in graph[node]:
            in_degree[j] -= 1
            if in_degree[j] == 0:
                queue.append(j)

    if index != size:
        raise("Cycle found")

    return toporder


def is_bipartite(graph):

    def bipartite_util(node, graph, colors):
        colors[node] = 1
        queue = deque()
        queue.append(node)

        while queue:
            n = queue.popleft()

            for i in graph[n]:
                if colors[i] == -1:
                    colors[i] = 1 - colors[n]
                    queue.append(i)
                elif colors[i] == colors[n]:
                    return False

        return True

    colors = [-1] * len(graph)
    for i in graph:
        if colors[i] == -1:
            if not bipartite_util(i, graph, colors):
                return False

    return True


def single_source_shortest_path_dag(graph):
    """
        Given a graphs  node -> [ (node, weight), ...]
        Complexity: time = O(V+E)

    """
    toporder = topological_sort_queue(graph)
    distance = [None] * len(graph)
    distance[toporder[0]] = 0

    for i in range(len(toporder)):

        for node, weight in graph[i]:
            if distance[node] is None:
                distance[node] = weight
            else:
                new_dist = distance[node] + weight
                distance[node] = min(distance[node], new_dist)

    return distance


def djikstra_shortest_path(graph, src, end=None):
    import math

    distance = [math.inf] * len(graph)
    visited = [False] * len(graph)
    prev = [None] * len(graph)

    q = [(0, src)]
    distance[src] = 0
    heapq.heapify(q)

    while q:

        weight, node = heapq.heappop()
        visited[node] = True

        for w, n in graph[node]:
            # Optimization because lower weight edge is already seen
            if visited[n]:
                continue

            new_dist = min(distance[n], distance[node] + weight)
            if new_dist < distance[n]:
                prev[n] = node
                distance[n] = new_dist
                heapq.heappush((distance[n], n))

        if not end:
            if node == end:
                return distance

    if not end:
        path = [end]
        i = prev[end]
        while i is not None:
            path.append(0, prev[i])
            i = prev[i]

        print(path)


def breadth_first_search(graph, start, end):

    def bfs_util(start, end):

        queue = deque()
        visited = [False] * len(graph)

        queue.append(start)
        visited[start] = True

        while queue:
            n = queue.popleft()

            if n == end:
                return True

            for i in graph[n]:
                if not visited[i]:
                    visited[i] = True
                    dist[i] = dist[n] + 1
                    prev[i] = n
                    queue.append(i)

        return False

    dist = [0] * len(graph)
    prev = [None] * len(graph)
    if not bfs_util(start, end):
        print("Not Found")
    else:
        index = end
        while prev[index] is not None:
            print(index)
            index = prev[index]


def rat_in_maze(matrix, mat_len):

    def is_safe(x, y, matrix, N, visited):
        if x < 0 and x >= N or y < 0 and y >= N:
            return False
        if matrix[x][y] == 0:
            return False
        if visited[x][y] == 1:
            return False

        return True

    def dfs_util(x, y, matrix, visited, path):

        visited[x][y] = 1

        if x == mat_len-1 and y == mat_len-1 and matrix[x][y] == 1:
            all_path.append(path)
            visited[x][y] = 0
            return

        for i in range(4):
            new_row = x + rc[i]
            new_col = y + cc[i]

            if is_safe(new_col, new_row, matrix, mat_len, visited):
                dfs_util(new_col, new_row, matrix, visited, path + dc[i])

        visited[x][y] = 0
        path.pop()
        return

    visited = [[0 for x in range(mat_len)] for y in range(mat_len)]
    all_path = []
    dc = ["U", "D", "L", "R"]
    rc = [-1, 1, 0, 0]
    cc = [0, 0, -1, 1]

    dfs_util(0, 0, matrix`, visited, [])


def dijkstra_maze(graph):

    rowmax = len(graph)
    colmax = len(graph[0]

    visited = [ [0 for i in colmax] for j in  rowmax]
    cost = [ [ math.inf for i in colmax] for j in rowax]

    queue = [(graph[0][0], 0, 0)]
    heapq.heapify(queue)
    visited [0][0] = 1

    dy = [-1, 1, 0, 0]
    dx = [ 0, 0, 1, -1]

    while queue:
        cost, x, y = heapq.heapop(queue)
        visited[x][y] = 1

        for i in range(len(4)):
            new_x = x + dx[i]
            new_y = y + dx[i]

            if is_safe(new_x, new_y) and not visited[new_x][new_y]:
                    new_cost = min(graph[new_x][new_y], cost[x][x] + graph[new_x][new_y])
                    if new_cost < cost[new_x][new_y]:
                        cost[new_x][new_y] = new_cost
                        queue.heappush((new_cost, new_x, new_y))

                    if new_x, new_y ==  colmax-1, rowmax-1:
                        break

    return cost[colmax-1][rowmax-1]


def hamiltonian_cycle(graph):

    def is_safe(start, end, graph, visited):
        if graph[start][end] == 1 and not visited(end):
            return True

        return False

    def dfs_util(node, visited, path, n):

        if len(path) == n:
            if graph[node][start_node]:
                print(path)
                return True
            return False

        for i in range(len(graph)):
            if is_safe(node, i, graph, visited):
                visited[i] = True
                path.append(i)

                if dfs_util(i, visited, path, n):
                    return True

                visited[i] = False
                path.pop()

        return False


    visited = [False] * len(graph)
    start_node = 0
    visited[start_node] = True
    path = [start_node]

    return dfs_util(start_node, visited, path, len(graph))


def all_hamiltonian_cycle(graph):

        def is_safe(start, end, graph, visited):
            if graph[start][end] == 1 and not visited(end):
                return True
            return False

        def dfs_util(node, graph, visited, path, n):

            if len(path) == n:
                if graph[node][start_node] == 1:
                    all_paths.append(path)
                 return

            for i in range(n):
                if is_safe(node, i, graph, visited):
                    visited[i] = True
                    path.append(i)

                    dfs_util(i, graph, visited, path, n)

                    visited[i] = False
                    path.pop()


        all_paths  = []
        visited = [False] * len(graph)
        n = len(graph)
        start_node = 0
        path = [start_node]
        visited[start_node] = True

        all_hamiltonian_cycle(start_node, graph, visited, n)

        print(all_paths)


def prims_mst(graph):

    mst_max_count = len(graph) - 1
    mst_cost = 0
    edge_count = 0

    mst = [None] * len(graph)
    visited = [False] * len(graph)
    heap = []

    # Starting with node 0
    for i in graph[0]:
        heap.heappush((i[1], i[0], 0))
    visited[0] = True

    while heap:
        weight, to, from = heapq.heappop(heap)
        if not visited[to]:
            mst_cost += weight
            mst[edge_count] = to
            edge_count += 1

            for next_weight, next_item in graph[to]:
                heapq.heappush((next_weight, next_item, to))

    if edge_count != mst_max_count:
        return None

    return mst
