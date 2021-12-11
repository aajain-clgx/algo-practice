#!/usr/bin/env python3

from collections import defaultdict, deque, heapq
import math

"""
    Questions to ask on any graph problem:

    * Directed or Undirected edge
    * Weighted or unweighted edge?
    * Cyclic or acyclic ?

    Most popular techniques:

    * DFS with backtracking + greedy
    * BFS  (when thinking shortest)
    * BFS variations to real world problems (currency exchange etc)
    * Shortest path variations (Djikstra)
    * Connected Components / Union Find
    * TopSort
    * Variations of coloring
    * Hard: Euler path, Hamiltonian path,  Strongly connected components


    Complexity


    What applies to directed vs undirected path?

"""


class Graph:

    def __init__(self):
        self.graph = defaultdict(list)

    def addnode(self, node, link_node):
        self.graph[node].append(link_node)

    def bfs(self, node):

        visited = [None] * len(self.graph)
        queue = deque()

        queue.append(node)
        visited[node] = True

        while queue:
            n = queue.popleft()
            for i in self.graph[n]:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True

    def dfs(self, start):

        def dfs_util(node, visited):
            visited[node] = True
            for n in self.graph[node]:
                if not visited[n]:
                    self.dfs(n, self.graph)

        visited = [None] * len(self.graph)
        dfs_util(start, visited)

    def dfs_cycle(self):

        def dfs_cycle_util(start):
            visited[start] = True
            path[start] = True

            for n in self.graph[start]:
                if path[n]:
                    return True
                else:
                    if not visited[n]:
                        return dfs_cycle_util(n)

            path[start] = False
            return False

        visited = [False] * len(self.graph)
        path = [False] * len(self.graph)

        for node in self.graph:
            if not visited[node]:
                if dfs_cycle_util(node):
                    return True

        return False

    def def_cycle_color(self):

        def dfs_util(node):
            color[node] = "GREY"

            for n in self.graph[node]:
                if color[node] == "WHITE":
                    if dfs_util(n):
                        return True
                elif color[node] == "GREY":
                    return True

            color[node] = "BLACK"
            return False

        color = ["WHITE"] * len(self.graph)

        for node in self.graph:
            if color[node] == "WHITE" and dfs_util(node):
                return True

        return False

    def topological_sort(self):

        def top_util(node):
            visited[node] = True

            for i in self.graph[node]:
                if not visited[node]:
                    top_util(node)

            toporder.appendleft(node)

        visited = [False] * len(self.graph)
        toporder = deque()

        for i in self.graph:
            if not visited[i]:
                top_util(i)

        return toporder

    def topsort2(self):

        size = len(self.graph)
        in_degree = [0] * size
        queue = deque()

        for i in self.graph:
            for edge in self.graph[i]:
                in_degree[edge] += 1

        for j in range(size):
            if in_degree[j] == 0:
                queue.append(j)

        index = 0
        topsort = [0] * size

        while queue:

            n = queue.popleft()
            topsort[index] = n
            index += 1

            for j in self.graph[n]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

        if index != len(self.graph):
            print("Cycle found")

        return topsort

    def undirected_cycle(self):

        def cycle_util(node, parent):

            visited[node] = True
            for n in self.graph[node]:
                if not visited[n]:
                    if cycle_util(n, node):
                        return True
                elif n != parent:
                    return True
            return False

        visited = [False] * len(self.graph)
        for i in self.graph:
            if not visited[i]:
                if cycle_util(i, -1):
                    return True

        return False

    def bipartite(self):

        def bipartite_util(start):
            queue = deque()
            queue.append(start)
            color[start] = 1

            while queue:
                n = queue.popleft()
                for i in self.graph[n]:
                    if color[i] == -1:
                        color[i] = 1 - color[n]
                        queue.append(i)

                    elif color[i] == color[n]:
                        return False

            return True

        color = [-1] * len(self.graph)
        for i in self.graph:
            if not bipartite_util(i):
                return False

        return True

    def djikstra(self, start, end=None):

        distance = [math.inf] * len(self.graph)
        prev = [None] * len(self.graph)
        visited = [False] * len(self.graph)

        queue = []
        queue.append((0, start))
        heapq.heapify(queue)
        visited[start] = True
        distance[start] = 0

        while queue:
            cost, node = heapq.heappop()
            visited[node] = True

            for n in self.graph[node]:
                if not visited[n]:
                    new_value = min(distance[n], distance[node] + cost)
                    if new_value < distance[n]:
                        distance[n] = new_value
                        prev[n] = node
                        heapq.heappush((distance[n], n))

            if node == end:
                return distance, prev

        return distance, prev

    def bellman_ford(self):
        dist = [math.inf] * len(self.graph)

        for i in self.graph:
            for node, weight in self.graph[i]:
                new_dist = min(dist[node], dist[i] + weight)
                if new_dist < dist[node]:
                    dist[node] = new_dist

        for i in self.graph:
            for node, weight in self.graph[i]:
                if dist[node] > dist[i] + weight:
                    dist[node] = math.inf

        return dist

    def tarjan_scc(self):

        def dfs_util(node):
            nonlocal time

            disc[node] = time
            low[node] = time
            time += 1
            stack.append(node)
            stack_p[node] = True

            for n in self.graph[node]:
                if disc[n] == -1:
                    dfs_util(n)
                    low[n] = min(low[n], low[node])

                elif stack[n]:
                    low[n] = min(low[n], disc[node])

            if low[node] == disc[node]:
                while stack[-1] != node:
                    item = stack.pop()
                    low[item] = low[node]
                    stack_p[item] = False

        low = [-1] * len(self.graph)
        disc = [-1] * len(self.graph)
        time = 0
        stack = []
        stack_p = [False] * len(self.graph)

        for i in self.graph:
            if disc[i] == -1:
                dfs_util(i)

        return low

    def bridges(self):

        def dfs_util(node):
            nonlocal time

            disc[node] = time
            low[node] = time
            time += 1
            stack.append(time)
            stack_p[node] = True

            for n in self.graph[node]:
                if disc[n] == -1:
                    dfs_util(n)
                    low[n] = min(low[n], low[node])

                    if low[n] > disc[node]:
                        bridge_list.add((node, n))

                elif stack_p[n]:
                    low[n] = min(low[n], disc[node])

        disc = [-1] * len(self.graph)
        low = [-1] * len(self.graph)
        stack = []
        time = 0
        bridge_list = []
        stack_p = [False] * len(self.graph)

        for i in self.graph:
            if disc[i] == -1:
                dfs_util(i)

    def prims_mst(self):

        mst = [0] * len(self.graph)
        mst_count = 0
        mst_cost = 0
        visited = [False] * len(self.graph)

        queue = []
        for i in len(self.graph[0]):
            queue.append((i[0], i[1], 0))
        visited[0] = True

        heapq.heapify(queue)

        while queue:

            weight, node_to, node_from = heapq.heappop(queue)
            if not visited[node_to]:
                mst_cost += weight
                mst[mst_count] = node_to
                mst_count += 1
                visited[node_to] = True

                for i in self.graph[node_to]:
                    queue.append((i[0], i[1], node_to))

        if mst_count != len(self.graph):
            print("No MST")

        return mst, mst_cost


def djikstra(maze):

    rowmax = len(maze)
    colmax = len(maze[0])

    cost = [[math.inf for i in range(colmax)] for j in rowmax]
    visited = [[False for i in range(colmax)] for j in rowmax]
    path = [[None for i in range(colmax)] for j in rowmax]

    queue = [(maze[0][0], 0, 0)]
    heapq.heapify(queue)

    visited[0][0] = True
    cost[0][0] = 0

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    while queue:

        weight, parent_x, parent_y = heapq.heappop(queue)
        visited[parent_x][parent_y] = True

        for i in range(len(dx)):
            x = parent_x + dx
            y = parent_y + dy

            if x >= 0 and x < colmax and y >= 0 \
                    and y < rowmax and not visited[x][y]:
                new_cost = min(cost[x][y], cost[parent_x][parent_y] + weight)
                if new_cost < cost[x][y]:
                    cost[x][y] = new_cost
                    path[x][y] = (parent_x, parent_y)
                    heapq.heappush(queue, (cost[x][y], x, y))

    return cost[colmax-1][rowmax-1], path


def hamiltonian_cycle(graph):

    def is_safe(x, y):
        if x >= 0 and x < size and y >= 0 and y < size:
            return True

        return False

    def dfs_util(node, path):

        if len(path) == n and graph[path[-1]]path[0]]] == 1:
            all_cycles.append(path)
            return

        for i in range(size):

            if is_safe(node, i) and not visited[node][i]:

                path.append(i)
                visited[i] = True

                dfs_util(i, path)

                path.pop()
                visited[node][i] = False

    size = len(graph)
    visited = [False] * size
    all_cycles = []
    dfs_util(0, [0])


def m_colorable(graph, m):

    def is_safe(node, color):

        for i in range(len(graph)):
            if graph[node][i] and  color[i] == color:
                return False
        return True

    def dfs_util(node):

        if node == len(graph)-1:
            return True

        for c in range(1, m):
            if is_safe(node, c):
                color[node] = c
                if dfs_util(node+1):
                    return True

                color[node] = 0


        return False

    color = [0] * len(graph)
    return dfs_util(0)


def nqueen(graph):

    def is_safe(col, row):

        if col in cols or col+row in diagonal or row-col in antidiagonal:
            return False
        return True

    def dfs_util(row):

        if row == size:
            return True

        for col in range(size):
            if is_safe(col, row):
                cols.add(col)
                diagonal.add(col+row)
                antidiagonal.add(row-col)

                if dfs_util(row+1):
                    return True

                cols.remove(col)
                diagonal.remove(row+col)
                antidiagonal.remove(row-col)

        return False

    size = len(graph)
    cols = set()
    diagonal = set()
    antidiagonal = set()

    return dfs_util(0)


def bfs_search(graph, start, end):

    queue = deque()
    visited = [False] * len(graph)
    level = [0] *  len(graph)
    path = [-1] * len(graph)

    queue.append(start)
    visited[start] = True

    while queue:
        n = queue.popleft()
        if n == end:
            return level, path

        for i in self.graph[n]:
            if not visited[i]:
                level[i] = level[n] +1
                path[i] = n
                visited[i] = True
                queue.append(i)


    return  None, None


def breath_first_0_1(graph, start, end):
    queue = deque()
    visited = set()
    dist = [math.inf]*len(graph)

    queue.append(start)

    while queue:
        node = queue.popleft()
        visited.add(node)

        if node == end:
            break

        for i in range(len(graph)):
            if not visited(i) and graph[node][i]:
                new_dist = min(dist[i], dist[node] + graph[node][i])
                if new_dist < dist[i]:
                    if graph[node][i] == 0:
                        queue.appendleft(i)
                    else:
                        queue.append(i)


    return dist[end]


def longest_increasing_matrix(graph):

    def is_safe(i, j):
        if i < rowmax and i >= 0 and j < colmax and j >=0:
            return True
        return False


    def dfs(i, j):

        if cache[i][j] != 0:
            return cache[i][j]

        ans = 1
        for i in range(4):
            row = i + dy
            col = j + dx
            if is_safe(row, col) and graph[row, col] > graph[i][j]:
                ans = max(ans, 1 + dfs(row, col)

        cache[i][j] = ans
        return ans


    dx = [0, 0, -1, 1]
    dy = [1, -1, 0, 0]

    colmax = len(graph[0])
    rowmax = len(graph)
    cache = [[0 for i in range(rowmax)] for j in range colmax]

    max_len = 0
    for i in range(rowmax):
        for j in range(colmax)
            max_len = max(max_len, dfs(i, j))

    return max_len

def eulerian_path(graph):

    def euler_path_exists(graph):

        end_nodes = 0
        start_nodes = 0

        for node in graph:
            for edge in graph[node]:
                in_degree[edge] += 1
                out_degre[node] += 1

        for i in range(len(graph)):
            if abs(out_degree[i]-in_degree[i]) > 1:
                return False

            elif out_degree[i] == in_degree[i]:
                continue
            elif out_degree[i] - in_degree[i] == 1
                start_nodes += 1
            elif in_degree[i] - out_degree[i] == 1
                end_nodes += 1

        if end_nodes == 1 and start_edges == 1 or \
                end_nodes == 0 and start_edges == 0:
            return True


    def find_starting_node(graph):

        node = -1

        for i in range(len(graph)):
            if out_degree[i] - in_degree[i] == 1:
                node = i
            elif out_degree[i] > 0:
                node = i

        return node


    def euler_dfs(node):

        while out_degree[node] != 0:
           next_node = graph[node][out_degree[node]]
           out_degree[node] -= 1

           euler_dfs(next_node)

        euler_path.appendleft(node)


    in_degree = [0] * len(graph)
    out_degree = [0] * len(graph)
    euler_path = deque()


    if euler_path_exists(graph):
        start_node = find_starting_node(graph)
        euler_dfs(start_node)

        print(euler_path)

    else:
        print("No such path exists")
