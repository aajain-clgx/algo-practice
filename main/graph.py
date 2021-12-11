#!/usr/bin/env python3

from collections import deque
import math
import pdb
import heapq

def breadth_first(graph, start):

    visited, queue = [], deque(start)

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            
            neighbors = graph[node]
            for n in neighbors:
                queue.append(n)

    return visited


def breadth_first_dijkstra(graph, start):

    visited, pqueue = [], []
    heapq.heappush(pqueue, (0, start))
    cost_so_far = {start:0}
    last_visited = {start: None}
    
    while pqueue:

        _,node = heapq.heappop(pqueue)
        if node not in visited:
            visited.append(node)
            
            for n in graph[node]:
                new_cost = cost_so_far[node] + graph[node][n]
                if n not in cost_so_far or new_cost < cost_so_far[n]:
                    cost_so_far[n] = new_cos
                    last_visited[n] = node
                    priority = new_cost
                    heapq.heappush(pqueue,(priority, n))
                    
    return cost_so_far, last_visited


def breadth_first_path(graph, start, goal):

    visited, queue = [], deque([start])

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node not in visited:
            visited.append(node)
            neighbors = graph[node]
            
            for n in neighbors:
                newpath = list(path)
                newpath.append(n)
                queue.append(newpath)
                
                if n == goal:
                    return newpath

    print("No path found")
        

def depth_first(graph, start):
    visited, stack = [], [start]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
    
            neighbors = graph[node]
            for n in neighbors:
                stack.append(n)

    return visited


def depth_first_recursive(graph, start, visited=None):
        if visited is None:
            visited = []

        visited.append(start)
        for n in graph[start]:
            if n not in visited:
                depth_first_recursive(graph, n, visited)

        return visited


def breadth_first_recursive(graph, start, queue=None, visited=None):
        if visited is None and queue is None:
            visited, queue = [], deque(start)

        if not queue:
            return
        node = queue.popleft()
                    
        for n in graph[node]:
            if n not in visited:
                visited.append(n)
                queue.append(n)
        breadth_first_recursive(graph, n, queue, visited)
        

def depth_first_path(graph, start, goal):

    visited, stack = [], [[start]]
    while stack:
        
        path = stack.pop()
        node = path[-1]

        if node not in visited:
            visited.append(node)

            for n in graph[node]:
                newpath = list(path)
                newpath.append(n)
                stack.append(newpath)
            
                if n == goal:
                    return newpath

    print("No path found")



def flood_fill_dfs(graph, startx, starty, m, n, replacement, visited = None):
    """ Given a MxN matrix, replace start node and surrounding node with 
        target color
    """
 
    def issafe(x, y,color):
        return (x < m and y < n and graph[m][n] == color)
         
    if visited is None:
        visited = []

    visited.append([startx, starty])
    color = graph[startx][starty]
    
    graph[startx][starty] = replacement

    for i in range(-1, 2):
        for j in range(-1, 2):
            if issafe(startx + i, starty + y, color) and [startx+i, starty+j] not in visited:
                flood_fill_dfs(graph, startx+i, starty+j, m, n, replacement, visited)
    

def flood_fill_bfs(graph, startx, starty, m, n, replacement):

    def issafe(x, y,color):
        return (x < m and y < n and graph[m][n] == color)
         
    visited, queue = [], deque([startx, starty])
    color = graph[startx][starty]
    
    while queue:
        node = queue.popleft()
        graph[node[0]][node[1]] = replacement

        if node not in visited:
            visited.append([node[0],node[1]]) 
            for i in range(-1, 1):
                for j in range(-1, 1):
                    if issafe(startx + i, starty+ j, color):
                        queue.append([startx+i, starty+j])


def dfs_directed_cyclic(graph):

    def is_cyclic(graph, node, visited= None, stack = None):
        if visited is None and stack is None:
            visited = {}, stack = {}

        visited[node] = True
        stack[node] = True

        for i in graph[node]:
            if i not in visited and is_cyclic(graph, i, visited, stack):
                return True
            else:
                if stack[node]: 
                    return True

        stack[node] = False
        return False
    

    visited = {}
    stack = {}

    for i in (graph.keys() - visited.keys()):
        gcycle = is_cyclic(graph, i, visited, stack)
        if gcycle:
            return True

    return False    



def dfs_undirected_cycle(graph):


    def is_cyclic(graph, node, parent, visited=None):
        if visited is None:
            visited = {}
        
        visited[node] = True

        for n in graph[node]:
            if n not in visited:
                if (dfs_undirected_cycle(graph, n, node, visited):
                    return True

            else:
                if n != node:
                    return True
    
        return False


class UnionFind(object):

    def __init__(self, size):
        self.size = [1] * size
        self.parent = list(range(size))

    def find(self, node):

        # Find parent
        root = node
        while root != self.parent[root]
                root = self.parent[root]

        # Compress path
        while node != root:
            nextnode = self.parent[node]
            self.parent[node] = root
            node = nextnode

        return root
  

    def connected(self, node1, node2):
        return self.find(node1) == self.find(node2)


    def unify(self, node1, node2):

        root1 = self.find(node1)
        root2 = self.find(node2)
        
        if root1 == root2:
            return
        if self.size[root1] < self.size[root2]:
            self.size[root2] += self.size[root1]
            self.parent[root1] = root2
        else:
            self.size[root1] += self.size[root2]
            self.parent[root2] = root1
        
        
    
def articulation_points(graph):


    def articulation_util(graph, start, visited, disc, low, parent, time):

        if visited is None:
            visited, low, disc, parent = [], {}, {}, {}
        
        disc[start] = time
        low[start] = time
        visited[start] = time
        childrent = 0

        for n in graph[start]:
            if n not in visited:
                visited[n] = True
                parent[n] = start
                children += 1
                articulation_util(graph, n, visited, disc, low, parent, time+1)
                low[start] = min(low[n], low[start])
                
                # Condition where start is root
                if parent[start] == None and children > 1:
                    print(start)

                # Condition when a backedge is found
                if parent[start] is not None and low[n] > disc[vertax]:
                    print(start)

            elif parent[n] != start:
                low[start]  = min(low[n], low[start])

       
    visited = disc = low = parent = {}
    time = 0

    for n in graph.keys():

        if n not in visited:
            parent[n] = None
            articuation_util(graph, n, visited, disc, low, parent, time)
    

    

def dijkstra(graph, start):

    visited = set()
    previous = dict.fromkeys(list(graph.keys()), None)
    delta = dict.fromkeys(list(graph.keys()), math.inf)
    v = set(graph.keys())
    delta[start] = 0
 
    while visited != v:
        
        # closest vertex not yet visited
        node = min(set(delta.keys()) - visited, key=delta.get)

        # all neighbors
        for neighbor in graph[node].keys():
            if node not in visited:
                path = delta[node] + graph[node][neighbor]
                if path < delta[neighbor]:
                    delta[neighbor] = path
                    previous[neighbor] = node
        visited.add(node)
                
    return delta, previous
        
        
            
def main():
    graph = {
        'A': ['B', 'C', 'E'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F', 'G'],
        'D': ['B'],
        'E': ['A', 'B','D'],
        'F': ['C'],
        'G': ['C']
    }

    print(breadth_first(graph, 'B'))
    print(breadth_first_path(graph, 'B', 'G'))

    print("Depth First Search")
    print(depth_first(graph, 'B'))
    print(depth_first_recursive(graph, 'B'))
    print(depth_first_path(graph, 'B', 'G'))

    graph = {
        'A' : {'B':2, 'C':8, 'D':5},
        'B': {'C': 1},
        'C': {'E': 3},
        'D': {'E': 4},
        'E': {}
    }

    print(dijkstra(graph, 'A'))



if __name__ == '__main__':
    main()

