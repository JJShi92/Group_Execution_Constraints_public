import copy
from typing import *

from operator import itemgetter
from collections import defaultdict
from collections import deque
import numpy as np
import math
import sys
sys.path.append('../')
from generators import generator_pure_dict as gen

class Graph:
    #def __init__(self, graph: Dict[int, List[int]], weights: List[int], priorities: List[int] = [], group: List[int] = []):
    #    self.graph = graph
    #    self.weights = weights
    #    self.priorities = priorities if priorities else [0 for i in range(len(graph.keys()))]
    #    self.group = group if group else [0 for i in range(len(graph.keys()))]

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = defaultdict(list)
        self.weights = List[int]
        self.priorities = List[int]
        # self.priorities = priorities if priorities else [0 for i in range(len(graph.keys()))]
        self.group = List[int]
        self.utilization = float
        self.period = int
        self.deadline = int
        self.paths = List[int]
        self.cp = int # length of the critical path
        self.parallelism_table = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def __getitem__(self, item):
        return self.graph.__getitem__(item)

    def __setitem__(self, key, value):
        self.graph.__setitem__(key, value)

    # find the length of a give path
    def path_length(self, path):
        length = 0
        for i in range(len(path)):
            length = length + self.weights[path[i]]

        return length
    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''

    def printAllPathsUtil(self, u, d, visited, path):

        # Mark the current node as visited and store in path
        visited[u] = True
        path.append(u)

        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            print('Path length verification @ ', time.time())
            cp_length = self.path_length(path)

            if cp_length > self.period:
                self.cp = cp_length
                return

            if cp_length > self.cp:
                self.cp = cp_length
                self.paths = path
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                if visited[i] == False:
                    self.printAllPathsUtil(i, d, visited, path)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u] = False

    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d):

        # Mark all the vertices as not visited
        visited = [False] * (self.V)

        # Create an array to store paths
        path = []
        self.paths = []
        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path)

    # build the parallelism table for greedy group strategy
    # the common strating node (read) and common ending node (write) are not included into the table
    def find_parallelism(self):
        all_successors = defaultdict(list)
        # find all the successors
        for i in range(self.V - 2, 0, -1):
            all_successors[i] = self.graph[i]
            if self.V-1 in all_successors[i]:
                all_successors[i].remove((self.V-1))
            if len(all_successors[i]) > 0:
                for j in range(len(all_successors[i])):
                    all_successors[i].extend(all_successors[all_successors[i][j]])
                    # remove all duplicates in the successors list
                    all_successors[i] = list(set(all_successors[i]))

        # find all the possible parallelism for each node
        for i in range(1, self.V-1):
            temp_list = set(np.arange(i+1, self.V-1))
            self.parallelism_table[i].extend(list(temp_list - set(all_successors[i])))
            if (len(self.parallelism_table[i]) > 0):
                for j in range(len(self.parallelism_table[i])):
                    if self.parallelism_table[i][j] > i:
                     self.parallelism_table[self.parallelism_table[i][j]].append(i)

        self.parallelism_table = dict(sorted(self.parallelism_table.items()))
'''
class DAGTask:
    def __init__(self, graph: Graph, period: int, deadline: int):
        self.graph = graph
        self.period = period
        self.deadline = deadline
'''

def remove_nodes(graph: Dict[int, List[int]], vertices: List[int]):
    for vertex, adjacent in graph.copy().items():
        if vertex in vertices:
            graph.pop(vertex)
        else:
            for vertex in adjacent:
                if vertex in vertices:
                    adjacent.remove(vertex)


def find_predecessors(graph: Dict[int, List[int]], vertex: int) -> List[int]:
    precs = []
    for key in graph:
        if vertex in graph[key]:
            precs.append(key)
    return precs


def find_successors(graph: Dict[int, List[int]], vertex: int) -> List[int]:
    if vertex in graph:
        return graph[vertex]
    else:
        return []


def find_ancestors(graph: Dict[int, List[int]], vertex: int, path: List[int] = []) -> List[int]:
    graph_copy = copy.deepcopy(graph)
    return find_ancestors_iter(graph_copy, vertex, path=[])


def find_ancestors_iter(graph: Dict[int, List[int]], vertex: int, path: List[int] = []):
    if vertex == 0:
        return []

    predecessors = find_predecessors(graph, vertex)
    if not predecessors:
        return []
    remove_nodes(graph, [vertex])

    for vertex in predecessors:
        path.append(vertex)
        find_ancestors_iter(graph, vertex, path)

    return list(set(path))


def find_descendants(graph: Dict[int, List[int]], vertex: int, path: List[int] = []):
    graph_copy = copy.deepcopy(graph)
    return find_descendants_iter(graph_copy, vertex, path=[])


def find_descendants_iter(graph: Dict[int, List[int]], vertex: int, path: List[int] = []):

    if vertex == max(graph.keys()):
        return []

    successors = find_successors(graph, vertex)
    if not successors:
        return []

    remove_nodes(graph, [vertex])

    for vertex in successors:
        path.append(vertex)
        find_descendants_iter(graph, vertex, path)

    return list(set(path))


def volume(vertex_set: List[int], wcet: List[int]):
    return sum(wcet[vertex] for vertex in vertex_set)


def delta_set_vertex(graph: Dict[int, List[int]], vertex: int, groups: List[int]) -> Set[int]:
    int_set_vertex = set()
    ancestors = find_ancestors(graph, vertex)
    descendants = find_descendants(graph, vertex)
    for u in graph.keys():
        if u is not vertex:
            if u not in ancestors and u not in descendants:
                if groups[u] == groups[vertex] and groups[u] > 0:
                    int_set_vertex.add(u)
    return int_set_vertex


def longest_measure(dag: Graph):
    groups = copy.deepcopy(dag.group)
    delta_path = [[set() for vertex in dag.graph.keys()] for vertex in dag.graph.keys()]
    delta_sets = [delta_set_vertex(dag.graph, vertex, groups) for vertex in dag.graph.keys()]
    num_nodes = max(dag.graph.keys()) + 1

    cost = np.zeros((num_nodes, num_nodes))
    changed = True
    for k in range(num_nodes):
        if not changed:
            return cost[0, max(dag.graph.keys())]
        changed = False
        for i in dag.graph.keys():
            if k == 0:
                changed = True
                for j in range(num_nodes):
                    if j in dag.graph[i]:
                        delta_path[i][j] = delta_sets[i].union(delta_sets[j])
                        cost[i, j] = dag.weights[i] + dag.weights[j] + sum(dag.weights[v] for v in delta_path[i][j])
                    else:
                        cost[i, j] = -np.inf
            else:
                for j in dag.graph.keys():
                    for u in dag.graph.keys():
                        intersection = delta_path[i][u].intersection(delta_path[u][j])
                        max_cost = np.maximum(cost[i, j], cost[i, u]+cost[u, j]-dag.weights[u]-sum(dag.weights[v] for v in intersection))
                        if max_cost > cost[i, j]:
                            changed = True
                            cost[i, j] = max_cost
                            delta_path[i][j] = delta_path[i][u].union(delta_path[u][j])
    return cost[0, max(dag.graph.keys())]


def makespan(dag: Graph, num_cpu: int):
    return sum(dag.weights[v] for v in dag.graph.keys())/num_cpu + ((num_cpu-1)/num_cpu)*longest_measure(dag)



def main():

    g = Graph(10)
    g.graph = {0: [1, 4], 1: [2, 3], 2: [5], 3: [5], 4: [5], 5: [6], 6: [7, 8], 7: [9], 8: [9], 9: []}
    g.weights = [0, 3, 1, 2, 1, 1, 2, 5, 2, 0]
    g.priorities = [0, 0, 0, 0, 0, 0]
    g.group = [0, 1, 2, 1, 2, 0, 2, 2, 0, 0]
    g.period = 50
    print(makespan(g, 3))

if __name__ == "__main__":
    main()

