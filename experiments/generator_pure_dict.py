import copy
from drs import drs
import numpy as np
import random
from typing import *
from operator import itemgetter
from collections import defaultdict
import time

# Generates sets of DAG tasks

# number of available processors m \in {4, 8, 16, 32}
# number of tasks \in {[0.25m, m], [0.25m, 2m], [m, 2m]}
# num_nodes \in [10, 100] + 2
# period \in [1, 2, 5, 10, 20, 50, 100, 200, 1000]
# probability of precedence constraint \in {[0.1, 0.3], [0.4, 0.6], [0.7, 0.9]}
# scale is applied to make the generated float execution time to int type
# number of groups \in {[0.25m, m], [0.25m, 2m], [m, 2m]}, ungrouped subtasks can be treated as another group
# the probabilities that a sub-task is assigned to a group: 1) averagely assign; or
# 2) unbalanced assign, where the probabilities are generated in advance

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


"""
Graph Object Wrapper
"""
# build the parallelism table for greedy group strategy
# the common strating node (read) and common ending node (write) are not included into the table
def find_parallelism(graph_org:Graph):
    all_successors = defaultdict(list)
    graph = copy.deepcopy(graph_org)
    num_nodes = graph.V
    parallelism_table = defaultdict(list)
    # find all the successors
    for i in range(num_nodes - 2, 0, -1):
        all_successors[i] = copy.deepcopy(graph.graph[i])
        if num_nodes - 1 in all_successors[i]:
            all_successors[i].remove((num_nodes - 1))
        if len(all_successors[i]) > 0:
            for j in range(len(all_successors[i])):
                all_successors[i].extend(all_successors[all_successors[i][j]])
            # remove all duplicates in the successors list
            all_successors[i] = list(set(all_successors[i]))

    # find all the possible parallelism for each node
    for i in range(1, num_nodes - 1):
        temp_list = set(np.arange(i + 1, num_nodes - 1))
        parallelism_table[i].extend(list(temp_list - set(all_successors[i])))
        if (len(parallelism_table[i]) > 0):
            for j in range(len(parallelism_table[i])):
                if parallelism_table[i][j] > i:
                    parallelism_table[parallelism_table[i][j]].append(i)

    return dict(sorted(parallelism_table.items()))

def find_worst_path(graph: Graph, preemptive=False):
    return _find_worst_path(graph.graph, graph.weights, graph.group, preemptive)


"""
Find the length of a longest path with respect to vol(v) + delay(v) 
under non-preemptive scheduling
"""
def _find_worst_path(graph: Dict[int, List[int]], weights: List[int], group: List[int], preemptive=False):
    if preemptive is False:
        base_cost = [weights[v] + find_delay_vertex_np(graph, weights, v, group) for v in graph.keys()]
    else:
        base_cost = [weights[v] + find_delay_vertex_p(graph, weights, v, group) for v in graph.keys()]
    def dfs(graph, vertex: int, cost: List[int], path: List[List[int]], visited: List[bool]):
        visited[vertex] = True
        for connected_vertex in graph[vertex]:
            if not visited[connected_vertex]:
                dfs(graph, connected_vertex, cost, path, visited)
            if cost[vertex] > base_cost[vertex]+cost[connected_vertex]:
                cost[vertex] = cost[vertex]
            else:
                cost[vertex] = base_cost[vertex]+cost[connected_vertex]
                path[vertex] = [vertex] + path[connected_vertex]

    cost = copy.deepcopy(base_cost)
    visited = [False for v in graph.keys()]
    path = [[v] for v in graph.keys()]

    for vertex in graph.keys():
        if not visited[vertex]:
            dfs(graph, vertex, cost, path, visited)

    max_cost_index, max_cost = max(enumerate(cost), key=lambda item: item[1])
    return max_cost, path[max_cost_index], max_cost-find_volume_vertex_list(graph, weights, path[max_cost_index])

def find_longest_path(graph: Dict[int, List[int]], weights: List[int]):
    base_cost = [weights[v] for v in graph.keys()]
    def dfs(graph, vertex: int, cost: List[int], visited: List[bool]):
        visited[vertex] = True
        for connected_vertex in graph[vertex]:
            if not visited[connected_vertex]:
                dfs(graph, connected_vertex, cost, visited)
            if cost[vertex] > base_cost[vertex]+cost[connected_vertex]:
                cost[vertex] = cost[vertex]
            else:
                cost[vertex] = base_cost[vertex]+cost[connected_vertex]

    cost = copy.deepcopy(base_cost)
    visited = [False for v in graph.keys()]
    for vertex in graph.keys():
        if not visited[vertex]:
            dfs(graph, vertex, cost, visited)

    return max(enumerate(cost), key=lambda item: item[1])

def generate_tsk_dict(msets, num_processors, pc_prob, utilization, sparse, group_mode, group_prob, scale):
    dtype = np.float64
    tasksets = []
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
    pc_q_up = [0.3, 0.6, 0.9]
    pc_q_lo = [0.1, 0.4, 0.7]
    num_tasks = 10
    for i in range(msets):
        taskset = []
        if sparse == 0:
            num_tasks = random.randint(0.25 * num_processors, num_processors)
        if sparse == 1:
            num_tasks = random.randint(num_processors, 2 * num_processors)
        if sparse == 2:
            num_tasks = random.randint(0.25 * num_processors, 2 * num_processors)
        util_tasks = drs(num_tasks, utilization)
        j = 0
        while j < num_tasks:
            # add one common source node and one common end node
            num_nodes_real = random.randint(10, 100)
            num_nodes_all = num_nodes_real + 2
            # print(num_nodes_all)
            # generate the DAG task with graph class
            g_temp = Graph(num_nodes_all)
            period_id = random.randint(0, 8)
            period = periods[period_id]
            g_temp.period = period * scale

            util_nodes = drs(num_nodes_real, util_tasks[j])
            g_temp.utilization = util_tasks[j]

            WCETs_float = [i * period * scale for i in util_nodes]
            WCETs_int = list(map(int, WCETs_float))

            # one common souce node with 0 WCET
            # one common end node with 0 WCET
            WCETs_int.insert(0, 0)
            WCETs_int.append(0)
            # print(WCETs_int)
            g_temp.weights = WCETs_int
            g_temp.priorities = [0] * num_nodes_all

            # G(n, q) method to generate the precedence constraints
            pc_q = random.uniform(pc_q_lo[pc_prob], pc_q_up[pc_prob])
            # define the graph structure
            struct = defaultdict(list)

            for source in range (1, num_nodes_all - 1):
                for dest in range (source+1, num_nodes_all):
                    if random.uniform(0, 1) < pc_q:
                        struct[source].append(dest)

            # check the connection to common source node
            all_list = range(1, num_nodes_all - 1)
            set_all_list = set(all_list)
            for node in range(1, num_nodes_all - 1):
                set_all_list = set_all_list - set(struct[node])

            # print(set_all_list)
            not_connected_source = list(set_all_list)
            if len(not_connected_source) > 0:
                for node in range(len(not_connected_source)):
                    struct[0].append(not_connected_source[node])

            # check the connection to common end node
            for node in range(1, num_nodes_all - 1):
                if len(struct[node]) == 0:
                    struct[node].append(num_nodes_all-1)
                if len(struct[node]) < 1:
                    print('Length3 : ', len(struct[node]))
                    print('something wrong 0!!!')

            # append the last item
            struct[num_nodes_all-1]

            g_temp.graph = copy.deepcopy(dict(sorted(struct.items())))


            # find the parallelism table for the generated graph
            g_temp.parallelism_table = find_parallelism(g_temp)

            g_temp.graph = copy.deepcopy(dict(sorted(struct.items())))

            for node in range(1, num_nodes_all - 1):
                if len(g_temp.graph[node]) < 1:
                    print('something wrong!!!')

            # print(g_temp.graph[0])
            # define the groups
            num_groups_candidate = [[0.25 * num_processors, num_processors], [0.25 * num_processors, 2 * num_processors], [num_processors, 2 * num_processors]]
            num_groups = random.randint(num_groups_candidate[group_mode][0], num_groups_candidate[group_mode][1])
            # add one group for ungrouped nodes
            num_groups = num_groups + 1
            # define the distribution of different groups
            ungroup_prob = [0.1, 0.3, 0.5]
            gp_probabilities = []
            gp_probabilities.append(ungroup_prob[group_prob])
            gp_probabilities.extend(drs(num_groups - 1, 1 - ungroup_prob[group_prob]))

            if abs(sum(gp_probabilities) - 1) > 0.01:
                print('Something Wrong!')

            '''
            if group_prob == 0:
                gp_probabilities = [1/num_groups] * num_groups
            else:
                gp_probabilities = np.random.dirichlet(np.ones(num_groups),size=1)[0]
            '''

            group_info = []
            for i in range(0, num_nodes_all):
                group_info.append(np.random.choice(np.arange(0, num_groups), p=gp_probabilities))

            g_temp.group = group_info
            # the common starting node and the common ending node have the 0 group
            g_temp.group[0] = g_temp.group[-1] = 0

            # check if the longest path is longer than the period already
            # the total execution time is no longer than the period
            if sum(g_temp.weights) <= g_temp.period:
                taskset.append(g_temp)
                j = j + 1
            else:
                # print('critical path has to be checked!')
                longest_path = find_longest_path(g_temp.graph, g_temp.weights)[1]
                if longest_path <= g_temp.period:
                    taskset.append(g_temp)
                    j = j + 1
        # append each set to a utilization specified big set
        tasksets.append(taskset)
    return tasksets

# convert task sets to different group allocation strategies
def tsk_dict_convertor(msets, num_processors, group_mode, group_prob, tsk_set_org):
    tsk_set = copy.deepcopy(tsk_set_org)
    for i in range(msets):
        for j in range(len(tsk_set[i])):

            # print(g_temp.graph[0])
            # define the groups
            num_groups_candidate = [[0.25 * num_processors, num_processors],
                                    [0.25 * num_processors, 2 * num_processors], [num_processors, 2 * num_processors]]
            num_groups = random.randint(num_groups_candidate[group_mode][0], num_groups_candidate[group_mode][1])
            # add one group for ungrouped nodes
            num_groups = num_groups + 1
            # define the distribution of different groups
            ungroup_prob = [0.1, 0.3, 0.5]
            gp_probabilities = []
            gp_probabilities.append(ungroup_prob[group_prob])
            gp_probabilities.extend(drs(num_groups - 1, 1 - ungroup_prob[group_prob]))

            if abs(sum(gp_probabilities) - 1) > 0.01:
                print('Something Wrong!')

            '''
            if group_prob == 0:
                gp_probabilities = [1/num_groups] * num_groups
            else:
                gp_probabilities = np.random.dirichlet(np.ones(num_groups),size=1)[0]
            '''

            group_info = []
            for k in range(int(tsk_set[i][j].V)):
                group_info.append(np.random.choice(np.arange(0, num_groups), p=gp_probabilities))

            group_info[0] = group_info[-1] = 0

            tsk_set[i][j].group = group_info

    return tsk_set

# add communication overheads
def tsk_dict_add_overheads(msets, overheads, tsk_set_org):
    tsk_set = copy.deepcopy(tsk_set_org)
    for i in range(msets):
        for j in range(len(tsk_set[i])):
            for k in range(int(tsk_set[i][j].V)):
                tsk_set[i][j].weights[k] = tsk_set[i][j].weights[k] * overheads

    return tsk_set

# add communication overheads
def tsk_dict_add_overheads_careful(msets, overheads, tsk_set_org):
    tsk_set = copy.deepcopy(tsk_set_org)
    for i in range(msets):
        for j in range(len(tsk_set[i])):
            for k in range(int(tsk_set[i][j].V)):
                if tsk_set[i][j].group[k] > 0:
                    tsk_set[i][j].weights[k] = tsk_set[i][j].weights[k] * overheads

    return tsk_set