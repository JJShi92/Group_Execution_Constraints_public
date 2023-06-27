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


"""
Very expensive => only use for testing purposes for small instances
"""
def find_all_paths(graph: Dict[int, List[int]], start_vertex: int, end_vertex: int, path: List[int] = []):
    path = path + [start_vertex]
    if start_vertex == end_vertex:
        return [path]

    if start_vertex not in graph:
        return []

    paths = []
    for vertex in graph[start_vertex]:
        if vertex not in path:
            extended_paths = find_all_paths(graph, vertex, end_vertex, path)
            for p in extended_paths:
                paths.append(p)

    return paths


"""
Graph Object Wrapper
"""
def find_worst_path(graph: Graph, preemptive=False):
    return _find_worst_path(graph.graph, graph.weights, graph.group, preemptive)


"""
Find the length of a longest path with respect to vol(v)
"""

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


"""
longest path length + longest path
"""
def find_longest_path_with_path(graph: Dict[int, List[int]], weights: List[int]) -> Tuple[List[int], int]:
    base_cost = [weights[v] for v in graph.keys()]
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
    return path[max_cost_index], max_cost

def find_volume_graph(graph: Dict[int, List[int]], weights: List[int]):
    return sum(weights[vertex] for vertex in graph.keys())

def find_volume_vertex_list(graph: Dict[int, List[int]], weights: List[int], vertex_list: List[int]):
    return sum(weights[vertex] for vertex in vertex_list)

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


"""
 Finds a list of vertexes from the 'graph' that can run in parallel with 'vertex'
"""
def find_parallel_vertex(graph: Dict[int, List[int]], vertex: int, path: List[int] = []) -> List[int]:
    desc, ance = find_descendants(graph, vertex, path), find_ancestors(graph, vertex, path)
    parallel_vertex = []
    for v in graph.keys():
        if v == vertex:
            continue
        if v in desc:
            continue
        if v in ance:
            continue
        parallel_vertex.append(v)
    return parallel_vertex

"""
 Finds the delay of 'vertex' with respect to 'group' 
 group[4] = 6 implies that vertex 4 is in group 6

 Use group[4] = 0 to denote that vertex 4 has no group constraint
"""
def find_delay_vertex_p(graph: Dict[int, List[int]], weights: List[int], vertex: int, group: List[int]) -> float:
    if group[vertex] == 0:
        return 0
    return sum(weights[v] for v in find_parallel_vertex(graph, vertex) if group[v] == group[vertex])


def find_delay_vertex_np(graph: Dict[int, List[int]], weights: List[int], vertex: int, group: List[int]) -> float:
    if group[vertex] == 0:
        return 0
    parallel_vertexes = find_parallel_vertex(graph, vertex)
    return sum(weights[v] for v in parallel_vertexes if group[v] == group[vertex]) + max((weights[v] for v in parallel_vertexes if group[v] == 0), default=0)


# the group merge strategy 0: average the parallelism
def merge_utilization(graph: Graph, available_processors):
    num_groups = max(graph.group)
    if num_groups <= available_processors:
        return graph
    else:
        group_util = deque()
        for i in range(1, num_groups + 1):
            group_util.append([i, 0])
        for i in range(len(graph.group)):
            if graph.group[i] > 0:
                group_util[graph.group[i]-1][1] = group_util[graph.group[i]-1][1] + graph.weights[i]/graph.period
        # sort the group utilizations decreasingly
        group_util = deque(sorted(group_util, reverse=True, key=lambda x: x[1]))

        # initialize the partition table
        partitioned = []
        partitioned_util = []
        for i in range(available_processors):
            partitioned.append([])
            partitioned_util.append(0)

        for i in range(num_groups):
            # select the processor with the least utilization
            next_pid = partitioned_util.index(min(partitioned_util))
            partitioned_util[next_pid] = partitioned_util[next_pid] + group_util[i][1]
            partitioned[next_pid].append(group_util[i][0])

        # update the group allocations
        new_group = copy.deepcopy(graph.group)
        for i in range(available_processors):
            for j in range(len(partitioned[i])):
                for k in range(len(graph.group)):
                    if graph.group[k] == partitioned[i][j]:
                        new_group[k] = i + 1

    return new_group


# the group merge strategy 1: greedy approach by fully utilizing the parallelism
def merge_greedy_table(graph: Graph):
    grp_look_up = dict()
    for vertex_id, group in enumerate(graph.group):
        if group not in grp_look_up:
            grp_look_up[group] = set()
        grp_look_up[group].add(vertex_id)

    #initialize and setup the look-up table for argmin
    cost = np.zeros((len(grp_look_up.keys()), len(grp_look_up.keys())))
    np.fill_diagonal(cost, np.inf)
    cost[0,:] = np.iinfo(np.int32).max
    cost[:,0] = np.iinfo(np.int32).max
    for group_id in grp_look_up.keys():
        if group_id == 0:
            continue
        for vertex_id in grp_look_up[group_id]:
            for parallel_vertex_id in graph.parallelism_table[vertex_id]:
                cost[group_id, graph.group[parallel_vertex_id]] += graph.weights[vertex_id]
                cost[graph.group[parallel_vertex_id], group_id] += graph.weights[vertex_id]

    all_groups = []
    while len(grp_look_up.keys()) > 2:
        all_groups.append(grp_look_up.copy())
        # make a grouping decision based on minimal cost
        group_id_base, group_id_merged = np.unravel_index(cost.argmin(), cost.shape)
        #print("Merge group %d with group %d" % (group_id_base, group_id_merged))
        grp_look_up[group_id_base].update(grp_look_up[group_id_merged])
        grp_look_up.pop(group_id_merged, None)
        # update cost matrix
        for vertex in (u for u in grp_look_up.keys() if u != 0 or u != group_id_merged):
            cost[group_id_base, vertex] = cost[group_id_base, vertex] + cost[group_id_merged, vertex]
            cost[vertex, group_id_base] = cost[vertex, group_id_base] + cost[vertex, group_id_merged]
        # remove merged group from candidate status
        cost[group_id_merged,:] = np.iinfo(np.int32).max
        cost[:, group_id_merged] = np.iinfo(np.int32).max

    all_groups.append(grp_look_up.copy())
    return all_groups

def merge_greedy_convert(modified_group, length):
    group_list = [0] * length
    for i in range(len(modified_group)):
        gp_list = list(modified_group[i])
        for j in range(len(gp_list)):
            group_list[gp_list[j]] = i

    return group_list

def merge_greedy(graph: Graph, max_num_groups: int):
    if len(graph.parallelism_table) < len(graph.weights)-2:
        graph.parallelism_table[len(graph.weights)-2] = []
    grp_look_up = dict()
    for vertex_id, group in enumerate(graph.group):
        if group not in grp_look_up:
            grp_look_up[group] = set()
        grp_look_up[group].add(vertex_id)

    groups = [0 for i in range(graph.V)]
    for new_key, key in enumerate(sorted(grp_look_up)):
        grp_look_up[new_key] = grp_look_up.pop(key, None)
        for vertex_id in grp_look_up[new_key]:
            groups[vertex_id] = new_key

    cost = np.zeros((len(grp_look_up.keys()), len(grp_look_up.keys())))
    np.fill_diagonal(cost, np.inf)
    cost[0, :] = np.inf
    cost[:, 0] = np.inf
    for group_id in grp_look_up.keys():
        if group_id == 0:
            continue
        for vertex_id in grp_look_up[group_id]:
            for parallel_vertex_id in graph.parallelism_table[vertex_id]:
                cost[group_id, groups[parallel_vertex_id]] += graph.weights[vertex_id]
                cost[groups[parallel_vertex_id], group_id] += graph.weights[vertex_id]

    while len(grp_look_up.keys()) - 1 > max_num_groups:

        # make a grouping decision based on minimal cost
        group_id_base, group_id_merged = np.unravel_index(cost.argmin(), cost.shape)
        # print("Merge group %d with group %d" % (group_id_base, group_id_merged))
        grp_look_up[group_id_base].update(grp_look_up[group_id_merged])
        grp_look_up.pop(group_id_merged, None)
        # update cost matrix
        for vertex in (u for u in grp_look_up.keys() if u != 0 or u != group_id_merged):
            cost[group_id_base, vertex] = cost[group_id_base, vertex] + cost[group_id_merged, vertex]
            cost[vertex, group_id_base] = cost[vertex, group_id_base] + cost[vertex, group_id_merged]
        # remove merged group from candidate status
        cost[group_id_merged, :] = np.inf
        cost[:, group_id_merged] = np.inf

    for key in grp_look_up:
        for vertex_id in grp_look_up[key]:
            groups[vertex_id] = key

    return groups

# when only one processor is available
def merge_one(graph:Graph):
    new_group = []
    for i in range(len(graph.group)):
        if graph.group[i] > 0:
            new_group.append(1)
        else:
            new_group.append(0)

    return new_group

# make span for a given graph and the number of available processors
def makespan(graph: Graph, parallelism: int, preemptive) -> float:
    # check the situation that the number of groups is larger than the number of available processors/parallelism
    #print(graph.graph)
    leng, path, delay = find_worst_path(graph, preemptive)
    vertex_set_graph = set(graph.graph.keys())
    path_vertex_set = set(path)
    interference = find_volume_vertex_list(graph.graph, graph.weights, list(vertex_set_graph-path_vertex_set))
    return leng + (float(interference)-delay)/parallelism

# make span for gang reservation
def makespan_gang(graph: Graph, parallelism: int, merge_strategy) -> float:
    # check the situation that the number of groups is larger than the number of available processors/parallelism
    if max(graph.group) > parallelism:
        if parallelism == 1:
            graph.group = merge_one(graph)
        else:
            if merge_strategy == 0:
                graph.group = merge_utilization(graph, parallelism)
            else:
                graph.group = merge_greedy(graph, parallelism)
    # makespan preemptive or non preemptive
    mkspan = []
    g_temp = copy.deepcopy(graph)
    mkspan.append(makespan(g_temp, parallelism, False))
    mkspan.append(makespan(graph, parallelism, True))

    return mkspan

# Gang reservation design
def gang_reservation_single(graph_org: Graph, maximum_processor, merge_strategy):
    graph = copy.deepcopy(graph_org)
    minimal_processor = math.ceil(sum(graph.weights)/graph.period)
    processor_makespan_np = []
    processor_makespan_p = []
    wasted_res_np = []
    wasted_res_p = []
    for i in range(minimal_processor, maximum_processor+1):
        graph = copy.deepcopy(graph_org)
        makespan_p = makespan_gang(graph, i, merge_strategy)
        processor_makespan_np.append(makespan_p[0])
        processor_makespan_p.append(makespan_p[1])

        wasted_res_np.append(i*makespan_p[0] - sum(graph.weights))
        wasted_res_p.append(i * makespan_p[1] - sum(graph.weights))

    # find the num_processor with minimal wasted resource
    # the index with the minimal value in a list
    p_id_np = wasted_res_np.index(min(wasted_res_np))
    p_id_p = wasted_res_p.index(min(wasted_res_p))
    # store the info [number of reserved processor, corresponding makespan]
    gang_reservation_np = [minimal_processor+p_id_np, processor_makespan_np[p_id_np]]
    gang_reservation_p = [minimal_processor + p_id_p, processor_makespan_p[p_id_p]]

    gang_reservation = []
    gang_reservation.append(gang_reservation_np)
    gang_reservation.append(gang_reservation_p)

    return gang_reservation

# Calculate the makespan for all the given DAG tasks
def calculate_makespan_all(taskset_org, available_processors_org):
    taskset = copy.deepcopy(taskset_org)
    available_processors = copy.deepcopy(available_processors_org)
    makespan_allset = []
    for i in range(len(taskset)):
        makespan_oneset = []
        for j in range(len(taskset[i])):
            makespan_onetask = []
            if max(taskset[i][j].group) <= available_processors:
                task = copy.deepcopy(taskset[i][j])
                mk_0 = mk_2 = makespan(task, available_processors, False)
                task = copy.deepcopy(taskset[i][j])
                mk_1 = mk_3 = makespan(task, available_processors, True)
            else:
                task = copy.deepcopy(taskset[i][j])
                # print(task.parallelism_table)
                if available_processors_org == 1:
                    group_temp_0 = group_temp_1 = merge_one(task)
                else:
                    group_temp_0 = merge_utilization(task, available_processors)
                    task = copy.deepcopy(taskset[i][j])
                    group_temp_1 = merge_greedy(task, available_processors)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_0
                mk_0 = makespan(task, available_processors, False)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_0
                mk_1 = makespan(task, available_processors, True)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_1
                mk_2 = makespan(task, available_processors, False)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_1
                mk_3 = makespan(task, available_processors, True)
            makespan_onetask.append(mk_0)
            makespan_onetask.append(mk_1)
            makespan_onetask.append(mk_2)
            makespan_onetask.append(mk_3)
            print(makespan_onetask)
            makespan_oneset.append(makespan_onetask)

        makespan_allset.append(makespan_oneset)

    return makespan_allset

# Calculate the makespan for all the given DAG tasks
def calculate_makespan_all_ungroupped(taskset_org, available_processors_org):
    taskset = copy.deepcopy(taskset_org)
    available_processors = copy.deepcopy(available_processors_org)
    makespan_allset = []
    for i in range(len(taskset)):
        makespan_oneset = []
        for j in range(len(taskset[i])):
            makespan_onetask = []
            taskset[i][j].group = [0] * taskset[i][j].V
            if max(taskset[i][j].group) <= available_processors:
                task = copy.deepcopy(taskset[i][j])
                mk_0 = mk_2 = makespan(task, available_processors, False)
                task = copy.deepcopy(taskset[i][j])
                mk_1 = mk_3 = makespan(task, available_processors, True)
            else:
                task = copy.deepcopy(taskset[i][j])
                # print(task.parallelism_table)
                if available_processors_org == 1:
                    group_temp_0 = group_temp_1 = merge_one(task)
                else:
                    group_temp_0 = merge_utilization(task, available_processors)
                    task = copy.deepcopy(taskset[i][j])
                    group_temp_1 = merge_greedy(task, available_processors)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_0
                mk_0 = makespan(task, available_processors, False)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_0
                mk_1 = makespan(task, available_processors, True)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_1
                mk_2 = makespan(task, available_processors, False)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_1
                mk_3 = makespan(task, available_processors, True)
            makespan_onetask.append(mk_0)
            makespan_onetask.append(mk_1)
            makespan_onetask.append(mk_2)
            makespan_onetask.append(mk_3)
            print(makespan_onetask)
            makespan_oneset.append(makespan_onetask)

        makespan_allset.append(makespan_oneset)

    return makespan_allset

# Calculate the makespan for all the given DAG tasks
def calculate_makespan_all_groupped(taskset_org, available_processors_org):
    taskset = copy.deepcopy(taskset_org)
    available_processors = copy.deepcopy(available_processors_org)
    makespan_allset = []
    for i in range(len(taskset)):
        makespan_oneset = []
        for j in range(len(taskset[i])):
            makespan_onetask = []
            taskset[i][j].group = list(range(0, taskset[i][j].V))
            taskset[i][j].group[-1] = 0
            if max(taskset[i][j].group) <= available_processors:
                task = copy.deepcopy(taskset[i][j])
                mk_0 = mk_2 = makespan(task, available_processors, False)
                task = copy.deepcopy(taskset[i][j])
                mk_1 = mk_3 = makespan(task, available_processors, True)
            else:
                task = copy.deepcopy(taskset[i][j])
                # print(task.parallelism_table)
                if available_processors_org == 1:
                    group_temp_0 = group_temp_1 = merge_one(task)
                else:
                    group_temp_0 = merge_utilization(task, available_processors)
                    task = copy.deepcopy(taskset[i][j])
                    group_temp_1 = merge_greedy(task, available_processors)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_0
                mk_0 = makespan(task, available_processors, False)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_0
                mk_1 = makespan(task, available_processors, True)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_1
                mk_2 = makespan(task, available_processors, False)
                task = copy.deepcopy(taskset[i][j])
                task.group = group_temp_1
                mk_3 = makespan(task, available_processors, True)
            makespan_onetask.append(mk_0)
            makespan_onetask.append(mk_1)
            makespan_onetask.append(mk_2)
            makespan_onetask.append(mk_3)
            print(makespan_onetask)
            makespan_oneset.append(makespan_onetask)

        makespan_allset.append(makespan_oneset)

    return makespan_allset

# Calculate the optimized Gang reservation for all the given DAG tasks
def calculate_gang_reservation_all(taskset_org, max_available_processors_org):
    taskset = copy.deepcopy(taskset_org)
    available_processors = copy.deepcopy(max_available_processors_org)
    gang_reservation_allset = []
    for i in range(len(taskset)):
        gang_reservation_oneset = []
        for j in range(len(taskset[i])):
            gang_reservation_onetask = []
            task = copy.deepcopy(taskset[i][j])
            mk_0 = gang_reservation_single(task, available_processors, 0)
            task = copy.deepcopy(taskset[i][j])
            mk_1 = gang_reservation_single(task, available_processors, 1)
            gang_reservation_onetask.append(mk_0[0])
            gang_reservation_onetask.append(mk_0[1])
            gang_reservation_onetask.append(mk_1[0])
            gang_reservation_onetask.append(mk_1[1])
            gang_reservation_onetask.append(task.period)
            print(gang_reservation_onetask)
            gang_reservation_oneset.append(gang_reservation_onetask)

        gang_reservation_allset.append(gang_reservation_oneset)

    return gang_reservation_allset

def calculate_longest_path_all(taskset_org):
    taskset = copy.deepcopy(taskset_org)
    makespan_allset = []
    for i in range(len(taskset)):
        makespan_oneset = []
        for j in range(len(taskset[i])):
            task = copy.deepcopy(taskset[i][j])
            makespan_onetask = find_longest_path(task.graph, task.weights)[1]
            print(makespan_onetask)

            makespan_oneset.append(makespan_onetask)

        makespan_allset.append(makespan_oneset)

    return makespan_allset

def calculate_wcet_all(taskset_org):
    taskset = copy.deepcopy(taskset_org)
    makespan_allset = []
    for i in range(len(taskset)):
        makespan_oneset = []
        for j in range(len(taskset[i])):
            task = copy.deepcopy(taskset[i][j])
            makespan_onetask = sum(task.weights)
            print(makespan_onetask)

            makespan_oneset.append(makespan_onetask)

        makespan_allset.append(makespan_oneset)

    return makespan_allset

def main():
    #graph = Graph({0: [1,2,3], 1: [4], 2: [4], 3: [5], 4: [5], 5: []}, [0, 8, 3, 6, 1, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 2, 0])
    # node ids are +1, i.e., 2 here is 3 in the figure without groups
    """
    tsk_all = np.load('../experiments/inputs/tasksets_m100_p8_u0.05_q1_s0_g0_r0.npy', allow_pickle=True)

    calculate_longest_path_all(tsk_all)

    # mkspan_all = calculate_makespan_all(tsk_all, 16)
    g_temp = tsk_all[0][0]
    print(g_temp.parallelism_table)
    print(merge_greedy(g_temp, 5))
    print(g_temp.parallelism_table)
    calculate_gang_reservation_all(g_temp, 12)
    # print(makespan(g_temp, 16, False, 1))
    return
    """

    g = Graph(10)
    g.graph = {0: [1,4], 1: [2,3], 2: [5], 3: [5], 4: [5], 5: [6], 6: [7,8], 7: [9], 8: [9], 9: []}
    g.weights = [0, 3, 1, 2, 1, 1, 2, 5, 2, 0]
    g.priorities = [0, 0, 0, 0, 0, 0]
    g.group = [0, 1, 2, 1, 3, 0, 2, 3, 0, 0]
    g.period = 50
    #print(makespan(g, 3, True))
    #print(gen.find_parallelism(g))
    #print(merge_greedy(g, 3))
    #print(makespan(g, 4, False, 1))
    #print(g.parallelism_table)
    print(find_longest_path(g.graph, g.weights))
    # print(max(g.group))
    # merge_utilization(g, 2)
    # print(g.group)
    # print(makespan(g, 10, False, 0))



    #for vertex in g.graph.keys():
    #    print(vertex+1, [x+1 for x in find_parallel_vertex(g.graph, vertex)])


if __name__ == "__main__":
    main()
