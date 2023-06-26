import numpy as np
import sys
sys.path.append('../')
from algorithms.graph import *


def minimal_feasible_ordinary(graph: Graph, period: float, deadline: float, max_cpu: int):
    """

    :rtype: object
    """
    volume = sum(graph.weights[vertex] for vertex in graph.graph.keys())
    weights = graph.weights.copy()
    covered = []
    for n in range(max_cpu):
        path_max, path_max_length = find_longest_path_with_path(graph.graph, weights)
        for vertex in path_max:
            weights[vertex] = 0 # set collected
        if n > 0:
            covered.append(covered[-1] + path_max_length)
        else:
            covered.append(path_max_length)
    min_sum_service_m, min_sum_service_n, min_sum_service = 0, 0, np.inf
    # choose the best one
    for m in range(max_cpu):
        for n in range(m+1):
            sum_budget = ((m-n+1) * covered[0]) + (n * deadline) + volume - covered[n]
            if sum_budget/(m+1) > deadline:
                continue
            if sum_budget < min_sum_service:
                min_sum_service_m = m+1
                min_sum_service_n = (n+1)
                min_sum_service = sum_budget
    return min_sum_service_m, min_sum_service_n, min_sum_service


def enumerate_all_feasible_ordinary(graph: Graph, period: float, deadline: float, max_cpu: int):
    volume = sum(graph.weights[vertex] for vertex in graph.graph.keys())
    weights = graph.weights.copy()
    covered = []
    for n in range(max_cpu):
        path_max, path_max_length = find_longest_path_with_path(graph.graph, weights)
        for vertex in path_max:
            weights[vertex] = 0 # set collected
        if n > 0:
            covered.append(covered[-1] + path_max_length)
        else:
            covered.append(path_max_length)
    feasible = [] # (num_res, num_path, sum_budget)
    for m in range(max_cpu):
        for n in range(m+1):
            sum_budget = ((m-n+1) * covered[0]) + (n * deadline) + volume - covered[n]
            if sum_budget/(m+1) > deadline:
                continue
            feasible.append((m+1, n+1, sum_budget))
    return feasible


def minimal_feasible_gang(graph: Graph, period: float, deadline: float, max_cpu: int):
    volume = sum(graph.weights[vertex] for vertex in graph.graph.keys())
    weights = graph.weights.copy()
    covered = []
    for n in range(max_cpu):
        path_max, path_max_length = find_longest_path_with_path(graph.graph, weights)
        for vertex in path_max:
            weights[vertex] = 0 # set collected
        if n > 0:
            covered.append(covered[-1] + path_max_length)
        else:
            covered.append(path_max_length)
    min_waste_n,  min_waste_budget, min_waste = 0, 0, np.inf
    for n in range(max_cpu):
        gang_budget = covered[0] + (volume-covered[n])/(max_cpu-n)
        if gang_budget > deadline:
            continue
        else:
            waste = (n+1) * gang_budget-volume
            if waste < min_waste:
                min_waste_budget = gang_budget
                min_waste = waste
                min_waste_n = n+1
    return min_waste_n, min_waste_budget, min_waste


def enumerate_all_feasible_gang(graph: Graph, period: float, deadline: float, max_cpu: int):
    volume = sum(graph.weights[vertex] for vertex in graph.graph.keys())
    weights = graph.weights.copy()
    covered = []
    for n in range(max_cpu):
        path_max, path_max_length = find_longest_path_with_path(graph.graph, weights)
        for vertex in path_max:
            weights[vertex] = 0 # set collected
        if n > 0:
            covered.append(covered[-1] + path_max_length)
        else:
            covered.append(path_max_length)

    feasible = [] # (num_res, num_path, sum_budget)
    for m in range(max_cpu):
        for n in range(m+1):
            gang_budget = covered[0] + (volume-covered[n])/(max_cpu-n)
            if gang_budget > deadline:
                continue
            feasible.append((m+1, n+1, gang_budget))
    return feasible


def makespan_uet_rtss(graph: Graph, max_cpu: int):
    volume = sum(graph.weights[vertex] for vertex in graph.graph.keys())
    weights = graph.weights.copy()
    covered = []
    for n in range(max_cpu):
        path_max, path_max_length = find_longest_path_with_path(graph.graph, weights)
        for vertex in path_max:
            weights[vertex] = 0 # set collected
        if n > 0:
            covered.append(covered[-1] + path_max_length)
        else:
            covered.append(path_max_length)
    min_makespan_n, min_makespan = 0, np.inf
    for n in range(max_cpu):
        makespan_n, makespan = n+1, covered[0] + ((volume-covered[n])/(max_cpu-n))
        if makespan < min_makespan:
            min_makespan_n = n+1
            min_makespan = makespan
    return min_makespan_n, min_makespan

# Calculate the makespan for all the given DAG tasks
def calculate_makespan_all(taskset_org, available_processors_org):
    taskset = copy.deepcopy(taskset_org)
    available_processors = copy.deepcopy(available_processors_org)
    makespan_allset = []
    for i in range(len(taskset)):
        makespan_oneset = []
        for j in range(len(taskset[i])):
            task = copy.deepcopy(taskset[i][j])
            makespan_onetask = makespan_uet_rtss(task, available_processors)

            makespan_oneset.append(makespan_onetask)

        makespan_allset.append(makespan_oneset)

    return makespan_allset

def main():
    # graph = Graph({0: [1,2,3], 1: [4], 2: [4], 3: [5], 4: [5], 5: []}, [0, 8, 3, 6, 1, 0], [0, 0, 0, 0, 0, 0])
    # print(graph.graph)

    g = Graph(10)
    g.graph = {0: [1, 4], 1: [2, 3], 2: [5], 3: [5], 4: [5], 5: [6], 6: [7, 8], 7: [9], 8: [9], 9: []}
    g.weights = [1, 3, 1, 2, 1, 1, 2, 5, 2, 1]
    g.priorities = [0, 0, 0, 0, 0, 0]
    g.group = [0, 1, 2, 1, 3, 0, 2, 3, 0, 0]
    g.period = 50

    print(makespan_uet_rtss(g, 2))

if __name__ == "__main__":
    main()
