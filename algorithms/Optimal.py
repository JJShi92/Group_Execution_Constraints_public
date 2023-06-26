import sys
sys.path.append('../')
from algorithms.graph import *


def makespan(graph: Dict[int, List[int]], weights: List[int], num_cpu: int) -> float:
    path, leng = find_longest_path(graph, 0, max(graph.keys()), weights)
    vol = sum(weights[vertex] for vertex in graph.keys())
    return max(vol/num_cpu, leng)


def makespan_lower_bound(graph: Graph, num_cpu: int) -> float:
    _, leng = find_longest_path(graph.graph, graph.weights)
    return max(sum(graph.weights[vertex] for vertex in graph.graph.keys())/num_cpu, leng)

# Calculate the makespan for all the given DAG tasks
def calculate_makespan_all(taskset_org, available_processors_org):
    taskset = copy.deepcopy(taskset_org)
    available_processors = copy.deepcopy(available_processors_org)
    makespan_allset = []
    for i in range(len(taskset)):
        makespan_oneset = []
        for j in range(len(taskset[i])):
            task = copy.deepcopy(taskset[i][j])
            makespan_onetask = makespan_lower_bound(task, available_processors)

            makespan_oneset.append(makespan_onetask)

        makespan_allset.append(makespan_oneset)

    return makespan_allset

def main():
    g = Graph(10)
    g.graph = {0: [1, 4], 1: [2, 3], 2: [5], 3: [5], 4: [5], 5: [6], 6: [7, 8], 7: [9], 8: [9], 9: []}
    g.weights = [1, 3, 1, 2, 1, 1, 2, 5, 2, 1]
    g.priorities = [0, 0, 0, 0, 0, 0]
    g.group = [0, 1, 2, 1, 3, 0, 2, 3, 0, 0]
    g.period = 50

    print(makespan_lower_bound(g, 2))

if __name__ == "__main__":
    main()