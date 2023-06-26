import sys
sys.path.append('../')
from algorithms.graph import *


class Alpha(tuple):
    def __new__(self, u, v, r):
        return tuple.__new__(Alpha, (u, v, r))

    def __ge__(self, other):
        if self[0] == other[0] and self[1] == other[1]:
            return self[2] >= other[2]
        return False

    def __le__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        if self[0] == other[0] and self[1] == other[1]:
            return self[2] == other[2]
        return False


def interference_set_path(graph: Dict[int, List[int]], path: List[int], priorities: List[int]) -> List[int]:
    int_set_path = set()
    for vertex in path:
        int_set_path.update(set(interference_set_vertex(graph, vertex, priorities)))
    return list(int_set_path)


def leng(path: List[int], wcet: List[int]):
    return sum(wcet[vertex] for vertex in path)


def response_path(graph: Dict[int, List[int]], priorities: List[int], path: List[int], wcets: List[int], m: int):
    return leng(path, wcets) + (leng(interference_set_path(graph, path, priorities), wcets)/m)


def exhaustive_bound(graph: Dict[int, List[int]], priorities: List[int], wcets: List[int], m: int):
    return max(response_path(graph, priorities, path, wcets, m) for path in find_all_paths(graph, 0, max(graph.keys())))


def longest_path_priority_assignment(graph: Dict[int, List[int]], wcets: List[int]):
    """
    Requires that source and sink nodes are given by id 0 and max(graph.keys())
    respectively.
    Computes all paths longest paths and then
    determines the length of the longest path that any vertex lies on.
    """
    num_nodes = max(graph.keys())+1
    cost = np.zeros((num_nodes, num_nodes))
    for k in graph.keys():
        for i in graph.keys():
            if k == 0:
                cost[i, :] = [wcets[i] + wcets[idx] if idx in graph[i] else -np.inf for idx in graph.keys()]
            else:
                cost[i, :] = np.maximum(cost[i, :], cost[i, k] + cost[k, :] - wcets[k])

    table = dict()
    for vertex in graph.keys():
        if vertex == 0 or vertex == max(graph.keys()):
            continue
        length = cost[0, vertex] + cost[vertex, max(graph.keys())] - wcets[vertex]
        if length in table:
            table[length].append(vertex)
        else:
            table[length] = [vertex]

    # prioritize according to rank-order of path lengh
    # the vertex lies on. lower number implies higher priority
    priority_list = [0 for i in range(max(graph.keys())+1)]
    for rank, key in enumerate(sorted(table.keys(), reverse=True)):
        for vertex in table[key]:
            priority_list[vertex] = rank
    return priority_list


def connection_vertex(graph: Dict[int, List[int]], priorities: List[int], vertex_u: int, vertex_v: int):
    source, sink = 0, max(graph.keys())
    if vertex_u is source and vertex_v is sink:
        return None
    elif vertex_u is source and vertex_v is not sink:
        return vertex_v
    elif vertex_u is not source and vertex_v is sink:
        return vertex_u
    elif vertex_u is not source and vertex_v is not sink:
        if priorities[vertex_u] >= priorities[vertex_v]:
            return vertex_u
        else:
            return vertex_v
    else:
        return None


def interference_set_vertex(graph: Dict[int, List[int]], vertex: int, priorities: List[int]) -> Set[int]:
    int_set_vertex = set()
    ancestors = find_ancestors(graph, vertex)
    descendants = find_descendants(graph, vertex)
    for u in graph.keys():
        if u is not vertex:
            if u not in ancestors and u not in descendants:
                if priorities[u] < priorities[vertex]:
                    int_set_vertex.add(u)
    return int_set_vertex


def connection_vertex_mat(graph: Dict[int, List[int]], priorities: List[int]):
    num_nodes = max(graph.keys())+1
    connection_mat = np.zeros((num_nodes, num_nodes))
    source, sink = 0, max(graph.keys())
    for vertex_u in graph.keys():
        for vertex_v in graph.keys():
            if vertex_u == vertex_v:
                connection_mat[vertex_u, vertex_v] = None
            elif vertex_u is source and vertex_v is sink:
                connection_mat[vertex_u, vertex_v] = None
            elif vertex_u is source and vertex_v is not sink:
                connection_mat[vertex_u, vertex_v] = vertex_v
            elif vertex_u is not source and vertex_v is sink:
                connection_mat[vertex_u, vertex_v] = vertex_u
            elif vertex_u is not source and vertex_v is not sink:
                if priorities[vertex_u] >= priorities[vertex_v]:
                    connection_mat[vertex_u, vertex_v] = vertex_u
                else:
                    connection_mat[vertex_u, vertex_v] = vertex_v
            else:
                connection_mat[vertex_u, vertex_v] = None
    return connection_mat


def compose(graph: Dict[int, List[int]], wcets: List[int], priorities: List[int], m: int, lhs: Alpha, rhs: Alpha):
    if lhs == rhs:
        return lhs
    if lhs[1] != rhs[0]:
        return None
    connection_vertex_a = connection_vertex(graph, priorities, lhs[0], lhs[1])
    connection_vertex_b = connection_vertex(graph, priorities, rhs[0], rhs[1])
    if connection_vertex_a == connection_vertex_b:
        V = set(interference_set_vertex(graph, lhs[1], priorities))
        U = set(interference_set_vertex(graph, lhs[0], priorities))
        W = set(interference_set_vertex(graph, rhs[1], priorities))
        composed = Alpha(lhs[0], rhs[1], lhs[2] + rhs[2] - wcets[connection_vertex_a]
                         - (leng(list(V.union(U.intersection(W))), wcets) / m))
        return composed
    return None


def makespan(graph: Dict[int, List[int]], priorities: List[int], weights: List[int], num_cpu: int):
    tuple_set = []
    #print(max(graph.keys()))
    for u in graph.keys():
        for v in graph[u]:
            #print("append tuples")
            tuple_set.append(Alpha(u, v, response_path(graph, priorities, [u, v], weights, num_cpu)))

    #print("start copy")
    tuple_set_base = tuple_set.copy()
    #rint("end copy")
    while True:
        tuple_set_pre = tuple_set_base.copy()
        for alpha in tuple_set_base:
            for beta in tuple_set_base:
                if alpha == beta:
                    continue
                else:
                    #print("[Attempt] %s + %s" % (alpha, beta))
                    composed = compose(graph, weights, priorities, num_cpu, alpha, beta)
                    if composed:
                        #print("[Composed] %s + %s -> %s" % (alpha, beta, composed))
                        if any(other.__ge__(composed) for other in tuple_set_base):
                            #print("no domination")
                            continue

                        elif any(composed.__ge__(other) for other in tuple_set_base):
                            #print(tuple_set_base)
                            for other in tuple_set_base:
                                if composed.__ge__(other):
                                    #print("FOUND %s >= %s" % (composed, other))
                                    #print("Remove %s and Append %s" % (other, composed))
                                    tuple_set_base.remove(other)
                                    tuple_set_base.append(composed)
                        else:
                            #print("append")
                            tuple_set_base.append(composed)
                    else:
                        pass
                        #print("Not composable %s %s" % (alpha, beta))
        if tuple_set_pre == tuple_set_base:
            break

    return max(alpha[2] for alpha in tuple_set_base if alpha[0] == 0 and alpha[1] == max(graph.keys()))


def interference(graph: Dict[int, List[int]], vertex: int, priorities: List[int]) -> Set[int]:
    int_set_vertex = set()
    ancestors = find_ancestors(graph, vertex)
    descendants = find_descendants(graph, vertex)
    for u in graph.keys():
        if u is not vertex:
            if u not in ancestors and u not in descendants:
                if priorities[u] < priorities[vertex]:
                    int_set_vertex.add(u)
    return int_set_vertex


def makespanHe(graph: Dict[int, List[int]], priorities: List[int], weights: List[int], num_cpu: int):
    int_mat = [interference(graph, vertex, priorities) for vertex in graph.keys()]
    con_mat = connection_vertex_mat(graph, priorities)
    num_nodes = max(graph.keys()) + 1

    cost = np.zeros((num_nodes, num_nodes))
    changed = True
    for k in range(num_nodes):
        if not changed:
            return cost[0, max(graph.keys())]
        changed = False
        for i in graph.keys():
            if k == 0:
                changed = True
                for j in range(num_nodes):
                    if j in graph[i]:
                        cost[i, j] = weights[i] + weights[j] + (1.0/num_cpu)*sum(weights[v]
                                                                                 for v in int_mat[i].union(int_mat[j]))
                    else:
                        cost[i, j] = -np.inf
            else:
                for j in graph.keys():
                    intersection = int_mat[i].intersection(int_mat[j])
                    for u in graph.keys():
                        if con_mat[i, u] == con_mat[u, j]:
                            max_cost = np.maximum(cost[i, j], cost[i, u]+cost[u, j]-weights[u]-(1.0/num_cpu)*sum(weights[v] for v in int_mat[u].union(intersection)))
                            if max_cost > cost[i, j]:
                                changed = True
                            cost[i, j] = max_cost
    return cost[0, max(graph.keys())]


def makespan_he_ecrts(task: Graph, num_cpu: int):
    # sanitize task model by connecting all unconnected
    # vertexes to the sink vertex.
    #for key in task.graph.keys():
    #    if not task.graph[key]:
    #        sink = max(task.graph.keys())
    #        if key == sink:
    #            continue
    #        task.graph[key].append(sink)
    priority_list = longest_path_priority_assignment(task.graph, task.weights)
    task.priorities = priority_list
    return makespanHe(task.graph, task.priorities, task.weights, num_cpu)

# Calculate the makespan for all the given DAG tasks
def calculate_makespan_all(taskset_org, available_processors_org):
    taskset = copy.deepcopy(taskset_org)
    available_processors = copy.deepcopy(available_processors_org)
    makespan_allset = []
    for i in range(len(taskset)):
        print('Processing set: ', i)
        makespan_oneset = []
        for j in range(len(taskset[i])):
            print('Processing task: ', j)
            task = copy.deepcopy(taskset[i][j])
            makespan_onetask = makespan_he_ecrts(task, available_processors)

            makespan_oneset.append(makespan_onetask)

        makespan_allset.append(makespan_oneset)

    return makespan_allset

def main():
    g = Graph(10)
    g.graph = {0: [1, 4], 1: [2, 3], 2: [5], 3: [5], 4: [5], 5: [6], 6: [7, 8], 7: [9], 8: [9], 9: []}
    g.weights = [0, 3, 1, 2, 1, 1, 2, 5, 2, 0]
    g.priorities = [0, 0, 0, 0, 0, 0]
    g.group = [0, 1, 2, 1, 3, 0, 2, 3, 0, 0]
    g.period = 50

    tsk_all = np.load('../experiments/inputs/tasksets_m100_p8_u0.05_q0_s0_g0_r0.npy', allow_pickle=True)

    #print(task.graph.keys())
    #print(max(task.graph.keys()))
    #print('verties: ', task.V)
    #print('len periorities: ', len(task.priorities))
    #print(longest_path_priority_assignment_new(task.graph, task.weights))


    #print(makespan_he_ecrts(tsk_all[0][1], 8))




    #calculate_makespan_all(tsk_all, 8)


    print(makespan_he_ecrts(g, 2))
    #print(longest_path_priority_assignment_new(g.graph, g.weights))
    #print(longest_path_priority_assignment(g.graph, g.weights))

    #print(g.priorities)
    #print(g.priorities)

    '''
    G = {0: [1,2,3], 1: [4], 2: [4], 3: [5], 4: [5], 5: []}
    C = [0, 8, 3, 6, 1, 0]
    M = 3
    L = [0, 2, 4, 5]
    PE = [0, 1, 5, 4, 2, 3]
    l, P = longest_path_priority_assignment(G, C)
    #print(exhaustive_bound(G, PE, C, 2))
    graph = Graph({0: [1,2,3], 1: [4], 2: [4], 3: [5], 4: [5], 5: []}, [0, 8, 3, 6, 1, 0])
    #print(connection_vertex(G, P, 2, 4))
    graph2 = Graph({0: [1], 1: [2, 4, 7], 2: [3], 3: [10], 4: [5], 5: [6,9], 6: [10], 7: [5, 8], 8: [10], 9: [10], 10: []},
               [0, 3, 3, 1, 1, 2, 3, 2, 2, 1, 0], [0, 1, 8, 9, 6, 2, 3, 4, 10, 7, 5])

    l, P = longest_path_priority_assignment(graph.graph, graph.weights)

    print(makespan_he_ecrts(graph, 4))

    '''





if __name__ == "__main__":
    main()

