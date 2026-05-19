import itertools
import heapq
import pickle

import numpy as np

import map_data
import map_storage


def build_weighted_graph(nodes: dict, edges: list) -> dict:
    graph = {name: [] for name in nodes}
    for edge in edges:
        if len(edge) < 2:
            continue
        n1_name, n2_name = edge[0], edge[1]
        if n1_name not in nodes or n2_name not in nodes:
            continue
        p1 = nodes[n1_name]
        p2 = nodes[n2_name]
        distance = np.linalg.norm(p1 - p2)
        graph[n1_name].append((n2_name, distance))
        graph[n2_name].append((n1_name, distance))
    return graph


def a_star_pathfinding(graph: dict, start_name: str, goal_name: str) -> list:
    if start_name not in graph or goal_name not in graph:
        return []
    open_set = [(0, start_name)]
    came_from = {}
    g_score = {name: float("inf") for name in graph}
    g_score[start_name] = 0

    while open_set:
        _, current_name = heapq.heappop(open_set)
        if current_name == goal_name:
            path_names = []
            temp = current_name
            while temp in came_from:
                path_names.append(temp)
                temp = came_from[temp]
            path_names.append(start_name)
            return list(reversed(path_names))

        for neighbor_name, weight in graph[current_name]:
            tentative_g_score = g_score[current_name] + weight
            if tentative_g_score < g_score[neighbor_name]:
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor_name))
    return []


def main():
    road_graph = build_weighted_graph(map_data.NODES, map_data.EDGES)
    route_cache = {}
    fuel_zones = getattr(map_data, "FUEL_ZONES", [])
    all_targets = list(set(map_data.LOAD_ZONES + map_data.DUMP_ZONES + fuel_zones))

    for start, end in itertools.permutations(all_targets, 2):
        path = a_star_pathfinding(road_graph, start, end)
        if path:
            route_cache[(start, end)] = path

    data = pickle.dumps({"road_graph": road_graph, "route_cache": route_cache})
    out = "map_cache.pkl"
    map_storage.write_binary_file(
        out,
        data,
        copy_targets=[map_storage.legacy_path(out), map_storage.simulation_path(out)],
    )


if __name__ == "__main__":
    main()
