import json
import pickle
import heapq
import itertools
import math

def build_weighted_graph(nodes: dict, edges: list) -> dict:
    graph = {name: [] for name in nodes}
    for edge in edges:
        if len(edge) < 2: continue
        n1, n2 = edge[0], edge[1]
        if n1 not in nodes or n2 not in nodes: continue
        p1, p2 = nodes[n1], nodes[n2]
        dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        graph[n1].append((n2, dist))
        graph[n2].append((n1, dist))
    return graph

def a_star(graph: dict, start: str, goal: str) -> list:
    if start not in graph or goal not in graph: return []
    open_set = [(0, start)]
    came_from = {}
    g_score = {name: float('inf') for name in graph}; g_score[start] = 0
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))
            
        for neighbor, weight in graph[current]:
            tentative_g = g_score[current] + weight
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))
    return []

def main():
    with open('Simulation/Map/map_data.json', 'r') as f:
        data = json.load(f)
    
    nodes = data.get('NODES', {})
    edges = data.get('EDGES', [])
    road_graph = build_weighted_graph(nodes, edges)
    
    route_cache = {}
    all_targets = list(set(data.get('LOAD_ZONES', []) + data.get('DUMP_ZONES', []) + data.get('FUEL_ZONES', [])))
    
    count = 0
    for s, e in itertools.permutations(all_targets, 2):
        path = a_star(road_graph, s, e)
        if path:
            route_cache[(s, e)] = path
        count += 1
    
    cache_data = {
        'road_graph': road_graph,
        'route_cache': route_cache
    }
    
    with open('MAP/map_cache.pkl', 'wb') as f:
        pickle.dump(cache_data, f)
    with open('Simulation/Map/map_cache.pkl', 'wb') as f:
        pickle.dump(cache_data, f)
        
    print("Cache regenerated.")

if __name__ == '__main__':
    main()
