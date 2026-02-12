import numpy as np
import pickle
import map_data
import os
import heapq
import itertools

def build_weighted_graph(nodes: dict, edges: list) -> dict:
    """
    Builds a weighted graph where edge weights are Euclidean distances between nodes.
    """
    graph = {name: [] for name in nodes}
    for edge in edges:
        # Check if edge has at least 2 elements (source, dest)
        if len(edge) < 2:
            continue
            
        n1_name, n2_name = edge[0], edge[1]
        
        if n1_name not in nodes or n2_name not in nodes:
            print(f"Warning: Edge nodes not found in NODES dict: {n1_name}, {n2_name}")
            continue

        p1 = nodes[n1_name]
        p2 = nodes[n2_name]
        distance = np.linalg.norm(p1 - p2)
        
        # Add bidirectional connection
        graph[n1_name].append((n2_name, distance))
        graph[n2_name].append((n1_name, distance))
    return graph

def a_star_pathfinding(graph: dict, start_name: str, goal_name: str) -> list:
    """Finds the shortest path between two nodes using A*."""
    if start_name not in graph or goal_name not in graph:
        return []
        
    open_set = [(0, start_name)] # (f_score, node_name)
    came_from = {}
    g_score = {name: float('inf') for name in graph}; g_score[start_name] = 0
    
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
    print("Building road graph from map_data...")
    if not hasattr(map_data, 'NODES') or not hasattr(map_data, 'EDGES'):
        print("Error: map_data.py is missing NODES or EDGES.")
        return

    road_graph = build_weighted_graph(map_data.NODES, map_data.EDGES)
    
    # --- ROUTE CACHING ---
    print("Pre-calculating optimal routes between zones...")
    route_cache = {}
    
    # Combine lists but keep unique
    # We check for FUEL_ZONES attribute just in case map_data.py isn't updated yet in some environments, though it should be.
    fuel_zones = getattr(map_data, 'FUEL_ZONES', [])
    all_targets = list(set(map_data.LOAD_ZONES + map_data.DUMP_ZONES + fuel_zones))
    
    count = 0
    total_pairs = len(all_targets) * (len(all_targets) - 1)
    
    # We use permutations because path A->B might differ from B->A (though graph is undirected here, 
    # it's good practice for A* cache to be explicit)
    for start, end in itertools.permutations(all_targets, 2):
        path = a_star_pathfinding(road_graph, start, end)
        if path:
            route_cache[(start, end)] = path
        
        count += 1
        if count % 100 == 0:
            print(f"  Cached {count}/{total_pairs} routes...", end='\r')
            
    print(f"\nCached {len(route_cache)} valid routes.")

    output_file = 'map_cache.pkl'
    print(f"Saving graph and route cache to {output_file}...")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump({
                'road_graph': road_graph,
                'route_cache': route_cache
            }, f)
        print("Done! map_cache.pkl regenerated successfully.")
    except Exception as e:
        print(f"Error saving pickle file: {e}")

if __name__ == "__main__":
    main()
