import numpy as np
import pickle
import map_data
import os

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

def main():
    print("Building road graph from map_data...")
    if not hasattr(map_data, 'NODES') or not hasattr(map_data, 'EDGES'):
        print("Error: map_data.py is missing NODES or EDGES.")
        return

    road_graph = build_weighted_graph(map_data.NODES, map_data.EDGES)
    
    output_file = 'map_cache.pkl'
    print(f"Saving graph to {output_file}...")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump({'road_graph': road_graph}, f)
        print("Done! map_cache.pkl regenerated successfully.")
    except Exception as e:
        print(f"Error saving pickle file: {e}")

if __name__ == "__main__":
    main()
