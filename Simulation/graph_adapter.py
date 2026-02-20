import pickle
import numpy as np
import map_loader as map_data
import math
from config import POINTS_PER_SEGMENT

class GraphAdapter:
    def __init__(self, map_cache_path='map_cache.pkl', waypoints_path='waypoints.pkl'):
        self.road_graph = {}
        self.route_cache = {}
        self.waypoints_map = {}
        
        # Load Map Cache (The "Logical" Route)
        try:
            with open(map_cache_path, 'rb') as f:
                data = pickle.load(f)
                self.road_graph = data.get('road_graph', {})
                self.route_cache = data.get('route_cache', {})
            print(f"GraphAdapter: Loaded {len(self.route_cache)} logical routes from {map_cache_path}.")
        except FileNotFoundError:
            print(f"GraphAdapter Error: {map_cache_path} not found.")

        # Load Waypoints (The "Physical" Curve)
        try:
            with open(waypoints_path, 'rb') as f:
                self.waypoints_map = pickle.load(f)
            print(f"GraphAdapter: Loaded {len(self.waypoints_map)} physical segments from {waypoints_path}.")
        except FileNotFoundError:
            print(f"GraphAdapter Error: {waypoints_path} not found.")

    def _get_path_length(self, waypoints):
        """Calculates total length of a waypoint list in meters."""
        if not waypoints or len(waypoints) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(waypoints) - 1):
            p1 = np.array(waypoints[i])
            p2 = np.array(waypoints[i+1])
            dist = np.linalg.norm(p1 - p2)
            length += dist
        return length

    def _reconstruct_path_waypoints(self, route_node_names):
        """
        Stitches together the high-res waypoints for a given sequence of nodes.
        Logic mirrored from main.py to ensure consistency.
        """
        final_waypoints = []
        if not route_node_names: return []

        # If it's just a start and end node with no intermediates, check if they are directly connected in map_data
        # But usually route_node_names comes from A* so it has the full path.
        
        # We need to find the segment for each pair in the route
        for i in range(len(route_node_names) - 1):
            seg_start, seg_end = route_node_names[i], route_node_names[i+1]
            found_segment = False
            
            # Search through the visual chains to find the segment connecting these two nodes
            for chain_tuple, waypoints in self.waypoints_map.items():
                try:
                    # Try Forward Direction
                    if seg_start in chain_tuple:
                        idx = chain_tuple.index(seg_start)
                        # Check if next node in chain is our target
                        if idx + 1 < len(chain_tuple) and chain_tuple[idx+1] == seg_end:
                            start_wp_idx = idx * POINTS_PER_SEGMENT
                            end_wp_idx = (idx + 1) * POINTS_PER_SEGMENT
                            
                            # Valid bounds check
                            if start_wp_idx < len(waypoints):
                                # Extract the slice of points for this segment
                                segment = waypoints[start_wp_idx : min(end_wp_idx, len(waypoints))]
                                final_waypoints.extend(segment)
                                found_segment = True
                                break
                    
                    # Try Reverse Direction
                    if seg_end in chain_tuple:
                        idx = chain_tuple.index(seg_end)
                        if idx + 1 < len(chain_tuple) and chain_tuple[idx+1] == seg_start:
                            start_wp_idx = idx * POINTS_PER_SEGMENT
                            end_wp_idx = (idx + 1) * POINTS_PER_SEGMENT
                            
                            if start_wp_idx < len(waypoints):
                                segment = waypoints[start_wp_idx : min(end_wp_idx, len(waypoints))]
                                # Reverse the points to match our direction
                                # Note: [::-1] reverses the list
                                final_waypoints.extend(segment[::-1]) 
                                found_segment = True
                                break
                except ValueError:
                    continue
            
            # Fallback: If no curvy path found, just use straight line between nodes
            if not found_segment:
                # This happens for short connector segments or if waypoints aren't generated for a link
                if seg_start in map_data.NODES:
                     # If final_waypoints is empty, add start node
                     if not final_waypoints:
                         final_waypoints.append(map_data.NODES[seg_start])
                
                if seg_end in map_data.NODES:
                    final_waypoints.append(map_data.NODES[seg_end])

        # Ensure the final node is included if we built a path
        if final_waypoints and route_node_names[-1] in map_data.NODES:
            # Check distance to avoid duplicate point if already close
            last_wp = final_waypoints[-1]
            end_node_pos = map_data.NODES[route_node_names[-1]]
            if np.linalg.norm(np.array(last_wp) - end_node_pos) > 0.1:
                final_waypoints.append(end_node_pos)
            
        return final_waypoints

    def get_distance_matrix(self, sources, destinations):
        """
        Returns a dictionary {(start, end): distance_meters} for all pairs.
        Uses exact waypoint geometry for precision.
        """
        matrix = {}
        count = 0
        total = len(sources) * len(destinations)
        
        print(f"GraphAdapter: Computing precise distances for {total} pairs...")
        
        for start in sources:
            for end in destinations:
                if start == end:
                    matrix[(start, end)] = 0.0
                    continue

                # 1. Get the Logical Route (Node Sequence)
                route_nodes = self.route_cache.get((start, end))
                
                # If not in cache, we can't compute distance easily without A*
                # For this adaptor, we assume map_cache.pkl is complete (generated by generate_map_cache.py)
                if not route_nodes:
                    matrix[(start, end)] = float('inf')
                    continue

                # 2. Reconstruct the Physical Path (Waypoints)
                physical_waypoints = self._reconstruct_path_waypoints(route_nodes)
                
                # 3. Calculate Exact Length
                dist = self._get_path_length(physical_waypoints)
                
                # Sanity check: If dist is 0 but nodes are different, something failed or points are missing.
                # Use straight-line Euclidean as safety fallback.
                if dist < 0.1 and start != end:
                    p1 = map_data.NODES.get(start, np.array([0,0]))
                    p2 = map_data.NODES.get(end, np.array([0,0]))
                    dist = np.linalg.norm(p1 - p2)
                
                matrix[(start, end)] = dist
                count += 1
                
        print(f"GraphAdapter: Matrix calculation complete.")
        return matrix

# --- Test Execution ---
if __name__ == "__main__":
    adapter = GraphAdapter()
    
    # Test with real zones
    # We take a few samples to verify
    load_samples = map_data.LOAD_ZONES[:3] 
    dump_samples = map_data.DUMP_ZONES[:3] 
    
    matrix = adapter.get_distance_matrix(load_samples, dump_samples)
    
    print("\n--- Distance Matrix Sample (Meters) ---")
    for (src, dst), dist in matrix.items():
        print(f"{src} -> {dst}: {dist:.2f} m")
