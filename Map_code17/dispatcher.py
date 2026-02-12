import numpy as np
import map_data
from config import *
import copy

class Dispatcher:
    def __init__(self, road_graph, coal_capacities=None):
        self.base_road_graph = copy.deepcopy(road_graph)
        self.current_road_graph = copy.deepcopy(road_graph)
        
        # Centralized State Tracking
        # Structure: { 'node_name': { 'en_route': 0, 'service_time': 30.0, 'coal_remaining': X } }
        self.site_states = {}
        
        # Coal capacities from config (None means unlimited/untracked)
        self.coal_capacities = coal_capacities or {}
        
        # Initialize States
        for zone in map_data.LOAD_ZONES:
            coal_remaining = self.coal_capacities.get(zone, float('inf'))
            self.site_states[zone] = {
                'en_route': 0, 
                'service_time': LOAD_UNLOAD_TIME_S,
                'coal_remaining': coal_remaining
            }
            
        for zone in map_data.DUMP_ZONES:
            self.site_states[zone] = {
                'en_route': 0, 
                'service_time': LOAD_UNLOAD_TIME_S,
                'coal_dumped': 0  # Track total coal dumped at this site
            }


    def assign_task(self, car):
        """
        Determines the best destination for the car based on a Cost Function.
        Returns: target_node_name (str)
        """
        current_pos = np.array([car.x_m, car.y_m])
        is_loaded = (car.current_mass_kg > MASS_KG + 100) # Simple threshold
        
        candidates = map_data.DUMP_ZONES if is_loaded else map_data.LOAD_ZONES
        
        best_site = None
        best_score = float('inf')
        
        for site in candidates:
            # Skip depleted mines (only for load zones)
            if not is_loaded:
                state = self.site_states.get(site, {})
                coal_remaining = state.get('coal_remaining', float('inf'))
                if coal_remaining <= 0:
                    continue  # Skip this mine, it's depleted
            
            # 1. Estimate Travel Time (Heuristic: Euclidean / Speed)
            site_pos = map_data.NODES[site]
            dist = np.linalg.norm(site_pos - current_pos)
            
            # Use strict empty/loaded speeds for better estimation
            est_speed = SPEED_MS_LOADED if is_loaded else SPEED_MS_EMPTY
            travel_time = dist / max(est_speed, 1.0)
            
            # 2. Estimate Queue/Service Time
            state = self.site_states.get(site, {'en_route': 0, 'service_time': 30.0})
            wait_time = state['en_route'] * state['service_time']
            
            # 3. Cost Function
            # Score = Travel Time + Wait Time
            score = travel_time + wait_time
            
            if score < best_score:
                best_score = score
                best_site = site
                
        # Reservation: Increment en_route counter for the chosen site
        if best_site:
            self.site_states[best_site]['en_route'] += 1
            # print(f"Dispatcher: Assigned {car.id} to {best_site} (Score: {best_score:.1f}, Queue: {self.site_states[best_site]['en_route']})")
            
        return best_site

    def release_reservation(self, site_name):
        """Decrements the en_route counter when a truck finishes its task."""
        if site_name in self.site_states:
            if self.site_states[site_name]['en_route'] > 0:
                self.site_states[site_name]['en_route'] -= 1
                # print(f"Dispatcher: Truck finished at {site_name}. Queue: {self.site_states[site_name]['en_route']}")

    def consume_coal(self, site_name, amount):
        """
        Decrements coal at a load zone when a truck picks up coal.
        Returns the actual amount consumed (may be less if mine depleted).
        """
        if site_name in self.site_states:
            state = self.site_states[site_name]
            if 'coal_remaining' in state:
                coal_remaining = state['coal_remaining']
                if coal_remaining == float('inf'):
                    return amount  # Unlimited coal
                actual_amount = min(amount, coal_remaining)
                state['coal_remaining'] -= actual_amount
                if actual_amount > 0:
                    print(f"Coal consumed at {site_name}: {actual_amount} kg (remaining: {state['coal_remaining']} kg)")
                return actual_amount
        return amount  # Default: return requested amount

    def get_coal_remaining(self, site_name):
        """Get remaining coal at a load zone."""
        if site_name in self.site_states:
            return self.site_states[site_name].get('coal_remaining', float('inf'))
        return float('inf')

    def record_coal_dumped(self, site_name, amount):
        """Record coal dumped at a dump site."""
        if site_name in self.site_states:
            if 'coal_dumped' in self.site_states[site_name]:
                self.site_states[site_name]['coal_dumped'] += amount
                print(f"Coal dumped at {site_name}: {amount} kg (total: {self.site_states[site_name]['coal_dumped']} kg)")

    def update_traffic_weights(self, cars):
        """
        Updates self.current_road_graph weights based on traffic congestion.
        Heavy edges get higher weights to discourage routing.
        """
        # Reset to base weights
        self.current_road_graph = copy.deepcopy(self.base_road_graph)
        
        # Count cars near edges
        # Optimization: Map cars to edges coarsely
        # For a small fleet (5-20), O(N*E) is acceptable. 
        # For simplicity, we just check which node the car is 'targeting' or closest to?
        # Better: Find the car's current segment on its path.
        
        edge_counts = {}
        
        for car in cars:
            if not car.path: continue
            
            # Find which edge the car is currently traversing
            # This requires mapping the car's s_path_m back to graph nodes.
            # This is complex with the current "stitched path" logic.
            # Alternative Proxy: Find the closest graph node to the car.
            
            closest_node = None
            min_dist = float('inf')
            
            # Optimization: Only check nodes in the car's current path segment?
            # Let's stick to a simpler heuristic:
            # If a car is "Going to X", penalize the area around X? No, that punishes the destination.
            
            # Correct Approach:
            # Car has car.path.
            # car.path was built from nodes.
            # We don't easily know WHICH graph edge corresponds to `s_path_m`.
            # Let's use a spatial lookup or just iterate edges in the graph?
            
            # Simple Congestion:
            # 1. Find closest node to car.
            # 2. Penalize all edges connected to that node.
            
            c_pos = np.array([car.x_m, car.y_m])
            # Only check nodes reasonably close to avoid O(N_nodes) if possible, 
            # but N_nodes is small (~100). fast enough.
            
            for name, pos in map_data.NODES.items():
                d = np.linalg.norm(pos - c_pos)
                if d < min_dist:
                    min_dist = d
                    closest_node = name
            
            if closest_node and min_dist < 20.0: # Only count if actually ON the road network
                edge_counts[closest_node] = edge_counts.get(closest_node, 0) + 1

        # Apply Penalties
        # Graph structure: { 'NodeA': [('NodeB', weight), ...], ... }
        ALPHA = 2.0 # Congestion Multiplier Factor
        
        for node, count in edge_counts.items():
            if count > 0:
                # Penalize incoming edges TO this node? Or outgoing?
                # Penalize edges connected to this node.
                
                # Penalize Outgoing from Node
                if node in self.current_road_graph:
                    new_list = []
                    for target, weight in self.current_road_graph[node]:
                        # W_new = W_old * (1 + count * 0.5)
                        new_weight = weight * (1.0 + count * ALPHA)
                        new_list.append((target, new_weight))
                    self.current_road_graph[node] = new_list
                    
                # Penalize Incoming (Harder to find in adjacency list without reverse map)
                # We iterate all nodes to find links TO 'node'
                for start_node, edges in self.current_road_graph.items():
                    new_edges = []
                    changed = False
                    for target, weight in edges:
                        if target == node:
                            new_weight = weight * (1.0 + count * ALPHA)
                            new_edges.append((target, new_weight))
                            changed = True
                        else:
                            new_edges.append((target, weight))
                    if changed:
                        self.current_road_graph[start_node] = new_edges

    def get_graph(self):
        return self.current_road_graph
