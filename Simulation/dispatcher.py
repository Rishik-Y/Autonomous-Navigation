import numpy as np
from Map import map_loader as map_data
from config import *
import copy
from Algorithm.planner_registry import load_global_planner

class Dispatcher:
    def __init__(self, road_graph, coal_capacities=None, global_planner_name=None):
        self.base_road_graph = copy.deepcopy(road_graph)
        self.current_road_graph = copy.deepcopy(road_graph)
        
        # Global Optimizer Integration
        try:
            planner_name = global_planner_name or DEFAULT_GLOBAL_PLANNER
            self.optimizer = load_global_planner(planner_name)
            self.use_global_optimization = True
        except Exception as e:
            print(f"Dispatcher Warning: Global planner failed to init ({e}). Using local fallback only.")
            self.optimizer = None
            self.use_global_optimization = False
            
        self.pending_assignments = {} # {truck_id: [target_node_1, target_node_2, ...]}
        
        # Centralized State Tracking
        # Structure: { 'node_name': { 'en_route': 0, 'service_time': 30.0, 'coal_remaining': X } }
        self.site_states = {}
        
        # Coal capacities from config (None means unlimited/untracked)
        self.coal_capacities = coal_capacities or {}
        
        # Initialize States
        for zone in map_data.LOAD_ZONES:
            raw_capacity = self.coal_capacities.get(zone, float('inf'))
            
            # Reverted unit fix as per user instruction. 100 means 100.
            coal_remaining = raw_capacity
                
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

        # --- Junction Reservation System ---
        # Max trucks allowed inside a single junction simultaneously.
        self.MAX_JUNCTION_TRUCKS = 2

        # Detect hub nodes: intersections with ≥2 incoming AND ≥2 outgoing roads.
        # Computed once at startup — zero runtime overhead.
        in_degree = {node: 0 for node in road_graph}
        for node, edges in road_graph.items():
            for target, _ in edges:
                if target in in_degree:
                    in_degree[target] += 1

        terminal_nodes = set(map_data.LOAD_ZONES + map_data.DUMP_ZONES)
        self.hub_nodes = set(
            node for node in road_graph
            if in_degree.get(node, 0) >= 2
            and len(road_graph[node]) >= 2
            and node not in terminal_nodes  # terminals are handled by site_states
        )
        print(f"Dispatcher: Detected {len(self.hub_nodes)} junction nodes for yield control.")
        # Junction State: { node_name: {'count': 0, 'heading': 0.0} }
        self.junction_states = {node: {'count': 0, 'heading': 0.0} for node in self.hub_nodes}
        # Track which truck holds which junction slot
        self.junction_holders = {}  # { (node_name, car_id): True }

    def update_global_plan(self, trucks):
        """
        Runs the Global Optimizer to update fleet assignments.
        Should be called periodically (e.g. every 30s).
        """
        if not self.use_global_optimization or not self.optimizer:
            return

        # print("Dispatcher: Running Global Optimization (Swarm Sim)...")
        # Returns dict {truck_id: [mine_A, mine_B, ...]}
        new_assignments_map = self.optimizer.optimize_assignments(trucks, self.site_states)
        
        # Merge new assignments
        # We overwrite the entire pending queue because the new plan is fresher/better
        if new_assignments_map:
            for t_id, plan_list in new_assignments_map.items():
                self.pending_assignments[t_id] = plan_list
            
            # print(f"Dispatcher: Updated global plans for {len(new_assignments_map)} trucks.")

    def assign_task(self, car):
        """
        Determines the best destination for the car.
        Priority 1: Global Plan Queue (Pop next target)
        Priority 2: Local Greedy Heuristic (Fallback)
        Returns: target_node_name (str)
        """
        current_pos = np.array([car.x_m, car.y_m])
        is_loaded = (car.current_mass_kg > MASS_KG + 100) # Simple threshold
        
        # --- 1. Global Optimizer Queue ---
        if self.use_global_optimization and car.id in self.pending_assignments:
            plan = self.pending_assignments[car.id]
            if len(plan) > 0:
                target = plan.pop(0)
                # print(f"Dispatcher: Truck {car.id} following global plan -> {target}")
                self.site_states[target]['en_route'] += 1
                return target
                
        # --- 2. Fallback Greedy Dispatch ---
        target_list = map_data.DUMP_ZONES if is_loaded else map_data.LOAD_ZONES
        
        best_site = None
        best_score = float('inf')
        
        for site in target_list:
            # Skip empty mines
            if not is_loaded and self.site_states[site]['coal_remaining'] <= 0:
                continue
                
            # 1. Estimate Travel Time
            if self.use_global_optimization and self.optimizer:
                 # Use high-fidelity distance if available
                 # We need to find nearest node to car if not at a node
                 start_node = car.current_node_name if car.current_node_name else None
                 if start_node:
                     time_est = self.optimizer.get_travel_time(start_node, site)
                     travel_time = time_est
                 else:
                     # Euclidean fallback
                     site_pos = map_data.NODES[site]
                     dist = np.linalg.norm(site_pos - current_pos)
                     est_speed = SPEED_MS_LOADED if is_loaded else SPEED_MS_EMPTY
                     travel_time = dist / max(est_speed, 1.0)
            else:
                site_pos = map_data.NODES[site]
                dist = np.linalg.norm(site_pos - current_pos)
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

    def request_junction(self, node_name, car_id, heading):
        """
        Directional junction reservation.
        If empty, truck claims it and sets the active heading.
        Subsequent trucks going the SAME direction (<60 deg diff) can enter freely (platooning).
        Cross-traffic must yield until the junction count hits 0.
        """
        import math
        key = (node_name, car_id)
        if key in self.junction_holders:
            return True  # Already inside
            
        state = self.junction_states.setdefault(node_name, {'count': 0, 'heading': 0.0})
        
        if state['count'] == 0:
            # Junction is empty, claim it
            state['heading'] = heading
            state['count'] = 1
            self.junction_holders[key] = True
            return True
            
        # Junction is active, check direction compatibility
        angle_diff = (heading - state['heading'] + math.pi) % (2 * math.pi) - math.pi
        if abs(angle_diff) < math.radians(60):
            # Same direction platoon — let them in!
            state['count'] += 1
            self.junction_holders[key] = True
            return True
            
        return False  # Cross-traffic — yield!

    def release_junction(self, node_name, car_id):
        """Release the truck's hold on the junction."""
        key = (node_name, car_id)
        if key in self.junction_holders:
            del self.junction_holders[key]
            if node_name in self.junction_states:
                self.junction_states[node_name]['count'] = max(0, self.junction_states[node_name]['count'] - 1)

    def count_yielding_trucks(self, cars):
        """Returns how many trucks are currently in YIELDING_AT_JUNCTION state."""
        return sum(1 for c in cars if c.op_state == "YIELDING_AT_JUNCTION")

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
