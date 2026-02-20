import map_data
import itertools
import heapq
import copy
import time
from collections import defaultdict
from config import *
from graph_adapter import GraphAdapter

class GlobalOptimizer:
    def __init__(self):
        self.adapter = GraphAdapter()
        self.load_zones = map_data.LOAD_ZONES
        self.dump_zones = map_data.DUMP_ZONES
        
        # Pre-compute distances for critical pairs
        self.static_dist_matrix = self.adapter.get_distance_matrix(
            self.load_zones + self.dump_zones, 
            self.load_zones + self.dump_zones
        )
        print("GlobalOptimizer: initialized with static distance matrix.")

    def get_travel_time(self, start, end):
        """
        Returns estimated travel time in seconds based on physics config.
        """
        # If start/end are same, time is 0
        if start == end: return 0.0

        dist = self.static_dist_matrix.get((start, end))
        if dist is None:
            # If not in matrix (e.g. from 'main_hub' or random road point), we need dynamic calculation
            # But graph_adapter is static. 
            # Fallback: Euclidean estimate / speed
            p1 = map_data.NODES.get(start)
            p2 = map_data.NODES.get(end)
            if p1 is not None and p2 is not None:
                import numpy as np
                dist = np.linalg.norm(p1 - p2) * 1.5 # 1.5 factor for road curviness
            else:
                return 60.0 # Default fallback penalty

        # Speed model
        if end in map_data.DUMP_ZONES:
            speed = SPEED_MS_LOADED
        else:
            speed = SPEED_MS_EMPTY
            
        return max(1.0, dist / speed)

    def optimize_assignments(self, trucks, site_states):
        """
        Runs a Global Optimization to assign target mines to trucks.
        Returns: {truck_id: [target_node_1, target_node_2, ...]}
        """
        
        # 1. Identify Active Mines
        active_mines = []
        for name, state in site_states.items():
            if name in map_data.LOAD_ZONES and state.get('coal_remaining', 0) > 0:
                active_mines.append(name)
        
        if not active_mines:
            return {}

        # 2. Select Critical Subset (Top N) to prevent DP explosion
        # Sort by most coal remaining? Or purely distance?
        # Let's take top 6 mines with most coal for the "Global Plan"
        # (DP with 6 mines and 3-5 trucks is manageable)
        active_mines.sort(key=lambda m: site_states[m].get('coal_remaining', 0), reverse=True)
        critical_mines = active_mines[:6]
        
        # If we have free trucks, we need to assign them.
        free_trucks = [t for t in trucks if t.op_state in ("IDLE", "UNLOADING", "RETURNING_TO_START") or (t.op_state == "GOING_TO_ENDPOINT" and not t.target_node_name)]
        
        if not free_trucks:
            return {}

        # 3. Setup DP State
        # State: (coal_remaining_tuple)
        truck_capacity = CARGO_TON * 1000 # kg
        
        # Normalize coal to "Truck Loads" to simplify state space
        # e.g. 5000kg -> 5 loads
        state_coal = []
        for m in critical_mines:
            kg = site_states[m].get('coal_remaining', 0)
            
            # Handle infinite coal (e.g. if not configured)
            if kg == float('inf'):
                 loads = 999 
            else:
                 loads = math.ceil(kg / truck_capacity)
            
            # Cap max loads for planning horizon (e.g. plan only next 2 loads per mine)
            state_coal.append(min(loads, 2)) 
            
        initial_state = tuple(state_coal)
        print(f"GlobalOptimizer: Planning for mines: {critical_mines} with state {initial_state}")
        
        # Truck locations (approximate to nearest node)
        truck_locs = []
        truck_times = []
        
        # We only plan for the free trucks to save complexity
        
        free_trucks = [t for t in free_trucks] # (Already filtered above)
        print(f"GlobalOptimizer: Planning for {len(free_trucks)} free trucks.")

        sim_truck_ids = [t.id for t in free_trucks]
        for t in free_trucks:
            # Use current node or default to main hub
            loc = t.current_node_name if t.current_node_name in map_data.NODES else "main_hub"
            truck_locs.append(loc)
            truck_times.append(0) # Relative start time 0
            
        # 4. Run DP Solver (Adapted from Algorithm.py)
        # We need to generate a SEQUENCE of assignments (Lookahead > 1)
        plan_depth = 5
        full_assignments = defaultdict(list)
        
        # Clone state for simulation
        sim_state = list(initial_state)
        sim_truck_locs = list(truck_locs)
        sim_truck_times = list(truck_times)
        
        # Iterative Simulation to build the queue
        for step in range(plan_depth):
            # Check if any coal left
            if sum(sim_state) <= 0:
                break
                
            # Solve for this step
            # We pass the SIMULATED state
            best_moves = self.solve_dp_greedy_lookahead(
                tuple(sim_state), 
                tuple(sim_truck_times), 
                tuple(sim_truck_locs), 
                critical_mines,
                truck_capacity
            )
            
            # Record moves and Update Simulation State
            step_has_moves = False
            for i, target_mine in enumerate(best_moves):
                truck_id = sim_truck_ids[i]
                if target_mine:
                    step_has_moves = True
                    full_assignments[truck_id].append(target_mine)
                    
                    # UPDATE SIM STATE for next step
                    # 1. Location becomes the mine
                    sim_truck_locs[i] = target_mine
                    
                    # 2. Time increases (Travel + Service)
                    travel = self.get_travel_time(sim_truck_locs[i], target_mine)
                    sim_truck_times[i] += travel + LOAD_UNLOAD_TIME_S
                    
                    # 3. Coal decreases
                    # Note: critical_mines vs active_mines mismatch potential
                    # We used critical_mines for DP state, so we find index in THAT
                    if target_mine in critical_mines:
                        c_idx = critical_mines.index(target_mine)
                        sim_state[c_idx] = max(0, sim_state[c_idx] - 1)
                        
                    # 4. Simulate Trip to Dump (Implicitly needed for next Load)
                    # We assume nearest dump for time calculation
                    # For simplicty, add fixed return time or calc to "main_hub"
                    return_time = self.get_travel_time(target_mine, "main_hub")
                    sim_truck_times[i] += return_time + LOAD_UNLOAD_TIME_S
                    sim_truck_locs[i] = "main_hub" # Reset to hub/dump for next iteration start

            if not step_has_moves:
                break
                
        return full_assignments

    def solve_dp_greedy_lookahead(self, state, truck_times, truck_locations, mines, capacity):
        """
        A simplified 1-step lookahead + Heuristic (Cost Function) to replace full DP.
        Full DP for 5 trucks * 6 mines is still too slow for real-time (60fps).
        """
        num_trucks = len(truck_locations)
        best_moves = [None] * num_trucks
        
        # We assign trucks one by one using a "Regret-based" or "Best Fit" approach
        # But to be "Smarter", we check all combinations of (Truck -> Mine) for the first step.
        
        # Generate all valid moves for each truck
        # Move = (Mine Index)
        valid_moves = [] # List of lists
        
        active_indices = [i for i, loads in enumerate(state) if loads > 0]
        if not active_indices:
            return best_moves

        # For each truck, try every active mine
        # We want to find the combination of assignments (T1->M1, T2->M2...) that minimizes cost
        
        # Cost = Travel Time + Wait Time
        # But we must coordinate! If T1 goes to M1, T2 arriving at M1 later pays a penalty.
        
        # Simple Permutation Search (manageable for ~3-5 trucks)
        # If trucks > 4, we fallback to greedy
        
        if num_trucks > 4:
            # Greedy fallback
            chosen_mines = []
            for i in range(num_trucks):
                best_mine = None
                best_cost = float('inf')
                start = truck_locations[i]
                
                for m_idx in active_indices:
                    mine = mines[m_idx]
                    cost = self.get_travel_time(start, mine)
                    # Penalty for congestion (if other trucks chose this)
                    cost += chosen_mines.count(mine) * LOAD_UNLOAD_TIME_S 
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_mine = mine
                
                if best_mine:
                    best_moves[i] = best_mine
                    chosen_mines.append(best_mine)
            return best_moves

        # Exact Combination Search for small fleets
        best_total_cost = float('inf')
        best_combination = [None] * num_trucks
        
        # Create list of choices for each truck: [None] + [Mine1, Mine2...]
        choices = []
        for _ in range(num_trucks):
            opts = [None] # Option to do nothing/wait
            opts.extend([mines[i] for i in active_indices])
            choices.append(opts)
            
        # Iterate all combinations (Cartesian product)
        # 5 mines, 3 trucks => 6^3 = 216 combinations. Very fast.
        for combination in itertools.product(*choices):
            # Calculate cost of this assignment set
            # Max Makespan? Or Sum of completion times? 
            # Sum of completion times usually minimizes average wait.
            
            current_cost = 0
            mine_arrivals = defaultdict(list)
            
            valid_combo = False
            for t_idx, mine in enumerate(combination):
                if mine:
                    valid_combo = True
                    t_time = self.get_travel_time(truck_locations[t_idx], mine)
                    mine_arrivals[mine].append(t_time)
            
            if not valid_combo: continue
            
            # Add queue penalties
            for mine, arrivals in mine_arrivals.items():
                arrivals.sort()
                # First truck arrives at t. Finishes at t + LOAD.
                # Second truck arrives at t2. Starts at max(t2, t+LOAD).
                finish_time = 0
                for arr in arrivals:
                    start_service = max(arr, finish_time)
                    finish_time = start_service + LOAD_UNLOAD_TIME_S
                    current_cost += finish_time # Objective: Minimize total finish time
            
            if current_cost < best_total_cost:
                best_total_cost = current_cost
                best_combination = list(combination)
                
        return best_combination
