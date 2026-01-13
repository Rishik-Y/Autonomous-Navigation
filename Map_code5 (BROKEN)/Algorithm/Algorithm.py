import csv
from collections import defaultdict
import heapq
import itertools
import copy
import os

# Load/unload time in seconds (can be changed for testing)
LOAD_UNLOAD_TIME = 1

class Truck:
    def __init__(self, truck_id, capacity, location):
        self.truck_id = truck_id
        self.capacity = capacity
        self.location = location
        self.loaded = 0
        self.total_time = 0
        self.route = []

    def __str__(self):
        return f"Truck {self.truck_id}: Location={self.location}, Capacity={self.capacity}kg, Loaded={self.loaded}kg, Total Time={self.total_time}s"

# Dijkstra's algorithm for shortest path (fastest path since 1km=1s)
def dijkstra(graph, start, end):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == end:
            return (cost, path)
        for neighbor in graph[node]:
            if neighbor[0] not in visited:
                heapq.heappush(queue, (cost + neighbor[1], neighbor[0], path))
    return (float('inf'), [])

def dp_min_time_multi_truck(node_capacities, truck_capacity, adjacency, dump_site, num_trucks=3):
    mines = [node for node in node_capacities if node != dump_site]
    initial_state = tuple(node_capacities[mine] for mine in mines)

    memo = {}
    choice = {}

    def dp(state, truck_times, truck_locations):
        key = (state, truck_times, truck_locations)

        if key in memo:
            return memo[key]

        # Base case: all mines depleted
        if all(coal == 0 for coal in state):
            # All trucks return to dump site if not already there
            max_return_time = 0
            for i, location in enumerate(truck_locations):
                if location != dump_site:
                    t, _ = dijkstra(adjacency, location, dump_site)
                    max_return_time = max(max_return_time, truck_times[i] + t)
                else:
                    max_return_time = max(max_return_time, truck_times[i])

            memo[key] = max_return_time
            choice[key] = None
            return max_return_time

        min_makespan = float('inf')
        best_assignment = None

        # Try assigning next trip to each available truck
        for truck_id in range(num_trucks):
            current_truck_time = truck_times[truck_id]
            current_location = truck_locations[truck_id]

            # Try all possible trips for this truck
            for r in range(1, len(mines)+1):
                active_mines = [i for i, coal in enumerate(state) if coal > 0]
                if r > len(active_mines):
                    continue

                for combo in itertools.combinations(active_mines, r):
                    # Check capacity constraint
                    coal_to_pick = [min(state[i], truck_capacity) for i in combo]
                    if sum(coal_to_pick) > truck_capacity:
                        continue

                    # Try all orders of visiting mines
                    for order in itertools.permutations(combo):
                        # Calculate trip time: current_location -> mines -> dump_site
                        route = [current_location] + [mines[i] for i in order] + [dump_site]
                        trip_time = 0

                        for i in range(len(route)-1):
                            t, _ = dijkstra(adjacency, route[i], route[i+1])
                            trip_time += t

                        # Add load/unload times
                        trip_time += LOAD_UNLOAD_TIME * len(order)  # Loading at each mine
                        trip_time += LOAD_UNLOAD_TIME  # Unloading at dump site

                        # Update state
                        new_state = list(state)
                        remaining_capacity = truck_capacity
                        for i in order:
                            take = min(new_state[i], remaining_capacity)
                            remaining_capacity -= take
                            new_state[i] -= take

                        # Update truck times and locations
                        new_truck_times = list(truck_times)
                        new_truck_locations = list(truck_locations)
                        new_truck_times[truck_id] = current_truck_time + trip_time
                        new_truck_locations[truck_id] = dump_site

                        # Recurse
                        makespan = dp(tuple(new_state), tuple(new_truck_times), tuple(new_truck_locations))

                        if makespan < min_makespan:
                            min_makespan = makespan
                            best_assignment = (truck_id, order, [mines[i] for i in order], route, trip_time)

        memo[key] = min_makespan
        choice[key] = best_assignment
        return min_makespan

    initial_truck_times = tuple([0] * num_trucks)
    initial_truck_locations = tuple([dump_site] * num_trucks)
    min_total_time = dp(initial_state, initial_truck_times, initial_truck_locations)

    return min_total_time, memo, choice

def get_optimal_schedule(road_graph, active_mines, mine_capacities, truck_capacity, dump_site, num_trucks):
    """
    Generates the optimal schedule for the simulation.
    
    Args:
        road_graph: The adjacency list from the map data (dict).
        active_mines: List of active mine node names.
        mine_capacities: Dict mapping mine names to coal capacity.
        truck_capacity: Capacity of a truck.
        dump_site: Name of the dump site node.
        num_trucks: Number of trucks.
        
    Returns:
        list of lists: A list where each element is a list of actions for a truck.
                       Action format: {'type': 'travel'|'load'|'unload', 'target': node_name, 'duration': seconds}
    """
    
    # 1. Convert road_graph to simple adjacency list for the algorithm
    # The algorithm expects adjacency[src] = [(dst, dist), ...]
    # The map_data road_graph is {src: [(dst, weight), ...]}
    # We need to filter it to only include relevant nodes if we want to optimize, 
    # but the algorithm uses Dijkstra internally so we can pass the full graph.
    # However, the algorithm assumes integer weights in some places or specific structure.
    # Let's adapt the graph.
    
    adjacency = defaultdict(list)
    for src, neighbors in road_graph.items():
        for dst, weight in neighbors:
            # Weight in the map is distance in meters. 
            # The algorithm treats distance as time (1m = 1s? No, 1km=1s in comment, but let's assume 1 unit = 1 unit).
            # We'll just use the raw distance.
            adjacency[src].append((dst, weight))
            
    # 2. Filter capacities to only active mines
    filtered_capacities = {k: v for k, v in mine_capacities.items() if k in active_mines}
    
    # 3. Run DP
    print("Running DP Algorithm...")
    min_time, memo, choice = dp_min_time_multi_truck(filtered_capacities, truck_capacity, adjacency, dump_site, num_trucks)
    print(f"DP Finished. Min makespan: {min_time}")
    
    # 4. Reconstruct Schedule
    mines = [node for node in filtered_capacities if node != dump_site]
    initial_state = tuple(filtered_capacities[mine] for mine in mines)
    
    state = initial_state
    truck_times = [0] * num_trucks
    truck_locations = [dump_site] * num_trucks
    truck_schedules = [[] for _ in range(num_trucks)]

    while not all(coal == 0 for coal in state):
        key = (tuple(state), tuple(truck_times), tuple(truck_locations))
        assignment = choice.get(key)
        if assignment is None:
            break

        truck_id, order, mines_order, route, trip_time = assignment

        # Build detailed action list for this trip
        current_location = truck_locations[truck_id]
        
        # The route list from DP is [start, mine1, mine2, ..., dump]
        # We need to navigate between them.
        
        # 1. Travel to first mine
        # 2. Load
        # 3. Travel to next mine...
        # 4. Travel to dump
        # 5. Unload
        
        for idx, mine in enumerate(mines_order):
            # Travel from current to mine
            # We don't need the full path here, just the target. The simulation will handle pathfinding.
            # But wait, the simulation needs to know WHERE to go.
            truck_schedules[truck_id].append({'type': 'travel', 'target': mine})
            truck_schedules[truck_id].append({'type': 'load', 'target': mine, 'duration': LOAD_UNLOAD_TIME})
            current_location = mine
            
        # Return to dump site
        truck_schedules[truck_id].append({'type': 'travel', 'target': dump_site})
        truck_schedules[truck_id].append({'type': 'unload', 'target': dump_site, 'duration': LOAD_UNLOAD_TIME})

        # Update state
        new_state = list(state)
        remaining_capacity = truck_capacity
        for i in order:
            take = min(new_state[i], remaining_capacity)
            remaining_capacity -= take
            new_state[i] -= take
        state = tuple(new_state)

        # Update truck time and location
        truck_times[truck_id] += trip_time
        truck_locations[truck_id] = dump_site
        
    return truck_schedules

if __name__ == "__main__":
    # Test block
    pass


