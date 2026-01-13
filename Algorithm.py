import csv
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import itertools
import copy

# Load/unload time in seconds (can be changed for testing)
LOAD_UNLOAD_TIME = 1

# Read edges from CSV
edges = []
with open('edges.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        edges.append((row['source'], row['destination'], int(row['distance'])))

# Build graph
G = nx.Graph()
for src, dst, dist in edges:
    G.add_edge(src, dst, weight=dist)

# Draw graph
# Use edge distances to influence layout
edge_lengths = {(src, dst): dist for src, dst, dist in edges}

# Kamada-Kawai layout uses edge weights as distances
G_for_layout = nx.Graph()
for src, dst, dist in edges:
    G_for_layout.add_edge(src, dst, weight=dist)

pos = nx.kamada_kawai_layout(G_for_layout, weight='weight')
plt.figure(figsize=(8,6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1200, font_size=12, font_weight='bold', edge_color='gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Coal Mine Network Map (Edge Lengths = Distances)")
# Save the graph as an image instead of displaying it
plt.savefig("map.png")
# plt.show()  # Commented out to avoid display issues

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

# Read node capacities
node_capacities = {}
with open('nodes.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        node_capacities[row['source']] = int(row['source_capacity'])

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

# Build adjacency list for Dijkstra
adjacency = defaultdict(list)
for src, dst, dist in edges:
    adjacency[src].append((dst, dist))
    adjacency[dst].append((src, dist))

# Initialize truck
truck = Truck(truck_id=1, capacity=50, location='Dump_site')

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

def dp_min_time_multi_truck_with_procedure(node_capacities, truck_capacity, adjacency, dump_site, num_trucks=3):
    mines = [node for node in node_capacities if node != dump_site]
    initial_state = tuple(node_capacities[mine] for mine in mines)

    min_total_time, memo, choice = dp_min_time_multi_truck(node_capacities, truck_capacity, adjacency, dump_site, num_trucks)

    # Reconstruct the optimal procedure
    print(f"--- DP-based Optimal {num_trucks}-Truck Procedure ---")
    print(f"Minimum makespan: {min_total_time}s")

    state = initial_state
    truck_times = tuple([0] * num_trucks)
    truck_locations = tuple([dump_site] * num_trucks)

    step = 1
    cumulative_times = [0] * num_trucks

    while not all(coal == 0 for coal in state):
        key = (state, truck_times, truck_locations)
        assignment = choice[key]

        if assignment is None:
            break

        truck_id, order, mines_order, route, trip_time = assignment

        print(f"\nTrip {step}: Truck {truck_id + 1}")
        print(f"Route: {' -> '.join(route)} (Trip time: {trip_time}s)")

        # Update state and truck info
        new_state = list(state)
        remaining_capacity = truck_capacity
        collected_this_trip = 0

        current_location = truck_locations[truck_id]

        # Detailed step-by-step for this trip
        for idx, mine in enumerate(mines_order):
            t, path = dijkstra(adjacency, current_location, mine)
            print(f"  {current_location} -> {mine}: Time Taken: {t}s")

            take = min(new_state[order[idx]], remaining_capacity)
            print(f"  Truck loaded {take}kg coal at {mine}")
            print(f"  Loading time at {mine}: {LOAD_UNLOAD_TIME}s")

            remaining_capacity -= take
            new_state[order[idx]] -= take
            collected_this_trip += take
            current_location = mine

        # Return to dump site
        t, path = dijkstra(adjacency, current_location, dump_site)
        print(f"  {current_location} -> {' -> '.join(path[1:])}: Time Taken: {t}s")
        print(f"  Truck unloaded at {dump_site}: Collected: {collected_this_trip}kg")
        print(f"  Unloading time at {dump_site}: {LOAD_UNLOAD_TIME}s")

        # Update truck times and locations
        new_truck_times = list(truck_times)
        new_truck_locations = list(truck_locations)
        new_truck_times[truck_id] += trip_time
        new_truck_locations[truck_id] = dump_site
        cumulative_times[truck_id] = new_truck_times[truck_id]

        print(f"  Truck {truck_id + 1} completion time: {cumulative_times[truck_id]}s")

        state = tuple(new_state)
        truck_times = tuple(new_truck_times)
        truck_locations = tuple(new_truck_locations)
        step += 1

    # Final summary
    print(f"\n--- Final Summary ---")
    for truck_id in range(num_trucks):
        print(f"Truck {truck_id + 1} finished at: {cumulative_times[truck_id]}s")

    print(f"Overall makespan: {max(cumulative_times)}s")
    print(f"All mines depleted. Total trips: {step-1}.")

    return min_total_time

import time

def print_multi_truck_table(truck_states, current_time):
    """Display progress for multiple trucks with current time"""
    bar_length = 22
    print('\033[H\033[J', end='')  # Clear screen

    for truck_id, state in enumerate(truck_states):
        status = state['status']
        progress = state['progress']
        percent = state['percent']
        capacity = state['capacity']

        bar = '#' * progress + '_' * (bar_length - progress)

        table = f"""
+----------------------+----------------------+
| Truck {truck_id + 1:<3}                                   |
| Status: {status:<22}              |
| Progress: {percent:3}% [{bar}]     |
| Current Capacity: {capacity:<5}Kg                   |
+----------------------+----------------------+
"""
        print(table)

    # Display current time at the bottom
    print(f"\nCurrent Simulation Time: {current_time}s")

def realtime_multi_truck_progress(node_capacities, truck_capacity, adjacency, dump_site, num_trucks=3):
    """Real-time parallel simulation for multiple trucks"""
    mines = [node for node in node_capacities if node != dump_site]
    initial_state = tuple(node_capacities[mine] for mine in mines)

    # Get optimal solution first
    min_total_time, memo, choice = dp_min_time_multi_truck(node_capacities, truck_capacity, adjacency, dump_site, num_trucks)

    # Reconstruct all trips for each truck
    state = initial_state
    truck_times = [0] * num_trucks
    truck_locations = [dump_site] * num_trucks
    truck_schedules = [[] for _ in range(num_trucks)]  # List of actions per truck

    while not all(coal == 0 for coal in state):
        key = (tuple(state), tuple(truck_times), tuple(truck_locations))
        assignment = choice.get(key)
        if assignment is None:
            break

        truck_id, order, mines_order, route, trip_time = assignment

        # Build detailed action list for this trip
        actions = []
        current_location = truck_locations[truck_id]
        for idx, mine in enumerate(mines_order):
            t, _ = dijkstra(adjacency, current_location, mine)
            actions.append({'type': 'travel', 'to': mine, 'duration': t})
            actions.append({'type': 'load', 'at': mine, 'duration': LOAD_UNLOAD_TIME})
            current_location = mine
        t, _ = dijkstra(adjacency, current_location, dump_site)
        actions.append({'type': 'travel', 'to': dump_site, 'duration': t})
        actions.append({'type': 'unload', 'at': dump_site, 'duration': LOAD_UNLOAD_TIME})

        truck_schedules[truck_id].append(actions)

        # Update state (as before)
        new_state = list(state)
        remaining_capacity = truck_capacity
        for i in order:
            take = min(new_state[i], remaining_capacity)
            remaining_capacity -= take
            new_state[i] -= take
        state = tuple(new_state)

        # Update truck time and location (for reconstruction only)
        truck_times[truck_id] += trip_time
        truck_locations[truck_id] = dump_site

    # Now simulate the timeline
    global_time = 0
    truck_states = [{
        'status': 'Waiting...',
        'progress': 0,
        'percent': 0,
        'capacity': truck_capacity,
        'current_trip': 0,
        'current_action': 0,
        'action_time_left': 0,
        'loaded': 0
    } for _ in range(num_trucks)]

    print('\n' * (num_trucks * 7 + 2))  # Space for tables + time

    all_done = False
    while not all_done:
        all_done = True
        for truck_id in range(num_trucks):
            truck = truck_states[truck_id]
            schedule = truck_schedules[truck_id]

            if truck['current_trip'] >= len(schedule):
                truck['status'] = 'Idle'
                truck['progress'] = 22
                truck['percent'] = 100
                continue

            all_done = False
            actions = schedule[truck['current_trip']]

            if truck['action_time_left'] <= 0:
                # Start next action
                action = actions[truck['current_action']]
                truck['action_time_left'] = action['duration']
                if action['type'] == 'travel':
                    truck['status'] = f"Travel to {action['to']}"
                elif action['type'] == 'load':
                    truck['status'] = f"Loading at {action['at']}"
                    # Simulate loading (increase loaded)
                    truck['loaded'] = min(truck['loaded'] + 1, truck_capacity)  # Simplified; adjust if needed
                elif action['type'] == 'unload':
                    truck['status'] = f"Unloading at {action['at']}"
                    # Simulate unloading (reset loaded)
                    truck['loaded'] = 0
                truck['progress'] = 0
                truck['percent'] = 0

            # Advance this truck's action by 1 second
            if truck['action_time_left'] > 0:
                truck['action_time_left'] -= 1
                total_duration = actions[truck['current_action']]['duration']
                elapsed = total_duration - truck['action_time_left']
                truck['progress'] = int((elapsed / total_duration) * 22)
                truck['percent'] = int((elapsed / total_duration) * 100)
                truck['capacity'] = truck_capacity - truck['loaded']  # Remaining capacity

                if truck['action_time_left'] == 0:
                    # Action complete, move to next
                    truck['current_action'] += 1
                    if truck['current_action'] >= len(actions):
                        # Trip complete, move to next trip
                        truck['current_trip'] += 1
                        truck['current_action'] = 0

        # Update display
        print_multi_truck_table(truck_states, global_time)
        time.sleep(1)  # Real-time delay

        global_time += 1
        if global_time >= min_total_time:
            all_done = True

    # Final display
    final_states = [{
        'status': 'All mines depleted',
        'progress': 22,
        'percent': 100,
        'capacity': truck_capacity
    } for _ in range(num_trucks)]
    print_multi_truck_table(final_states, global_time)
    print(f"\nAll mines depleted. Total makespan: {min_total_time}s.")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Direct solution (minimal time, full procedure)")
    print("2. Realtime progress (step-by-step visualization)")
    print("\nChoose number of trucks:")

    mode = input("Enter 1 for Direct solution or 2 for Realtime progress: ").strip()
    num_trucks = int(input("Enter number of trucks (e.g., 1 for single truck, 3 for multi-truck): "))

    dp_node_capacities = copy.deepcopy(node_capacities)

    if mode == "1":
        dp_min_time_multi_truck_with_procedure(dp_node_capacities, truck.capacity, adjacency, 'Dump_site', num_trucks)
    elif mode == "2":
        print(f"--- Realtime progress mode selected ({num_trucks} trucks) ---")
        realtime_multi_truck_progress(dp_node_capacities, truck.capacity, adjacency, 'Dump_site', num_trucks)
    else:
        print("Invalid option. Please run again and select 1 or 2.")

