import pygame
import numpy as np
import math
import random
import pickle
import os
import json

# --- Module Imports ---
import map_data
from config import *
from utils import Path, KalmanFilter, a_star_pathfinding
from car import Car
from dispatcher import Dispatcher
from graphics import grid_to_screen, screen_to_grid, draw_road_network, draw_active_path
from tooltip_overlay import get_hovered_entity, draw_tooltip

def load_mine_config():
    """Load configuration from mine_config.json, fallback to defaults."""
    config = {
        "truck_count": 5,
        "coal_capacities": {}
    }
    
    config_file = "mine_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                loaded = json.load(f)
                config["truck_count"] = loaded.get("truck_count", 5)
                config["coal_capacities"] = loaded.get("coal_capacities", {})
                print(f"Loaded mine config: {config['truck_count']} trucks, {len(config['coal_capacities'])} mines configured.")
        except Exception as e:
            print(f"Error loading mine config: {e}, using defaults.")
    else:
        # Fallback to truck.txt for backwards compatibility
        if os.path.exists("truck.txt"):
            try:
                with open("truck.txt", "r") as f:
                    config["truck_count"] = int(f.read().strip())
            except ValueError:
                pass
        print(f"No mine_config.json found, using defaults with {config['truck_count']} trucks.")
    
    return config

def get_path_from_nodes(route_node_names, waypoints_map):
    """Stitches together pre-calculated waypoints to form a complete path."""
    final_waypoints = []
    if not route_node_names: return []

    for i in range(len(route_node_names) - 1):
        seg_start, seg_end = route_node_names[i], route_node_names[i+1]
        found_segment = False
        for chain_tuple, waypoints in waypoints_map.items():
            try:
                # Find segment in forward direction
                idx = chain_tuple.index(seg_start)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx+1] == seg_end:
                    start_wp_idx = idx * POINTS_PER_SEGMENT
                    end_wp_idx = (idx + 1) * POINTS_PER_SEGMENT
                    final_waypoints.extend(waypoints[start_wp_idx:end_wp_idx])
                    found_segment = True
                    break
                # Find segment in reverse direction
                idx = chain_tuple.index(seg_end)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx+1] == seg_start:
                    start_wp_idx = idx * POINTS_PER_SEGMENT
                    end_wp_idx = (idx + 1) * POINTS_PER_SEGMENT
                    segment = waypoints[start_wp_idx:end_wp_idx+1]
                    final_waypoints.extend(segment[::-1][:-1]) 
                    found_segment = True
                    break
            except ValueError:
                continue 
        if not found_segment:
            # print(f"!!! Warning: Could not find waypoint segment for {seg_start} -> {seg_end}")
            pass

    if final_waypoints and route_node_names[-1] in map_data.NODES:
        final_waypoints.append(map_data.NODES[route_node_names[-1]])
    return final_waypoints

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pro Trucker Fleet - Advanced Dispatcher + Map Editing Preserved")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # --- Load Mine Configuration ---
    mine_config = load_mine_config()
    truck_count = mine_config["truck_count"]
    coal_capacities = mine_config["coal_capacities"]
    print(f"Starting simulation with {truck_count} trucks.")

    # --- Load Pre-computed Data ---
    waypoints_filepath = 'waypoints.pkl'
    cache_filename = 'map_cache.pkl'
    
    if not os.path.exists(cache_filename):
         print(f"Warning: Local '{cache_filename}' not found. Pathfinding might fail if graph is missing.")

    if not os.path.exists(waypoints_filepath):
        print(f"Error: '{waypoints_filepath}' not found.")
        return
    with open(waypoints_filepath, 'rb') as f:
        waypoints_map = pickle.load(f)

    if not os.path.exists(cache_filename):
        print(f"Error: '{cache_filename}' not found.")
        return
    with open(cache_filename, 'rb') as f:
        cache_data = pickle.load(f)
        road_graph = cache_data.get('road_graph', {})
        route_cache = cache_data.get('route_cache', {})

    # --- RECURSIVE SPUR PATCHING (Fix Dead Ends & Long Cul-de-Sacs) ---
    all_terminals = set(map_data.LOAD_ZONES + map_data.DUMP_ZONES)
    
    # 1. Build Graph Stats
    incoming_map = {} # target -> [sources]
    outgoing_counts = {} # source -> count
    
    for start_node, edges in road_graph.items():
        outgoing_counts[start_node] = len(edges)
        for target, weight in edges:
            if target not in incoming_map: incoming_map[target] = []
            incoming_map[target].append(start_node)

    # 2. Recursive Patch Function
    def trace_back_and_patch(current_node, visited):
        if current_node in visited: return
        visited.add(current_node)

        # Get parent (Assumes spurs have 1 incoming path usually, but we handle the first one found)
        parents = incoming_map.get(current_node, [])
        if not parents: return # Reached a root or disconnected node
        
        parent = parents[0] # Take the first parent (Spurs are usually linear)
        
        # Add Reverse Edge: Current -> Parent
        if current_node not in road_graph: road_graph[current_node] = []
        
        # Check if edge already exists
        has_edge = any(target == parent for target, _ in road_graph[current_node])
        if not has_edge:
            p1 = map_data.NODES[current_node]
            p2 = map_data.NODES[parent]
            dist = float(np.linalg.norm(p1 - p2))
            road_graph[current_node].append((parent, dist))
            print(f"Patched Spur: {current_node} -> {parent}")
            
        # Recurse UP if the parent is a "Spur Node"
        parent_out_degree = outgoing_counts.get(parent, 0)
        parent_in_degree = len(incoming_map.get(parent, []))
        
        is_hub = "hub" in parent or (parent_in_degree > 1 and parent_out_degree > 1)
        
        if not is_hub:
            trace_back_and_patch(parent, visited)

    # 3. Execute Patching
    for zone in all_terminals:
        trace_back_and_patch(zone, set())

    # --- Initialize Components ---
    dispatcher = Dispatcher(road_graph, coal_capacities=coal_capacities)
    cars = []
    kfs = []

    # Spawn trucks
    start_nodes = list(map_data.DUMP_ZONES) 
    random.shuffle(start_nodes)

    # First Pass: Create Trucks (without assigning task yet, or dummy task)
    for i in range(truck_count):
        start_node = start_nodes[i % len(start_nodes)]
        # We temporarily set dummy values; will be fixed by Global Update below
        new_car = Car(i + 1, 0, 0, 0)
        new_car.current_node_name = start_node
        # Position at start node
        if start_node in map_data.NODES:
            new_car.x_m = map_data.NODES[start_node][0]
            new_car.y_m = map_data.NODES[start_node][1]
        
        cars.append(new_car)
        kfs.append(KalmanFilter(dt=1.0/60.0, start_x=new_car.x_m, start_y=new_car.y_m))

    # --- INITIAL GLOBAL OPTIMIZATION (Pre-Assign) ---
    print("Running initial Global Optimization...")
    dispatcher.update_global_plan(cars)

    # Second Pass: Assign Initial Tasks (Now using the Global Queue)
    for idx, car in enumerate(cars):
        target_node = dispatcher.assign_task(car)
        car.target_node_name = target_node
        
        route = a_star_pathfinding(road_graph, car.current_node_name, target_node, cache=route_cache)
        if not route: continue
        
        wp = get_path_from_nodes(route, waypoints_map)
        if not wp or len(wp) < 2: continue

        pos = wp[0]
        angle = math.atan2(wp[1][1] - wp[0][1], wp[1][0] - wp[0][0])
        
        # Update Car with correct path
        car.x_m, car.y_m, car.angle = pos[0], pos[1], angle
        car.path = Path(wp)
        car.op_state = "GOING_TO_ENDPOINT"
        car.desired_speed_ms = SPEED_MS_EMPTY 
        
        # Reset Kalman Filter to match path start
        kfs[idx] = KalmanFilter(dt=1.0/60.0, start_x=pos[0], start_y=pos[1])

    # Map View Setup
    all_nodes_m = list(map_data.NODES.values())
    min_x_m, max_x_m = min(p[0] for p in all_nodes_m), max(p[0] for p in all_nodes_m)
    min_y_m, max_y_m = min(p[1] for p in all_nodes_m), max(p[1] for p in all_nodes_m)
    map_w_m, map_h_m = max(1.0, max_x_m - min_x_m), max(1.0, max_y_m - min_y_m)
    scale = min((WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS))
    pan = [PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale)]
    mouse_dragging, last_mouse_pos = False, None
    selected_car_idx = 0

    # --- Initial MPC run for all trucks ---
    for car in cars:
        car.run_mpc([])

    # --- Timers ---
    mpc_timer = 0.0
    MPC_INTERVAL = 0.1 # 10Hz
    
    traffic_update_timer = 0.0
    TRAFFIC_UPDATE_INTERVAL = 1.0 # 1Hz
    
    global_opt_timer = 0.0
    GLOBAL_OPT_INTERVAL = 30.0 # Run heavy optimization every 30 seconds

    # --- Main Loop ---
    running = True
    while running:
        dt = clock.tick(60) / 1000.0 # Physics runs at ~60Hz
        if dt == 0: continue
        
        mpc_timer += dt
        traffic_update_timer += dt
        global_opt_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    selected_car_idx = (selected_car_idx + 1) % len(cars)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3: mouse_dragging, last_mouse_pos = True, event.pos
                elif event.button in (4, 5):
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos_m = screen_to_grid(event.pos, scale, pan); scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos_m, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]; pan[1] += event.pos[1] - new_screen_pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3: mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0] += dx; pan[1] += dy; last_mouse_pos = event.pos

        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)
        g_to_s.scale = scale
        
        draw_road_network(screen, g_to_s, scale, waypoints_map)

        # --- Traffic Update (1Hz) ---
        if traffic_update_timer >= TRAFFIC_UPDATE_INTERVAL:
            traffic_update_timer = 0.0
            dispatcher.update_traffic_weights(cars)

        # --- Global Optimization Update (0.2Hz) ---
        if global_opt_timer >= GLOBAL_OPT_INTERVAL:
            global_opt_timer = 0.0
            dispatcher.update_global_plan(cars)

        # --- MPC Update Step (10Hz) ---
        if mpc_timer >= MPC_INTERVAL:
            mpc_timer = 0.0 
            
            # Gather all PLANNED trajectories (Sync Step)
            all_trajectories = [c.planned_trajectory for c in cars]

            for i, car in enumerate(cars):
                # Skip MPC during K-turn maneuver
                if car.op_state == "TURNING_AROUND":
                    continue
                # Cooperative Collision Avoidance: Share trajectories
                other_trajectories = [all_trajectories[j] for j in range(len(cars)) if j != i]
                car.run_mpc(other_trajectories, other_cars=cars)

        # --- Physics Update Step (60Hz) ---
        for idx, car in enumerate(cars):
            kf = kfs[idx]
            est_pos_m = np.array([kf.x[0], kf.x[2]])
            est_vel_m = np.array([kf.x[1], kf.x[3]])

            # Project position to path (skip during K-turn)
            if car.path and car.op_state != "TURNING_AROUND":
                car.s_path_m = car.path.project(est_pos_m, car.s_path_m)

            # High-level Logic (State Machine)
            # Pass dispatcher to handle reservations
            direction, base_speed_ms = car.update_op_state(dt, dispatcher)
            car.desired_speed_ms = base_speed_ms 

            if car.needs_new_path:
                car.needs_new_path = False
                car.current_node_name = car.target_node_name
                new_target = dispatcher.assign_task(car)
                car.target_node_name = new_target
                
                print(f"Car {car.id}: Routing {car.current_node_name} -> {new_target}")
                
                # USE DISPATCHER'S WEIGHTED GRAPH AND ROUTE CACHE
                route_node_names = a_star_pathfinding(dispatcher.get_graph(), car.current_node_name, new_target, cache=route_cache)
                
                if route_node_names:
                    waypoints_m = get_path_from_nodes(route_node_names, waypoints_map)
                    if waypoints_m and len(waypoints_m) >= 2:
                        car.path = Path(waypoints_m)
                        car.s_path_m = 0.0
                        
                        # --- K-TURN SETUP: Calculate target heading for maneuver ---
                        car.turn_target_angle = math.atan2(
                            waypoints_m[1][1] - waypoints_m[0][1],
                            waypoints_m[1][0] - waypoints_m[0][0]
                        )
                        car.current_mpc_control = np.zeros(2)
                        car.mpc.prev_u = np.zeros_like(car.mpc.prev_u)
                        
                        print(f"Car {car.id}: K-turn to heading {math.degrees(car.turn_target_angle):.1f}°")
                    else:
                        print(f"Car {car.id}: Path too short!")
                else:
                    print(f"Car {car.id}: No route found!")

            # --- Physics / Maneuver ---
            if car.op_state == "TURNING_AROUND":
                # Execute K-turn maneuver (replaces normal move)
                turn_done = car.execute_turn_step(dt)
                if turn_done:
                    # Turn complete - re-sync path projection and Kalman filter
                    if car.path:
                        car.s_path_m = car.path.project(np.array([car.x_m, car.y_m]), 0.0)
                    kf.x[0] = car.x_m
                    kf.x[1] = 0.0
                    kf.x[2] = car.y_m
                    kf.x[3] = 0.0
                    car.run_mpc([], other_cars=cars)
                    print(f"Car {car.id}: K-turn complete, now {car.op_state}")
            else:
                # --- Hard collision safety (before move) ---
                # Emergency braking if a truck is ahead and dangerously close.
                # This overrides MPC to prevent physical overlap.
                heading_vec = np.array([math.cos(car.angle), math.sin(car.angle)])
                lateral_vec = np.array([-math.sin(car.angle), math.cos(car.angle)])
                for j, other in enumerate(cars):
                    if j == idx or other.op_state in ("LOADING", "UNLOADING", "TURNING_AROUND"):
                        continue
                    diff = np.array([other.x_m - car.x_m, other.y_m - car.y_m])
                    dist = np.linalg.norm(diff)
                    if dist > SAFE_DISTANCE_M * 1.5:
                        continue
                    longitudinal = np.dot(diff, heading_vec)
                    lateral = np.dot(diff, lateral_vec)
                    # Only brake for trucks AHEAD in SAME lane
                    if longitudinal > 0 and abs(lateral) < 4.0:
                        # Proportional emergency brake — harder when closer
                        if dist < SAFE_DISTANCE_M:
                            brake_factor = max(0.0, 1.0 - dist / SAFE_DISTANCE_M)
                            car.current_mpc_control[0] = min(
                                car.current_mpc_control[0],
                                -brake_factor * MAX_BRAKE_DECEL
                            )

                # Normal MPC-controlled movement
                car.move(dt)

            # Kalman Filter
            accel_vec_m = np.array([car.accel_ms2 * math.cos(car.angle), car.accel_ms2 * math.sin(car.angle)])
            kf.predict(u=accel_vec_m)
            kf.update(z=car.get_noisy_measurement())

            # Draw
            if car.path and idx == selected_car_idx:
                 draw_active_path(screen, car.path, g_to_s, scale)
            car.draw(screen, g_to_s, is_selected=(idx == selected_car_idx))

        # HUD
        if cars:
            sel_car = cars[selected_car_idx]
            hud_texts = [
                f"Truck ID: {sel_car.id} (TAB to switch)",
                f"Speed: {sel_car.speed_ms * 3.6:.1f} km/h",
                f"Mass: {sel_car.current_mass_kg:.0f} kg",
                f"State: {sel_car.op_state}",
                f"Dispatcher: Advanced (Swarm Plan)",
                f"Active Trucks: {len(cars)}"
            ]
            for i, text in enumerate(hud_texts):
                screen.blit(font.render(text, True, (0, 0, 0)), (10, 10 + i * 22))

        # --- Hover Tooltips ---
        mouse_pos = pygame.mouse.get_pos()
        hovered = get_hovered_entity(mouse_pos, scale, pan, cars, dispatcher)
        if hovered:
            draw_tooltip(screen, mouse_pos, hovered, font)

        pygame.display.flip()
    pygame.quit()

if __name__ == '__main__':
    run_simulation()
