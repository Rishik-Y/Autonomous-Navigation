import pygame
import numpy as np
import math
import random
import pickle
import os

# --- Module Imports ---
import map_data
from config import *
from utils import Path, KalmanFilter, a_star_pathfinding
from car import Car
from dispatcher import Dispatcher
from graphics import grid_to_screen, screen_to_grid, draw_road_network, draw_active_path

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
            print(f"!!! Warning: Could not find waypoint segment for {seg_start} -> {seg_end}")

    if final_waypoints:
        final_waypoints.append(map_data.NODES[route_node_names[-1]])
    return final_waypoints

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pro Trucker Fleet - Modular Architecture")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # --- Load Trucks Count ---
    truck_count = 3
    if os.path.exists("truck.txt"):
        try:
            with open("truck.txt", "r") as f:
                truck_count = int(f.read().strip())
        except ValueError:
            print("Invalid truck.txt, defaulting to 3.")
    print(f"Starting simulation with {truck_count} trucks.")

    # --- Load Pre-computed Data ---
    # NOTE: Ensure these paths are correct for your machine
    waypoints_filepath = 'waypoints.pkl'
    cache_filename = r'C:\Users\DAIICT A\Downloads\Sub\Using_MAp\Map_code6\map_cache.pkl'

    if not os.path.exists(waypoints_filepath):
        print(f"Error: '{waypoints_filepath}' not found.")
        return
    with open(waypoints_filepath, 'rb') as f:
        waypoints_map = pickle.load(f)

    if not os.path.exists(cache_filename):
        print(f"Error: '{cache_filename}' not found.")
        return
    with open(cache_filename, 'rb') as f:
        road_graph = pickle.load(f)['road_graph']

    # --- Initialize Components ---
    dispatcher = Dispatcher(road_graph)
    cars = []
    kfs = []

    # Spawn trucks
    start_nodes = list(map_data.DUMP_ZONES) 
    random.shuffle(start_nodes)

    for i in range(truck_count):
        start_node = start_nodes[i % len(start_nodes)]
        target_node = dispatcher.assign_task(Car(i, 0, 0)) 
        
        route = a_star_pathfinding(road_graph, start_node, target_node)
        if not route: continue
        
        wp = get_path_from_nodes(route, waypoints_map)
        if not wp or len(wp) < 2: continue

        pos = wp[0]
        angle = math.atan2(wp[1][1] - wp[0][1], wp[1][0] - wp[0][0])
        
        new_car = Car(i + 1, pos[0], pos[1], angle)
        new_car.path = Path(wp)
        new_car.current_node_name = start_node
        new_car.target_node_name = target_node
        new_car.op_state = "GOING_TO_ENDPOINT" 

        cars.append(new_car)
        kfs.append(KalmanFilter(dt=1.0/60.0, start_x=pos[0], start_y=pos[1]))

    # Map View Setup
    all_nodes_m = list(map_data.NODES.values())
    min_x_m, max_x_m = min(p[0] for p in all_nodes_m), max(p[0] for p in all_nodes_m)
    min_y_m, max_y_m = min(p[1] for p in all_nodes_m), max(p[1] for p in all_nodes_m)
    map_w_m, map_h_m = max(1.0, max_x_m - min_x_m), max(1.0, max_y_m - min_y_m)
    scale = min((WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS))
    pan = [PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale)]
    mouse_dragging, last_mouse_pos = False, None
    selected_car_idx = 0

    # --- Main Loop ---
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue

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

        # Update and Draw All Cars
        for idx, car in enumerate(cars):
            kf = kfs[idx]
            est_pos_m = np.array([kf.x[0], kf.x[2]])
            est_vel_m = np.array([kf.x[1], kf.x[3]])
            est_speed_ms = np.linalg.norm(est_vel_m)
            if car.path:
                car.s_path_m = car.path.project(est_pos_m, car.s_path_m)

            direction, base_speed_ms = car.update_op_state(dt)
            
            if direction == 0 and car.op_timer <= 0:
                if car.op_state == "RETURNING_TO_START":
                    dispatcher.record_load(car.target_node_name)

                car.current_node_name = car.target_node_name
                new_target = dispatcher.assign_task(car)
                car.target_node_name = new_target
                print(f"Car {car.id}: Routing {car.current_node_name} -> {new_target}")
                
                route_node_names = a_star_pathfinding(road_graph, car.current_node_name, new_target)
                if route_node_names:
                    waypoints_m = get_path_from_nodes(route_node_names, waypoints_map)
                    car.path = Path(waypoints_m)
                    car.s_path_m = car.path.project(est_pos_m, 0.0)
                else:
                    print(f"Car {car.id}: No route found!")

            accel_cmd = 0.0
            steer_input = 0.0
            
            if direction != 0 and car.path:
                ld = np.clip(est_speed_ms * LOOKAHEAD_GAIN, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)
                s_target_steer = car.s_path_m + direction * ld
                p_target = car.path.point_at(s_target_steer)
                
                dx_local = (p_target[0] - est_pos_m[0]) * math.cos(car.angle) + (p_target[1] - est_pos_m[1]) * math.sin(car.angle)
                dy_local = -(p_target[0] - est_pos_m[0]) * math.sin(car.angle) + (p_target[1] - est_pos_m[1]) * math.cos(car.angle)
                alpha = math.atan2(dy_local, max(dx_local, 1e-3))
                steer_input = np.clip(math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld), -STEER_MAX_RAD, STEER_MAX_RAD)

                lookahead_dist_speed = 15.0
                max_curvature = 0.0
                for step in np.linspace(0, lookahead_dist_speed, 10):
                    s_check = car.s_path_m + direction * step
                    if s_check > car.path.length or s_check < 0: break
                    max_curvature = max(max_curvature, car.path.get_curvature_at(s_check))
                
                v_turn_cap = math.sqrt(MAX_LAT_ACCEL / max(max_curvature, 1e-4)) if max_curvature > 1e-4 else base_speed_ms
                dist_to_target = abs(car.path.length - car.s_path_m)
                v_stop_cap = math.sqrt(2 * MAX_BRAKE_DECEL * max(0, dist_to_target - 2.0))

                car.desired_speed_ms = min(base_speed_ms, v_turn_cap, v_stop_cap)
                if car.check_collision(cars):
                    car.desired_speed_ms = 0.0
                
                speed_error_ms = car.desired_speed_ms - est_speed_ms
                accel_cmd = np.clip(speed_error_ms / 0.4, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
            else:
                 car.desired_speed_ms = 0.0
                 speed_error_ms = 0.0 - est_speed_ms
                 accel_cmd = np.clip(speed_error_ms / 0.4, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)

            car.move(accel_cmd, steer_input, dt)
            accel_vec_m = np.array([car.accel_ms2 * math.cos(car.angle), car.accel_ms2 * math.sin(car.angle)])
            kf.predict(u=accel_vec_m)
            kf.update(z=car.get_noisy_measurement())

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
                f"Target: {sel_car.target_node_name}",
                f"Active Trucks: {len(cars)}"
            ]
            for i, text in enumerate(hud_texts):
                screen.blit(font.render(text, True, (0, 0, 0)), (10, 10 + i * 22))

        pygame.display.flip()
    pygame.quit()

if __name__ == '__main__':
    run_simulation()