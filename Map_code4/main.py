import pygame
import numpy as np
import math
import heapq
import random
import pickle
import os

# --- IMPORTS ---
import map_data  # Use the local map_data.py

# --- VISUAL & GAME SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, GRAY, BLUE_ACTIVE, RED, PURPLE_NODE = (255, 255, 255), (100, 100, 100), (0, 100, 200, 150), (255, 0, 0), (150, 0, 150)
CAR_COLOR = (0, 80, 200)
ROAD_WIDTH_M = 4.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS
POINTS_PER_SEGMENT = 20 # Must match the value from Waypoint_Editor.py

# --- Car Physics Parameters (From try12.py) ---
CAR_LENGTH_M = 4.5
CAR_WIDTH_M = 2.0
WHEELBASE_M = 2.8
MASS_KG = 1500.0
CARGO_TON = 1.0
P_MAX_W = 80_000.0
CD = 0.35
FRONTAL_AREA = 2.2
CRR = 0.01
MAX_ACCEL_CMD = 1.5
MAX_BRAKE_DECEL = 1.5
SPEED_KMPH_EMPTY = 35.0
SPEED_KMPH_LOADED = 25.0
SPEED_MS_EMPTY = SPEED_KMPH_EMPTY / 3.6
SPEED_MS_LOADED = SPEED_KMPH_LOADED / 3.6
STEER_MAX_DEG = 35.0
STEER_RATE_DEGPS = 270.0
STEER_MAX_RAD = math.radians(STEER_MAX_DEG)
STEER_RATE_RADPS = math.radians(STEER_RATE_DEGPS)
JERK_LIMIT = 1.0
LOAD_UNLOAD_TIME_S = 3.0

# --- Controller Parameters (From try12.py) ---
LOOKAHEAD_GAIN = 0.8
LOOKAHEAD_MIN_M = 4.0
LOOKAHEAD_MAX_M = 15.0
MAX_LAT_ACCEL = 2.0
SENSOR_NOISE_STD_DEV = 0.5

# --- HELPER FUNCTIONS ---
def a_star_pathfinding(graph: dict, start_name: str, goal_name: str) -> list[str]:
    """Finds the shortest path between two nodes using A* and returns a list of node names."""
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
                # Heuristic (h) is not needed for f_score since we are just looking for the cheapest path
                heapq.heappush(open_set, (tentative_g_score, neighbor_name))
    return []

# --- Physics Functions (FROM try12.py) ---
def resist_forces(v_ms: float, mass_kg: float) -> float:
    return CRR * mass_kg * 9.81 + 0.5 * 1.225 * CD * FRONTAL_AREA * v_ms**2

def traction_force_from_power(v_ms: float, throttle: float) -> float:
    return (P_MAX_W * np.clip(throttle, 0, 1)) / max(v_ms, 0.5)

def brake_force_from_command(brake_cmd: float, mass_kg: float) -> float:
    return np.clip(brake_cmd, 0, 1) * mass_kg * MAX_BRAKE_DECEL

# --- Path Class (FROM try12.py) ---
class Path:
    def __init__(self, waypoints: list[np.ndarray]):
        if not waypoints or len(waypoints) < 2:
            self.wp = [np.array([0,0]), np.array([1,1])] # Create a dummy path
        else:
            self.wp = waypoints
        self.s = [0.0]
        for i in range(1, len(self.wp)):
            dist = np.linalg.norm(self.wp[i] - self.wp[i-1])
            self.s.append(self.s[-1] + dist)
        self.length = self.s[-1]
        if self.length < 1e-6: self.length = 1e-6

    def get_segment_index(self, s: float) -> int:
        s = np.clip(s, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]: return i
        return len(self.s) - 2

    def point_at(self, s_query: float) -> np.ndarray:
        s = np.clip(s_query, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]:
                s_base, s_end = self.s[i], self.s[i+1]
                p_base, p_end = self.wp[i], self.wp[i + 1]
                if s_end - s_base < 1e-6: return p_base
                t = (s - s_base) / (s_end - s_base)
                return p_base + t * (p_end - p_base)
        return self.wp[-1]

    def project(self, p: np.ndarray, s_hint: float) -> float:
        idx_hint = self.get_segment_index(s_hint)
        search_window = 20
        idx_start = max(0, idx_hint - search_window)
        idx_end = min(len(self.wp) - 1, idx_hint + search_window)
        min_dist_sq, best_s = float('inf'), s_hint
        for i in range(idx_start, idx_end):
            a, b = self.wp[i], self.wp[i + 1]
            seg_vec = b - a; ap = p - a; dot_val = np.dot(seg_vec, seg_vec)
            if dot_val < 1e-9: t = 0.0
            else: t = np.clip(np.dot(ap, seg_vec) / dot_val, 0, 1)
            proj = a + t * seg_vec; dist_sq = np.sum((p - proj)**2)
            if dist_sq < min_dist_sq: min_dist_sq, best_s = dist_sq, self.s[i] + t * np.linalg.norm(seg_vec)
        return best_s

    def get_curvature_at(self, s: float) -> float:
        s_before, s_at, s_after = self.point_at(s - 1.0), self.point_at(s), self.point_at(s + 1.0)
        area = 0.5 * abs(s_before[0]*(s_at[1]-s_after[1]) + s_at[0]*(s_after[1]-s_before[1]) + s_after[0]*(s_before[1]-s_at[1]))
        d1, d2, d3 = np.linalg.norm(s_at - s_before), np.linalg.norm(s_after - s_at), np.linalg.norm(s_after - s_before)
        if d1*d2*d3 < 1e-6: return 0.0
        return (4 * area) / (d1 * d2 * d3)

# --- Kalman Filter Class (FROM try12.py) ---
class KalmanFilter:
    def __init__(self, dt, start_x, start_y):
        self.x = np.zeros(4); self.x[0] = start_x; self.x[2] = start_y
        self.dt = dt
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt**2, 0], [dt, 0], [0, 0.5 * dt**2], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * (SENSOR_NOISE_STD_DEV**2)
        self.P = np.eye(4)

    def predict(self, u):
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P
        except np.linalg.LinAlgError:
            pass # Ignore singular matrix errors

# --- Car Class (FROM try12.py) ---
class Car:
    def __init__(self, x_m, y_m, angle=0):
        self.x_m, self.y_m, self.angle = x_m, y_m, angle
        self.speed_ms, self.accel_ms2, self.steer_angle = 0.0, 0.0, 0.0
        self.current_mass_kg = MASS_KG
        self.op_state = "GOING_TO_ENDPOINT"
        self.op_timer = 0.0
        self.a_cmd_prev = 0.0
        self.s_path_m = 0.0
        self.current_node_name = ""
        self.target_node_name = ""

    def move(self, accel_cmd, steer_input, dt):
        accel_cmd = np.clip(accel_cmd, self.a_cmd_prev - JERK_LIMIT * dt, self.a_cmd_prev + JERK_LIMIT * dt)
        self.a_cmd_prev = accel_cmd
        F_resist = resist_forces(self.speed_ms, self.current_mass_kg)
        F_needed = self.current_mass_kg * accel_cmd
        if F_needed >= 0:
            throttle = ((F_needed + F_resist) * max(self.speed_ms, 0.5)) / P_MAX_W
            brake = 0.0
        else:
            brake = (-F_needed + F_resist) / (self.current_mass_kg * MAX_BRAKE_DECEL)
            throttle = 0.0
        self.steer_angle = np.clip(steer_input, self.steer_angle - STEER_RATE_RADPS * dt, self.steer_angle + STEER_RATE_RADPS * dt)
        F_trac = traction_force_from_power(self.speed_ms, throttle)
        F_brake = brake_force_from_command(brake, self.current_mass_kg)
        F_net = F_trac - F_resist - F_brake
        self.accel_ms2 = F_net / self.current_mass_kg
        self.speed_ms = max(0.0, self.speed_ms + self.accel_ms2 * dt)
        if abs(self.steer_angle) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(self.steer_angle)
            angular_velocity = self.speed_ms / turn_radius
            self.angle += angular_velocity * dt
        self.x_m += self.speed_ms * math.cos(self.angle) * dt
        self.y_m += self.speed_ms * math.sin(self.angle) * dt

    def update_op_state(self, dt, path_length_m, current_s_m):
        # This state machine is adapted for a network, not just a simple back-and-forth
        if self.op_state == "GOING_TO_ENDPOINT":
            if current_s_m >= path_length_m - 2.0 and self.speed_ms < 0.5:
                self.op_state = "LOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
                return 0, 0.0 # Signal to stop and get a new path
            return +1, SPEED_MS_EMPTY
        elif self.op_state == "LOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else:
                self.op_state = "RETURNING_TO_START"
                self.current_mass_kg = MASS_KG + CARGO_TON * 1000
            return 0, 0.0
        elif self.op_state == "RETURNING_TO_START":
            if current_s_m >= path_length_m - 2.0 and self.speed_ms < 0.5:
                self.op_state = "UNLOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
                return 0, 0.0 # Signal to stop and get a new path
            return +1, SPEED_MS_LOADED # Always drive forward on the new path
        elif self.op_state == "UNLOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else:
                self.op_state = "GOING_TO_ENDPOINT"
                self.current_mass_kg = MASS_KG
            return 0, 0.0
        return 0, 0.0

    def get_noisy_measurement(self):
        return np.array([self.x_m + np.random.normal(0, SENSOR_NOISE_STD_DEV), self.y_m + np.random.normal(0, SENSOR_NOISE_STD_DEV)])

    def draw(self, screen, g_to_s):
        car_center_screen = g_to_s((self.x_m, self.y_m))
        length_px = CAR_LENGTH_M * METERS_TO_PIXELS * g_to_s.scale
        width_px = CAR_WIDTH_M * METERS_TO_PIXELS * g_to_s.scale
        if length_px < 1 or width_px < 1: return
        car_surface = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
        cargo_ratio = (self.current_mass_kg - MASS_KG) / (CARGO_TON * 1000)
        body_color = tuple(np.array(CAR_COLOR) * (1 - cargo_ratio) + np.array([120, 120, 120]) * cargo_ratio)
        car_surface.fill(body_color)
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        rect = rotated_surface.get_rect(center=car_center_screen)
        screen.blit(rotated_surface, rect.topleft)

# --- Drawing Functions ---
def grid_to_screen(pos_m, scale, pan):
    pos_m_np = np.array(pos_m)
    pos_px = pos_m_np * METERS_TO_PIXELS
    return (int(pos_px[0] * scale + pan[0]), int(pos_px[1] * scale + pan[1]))

def screen_to_grid(pos_px, scale, pan):
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return (grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS)

def draw_road_network(screen, g_to_s, scale, waypoints_map):
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    for waypoints in waypoints_map.values():
        if len(waypoints) < 2: continue
        road_px = [g_to_s(p) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    
    for node_name, pos_m in map_data.NODES.items():
        if node_name in map_data.LOAD_ZONES: color = (0, 200, 0)
        elif node_name in map_data.DUMP_ZONES: color = (200, 0, 0)
        else: color = PURPLE_NODE
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(2, int(scale * 4)))

def draw_active_path(screen, path: Path, g_to_s, scale):
    if len(path.wp) < 2: return
    road_px = [g_to_s(p) for p in path.wp]
    if len(road_px) > 1:
        road_width_px = max(2, int((ROAD_WIDTH_M - 1.0) * METERS_TO_PIXELS * scale))
        pygame.draw.lines(screen, BLUE_ACTIVE, False, road_px, road_width_px)

def get_path_from_nodes(route_node_names, waypoints_map):
    """Stitches together pre-calculated waypoints to form a complete path."""
    final_waypoints = []
    if not route_node_names:
        return []

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
                    # Add segment, excluding the last point if it's not the final segment
                    final_waypoints.extend(waypoints[start_wp_idx:end_wp_idx])
                    found_segment = True
                    break
                
                # Find segment in reverse direction
                idx = chain_tuple.index(seg_end)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx+1] == seg_start:
                    start_wp_idx = idx * POINTS_PER_SEGMENT
                    end_wp_idx = (idx + 1) * POINTS_PER_SEGMENT
                    # Get segment and reverse it
                    segment = waypoints[start_wp_idx:end_wp_idx+1]
                    final_waypoints.extend(segment[::-1][:-1]) # Exclude last point after reversing
                    found_segment = True
                    break

            except ValueError:
                continue # Node not in this chain
        
        if not found_segment:
            print(f"!!! Warning: Could not find waypoint segment for {seg_start} -> {seg_end}")

    # Add the very last point of the entire path
    if final_waypoints:
        final_waypoints.append(map_data.NODES[route_node_names[-1]])

    return final_waypoints

# --- Main Simulation ---
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pro Trucker with Pre-calculated Waypoints")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # --- Load Pre-computed Data ---
    waypoints_filepath = 'waypoints.pkl'
    if not os.path.exists(waypoints_filepath):
        print(f"Error: Waypoint file '{waypoints_filepath}' not found.")
        print(f"Please run 'python Waypoint_Editor.py', press 'A' then 'S' to generate the file.")
        return
        
    with open(waypoints_filepath, 'rb') as f:
        waypoints_map = pickle.load(f)
    print(f"Loaded {len(waypoints_map)} pre-calculated road paths.")

    cache_filename = 'map_cache.pkl'
    if not os.path.exists(cache_filename):
        print(f"Error: Map cache file '{cache_filename}' not found.")
        print(f"Please run 'python map_data.py' to generate the cache file first.")
        return

    with open(cache_filename, 'rb') as f:
        road_graph = pickle.load(f)['road_graph']

    # --- Create the first path ---
    start_node_name = random.choice(map_data.DUMP_ZONES)
    goal_node_name = random.choice(map_data.LOAD_ZONES)
    route_node_names = a_star_pathfinding(road_graph, start_node_name, goal_node_name)
    
    if not route_node_names:
        print(f"Error: A* could not find an initial path from {start_node_name} to {goal_node_name}. Exiting.")
        return

    waypoints_m = get_path_from_nodes(route_node_names, waypoints_map)
    path = Path(waypoints_m)

    # --- Setup Car, KF, and View ---
    if not waypoints_m or len(waypoints_m) < 2:
        print("Error: Could not generate valid waypoints for the initial path. Exiting.")
        return
        
    initial_pos = waypoints_m[0]
    initial_angle = math.atan2(waypoints_m[1][1] - waypoints_m[0][1], waypoints_m[1][0] - waypoints_m[0][0])
    car = Car(initial_pos[0], initial_pos[1], angle=initial_angle)
    car.current_node_name = start_node_name
    car.target_node_name = goal_node_name

    kf = KalmanFilter(dt=1.0/60.0, start_x=car.x_m, start_y=car.y_m)

    all_nodes_m = list(map_data.NODES.values())
    min_x_m, max_x_m = min(p[0] for p in all_nodes_m), max(p[0] for p in all_nodes_m)
    min_y_m, max_y_m = min(p[1] for p in all_nodes_m), max(p[1] for p in all_nodes_m)
    map_w_m, map_h_m = max(1.0, max_x_m - min_x_m), max(1.0, max_y_m - min_y_m)
    scale = min((WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS))
    pan = [PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale)]
    mouse_dragging, last_mouse_pos = False, None

    # --- Main Loop ---
    running = True
    desired_speed_ms = 0.0 # Initialize
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3: mouse_dragging, last_mouse_pos = True, event.pos # Right-click
                elif event.button in (4, 5):
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos_m = screen_to_grid(event.pos, scale, pan); scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos_m, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]; pan[1] += event.pos[1] - new_screen_pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3: mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0] += dx; pan[1] += dy; last_mouse_pos = event.pos

        # --- State Estimation ---
        est_pos_m = np.array([kf.x[0], kf.x[2]])
        est_vel_m = np.array([kf.x[1], kf.x[3]])
        est_speed_ms = np.linalg.norm(est_vel_m)
        car.s_path_m = path.project(est_pos_m, car.s_path_m)

        # --- State Machine & Rerouting ---
        direction, base_speed_ms = car.update_op_state(dt, path.length, car.s_path_m)

        if direction == 0 and car.op_timer <= 0: # Time to get a new path
            prev_state = car.op_state
            
            if prev_state == "LOADING":
                car.current_node_name = car.target_node_name
                new_target = random.choice(map_data.DUMP_ZONES)
                car.target_node_name = new_target
                
            elif prev_state == "UNLOADING":
                car.current_node_name = car.target_node_name
                new_target = random.choice(map_data.LOAD_ZONES)
                car.target_node_name = new_target

            else: # Should not happen, but as a fallback
                new_target = random.choice(map_data.LOAD_ZONES)

            print(f"New route: {car.current_node_name} -> {new_target}")
            route_node_names = a_star_pathfinding(road_graph, car.current_node_name, new_target)
            if route_node_names:
                waypoints_m = get_path_from_nodes(route_node_names, waypoints_map)
                path = Path(waypoints_m)
                car.s_path_m = path.project(est_pos_m, 0.0) # Project onto new path
            else:
                print(f"!!! A* failed to find path from {car.current_node_name} to {new_target}")
                pass

        # --- Controller Logic ---
        if direction != 0:
            ld = np.clip(est_speed_ms * LOOKAHEAD_GAIN, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)
            s_target_steer = car.s_path_m + direction * ld
            p_target = path.point_at(s_target_steer)
            dx_local = (p_target[0] - est_pos_m[0]) * math.cos(car.angle) + (p_target[1] - est_pos_m[1]) * math.sin(car.angle)
            dy_local = -(p_target[0] - est_pos_m[0]) * math.sin(car.angle) + (p_target[1] - est_pos_m[1]) * math.cos(car.angle)
            alpha = math.atan2(dy_local, max(dx_local, 1e-3))
            steer_input = np.clip(math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld), -STEER_MAX_RAD, STEER_MAX_RAD)

            lookahead_dist_speed = 15.0
            max_curvature = 0.0
            for step in np.linspace(0, lookahead_dist_speed, 10):
                s_check = car.s_path_m + direction * step
                if s_check > path.length or s_check < 0: break
                max_curvature = max(max_curvature, path.get_curvature_at(s_check))
            v_turn_cap = math.sqrt(MAX_LAT_ACCEL / max(max_curvature, 1e-4)) if max_curvature > 1e-4 else base_speed_ms
            
            dist_to_target = abs(path.length - car.s_path_m)
            v_stop_cap = math.sqrt(2 * MAX_BRAKE_DECEL * max(0, dist_to_target - 2.0))

            desired_speed_ms = min(base_speed_ms, v_turn_cap, v_stop_cap)
            speed_error_ms = desired_speed_ms - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / 0.4, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
        else:
            steer_input = 0.0
            speed_error_ms = 0.0 - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / 0.4, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)

        # --- Update Physics and KF ---
        car.move(accel_cmd, steer_input, dt)
        accel_vec_m = np.array([car.accel_ms2 * math.cos(car.angle), car.accel_ms2 * math.sin(car.angle)])
        kf.predict(u=accel_vec_m)
        kf.update(z=car.get_noisy_measurement())

        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)
        g_to_s.scale = scale
        
        draw_road_network(screen, g_to_s, scale, waypoints_map)
        draw_active_path(screen, path, g_to_s, scale)
        car.draw(screen, g_to_s)

        kf_pos_m = (kf.x[0], kf.x[2])
        pygame.draw.circle(screen, RED, g_to_s(kf_pos_m), 6, 2)

        # HUD
        hud_texts = [
            f"Speed: {car.speed_ms * 3.6:.1f} km/h (Target: {desired_speed_ms*3.6:.1f})",
            f"Mass: {car.current_mass_kg:.0f} kg",
            f"State: {car.op_state}",
            f"Accel: {car.accel_ms2:.2f} m/s² (Cmd: {car.a_cmd_prev:.2f})",
            f"Steer: {math.degrees(car.steer_angle):.1f} deg",
            f"Path: {car.s_path_m:.1f}m / {path.length:.1f}m"
        ]

        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 22))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_simulation()
