import pygame
import numpy as np
import math
import heapq
import random

# --- IMPORTS ---
from truck_system import Truck, Path 
import map_data as map 

# --- VISUAL & GAME SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, GRAY, BLUE_ACTIVE, RED, PURPLE_NODE = (255, 255, 255), (100, 100, 100), (0, 100, 200, 150), (255, 0, 0), (150, 0, 150)
ROAD_WIDTH_M = 4.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 3.0
PIXELS_TO_METERS = 1.0 / 3.0
SMOOTH_ITERATIONS = 5

# --- SIMULATION SETTINGS ---
NUM_PLATOONS = 2     # 20 groups
TRUCKS_PER_PLATOON = 5 # 5 trucks per group = 100 trucks total

# --- CONTROLLER TUNING ---
LOOKAHEAD_GAIN = 1.2
LOOKAHEAD_MIN_M = 4.0
LOOKAHEAD_MAX_M = 35.0
MAX_LAT_ACCEL = 4.5
MAX_BRAKE_DECEL = 1.5 
MAX_ACCEL_CMD = 1.5

# --- GLOBAL CACHE ---
PRE_CALCULATED_SPLINES = [] 

# --- HELPER FUNCTIONS (Unchanged) ---
def catmull_rom_point(t, p0, p1, p2, p3):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t**2) + (-p0 + 3 * p1 - 3 * p2 + p3) * (t**3))
def generate_curvy_path_from_nodes(node_list: list[np.ndarray], points_per_segment=20) -> list[np.ndarray]:
    all_waypoints_m = []
    if not node_list or len(node_list) < 2: return []
    for i in range(len(node_list) - 1):
        p0 = node_list[0] if i == 0 else node_list[i - 1]
        p1 = node_list[i]
        p2 = node_list[i + 1]
        p3 = node_list[-1] if i >= len(node_list) - 2 else node_list[i + 2]
        if i == 0: all_waypoints_m.append(p1)
        for j in range(1, points_per_segment + 1):
            t = j / float(points_per_segment)
            point = catmull_rom_point(t, p0, p1, p2, p3)
            all_waypoints_m.append(point)
    return all_waypoints_m
def chaikin_smoother(points, iterations):
    for _ in range(iterations):
        new_points = []; new_points.append(points[0])
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]; q = (1 - 0.25) * p1 + 0.25 * p2; r = 0.25 * p1 + (1 - 0.25) * p2
            new_points.extend([q, r])
        new_points.append(points[-1]); points = new_points
    return points
def build_weighted_graph(nodes: dict, edges: list) -> dict:
    graph = {name: [] for name in nodes}
    for n1_name, n2_name in edges:
        p1 = nodes[n1_name]
        p2 = nodes[n2_name]
        distance = np.linalg.norm(p1 - p2)
        graph[n1_name].append((n2_name, distance))
        graph[n2_name].append((n1_name, distance))
    return graph
def a_star_pathfinding(graph: dict, nodes_coords: dict, start_name: str, goal_name: str) -> list[np.ndarray]:
    def h(node_name): return np.linalg.norm(nodes_coords[node_name] - nodes_coords[goal_name])
    open_set = [(h(start_name), start_name)]
    came_from = {}
    g_score = {name: float('inf') for name in nodes_coords}; g_score[start_name] = 0
    f_score = {name: float('inf') for name in nodes_coords}; f_score[start_name] = h(start_name)
    while open_set:
        current_f, current_name = heapq.heappop(open_set)
        if current_name == goal_name:
            path_nodes = []
            temp = current_name
            while temp in came_from:
                path_nodes.append(nodes_coords[temp])
                temp = came_from[temp]
            path_nodes.append(nodes_coords[start_name])
            return list(reversed(path_nodes))
        for neighbor_name, weight in graph[current_name]:
            tentative_g_score = g_score[current_name] + weight
            if tentative_g_score < g_score[neighbor_name]:
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative_g_score
                f_score[neighbor_name] = tentative_g_score + h(neighbor_name)
                if (f_score[neighbor_name], neighbor_name) not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor_name], neighbor_name))
    return []

# --- DRAWING ---
def grid_to_screen(pos_m, scale, pan):  
    pos_m_np = np.array(pos_m); pos_px = pos_m_np * METERS_TO_PIXELS
    return (int(pos_px[0] * scale + pan[0]), int(pos_px[1] * scale + pan[1]))
def screen_to_grid(pos_px, scale, pan):  
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return (grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS)
def draw_road_network(screen, g_to_s, scale):
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    for waypoints in PRE_CALCULATED_SPLINES:
        if len(waypoints) < 2: continue
        road_px = [g_to_s(p) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    for node_name, pos_m in map.NODES.items():
        if node_name in map.LOAD_ZONES: color = (0, 200, 0)
        elif node_name in map.DUMP_ZONES: color = (200, 0, 0) 
        else: color = PURPLE_NODE
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(2, int(scale)))
def draw_active_path(screen, path: Path, g_to_s, scale):
    if len(path.wp) < 2: return
    smoothed_points_m = chaikin_smoother(path.wp, SMOOTH_ITERATIONS); road_px = [g_to_s(p) for p in smoothed_points_m]
    if len(road_px) > 1:
        road_width_px = max(2, int((ROAD_WIDTH_M + 0.5) * METERS_TO_PIXELS * scale))
        pygame.draw.lines(screen, BLUE_ACTIVE, False, road_px, road_width_px)

# --- PLATOON MANAGEMENT ---
class Platoon:
    def __init__(self, road_graph):
        self.trucks = [] # [Leader, Follower1, Follower2...]
        self.path = None
        self.graph = road_graph
        self.spawn()
        
    def spawn(self):
        # Pick start/end for Leader
        start_node = random.choice(map.DUMP_ZONES)
        target_node = random.choice(map.LOAD_ZONES)
        
        # Calculate Path
        route_nodes = a_star_pathfinding(self.graph, map.NODES, start_node, target_node)
        if not route_nodes: return
        waypoints_m = generate_curvy_path_from_nodes(route_nodes)
        self.path = Path(waypoints_m)
        
        # Spawn Leader
        start_pos = map.NODES[start_node]
        angle = 0.0
        if len(waypoints_m) > 1:
            angle = math.atan2(waypoints_m[1][1] - waypoints_m[0][1], waypoints_m[1][0] - waypoints_m[0][0])
            
        leader = Truck(start_pos[0], start_pos[1], angle=angle, start_node=start_node)
        leader.target_node_name = target_node
        leader.is_leader = True
        self.trucks.append(leader)
        
        # Spawn Followers
        for i in range(TRUCKS_PER_PLATOON - 1):
            # Spawn slightly behind the previous truck
            prev_truck = self.trucks[-1]
            offset_x = -10.0 * math.cos(angle)
            offset_y = -10.0 * math.sin(angle)
            
            follower = Truck(prev_truck.x_m + offset_x, prev_truck.y_m + offset_y, angle=angle, start_node=start_node)
            follower.is_leader = False
            follower.leader_truck = leader
            follower.front_truck = prev_truck # Follow the guy in front
            self.trucks.append(follower)

    def update_logic(self, dt, road_graph):
        leader = self.trucks[0]
        
        # --- LEADER LOGIC (Pathfinding) ---
        est_pos_m = np.array([leader.kf.x[0], leader.kf.x[2]])
        est_vel_m = np.array([leader.kf.x[1], leader.kf.x[3]])
        leader.s_path_m = self.path.project(est_pos_m, leader.s_path_m)
        
        prev_op_state = leader.op_state
        direction, base_speed_ms = leader.update_op_state(dt, self.path.length, leader.s_path_m)
        
        # Rerouting (Only Leader decides)
        if leader.op_state != prev_op_state:
            new_target = None
            if leader.op_state == "RETURNING_TO_START":
                leader.current_node_name = leader.target_node_name
                new_target = random.choice(map.DUMP_ZONES)
            elif leader.op_state == "GOING_TO_ENDPOINT":
                leader.current_node_name = leader.target_node_name
                new_target = random.choice(map.LOAD_ZONES)
            
            if new_target:
                leader.target_node_name = new_target
                route_nodes = a_star_pathfinding(road_graph, map.NODES, leader.current_node_name, new_target)
                if route_nodes:
                    # Update Path Object Shared by ALL
                    waypoints_m = generate_curvy_path_from_nodes(route_nodes) if leader.op_state == "GOING_TO_ENDPOINT" else generate_curvy_path_from_nodes(list(reversed(route_nodes)))
                    self.path = Path(waypoints_m) 
                    
                    # Reset Leader S
                    leader.s_path_m = self.path.project(est_pos_m, 0.0 if direction == 1 else self.path.length)
                    
                    # Followers just reset their S tracking, they don't need new path gen
                    for t in self.trucks[1:]:
                        t.s_path_m = leader.s_path_m - 10.0 # Rough estimation

        # --- FOLLOWER LOGIC (Copy Leader's Path Data) ---
        for t in self.trucks[1:]:
             # Followers just project onto the Leader's path object
             f_pos = np.array([t.kf.x[0], t.kf.x[2]])
             t.s_path_m = self.path.project(f_pos, t.s_path_m)
             
        return direction, base_speed_ms

# --- MAIN SIMULATION ---
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE) 
    pygame.display.set_caption(f"Pro Driver: {NUM_PLATOONS*TRUCKS_PER_PLATOON} Trucks (Leader-Follower)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # 1. Build Graph
    print("Building road graph...")
    road_graph = build_weighted_graph(map.NODES, map.EDGES)
    print("Pre-calculating visuals...")
    for chain in map.VISUAL_ROAD_CHAINS:
        node_coords_list = [map.NODES[node_name] for node_name in chain if node_name in map.NODES]
        if len(node_coords_list) < 2: continue
        spline_waypoints = generate_curvy_path_from_nodes(node_coords_list)
        PRE_CALCULATED_SPLINES.append(spline_waypoints)
    
    # 2. Initialize Platoons
    platoons = []
    print(f"Spawning {NUM_PLATOONS} platoons...")
    for _ in range(NUM_PLATOONS):
        p = Platoon(road_graph)
        if p.trucks: platoons.append(p)

    # 3. View Setup
    all_nodes_m = list(map.NODES.values())
    min_x_m, max_x_m = min(p[0] for p in all_nodes_m), max(p[0] for p in all_nodes_m)
    min_y_m, max_y_m = min(p[1] for p in all_nodes_m), max(p[1] for p in all_nodes_m)
    img_w_m, img_h_m = max(1.0, max_x_m - min_x_m), max(1.0, max_y_m - min_y_m)
    scale = min((WIDTH - PADDING * 2) / (img_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (img_h_m * METERS_TO_PIXELS))
    pan = [ PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale) ]
    mouse_dragging, last_mouse_pos = False, None

    running = True
    
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: mouse_dragging, last_mouse_pos = True, event.pos
                elif event.button in (4, 5):
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos_m = screen_to_grid(event.pos, scale, pan); scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos_m, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]; pan[1] += event.pos[1] - new_screen_pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0], pan[1] = pan[0] + dx, pan[1] + dy; last_mouse_pos = event.pos

        # --- UPDATE LOOP ---
        for platoon in platoons:
            # Update Platoon Logic (Rerouting, Path Projecting)
            direction, base_speed_ms = platoon.update_logic(dt, road_graph)
            
            # Update Physics for Every Truck in Platoon
            for truck in platoon.trucks:
                current_path = platoon.path
                
                # 1. Get State
                est_pos_m = np.array([truck.kf.x[0], truck.kf.x[2]])
                est_vel_m = np.array([truck.kf.x[1], truck.kf.x[3]])
                est_speed_ms = np.linalg.norm(est_vel_m)
                
                accel_cmd, steer_input = 0.0, 0.0
                
                # 2. Determine Speed Command
                if truck.is_leader:
                    # --- LEADER: STANDARD LOGIC ---
                    if direction != 0:
                        curv_check = current_path.get_curvature_at(truck.s_path_m + direction * 10.0)
                        v_turn = max(2.5, math.sqrt(MAX_LAT_ACCEL / max(1e-4, abs(curv_check))))
                        desired_speed = min(base_speed_ms, v_turn)
                        
                        dist_end = abs((0.0 if direction == -1 else current_path.length) - truck.s_path_m)
                        if dist_end < 30.0: desired_speed = min(desired_speed, base_speed_ms * (dist_end / 30.0))
                        
                        accel_cmd = np.clip((desired_speed - est_speed_ms) / 0.4, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
                    else:
                        accel_cmd = np.clip((0.0 - est_speed_ms) / 0.4, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
                else:
                    # --- FOLLOWER: ADAPTIVE CRUISE CONTROL ---
                    # Steer logic is same as leader (follow path), but Speed logic relies on front truck
                    accel_cmd = truck.get_acc_control(dt)

                # 3. Determine Steer Command (Same for everyone: Follow the Path)
                if direction != 0:
                    ld = np.clip(est_speed_ms * LOOKAHEAD_GAIN, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)
                    s_target = truck.s_path_m + direction * ld
                    p_target = current_path.point_at(s_target)
                    
                    dx_local = (p_target[0] - est_pos_m[0]) * math.cos(truck.angle) + (p_target[1] - est_pos_m[1]) * math.sin(truck.angle)
                    dy_local = -(p_target[0] - est_pos_m[0]) * math.sin(truck.angle) + (p_target[1] - est_pos_m[1]) * math.cos(truck.angle)
                    alpha = math.atan2(dy_local, max(dx_local, 1e-3))
                    steer_input = np.clip(math.atan2(2.0 * 2.8 * math.sin(alpha), ld), -0.6, 0.6)
                
                truck.move(accel_cmd, steer_input, dt)

        # --- DRAWING ---
        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan); g_to_s.scale = scale
        
        draw_road_network(screen, g_to_s, scale)
        
        # Draw active paths for ALL platoons
        for p in platoons:
             draw_active_path(screen, p.path, g_to_s, scale) 
        
        # Draw all trucks
        for p in platoons:
            for t in p.trucks:
                t.draw(screen, g_to_s)

        # HUD
        s = font.render(f"Platoons: {NUM_PLATOONS} | Total Trucks: {NUM_PLATOONS*TRUCKS_PER_PLATOON}", True, (0,0,0))
        screen.blit(s, (10, 10))
            
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_simulation()