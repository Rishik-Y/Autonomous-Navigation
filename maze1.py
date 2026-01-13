import pygame
import numpy as np
import math
import networkx as nx
import random
from collections import deque, defaultdict # Added for site placement logic

# --- Constants from Maze Code ---
GRID_N, REMOVE_RATIO        = 25, 0.60
DEAD_END_PRUNE_RATIO        = 0.50
EDGE_MIN_KM, EDGE_MAX_KM    = 1.0, 2.0
LOAD_SITES, MAX_LOAD_CAP    = 5, 4
FUEL_SITES, STORE_SITES     = 3, 5
PARK_BAYS                   = 3

# --- Constants ---
# Screen dimensions
WIDTH, HEIGHT = 900, 900 # Increased size a bit

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50) # For intersections
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128) # For Parking

# Simulation parameters
FPS = 60
GRID_UNIT_TO_METERS = 8.0 # How many meters one unit of the 25x25 grid represents
METERS_TO_PIXELS = 4 # Scale from meters to pixels
ROAD_WIDTH_M = 10
ROAD_WIDTH_PX = int(ROAD_WIDTH_M * METERS_TO_PIXELS)

# Car parameters
CAR_LENGTH_M = 4.5
CAR_WIDTH_M = 2.0
TARGET_SPEED_KMPH = 40.0 # Increased speed a bit for the larger map
TARGET_SPEED_MPS = TARGET_SPEED_KMPH * 1000 / 3600
MAX_ACCELERATION = 2.0
MAX_STEER_ANGLE = math.pi / 4

# --- Kalman Filter Class (Unchanged) ---
class KalmanFilter:
    def __init__(self, dt):
        self.x = np.zeros(4)
        self.dt = dt
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt**2, 0], [dt, 0], [0, 0.5 * dt**2], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 1.0
        self.P = np.eye(4)

    def predict(self, u):
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

# --- Car Class (Unchanged) ---
class Car:
    def __init__(self, x_m, y_m, angle=0):
        self.x_m = x_m
        self.y_m = y_m
        self.angle = angle
        self.speed_mps = 0.0
        self.steer_angle = 0.0
        car_len_px = max(4, int(CAR_LENGTH_M * METERS_TO_PIXELS)) # Ensure min 4px
        car_wid_px = max(2, int(CAR_WIDTH_M * METERS_TO_PIXELS)) # Ensure min 2px
        self.original_image = pygame.Surface((car_len_px, car_wid_px), pygame.SRCALPHA)
        self.original_image.fill(BLUE)
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(self.x_m * METERS_TO_PIXELS, self.y_m * METERS_TO_PIXELS))

    def move(self, throttle, steer_input, dt):
        self.speed_mps += throttle * dt
        self.speed_mps = max(0, min(self.speed_mps, TARGET_SPEED_MPS * 1.5))
        self.steer_angle = max(-MAX_STEER_ANGLE, min(steer_input, MAX_STEER_ANGLE))
        
        if self.steer_angle != 0:
            turn_radius = CAR_LENGTH_M / math.tan(self.steer_angle)
            angular_velocity = self.speed_mps / turn_radius
            self.angle += angular_velocity * dt
        
        self.x_m += self.speed_mps * math.cos(self.angle) * dt
        self.y_m += self.speed_mps * math.sin(self.angle) * dt

    def get_noisy_measurement(self, noise_std_dev=0.5):
        noisy_x = self.x_m + np.random.normal(0, noise_std_dev)
        noisy_y = self.y_m + np.random.normal(0, noise_std_dev)
        return np.array([noisy_x, noisy_y])

    def draw(self, screen, world_offset_px):
        self.image = pygame.transform.rotate(self.original_image, -math.degrees(self.angle))
        center_px = (
            self.x_m * METERS_TO_PIXELS + world_offset_px[0],
            self.y_m * METERS_TO_PIXELS + world_offset_px[1]
        )
        self.rect = self.image.get_rect(center=center_px)
        screen.blit(self.image, self.rect.topleft)

# --- Helper functions from Maze Code ---
def _perp(dx,dy): 
    ux,uy = -dy,dx
    n = math.hypot(ux,uy)
    return (0,0) if n==0 else (ux/n,uy/n)

def _ctrl(p0,p2,s=.35):
    (x0,y0),(x2,y2)=p0,p2
    mx,my=((x0+x2)/2,(y0+y2)/2)
    dx,dy=x2-x0,y2-y0
    px,py=_perp(dx,dy)
    return mx+random.choice((-1,1))*s*px, my+random.choice((-1,1))*s*py

def _qbez(p0,p1,p2,t):
    u=1-t
    return (u*u*p0[0]+2*u*t*p1[0]+t*t*p2[0],
            u*u*p0[1]+2*u*t*p1[1]+t*t*p2[1])

# --- Maze Generation Functions from Maze Code ---
def build_maze(n,drop):
    G = nx.grid_2d_graph(n,n)
    POS = {(x,y):(float(x),float(y)) for x,y in G}
    for u,v in G.edges():
        wt  = random.uniform(EDGE_MIN_KM, EDGE_MAX_KM)
        G.edges[u,v]['weight'] = wt
        G.edges[u,v]['ctrl']   = _ctrl(POS[u], POS[v])

    original_edges = list(G.edges(data=True))
    edges_to_remove = list(G.edges()); random.shuffle(edges_to_remove)
    for u,v in edges_to_remove[:int(drop*len(edges_to_remove))]:
        G.remove_edge(u,v)
        if not nx.is_connected(G):
            d = next((d for ou,ov,d in original_edges if (ou,ov)==(u,v) or (ou,ov)==(v,u)), None)
            if d: G.add_edge(u, v, **d)

    # Manual connectivity fixes (optional, but in original)
    for u,v in [((8,18),(9,18)),((11,14),(12,14)),((19,21),(19,22))]:
        if (u in G and v in G) and not G.has_edge(u,v):
            d = next((d for ou,ov,d in original_edges if (ou,ov)==(u,v) or (ou,ov)==(v,u)), None)
            if d: G.add_edge(u, v, **d)
            else:
                wt = random.uniform(EDGE_MIN_KM, EDGE_MAX_KM)
                G.add_edge(u, v, weight=wt, ctrl=_ctrl(POS[u], POS[v]))
    return G, POS

def prune_leaves(G,keep):
    leaves=[n for n in G if G.degree[n]==1 and n not in keep]
    random.shuffle(leaves)
    for n in leaves[:int(len(leaves)*DEAD_END_PRUNE_RATIO)]:
        if n in G: G.remove_node(n)
    return G

def find_closest_node(target_coord, available_nodes):
    return min(available_nodes, key=lambda node: math.dist(node, target_coord))

# --- NEW Pygame Drawing Functions ---
def world_to_screen(pos_m, world_offset_px):
    """Converts world meters to screen pixels."""
    px = pos_m[0] * METERS_TO_PIXELS + world_offset_px[0]
    py = pos_m[1] * METERS_TO_PIXELS + world_offset_px[1]
    return (int(px), int(py))

def draw_maze(screen, G, POS_M, SPECIAL, world_offset_px, font):
    """Draws the entire networkx maze using curved roads and sites."""
    
    # 1. Draw Roads
    for u, v, data in G.edges(data=True):
        p0_m = POS_M[u]
        p2_m = POS_M[v]
        
        # Get the control point. It's in grid units, so scale it.
        ctrl_grid = data['ctrl']
        p1_m = (ctrl_grid[0] * GRID_UNIT_TO_METERS, ctrl_grid[1] * GRID_UNIT_TO_METERS)

        # Sample the Bezier curve
        curve_points_m = []
        num_segments = 20 # More segments = smoother curve
        for i in range(num_segments + 1):
            t = i / num_segments
            point_m = _qbez(p0_m, p1_m, p2_m, t)
            curve_points_m.append(point_m)
        
        # Convert to screen coordinates
        curve_points_px = [world_to_screen(p, world_offset_px) for p in curve_points_m]
        
        # Draw the segmented line
        pygame.draw.lines(screen, GRAY, False, curve_points_px, ROAD_WIDTH_PX)

    # 2. Draw Intersections (Hubs)
    for node, pos_m in POS_M.items():
        if node in G: # Only draw nodes still in the graph
            pos_px = world_to_screen(pos_m, world_offset_px)
            pygame.draw.circle(screen, DARK_GRAY, pos_px, ROAD_WIDTH_PX // 2 + 2) # Draw a hub

    # 3. Draw Special Sites
    site_colors = {'L': RED, 'S': GREEN, 'F': BLUE, 'P': PURPLE}
    site_marker_radius = int(ROAD_WIDTH_PX * 0.7)
    
    for label, node in SPECIAL.items():
        if node not in G: continue
        pos_m = POS_M[node]
        pos_px = world_to_screen(pos_m, world_offset_px)
        color = site_colors.get(label[0], WHITE)
        
        pygame.draw.circle(screen, color, pos_px, site_marker_radius)
        pygame.draw.circle(screen, BLACK, pos_px, site_marker_radius, 2) # Outline
        
        # Draw label
        label_text = font.render(label, True, BLACK)
        screen.blit(label_text, (pos_px[0] + 10, pos_px[1] - 10))


# --- Main simulation ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Autonomous Car: Networkx Maze Challenge")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14, bold=True)
    hud_font = pygame.font.SysFont("Arial", 18, bold=True)

    # --- NEW: Generate the maze ---
    random.seed(7); np.random.seed(7)
    G, POS_GRID = build_maze(GRID_N, REMOVE_RATIO)
    
    # --- NEW: Place Sites ---
    available_nodes = list(G.nodes())
    target_coords = {
        'L1': (5,20), 'L2': (20,23), 'L4': (10,17), 'S4': (15,19), 'L5': (18,10),
        'S3': (21,15), 'F1': (12,13), 'S5': (14,11), 'S1': (22,6), 'F2': (8,5),
        'S2': (4,6),  'P1': (3,22),  'P2': (23,2),  'P3': (2,4),  'F3': (18,21),
        'L3': (3,16),
    }
    site_count_var_map = {'L':'LOAD_SITES','S':'STORE_SITES','F':'FUEL_SITES','P':'PARK_BAYS'}
    site_nodes = {}
    
    # Use globals() to access the constants defined at the top
    sim_globals = globals() 
    
    for code, target in target_coords.items():
        if not available_nodes: break
        node_type = code[0]
        limit_var_name = site_count_var_map.get(node_type)
        if limit_var_name and len(site_nodes.get(node_type, [])) < sim_globals[limit_var_name]:
            closest = find_closest_node(target, available_nodes)
            if closest in available_nodes:
                site_nodes[node_type] = site_nodes.get(node_type, []) + [closest]
                available_nodes.remove(closest)

    park_nodes_list  = site_nodes.get('P', [])
    if not park_nodes_list: # Ensure there's at least one park node
        park_nodes_list = [random.choice(list(G.nodes()))]
        site_nodes['P'] = park_nodes_list

    SPECIAL = {**{f'L{i+1}':n for i,n in enumerate(site_nodes.get('L', []))},
               **{f'S{i+1}':n for i,n in enumerate(site_nodes.get('S', []))},
               **{f'F{i+1}':n for i,n in enumerate(site_nodes.get('F', []))},
               **{f'P{i+1}':n for i,n in enumerate(park_nodes_list)}}
    
    # --- NEW: Prune leaves, keeping sites safe ---
    G = prune_leaves(G, set(SPECIAL.values()))

    # --- NEW: Convert grid positions to metric positions (meters) ---
    POS_M = {node: (pos[0] * GRID_UNIT_TO_METERS, pos[1] * GRID_UNIT_TO_METERS) 
             for node, pos in POS_GRID.items() if node in G}
    
    # --- NEW: Define A* path helper (needs G and POS_M from local scope) ---
    def astar_path_safe(src, dst):
        if src == dst: return [src]
        # Heuristic: Euclidean distance in meters
        h = lambda a,b: math.dist(POS_M[a], POS_M[b]) 
        try:
            # Use 'weight' (edge_km) as the cost
            return nx.astar_path(G, src, dst, heuristic=h, weight='weight')
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            print(f"Warning: Could not find path from {src} to {dst}")
            return [src]

    # --- NEW: Create a path for the car using A* ---
    start_node = random.choice(park_nodes_list)
    end_node = random.choice(park_nodes_list)
    while end_node == start_node and len(park_nodes_list) > 1:
        end_node = random.choice(park_nodes_list)
        
    path_nodes = astar_path_safe(start_node, end_node)

    # Convert node path to metric waypoint list
    waypoints_m = [np.array(POS_M[node]) for node in path_nodes]

    if len(waypoints_m) < 2:
        print("Error: Could not find a valid A* path. Exiting.")
        running = False
        waypoints_m = [np.array([0,0]), np.array([1,1])] # Dummy path
        
    current_waypoint_idx = 0

    # --- NEW: Center the maze world on the screen ---
    world_size_m = (GRID_N * GRID_UNIT_TO_METERS, GRID_N * GRID_UNIT_TO_METERS)
    world_offset_px = (
        (WIDTH - world_size_m[0] * METERS_TO_PIXELS) / 2, 
        (HEIGHT - world_size_m[1] * METERS_TO_PIXELS) / 2
    )

    # Convert waypoints from meters to pixel coordinates for drawing
    waypoints_px = [world_to_screen(p, world_offset_px) for p in waypoints_m]

    # Initialize car at the first waypoint
    initial_angle = math.atan2(
        waypoints_m[1][1] - waypoints_m[0][1], 
        waypoints_m[1][0] - waypoints_m[0][0]
    )
    car = Car(waypoints_m[0][0], waypoints_m[0][1], angle=initial_angle)
    kf = KalmanFilter(dt=1.0/FPS)
    kf.x = np.array([car.x_m, 0, car.y_m, 0])
    
    direction = 1  # 1 for forward, -1 for backward

    # --- Main Loop ---
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        if dt == 0: continue # Avoid division by zero if frame rate is too high

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Navigation Logic ---
        target_waypoint = waypoints_m[current_waypoint_idx]
        est_pos = np.array([kf.x[0], kf.x[2]])
        dist_to_waypoint = np.linalg.norm(target_waypoint - est_pos)

        # Waypoint switching and reversing logic
        if dist_to_waypoint < 7.0: # Threshold to switch to next waypoint
            if direction == 1 and current_waypoint_idx == len(waypoints_m) - 1:
                direction = -1
                # Optional: Find a new path
                # start_node = path_nodes[-1]
                # end_node = random.choice(park_nodes_list)
                # path_nodes = astar_path_safe(start_node, end_node)
                # waypoints_m = [np.array(POS_M[node]) for node in path_nodes]
                # waypoints_px = [world_to_screen(p, world_offset_px) for p in waypoints_m]
                # current_waypoint_idx = 0 # Reset to start of new path
                
            elif direction == -1 and current_waypoint_idx == 0:
                direction = 1
            else:
                current_waypoint_idx += direction
            
            # Ensure index is valid after logic
            current_waypoint_idx = max(0, min(len(waypoints_m) - 1, current_waypoint_idx))


        # Control and Physics Logic
        z = car.get_noisy_measurement()
        est_vel = np.array([kf.x[1], kf.x[3]])
        est_speed = np.linalg.norm(est_vel)
        
        turn_slowdown_distance = 25.0
        if dist_to_waypoint < turn_slowdown_distance and est_speed > 5.0:
            desired_speed = TARGET_SPEED_MPS * (dist_to_waypoint / turn_slowdown_distance)
            desired_speed = max(desired_speed, 4.0)
        else:
            desired_speed = TARGET_SPEED_MPS

        speed_error = desired_speed - est_speed
        throttle = np.clip(speed_error, -MAX_ACCELERATION, MAX_ACCELERATION)
        
        # Update target waypoint for controller
        target_waypoint = waypoints_m[current_waypoint_idx] 
        vector_to_target = target_waypoint - est_pos
        angle_to_target = math.atan2(vector_to_target[1], vector_to_target[0])
        
        angle_error = (angle_to_target - car.angle)
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi # Normalize
        steer_input = np.clip(angle_error * 2.5, -MAX_STEER_ANGLE, MAX_STEER_ANGLE) # Steering gain
        
        accel_vector = np.array([throttle * math.cos(car.angle), throttle * math.sin(car.angle)])
        kf.predict(u=accel_vector)
        kf.update(z=z)
        car.move(throttle, steer_input, dt)

        # --- Drawing ---
        screen.fill(WHITE)
        
        # --- MODIFIED: Draw the full maze ---
        draw_maze(screen, G, POS_M, SPECIAL, world_offset_px, font)
        
        # --- Draw car's current path ---
        # Draw all waypoints as small circles
        for wp_px in waypoints_px:
            pygame.draw.circle(screen, RED, wp_px, 4)
        
        # Draw path lines
        if len(waypoints_px) > 1:
            pygame.draw.lines(screen, RED, False, waypoints_px, 1)

        # Dynamic START and END markers
        start_pos_px, end_pos_px = waypoints_px[0], waypoints_px[-1]
        
        start_text = hud_font.render("START", True, BLACK)
        end_text = hud_font.render("END", True, BLACK)
        
        screen.blit(start_text, (start_pos_px[0] + 10, start_pos_px[1] + 10))
        screen.blit(end_text, (end_pos_px[0] + 10, end_pos_px[1] + 10))
        
        # Visualizations (Path line, KF estimate)
        target_wp_px = world_to_screen(target_waypoint, world_offset_px)
        car_px_center = (car.x_m * METERS_TO_PIXELS + world_offset_px[0], car.y_m * METERS_TO_PIXELS + world_offset_px[1])
        pygame.draw.line(screen, YELLOW, car_px_center, target_wp_px, 2)
        
        car.draw(screen, world_offset_px)
        
        kf_pos_px = (kf.x[0] * METERS_TO_PIXELS + world_offset_px[0], kf.x[2] * METERS_TO_PIXELS + world_offset_px[1])
        pygame.draw.circle(screen, RED, kf_pos_px, 8, 2)

        # HUD
        speed_text = hud_font.render(f"Speed: {car.speed_mps * 3.6:.1f} km/h", True, BLACK)
        direction_str = "Forward" if direction == 1 else "Backward"
        state_text = hud_font.render(f"Direction: {direction_str}", True, BLACK)
        target_text = hud_font.render(f"Target WP: {current_waypoint_idx+1}/{len(waypoints_m)}", True, BLACK)
        
        screen.blit(speed_text, (10, 10))
        screen.blit(state_text, (10, 30))
        screen.blit(target_text, (10, 50))


        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
