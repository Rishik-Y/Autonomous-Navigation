import pygame
import networkx as nx
import math
import random
import numpy as np
from collections import deque, defaultdict

# --- Constants ---
# Screen dimensions
WIDTH, HEIGHT = 900, 900

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100) # Road color
DARK_GRAY = (50, 50, 50) # Intersection color
RED = (255, 0, 0)       # Load sites
GREEN = (0, 200, 0)     # Store sites
BLUE = (0, 0, 255)      # Fuel sites
PURPLE = (128, 0, 128)  # Park sites
YELLOW = (255, 255, 0)  # Path line
CAR_COLOR = (200, 0, 0)
HUD_BG = (240, 240, 240)
HUD_TEXT = (10, 10, 10)

# --- View Constants ---
ROAD_WIDTH_PX = 50      # Wide roads
SITE_RADIUS_PX = 25
ZOOM_FACTOR = 1.1
PADDING = 50

# --- Maze Constants ---
GRID_N, REMOVE_RATIO        = 25, 0.60
DEAD_END_PRUNE_RATIO        = 0.50
EDGE_MIN_KM, EDGE_MAX_KM    = 1.0, 2.0
LOAD_SITES, STORE_SITES     = 5, 5
FUEL_SITES, PARK_BAYS       = 3, 3

# --- Car Simulation Constants (in Grid Units) ---
# The car's physics now operate in the 0-24 grid space
TARGET_SPEED_GUPS = 4.0  # Grid Units Per Second
MAX_ACCELERATION_GUPS = 2.0
MAX_STEER_ANGLE = math.pi / 4 # Radians
CAR_LENGTH_GU = 0.4  # Car length in Grid Units
CAR_WIDTH_GU = 0.2
WAYPOINT_THRESHOLD_GU = 1.0 # How close to get to a waypoint (in grid units)

# --- Kalman Filter Class ---
class KalmanFilter:
    def __init__(self, dt):
        self.x = np.zeros(4) # [x, vx, y, vy]
        self.dt = dt
        # State transition matrix
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        # Control input matrix
        self.B = np.array([[0.5 * dt**2, 0], [dt, 0], [0, 0.5 * dt**2], [0, dt]])
        # Measurement matrix (we only measure position x, y)
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        # Process noise
        self.Q = np.eye(4) * 0.1
        # Measurement noise
        self.R = np.eye(2) * 1.0
        self.P = np.eye(4) # Error covariance

    def predict(self, u):
        # u = [ax, ay] (control input, acceleration)
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        # z = [x, y] (measurement)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

# --- Car Class ---
class Car:
    def __init__(self, x_grid, y_grid, angle=0):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.angle = angle # radians
        self.speed_gups = 0.0 # Grid Units Per Second
        self.steer_angle = 0.0
        self.length_gu = CAR_LENGTH_GU
        self.width_gu = CAR_WIDTH_GU

    def move(self, throttle, steer_input, dt):
        # --- Physics Update ---
        self.speed_gups += throttle * dt
        self.speed_gups = max(0, min(self.speed_gups, TARGET_SPEED_GUPS * 1.5))
        self.steer_angle = max(-MAX_STEER_ANGLE, min(steer_input, MAX_STEER_ANGLE))
        
        # Ackermann steering model
        if self.steer_angle != 0:
            turn_radius = self.length_gu / math.tan(self.steer_angle)
            angular_velocity = self.speed_gups / turn_radius
            self.angle += angular_velocity * dt
        
        self.x_grid += self.speed_gups * math.cos(self.angle) * dt
        self.y_grid += self.speed_gups * math.sin(self.angle) * dt

    def get_noisy_measurement(self, noise_std_dev=0.05):
        """Returns noisy position in grid units."""
        noisy_x = self.x_grid + np.random.normal(0, noise_std_dev)
        noisy_y = self.y_grid + np.random.normal(0, noise_std_dev)
        return np.array([noisy_x, noisy_y])

    def draw(self, screen, grid_to_screen_func, scale):
        """Draws the car, scaling it based on the current zoom level."""
        # Convert car dimensions from grid units to pixels
        car_len_px = max(4, int(self.length_gu * scale))
        car_wid_px = max(2, int(self.width_gu * scale))
        
        # Create a car surface (drawing it fresh each time)
        car_surface = pygame.Surface((car_len_px, car_wid_px), pygame.SRCALPHA)
        car_surface.fill(CAR_COLOR)
        
        # Add a "windshield" to show direction
        pygame.draw.rect(car_surface, BLACK, (car_len_px * 0.7, 0, car_len_px * 0.3, car_wid_px))
        
        # Rotate the surface
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        
        # Get screen position from grid position
        center_px = grid_to_screen_func((self.x_grid, self.y_grid))
        rect = rotated_surface.get_rect(center=center_px)
        screen.blit(rotated_surface, rect.topleft)

# --- Bezier and Maze Functions (from original) ---
def _perp(dx,dy): 
    ux,uy = -dy,dx; n = math.hypot(ux,uy)
    return (0,0) if n==0 else (ux/n,uy/n)

def _ctrl(p0,p2,s=.35):
    (x0,y0),(x2,y2)=p0,p2; mx,my=((x0+x2)/2,(y0+y2)/2)
    dx,dy=x2-x0,y2-y0; px,py=_perp(dx,dy)
    return mx+random.choice((-1,1))*s*px, my+random.choice((-1,1))*s*py

def _qbez(p0,p1,p2,t):
    u=1-t
    return (u*u*p0[0]+2*u*t*p1[0]+t*t*p2[0],
            u*u*p0[1]+2*u*t*p1[1]+t*t*p2[1])

def build_maze(n,drop):
    G = nx.grid_2d_graph(n,n); POS = {(x,y):(float(x),float(y)) for x,y in G}
    for u,v in G.edges():
        G.edges[u,v]['weight'] = random.uniform(EDGE_MIN_KM, EDGE_MAX_KM)
        G.edges[u,v]['ctrl']   = _ctrl(POS[u], POS[v])
    original_edges = list(G.edges(data=True))
    edges_to_remove = list(G.edges()); random.shuffle(edges_to_remove)
    for u,v in edges_to_remove[:int(drop*len(edges_to_remove))]:
        G.remove_edge(u,v)
        if not nx.is_connected(G):
            d = next((d for ou,ov,d in original_edges if (ou,ov)==(u,v) or (ou,ov)==(v,u)), None)
            if d: G.add_edge(u, v, **d)
    for u,v in [((8,18),(9,18)),((11,14),(12,14)),((19,21),(19,22))]:
        if (u in G and v in G) and not G.has_edge(u,v):
            d = next((d for ou,ov,d in original_edges if (ou,ov)==(u,v) or (ou,ov)==(v,u)), None)
            if d: G.add_edge(u, v, **d)
            else: G.add_edge(u, v, weight=random.uniform(EDGE_MIN_KM, EDGE_MAX_KM), ctrl=_ctrl(POS[u], POS[v]))
    return G, POS

def prune_leaves(G,keep):
    leaves=[n for n in G if G.degree[n]==1 and n not in keep]
    random.shuffle(leaves);
    for n in leaves[:int(len(leaves)*DEAD_END_PRUNE_RATIO)]:
        if n in G: G.remove_node(n)
    return G

def find_closest_node(target_coord, available_nodes):
    return min(available_nodes, key=lambda node: math.dist(node, target_coord))

# --- NEW Pathfinding Class ---
class AStarPather:
    def __init__(self, G, POS):
        self.G = G
        self.POS = POS # Keep grid positions
    
    def find_node_path(self, src_node, dst_node):
        """Finds the shortest path of NODES using A*."""
        h = lambda a,b: math.dist(self.POS[a], self.POS[b])
        try:
            return nx.astar_path(self.G, src_node, dst_node, heuristic=h, weight='weight')
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            print(f"Warning: Could not find path from {src_node} to {dst_node}")
            return [src_node]

    def create_path_waypoints(self, path_nodes, segments_per_road=20):
        """
        *** THIS IS THE KEY FUNCTION ***
        Converts a list of nodes [n1, n2, n3] into a high-resolution
        list of waypoints that follow the Bezier curves.
        """
        waypoints_grid = []
        if not path_nodes:
            return []

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            
            # Check for edge existence
            if not self.G.has_edge(u, v):
                print(f"Warning: No edge between {u} and {v}, path is broken.")
                continue

            edge_data = self.G.get_edge_data(u, v)
            p0_grid = self.POS[u]
            p2_grid = self.POS[v]
            p1_grid = edge_data['ctrl']
            
            # Add points for this curve
            # Start at t=0 only for the very first segment
            start_t_index = 0 if i == 0 else 1
            
            for t_step in range(start_t_index, segments_per_road + 1):
                t = t_step / segments_per_road
                point_grid = _qbez(p0_grid, p1_grid, p2_grid, t)
                waypoints_grid.append(point_grid)
                
        # Convert to numpy arrays for easier math
        return [np.array(p) for p in waypoints_grid]

# --- Coordinate and Drawing Functions ---
def grid_to_screen(grid_pos, scale, pan_offset):
    px = (grid_pos[0] * scale) + pan_offset[0]
    py = (grid_pos[1] * scale) + pan_offset[1]
    return (int(px), int(py))

def screen_to_grid(screen_pos, scale, pan_offset):
    gx = (screen_pos[0] - pan_offset[0]) / scale
    gy = (screen_pos[1] - pan_offset[1]) / scale
    return (gx, gy)

def draw_maze_network(screen, G, POS, SPECIAL, grid_to_screen_func, font):
    # 1. Draw Roads
    for u, v, data in G.edges(data=True):
        p0_grid = POS[u]; p2_grid = POS[v]; p1_grid = data['ctrl']
        curve_points_grid = []
        num_segments = 20
        for i in range(num_segments + 1):
            t = i / num_segments
            curve_points_grid.append(_qbez(p0_grid, p1_grid, p2_grid, t))
        curve_points_px = [grid_to_screen_func(p) for p in curve_points_grid]
        pygame.draw.lines(screen, GRAY, False, curve_points_px, ROAD_WIDTH_PX)

    # 2. Draw Intersections
    for node in G.nodes():
        pos_px = grid_to_screen_func(POS[node])
        pygame.draw.circle(screen, DARK_GRAY, pos_px, ROAD_WIDTH_PX // 2 + 2)

    # 3. Draw Special Sites
    site_colors = {'L': RED, 'S': GREEN, 'F': BLUE, 'P': PURPLE}
    for label, node in SPECIAL.items():
        if node not in G: continue
        pos_px = grid_to_screen_func(POS[node])
        color = site_colors.get(label[0], WHITE)
        pygame.draw.circle(screen, color, pos_px, SITE_RADIUS_PX)
        pygame.draw.circle(screen, BLACK, pos_px, SITE_RADIUS_PX, 3)
        label_text = font.render(label, True, BLACK)
        text_rect = label_text.get_rect(center=(pos_px[0], pos_px[1] - SITE_RADIUS_PX - 10))
        screen.blit(label_text, text_rect)

# --- Main ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Car Simulation on Curved Roads")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16, bold=True)
    hud_font = pygame.font.SysFont("Arial", 18, bold=True)

    # --- 1. Generate the Maze ---
    random.seed(7)
    G, POS = build_maze(GRID_N, REMOVE_RATIO)
    
    # --- 2. Place Sites ---
    available_nodes = list(G.nodes())
    target_coords = {
        'L1': (5,20), 'L2': (20,23), 'L4': (10,17), 'S4': (15,19), 'L5': (18,10),
        'S3': (21,15), 'F1': (12,13), 'S5': (14,11), 'S1': (22,6), 'F2': (8,5),
        'S2': (4,6),  'P1': (3,22),  'P2': (23,2),  'P3': (2,4),  'F3': (18,21),
        'L3': (3,16),
    }
    site_count_var_map = {'L':'LOAD_SITES','S':'STORE_SITES','F':'FUEL_SITES','P':'PARK_BAYS'}
    site_nodes = {}
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
    if not park_nodes_list: park_nodes_list = [random.choice(list(G.nodes()))]

    SPECIAL = {**{f'L{i+1}':n for i,n in enumerate(site_nodes.get('L', []))},
               **{f'S{i+1}':n for i,n in enumerate(site_nodes.get('S', []))},
               **{f'F{i+1}':n for i,n in enumerate(site_nodes.get('F', []))},
               **{f'P{i+1}':n for i,n in enumerate(park_nodes_list)}}
    
    G = prune_leaves(G, set(SPECIAL.values()))

    # --- 3. Setup Path and Car ---
    pather = AStarPather(G, POS)
    start_node = random.choice(park_nodes_list)
    end_node = random.choice(list(site_nodes.get('F', park_nodes_list)))
    
    node_path = pather.find_node_path(start_node, end_node)
    waypoints_grid = pather.create_path_waypoints(node_path)

    if not waypoints_grid:
        print("Error: Could not create waypoints. Exiting.")
        return

    # Initialize Car
    initial_angle = math.atan2(waypoints_grid[1][1] - waypoints_grid[0][1], 
                              waypoints_grid[1][0] - waypoints_grid[0][0])
    car = Car(waypoints_grid[0][0], waypoints_grid[0][1], angle=initial_angle)
    
    # Initialize Kalman Filter
    kf = KalmanFilter(dt=1.0/60.0) # Assume 60 FPS for dt
    kf.x = np.array([car.x_grid, 0, car.y_grid, 0])
    
    current_waypoint_idx = 0
    direction = 1 # 1 for forward, -1 for backward

    # --- 4. Setup View Controls ---
    scale = (min(WIDTH, HEIGHT) - PADDING * 2) / GRID_N
    pan_offset_px = [PADDING, PADDING]
    mouse_dragging = False
    last_mouse_pos = None

    # --- 5. Main Loop ---
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue
        
        # --- 5a. Event Handling (Pan/Zoom/Quit) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    mouse_dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4: # Scroll up
                    mouse_pos_screen = pygame.mouse.get_pos()
                    grid_pos_before = screen_to_grid(mouse_pos_screen, scale, pan_offset_px)
                    scale *= ZOOM_FACTOR
                    screen_pos_after_scale = grid_to_screen(grid_pos_before, scale, pan_offset_px)
                    pan_offset_px[0] += mouse_pos_screen[0] - screen_pos_after_scale[0]
                    pan_offset_px[1] += mouse_pos_screen[1] - screen_pos_after_scale[1]
                elif event.button == 5: # Scroll down
                    mouse_pos_screen = pygame.mouse.get_pos()
                    grid_pos_before = screen_to_grid(mouse_pos_screen, scale, pan_offset_px)
                    scale /= ZOOM_FACTOR
                    screen_pos_after_scale = grid_to_screen(grid_pos_before, scale, pan_offset_px)
                    pan_offset_px[0] += mouse_pos_screen[0] - screen_pos_after_scale[0]
                    pan_offset_px[1] += mouse_pos_screen[1] - screen_pos_after_scale[1]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    current_mouse_pos = pygame.mouse.get_pos()
                    dx = current_mouse_pos[0] - last_mouse_pos[0]
                    dy = current_mouse_pos[1] - last_mouse_pos[1]
                    pan_offset_px[0] += dx; pan_offset_px[1] += dy
                    last_mouse_pos = current_mouse_pos

        # --- 5b. Simulation Logic ---
        
        # Get Kalman Filter's estimated position
        est_pos_grid = np.array([kf.x[0], kf.x[2]])
        
        # Navigation Logic
        target_waypoint_grid = waypoints_grid[current_waypoint_idx]
        dist_to_waypoint = np.linalg.norm(target_waypoint_grid - est_pos_grid)

        # Waypoint switching and reversing
        if dist_to_waypoint < WAYPOINT_THRESHOLD_GU:
            if direction == 1 and current_waypoint_idx == len(waypoints_grid) - 1:
                direction = -1 # Reverse
            elif direction == -1 and current_waypoint_idx == 0:
                direction = 1 # Go forward again
            else:
                current_waypoint_idx += direction
        
        # Controller Logic
        target_waypoint_grid = waypoints_grid[current_waypoint_idx]
        est_vel_grid = np.array([kf.x[1], kf.x[3]])
        est_speed_gups = np.linalg.norm(est_vel_grid)
        
        # --- Speed Control ---
        speed_error = TARGET_SPEED_GUPS - est_speed_gups
        throttle = np.clip(speed_error, -MAX_ACCELERATION_GUPS, MAX_ACCELERATION_GUPS)
        
        # --- Steering Control (PID-like) ---
        vector_to_target = target_waypoint_grid - est_pos_grid
        angle_to_target = math.atan2(vector_to_target[1], vector_to_target[0])
        
        angle_error = (angle_to_target - car.angle)
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi # Normalize
        
        steer_input = np.clip(angle_error * 2.5, -MAX_STEER_ANGLE, MAX_STEER_ANGLE) # Steering gain
        
        # --- Update Kalman Filter ---
        # Get noisy measurement
        z = car.get_noisy_measurement() 
        # Predict next state based on controls
        accel_gups_vector = np.array([throttle * math.cos(car.angle), 
                                      throttle * math.sin(car.angle)])
        kf.predict(u=accel_gups_vector)
        # Update based on measurement
        kf.update(z=z) 
        
        # --- Update Car Physics ---
        car.move(throttle, steer_input, dt)

        # --- 5c. Drawing ---
        screen.fill(WHITE)
        
        # Create the grid-to-screen conversion function for this frame
        g_to_s_func = lambda pos: grid_to_screen(pos, scale, pan_offset_px)
        
        # Draw the maze
        draw_maze_network(screen, G, POS, SPECIAL, g_to_s_func, font)
        
        # Draw the high-resolution path
        if len(waypoints_grid) > 1:
            path_px = [g_to_s_func(p) for p in waypoints_grid]
            pygame.draw.lines(screen, YELLOW, False, path_px, 3)

        # Draw the car
        car.draw(screen, g_to_s_func, scale)
        
        # Draw KF estimate
        kf_pos_px = g_to_s_func((kf.x[0], kf.x[2]))
        pygame.draw.circle(screen, RED, kf_pos_px, 8, 2)
        
        # Draw target waypoint
        target_px = g_to_s_func(target_waypoint_grid)
        pygame.draw.circle(screen, GREEN, target_px, 10, 2)

        # Draw HUD
        hud_surface = pygame.Surface((250, 90), pygame.SRCALPHA)
        hud_surface.fill((*HUD_BG, 200))
        pygame.draw.rect(hud_surface, BLACK, hud_surface.get_rect(), 1)
        text1 = hud_font.render("Left Click + Drag to Pan", True, HUD_TEXT)
        text2 = hud_font.render("Mouse Wheel to Zoom", True, HUD_TEXT)
        text3 = hud_font.render(f"Zoom: {scale:.1f}x", True, HUD_TEXT)
        text4 = hud_font.render(f"Speed: {car.speed_gups:.1f} gu/s", True, HUD_TEXT)
        hud_surface.blit(text1, (10, 5)); hud_surface.blit(text2, (10, 25))
        hud_surface.blit(text3, (10, 45)); hud_surface.blit(text4, (10, 65))
        screen.blit(hud_surface, (10, 10))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
