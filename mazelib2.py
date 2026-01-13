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
SPEED_STRAIGHT_GUPS = 3.0   # 30 km/h equivalent
SPEED_CURVE_GUPS = 2.0      # Slower speed for curves
MAX_ACCELERATION_GUPS = 2.0
MAX_STEER_ANGLE = math.pi / 4 # Radians
CAR_LENGTH_GU = 0.4   # Car length in Grid Units
CAR_WIDTH_GU = 0.2
WAYPOINT_THRESHOLD_GU = 1.0 # How close to get to a waypoint (in grid units)

# --- Controller Look-ahead constants ---
LOOK_AHEAD_STEERING_IDX = 5   # Aim 5 waypoints ahead for steering
LOOK_AHEAD_SPEED_IDX = 10     # Check 10 waypoints ahead for curves
CURVE_THRESHOLD_DOT = 0.98    # Dot product threshold to detect a "straight" line

# --- NEW: Path Simplification ---
WAYPOINT_SIMPLIFICATION_THRESHOLD = 0.999 # Dot product threshold. Closer to 1.0 removes more points.


# --- Kalman Filter Class ---
class KalmanFilter:
    def __init__(self, dt):
        self.x = np.zeros(4) # [x, vx, y, vy]
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

# --- Car Class ---
class Car:
    def __init__(self, x_grid, y_grid, angle=0):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.angle = angle
        self.speed_gups = 0.0
        self.steer_angle = 0.0
        self.length_gu = CAR_LENGTH_GU
        self.width_gu = CAR_WIDTH_GU

    def move(self, throttle, steer_input, dt):
        # In a real scenario, TARGET_SPEED_GUPS would be dynamic.
        # For simplicity, we just use a high cap here.
        max_speed = SPEED_STRAIGHT_GUPS * 1.5
        self.speed_gups += throttle * dt
        self.speed_gups = max(0, min(self.speed_gups, max_speed))
        self.steer_angle = max(-MAX_STEER_ANGLE, min(steer_input, MAX_STEER_ANGLE))
        
        if self.steer_angle != 0:
            turn_radius = self.length_gu / math.tan(self.steer_angle)
            angular_velocity = self.speed_gups / turn_radius
            self.angle += angular_velocity * dt
        
        self.x_grid += self.speed_gups * math.cos(self.angle) * dt
        self.y_grid += self.speed_gups * math.sin(self.angle) * dt

    def get_noisy_measurement(self, noise_std_dev=0.05):
        noisy_x = self.x_grid + np.random.normal(0, noise_std_dev)
        noisy_y = self.y_grid + np.random.normal(0, noise_std_dev)
        return np.array([noisy_x, noisy_y])

    def draw(self, screen, grid_to_screen_func, scale):
        car_len_px = max(4, int(self.length_gu * scale))
        car_wid_px = max(2, int(self.width_gu * scale))
        car_surface = pygame.Surface((car_len_px, car_wid_px), pygame.SRCALPHA)
        car_surface.fill(CAR_COLOR)
        pygame.draw.rect(car_surface, BLACK, (car_len_px * 0.7, 0, car_len_px * 0.3, car_wid_px))
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        center_px = grid_to_screen_func((self.x_grid, self.y_grid))
        rect = rotated_surface.get_rect(center=center_px)
        screen.blit(rotated_surface, rect.topleft)

# --- Bezier and Maze Functions ---
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

# --- Pathfinding Class ---
class AStarPather:
    def __init__(self, G, POS):
        self.G = G
        self.POS = POS
    
    def find_node_path(self, src_node, dst_node):
        h = lambda a,b: math.dist(self.POS[a], self.POS[b])
        try:
            return nx.astar_path(self.G, src_node, dst_node, heuristic=h, weight='weight')
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            print(f"Warning: Could not find path from {src_node} to {dst_node}")
            return [src_node]

    def create_full_display_path(self, path_nodes, segments_per_road=20):
        """Generates a high-resolution path for drawing purposes only."""
        full_waypoints_grid = []
        if not path_nodes or len(path_nodes) < 2:
            return []
        
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            if not self.G.has_edge(u, v): continue
            
            edge_data = self.G.get_edge_data(u, v)
            p0_grid, p2_grid, p1_grid = self.POS[u], self.POS[v], edge_data['ctrl']
            
            start_t_index = 0 if i == 0 else 1
            for t_step in range(start_t_index, segments_per_road + 1):
                t = t_step / segments_per_road
                full_waypoints_grid.append(np.array(_qbez(p0_grid, p1_grid, p2_grid, t)))
                
        return full_waypoints_grid

    def create_path_waypoints(self, path_nodes, segments_per_road=20):
        """
        Creates a waypoint path that is simplified to only include points
        at turns, reducing data storage.
        """
        # Step 1: Generate the full, high-resolution path first.
        full_waypoints_grid = self.create_full_display_path(path_nodes, segments_per_road)

        if len(full_waypoints_grid) < 3:
            return full_waypoints_grid # Not enough points to simplify

        # Step 2: Simplify the path by removing collinear points.
        simplified_waypoints = [full_waypoints_grid[0]]
        for i in range(1, len(full_waypoints_grid) - 1):
            p_prev = simplified_waypoints[-1]
            p_curr = full_waypoints_grid[i]
            p_next = full_waypoints_grid[i+1]

            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                dot_product = np.dot(v1 / norm_v1, v2 / norm_v2)
                # If the vectors are not pointing in the same direction (i.e., it's a turn), keep the point.
                if dot_product < WAYPOINT_SIMPLIFICATION_THRESHOLD:
                    simplified_waypoints.append(p_curr)
        
        # Always add the very last point to complete the path.
        simplified_waypoints.append(full_waypoints_grid[-1])
        
        print(f"Path simplified: {len(full_waypoints_grid)} waypoints -> {len(simplified_waypoints)} waypoints.")
        return simplified_waypoints

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
    for u, v, data in G.edges(data=True):
        p0_grid = POS[u]; p2_grid = POS[v]; p1_grid = data['ctrl']
        curve_points_grid = [_qbez(p0_grid, p1_grid, p2_grid, i / 20.0) for i in range(21)]
        curve_points_px = [grid_to_screen_func(p) for p in curve_points_grid]
        pygame.draw.lines(screen, GRAY, False, curve_points_px, ROAD_WIDTH_PX)

    for node in G.nodes():
        pos_px = grid_to_screen_func(POS[node])
        pygame.draw.circle(screen, DARK_GRAY, pos_px, ROAD_WIDTH_PX // 2 + 2)

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

    random.seed(7)
    G, POS = build_maze(GRID_N, REMOVE_RATIO)
    
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

    pather = AStarPather(G, POS)
    start_node = random.choice(park_nodes_list)
    end_node = random.choice(list(site_nodes.get('F', park_nodes_list)))
    
    # --- MODIFIED: Generate BOTH paths ---
    node_path = pather.find_node_path(start_node, end_node)
    waypoints_grid = pather.create_path_waypoints(node_path) # Simplified path for the car
    display_path_grid = pather.create_full_display_path(node_path) # Full path for drawing

    if not waypoints_grid:
        print("Error: Could not create waypoints. Exiting.")
        return

    initial_angle = math.atan2(waypoints_grid[1][1] - waypoints_grid[0][1], 
                               waypoints_grid[1][0] - waypoints_grid[0][0])
    car = Car(waypoints_grid[0][0], waypoints_grid[0][1], angle=initial_angle)
    
    kf = KalmanFilter(dt=1.0/60.0)
    kf.x = np.array([car.x_grid, 0, car.y_grid, 0])
    
    current_waypoint_idx = 0
    direction = 1

    scale = (min(WIDTH, HEIGHT) - PADDING * 2) / GRID_N
    pan_offset_px = [PADDING, PADDING]
    mouse_dragging = False
    last_mouse_pos = None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    mouse_dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.button in (4, 5): # Scroll wheel
                    mouse_pos_screen = pygame.mouse.get_pos()
                    grid_pos_before = screen_to_grid(mouse_pos_screen, scale, pan_offset_px)
                    zoom_change = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    scale *= zoom_change
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

        est_pos_grid = np.array([kf.x[0], kf.x[2]])
        
        # --- Navigation ---
        dist_to_progress_target = np.linalg.norm(waypoints_grid[current_waypoint_idx] - est_pos_grid)
        if dist_to_progress_target < WAYPOINT_THRESHOLD_GU:
            if direction == 1 and current_waypoint_idx == len(waypoints_grid) - 1:
                direction = -1
            elif direction == -1 and current_waypoint_idx == 0:
                direction = 1
            else:
                current_waypoint_idx += direction
        
        # --- Speed Control ---
        speed_look_ahead_idx = np.clip(current_waypoint_idx + direction * LOOK_AHEAD_SPEED_IDX, 0, len(waypoints_grid)-1)
        next_point_idx = np.clip(current_waypoint_idx + direction, 0, len(waypoints_grid)-1)
        v1 = waypoints_grid[next_point_idx] - est_pos_grid
        v2 = waypoints_grid[speed_look_ahead_idx] - waypoints_grid[next_point_idx]

        if np.linalg.norm(v1) < 0.1 or np.linalg.norm(v2) < 0.1:
            is_straight = True
        else:
            dot_product = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
            is_straight = dot_product > CURVE_THRESHOLD_DOT

        desired_speed_gups = SPEED_STRAIGHT_GUPS if is_straight else SPEED_CURVE_GUPS
        est_speed_gups = np.linalg.norm([kf.x[1], kf.x[3]])
        speed_error = desired_speed_gups - est_speed_gups
        throttle = np.clip(speed_error, -MAX_ACCELERATION_GUPS, MAX_ACCELERATION_GUPS)
        
        # --- Steering Control ---
        steering_target_idx = np.clip(current_waypoint_idx + direction * LOOK_AHEAD_STEERING_IDX, 0, len(waypoints_grid)-1)
        vector_to_target = waypoints_grid[steering_target_idx] - est_pos_grid
        angle_to_target = math.atan2(vector_to_target[1], vector_to_target[0])
        angle_error = (angle_to_target - car.angle + math.pi) % (2 * math.pi) - math.pi
        steer_input = np.clip(angle_error * 2.5, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
        
        # --- Kalman Filter and Car Update ---
        z = car.get_noisy_measurement() 
        accel_gups_vector = np.array([throttle * math.cos(car.angle), throttle * math.sin(car.angle)])
        kf.predict(u=accel_gups_vector)
        kf.update(z=z) 
        car.move(throttle, steer_input, dt)

        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s_func = lambda pos: grid_to_screen(pos, scale, pan_offset_px)
        draw_maze_network(screen, G, POS, SPECIAL, g_to_s_func, font)
        
        # MODIFIED: Draw the full path for visuals
        if len(display_path_grid) > 1:
            path_px = [g_to_s_func(p) for p in display_path_grid]
            pygame.draw.lines(screen, YELLOW, False, path_px, 3)

        car.draw(screen, g_to_s_func, scale)
        kf_pos_px = g_to_s_func((kf.x[0], kf.x[2]))
        pygame.draw.circle(screen, RED, kf_pos_px, 8, 2)
        
        steering_px = g_to_s_func(waypoints_grid[steering_target_idx])
        pygame.draw.circle(screen, GREEN, steering_px, 10, 2)
        pygame.draw.line(screen, GREEN, g_to_s_func(est_pos_grid), steering_px, 2)

        # Draw HUD
        hud_surface = pygame.Surface((250, 110), pygame.SRCALPHA)
        hud_surface.fill((*HUD_BG, 200))
        pygame.draw.rect(hud_surface, BLACK, hud_surface.get_rect(), 1)
        texts = [
            "Left Click + Drag to Pan", "Mouse Wheel to Zoom",
            f"Zoom: {scale:.1f}x", f"Speed: {car.speed_gups:.1f} gu/s",
            f"Target: {desired_speed_gups:.1f} gu/s"
        ]
        for i, text in enumerate(texts):
            rendered_text = hud_font.render(text, True, HUD_TEXT)
            hud_surface.blit(rendered_text, (10, 5 + i * 20))
        screen.blit(hud_surface, (10, 10))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()