import pygame
import numpy as np
import math
import heapq  # <-- Added for A* Priority Queue

# --- Constants ---

# --- NEW: Redundant "Coal Mine" Layout ---
# We are adding a "shortcut" path.
NODES = {
    # Start area
    "start_zone":   np.array([20.0, 150.0]),
    "start_exit":   np.array([50.0, 150.0]),
    
    # --- MAIN ROUTE ---
    "main_1": np.array([100.0, 140.0]),
    "main_2":  np.array([120.0, 120.0]),
    "main_3":  np.array([100.0, 100.0]),
    
    # --- SHORTCUT ROUTE ---
    "shortcut_1": np.array([70.0, 130.0]),
    "shortcut_2": np.array([80.0, 110.0]),
    
    # --- REJOIN & LOADING ---
    "load_entry":    np.array([50.0, 90.0]),
    "load_zone":     np.array([30.0, 80.0]),
    
    # --- DUMP ROUTE ---
    "dump_uturn_1":  np.array([10.0, 60.0]),
    "dump_uturn_2":  np.array([130.0, 40.0]),
    "dump_uturn_3":  np.array([140.0, 100.0]),
    "dump_zone":     np.array([130.0, 160.0]),
}

# 2. Define the connections (edges)
# Now we have two paths from start_exit to load_entry!
EDGES = [
    ("start_zone", "start_exit"),
    
    # Path 1: The "Main" U-turn
    ("start_exit", "main_1"),
    ("main_1", "main_2"),
    ("main_2", "main_3"),
    ("main_3", "load_entry"),
    
    # Path 2: The "Shortcut"
    ("start_exit", "shortcut_1"),
    ("shortcut_1", "shortcut_2"),
    ("shortcut_2", "load_entry"),
    
    # Rest of the path
    ("load_entry", "load_zone"),
    ("load_zone", "dump_uturn_1"),
    ("dump_uturn_1", "dump_uturn_2"),
    ("dump_uturn_2", "dump_uturn_3"),
    ("dump_uturn_3", "dump_zone"),
    ("dump_zone", "start_zone") 
]

# (We no longer need ROUTE_TO_LOAD or ROUTE_TO_DUMP lists)

# --- Main Settings ---
SMOOTH_ITERATIONS = 5 # Smoothing for the *active* path (Chaikin)

# --- Pygame Display ---
WIDTH, HEIGHT = 1000, 800
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLUE_ACTIVE = (0, 100, 200, 150) # Active path color
RED = (255, 0, 0)
PURPLE_NODE = (150, 0, 150)
CAR_COLOR = (0, 80, 200)
ROAD_WIDTH_M = 5.0 # Road width in meters
ZOOM_FACTOR = 1.1
PADDING = 50

# --- Pixel to Meter conversion ---
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS

# --- Car Physics Parameters (Unchanged) ---
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

# --- Controller Parameters (Unchanged) ---
LOOKAHEAD_GAIN = 0.8
LOOKAHEAD_MIN_M = 4.0
LOOKAHEAD_MAX_M = 15.0
MAX_LAT_ACCEL = 2.0
SENSOR_NOISE_STD_DEV = 0.5

# --- Physics Functions (Unchanged) ---
def resist_forces(v_ms: float, mass_kg: float) -> float:
    return CRR * mass_kg * 9.81 + 0.5 * 1.225 * CD * FRONTAL_AREA * v_ms**2

def traction_force_from_power(v_ms: float, throttle: float) -> float:
    return (P_MAX_W * np.clip(throttle, 0, 1)) / max(v_ms, 0.5)

def brake_force_from_command(brake_cmd: float, mass_kg: float) -> float:
    return np.clip(brake_cmd, 0, 1) * mass_kg * MAX_BRAKE_DECEL

# --- NEW: A* Pathfinding Code ---
def build_weighted_graph(nodes: dict, edges: list) -> dict:
    """
    Builds a weighted adjacency list for the A* algorithm.
    The 'cost' of each edge is its real-world distance in meters.
    """
    graph = {name: [] for name in nodes}
    for n1_name, n2_name in edges:
        p1 = nodes[n1_name]
        p2 = nodes[n2_name]
        distance = np.linalg.norm(p1 - p2)
        
        # Add the edge in both directions
        graph[n1_name].append((n2_name, distance))
        graph[n2_name].append((n1_name, distance))
    return graph

def a_star_pathfinding(graph: dict, nodes_coords: dict, start_name: str, goal_name: str) -> list[np.ndarray]:
    """
    Finds the shortest path using A* algorithm.
    Returns a list of node coordinates (np.ndarray) for the path.
    """
    
    # Heuristic function (h): Euclidean distance "as the crow flies"
    def h(node_name):
        return np.linalg.norm(nodes_coords[node_name] - nodes_coords[goal_name])

    # Priority queue: stores (f_score, node_name)
    open_set = [(h(start_name), start_name)]
    
    # came_from[node] = previous_node
    came_from = {}
    
    # g_score[node] = cost from start to node
    g_score = {name: float('inf') for name in nodes_coords}
    g_score[start_name] = 0
    
    # f_score[node] = g_score + h
    f_score = {name: float('inf') for name in nodes_coords}
    f_score[start_name] = h(start_name)

    while open_set:
        # Get the node with the lowest f_score
        current_f, current_name = heapq.heappop(open_set)
        
        if current_name == goal_name:
            # --- Path Found! Reconstruct it ---
            path_nodes = []
            temp = current_name
            # Need to loop until temp is start_name, which won't be in came_from
            while temp in came_from:
                path_nodes.append(nodes_coords[temp])
                temp = came_from[temp]
            path_nodes.append(nodes_coords[start_name]) # Add the start node
            return list(reversed(path_nodes)) # Return from start to goal

        # Check neighbors
        for neighbor_name, weight in graph[current_name]:
            tentative_g_score = g_score[current_name] + weight
            
            if tentative_g_score < g_score[neighbor_name]:
                # This path is better than the previous one. Record it!
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative_g_score
                f_score[neighbor_name] = tentative_g_score + h(neighbor_name)
                
                # Add to priority queue
                if (f_score[neighbor_name], neighbor_name) not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor_name], neighbor_name))
                    
    print(f"A* Error: Path not found from {start_name} to {goal_name}")
    return [] # Path not found

# --- NEW: Catmull-Rom Spline Path Generation ---
def catmull_rom_point(t, p0, p1, p2, p3):
    """
    Calculates a single point on a Catmull-Rom spline.
    t: 0.0 to 1.0 (parameter)
    p0, p1, p2, p3: The four control points (as np.array)
    """
    # Catmull-Rom matrix calculation
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t**2) +
        (-p0 + 3 * p1 - 3 * p2 + p3) * (t**3)
    )

def generate_curvy_path_from_nodes(node_list: list[np.ndarray], points_per_segment=20) -> list[np.ndarray]:
    """
    Creates a detailed list of waypoints by connecting nodes
    with a smooth Catmull-Rom spline.
    """
    all_waypoints_m = []
    
    if not node_list or len(node_list) < 2:
        return []

    # Iterate through the segments (from the 2nd point to the 2nd-to-last)
    # A spline segment from p1 to p2 requires 4 points: p0, p1, p2, p3
    for i in range(len(node_list) - 1):
        
        # --- Handle boundary conditions for the 4 points ---
        # If we're at the first point (i=0), p0 doesn't exist.
        # We "ghost" it by using p1, which creates a smooth start.
        p0 = node_list[0] if i == 0 else node_list[i - 1]
        
        # p1 is the start of our current segment
        p1 = node_list[i]
        
        # p2 is the end of our current segment
        p2 = node_list[i + 1]
        
        # If we're at the last segment, p3 doesn't exist.
        # We "ghost" it by using p2, which creates a smooth end.
        p3 = node_list[-1] if i >= len(node_list) - 2 else node_list[i + 2]
        
        # --- Generate the points for this segment ---
        # Add the first point of the segment (p1)
        if i == 0:
            all_waypoints_m.append(p1)
            
        # Generate the interpolated points (t from 0 to 1)
        # We start from j=1 because we already added the '0' point (p1)
        for j in range(1, points_per_segment + 1):
            t = j / float(points_per_segment)
            point = catmull_rom_point(t, p0, p1, p2, p3)
            all_waypoints_m.append(point)

    print(f"Generated curvy path with {len(all_waypoints_m)} waypoints.")
    return all_waypoints_m

# --- Path Class (Unchanged) ---
class Path:
    def __init__(self, waypoints: list[np.ndarray]):
        if not waypoints or len(waypoints) < 2:
            print("Warning: Path created with < 2 waypoints. Using fallback.")
            self.wp = [np.array([0,0]), np.array([1,1])]
        else:
            self.wp = waypoints
            
        self.s = [0.0]  # s-coordinates (distance along path)
        for i in range(1, len(self.wp)):
            dist = np.linalg.norm(self.wp[i] - self.wp[i-1])
            self.s.append(self.s[-1] + dist)
        
        self.length = self.s[-1]
        if self.length < 1e-6:
            print("Warning: Path length is near zero.")
            self.length = 1e-6 # Avoid division by zero

    def get_segment_index(self, s: float) -> int:
        s = np.clip(s, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]:
                return i
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

        min_dist_sq = float('inf')
        best_s = s_hint

        for i in range(idx_start, idx_end):
            a, b = self.wp[i], self.wp[i + 1]
            seg_vec = b - a
            ap = p - a
            dot_val = np.dot(seg_vec, seg_vec)
            
            if dot_val < 1e-9:
                t = 0.0
            else:
                t = np.clip(np.dot(ap, seg_vec) / dot_val, 0, 1)
            
            proj = a + t * seg_vec
            dist_sq = np.sum((p - proj)**2)
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_s = self.s[i] + t * np.linalg.norm(seg_vec)
                
        return best_s

    def get_curvature_at(self, s: float) -> float:
        s_before = self.point_at(s - 1.0) # 1 meter behind
        s_at = self.point_at(s)
        s_after = self.point_at(s + 1.0) # 1 meter ahead
        
        area = 0.5 * abs(s_before[0]*(s_at[1]-s_after[1]) + s_at[0]*(s_after[1]-s_before[1]) + s_after[0]*(s_before[1]-s_at[1]))
        d1 = np.linalg.norm(s_at - s_before)
        d2 = np.linalg.norm(s_after - s_at)
        d3 = np.linalg.norm(s_after - s_before)
        
        if d1*d2*d3 < 1e-6: return 0.0
        return (4 * area) / (d1 * d2 * d3)

# --- Kalman Filter Class (Unchanged) ---
class KalmanFilter:
    def __init__(self, dt):
        self.x = np.zeros(4)  # [x_m, vx_ms, y_m, vy_ms]
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
            # Handle singular matrix, e.g., by skipping update
            print("Warning: Kalman filter update skipped (singular matrix)")
            pass


# --- Enhanced Car Class (Unchanged) ---
class Car:
    def __init__(self, x_m, y_m, angle=0):
        self.x_m = x_m
        self.y_m = y_m
        self.angle = angle
        self.speed_ms = 0.0
        self.accel_ms2 = 0.0
        self.steer_angle = 0.0
        
        self.current_mass_kg = MASS_KG
        self.op_state = "GOING_TO_ENDPOINT"
        self.op_timer = 0.0
        self.a_cmd_prev = 0.0
        self.s_path_m = 0.0

    def move(self, accel_cmd, steer_input, dt):
        # 1. Jerk limiting
        accel_cmd = np.clip(
            accel_cmd,
            self.a_cmd_prev - JERK_LIMIT * dt,
            self.a_cmd_prev + JERK_LIMIT * dt
        )
        self.a_cmd_prev = accel_cmd

        # 2. Calculate forces
        F_resist = resist_forces(self.speed_ms, self.current_mass_kg)
        F_needed = self.current_mass_kg * accel_cmd

        if F_needed >= 0:
            F_engine = F_needed + F_resist
            throttle = (F_engine * max(self.speed_ms, 0.5)) / P_MAX_W
            brake = 0.0
        else:
            F_brake_req = -F_needed + F_resist
            brake = F_brake_req / (self.current_mass_kg * MAX_BRAKE_DECEL)
            throttle = 0.0

        # 3. Apply steering rate limit
        self.steer_angle = np.clip(
            steer_input,
            self.steer_angle - STEER_RATE_RADPS * dt,
            self.steer_angle + STEER_RATE_RADPS * dt
        )

        # 4. Calculate net force and update kinematics
        F_trac = traction_force_from_power(self.speed_ms, throttle)
        F_brake = brake_force_from_command(brake, self.current_mass_kg)
        F_net = F_trac - F_resist - F_brake

        self.accel_ms2 = F_net / self.current_mass_kg
        self.speed_ms = max(0.0, self.speed_ms + self.accel_ms2 * dt)

        # 5. Kinematic bicycle model
        if abs(self.steer_angle) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(self.steer_angle)
            angular_velocity = self.speed_ms / turn_radius
            self.angle += angular_velocity * dt

        # 6. Update position
        self.x_m += self.speed_ms * math.cos(self.angle) * dt
        self.y_m += self.speed_ms * math.sin(self.angle) * dt

    def update_op_state(self, dt, path_length_m, current_s_m):
        """
        Updates the car's operational state.
        Returns: (direction, target_speed_cap)
        """
        if self.op_state == "GOING_TO_ENDPOINT":
            if current_s_m >= path_length_m - 2.0 and self.speed_ms < 0.5:
                self.op_state = "LOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
            return +1, SPEED_MS_EMPTY
            
        elif self.op_state == "LOADING":
            if self.op_timer > 0:
                self.op_timer -= dt
            else:
                self.op_state = "RETURNING_TO_START" # Main loop will catch this
                self.current_mass_kg = MASS_KG + CARGO_TON * 1000
            return 0, 0.0
            
        elif self.op_state == "RETURNING_TO_START":
            if current_s_m <= 2.0 and self.speed_ms < 0.5:
                self.op_state = "UNLOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
            return -1, SPEED_MS_LOADED # NOTE: This path goes backward (s=max to s=0)
            
        elif self.op_state == "UNLOADING":
            if self.op_timer > 0:
                self.op_timer -= dt
            else:
                self.op_state = "GOING_TO_ENDPOINT" # Main loop will catch this
                self.current_mass_kg = MASS_KG
            return 0, 0.0
        
        return 0, 0.0 # Default case

    def get_noisy_measurement(self):
        return np.array([self.x_m + np.random.normal(0, SENSOR_NOISE_STD_DEV),
                         self.y_m + np.random.normal(0, SENSOR_NOISE_STD_DEV)])

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

# --- Path Processing Functions ---
def chaikin_smoother(points, iterations):
    """
    Smooths a path using Chaikin's algorithm.
    This is used for the *active* path visualization, not the generation.
    """
    for _ in range(iterations):
        new_points = []
        if not points: return []
        new_points.append(points[0])
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            q = (1 - 0.25) * p1 + 0.25 * p2
            r = 0.25 * p1 + (1 - 0.25) * p2
            new_points.extend([q, r])
        new_points.append(points[-1])
        points = new_points
    return points

# --- Coordinate and Drawing Functions ---
def grid_to_screen(pos_m, scale, pan):  
    """Converts METERS (grid) to screen pixels."""
    pos_m_np = np.array(pos_m) 
    pos_px = pos_m_np * METERS_TO_PIXELS
    return (int(pos_px[0] * scale + pan[0]), int(pos_px[1] * scale + pan[1]))

def screen_to_grid(pos_px, scale, pan):  
    """Converts screen pixels to METERS (grid)."""
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return (grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS)

def draw_active_path(screen, path: Path, g_to_s, scale):
    """Draws a smoothed road for the ACTIVE Path object (in meters)."""
    if len(path.wp) < 2: return
    
    # We use Chaikin here for a slightly different visual style on the active path
    smoothed_points_m = chaikin_smoother(path.wp, SMOOTH_ITERATIONS)
    road_px = [g_to_s(p) for p in smoothed_points_m]
    
    if len(road_px) > 1:
        # Create a temporary surface for alpha drawing
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.lines(s, BLUE_ACTIVE, False, road_px, max(2, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale)))
        screen.blit(s, (0,0))

# --- MODIFIED: Function to draw the entire network with CURVES ---
def draw_road_network(screen, g_to_s, scale):
    """
    Draws the entire road graph with smooth Catmull-Rom splines for each edge.
    It also draws the nodes.
    """
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    points_per_mini_segment = 5 # How many points to generate for each small edge spline

    # Convert node names to a list for easier indexing
    # We use the *order* from the dictionary keys.
    node_names_list = list(NODES.keys())
    node_name_to_index = {name: i for i, name in enumerate(node_names_list)}
    
    # Iterate through all defined EDGES to draw the full network
    for node1_name, node2_name in EDGES:
        p1_idx = node_name_to_index[node1_name]
        p2_idx = node_name_to_index[node2_name]

        # p0: Node before p1.
        p0_idx = max(0, p1_idx - 1)
        p0 = NODES[node_names_list[p0_idx]]

        # p1: The current start node of our edge
        p1 = NODES[node1_name]

        # p2: The current end node of our edge
        p2 = NODES[node2_name]

        # p3: Node after p2.
        p3_idx = min(len(node_names_list) - 1, p2_idx + 1)
        p3 = NODES[node_names_list[p3_idx]]

        # Generate points for this specific edge spline
        segment_points_m = []
        for j in range(points_per_mini_segment + 1):
            t = j / float(points_per_mini_segment)
            point = catmull_rom_point(t, p0, p1, p2, p3)
            segment_points_m.append(point)
        
        # Convert to screen coordinates and draw the line
        segment_points_px = [g_to_s(p) for p in segment_points_m]
        if len(segment_points_px) > 1:
            pygame.draw.lines(screen, GRAY, False, segment_points_px, road_width_px)
            
    # Draw nodes as circles (on top, so they are always visible)
    for node_name, pos_m in NODES.items():
        pygame.draw.circle(screen, PURPLE_NODE, g_to_s(pos_m), 6)


# --- Main Simulation ---
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("A* Pathfinding Simulation with Spline Roads")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # --- NEW: Step 1: Build the Graph and Find Paths with A* ---
    
    # Build the "brain" for the A* algorithm
    print("Building weighted road network graph...")
    road_graph = build_weighted_graph(NODES, EDGES)

    # --- Calculate the path to the load zone ---
    print("A* calculating path from 'start_zone' to 'load_zone'...")
    route_nodes_to_load = a_star_pathfinding(road_graph, NODES, "start_zone", "load_zone")
    
    if not route_nodes_to_load:
        print("FATAL ERROR: A* could not find path to load_zone. Exiting.")
        return
        
    print(f"A* found 'TO_LOAD' path with {len(route_nodes_to_load)} nodes.")
    waypoints_to_load_m = generate_curvy_path_from_nodes(route_nodes_to_load)
    path_to_load = Path(waypoints_to_load_m)


    # --- Calculate the path from the load zone back to the start ---
    print("A* calculating path from 'load_zone' to 'start_zone'...")
    # This will find the shortest path via the dump_zone
    route_nodes_to_dump = a_star_pathfinding(road_graph, NODES, "load_zone", "start_zone")
    
    if not route_nodes_to_dump:
        print("FATAL ERROR: A* could not find path from load_zone. Exiting.")
        return

    print(f"A* found 'TO_DUMP' path with {len(route_nodes_to_dump)} nodes.")
    # We must REVERSE the node list for the spline generator
    # because the state machine travels s=max to s=0 on the return trip.
    waypoints_to_dump_m = generate_curvy_path_from_nodes(list(reversed(route_nodes_to_dump)))
    path_to_dump = Path(waypoints_to_dump_m)

    # --- End of A* setup ---

    current_path = path_to_load
    path_length_m = current_path.length

    # --- Setup Car, KF, and View ---
    if not waypoints_to_load_m:
        print("ERROR: 'path_to_load' is empty. Cannot start.")
        return

    initial_angle = math.atan2(waypoints_to_load_m[1][1] - waypoints_to_load_m[0][1], 
                               waypoints_to_load_m[1][0] - waypoints_to_load_m[0][0])
    car = Car(waypoints_to_load_m[0][0], waypoints_to_load_m[0][1], angle=initial_angle)
    
    kf = KalmanFilter(dt=1.0/60.0)
    kf.x = np.array([car.x_m, 0, car.y_m, 0])
    
    # --- Base camera view on NODES ---
    all_nodes_m = list(NODES.values())
    min_x_m = min(p[0] for p in all_nodes_m)
    max_x_m = max(p[0] for p in all_nodes_m)
    min_y_m = min(p[1] for p in all_nodes_m)
    max_y_m = max(p[1] for p in all_nodes_m)

    img_w_m = max(1.0, max_x_m - min_x_m)
    img_h_m = max(1.0, max_y_m - min_y_m)
    
    scale = min((WIDTH - PADDING * 2) / (img_w_m * METERS_TO_PIXELS), 
                (HEIGHT - PADDING * 2) / (img_h_m * METERS_TO_PIXELS))
    
    pan = [
        PADDING - (min_x_m * METERS_TO_PIXELS * scale),
        PADDING - (min_y_m * METERS_TO_PIXELS * scale)
    ]

    mouse_dragging = False
    last_mouse_pos = None

    # --- Main Loop ---
    running = True
    desired_speed_ms = 0.0 # Initialize for HUD
    
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    mouse_dragging = True
                    last_mouse_pos = event.pos
                elif event.button in (4, 5):
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos_m = screen_to_grid(event.pos, scale, pan)
                    scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos_m, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]
                    pan[1] += event.pos[1] - new_screen_pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: 
                mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0] += dx
                pan[1] += dy
                last_mouse_pos = event.pos

        # --- Step 2: Get State (all in METERS) ---
        est_pos_m = np.array([kf.x[0], kf.x[2]])
        est_vel_m = np.array([kf.x[1], kf.x[3]])
        est_speed_ms = np.linalg.norm(est_vel_m)
        
        est_s_path_m = current_path.project(est_pos_m, car.s_path_m)
        car.s_path_m = est_s_path_m 

        # --- Step 3: Update State Machine ---
        prev_op_state = car.op_state
        direction, base_speed_ms = car.update_op_state(dt, path_length_m, est_s_path_m)
        new_op_state = car.op_state

        # --- Path and State-Machine Management ---
        if new_op_state != prev_op_state:
            print(f"State change: {prev_op_state} -> {new_op_state}")
            
            if new_op_state == "RETURNING_TO_START":
                current_path = path_to_dump
                car.s_path_m = current_path.project(est_pos_m, current_path.length) 
                print(f"Swapped to 'path_to_dump' (Length: {current_path.length:.1f}m)")

            elif new_op_state == "GOING_TO_ENDPOINT":
                current_path = path_to_load
                car.s_path_m = current_path.project(est_pos_m, 0.0)
                print(f"Swapped to 'path_to_load' (Length: {current_path.length:.1f}m)")
        
        path_length_m = current_path.length 

        # --- Step 4: Controller Logic ---
        if direction != 0:
            # === STEERING (Pure Pursuit) ===
            ld = np.clip(est_speed_ms * LOOKAHEAD_GAIN, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)
            s_target_steer = est_s_path_m + direction * ld
            p_target = current_path.point_at(s_target_steer)
            
            dx_local = (p_target[0] - est_pos_m[0]) * math.cos(car.angle) + \
                       (p_target[1] - est_pos_m[1]) * math.sin(car.angle)
            dy_local = -(p_target[0] - est_pos_m[0]) * math.sin(car.angle) + \
                        (p_target[1] - est_pos_m[1]) * math.cos(car.angle)
            
            alpha = math.atan2(dy_local, max(dx_local, 1e-3))
            steer_input = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld)
            steer_input = np.clip(steer_input, -STEER_MAX_RAD, STEER_MAX_RAD)

            # === SPEED (Curvature & Braking) ===
            lookahead_dist_speed = 15.0
            max_curvature = 0.0
            
            check_steps = np.linspace(0, lookahead_dist_speed, 10)
            for step in check_steps:
                s_check = est_s_path_m + direction * step
                curvature = current_path.get_curvature_at(s_check)
                max_curvature = max(max_curvature, abs(curvature))

            if max_curvature > 1e-4:
                v_turn_cap = math.sqrt(MAX_LAT_ACCEL / max_curvature)
            else:
                v_turn_cap = base_speed_ms
            
            target_s = 0.0 if direction == -1 else path_length_m
            dist_to_target = abs(target_s - est_s_path_m)
            v_stop_cap = math.sqrt(2 * MAX_BRAKE_DECEL * max(0, dist_to_target - 1.0))

            desired_speed_ms = min(base_speed_ms, v_turn_cap, v_stop_cap, 50.0/3.6) # Hard cap
            speed_error_ms = desired_speed_ms - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / max(0.4, dt), -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)

        else: # (direction == 0) -> Loading/unloading
            steer_input = 0.0
            speed_error_ms = 0.0 - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / max(0.4, dt), -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
            desired_speed_ms = 0.0 

        # --- Step 5: Update Physics and KF ---
        car.move(accel_cmd, steer_input, dt)

        accel_vec_m = np.array([
            car.accel_ms2 * math.cos(car.angle),
            car.accel_ms2 * math.sin(car.angle)
        ])
        kf.predict(u=accel_vec_m)
        kf.update(z=car.get_noisy_measurement())

        # --- Step 6: Drawing ---
        screen.fill(WHITE)
        
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)
        g_to_s.scale = scale
        
        draw_road_network(screen, g_to_s, scale)     # Draw the whole graph (curvy gray lines)
        draw_active_path(screen, current_path, g_to_s, scale) # Draw the active path (blue)
        car.draw(screen, g_to_s)

        kf_pos_m = (kf.x[0], kf.x[2])
        kf_screen = g_to_s(kf_pos_m)
        pygame.draw.circle(screen, RED, kf_screen, 6, 2)

        # HUD
        speed_kmph = car.speed_ms * 3.6
        hud_texts = [
            f"Speed: {speed_kmph:.1f} km/h (Target: {desired_speed_ms*3.6:.1f})",
            f"Mass: {car.current_mass_kg:.0f} kg",
            f"State: {car.op_state}",
            f"Accel: {car.accel_ms2:.2f} m/s² (Cmd: {car.a_cmd_prev:.2f})",
            f"Steer: {math.degrees(car.steer_angle):.1f} deg",
            f"Path: {est_s_path_m:.1f}m / {path_length_m:.1f}m (Dir: {direction})"
        ]

        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 22))

        pygame.display.flip()

    pygame.quit()

# --- Main Block ---
if __name__ == '__main__':
    print("Starting graph-based simulation with A* and Splines...")
    run_simulation()