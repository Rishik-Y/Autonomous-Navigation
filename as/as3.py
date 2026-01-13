import pygame
import numpy as np
import math

# --- Constants ---
# --- NEW: Define your mine layout as a Graph ---
# 1. Define the key locations (nodes) in METERS# --- NEW: Define your "Coal Mine" layout as a Graph ---
# We add more nodes to define the *shape* of the curves
NODES = {
    # Start area (bottom left)
    "start_zone":   np.array([20.0, 150.0]),
    "start_exit":   np.array([50.0, 150.0]),
    
    # First big U-turn (going up and right)
    "uturn_1_entry": np.array([100.0, 140.0]),
    "uturn_1_apex":  np.array([120.0, 120.0]), # <-- The point of the curve
    "uturn_1_exit":  np.array([100.0, 100.0]),
    
    # Loading area (top left)
    "load_entry":    np.array([50.0, 90.0]),
    "load_zone":     np.array([30.0, 80.0]),
    
    # Second U-turn (the "dump" route)
    "dump_uturn_1":  np.array([10.0, 60.0]),
    "dump_uturn_2":  np.array([130.0, 40.0]), # <-- A big wide turn
    "dump_uturn_3":  np.array([140.0, 100.0]),
    
    # Dump area
    "dump_zone":     np.array([130.0, 160.0]),
}

# 2. Define the connections (edges) - This is our "visual" map
EDGES = [
    ("start_zone", "start_exit"),
    ("start_exit", "uturn_1_entry"),
    ("uturn_1_entry", "uturn_1_apex"),
    ("uturn_1_apex", "uturn_1_exit"),
    ("uturn_1_exit", "load_entry"),
    ("load_entry", "load_zone"),
    
    # The return path
    ("load_zone", "dump_uturn_1"),
    ("dump_uturn_1", "dump_uturn_2"),
    ("dump_uturn_2", "dump_uturn_3"),
    ("dump_uturn_3", "dump_zone"),
    
    # Path from dump back to start
    ("dump_zone", "start_zone") 
]

# 3. Define the *actual routes* the car will take
#    These lists are now the "control points" for our spline
ROUTE_TO_LOAD = [
    NODES["start_zone"], 
    NODES["start_exit"],
    NODES["uturn_1_entry"],
    NODES["uturn_1_apex"],
    NODES["uturn_1_exit"],
    NODES["load_entry"],
    NODES["load_zone"]
]

ROUTE_TO_DUMP = [
    NODES["load_zone"],
    NODES["dump_uturn_1"],
    NODES["dump_uturn_2"],
    NODES["dump_uturn_3"],
    NODES["dump_zone"],
    NODES["start_zone"] # <-- Go all the way back to the start
]
# --- Main Settings ---
SMOOTH_ITERATIONS = 5

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
# This is an estimate; you may need to tune it.
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

# # --- NEW: Path Generation Function (from your solution) ---
# def generate_path_from_nodes(node_list: list[np.ndarray]) -> list[np.ndarray]:
#     """
#     Creates a detailed list of waypoints by connecting nodes
#     with straight lines.
#     """
#     all_waypoints_m = []
    
#     # Use a smaller step size for smoother paths
#     WAYPOINT_SPACING_M = 1.0  # Generate a waypoint every 1 meter

#     if not node_list:
#         return []

#     # Add the first point
#     all_waypoints_m.append(node_list[0])

#     for i in range(len(node_list) - 1):
#         p_start = node_list[i]
#         p_end = node_list[i+1]
        
#         vec = p_end - p_start
#         dist = np.linalg.norm(vec)
#         if dist < 1e-6: continue
        
#         direction = vec / dist
        
#         # Add points along the line
#         num_steps = int(dist / WAYPOINT_SPACING_M)
#         for j in range(1, num_steps + 1): # Start from 1, we already added p_start
#             all_waypoints_m.append(p_start + direction * (j * WAYPOINT_SPACING_M))
            
#     # Ensure the very last point is added if spacing wasn't perfect
#     if np.linalg.norm(all_waypoints_m[-1] - node_list[-1]) > 1e-3:
#         all_waypoints_m.append(node_list[-1])
        
#     print(f"Generated path with {len(all_waypoints_m)} waypoints.")
#     return all_waypoints_m

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
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

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

# --- Path Processing Functions (Unchanged, work in pixels) ---
def chaikin_smoother(points, iterations):
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

# --- MODIFIED: Renamed to draw_active_path and draws in BLUE ---
def draw_active_path(screen, path: Path, g_to_s, scale):
    """Draws a smoothed road for the ACTIVE Path object (in meters)."""
    if len(path.wp) < 2: return
    
    smoothed_points_m = chaikin_smoother(path.wp, SMOOTH_ITERATIONS)
    road_px = [g_to_s(p) for p in smoothed_points_m]
    
    if len(road_px) > 1:
        # Create a temporary surface for alpha drawing
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.lines(s, BLUE_ACTIVE, False, road_px, max(2, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale)))
        screen.blit(s, (0,0))

# --- NEW/MODIFIED: Function to draw the entire road network graph with splines ---
def draw_road_network(screen, g_to_s, scale):
    """
    Draws the entire road graph with smooth Catmull-Rom splines for each edge.
    It also draws the nodes.
    """
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    points_per_mini_segment = 5 # How many points to generate for each small edge spline

    # Convert node names to a list for easier indexing
    node_names_list = list(NODES.keys())
    
    # Iterate through all defined EDGES to draw the full network
    for node1_name, node2_name in EDGES:
        p1_idx = node_names_list.index(node1_name)
        p2_idx = node_names_list.index(node2_name)

        # To draw a spline segment from p1 to p2, we need p0 and p3 as neighbors.
        # We'll 'ghost' points if they don't exist, similar to generate_curvy_path_from_nodes.

        # p0: Node before p1. If p1 is the first node, use p1 itself.
        p0_name = node_names_list[max(0, p1_idx - 1)]
        p0 = NODES[p0_name] if p1_idx > 0 else NODES[node1_name] # Clamp to p1 if at start

        # p1: The current start node of our edge
        p1 = NODES[node1_name]

        # p2: The current end node of our edge
        p2 = NODES[node2_name]

        # p3: Node after p2. If p2 is the last node, use p2 itself.
        p3_name = node_names_list[min(len(node_names_list) - 1, p2_idx + 1)]
        p3 = NODES[p3_name] if p2_idx < len(node_names_list) - 1 else NODES[node2_name] # Clamp to p2 if at end

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
# --- MODIFIED: No longer takes waypoints_px ---
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Graph-Based Physics Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

   # --- NEW: Step 1: Generate all paths from the Graph ---
    print("Generating 'TO_LOAD' path...")
    waypoints_to_load_m = generate_curvy_path_from_nodes(ROUTE_TO_LOAD) # <-- CHANGED
    path_to_load = Path(waypoints_to_load_m)

    print("Generating 'TO_DUMP' path...")
    waypoints_to_dump_m = generate_curvy_path_from_nodes(list(reversed(ROUTE_TO_DUMP))) # <-- CHANGED
    path_to_dump = Path(waypoints_to_dump_m)
    # --- END MODIFIED ---

    current_path = path_to_load
    path_length_m = current_path.length

    # --- Setup Car, KF, and View (ALL IN METERS) ---
    if not waypoints_to_load_m:
        print("ERROR: 'path_to_load' is empty. Cannot start.")
        return

    initial_angle = math.atan2(waypoints_to_load_m[1][1] - waypoints_to_load_m[0][1], 
                               waypoints_to_load_m[1][0] - waypoints_to_load_m[0][0])
    car = Car(waypoints_to_load_m[0][0], waypoints_to_load_m[0][1], angle=initial_angle)
    
    kf = KalmanFilter(dt=1.0/60.0)
    kf.x = np.array([car.x_m, 0, car.y_m, 0])
    
    # --- MODIFIED: Base camera view on NODES, not image size ---
    all_nodes_m = list(NODES.values())
    min_x_m = min(p[0] for p in all_nodes_m)
    max_x_m = max(p[0] for p in all_nodes_m)
    min_y_m = min(p[1] for p in all_nodes_m)
    max_y_m = max(p[1] for p in all_nodes_m)

    img_w_m = max_x_m - min_x_m
    img_h_m = max_y_m - min_y_m
    
    if img_w_m < 1 or img_h_m < 1:
        print("Warning: Invalid node dimensions. Using fallback zoom.")
        img_w_m, img_h_m = 100, 100

    scale = min((WIDTH - PADDING * 2) / (img_w_m * METERS_TO_PIXELS), 
                (HEIGHT - PADDING * 2) / (img_h_m * METERS_TO_PIXELS))
    
    # Pan to center the graph
    pan = [
        PADDING - (min_x_m * METERS_TO_PIXELS * scale),
        PADDING - (min_y_m * METERS_TO_PIXELS * scale)
    ]
    # --- END MODIFIED ---

    mouse_dragging = False
    last_mouse_pos = None

    # --- Main Loop ---
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
        car.s_path_m = est_s_path_m # Update the car's last known 's'

        # --- Step 3: Update State Machine ---
        prev_op_state = car.op_state
        direction, base_speed_ms = car.update_op_state(dt, path_length_m, est_s_path_m)
        new_op_state = car.op_state

        # --- NEW: Path and State-Machine Management ---
        if new_op_state != prev_op_state:
            print(f"State change: {prev_op_state} -> {new_op_state}")
            
            # Finished loading, swap to DUMP path
            if new_op_state == "RETURNING_TO_START":
                current_path = path_to_dump
                # Project car onto the *end* of the new path (since it's reversed)
                car.s_path_m = current_path.project(est_pos_m, current_path.length) 
                print(f"Swapped to 'path_to_dump' (Length: {current_path.length:.1f}m)")

            # Finished unloading, swap to LOAD path
            elif new_op_state == "GOING_TO_ENDPOINT":
                current_path = path_to_load
                # Project car onto the *start* of the new path
                car.s_path_m = current_path.project(est_pos_m, 0.0)
                print(f"Swapped to 'path_to_load' (Length: {current_path.length:.1f}m)")
        
        path_length_m = current_path.length # Update path length for controllers
        # --- END NEW ---

        # --- Step 4: Controller Logic (Unchanged, works on current_path) ---
        if direction != 0:
            # === ADVANCED STEERING (Pure Pursuit) ===
            ld = np.clip(est_speed_ms * LOOKAHEAD_GAIN, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)
            # --- MODIFIED: Use `path_length_m` for s_target_steer calculation ---
            # When direction is -1, we subtract ld from s.
            s_target_steer = est_s_path_m + direction * ld
            p_target = current_path.point_at(s_target_steer)
            # --- END MODIFIED ---
            
            dx_local = (p_target[0] - est_pos_m[0]) * math.cos(car.angle) + \
                       (p_target[1] - est_pos_m[1]) * math.sin(car.angle)
            dy_local = -(p_target[0] - est_pos_m[0]) * math.sin(car.angle) + \
                        (p_target[1] - est_pos_m[1]) * math.cos(car.angle)
            
            alpha = math.atan2(dy_local, max(dx_local, 1e-3))
            steer_input = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld)
            steer_input = np.clip(steer_input, -STEER_MAX_RAD, STEER_MAX_RAD)

            # === ADVANCED SPEED (Curvature & Braking) ===
            lookahead_dist_speed = 15.0
            max_curvature = 0.0
            
            check_steps = np.linspace(0, lookahead_dist_speed, 10)
            for step in check_steps:
                s_check = est_s_path_m + direction * step
                # No need to check s_check < 0, as point_at() handles clipping
                if s_check > current_path.length:
                    break
                curvature = current_path.get_curvature_at(s_check)
                max_curvature = max(max_curvature, curvature)

            if max_curvature > 1e-4:
                v_turn_cap = math.sqrt(MAX_LAT_ACCEL / max_curvature)
            else:
                v_turn_cap = base_speed_ms
            
            # Target 's' is 0.0 if going backward, or path_length_m if going forward
            target_s = 0.0 if direction == -1 else path_length_m
            dist_to_target = abs(target_s - est_s_path_m)
            v_stop_cap = math.sqrt(2 * MAX_BRAKE_DECEL * max(0, dist_to_target - 1.0)) # Stop 1m early

            desired_speed_ms = min(base_speed_ms, v_turn_cap, v_stop_cap)
            speed_error_ms = desired_speed_ms - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / max(0.4, dt), -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)

        else: # (direction == 0) -> We are loading/unloading
            steer_input = 0.0
            speed_error_ms = 0.0 - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / max(0.4, dt), -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
            desired_speed_ms = 0.0 # For HUD display

        # --- Step 5: Update Physics and KF (Unchanged) ---
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
        
        draw_road_network(screen, g_to_s, scale)     # <-- NEW: Draw the whole graph
        draw_active_path(screen, current_path, g_to_s, scale) # <-- Draw the active path
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

# --- MODIFIED: Main Block ---
# Removed all caching, OS checks, and image loading.
if __name__ == '__main__':
    print("Starting graph-based simulation...")
    run_simulation()
