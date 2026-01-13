import pygame
import cv2
import cv2.ximgproc as ximgproc
import numpy as np
import math
from scipy.spatial import KDTree
import os
import pickle

# --- Constants ---
# --- Main Settings ---
IMAGE_FILE = r'C:\Users\DAIICT A\Downloads\Sub\smtgimg_to_road\gimp.png'
WAYPOINT_CACHE_FILE = 'waypoints_1000.pkl'
WAYPOINT_STEP_SIZE = 10  # How many pixels to step along the thinned line for each waypoint
SMOOTH_ITERATIONS = 5

# --- Pygame Display ---
WIDTH, HEIGHT = 1000, 800
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
CAR_COLOR = (0, 80, 200)
ROAD_WIDTH_M = 5.0 # Road width in meters
ZOOM_FACTOR = 1.1
PADDING = 50

# --- Pixel to Meter conversion ---
# This is an estimate; you may need to tune it.
# If the road image is 800px wide and represents 100m, this value would be 8.0
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS

# --- Car Physics Parameters (From gemini3.py) ---
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

# --- Controller Parameters (From gemini3.py) ---
LOOKAHEAD_GAIN = 0.8        # Gain for Pure Pursuit
LOOKAHEAD_MIN_M = 4.0       # Min look-ahead distance
LOOKAHEAD_MAX_M = 15.0      # Max look-ahead distance
MAX_LAT_ACCEL = 2.0         # Max lateral G-force for turns (m/s^2)
SENSOR_NOISE_STD_DEV = 0.5  # In meters

# --- Physics Functions (FROM GEMINI3.PY) ---
def resist_forces(v_ms: float, mass_kg: float) -> float:
    return CRR * mass_kg * 9.81 + 0.5 * 1.225 * CD * FRONTAL_AREA * v_ms**2

def traction_force_from_power(v_ms: float, throttle: float) -> float:
    return (P_MAX_W * np.clip(throttle, 0, 1)) / max(v_ms, 0.5)

def brake_force_from_command(brake_cmd: float, mass_kg: float) -> float:
    return np.clip(brake_cmd, 0, 1) * mass_kg * MAX_BRAKE_DECEL

# --- NEW: Path Class (FROM GEMINI3.PY) ---
# This class is essential for the advanced controllers.
# It allows us to work with the path in meters (s-coordinates).
class Path:
    def __init__(self, waypoints: list[np.ndarray]):
        if not waypoints:
            raise ValueError("Waypoints list cannot be empty")
            
        self.wp = waypoints
        self.s = [0.0]  # s-coordinates (distance along path)
        for i in range(1, len(self.wp)):
            dist = np.linalg.norm(self.wp[i] - self.wp[i-1])
            self.s.append(self.s[-1] + dist)
        
        self.length = self.s[-1]
        if self.length < 1e-6:
             print("Warning: Path length is near zero.")
             self.length = 1e-6 # Avoid division by zero

    # In class Path:
    def get_segment_index(self, s: float) -> int:
        """Helper to find the segment index that contains a given s-value."""
        s = np.clip(s, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]:
                return i
        return len(self.s) - 2 # Return last segment


    def point_at(self, s_query: float) -> np.ndarray:
        """Gets the (x, y) point in meters at a distance 's' along the path."""
        s = np.clip(s_query, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]:
                s_base, s_end = self.s[i], self.s[i+1]
                p_base, p_end = self.wp[i], self.wp[i + 1]
                if s_end - s_base < 1e-6: return p_base
                t = (s - s_base) / (s_end - s_base)
                return p_base + t * (p_end - p_base)
        return self.wp[-1]

# In class Path:
    def project(self, p: np.ndarray, s_hint: float) -> float:
        """Finds the 's' coordinate closest to point 'p', searching near 's_hint'."""
        
        # 1. Find the segment index for our hint
        idx_hint = self.get_segment_index(s_hint)
        
        # 2. Define a search window (e.g., 20 segments back, 20 segments forward)
        #    This prevents the projection from snapping to the wrong side of the loop.
        search_window = 20
        idx_start = max(0, idx_hint - search_window)
        idx_end = min(len(self.wp) - 1, idx_hint + search_window)

        min_dist_sq = float('inf')
        best_s = s_hint # Default to hint if no better solution found

        # 3. Search only within the window
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
        """Calculates path curvature at 's' (used for speed control)."""
        s_before = self.point_at(s - 1.0) # 1 meter behind
        s_at = self.point_at(s)
        s_after = self.point_at(s + 1.0) # 1 meter ahead
        
        # Using Menger curvature formula
        area = 0.5 * abs(s_before[0]*(s_at[1]-s_after[1]) + s_at[0]*(s_after[1]-s_before[1]) + s_after[0]*(s_before[1]-s_at[1]))
        d1 = np.linalg.norm(s_at - s_before)
        d2 = np.linalg.norm(s_after - s_at)
        d3 = np.linalg.norm(s_after - s_before)
        
        if d1*d2*d3 < 1e-6: return 0.0 # Avoid division by zero if points are colinear
        return (4 * area) / (d1 * d2 * d3)

# --- Kalman Filter Class (Unchanged, but now operates in METERS) ---
class KalmanFilter:
    def __init__(self, dt):
        self.x = np.zeros(4)  # [x_m, vx_ms, y_m, vy_ms]
        self.dt = dt
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt**2, 0], [dt, 0], [0, 0.5 * dt**2], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * (SENSOR_NOISE_STD_DEV**2) # Use sensor noise constant
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

# --- Enhanced Car Class (Operates ENTIRELY in METERS) ---
class Car:
# In class Car:
    def __init__(self, x_m, y_m, angle=0):
        # State is entirely in SI units (meters, radians, m/s)
        self.x_m = x_m
        self.y_m = y_m
        self.angle = angle  # radians
        self.speed_ms = 0.0
        self.accel_ms2 = 0.0
        self.steer_angle = 0.0
        
        self.current_mass_kg = MASS_KG
        self.op_state = "GOING_TO_ENDPOINT"
        self.op_timer = 0.0
        self.a_cmd_prev = 0.0
        self.s_path_m = 0.0  # <--- ADD THIS LINE

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
                self.op_state = "RETURNING_TO_START"
                self.current_mass_kg = MASS_KG + CARGO_TON * 1000
            return 0, 0.0
            
        elif self.op_state == "RETURNING_TO_START":
            if current_s_m <= 2.0 and self.speed_ms < 0.5:
                self.op_state = "UNLOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
            return -1, SPEED_MS_LOADED
            
        elif self.op_state == "UNLOADING":
            if self.op_timer > 0:
                self.op_timer -= dt
            else:
                self.op_state = "GOING_TO_ENDPOINT"
                self.current_mass_kg = MASS_KG
            return 0, 0.0
        
        return 0, 0.0 # Default case

    def get_noisy_measurement(self):
        """Returns noisy position in METERS."""
        return np.array([self.x_m + np.random.normal(0, SENSOR_NOISE_STD_DEV),
                         self.y_m + np.random.normal(0, SENSOR_NOISE_STD_DEV)])

    def draw(self, screen, g_to_s):
        """Draws the car. g_to_s is the meter-to-screen conversion function."""
        car_center_screen = g_to_s((self.x_m, self.y_m))
        
        # Calculate pixel dimensions for drawing
        length_px = CAR_LENGTH_M * METERS_TO_PIXELS * (g_to_s.scale / METERS_TO_PIXELS)
        width_px = CAR_WIDTH_M * METERS_TO_PIXELS * (g_to_s.scale / METERS_TO_PIXELS)

        if length_px < 1 or width_px < 1: return # Don't draw if too small
        
        car_surface = pygame.Surface((length_px, width_px), pygame.SRCALPHA)

        cargo_ratio = (self.current_mass_kg - MASS_KG) / (CARGO_TON * 1000)
        body_color = tuple(np.array(CAR_COLOR) * (1 - cargo_ratio) + np.array([120, 120, 120]) * cargo_ratio)
        
        car_surface.fill(body_color)
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        rect = rotated_surface.get_rect(center=car_center_screen)
        screen.blit(rotated_surface, rect.topleft)

# --- Path Processing Functions (Unchanged, work in pixels) ---
def extract_waypoints_from_mask(road_mask, step_size):
    if road_mask is None or road_mask.size == 0: return []
    thinned = ximgproc.thinning(road_mask)
    points = np.argwhere(thinned > 0)
    if len(points) < 2: return []
    points = points[:, ::-1]
    pixel_tree = KDTree(points)
    start_index = np.argmin(points[:, 1] * thinned.shape[1] + points[:, 0])
    ordered_points = []
    visited_indices = set()
    current_idx = start_index
    while len(visited_indices) < len(points):
        if current_idx in visited_indices: break
        ordered_points.append(points[current_idx])
        visited_indices.add(current_idx)
        distances, indices = pixel_tree.query(points[current_idx], k=10)
        next_idx = -1
        for i in indices:
            if i not in visited_indices: next_idx = i; break
        if next_idx == -1: break
        current_idx = next_idx
    waypoints = [np.array(p) for p in ordered_points[::step_size]]
    if ordered_points and list(waypoints[-1]) != list(ordered_points[-1]):
        waypoints.append(np.array(ordered_points[-1]))
    print(f"Extracted {len(waypoints)} waypoints (in pixels) from image.")
    return waypoints

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
    # --- FIX ---
    # Ensure pos_m is a numpy array for element-wise multiplication
    pos_m_np = np.array(pos_m) 
    # --- END FIX ---
    
    pos_px = pos_m_np * METERS_TO_PIXELS # Convert meters to base pixels
    return (int(pos_px[0] * scale + pan[0]), int(pos_px[1] * scale + pan[1]))

def screen_to_grid(pos_px, scale, pan): 
    """Converts screen pixels to METERS (grid)."""
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return (grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS)

def draw_road(screen, path: Path, g_to_s, scale):
    """Draws a smoothed road from the Path object (in meters)."""
    if len(path.wp) < 2: return
    # We can use the Chaikin smoother on the meter-based waypoints
    smoothed_points_m = chaikin_smoother(path.wp, SMOOTH_ITERATIONS)
    road_px = [g_to_s(p) for p in smoothed_points_m]
    if len(road_px) > 1:
        pygame.draw.lines(screen, GRAY, False, road_px, max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale)))

# --- Main Simulation ---
def run_simulation(waypoints_px):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("FIXED: Enhanced Physics Simulation - OpenCV Road")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # --- Step 1: Convert all path data to METERS ---
    waypoints_m = [wp * PIXELS_TO_METERS for wp in waypoints_px]
    path = Path(waypoints_m)
    path_length_m = path.length

    # --- Setup Car, KF, and View (ALL IN METERS) ---
    initial_angle = math.atan2(waypoints_m[1][1] - waypoints_m[0][1], waypoints_m[1][0] - waypoints_m[0][0])
    car = Car(waypoints_m[0][0], waypoints_m[0][1], angle=initial_angle)
    
    kf = KalmanFilter(dt=1.0/60.0)
    kf.x = np.array([car.x_m, 0, car.y_m, 0]) # Init KF in METERS
    
    img_w_m = max(p[0] for p in waypoints_m)
    img_h_m = max(p[1] for p in waypoints_m)
    
    if img_w_m == 0 or img_h_m == 0:
        print("Error: Invalid path dimensions.")
        return

    scale = min((WIDTH - PADDING * 2) / (img_w_m * METERS_TO_PIXELS), 
                (HEIGHT - PADDING * 2) / (img_h_m * METERS_TO_PIXELS))
    
    pan = [PADDING, PADDING]
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
        
        # --- MODIFIED ---
        # Give the project function a HINT (car.s_path_m) and store the result
        est_s_path_m = path.project(est_pos_m, car.s_path_m)
        car.s_path_m = est_s_path_m # Update the car's last known 's'
        # --- END MODIFIED ---

        # --- Step 3: Update State Machine ---
        direction, base_speed_ms = car.update_op_state(dt, path_length_m, est_s_path_m)

        # --- Step 4: NEW Physics-Based Controller Logic (from gemini3.py) ---
        if direction != 0:
            # === ADVANCED STEERING (Pure Pursuit) ===
            ld = np.clip(est_speed_ms * LOOKAHEAD_GAIN, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)
            s_target_steer = est_s_path_m + direction * ld
            p_target = path.point_at(s_target_steer)
            
            # Convert target to car's local frame
            dx_local = (p_target[0] - est_pos_m[0]) * math.cos(car.angle) + \
                       (p_target[1] - est_pos_m[1]) * math.sin(car.angle)
            dy_local = -(p_target[0] - est_pos_m[0]) * math.sin(car.angle) + \
                        (p_target[1] - est_pos_m[1]) * math.cos(car.angle)
            
            alpha = math.atan2(dy_local, max(dx_local, 1e-3))
            steer_input = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld)
            steer_input = np.clip(steer_input, -STEER_MAX_RAD, STEER_MAX_RAD)

            # === ADVANCED SPEED (Curvature & Braking) ===
            # 1. Check for curvature ahead
            lookahead_dist_speed = 15.0 # How many meters ahead to check for turns
            s_check = est_s_path_m
            max_curvature = 0.0
            
            check_steps = np.linspace(0, lookahead_dist_speed, 10)
            for step in check_steps:
                s_check = est_s_path_m + direction * step
                if s_check > path.length or s_check < 0:
                    break
                curvature = path.get_curvature_at(s_check)
                max_curvature = max(max_curvature, curvature)

            # 2. Calculate speed cap based on curvature
            if max_curvature > 1e-4:
                v_turn_cap = math.sqrt(MAX_LAT_ACCEL / max_curvature)
            else:
                v_turn_cap = base_speed_ms # No curve, use base speed
            
            # 3. Calculate speed cap for stopping at the end
            target_s = path.length if direction == 1 else 0.0
            dist_to_target = abs(target_s - est_s_path_m)
            v_stop_cap = math.sqrt(2 * MAX_BRAKE_DECEL * max(0, dist_to_target))

            # 4. Final target speed and acceleration command
            desired_speed_ms = min(base_speed_ms, v_turn_cap, v_stop_cap)
            speed_error_ms = desired_speed_ms - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / max(0.4, dt), -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)

        else: # (direction == 0) -> We are loading/unloading
            steer_input = 0.0
            # Command a hard brake to ensure the car stops
            speed_error_ms = 0.0 - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / max(0.4, dt), -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)

        # --- Step 5: Update Physics and KF (all in METERS) ---
        car.move(accel_cmd, steer_input, dt)

        accel_vec_m = np.array([
            car.accel_ms2 * math.cos(car.angle),
            car.accel_ms2 * math.sin(car.angle)
        ])
        kf.predict(u=accel_vec_m) # Predict with m/s^2
        kf.update(z=car.get_noisy_measurement()) # Update with noisy meters

        # --- Step 6: Drawing ---
        screen.fill(WHITE)
        
        # Create a closure for the meter-to-screen function
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)
        g_to_s.scale = scale # Attach scale for the car's draw function
        
        draw_road(screen, path, g_to_s, scale)
        car.draw(screen, g_to_s)

        # Draw KF estimate
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
            f"Path: {est_s_path_m:.1f}m / {path_length_m:.1f}m"
        ]

        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 22))

        pygame.display.flip()

    pygame.quit()

# --- Main Block with Caching (Unchanged) ---
if __name__ == '__main__':
    if os.path.exists(WAYPOINT_CACHE_FILE):
        print(f"Loading cached waypoints from '{WAYPOINT_CACHE_FILE}'...")
        with open(WAYPOINT_CACHE_FILE, 'rb') as f:
            final_waypoints_px = pickle.load(f)
    else:
        print("No cache found. Processing image...")
        if not os.path.exists(IMAGE_FILE):
            print(f"ERROR: Image file not found at '{IMAGE_FILE}'. Please check the file name.")
            final_waypoints_px = []
        else:
            img = cv2.imread(IMAGE_FILE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, road_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            final_waypoints_px = extract_waypoints_from_mask(road_mask, step_size=WAYPOINT_STEP_SIZE)

            if final_waypoints_px and len(final_waypoints_px) > 1:
                print(f"Saving waypoints to cache '{WAYPOINT_CACHE_FILE}'...")
                with open(WAYPOINT_CACHE_FILE, 'wb') as f:
                    pickle.dump(final_waypoints_px, f)

    if final_waypoints_px and len(final_waypoints_px) > 1:
        print("Waypoints loaded. Starting enhanced physics simulation...")
        run_simulation(final_waypoints_px)
    else:
        print("Failed to generate or load a valid path. Exiting.")