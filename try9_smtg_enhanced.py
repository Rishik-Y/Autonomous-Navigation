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

# --- NEW: Define a cache file path ---
WAYPOINT_CACHE_FILE = 'waypoints_1000.pkl'
WAYPOINT_STEP_SIZE = 10
SMOOTH_ITERATIONS = 5

# --- Pygame Display ---
WIDTH, HEIGHT = 1000, 800
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
CAR_COLOR = (0, 80, 200)
ROAD_WIDTH_PX = 50
ZOOM_FACTOR = 1.1
PADDING = 50

# --- Car Physics Parameters (ENHANCED FROM GEMINI FILES) ---
# Physical dimensions
CAR_LENGTH_M = 4.5  # meters (from gemini1.py)
CAR_WIDTH_M = 2.0   # meters (from gemini1.py)
WHEELBASE_M = 2.8   # meters (from gemini3.py)

# Mass and cargo
MASS_KG = 1500.0    # Empty vehicle mass (from gemini3.py)
CARGO_TON = 1.0     # Cargo capacity in tons (from gemini3.py)

# Power and drag (from gemini3.py)
P_MAX_W = 80_000.0           # Maximum power in Watts
CD = 0.35                    # Drag coefficient
FRONTAL_AREA = 2.2          # Frontal area in m²
CRR = 0.01                  # Rolling resistance coefficient

# Acceleration/Braking limits (from gemini3.py)
MAX_ACCEL_CMD = 1.5         # Max acceleration in m/s²
MAX_BRAKE_DECEL = 1.5       # Max deceleration in m/s²

# Speed limits (from gemini3.py)
SPEED_KMPH_EMPTY = 35.0     # Max speed when empty
SPEED_KMPH_LOADED = 25.0    # Max speed when loaded
TURN_SPEED_KMPH = 20.0      # Max speed for sharp turns

# Convert to m/s
SPEED_MS_EMPTY = SPEED_KMPH_EMPTY / 3.6
SPEED_MS_LOADED = SPEED_KMPH_LOADED / 3.6
TURN_SPEED_MS = TURN_SPEED_KMPH / 3.6

# Steering (from gemini1.py & gemini3.py)
MAX_STEER_ANGLE = math.pi / 4  # radians (from gemini1.py)
STEER_MAX_DEG = 35.0            # degrees (from gemini3.py)
STEER_RATE_DEGPS = 270.0        # steering rate deg/s
STEER_MAX_RAD = math.radians(STEER_MAX_DEG)
STEER_RATE_RADPS = math.radians(STEER_RATE_DEGPS)

# Control parameters
JERK_LIMIT = 1.0  # m/s³ (from gemini3.py)

# Operational Logic (from gemini3.py)
LOAD_UNLOAD_TIME_S = 3.0

# --- Pixel to Meter conversion ---
METERS_TO_PIXELS = 6  # Same as original try8
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS

# Convert pixel constants to meters
CAR_LENGTH_PX = CAR_LENGTH_M * METERS_TO_PIXELS
CAR_WIDTH_PX = CAR_WIDTH_M * METERS_TO_PIXELS
WAYPOINT_THRESHOLD_PX = 40.0

# --- Controller Look-ahead ---
LOOK_AHEAD_STEERING_IDX = 4
LOOK_AHEAD_SPEED_IDX = 12
CURVE_THRESHOLD_DOT = 0.97

# --- Physics Functions (FROM GEMINI3.PY) ---
def resist_forces(v_ms: float, mass_kg: float) -> float:
    """Calculate resistance forces (rolling + air drag)"""
    return CRR * mass_kg * 9.81 + 0.5 * 1.225 * CD * FRONTAL_AREA * v_ms**2

def traction_force_from_power(v_ms: float, throttle: float) -> float:
    """Calculate traction force from engine power"""
    return (P_MAX_W * np.clip(throttle, 0, 1)) / max(v_ms, 0.5)

def brake_force_from_command(brake_cmd: float, mass_kg: float) -> float:
    """Calculate braking force"""
    return np.clip(brake_cmd, 0, 1) * mass_kg * MAX_BRAKE_DECEL

# --- Kalman Filter Class ---
class KalmanFilter:
    def __init__(self, dt):
        self.x = np.zeros(4)  # [x, vx, y, vy]
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

# --- Enhanced Car Class (WITH PHYSICS FROM GEMINI FILES) ---
class Car:
    def __init__(self, x_px, y_px, angle=0):
        # Position in pixels
        self.x = x_px
        self.y = y_px
        self.angle = angle  # radians

        # Physics state (in SI units - meters and m/s)
        self.x_m = x_px * PIXELS_TO_METERS
        self.y_m = y_px * PIXELS_TO_METERS
        self.speed_ms = 0.0  # Speed in m/s
        self.accel_ms2 = 0.0  # Acceleration in m/s²

        self.steer_angle = 0.0
        self.length = CAR_LENGTH_PX
        self.width = CAR_WIDTH_PX

        # Mass and cargo (from gemini3.py)
        self.current_mass_kg = MASS_KG

        # Operational state (from gemini3.py)
        self.op_state = "GOING_TO_ENDPOINT"
        self.op_timer = 0.0

        # Controller state (for jerk limiting)
        self.a_cmd_prev = 0.0

    def move(self, throttle_or_accel_cmd, steer_input, dt):
        """
        Enhanced move function with realistic physics.
        throttle_or_accel_cmd: desired acceleration in m/s²
        steer_input: desired steering angle in radians
        dt: time step in seconds
        """
        # Convert pixel position to meters for physics calculations
        self.x_m = self.x * PIXELS_TO_METERS
        self.y_m = self.y * PIXELS_TO_METERS

        # Jerk limiting (from gemini3.py)
        accel_cmd = np.clip(
            throttle_or_accel_cmd,
            self.a_cmd_prev - JERK_LIMIT * dt,
            self.a_cmd_prev + JERK_LIMIT * dt
        )
        self.a_cmd_prev = accel_cmd

        # Calculate forces
        F_resist = resist_forces(self.speed_ms, self.current_mass_kg)
        F_needed = self.current_mass_kg * accel_cmd

        if F_needed >= 0:
            # Accelerating
            F_engine = F_needed + F_resist
            throttle = (F_engine * max(self.speed_ms, 0.5)) / P_MAX_W
            brake = 0.0
        else:
            # Braking
            F_brake_req = -F_needed + F_resist
            brake = F_brake_req / (self.current_mass_kg * MAX_BRAKE_DECEL)
            throttle = 0.0

        # Apply steering rate limit
        self.steer_angle = np.clip(
            steer_input,
            self.steer_angle - STEER_RATE_RADPS * dt,
            self.steer_angle + STEER_RATE_RADPS * dt
        )

        # Calculate net force
        F_trac = traction_force_from_power(self.speed_ms, throttle)
        F_brake = brake_force_from_command(brake, self.current_mass_kg)
        F_res = resist_forces(self.speed_ms, self.current_mass_kg)
        F_net = F_trac - F_res - F_brake

        # Update acceleration and velocity
        self.accel_ms2 = F_net / self.current_mass_kg
        self.speed_ms = max(0.0, self.speed_ms + self.accel_ms2 * dt)

        # Kinematic bicycle model (from gemini1.py & gemini3.py)
        if abs(self.steer_angle) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(self.steer_angle)
            angular_velocity = self.speed_ms / turn_radius
            self.angle += angular_velocity * dt

        # Update position (in meters first)
        self.x_m += self.speed_ms * math.cos(self.angle) * dt
        self.y_m += self.speed_ms * math.sin(self.angle) * dt

        # Convert back to pixels
        self.x = self.x_m * METERS_TO_PIXELS
        self.y = self.y_m * METERS_TO_PIXELS

    def update_op_state(self, dt, path_length_px, current_s_path_px):
        """Update operational state for loading/unloading (from gemini3.py)"""
        current_s_m = current_s_path_px * PIXELS_TO_METERS
        path_length_m = path_length_px * PIXELS_TO_METERS

        if self.op_state == "GOING_TO_ENDPOINT":
            if current_s_m >= path_length_m - 2.0 and self.speed_ms < 0.5:
                self.op_state = "LOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
        elif self.op_state == "LOADING":
            if self.op_timer > 0:
                self.op_timer -= dt
            else:
                self.op_state = "RETURNING_TO_START"
                self.current_mass_kg = MASS_KG + CARGO_TON * 1000
        elif self.op_state == "RETURNING_TO_START":
            if current_s_m <= 2.0 and self.speed_ms < 0.5:
                self.op_state = "UNLOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
        elif self.op_state == "UNLOADING":
            if self.op_timer > 0:
                self.op_timer -= dt
            else:
                self.op_state = "GOING_TO_ENDPOINT"
                self.current_mass_kg = MASS_KG

    def get_noisy_measurement(self, noise_std_dev=0.5):
        return np.array([self.x + np.random.normal(0, noise_std_dev),
                        self.y + np.random.normal(0, noise_std_dev)])

    def draw(self, screen, g_to_s):
        car_center_px = g_to_s((self.x, self.y))
        car_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)

        # Color changes based on cargo (from gemini3.py)
        cargo_ratio = (self.current_mass_kg - MASS_KG) / (CARGO_TON * 1000)
        body_color = tuple(np.array(CAR_COLOR) * (1 - cargo_ratio) + np.array([120, 120, 120]) * cargo_ratio)

        car_surface.fill(body_color)
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        rect = rotated_surface.get_rect(center=car_center_px)
        screen.blit(rotated_surface, rect.topleft)

# --- Path Processing Functions ---
def extract_waypoints_from_mask(road_mask, step_size):
    """Takes a binary road mask, thins it, and extracts ordered waypoints."""
    if road_mask is None or road_mask.size == 0: return []
    thinned = ximgproc.thinning(road_mask)
    points = np.argwhere(thinned > 0)
    if len(points) < 2: return []
    points = points[:, ::-1]  # Swap to (x, y)
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
    print(f"Extracted {len(waypoints)} waypoints from image.")
    return waypoints

def chaikin_smoother(points, iterations):
    """Smooths a path using Chaikin's algorithm."""
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
def grid_to_screen(pos, scale, pan): 
    return (int(pos[0] * scale + pan[0]), int(pos[1] * scale + pan[1]))

def screen_to_grid(pos, scale, pan): 
    return ((pos[0] - pan[0]) / scale, (pos[1] - pan[1]) / scale)

def draw_road(screen, waypoints, g_to_s, scale):
    """Draws a smoothed road from waypoints."""
    if len(waypoints) < 2: return
    smoothed_points = chaikin_smoother(waypoints, SMOOTH_ITERATIONS)
    road_px = [g_to_s(p) for p in smoothed_points]
    if len(road_px) > 1:
        pygame.draw.lines(screen, GRAY, False, road_px, max(1, int(ROAD_WIDTH_PX * scale)))

# --- Main Simulation ---
def run_simulation(waypoints):
    """Initializes Pygame and runs the car simulation with the given waypoints."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Enhanced Physics Simulation - OpenCV Road")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # --- Setup Car and KF ---
    initial_angle = math.atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])
    car = Car(waypoints[0][0], waypoints[0][1], angle=initial_angle)
    kf = KalmanFilter(dt=1.0/60.0)
    kf.x = np.array([car.x, 0, car.y, 0])
    current_waypoint_idx = 0
    direction = 1

    # Calculate path length in pixels
    path_length_px = sum(np.linalg.norm(waypoints[i+1] - waypoints[i]) 
                         for i in range(len(waypoints)-1))

    # --- Setup View ---
    img_w = max(p[0] for p in waypoints)
    img_h = max(p[1] for p in waypoints)
    scale = min((WIDTH - PADDING * 2) / img_w, (HEIGHT - PADDING * 2) / img_h)
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
            # Pan and Zoom event handling
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    mouse_dragging = True
                    last_mouse_pos = event.pos
                elif event.button in (4, 5):  # Zoom
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos = screen_to_grid(event.pos, scale, pan)
                    scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]
                    pan[1] += event.pos[1] - new_screen_pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: 
                mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0] += dx
                pan[1] += dy
                last_mouse_pos = event.pos

        # --- Car AI Logic with Enhanced Physics ---
        est_pos = np.array([kf.x[0], kf.x[2]])
        est_vel = np.array([kf.x[1], kf.x[3]])
        est_speed_pxps = np.linalg.norm(est_vel)

        # Calculate path progress
        current_s_path_px = 0
        for i in range(current_waypoint_idx):
            current_s_path_px += np.linalg.norm(waypoints[i+1] - waypoints[i])
        current_s_path_px += np.linalg.norm(est_pos - waypoints[current_waypoint_idx])

        # Update operational state
        car.update_op_state(dt, path_length_px, current_s_path_px)

        # Waypoint switching
        if np.linalg.norm(waypoints[current_waypoint_idx] - est_pos) < WAYPOINT_THRESHOLD_PX:
            if direction == 1 and current_waypoint_idx >= len(waypoints) - 1: 
                direction = -1
            elif direction == -1 and current_waypoint_idx <= 0: 
                direction = 1
            else: 
                current_waypoint_idx += direction

        # Speed Control with Physics
        speed_idx = np.clip(current_waypoint_idx + (direction * LOOK_AHEAD_SPEED_IDX), 
                           0, len(waypoints) - 1)
        next_idx = np.clip(current_waypoint_idx + direction, 0, len(waypoints) - 1)

        v1 = waypoints[next_idx] - est_pos
        v2 = waypoints[speed_idx] - waypoints[next_idx]

        is_straight = True
        if np.linalg.norm(v1) > 1 and np.linalg.norm(v2) > 1:
            dot = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
            is_straight = dot > CURVE_THRESHOLD_DOT

        # Determine target speed based on state and curvature
        if car.op_state in ["LOADING", "UNLOADING"]:
            desired_speed_ms = 0.0
        else:
            if car.current_mass_kg > MASS_KG:
                base_speed_ms = SPEED_MS_LOADED
            else:
                base_speed_ms = SPEED_MS_EMPTY

            if is_straight:
                desired_speed_ms = base_speed_ms
            else:
                desired_speed_ms = min(base_speed_ms, TURN_SPEED_MS)

        # Convert to pixels for comparison
        desired_speed_pxps = desired_speed_ms * METERS_TO_PIXELS

        # Calculate acceleration command (in m/s²)
        est_speed_ms = est_speed_pxps * PIXELS_TO_METERS
        speed_error_ms = desired_speed_ms - est_speed_ms
        accel_cmd = np.clip(speed_error_ms / max(0.4, dt), 
                           -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)

        # Steering Control
        steer_idx = np.clip(current_waypoint_idx + (direction * LOOK_AHEAD_STEERING_IDX), 
                           0, len(waypoints) - 1)
        vec_to_target = waypoints[steer_idx] - est_pos
        angle_to_target = math.atan2(vec_to_target[1], vec_to_target[0])
        angle_err = (angle_to_target - car.angle + math.pi) % (2 * math.pi) - math.pi
        steer_input = np.clip(angle_err * 2.5, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)

        # --- Update Physics and KF ---
        car.move(accel_cmd, steer_input, dt)

        # Update Kalman filter
        accel_vec = np.array([
            car.accel_ms2 * math.cos(car.angle) * METERS_TO_PIXELS,
            car.accel_ms2 * math.sin(car.angle) * METERS_TO_PIXELS
        ])
        kf.predict(u=accel_vec)
        kf.update(z=car.get_noisy_measurement())

        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s = lambda pos: grid_to_screen(pos, scale, pan)
        draw_road(screen, waypoints, g_to_s, scale)
        car.draw(screen, g_to_s)

        # Draw KF estimate
        kf_pos_px = (kf.x[0], kf.x[2])
        kf_screen = g_to_s(kf_pos_px)
        pygame.draw.circle(screen, RED, kf_screen, 6, 2)

        # HUD with physics info
        speed_kmph = car.speed_ms * 3.6
        hud_texts = [
            f"Speed: {speed_kmph:.1f} km/h",
            f"Mass: {car.current_mass_kg:.0f} kg",
            f"State: {car.op_state}",
            f"Accel: {car.accel_ms2:.2f} m/s²",
            f"Direction: {'Forward' if direction == 1 else 'Backward'}"
        ]

        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 22))

        pygame.display.flip()

    pygame.quit()

# --- Main Block with Caching ---
if __name__ == '__main__':
    # --- Step 1: Check for cached waypoints ---
    if os.path.exists(WAYPOINT_CACHE_FILE):
        print(f"Loading cached waypoints from '{WAYPOINT_CACHE_FILE}'...")
        with open(WAYPOINT_CACHE_FILE, 'rb') as f:
            final_waypoints = pickle.load(f)
    else:
        # --- Step 2: Process Image (if no cache) ---
        print("No cache found. Processing image...")
        if not os.path.exists(IMAGE_FILE):
            print(f"ERROR: Image file not found at '{IMAGE_FILE}'. Please check the file name.")
            final_waypoints = []
        else:
            img = cv2.imread(IMAGE_FILE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, road_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            final_waypoints = extract_waypoints_from_mask(road_mask, step_size=WAYPOINT_STEP_SIZE)

            if final_waypoints and len(final_waypoints) > 1:
                print(f"Saving waypoints to cache '{WAYPOINT_CACHE_FILE}'...")
                with open(WAYPOINT_CACHE_FILE, 'wb') as f:
                    pickle.dump(final_waypoints, f)

    # --- Step 3: Run Simulation ---
    if final_waypoints and len(final_waypoints) > 1:
        print("Waypoints loaded. Starting enhanced physics simulation...")
        run_simulation(final_waypoints)
    else:
        print("Failed to generate or load a valid path. Exiting.")
