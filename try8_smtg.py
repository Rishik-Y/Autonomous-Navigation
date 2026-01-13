import pygame
import cv2
import cv2.ximgproc as ximgproc
import numpy as np
import math
from scipy.spatial import KDTree
import os
import pickle  # --- NEW: Import pickle for caching ---

# --- Constants ---
# --- Main Settings ---
IMAGE_FILE = r'C:\Users\DAIICT A\Downloads\Sub\smtgimg_to_road\gimp.png'
# --- NEW: Define a cache file path ---
WAYPOINT_CACHE_FILE = 'waypoints_1000.pkl'
WAYPOINT_STEP_SIZE = 8
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

# --- Car Simulation (in PIXEL units) ---
SPEED_STRAIGHT_PXPS = 250.0
SPEED_CURVE_PXPS = 150.0
MAX_ACCELERATION_PXPS = 150.0
MAX_STEER_ANGLE = math.pi / 4
CAR_LENGTH_PX = 40.0
CAR_WIDTH_PX = 20.0
WAYPOINT_THRESHOLD_PX = 40.0

# --- Controller Look-ahead ---
LOOK_AHEAD_STEERING_IDX = 4
LOOK_AHEAD_SPEED_IDX = 12
CURVE_THRESHOLD_DOT = 0.97


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
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle # radians
        self.speed_pxps = 0.0 # Pixels Per Second
        self.steer_angle = 0.0
        self.length = CAR_LENGTH_PX
        self.width = CAR_WIDTH_PX

    def move(self, throttle, steer_input, dt):
        self.speed_pxps += throttle * dt
        self.steer_angle = np.clip(steer_input, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
        if self.steer_angle != 0:
            turn_radius = self.length / math.tan(self.steer_angle)
            angular_velocity = self.speed_pxps / turn_radius
            self.angle += angular_velocity * dt
        self.x += self.speed_pxps * math.cos(self.angle) * dt
        self.y += self.speed_pxps * math.sin(self.angle) * dt

    def get_noisy_measurement(self, noise_std_dev=0.5):
        return np.array([self.x + np.random.normal(0, noise_std_dev),
                         self.y + np.random.normal(0, noise_std_dev)])

    def draw(self, screen, g_to_s):
        car_center_px = g_to_s((self.x, self.y))
        car_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        car_surface.fill(CAR_COLOR)
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
    points = points[:, ::-1] # Swap to (x, y)
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
def grid_to_screen(pos, scale, pan): return (int(pos[0] * scale + pan[0]), int(pos[1] * scale + pan[1]))
def screen_to_grid(pos, scale, pan): return ((pos[0] - pan[0]) / scale, (pos[1] - pan[1]) / scale)

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
    pygame.display.set_caption("OpenCV Road Simulation")
    clock = pygame.time.Clock()
    
    # --- Setup Car and KF ---
    initial_angle = math.atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])
    car = Car(waypoints[0][0], waypoints[0][1], angle=initial_angle)
    kf = KalmanFilter(dt=1.0/60.0)
    kf.x = np.array([car.x, 0, car.y, 0])
    current_waypoint_idx = 0
    direction = 1

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
            if event.type == pygame.QUIT: running = False
            # (Pan and Zoom event handling)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: mouse_dragging = True; last_mouse_pos = event.pos
                elif event.button in (4, 5): # Zoom
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos = screen_to_grid(event.pos, scale, pan)
                    scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]
                    pan[1] += event.pos[1] - new_screen_pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0] += dx; pan[1] += dy
                last_mouse_pos = event.pos

        # --- Car AI Logic ---
        est_pos = np.array([kf.x[0], kf.x[2]])
        if np.linalg.norm(waypoints[current_waypoint_idx] - est_pos) < WAYPOINT_THRESHOLD_PX:
            if direction == 1 and current_waypoint_idx >= len(waypoints) - 1: direction = -1
            elif direction == -1 and current_waypoint_idx <= 0: direction = 1
            else: current_waypoint_idx += direction
        
        # Speed Control
        speed_idx = np.clip(current_waypoint_idx + (direction * LOOK_AHEAD_SPEED_IDX), 0, len(waypoints) - 1)
        next_idx = np.clip(current_waypoint_idx + direction, 0, len(waypoints) - 1)
        v1 = waypoints[next_idx] - est_pos
        v2 = waypoints[speed_idx] - waypoints[next_idx]
        is_straight = True
        if np.linalg.norm(v1) > 1 and np.linalg.norm(v2) > 1:
            dot = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
            is_straight = dot > CURVE_THRESHOLD_DOT
        desired_speed = SPEED_STRAIGHT_PXPS if is_straight else SPEED_CURVE_PXPS
        throttle = np.clip(desired_speed - car.speed_pxps, -MAX_ACCELERATION_PXPS, MAX_ACCELERATION_PXPS)

        # Steering Control
        steer_idx = np.clip(current_waypoint_idx + (direction * LOOK_AHEAD_STEERING_IDX), 0, len(waypoints) - 1)
        vec_to_target = waypoints[steer_idx] - est_pos
        angle_to_target = math.atan2(vec_to_target[1], vec_to_target[0])
        angle_err = (angle_to_target - car.angle + math.pi) % (2 * math.pi) - math.pi
        steer_input = np.clip(angle_err * 2.5, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)

        # --- Update Physics and KF ---
        car.move(throttle, steer_input, dt)
        accel_vec = np.array([throttle * math.cos(car.angle), throttle * math.sin(car.angle)])
        kf.predict(u=accel_vec)
        kf.update(z=car.get_noisy_measurement())

        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s = lambda pos: grid_to_screen(pos, scale, pan)
        draw_road(screen, waypoints, g_to_s, scale)
        car.draw(screen, g_to_s)
        pygame.display.flip()

    pygame.quit()

# --- NEW: MODIFIED MAIN BLOCK FOR CACHING ---
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
            final_waypoints = [] # Set to empty list to avoid error
        else:
            img = cv2.imread(IMAGE_FILE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Use simple thresholding for high-contrast drawings
            _, road_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            
            # --- Extract Waypoints ---
            final_waypoints = extract_waypoints_from_mask(road_mask, step_size=WAYPOINT_STEP_SIZE)
            
            # --- Save to cache ---
            if final_waypoints and len(final_waypoints) > 1:
                print(f"Saving waypoints to cache '{WAYPOINT_CACHE_FILE}'...")
                with open(WAYPOINT_CACHE_FILE, 'wb') as f:
                    pickle.dump(final_waypoints, f)

    # --- Step 3: Run Simulation ---
    if final_waypoints and len(final_waypoints) > 1:
        print("Waypoints loaded. Starting Pygame simulation...")
        run_simulation(final_waypoints)
    else:
        print("Failed to generate or load a valid path. Exiting.")