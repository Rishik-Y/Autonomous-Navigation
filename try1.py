import pygame
import math
import numpy as np
from scipy.spatial import KDTree # Needed for fast path tracing
import os # To check if file exists

# --- Constants ---
# Screen dimensions
WIDTH, HEIGHT = 900, 900

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100) # Road color
RED = (255, 0, 0)
GREEN = (0, 200, 0)
YELLOW = (255, 255, 0)
CAR_COLOR = (200, 0, 0)
HUD_BG = (240, 240, 240)
HUD_TEXT = (10, 10, 10)

# --- View Constants ---
ROAD_WIDTH_PX = 50
ZOOM_FACTOR = 1.1
PADDING = 50

# --- Car Simulation Constants (in PIXEL Units) ---
# These values are scaled for a coordinate system based on image pixels
SPEED_STRAIGHT_PXPS = 250.0 # Pixels Per Second
SPEED_CURVE_PXPS = 150.0
MAX_ACCELERATION_PXPS = 150.0
MAX_STEER_ANGLE = math.pi / 4 # Radians
CAR_LENGTH_PX = 40.0 # Car length in pixels
CAR_WIDTH_PX = 20.0
WAYPOINT_THRESHOLD_PX = 30.0 # How close to get to a waypoint (in pixels)

# --- Controller Look-ahead constants ---
LOOK_AHEAD_STEERING_IDX = 5 # Aim 5 waypoints ahead for steering
LOOK_AHEAD_SPEED_IDX = 10   # Check 10 waypoints ahead for curves
CURVE_THRESHOLD_DOT = 0.98  # Dot product threshold to detect a "straight" line

# --- Kalman Filter Class (Unchanged) ---
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
        self.x = x # Position is now in pixels
        self.y = y
        self.angle = angle # radians
        self.speed_pxps = 0.0 # Pixels Per Second
        self.steer_angle = 0.0
        # Use pixel-based constants
        self.length = CAR_LENGTH_PX
        self.width = CAR_WIDTH_PX

    def move(self, throttle, steer_input, dt):
        self.speed_pxps += throttle * dt
        self.steer_angle = max(-MAX_STEER_ANGLE, min(steer_input, MAX_STEER_ANGLE))
        
        if self.steer_angle != 0:
            turn_radius = self.length / math.tan(self.steer_angle)
            angular_velocity = self.speed_pxps / turn_radius
            self.angle += angular_velocity * dt
        
        self.x += self.speed_pxps * math.cos(self.angle) * dt
        self.y += self.speed_pxps * math.sin(self.angle) * dt

    def get_noisy_measurement(self, noise_std_dev=0.5):
        noisy_x = self.x + np.random.normal(0, noise_std_dev)
        noisy_y = self.y + np.random.normal(0, noise_std_dev)
        return np.array([noisy_x, noisy_y])

    def draw(self, screen, grid_to_screen_func):
        # Car dimensions are now fixed in pixels, but scale with zoom
        car_center_px = grid_to_screen_func((self.x, self.y))
        
        # We create a new surface, rotate it, and blit it
        car_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        car_surface.fill(CAR_COLOR)
        pygame.draw.rect(car_surface, BLACK, (self.length * 0.7, 0, self.length * 0.3, self.width))
        
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        
        rect = rotated_surface.get_rect(center=car_center_px)
        screen.blit(rotated_surface, rect.topleft)

# --- Image Processing Function ---
def generate_waypoints_from_image(image_path, step_size=15):
    """Loads a PNG, finds a black path, and returns an ordered list of waypoints."""
    if not os.path.exists(image_path):
        print(f"--- ERROR ---")
        print(f"Image file not found at: '{image_path}'")
        print(f"Please make sure the image is in the same directory as the script.")
        print(f"---------------")
        return []

    print(f"Loading image from: {image_path}")
    image = pygame.image.load(image_path)
    image_w, image_h = image.get_size()
    
    path_pixels = []
    BLACK_THRESHOLD = 100
    for x in range(image_w):
        for y in range(image_h):
            # Check for alpha channel
            color = image.get_at((x, y))
            r, g, b = color.r, color.g, color.b
            if r + g + b < BLACK_THRESHOLD:
                path_pixels.append((x, y))

    if not path_pixels:
        print("--- ERROR ---")
        print(f"No black pixels found in '{image_path}'. The road could not be generated.")
        print(f"---------------")
        return []
    
    print(f"Found {len(path_pixels)} path pixels. Tracing path...")
    
    pixel_tree = KDTree(path_pixels)
    start_pixel_idx = np.argmin([p[1] * image_w + p[0] for p in path_pixels])
    
    ordered_path = []
    visited_indices = set()
    current_idx = start_pixel_idx
    
    for _ in range(len(path_pixels)):
        if current_idx in visited_indices: break # Stop if we're stuck
        
        ordered_path.append(path_pixels[current_idx])
        visited_indices.add(current_idx)
        
        distances, indices = pixel_tree.query(path_pixels[current_idx], k=min(20, len(path_pixels)))

        next_idx = -1
        for i in indices:
            if i not in visited_indices:
                next_idx = i
                break
        
        if next_idx == -1: break
        current_idx = next_idx

    # Simplify the path and convert to numpy arrays
    simplified_path = [np.array(p) for p in ordered_path[::step_size]]
    if ordered_path and list(simplified_path[-1]) != ordered_path[-1]:
         simplified_path.append(np.array(ordered_path[-1]))

    print(f"Path traced. Simplified to {len(simplified_path)} waypoints.")
    return simplified_path

# --- Coordinate and Drawing Functions ---
def grid_to_screen(pos, scale, pan_offset):
    px = (pos[0] * scale) + pan_offset[0]
    py = (pos[1] * scale) + pan_offset[1]
    return (int(px), int(py))

def screen_to_grid(screen_pos, scale, pan_offset):
    gx = (screen_pos[0] - pan_offset[0]) / scale
    gy = (screen_pos[1] - pan_offset[1]) / scale
    return (gx, gy)

# The corrected function
def draw_image_road(screen, waypoints, grid_to_screen_func, scale):
    """Draws the road by connecting waypoints with thick lines."""
    if len(waypoints) < 2: return
    
    road_points_px = [grid_to_screen_func(p) for p in waypoints]
    
    # Draw thick lines for the road
    # Now 'scale' is correctly defined within this function's scope
    pygame.draw.lines(screen, GRAY, False, road_points_px, int(ROAD_WIDTH_PX * scale))
    
    # Draw circles at joints to make them smooth
    for point_px in road_points_px:
        pygame.draw.circle(screen, GRAY, point_px, int(ROAD_WIDTH_PX * scale) // 2)

# --- Main ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Car Simulation on Image-Based Road")
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("Arial", 18, bold=True)

    IMAGE_FILE = os.path.join(os.path.dirname(__file__), 'try.png')
    waypoints = generate_waypoints_from_image(IMAGE_FILE, step_size=20)

    if not waypoints or len(waypoints) < 2:
        print("Could not generate a valid path. Exiting.")
        return

    # --- Setup Car ---
    initial_angle = math.atan2(waypoints[1][1] - waypoints[0][1], 
                               waypoints[1][0] - waypoints[0][0])
    car = Car(waypoints[0][0], waypoints[0][1], angle=initial_angle)
    
    kf = KalmanFilter(dt=1.0/60.0)
    kf.x = np.array([car.x, 0, car.y, 0])
    
    current_waypoint_idx = 0
    direction = 1

    # --- Setup View Controls ---
    temp_img = pygame.image.load(IMAGE_FILE)
    img_width, img_height = temp_img.get_size()
    
    scale_x = (WIDTH - PADDING * 2) / img_width
    scale_y = (HEIGHT - PADDING * 2) / img_height
    scale = min(scale_x, scale_y) 
    
    pan_offset_px = [PADDING, PADDING]
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

        # --- Simulation Logic ---
        est_pos = np.array([kf.x[0], kf.x[2]])
        progress_target = waypoints[current_waypoint_idx]
        
        if np.linalg.norm(progress_target - est_pos) < WAYPOINT_THRESHOLD_PX:
            if direction == 1 and current_waypoint_idx >= len(waypoints) - 1:
                direction = -1 # Reverse
            elif direction == -1 and current_waypoint_idx <= 0:
                direction = 1 # Go forward again
            else:
                current_waypoint_idx += direction
        
        # --- Speed Control ---
        speed_look_ahead_idx = np.clip(current_waypoint_idx + (direction * LOOK_AHEAD_SPEED_IDX), 0, len(waypoints) - 1)
        next_point_idx = np.clip(current_waypoint_idx + direction, 0, len(waypoints) - 1)

        v1 = waypoints[next_point_idx] - est_pos
        v2 = waypoints[speed_look_ahead_idx] - waypoints[next_point_idx]
        
        is_straight = True
        if np.linalg.norm(v1) > 1 and np.linalg.norm(v2) > 1:
            dot_product = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
            is_straight = dot_product > CURVE_THRESHOLD_DOT

        desired_speed = SPEED_STRAIGHT_PXPS if is_straight else SPEED_CURVE_PXPS
        speed_error = desired_speed - car.speed_pxps
        throttle = np.clip(speed_error, -MAX_ACCELERATION_PXPS, MAX_ACCELERATION_PXPS)
        
        # --- Steering Control ---
        steering_target_idx = np.clip(current_waypoint_idx + (direction * LOOK_AHEAD_STEERING_IDX), 0, len(waypoints) - 1)
        steering_target = waypoints[steering_target_idx]
        
        vector_to_target = steering_target - est_pos
        angle_to_target = math.atan2(vector_to_target[1], vector_to_target[0])
        
        angle_error = (angle_to_target - car.angle)
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        
        steer_input = np.clip(angle_error * 2.5, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
        
        # --- Update Car Physics and KF ---
        car.move(throttle, steer_input, dt)
        z = car.get_noisy_measurement()
        accel_vector = np.array([throttle * math.cos(car.angle), throttle * math.sin(car.angle)])
        kf.predict(u=accel_vector)
        kf.update(z=z)
        
        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s_func = lambda pos: grid_to_screen(pos, scale, pan_offset_px)
        
        draw_image_road(screen, waypoints, g_to_s_func, scale)        
        if len(waypoints) > 1:
            path_px = [g_to_s_func(p) for p in waypoints]
            pygame.draw.lines(screen, YELLOW, False, path_px, 3)

        car.draw(screen, g_to_s_func)
        
        # Draw HUD
        hud_surface = pygame.Surface((250, 110), pygame.SRCALPHA)
        hud_surface.fill((*HUD_BG, 200))
        text1 = hud_font.render("Left Click + Drag to Pan", True, HUD_TEXT)
        text2 = hud_font.render("Mouse Wheel to Zoom", True, HUD_TEXT)
        text3 = hud_font.render(f"Zoom: {scale:.2f}x", True, HUD_TEXT)
        text4 = hud_font.render(f"Speed: {car.speed_pxps:.1f} px/s", True, HUD_TEXT)
        text5 = hud_font.render(f"Target: {desired_speed:.1f} px/s", True, HUD_TEXT)
        hud_surface.blit(text1, (10, 5)); hud_surface.blit(text2, (10, 25))
        hud_surface.blit(text3, (10, 45)); hud_surface.blit(text4, (10, 65))
        hud_surface.blit(text5, (10, 85))
        screen.blit(hud_surface, (10, 10))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()