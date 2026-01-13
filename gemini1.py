import pygame
import numpy as np
import math

# --- Constants ---
# Screen dimensions
WIDTH, HEIGHT = 800, 800

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Simulation parameters
FPS = 60
METERS_TO_PIXELS = 6 # Increased scale to make the maze fill the screen better
ROAD_WIDTH_M = 10
ROAD_WIDTH_PX = int(ROAD_WIDTH_M * METERS_TO_PIXELS)

# Car parameters
CAR_LENGTH_M = 4.5
CAR_WIDTH_M = 2.0
TARGET_SPEED_KMPH = 30.0
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
        self.original_image = pygame.Surface((CAR_LENGTH_M * METERS_TO_PIXELS, CAR_WIDTH_M * METERS_TO_PIXELS), pygame.SRCALPHA)
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

# --- Main simulation ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Autonomous Car: Maze Challenge")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18, bold=True)
    
    # --- MODIFIED: Define waypoints for a maze-like path ---
    # Coordinates are in meters. (0,0) is top-left of the maze area.
    waypoints_m = [
        np.array([10, 90]),
        np.array([80, 90]),
        np.array([80, 50]),
        np.array([20, 50]),
        np.array([20, 20]),
        np.array([90, 20])
    ]

    # --- MODIFIED: Center the maze world on the screen ---
    world_size_m = (100, 100) # Assumed world dimensions for the maze
    world_offset_px = (
        (WIDTH - world_size_m[0] * METERS_TO_PIXELS) / 2, 
        (HEIGHT - world_size_m[1] * METERS_TO_PIXELS) / 2
    )

    # Convert waypoints from meters to pixel coordinates for drawing
    waypoints_px = [(p[0] * METERS_TO_PIXELS + world_offset_px[0], p[1] * METERS_TO_PIXELS + world_offset_px[1]) for p in waypoints_m]

    current_waypoint_idx = 0

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Navigation Logic ---
        target_waypoint = waypoints_m[current_waypoint_idx]
        est_pos = np.array([kf.x[0], kf.x[2]])
        dist_to_waypoint = np.linalg.norm(target_waypoint - est_pos)

        # Waypoint switching and reversing logic
        if dist_to_waypoint < 7.0: # Slightly larger threshold for turns
            if direction == 1 and current_waypoint_idx == len(waypoints_m) - 1:
                direction = -1
            elif direction == -1 and current_waypoint_idx == 0:
                direction = 1
            current_waypoint_idx += direction

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
        vector_to_target = waypoints_m[current_waypoint_idx] - est_pos
        angle_to_target = math.atan2(vector_to_target[1], vector_to_target[0])
        angle_error = (angle_to_target - car.angle)
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        steer_input = np.clip(angle_error * 2.5, -MAX_STEER_ANGLE, MAX_STEER_ANGLE) # Increased steering gain
        
        accel_vector = np.array([throttle * math.cos(car.angle), throttle * math.sin(car.angle)])
        kf.predict(u=accel_vector)
        kf.update(z=z)
        car.move(throttle, steer_input, dt)

        # --- Drawing ---
        screen.fill(WHITE)
        
        # --- MODIFIED: Draw the maze road using thick lines ---
        pygame.draw.lines(screen, GRAY, False, waypoints_px, ROAD_WIDTH_PX)
        
        # Draw all waypoints as small circles
        for wp_px in waypoints_px:
            pygame.draw.circle(screen, RED, wp_px, 4)
        
        # Dynamic START and END markers
        start_idx = 0 if direction == 1 else len(waypoints_m) - 1
        end_idx = len(waypoints_m) - 1 if direction == 1 else 0

        start_pos_px, end_pos_px = waypoints_px[start_idx], waypoints_px[end_idx]
        
        start_text = font.render("START", True, BLACK)
        end_text = font.render("END / TURN BACK", True, BLACK) # Changed text
        
        # Position text based on waypoint location
        screen.blit(start_text, (start_pos_px[0] - 80, start_pos_px[1] - 10))
        screen.blit(end_text, (end_pos_px[0] + 15, end_pos_px[1] - 10))
        
        # Visualizations (Path line, KF estimate)
        target_wp_px = waypoints_px[current_waypoint_idx]
        car_px_center = (car.x_m * METERS_TO_PIXELS + world_offset_px[0], car.y_m * METERS_TO_PIXELS + world_offset_px[1])
        pygame.draw.line(screen, YELLOW, car_px_center, target_wp_px, 2)
        
        car.draw(screen, world_offset_px)
        
        kf_pos_px = (kf.x[0] * METERS_TO_PIXELS + world_offset_px[0], kf.x[2] * METERS_TO_PIXELS + world_offset_px[1])
        pygame.draw.circle(screen, RED, kf_pos_px, 8, 2)

        # HUD
        speed_text = font.render(f"Speed: {car.speed_mps * 3.6:.1f} km/h", True, BLACK)
        direction_str = "Forward" if direction == 1 else "Backward"
        state_text = font.render(f"Direction: {direction_str}", True, BLACK)
        screen.blit(speed_text, (10, 10))
        screen.blit(state_text, (10, 30))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()