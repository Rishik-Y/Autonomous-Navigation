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
METERS_TO_PIXELS = 4
ROAD_WIDTH_M = 10
ROAD_SIDE_M = 100
TURN_RADIUS_M = 15.0
NUM_WAYPOINTS_PER_TURN = 5

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

# --- Path Generation Function (Unchanged) ---
def generate_rounded_square_path(side_length, radius, num_points):
    waypoints = []
    centers = [
        np.array([radius, radius]),
        np.array([side_length - radius, radius]),
        np.array([side_length - radius, side_length - radius]),
        np.array([radius, side_length - radius]),
    ]
    start_angles = [math.pi, -math.pi / 2, 0, math.pi / 2]
    
    for i in range(4):
        prev_turn_end_angle = start_angles[i-1] + math.pi / 2
        start_of_straight = centers[i-1] + radius * np.array([math.cos(prev_turn_end_angle), math.sin(prev_turn_end_angle)])
        waypoints.append(start_of_straight)
        
        for j in range(1, num_points + 1):
            angle = start_angles[i] + (j / num_points) * (math.pi / 2)
            wp = centers[i] + radius * np.array([math.cos(angle), math.sin(angle)])
            waypoints.append(wp)
    return waypoints

# --- Main simulation ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Autonomous Car: Reversing Path")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18, bold=True)
    
    total_world_dim_m = ROAD_SIDE_M
    world_offset_px = (
        (WIDTH - total_world_dim_m * METERS_TO_PIXELS) / 2, 
        (HEIGHT - total_world_dim_m * METERS_TO_PIXELS) / 2
    )

    road_rect_px = pygame.Rect(
        world_offset_px[0] - (ROAD_WIDTH_M / 2 * METERS_TO_PIXELS),
        world_offset_px[1] - (ROAD_WIDTH_M / 2 * METERS_TO_PIXELS),
        (ROAD_SIDE_M + ROAD_WIDTH_M) * METERS_TO_PIXELS,
        (ROAD_SIDE_M + ROAD_WIDTH_M) * METERS_TO_PIXELS
    )
    inner_grass_rect_px = pygame.Rect(
        world_offset_px[0] + (ROAD_WIDTH_M / 2 * METERS_TO_PIXELS),
        world_offset_px[1] + (ROAD_WIDTH_M / 2 * METERS_TO_PIXELS),
        (ROAD_SIDE_M - ROAD_WIDTH_M) * METERS_TO_PIXELS,
        (ROAD_SIDE_M - ROAD_WIDTH_M) * METERS_TO_PIXELS
    )

    waypoints = generate_rounded_square_path(ROAD_SIDE_M, TURN_RADIUS_M, NUM_WAYPOINTS_PER_TURN)
    current_waypoint_idx = 0

    initial_angle = math.atan2(waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0])
    car = Car(waypoints[0][0], waypoints[0][1], angle=initial_angle)
    kf = KalmanFilter(dt=1.0/FPS)
    kf.x = np.array([car.x_m, 0, car.y_m, 0])
    
    # --- MODIFIED: Use direction state instead of a 'finished' flag ---
    direction = 1  # 1 for forward, -1 for backward

    # --- Main Loop ---
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- MODIFIED: The car is always running, just its direction changes ---
        z = car.get_noisy_measurement()
        target_waypoint = waypoints[current_waypoint_idx]
        est_pos = np.array([kf.x[0], kf.x[2]])
        dist_to_waypoint = np.linalg.norm(target_waypoint - est_pos)

        # --- MODIFIED: Waypoint switching and reversing logic ---
        if dist_to_waypoint < 5.0:
            # If going forward and at the last waypoint, reverse.
            if direction == 1 and current_waypoint_idx == len(waypoints) - 1:
                direction = -1
            # If going backward and at the first waypoint, go forward again.
            elif direction == -1 and current_waypoint_idx == 0:
                direction = 1
            
            # Update the waypoint index based on the current direction
            current_waypoint_idx += direction

        # --- Control and Physics Logic (Unchanged) ---
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
        vector_to_target = waypoints[current_waypoint_idx] - est_pos # Always target the next waypoint
        angle_to_target = math.atan2(vector_to_target[1], vector_to_target[0])
        angle_error = (angle_to_target - car.angle)
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        steer_input = np.clip(angle_error * 2.0, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
        
        accel_vector = np.array([throttle * math.cos(car.angle), throttle * math.sin(car.angle)])
        kf.predict(u=accel_vector)
        kf.update(z=z)
        car.move(throttle, steer_input, dt)

        # --- Drawing ---
        screen.fill(WHITE)
        pygame.draw.rect(screen, GRAY, road_rect_px)
        pygame.draw.rect(screen, WHITE, inner_grass_rect_px)

        for wp in waypoints:
            wp_px = (wp[0] * METERS_TO_PIXELS + world_offset_px[0], 
                     wp[1] * METERS_TO_PIXELS + world_offset_px[1])
            pygame.draw.circle(screen, RED, wp_px, 3)
        
        # --- MODIFIED: Dynamic START and END markers ---
        start_idx = 0 if direction == 1 else len(waypoints) - 1
        end_idx = len(waypoints) - 1 if direction == 1 else 0

        start_pos_m, end_pos_m = waypoints[start_idx], waypoints[end_idx]
        start_pos_px = (start_pos_m[0] * METERS_TO_PIXELS + world_offset_px[0], start_pos_m[1] * METERS_TO_PIXELS + world_offset_px[1])
        end_pos_px = (end_pos_m[0] * METERS_TO_PIXELS + world_offset_px[0], end_pos_m[1] * METERS_TO_PIXELS + world_offset_px[1])
        screen.blit(font.render("START", True, BLACK), (start_pos_px[0] - 50, start_pos_px[1] - 30))
        screen.blit(font.render("END", True, BLACK), (end_pos_px[0] + 15, end_pos_px[1] - 10))
        
        # Draw path planning line and KF estimate (Unchanged)
        target_wp_px = (waypoints[current_waypoint_idx][0] * METERS_TO_PIXELS + world_offset_px[0], waypoints[current_waypoint_idx][1] * METERS_TO_PIXELS + world_offset_px[1])
        car_px_center = (car.x_m * METERS_TO_PIXELS + world_offset_px[0], car.y_m * METERS_TO_PIXELS + world_offset_px[1])
        pygame.draw.line(screen, YELLOW, car_px_center, target_wp_px, 2)

        car.draw(screen, world_offset_px)
        kf_pos_px = (kf.x[0] * METERS_TO_PIXELS + world_offset_px[0], kf.x[2] * METERS_TO_PIXELS + world_offset_px[1])
        pygame.draw.circle(screen, RED, kf_pos_px, 8, 2)

        # --- MODIFIED: Update HUD to show direction ---
        est_speed = np.linalg.norm(np.array([kf.x[1], kf.x[3]]))
        speed_text = font.render(f"Speed: {car.speed_mps * 3.6:.1f} km/h", True, BLACK)
        direction_str = "Forward" if direction == 1 else "Backward"
        state_text = font.render(f"Direction: {direction_str}", True, BLACK)
        screen.blit(speed_text, (10, 10))
        screen.blit(state_text, (10, 30))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()