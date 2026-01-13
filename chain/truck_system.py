import numpy as np
import math

# --- CONSTANTS ---
CAR_LENGTH_M, CAR_WIDTH_M, WHEELBASE_M = 4.5, 2.0, 2.8
MASS_KG, CARGO_TON = 1500.0, 1.0
P_MAX_W, CD, FRONTAL_AREA, CRR = 80_000.0, 0.35, 2.2, 0.01
MAX_ACCEL_CMD, MAX_BRAKE_DECEL = 1.5, 1.5
SPEED_KMPH_EMPTY, SPEED_KMPH_LOADED = 35.0, 25.0
SPEED_MS_EMPTY, SPEED_MS_LOADED = SPEED_KMPH_EMPTY / 3.6, SPEED_KMPH_LOADED / 3.6
METERS_TO_PIXELS = 3.0
CAR_COLOR = (0, 80, 200)

# --- TUNING ---
STEER_RATE_RADPS = math.radians(720.0)
JERK_LIMIT = 1.0
LOAD_UNLOAD_TIME_S = 3.0
SAFE_GAP_M = 10.0  # Distance to keep from the guy in front

# --- PHYSICS HELPERS ---
def resist_forces(v_ms, mass_kg):
    return CRR * mass_kg * 9.81 + 0.5 * 1.225 * CD * FRONTAL_AREA * v_ms**2

def traction_force_from_power(v_ms, throttle):
    return (P_MAX_W * np.clip(throttle, 0, 1)) / max(v_ms, 0.5)

def brake_force_from_command(brake_cmd, mass_kg):
    return np.clip(brake_cmd, 0, 1) * mass_kg * MAX_BRAKE_DECEL

# --- PATH CLASS ---
class Path:
    def __init__(self, waypoints: list[np.ndarray]):
        if not waypoints or len(waypoints) < 2: self.wp = [np.array([0,0]), np.array([1,1])]
        else: self.wp = waypoints
        self.s = [0.0]
        for i in range(1, len(self.wp)): self.s.append(self.s[-1] + np.linalg.norm(self.wp[i] - self.wp[i-1]))
        self.length = self.s[-1]
        if self.length < 1e-6: self.length = 1e-6

    def get_segment_index(self, s: float) -> int:
        s = np.clip(s, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]: return i
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
        min_dist_sq, best_s = float('inf'), s_hint
        for i in range(idx_start, idx_end):
            a, b = self.wp[i], self.wp[i + 1]
            seg_vec = b - a; ap = p - a; dot_val = np.dot(seg_vec, seg_vec)
            if dot_val < 1e-9: t = 0.0
            else: t = np.clip(np.dot(ap, seg_vec) / dot_val, 0, 1)
            proj = a + t * seg_vec; dist_sq = np.sum((p - proj)**2)
            if dist_sq < min_dist_sq: min_dist_sq, best_s = dist_sq, self.s[i] + t * np.linalg.norm(seg_vec)
        return best_s

    def get_curvature_at(self, s: float) -> float:
        s_before, s_at, s_after = self.point_at(s - 1.0), self.point_at(s), self.point_at(s + 1.0)
        area = 0.5 * abs(s_before[0]*(s_at[1]-s_after[1]) + s_at[0]*(s_after[1]-s_before[1]) + s_after[0]*(s_before[1]-s_at[1]))
        d1, d2, d3 = np.linalg.norm(s_at - s_before), np.linalg.norm(s_after - s_at), np.linalg.norm(s_after - s_before)
        if d1*d2*d3 < 1e-6: return 0.0
        return (4 * area) / (d1 * d2 * d3)

# --- KALMAN FILTER ---
class KalmanFilter:
    def __init__(self, dt, start_x, start_y):
        self.x = np.zeros(4); self.x[0] = start_x; self.x[2] = start_y; self.dt = dt
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt**2, 0], [dt, 0], [0, 0.5 * dt**2], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.eye(4) * 0.1; self.R = np.eye(2) * (0.5**2); self.P = np.eye(4)
    def predict(self, u): self.x = self.F @ self.x + self.B @ u; self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        y = z - self.H @ self.x; S = self.H @ self.P @ self.H.T + self.R
        try: K = self.P @ self.H.T @ np.linalg.inv(S); self.x = self.x + K @ y; self.P = (np.eye(4) - K @ self.H) @ self.P
        except: pass

# --- TRUCK (CAR) CLASS ---
class Truck:
    def __init__(self, x_m, y_m, angle=0, start_node="start_zone"):
        self.x_m, self.y_m, self.angle = x_m, y_m, angle
        self.speed_ms, self.accel_ms2, self.steer_angle = 0.0, 0.0, 0.0
        self.current_mass_kg = MASS_KG
        self.op_state = "GOING_TO_ENDPOINT" 
        self.op_timer = 0.0
        self.a_cmd_prev, self.s_path_m = 0.0, 0.0
        self.current_node_name = start_node 
        self.target_node_name = ""
        self.kf = KalmanFilter(1.0/60.0, x_m, y_m)
        
        # Platoon Identity
        self.is_leader = True
        self.front_truck = None  # The actual object of the truck ahead

    def move(self, accel_cmd, steer_input, dt):
        accel_cmd = np.clip(accel_cmd, self.a_cmd_prev - JERK_LIMIT * dt, self.a_cmd_prev + JERK_LIMIT * dt)
        self.a_cmd_prev = accel_cmd
        F_resist = resist_forces(self.speed_ms, self.current_mass_kg)
        F_needed = self.current_mass_kg * accel_cmd
        if F_needed >= 0: throttle = ((F_needed + F_resist) * max(self.speed_ms, 0.5)) / P_MAX_W; brake = 0.0
        else: brake = (-F_needed + F_resist) / (self.current_mass_kg * MAX_BRAKE_DECEL); throttle = 0.0
        self.steer_angle = np.clip(steer_input, self.steer_angle - STEER_RATE_RADPS * dt, self.steer_angle + STEER_RATE_RADPS * dt)
        F_trac = traction_force_from_power(self.speed_ms, throttle); F_brake_force = brake_force_from_command(brake, self.current_mass_kg)
        F_net = F_trac - F_resist - F_brake_force
        self.accel_ms2 = F_net / self.current_mass_kg
        self.speed_ms = max(0.0, self.speed_ms + self.accel_ms2 * dt)
        if abs(self.steer_angle) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(self.steer_angle); angular_velocity = self.speed_ms / turn_radius; self.angle += angular_velocity * dt
        self.x_m += self.speed_ms * math.cos(self.angle) * dt; self.y_m += self.speed_ms * math.sin(self.angle) * dt
        accel_vec = np.array([self.accel_ms2 * math.cos(self.angle), self.accel_ms2 * math.sin(self.angle)])
        self.kf.predict(accel_vec); self.kf.update(self.get_noisy_measurement())

    def get_noisy_measurement(self):
        return np.array([self.x_m + np.random.normal(0, 0.5), self.y_m + np.random.normal(0, 0.5)])

    # --- CHAINED ACC (Adaptive Cruise Control) ---
    def get_acc_control(self, dt):
        if not self.front_truck: return 0.0 
        
        # Use PATH DISTANCE (Arc Length) not Euclidean
        # This handles curves correctly
        dist_to_front = self.front_truck.s_path_m - self.s_path_m
        
        # Handle Loop-around logic (if front truck is at s=10 and I am at s=1000)
        if dist_to_front < -100: dist_to_front += 10000 # Hack for closed loops if needed, but A* usually linear
        
        gap_error = dist_to_front - SAFE_GAP_M
        
        # P-Controller: Kp = 1.0
        # If gap is large (>15m), drive fast. If gap is small, slow down.
        desired_catchup_speed = self.front_truck.speed_ms + (gap_error * 0.5)
        
        # Safety: Never reverse
        desired_catchup_speed = max(0.0, desired_catchup_speed)
        
        speed_error = desired_catchup_speed - self.speed_ms
        accel_cmd = np.clip(speed_error / 0.4, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
        
        # EMERGENCY BRAKE: If gap < 4m, slam brakes
        if dist_to_front < 4.0:
            accel_cmd = -MAX_BRAKE_DECEL
            
        return accel_cmd

    def update_op_state(self, dt, path_length_m, current_s_m):
        # Leader State Machine
        if self.op_state == "GOING_TO_ENDPOINT": 
            if current_s_m >= path_length_m - 2.0 and self.speed_ms < 0.5:
                self.op_state = "LOADING"; self.op_timer = LOAD_UNLOAD_TIME_S
            return +1, SPEED_MS_EMPTY
        elif self.op_state == "LOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else: self.op_state = "RETURNING_TO_START"; self.current_mass_kg = MASS_KG + CARGO_TON * 1000
            return 0, 0.0
        elif self.op_state == "RETURNING_TO_START": 
            if current_s_m <= 2.0 and self.speed_ms < 0.5: 
                self.op_state = "UNLOADING"; self.op_timer = LOAD_UNLOAD_TIME_S
            return -1, SPEED_MS_LOADED
        elif self.op_state == "UNLOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else: self.op_state = "GOING_TO_ENDPOINT"; self.current_mass_kg = MASS_KG
            return 0, 0.0
        return 0, 0.0

    def draw(self, screen, g_to_s):
        import pygame
        car_center_screen = g_to_s((self.x_m, self.y_m))
        length_px = CAR_LENGTH_M * METERS_TO_PIXELS * g_to_s.scale
        width_px = CAR_WIDTH_M * METERS_TO_PIXELS * g_to_s.scale
        if length_px < 1 or width_px < 1: return
        car_surface = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
        
        if self.is_leader:
            # Leaders = Blue/Grey (Based on Load)
            cargo_ratio = (self.current_mass_kg - MASS_KG) / (CARGO_TON * 1000)
            body_color = tuple(np.array(CAR_COLOR) * (1 - cargo_ratio) + np.array([120, 120, 120]) * cargo_ratio)
        else:
            # Followers = Green/Red (Red if braking/too close, Green if good)
            gap_ok = True
            if self.front_truck:
                dist = self.front_truck.s_path_m - self.s_path_m
                if dist < SAFE_GAP_M - 2.0: gap_ok = False
            body_color = (0, 200, 0) if gap_ok else (255, 100, 0)
            
        car_surface.fill(body_color)
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        rect = rotated_surface.get_rect(center=car_center_screen)
        screen.blit(rotated_surface, rect.topleft)