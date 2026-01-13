import pygame
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

# --- CONFIGURATION & CONSTANTS ---
# Simulation Setup
WIN_W, WIN_H = 1200, 800
FPS = 60
NUM_TRUCKS = 5
AUTO_MODE = True
FOLLOW_LEAD_TRUCK = True

# --- MODIFIED PARAMETERS ---
TARGET_SPEED_KMPH = 40.0         # <--- Max speed set to 40 km/h
MIN_FOLLOW_DISTANCE_M = 10.0     # <--- Minimum gap between trucks set to 10m
DESIRED_TIME_GAP_S = 1.0         # Reduced for tighter following

# Truck Parameters
TRUCK_LEN_M = 8.0
TRUCK_WID_M = 2.5
WHEELBASE_M = 4.5
STEER_MAX_DEG = 30.0
STEER_RATE_DEGPS = 180.0
MAX_ACCELERATION = 1.5
MAX_BRAKE_DECEL = 3.0

# Camera & UI
BASE_SCALE = 8.0
PAN_SPEED_MPS = 80.0
ZOOM_STEP = 1.1

# Derived Constants
TARGET_SPEED_MS = TARGET_SPEED_KMPH / 3.6
STEER_MAX_RAD = math.radians(STEER_MAX_DEG)
STEER_RATE_RADPS = math.radians(STEER_RATE_DEGPS)

# Colors
COLORS = {
    "GREY": (85, 85, 85), "DARK": (30, 30, 30), "WHITE": (240, 240, 240),
    "RED": (220, 40, 40), "GREEN": (60, 200, 80), "BLUE": (40, 120, 220),
}

# --- DATA CLASSES ---
@dataclass
class TruckState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = math.pi / 2
    v: float = 0.0
    a: float = 0.0
    delta: float = 0.0
    s_path: float = 0.0

@dataclass
class Camera:
    cx: float = 0.0
    cy: float = 0.0
    zoom: float = 1.0

# --- PATH CLASS ---
Vec2 = Tuple[float, float]

class Path:
    """Handles calculations related to the path the trucks follow."""
    def __init__(self, waypoints: List[Vec2]):
        self.wp = waypoints
        self.s = [0.0]
        for i in range(1, len(self.wp)):
            self.s.append(self.s[-1] + np.linalg.norm(np.array(self.wp[i]) - np.array(self.wp[i-1])))
        self.length = self.s[-1]

    def point_at(self, s_query: float) -> Vec2:
        s = np.clip(s_query, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]:
                s_base, s_end = self.s[i], self.s[i+1]
                p_base, p_end = np.array(self.wp[i]), np.array(self.wp[i+1])
                if s_end - s_base < 1e-6:
                    return tuple(p_base)
                t = (s - s_base) / (s_end - s_base)
                return tuple(p_base + t * (p_end - p_base))
        return self.wp[-1]

    def project(self, p: Vec2) -> float:
        """Finds the progress (s) along the path for a given point."""
        pos = np.array(p)
        best_s = 0.0
        min_dist = float('inf')
        for i in range(len(self.wp) - 1):
            a, b = np.array(self.wp[i]), np.array(self.wp[i+1])
            seg_vec = b - a
            dot_val = np.dot(seg_vec, seg_vec)
            if dot_val < 1e-9: continue
            t = np.dot(pos - a, seg_vec) / dot_val
            t_clamped = np.clip(t, 0, 1)
            projection = a + t_clamped * seg_vec
            dist = np.linalg.norm(pos - projection)
            if dist < min_dist:
                min_dist = dist
                best_s = self.s[i] + t_clamped * np.linalg.norm(seg_vec)
        return best_s

# --- CORE AGENT & FLEET CLASSES ---

class Truck:
    """Represents a single truck with its physics, state, and controllers."""
    def __init__(self, agent_id: str, path: Path, initial_s: float):
        self.id = agent_id
        self.path = path
        initial_pos = path.point_at(initial_s)
        
        # Calculate initial angle to align with the path
        p_ahead = path.point_at(initial_s + 1.0)
        initial_yaw = math.atan2(p_ahead[1] - initial_pos[1], p_ahead[0] - initial_pos[0])
        
        self.state = TruckState(x=initial_pos[0], y=initial_pos[1], s_path=initial_s, yaw=initial_yaw)
        self.color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 255))
        self.a_cmd_prev = 0.0

    def step(self, dt: float, leader: Optional['Truck'], path_end_s: float):
        # --- 1. Lateral Control (Steering) using Pure Pursuit ---
        lookahead_dist = np.clip(self.state.v * 1.5, 5.0, 30.0)
        s_target = self.state.s_path + lookahead_dist
        p_target = self.path.point_at(s_target)
        
        dx, dy = p_target[0] - self.state.x, p_target[1] - self.state.y
        # Transform target to truck's reference frame
        dx_local = dx * math.cos(self.state.yaw) + dy * math.sin(self.state.yaw)
        dy_local = -dx * math.sin(self.state.yaw) + dy * math.cos(self.state.yaw)

        alpha = math.atan2(dy_local, dx_local)
        delta_target = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), max(1e-3, lookahead_dist))
        
        delta_target = np.clip(delta_target, -STEER_MAX_RAD, STEER_MAX_RAD)
        self.state.delta = np.clip(delta_target, self.state.delta - STEER_RATE_RADPS * dt, self.state.delta + STEER_RATE_RADPS * dt)

        # --- 2. Longitudinal Control (Speed) ---
        target_v = TARGET_SPEED_MS

        if leader:
            dist_to_leader = leader.state.s_path - self.state.s_path - TRUCK_LEN_M
            desired_gap = self.state.v * DESIRED_TIME_GAP_S + MIN_FOLLOW_DISTANCE_M
            gap_error = dist_to_leader - desired_gap
            target_v = leader.state.v + 0.5 * gap_error 
            target_v = min(target_v, TARGET_SPEED_MS)

        dist_to_end = path_end_s - self.state.s_path
        if dist_to_end < 40.0: # Start braking 40m from the end
            stopping_v = math.sqrt(max(0, 2 * MAX_BRAKE_DECEL * max(0, dist_to_end)))
            target_v = min(target_v, stopping_v)
        
        accel_needed = (target_v - self.state.v) / max(0.2, dt)
        accel_cmd = np.clip(accel_needed, -MAX_BRAKE_DECEL, MAX_ACCELERATION)
        
        max_jerk = 5.0
        self.a_cmd_prev = np.clip(accel_cmd, self.a_cmd_prev - max_jerk * dt, self.a_cmd_prev + max_jerk * dt)
        self.state.a = self.a_cmd_prev
        
        # --- 3. Physics Update ---
        self.state.v += self.state.a * dt
        self.state.v = max(0.0, self.state.v)
        
        self.state.yaw += (self.state.v * math.tan(self.state.delta) / WHEELBASE_M) * dt
        self.state.x += self.state.v * math.cos(self.state.yaw) * dt
        self.state.y += self.state.v * math.sin(self.state.yaw) * dt
        
        self.state.s_path = self.path.project((self.state.x, self.state.y))

class Fleet:
    """Manages all the trucks in the simulation."""
    def __init__(self):
        self.agents: List[Truck] = []

    def add(self, agent: Truck):
        self.agents.append(agent)

    def update(self, dt: float, path_end_s: float):
        for i, agent in enumerate(self.agents):
            leader = self.agents[i-1] if i > 0 else None
            agent.step(dt, leader, path_end_s)

    def render(self, screen, cam, win_w, win_h):
        for agent in self.agents:
            draw_truck(screen, agent, cam, win_w, win_h)

# --- RENDERING & UI ---
def world_to_screen(xm, ym, cam, win_w, win_h):
    S = BASE_SCALE * cam.zoom
    xs = win_w / 2 + (xm - cam.cx) * S
    ys = win_h / 2 + (cam.cy - ym) * S
    return int(xs), int(ys)

def draw_truck(screen, truck, cam, win_w, win_h):
    S = BASE_SCALE * cam.zoom
    w_px = max(2, int(TRUCK_WID_M * S))
    h_px = max(4, int(TRUCK_LEN_M * S))
    
    surf = pygame.Surface((w_px, h_px), pygame.SRCALPHA)
    pygame.draw.rect(surf, truck.color, surf.get_rect(), border_radius=int(w_px/4))
    pygame.draw.rect(surf, COLORS["DARK"], pygame.Rect(0, 0, w_px, h_px // 4), border_top_left_radius=int(w_px/4), border_top_right_radius=int(w_px/4))
    
    rotated_surf = pygame.transform.rotate(surf, -math.degrees(truck.state.yaw))
    xs, ys = world_to_screen(truck.state.x, truck.state.y, cam, win_w, win_h)
    rect = rotated_surf.get_rect(center=(xs, ys))
    screen.blit(rotated_surf, rect)

def draw_road(screen, path, cam, win_w, win_h):
    S = BASE_SCALE * cam.zoom
    road_width_px = max(2, int(10 * S)) # 10m wide road
    
    points = [world_to_screen(p[0], p[1], cam, win_w, win_h) for p in path.wp]
    if len(points) > 1:
        pygame.draw.lines(screen, COLORS["GREY"], False, points, road_width_px)

def draw_hud(screen, font, truck):
    speed_kmh = truck.state.v * 3.6
    text = font.render(f"Leader Speed: {speed_kmh:.1f} km/h | Path: {truck.state.s_path:.1f} m", True, COLORS["WHITE"])
    screen.blit(text, (10, 10))

# --- MAIN APPLICATION ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Multi-Truck Maze Following")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)
    
    # --- Create the MAZE path from gemini1.py ---
    waypoints = [
        (10, 90), (80, 90), (80, 50),
        (20, 50), (20, 20), (90, 20)
    ]
    path = Path(waypoints)
    PATH_END_M = path.length
    PATH_START_M = 0.0
    
    # --- Create the fleet ---
    fleet = Fleet()
    initial_separation = TRUCK_LEN_M + MIN_FOLLOW_DISTANCE_M
    for i in range(NUM_TRUCKS):
        initial_s = PATH_START_M - i * initial_separation
        truck = Truck(f"truck-{i}", path, initial_s)
        if i == 0: truck.color = COLORS["GREEN"]
        fleet.add(truck)
        
    cam = Camera(cx=50, cy=55, zoom=4.0)
    
    running = True
    while running:
        dt = min(0.1, clock.tick(FPS) / 1000.0)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEWHEEL:
                factor = ZOOM_STEP if event.y > 0 else 1.0 / ZOOM_STEP
                cam.zoom *= factor
        
        if AUTO_MODE:
            fleet.update(dt, PATH_END_M)

        if FOLLOW_LEAD_TRUCK and fleet.agents:
            lead_truck = fleet.agents[0]
            cam.cx = lead_truck.state.x
            cam.cy = lead_truck.state.y

        # --- Drawing ---
        screen.fill(COLORS["DARK"])
        draw_road(screen, path, cam, WIN_W, WIN_H)
        fleet.render(screen, cam, WIN_W, WIN_H)
        
        if fleet.agents:
            draw_hud(screen, font, fleet.agents[0])
            
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()