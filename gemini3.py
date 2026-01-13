import pygame
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple

# --- 1. CONFIGURATION & CONSTANTS ---

# World & Display
WIN_W, WIN_H = 1200, 800
FPS = 60
METERS_TO_PIXELS_BASE = 8.0
ZOOM_STEP = 1.1

# Waypoints
WAYPOINTS_M = [
    (10, 90), (80, 90), (80, 50),
    (20, 50), (20, 20), (90, 20)
]

# Car Physics & Dimensions
CAR_LEN_M, CAR_WID_M, WHEELBASE_M = 4.5, 2.0, 2.8
MASS_KG, CARGO_TON = 1500.0, 1.0
P_MAX_W = 80_000.0
CD, FRONTAL_AREA, CRR = 0.35, 2.2, 0.01

# --- YOUR REQUESTED PARAMETERS ---
MAX_ACCEL_CMD = 1.5         # Max desired acceleration is 1.5 m/s^2
MAX_BRAKE_DECEL = 1.5       # Max desired deceleration is 1.5 m/s^2
SPEED_KMPH_EMPTY = 35.0     # Max speed is 35 km/h
SPEED_KMPH_LOADED = 25.0    # Max speed is 35 km/h
TURN_SPEED_KMPH = 20.0      # Max speed for sharp turns is 20 km/h
# ------------------------------------

# Operational Logic
LOAD_UNLOAD_TIME_S = 3.0

# Control System
STEER_MAX_DEG, STEER_RATE_DEGPS = 35.0, 270.0
LOOKAHEAD_GAIN, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M = 0.8, 4.0, 15.0
JERK_LIMIT = 1.0

# Kalman Filter
SENSOR_NOISE_STD_DEV = 0.75

# Derived Constants
SPEED_MS_EMPTY = SPEED_KMPH_EMPTY / 3.6
SPEED_MS_LOADED = SPEED_KMPH_LOADED / 3.6
TURN_SPEED_MS = TURN_SPEED_KMPH / 3.6
STEER_MAX_RAD = math.radians(STEER_MAX_DEG)
STEER_RATE_RADPS = math.radians(STEER_RATE_DEGPS)

# Colors
COLORS = { "DARK": (30, 30, 30), "GREY": (85, 85, 85), "WHITE": (240, 240, 240),
           "BLUE": (40, 120, 220), "RED": (220, 40, 40), "YELLOW": (240, 220, 0) }

# --- 2. DATA STRUCTURES ---
@dataclass
class CarState:
    x: float = 0.0; y: float = 0.0; yaw: float = 0.0; v: float = 0.0
    a: float = 0.0; delta: float = 0.0; s_path: float = 0.0

@dataclass
class ControllerState:
    a_cmd_prev: float = 0.0

@dataclass
class Camera:
    cx: float = 50.0; cy: float = 55.0; zoom: float = 4.0

# --- 3. CORE CLASSES & MODULES ---

class Path:
    def __init__(self, waypoints: List[Tuple[float, float]]):
        self.wp = [np.array(p) for p in waypoints]
        self.s = [0.0]
        for i in range(1, len(self.wp)):
            self.s.append(self.s[-1] + np.linalg.norm(self.wp[i] - self.wp[i-1]))
        self.length = self.s[-1]

    def point_at(self, s_query: float) -> np.ndarray:
        s = np.clip(s_query, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]:
                s_base, s_end, p_base, p_end = self.s[i], self.s[i+1], self.wp[i], self.wp[i+1]
                if s_end - s_base < 1e-6: return p_base
                t = (s - s_base) / (s_end - s_base)
                return p_base + t * (p_end - p_base)
        return self.wp[-1]

    def project(self, p: np.ndarray) -> float:
        min_dist_sq = float('inf')
        best_s = 0.0
        for i in range(len(self.wp) - 1):
            a, b = self.wp[i], self.wp[i + 1]
            seg_vec = b - a
            ap = p - a
            dot_val = np.dot(seg_vec, seg_vec)
            if dot_val < 1e-9:
                continue
            t = np.clip(np.dot(ap, seg_vec) / dot_val, 0, 1)
            proj = a + t * seg_vec  # ✅ define proj first
            dist_sq = np.sum((p - proj)**2)  # ✅ then use it
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_s = self.s[i] + t * np.linalg.norm(seg_vec)
        return best_s


    def get_curvature_at(self, s: float) -> float:
        # <<< FIX: Increased sampling distance from 1.0m to 3.0m for more stable results
        s_before, s_at, s_after = self.point_at(s - 3.0), self.point_at(s), self.point_at(s + 3.0)
        area = 0.5 * abs(s_before[0]*(s_at[1]-s_after[1]) + s_at[0]*(s_after[1]-s_before[1]) + s_after[0]*(s_before[1]-s_at[1]))
        d1, d2, d3 = np.linalg.norm(s_at - s_before), np.linalg.norm(s_after - s_at), np.linalg.norm(s_after - s_before)
        if d1*d2*d3 < 1e-6: return 0.0
        return (4 * area) / (d1 * d2 * d3)

class KalmanFilter:
    def __init__(self, dt):
        self.x=np.zeros(4); self.F=np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]]); self.B=np.array([[0.5*dt**2,0],[dt,0],[0,0.5*dt**2],[0,dt]]); self.H=np.array([[1,0,0,0],[0,0,1,0]]); self.Q,self.R,self.P=np.eye(4)*0.1,np.eye(2)*(SENSOR_NOISE_STD_DEV**2),np.eye(4)
    def predict(self, u: np.ndarray): self.x=self.F@self.x+self.B@u; self.P=self.F@self.P@self.F.T+self.Q
    def update(self, z: np.ndarray): y=z-self.H@self.x; S=self.H@self.P@self.H.T+self.R; K=self.P@self.H.T@np.linalg.inv(S); self.x=self.x+K@y; self.P=(np.eye(4)-K@self.H)@self.P

def resist_forces(v_ms: float, mass_kg: float) -> float: return CRR*mass_kg*9.81+0.5*1.225*CD*FRONTAL_AREA*v_ms**2
def traction_force_from_power(v_ms: float, throttle: float) -> float: return(P_MAX_W*np.clip(throttle,0,1))/max(v_ms,0.5)
def brake_force_from_command(brake_cmd: float, mass_kg: float) -> float: return np.clip(brake_cmd,0,1)*mass_kg*MAX_BRAKE_DECEL

class Car:
    def __init__(self, path: Path, dt: float):
        self.path = path
        p_start, p_ahead = self.path.point_at(0), self.path.point_at(1.0)
        initial_yaw = math.atan2(p_ahead[1]-p_start[1], p_ahead[0]-p_start[0]) + math.pi/2
        self.true_state = CarState(x=p_start[0], y=p_start[1], yaw=initial_yaw)
        self.kf = KalmanFilter(dt); self.kf.x = np.array([p_start[0], 0, p_start[1], 0])
        self.controller = ControllerState()
        self.op_state, self.op_timer, self.current_mass_kg = "GOING_TO_ENDPOINT", 0.0, MASS_KG

    def get_noisy_measurement(self) -> np.ndarray: return np.array([self.true_state.x, self.true_state.y])+np.random.normal(0, SENSOR_NOISE_STD_DEV, 2)

    def update(self, dt: float):
        target_s, target_v_cap, direction = self._update_op_state(dt)
        z = self.get_noisy_measurement()
        u_accel = np.array([
            self.true_state.a * math.cos(self.true_state.yaw),
            self.true_state.a * math.sin(self.true_state.yaw)
        ])
        self.kf.predict(u=u_accel)
        self.kf.update(z=z)
        
        est_pos = np.array([self.kf.x[0], self.kf.x[2]])
        est_vel = np.array([self.kf.x[1], self.kf.x[3]])
        est_speed = np.linalg.norm(est_vel)
        est_s_path = self.path.project(est_pos)

        if direction != 0:
            ld = np.clip(est_speed * LOOKAHEAD_GAIN, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)
            s_target_steer = est_s_path + direction * ld

            # ✅ Fixed: Curvature-based speed limiting
            lookahead_distance = 10.0  # How far ahead to check for a turn
            check_step = 0.5
            max_curvature = 0.0
            s_look = est_s_path
            while s_look < est_s_path + lookahead_distance and s_look < self.path.length:
                curvature = self.path.get_curvature_at(s_look)
                max_curvature = max(max_curvature, curvature)
                s_look += check_step

            # --- STEP 2: Determine speed cap based on that curvature ---
            MAX_LAT_ACCEL = 2.0  # m/s²
            if max_curvature > 1e-4:
                v_turn_cap = min(target_v_cap, math.sqrt(MAX_LAT_ACCEL / max_curvature))
            else:
                v_turn_cap = target_v_cap

            # --- STEP 3: Compute how far we need to slow down to turn speed ---
            v_cur = est_speed
            v_target = v_turn_cap
            if v_cur > v_target:
                d_required = (v_cur ** 2 - v_target ** 2) / (2 * MAX_BRAKE_DECEL)
                if lookahead_distance < d_required:
                    v_turn_cap = v_target  # Slow down NOW

            p_target = self.path.point_at(s_target_steer)
            dx_local = (p_target[0] - est_pos[0]) * math.cos(self.true_state.yaw) + \
                    (p_target[1] - est_pos[1]) * math.sin(self.true_state.yaw)
            dy_local = -(p_target[0] - est_pos[0]) * math.sin(self.true_state.yaw) + \
                        (p_target[1] - est_pos[1]) * math.cos(self.true_state.yaw)
            alpha = math.atan2(dy_local, max(dx_local, 1e-3))
            delta_target = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld)
            delta_cmd = np.clip(delta_target, -STEER_MAX_RAD, STEER_MAX_RAD)

            dist_to_target = abs(target_s - est_s_path)
            v_stop_cap = math.sqrt(2 * MAX_BRAKE_DECEL * dist_to_target)
            v_ref = min(target_v_cap, v_stop_cap, v_turn_cap)

            accel_needed = (v_ref - est_speed) / max(0.4, dt)
            accel_cmd = np.clip(accel_needed, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
            accel_cmd = np.clip(
                accel_cmd,
                self.controller.a_cmd_prev - JERK_LIMIT * dt,
                self.controller.a_cmd_prev + JERK_LIMIT * dt
            )
            self.controller.a_cmd_prev = accel_cmd

            # Apply force calculations
            F_resist = resist_forces(self.true_state.v, self.current_mass_kg)
            F_needed = self.current_mass_kg * accel_cmd

            if F_needed >= 0:
                F_engine = F_needed + F_resist
                throttle = (F_engine * max(self.true_state.v, 0.5)) / P_MAX_W
                brake = 0.0
            else:
                F_brake_req = -F_needed + F_resist
                brake = F_brake_req / (self.current_mass_kg * MAX_BRAKE_DECEL)
                throttle = 0.0
        else:
            delta_cmd, throttle, brake = 0.0, 0.0, 1.0

        self.true_state.delta = np.clip(
            delta_cmd,
            self.true_state.delta - STEER_RATE_RADPS * dt,
            self.true_state.delta + STEER_RATE_RADPS * dt
        )

        F_trac = traction_force_from_power(self.true_state.v, throttle)
        F_brake = brake_force_from_command(brake, self.current_mass_kg)
        F_res = resist_forces(self.true_state.v, self.current_mass_kg)
        F_net = F_trac - F_res - F_brake
        self.true_state.a = F_net / self.current_mass_kg
        self.true_state.v = max(0.0, self.true_state.v + self.true_state.a * dt)
        self.true_state.yaw += (self.true_state.v * math.tan(self.true_state.delta) / WHEELBASE_M) * dt
        self.true_state.x += self.true_state.v * math.cos(self.true_state.yaw) * dt
        self.true_state.y += self.true_state.v * math.sin(self.true_state.yaw) * dt
        self.true_state.s_path = self.path.project(np.array([self.true_state.x, self.true_state.y]))

    def _update_op_state(self, dt):
        if self.op_state=="GOING_TO_ENDPOINT":
            if self.true_state.s_path>=self.path.length-2.0 and self.true_state.v<0.5: self.op_state,self.op_timer="LOADING",LOAD_UNLOAD_TIME_S
            return self.path.length, SPEED_MS_EMPTY, +1
        elif self.op_state=="LOADING":
            if self.op_timer>0: self.op_timer-=dt
            else: self.op_state="RETURNING_TO_START"; self.current_mass_kg=MASS_KG+CARGO_TON*1000
            return self.true_state.s_path, 0, 0
        elif self.op_state=="RETURNING_TO_START":
            if self.true_state.s_path<=2.0 and self.true_state.v<0.5: self.op_state,self.op_timer="UNLOADING",LOAD_UNLOAD_TIME_S
            return 0.0, SPEED_MS_LOADED, -1
        elif self.op_state=="UNLOADING":
            if self.op_timer>0: self.op_timer-=dt
            else: self.op_state="GOING_TO_ENDPOINT"; self.current_mass_kg=MASS_KG
            return self.true_state.s_path, 0, 0
        return self.true_state.s_path, 0, 0

# --- 4. RENDERING & UI ---
def world_to_screen(xm, ym, cam: Camera, win_w, win_h): S=METERS_TO_PIXELS_BASE*cam.zoom; return int(win_w/2+(xm-cam.cx)*S), int(win_h/2+(cam.cy-ym)*S)

def draw_road(screen, path: Path, cam: Camera, win_w, win_h):
    S=METERS_TO_PIXELS_BASE*cam.zoom; points=[world_to_screen(p[0],p[1],cam,win_w,win_h) for p in path.wp]
    if len(points)>1: pygame.draw.lines(screen,COLORS["GREY"],False,points,max(2,int(10*S)))
    for p in path.wp: pygame.draw.circle(screen,COLORS["RED"],world_to_screen(p[0],p[1],cam,win_w,win_h),int(S*1.5))

def draw_car(screen, car: Car, cam: Camera, win_w, win_h):
    S=METERS_TO_PIXELS_BASE*cam.zoom; w,h=int(CAR_WID_M*S),int(CAR_LEN_M*S)
    if w<1 or h<1: return
    surf=pygame.Surface((w,h),pygame.SRCALPHA); cargo_ratio=(car.current_mass_kg-MASS_KG)/(CARGO_TON*1000)
    body_color=tuple(np.array(COLORS["BLUE"])*(1-cargo_ratio)+np.array([120,120,120])*cargo_ratio)
    pygame.draw.rect(surf,body_color,surf.get_rect(),border_radius=int(w/4))
    pygame.draw.rect(surf,COLORS["DARK"],pygame.Rect(0,0,w,h//4),border_top_left_radius=int(w/4),border_top_right_radius=int(w/4))
    rotated=pygame.transform.rotate(surf,-math.degrees(car.true_state.yaw)); xs,ys=world_to_screen(car.true_state.x,car.true_state.y,cam,win_w,win_h)
    screen.blit(rotated,rotated.get_rect(center=(xs,ys)))
    kf_x,kf_y=world_to_screen(car.kf.x[0],car.kf.x[2],cam,win_w,win_h); pygame.draw.circle(screen,COLORS["YELLOW"],(kf_x,kf_y),int(S*1.5),2)

def draw_hud(screen, font: pygame.font.Font, car: Car):
    texts=[f"Speed: {car.true_state.v*3.6:5.1f} km/h",f"Mass: {car.current_mass_kg:6.1f} kg",f"State: {car.op_state}",f"Accel: {car.true_state.a:5.2f} m/s^2","Yellow circle is KF estimate"]
    for i,text in enumerate(texts): screen.blit(font.render(text,True,COLORS["WHITE"]),(10,10+i*22))

# --- 5. MAIN APPLICATION LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Advanced Autonomous Car Simulation")
    clock, font = pygame.time.Clock(), pygame.font.SysFont("Consolas", 18)
    path = Path(WAYPOINTS_M)
    car = Car(path, 1.0 / FPS)
    cam = Camera()
    
    running = True
    while running:
        dt = min(0.1, clock.tick(FPS) / 1000.0)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEWHEEL: cam.zoom *= (ZOOM_STEP if event.y > 0 else 1.0/ZOOM_STEP)
        
        car.update(dt)
        
        screen.fill(COLORS["DARK"])
        draw_road(screen, path, cam, WIN_W, WIN_H)
        draw_car(screen, car, cam, WIN_W, WIN_H)
        draw_hud(screen, font, car)
        pygame.display.flip()
        
    pygame.quit()

if __name__ == '__main__':
    main()