import pygame
import numpy as np
import math
from config import *
from utils import resist_forces, traction_force_from_power, brake_force_from_command

class Car:
    def __init__(self, car_id, x_m, y_m, angle=0):
        self.id = car_id
        self.x_m, self.y_m, self.angle = x_m, y_m, angle
        self.speed_ms, self.accel_ms2, self.steer_angle = 0.0, 0.0, 0.0
        self.current_mass_kg = MASS_KG
        self.op_state = "GOING_TO_ENDPOINT"
        self.op_timer = 0.0
        self.a_cmd_prev = 0.0
        self.s_path_m = 0.0
        self.current_node_name = ""
        self.target_node_name = ""
        self.path = None 
        self.desired_speed_ms = 0.0

    def move(self, accel_cmd, steer_input, dt):
        accel_cmd = np.clip(accel_cmd, self.a_cmd_prev - JERK_LIMIT * dt, self.a_cmd_prev + JERK_LIMIT * dt)
        self.a_cmd_prev = accel_cmd
        F_resist = resist_forces(self.speed_ms, self.current_mass_kg)
        F_needed = self.current_mass_kg * accel_cmd
        if F_needed >= 0:
            throttle = ((F_needed + F_resist) * max(self.speed_ms, 0.5)) / P_MAX_W
            brake = 0.0
        else:
            brake = (-F_needed + F_resist) / (self.current_mass_kg * MAX_BRAKE_DECEL)
            throttle = 0.0
        self.steer_angle = np.clip(steer_input, self.steer_angle - STEER_RATE_RADPS * dt, self.steer_angle + STEER_RATE_RADPS * dt)
        F_trac = traction_force_from_power(self.speed_ms, throttle)
        F_brake = brake_force_from_command(brake, self.current_mass_kg)
        F_net = F_trac - F_resist - F_brake
        self.accel_ms2 = F_net / self.current_mass_kg
        self.speed_ms = max(0.0, self.speed_ms + self.accel_ms2 * dt)
        if abs(self.steer_angle) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(self.steer_angle)
            angular_velocity = self.speed_ms / turn_radius
            self.angle += angular_velocity * dt
        self.x_m += self.speed_ms * math.cos(self.angle) * dt
        self.y_m += self.speed_ms * math.sin(self.angle) * dt

    def update_op_state(self, dt):
        path_length_m = self.path.length if self.path else 0
        current_s_m = self.s_path_m
        
        if self.op_state == "GOING_TO_ENDPOINT":
            if current_s_m >= path_length_m - 2.0 and self.speed_ms < 0.5:
                self.op_state = "LOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
                return 0, 0.0 
            return +1, SPEED_MS_EMPTY
        elif self.op_state == "LOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else:
                self.op_state = "RETURNING_TO_START"
                self.current_mass_kg = MASS_KG + CARGO_TON * 1000
            return 0, 0.0
        elif self.op_state == "RETURNING_TO_START":
            if current_s_m >= path_length_m - 2.0 and self.speed_ms < 0.5:
                self.op_state = "UNLOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
                return 0, 0.0 
            return +1, SPEED_MS_LOADED 
        elif self.op_state == "UNLOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else:
                self.op_state = "GOING_TO_ENDPOINT"
                self.current_mass_kg = MASS_KG
            return 0, 0.0
        return 0, 0.0

    def get_noisy_measurement(self):
        return np.array([self.x_m + np.random.normal(0, SENSOR_NOISE_STD_DEV), self.y_m + np.random.normal(0, SENSOR_NOISE_STD_DEV)])

    def check_collision(self, other_cars):
        my_pos = np.array([self.x_m, self.y_m])
        my_heading = np.array([math.cos(self.angle), math.sin(self.angle)])
        for other in other_cars:
            if other.id == self.id: continue
            other_pos = np.array([other.x_m, other.y_m])
            dist = np.linalg.norm(other_pos - my_pos)
            if dist < SAFE_DISTANCE_M:
                to_other = other_pos - my_pos
                to_other_norm = np.linalg.norm(to_other)
                if to_other_norm > 1e-3:
                    to_other /= to_other_norm
                    dot_prod = np.dot(my_heading, to_other)
                    if dot_prod > 0.5:
                        return True
        return False

    def draw(self, screen, g_to_s, is_selected=False):
        car_center_screen = g_to_s((self.x_m, self.y_m))
        length_px = CAR_LENGTH_M * METERS_TO_PIXELS * g_to_s.scale
        width_px = CAR_WIDTH_M * METERS_TO_PIXELS * g_to_s.scale
        if length_px < 1 or width_px < 1: return
        car_surface = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
        
        if is_selected:
            body_color = RED
        else:
            cargo_ratio = (self.current_mass_kg - MASS_KG) / (CARGO_TON * 1000)
            body_color = tuple(np.array(CAR_COLOR) * (1 - cargo_ratio) + np.array([120, 120, 120]) * cargo_ratio)
            
        car_surface.fill(body_color)
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        rect = rotated_surface.get_rect(center=car_center_screen)
        screen.blit(rotated_surface, rect.topleft)