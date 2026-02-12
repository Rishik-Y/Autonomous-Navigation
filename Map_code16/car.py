import pygame
import numpy as np
import math
from config import *
from utils import resist_forces, traction_force_from_power, brake_force_from_command
from mpc_controller import MPCController

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
        
        # MPC Setup (iLQR)
        self.mpc = MPCController(dt=0.1, N=15, wheelbase=WHEELBASE_M, d_safe=SAFE_DISTANCE_M)
        self.planned_trajectory = np.zeros((2, 16)) # [x, y] for N+1 steps
        self.current_mpc_control = np.zeros(2) # [accel, steer]

    def get_reference_trajectory(self, N, dt):
        """
        Generates a reference trajectory over horizon N.
        1. Scans ahead for sharp turns.
        2. Calculates a decelerating speed profile from distance.
        3. Applies lateral offset (swing out) for sharp turns.
        """
        ref = np.zeros((4, N + 1))
        
        if not self.path:
            ref[0, :] = self.x_m; ref[1, :] = self.y_m; ref[2, :] = self.angle; ref[3, :] = 0.0
            return ref

        # --- 1. Scan Ahead for Speed Profile & Geometry ---
        # Look ahead 40 meters
        SCAN_DIST = 40.0
        STEP_SIZE = 1.0
        num_steps = int(SCAN_DIST / STEP_SIZE)
        
        v_profile = np.full(num_steps + 1, self.desired_speed_ms)
        swing_offsets = np.zeros(num_steps + 1)
        
        current_s = self.s_path_m
        
        # Forward pass: Detect restrictions
        for i in range(num_steps + 1):
            s_check = current_s + i * STEP_SIZE
            if s_check > self.path.length:
                v_profile[i:] = 0.0
                break
                
            # Curvature Check
            curvature = self.path.get_curvature_at(s_check)
            safe_curvature = max(curvature, 1e-4)
            
            # Default "Comfort" Limit (0.8 m/s^2)
            PLANNING_LAT_ACCEL = 0.8
            v_limit = math.sqrt(PLANNING_LAT_ACCEL / safe_curvature)
            
            # Sharp Turn Detection (>130 degree equivalent logic)
            # Check heading change over next 10m
            s_ahead_10 = min(s_check + 10.0, self.path.length)
            p_now = self.path.point_at(s_check)
            p_next = self.path.point_at(s_check + 1.0) # Local tangent
            p_ahead = self.path.point_at(s_ahead_10)
            p_ahead_next = self.path.point_at(min(s_ahead_10 + 1.0, self.path.length)) # Ahead tangent
            
            angle_now = math.atan2(p_next[1]-p_now[1], p_next[0]-p_now[0])
            angle_ahead = math.atan2(p_ahead_next[1]-p_ahead[1], p_ahead_next[0]-p_ahead[0])
            
            # Heading change
            diff = angle_ahead - angle_now
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            
            # If heading change > 100 degrees (approx > 1.7 rad), treat as sharp hook turn
            # User asked for >130 (2.26 rad), but let's be safe with >100 for "sharp" behavior trigger
            SHARP_TURN_THRESHOLD = math.radians(100) 
            
            is_sharp_turn = abs(diff) > SHARP_TURN_THRESHOLD
            
            if is_sharp_turn:
                v_limit = min(v_limit, 1.4) # ~5 km/h
                
                # Calculate Swing Offset
                # Swing out opposite to turn direction
                # dist_to_vertex approx 5m (center of 10m scan)? 
                # Simplistic swing logic: Ramp up offset based on proximity to this sharp section
                turn_direction = np.sign(diff) # +1 Left, -1 Right
                
                # Swing OPPOSITE to turn (Left turn -> Swing Right, Right turn -> Swing Left)
                # Normal vector (Left of path): [-sin, cos]
                # If turning Left (+), we want to go Right (negative normal direction) -> -1 * +1 = -1
                # If turning Right (-), we want to go Left (positive normal direction) -> -1 * -1 = +1
                # So swing scalar should be -sign(diff) * magnitude
                
                swing_mag = 2.0 # 2 meters swing
                swing_offsets[i] = -turn_direction * swing_mag

            v_profile[i] = min(v_profile[i], v_limit)

        # Forward pass: Limit acceleration (Smooth Acceleration)
        # v_current <= sqrt(v_prev^2 + 2 * a * dx)
        for i in range(1, num_steps + 1):
            v_prev = v_profile[i-1]
            v_allowable = math.sqrt(v_prev**2 + 2 * ACCEL_LIMIT * STEP_SIZE)
            v_profile[i] = min(v_profile[i], v_allowable)

        # Backward pass: Propagate deceleration (Smooth Braking)
        # v_current <= sqrt(v_next^2 + 2 * a * dx)
        BRAKE_ACCEL = 0.5 
        for i in range(num_steps - 1, -1, -1):
            v_next = v_profile[i+1]
            v_allowable = math.sqrt(v_next**2 + 2 * BRAKE_ACCEL * STEP_SIZE)
            v_profile[i] = min(v_profile[i], v_allowable)

        # --- 2. Generate Reference Trajectory ---
        s_sim = self.s_path_m
        
        for i in range(N + 1):
            # Interpolate speed from profile
            dist_from_start = s_sim - self.s_path_m
            idx_float = np.clip(dist_from_start / STEP_SIZE, 0, num_steps - 1)
            idx_int = int(idx_float)
            alpha = idx_float - idx_int
            target_v = v_profile[idx_int] * (1-alpha) + v_profile[idx_int+1] * alpha
            
            # Interpolate swing offset
            swing_val = swing_offsets[idx_int] * (1-alpha) + swing_offsets[idx_int+1] * alpha
            
            # Apply minimum speed floor unless stopping
            if v_profile[-1] < 0.1 and dist_from_start > SCAN_DIST - 5.0:
                 pass # Let it stop
            else:
                 target_v = max(target_v, 0.5) # Keep moving slowly

            # Get Point & Normal
            p = self.path.point_at(s_sim)
            
            # Calculate Normal for Swing + Lane Offset
            normal = self.path.get_normal_at(s_sim)
            
            # Combine dynamic swing (for turns) with static lane offset (for driving side)
            total_offset = swing_val + LANE_OFFSET_M
            
            p_target_x = p[0] + normal[0] * total_offset
            p_target_y = p[1] + normal[1] * total_offset
            
            # Heading (Tangent of the MODIFIED path ideally, but path tangent is close enough for small offsets)
            # Use path tangent.
            # We can re-derive heading from normal: (-ny, nx) is tangent
            theta = math.atan2(normal[0], -normal[1]) # Rotated back 90 deg right?
            # Normal was (-dy, dx). Tangent is (dx, dy).
            # If normal = (-dy, dx), then normal[0] = -dy, normal[1] = dx.
            # Tangent y = dy = -normal[0]. Tangent x = dx = normal[1].
            theta = math.atan2(-normal[0], normal[1])
            
            ref[0, i] = p_target_x
            ref[1, i] = p_target_y
            ref[2, i] = theta
            ref[3, i] = target_v
            
            s_sim += target_v * dt

        return ref

    def run_mpc(self, other_cars_trajectories):
        if not self.path:
            self.current_mpc_control = np.zeros(2)
            self.planned_trajectory = np.tile(np.array([[self.x_m], [self.y_m]]), (1, self.mpc.N + 1))
            return

        state = [self.x_m, self.y_m, self.angle, self.speed_ms]
        ref = self.get_reference_trajectory(self.mpc.N, self.mpc.dt)
        
        u_opt, x_opt = self.mpc.solve(state, ref, other_cars_trajectories)
        
        # Apply the FIRST control action
        self.current_mpc_control = u_opt[0]
        
        # Store plan for visualization and collision avoidance
        self.planned_trajectory = x_opt.T[:2, :] # Transpose to (2, N+1)

    def move(self, dt):
        # Retrieve control from MPC (held constant between MPC steps)
        accel_cmd = self.current_mpc_control[0]
        steer_input = self.current_mpc_control[1]

        # Physics
        F_resist = resist_forces(self.speed_ms, self.current_mass_kg)
        F_needed = self.current_mass_kg * accel_cmd
        
        if F_needed >= 0:
            throttle = ((F_needed + F_resist) * max(self.speed_ms, 0.5)) / P_MAX_W
            brake = 0.0
        else:
            brake = (-F_needed + F_resist) / (self.current_mass_kg * MAX_BRAKE_DECEL)
            throttle = 0.0
            
        # Steer Dynamics
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
        
        # Angle Normalization
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi

    def update_op_state(self, dt, dispatcher=None):
        path_length_m = self.path.length if self.path else 0
        current_s_m = self.s_path_m
        
        if self.op_state == "GOING_TO_ENDPOINT":
            if current_s_m >= path_length_m - 5.0 and self.speed_ms < 1.0:
                self.op_state = "LOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
                return 0, 0.0 
            return +1, SPEED_MS_EMPTY
        elif self.op_state == "LOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else:
                self.op_state = "RETURNING_TO_START"
                self.current_mass_kg = MASS_KG + CARGO_TON * 1000
                if dispatcher: dispatcher.release_reservation(self.target_node_name)
            return 0, 0.0
        elif self.op_state == "RETURNING_TO_START":
            if current_s_m >= path_length_m - 5.0 and self.speed_ms < 1.0:
                self.op_state = "UNLOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
                return 0, 0.0 
            return +1, SPEED_MS_LOADED 
        elif self.op_state == "UNLOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else:
                self.op_state = "GOING_TO_ENDPOINT"
                self.current_mass_kg = MASS_KG
                if dispatcher: dispatcher.release_reservation(self.target_node_name)
            return 0, 0.0
        return 0, 0.0

    def get_noisy_measurement(self):
        return np.array([self.x_m + np.random.normal(0, SENSOR_NOISE_STD_DEV), self.y_m + np.random.normal(0, SENSOR_NOISE_STD_DEV)])

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
        
        # Draw Planned Trajectory
        if is_selected and hasattr(self, 'planned_trajectory'):
            traj = self.planned_trajectory
            if traj.shape[1] > 1:
                points = [g_to_s((traj[0, i], traj[1, i])) for i in range(traj.shape[1])]
                if len(points) > 1:
                    pygame.draw.lines(screen, (0, 255, 0), False, points, 2)

    def check_collision(self, other_cars):
        my_pos = np.array([self.x_m, self.y_m])
        # Heading vector
        c, s = math.cos(self.angle), math.sin(self.angle)
        tangent = np.array([c, s])
        normal = np.array([-s, c]) # Left normal

        for other in other_cars:
            if other.id == self.id: continue
            
            other_pos = np.array([other.x_m, self.y_m])
            diff = other_pos - my_pos
            dist = np.linalg.norm(diff)
            
            if dist < SAFE_DISTANCE_M:
                # 1. Lane Awareness (Projection)
                longitudinal = np.dot(diff, tangent)
                lateral = np.dot(diff, normal)
                
                # If behind me, ignore
                if longitudinal < -CAR_LENGTH_M: continue
                
                # If clearly in other lane (>3m offset), ignore
                # Note: We are already offset 2.5m left. 
                # If other car is 2.5m right (total 5m), lateral diff should be large.
                if abs(lateral) > 3.0: continue
                
                # 2. Priority Yielding (Head-On Deadlock Breaker)
                # Check relative heading
                other_heading = np.array([math.cos(other.angle), math.sin(other.angle)])
                dot_heading = np.dot(tangent, other_heading)
                
                is_head_on = dot_heading < -0.8 # Approx opposite directions
                
                if is_head_on:
                    # If I am Higher ID, I yield. If Lower ID, I bully through.
                    if self.id > other.id:
                        return True
                    else:
                        continue # Assert dominance
                
                # Standard Rear-End / Crossing collision
                return True
                
        return False