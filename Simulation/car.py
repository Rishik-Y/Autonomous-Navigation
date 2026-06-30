import pygame
import numpy as np
import math
from config import *
from utils import resist_forces, traction_force_from_power, brake_force_from_command
from mpc_controller import MPCController
from cyberpunk_driver import CyberpunkDriver
from Map import map_loader as map_data

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
        self.needs_new_path = False
        
        # Cyberpunk Driver Logic
        self.cyberpunk_driver = CyberpunkDriver()
        self.last_traffic_target_ms = SPEED_MS_EMPTY
        
        # K-Turn maneuver state
        self.turn_target_angle = 0.0
        self.turn_next_state = ""
        self.turn_phase = 0        # 0=forward, 1=reverse
        self.turn_phase_timer = 0.0
        
        # MPC Setup (iLQR)
        self.mpc = MPCController(dt=0.1, N=20, wheelbase=WHEELBASE_M, d_safe=SAFE_DISTANCE_M)
        self.planned_trajectory = np.zeros((2, 21)) # [x, y] for N+1 steps
        self.current_mpc_control = np.zeros(2) # [accel, steer]

        # Junction yield state
        self.approaching_junction = None   # node name of the junction we're waiting for
        self.junction_granted = False      # True once dispatcher granted us a slot

    def get_reference_trajectory(self, N, dt, other_cars=None):
        """
        Generates a reference trajectory over horizon N.
        1. Scans ahead for sharp turns.
        2. Calculates a decelerating speed profile from distance.
        3. Applies lateral offset (swing out) for sharp turns.
        4. Reduces speed when following another truck.
        """
        ref = np.zeros((4, N + 1))
        
        if not self.path:
            ref[0, :] = self.x_m; ref[1, :] = self.y_m; ref[2, :] = self.angle; ref[3, :] = 0.0
            return ref

        # --- 0. Following detection: slow down if a truck is ahead on same road ---
        follow_speed_cap = self.desired_speed_ms
        if other_cars:
            follow_speed_cap = self.cyberpunk_driver.calculate_target_speed(self, other_cars, self.desired_speed_ms)
        
        # Store for failsafe usage in move()
        self.last_traffic_target_ms = follow_speed_cap

        # --- 1. Scan Ahead for Speed Profile & Geometry ---
        SCAN_DIST = 40.0
        STEP_SIZE = 1.0
        num_steps = int(SCAN_DIST / STEP_SIZE)
        
        base_speed = min(self.desired_speed_ms, follow_speed_cap)
        v_profile = np.full(num_steps + 1, base_speed)
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
            
            PLANNING_LAT_ACCEL = 0.8
            v_limit = math.sqrt(PLANNING_LAT_ACCEL / safe_curvature)
            
            # Sharp Turn Detection
            s_ahead_10 = min(s_check + 10.0, self.path.length)
            p_now = self.path.point_at(s_check)
            p_next = self.path.point_at(s_check + 1.0)
            p_ahead = self.path.point_at(s_ahead_10)
            p_ahead_next = self.path.point_at(min(s_ahead_10 + 1.0, self.path.length))
            
            angle_now = math.atan2(p_next[1]-p_now[1], p_next[0]-p_now[0])
            angle_ahead = math.atan2(p_ahead_next[1]-p_ahead[1], p_ahead_next[0]-p_ahead[0])
            
            diff = angle_ahead - angle_now
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            
            SHARP_TURN_THRESHOLD = math.radians(100) 
            is_sharp_turn = abs(diff) > SHARP_TURN_THRESHOLD
            
            if is_sharp_turn:
                v_limit = min(v_limit, 1.4)
                turn_direction = np.sign(diff)
                swing_mag = 2.0
                swing_offsets[i] = -turn_direction * swing_mag

            v_profile[i] = min(v_profile[i], v_limit)

        # Forward pass: Smooth Acceleration
        for i in range(1, num_steps + 1):
            v_prev = v_profile[i-1]
            v_allowable = math.sqrt(v_prev**2 + 2 * ACCEL_LIMIT * STEP_SIZE)
            v_profile[i] = min(v_profile[i], v_allowable)

        # Backward pass: Smooth Braking
        BRAKE_ACCEL = 0.5 
        for i in range(num_steps - 1, -1, -1):
            v_next = v_profile[i+1]
            v_allowable = math.sqrt(v_next**2 + 2 * BRAKE_ACCEL * STEP_SIZE)
            v_profile[i] = min(v_profile[i], v_allowable)

        # --- 2. Generate Reference Trajectory ---
        s_sim = self.s_path_m
        
        for i in range(N + 1):
            dist_from_start = s_sim - self.s_path_m
            idx_float = np.clip(dist_from_start / STEP_SIZE, 0, num_steps - 1)
            idx_int = int(idx_float)
            alpha = idx_float - idx_int
            target_v = v_profile[idx_int] * (1-alpha) + v_profile[idx_int+1] * alpha
            
            swing_val = swing_offsets[idx_int] * (1-alpha) + swing_offsets[idx_int+1] * alpha
            
            if v_profile[-1] < 0.1 and dist_from_start > SCAN_DIST - 5.0:
                 pass
            else:
                 target_v = max(target_v, 0.5)

            p = self.path.point_at(s_sim)
            normal = self.path.get_normal_at(s_sim)
            total_offset = swing_val + LANE_OFFSET_M
            
            p_target_x = p[0] + normal[0] * total_offset
            p_target_y = p[1] + normal[1] * total_offset
            
            theta = math.atan2(-normal[0], normal[1])
            
            ref[0, i] = p_target_x
            ref[1, i] = p_target_y
            ref[2, i] = theta
            ref[3, i] = target_v
            
            s_sim += target_v * dt

        return ref

    def run_mpc(self, all_traj_stack_or_list, own_idx=None, other_cars=None):
        """
        Run MPC solver.

        Can be called in two ways:
          1. Threaded batch (normal operation):
             run_mpc(all_traj_stack, own_idx=i, other_cars=cars)
             all_traj_stack: (num_cars, 2, N+1) ndarray; own_idx slices out self.
          2. Single warm-up / post-K-turn (no other trucks yet):
             run_mpc([]) or run_mpc([], other_cars=cars)
             Passes an empty array to the solver, which skips collision costs.
        """
        if not self.path:
            self.current_mpc_control = np.zeros(2)
            self.planned_trajectory = np.tile(np.array([[self.x_m], [self.y_m]]), (1, self.mpc.N + 1))
            return

        # Build the "other trajectories" stacked array
        if isinstance(all_traj_stack_or_list, np.ndarray) and own_idx is not None:
            # Fast path: delete own row from the (num_cars, 2, N+1) stack
            other_traj_arr = np.delete(all_traj_stack_or_list, own_idx, axis=0)
        else:
            # Legacy path: called with [] at startup or post-K-turn
            other_traj_arr = np.empty((0, 2, self.mpc.N + 1))

        state = [self.x_m, self.y_m, self.angle, self.speed_ms]
        ref = self.get_reference_trajectory(self.mpc.N, self.mpc.dt, other_cars=other_cars)
        
        u_opt, x_opt = self.mpc.solve(state, ref, other_traj_arr)
        
        # Apply the FIRST control action
        self.current_mpc_control = u_opt[0]
        
        # Store plan for visualization and collision avoidance
        self.planned_trajectory = x_opt.T[:2, :]  # shape: (2, N+1)


    def move(self, dt):
        """Apply MPC controls using the SAME bicycle kinematic model the MPC uses.
        This eliminates model mismatch — the truck moves exactly as the MPC predicts."""
        accel_cmd = self.current_mpc_control[0]
        
        # --- Pure Pursuit Steering (Async Fix) ---
        # Instead of blindly applying the MPC's raw steering command for the entire async delay 
        # (which causes massive sine-wave oscillations due to oversteering), we use a 60Hz 
        # pure pursuit controller to smoothly track the (x,y) path the MPC actually planned.
        steer_input = 0.0
        if hasattr(self, 'planned_trajectory') and self.planned_trajectory.shape[1] > 1:
            my_pos = np.array([self.x_m, self.y_m])
            # Lookahead scales with speed (e.g. 0.5s ahead, min 4m)
            lookahead_dist = max(4.0, self.speed_ms * 0.5)
            
            target_pt = None
            for i in range(1, self.planned_trajectory.shape[1]):
                pt = self.planned_trajectory[:, i]
                if np.linalg.norm(pt - my_pos) >= lookahead_dist:
                    target_pt = pt
                    break
                    
            if target_pt is None:
                target_pt = self.planned_trajectory[:, -1]
                
            dx = target_pt[0] - self.x_m
            dy = target_pt[1] - self.y_m
            target_angle = math.atan2(dy, dx)
            alpha = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            
            L_d = np.linalg.norm(target_pt - my_pos)
            if L_d > 0.1:
                steer_input = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), L_d)
        
        # --- FAILSAFE: Emergency Override ---
        # If the high-level Cyberpunk logic demands a STOP (target < 0.1 m/s), 
        # but we are still moving, force MAX BRAKING.
        # This prevents the MPC from "gliding" to a stop too slowly.
        if self.last_traffic_target_ms < 0.1 and self.speed_ms > 0.1:
            accel_cmd = -MAX_BRAKE_DECEL

        # Clamp commands
        accel_cmd = np.clip(accel_cmd, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
        steer_input = np.clip(steer_input, -STEER_MAX_RAD, STEER_MAX_RAD)

        # Steering rate limiting (keeps realism)
        max_steer_change = STEER_RATE_RADPS * dt
        steer_diff = np.clip(steer_input - self.steer_angle, -max_steer_change, max_steer_change)
        self.steer_angle += steer_diff
        self.steer_angle = np.clip(self.steer_angle, -STEER_MAX_RAD, STEER_MAX_RAD)

        # Acceleration with jerk limiting
        accel_diff = np.clip(accel_cmd - self.a_cmd_prev, -JERK_LIMIT * dt, JERK_LIMIT * dt)
        accel_cmd = self.a_cmd_prev + accel_diff
        self.a_cmd_prev = accel_cmd

        # Direct bicycle kinematics (matches MPC model exactly)
        self.accel_ms2 = accel_cmd
        self.speed_ms = max(0.0, self.speed_ms + self.accel_ms2 * dt)
        self.speed_ms = min(self.speed_ms, SPEED_MS_EMPTY)  # Cap max speed

        if abs(self.steer_angle) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(self.steer_angle)
            angular_velocity = self.speed_ms / turn_radius
            self.angle += angular_velocity * dt

        self.x_m += self.speed_ms * math.cos(self.angle) * dt
        self.y_m += self.speed_ms * math.sin(self.angle) * dt

        # Angle Normalization
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi

    def _check_junction_yield(self, dispatcher):
        """
        Checks if a junction hub node is within JUNCTION_APPROACH_M metres ahead on the path.
        Returns (yield_needed: bool, junction_node: str | None).
        Called only from GOING_TO_ENDPOINT and RETURNING_TO_START states.
        """
        if not dispatcher or not self.path:
            return False, None

        JUNCTION_APPROACH_M = 30.0
        JUNCTION_CLEAR_M    = 15.0  # behind us — means we've passed it
        my_pos = np.array([self.x_m, self.y_m])

        # If already tracking a junction, check for pass-through or slot release
        if self.approaching_junction is not None:
            node_pos = map_data.NODES.get(self.approaching_junction)
            if node_pos is not None:
                dist = np.linalg.norm(node_pos - my_pos)
                # Truck has passed the junction — release slot
                if self.junction_granted and dist > JUNCTION_CLEAR_M and self.s_path_m > 5.0:
                    # Check we are moving AWAY (junction is behind us on the path)
                    junction_s = self.path.project(node_pos, self.s_path_m)
                    if junction_s < self.s_path_m - 5.0:
                        dispatcher.release_junction(self.approaching_junction, self.id)
                        self.approaching_junction = None
                        self.junction_granted = False
                        return False, None

            if self.junction_granted:
                return False, self.approaching_junction  # Slot held — proceed
            else:
                # Try again this frame (idempotent)
                granted = dispatcher.request_junction(self.approaching_junction, self.id, self.angle)
                self.junction_granted = granted
                return not granted, self.approaching_junction

        # No junction tracked yet — scan path ahead for a hub node
        scan_s = self.s_path_m
        for _ in range(int(JUNCTION_APPROACH_M)):
            scan_s = min(scan_s + 1.0, self.path.length)
            scan_pos = self.path.point_at(scan_s)
            for node_name in dispatcher.hub_nodes:
                node_pos = map_data.NODES.get(node_name)
                if node_pos is None:
                    continue
                if np.linalg.norm(node_pos - scan_pos) < 5.0:
                    # Junction found ahead — request slot
                    self.approaching_junction = node_name
                    granted = dispatcher.request_junction(node_name, self.id, self.angle)
                    self.junction_granted = granted
                    return not granted, node_name
            if scan_s >= self.path.length:
                break

        return False, None

    def update_op_state(self, dt, dispatcher=None):
        path_length_m = self.path.length if self.path else 0
        current_s_m = self.s_path_m
        
        if self.op_state == "GOING_TO_ENDPOINT":
            if current_s_m >= path_length_m - 5.0 and self.speed_ms < 1.0:
                # Clear any held junction slot on arrival
                if self.approaching_junction and self.junction_granted:
                    dispatcher.release_junction(self.approaching_junction, self.id) if dispatcher else None
                self.approaching_junction = None
                self.junction_granted = False
                self.op_state = "LOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
                return 0, 0.0
            # Junction yield check
            yield_needed, _ = self._check_junction_yield(dispatcher)
            if yield_needed:
                return 0, 0.0   # Stop and wait for slot
            return +1, SPEED_MS_EMPTY
        elif self.op_state == "LOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else:
                self.op_state = "TURNING_AROUND"
                self.turn_next_state = "RETURNING_TO_START"
                self.needs_new_path = True
                self.turn_phase = 0
                self.turn_phase_timer = 0.0
                # Consume coal from mine - get actual amount available
                max_cargo = CARGO_TON * 1000
                if dispatcher:
                    actual_coal = dispatcher.consume_coal(self.target_node_name, max_cargo)
                    self.current_mass_kg = MASS_KG + actual_coal
                else:
                    self.current_mass_kg = MASS_KG + max_cargo
                if dispatcher: dispatcher.release_reservation(self.target_node_name)
                return +1, SPEED_MS_LOADED
            return 0, 0.0
        elif self.op_state == "TURNING_AROUND":
            # Maneuver handled by execute_turn_step() in main loop
            return 0, 0.0
        elif self.op_state == "RETURNING_TO_START":
            if current_s_m >= path_length_m - 5.0 and self.speed_ms < 1.0:
                # Clear any held junction slot on arrival
                if self.approaching_junction and self.junction_granted:
                    dispatcher.release_junction(self.approaching_junction, self.id) if dispatcher else None
                self.approaching_junction = None
                self.junction_granted = False
                self.op_state = "UNLOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
                return 0, 0.0
            # Junction yield check
            yield_needed, _ = self._check_junction_yield(dispatcher)
            if yield_needed:
                return 0, 0.0   # Stop and wait for slot
            return +1, SPEED_MS_LOADED
        elif self.op_state == "UNLOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else:
                self.op_state = "TURNING_AROUND"
                self.turn_next_state = "GOING_TO_ENDPOINT"
                self.needs_new_path = True
                self.turn_phase = 0
                self.turn_phase_timer = 0.0
                # Record coal dumped at dump site
                coal_to_dump = self.current_mass_kg - MASS_KG
                if dispatcher and coal_to_dump > 0:
                    dispatcher.record_coal_dumped(self.target_node_name, coal_to_dump)
                self.current_mass_kg = MASS_KG
                if dispatcher: dispatcher.release_reservation(self.target_node_name)
                return +1, SPEED_MS_EMPTY
            return 0, 0.0
        return 0, 0.0

    def execute_turn_step(self, dt):
        """Execute one step of a multi-point K-turn maneuver.
        Uses direct bicycle kinematics (no MPC/full physics).
        Returns True when the turn is complete."""
        TURN_FWD_SPEED = 2.0    # m/s forward crawl
        TURN_REV_SPEED = 1.5    # m/s reverse crawl
        FWD_DURATION = 1.5      # seconds per forward phase
        REV_DURATION = 1.0      # seconds per reverse phase
        HEADING_TOLERANCE = math.radians(15)  # "close enough" to snap

        # Check if heading is close enough to target
        angle_error = (self.turn_target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_error) < HEADING_TOLERANCE:
            # Turn complete — snap to exact heading and transition
            self.angle = self.turn_target_angle
            self.speed_ms = 0.0
            self.steer_angle = 0.0
            self.accel_ms2 = 0.0
            self.op_state = self.turn_next_state
            return True

        # Determine which direction to steer (shorter angular path)
        turn_sign = 1.0 if angle_error > 0 else -1.0

        if self.turn_phase == 0:
            # --- Phase 0: Steer hard + creep FORWARD ---
            steer = turn_sign * STEER_MAX_RAD
            speed = TURN_FWD_SPEED
            self.turn_phase_timer += dt
            if self.turn_phase_timer >= FWD_DURATION:
                self.turn_phase = 1
                self.turn_phase_timer = 0.0
        else:
            # --- Phase 1: Opposite steer + creep REVERSE ---
            # Opposite steer while reversing continues rotation
            # in the same direction (bicycle kinematics property)
            steer = -turn_sign * STEER_MAX_RAD
            speed = -TURN_REV_SPEED
            self.turn_phase_timer += dt
            if self.turn_phase_timer >= REV_DURATION:
                self.turn_phase = 0
                self.turn_phase_timer = 0.0

        # --- Apply bicycle kinematics directly ---
        self.steer_angle = steer
        if abs(steer) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(steer)
            angular_velocity = speed / turn_radius
            self.angle += angular_velocity * dt

        self.x_m += speed * math.cos(self.angle) * dt
        self.y_m += speed * math.sin(self.angle) * dt
        self.speed_ms = abs(speed)
        self.accel_ms2 = 0.0

        # Normalize angle
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

        return False

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
