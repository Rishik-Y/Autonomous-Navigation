import math
import numpy as np

from panda3d.core import BitMask32, CollisionBox, CollisionNode, Point3

from config import *
from mpc_controller import MPCController
from cyberpunk_driver import CyberpunkDriver
from utils import (
    aerodynamic_drag_force,
    estimate_max_uphill_speed_ms,
    gravity_force_along_slope,
    max_brake_force,
    max_traction_force,
    rolling_resistance_force,
)


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
        self.heightmap = None
        self.pitch_rad = 0.0
        self.roll_rad = 0.0
        self.throttle_pct = 0.0
        self.brake_pct = 0.0
        self.power_limited = False

        self.cyberpunk_driver = CyberpunkDriver()
        self.last_traffic_target_ms = SPEED_MS_EMPTY

        self.turn_target_angle = 0.0
        self.turn_next_state = ""
        self.turn_phase = 0
        self.turn_phase_timer = 0.0

        self.mpc = MPCController(dt=0.1, N=20, wheelbase=WHEELBASE_M, d_safe=SAFE_DISTANCE_M)
        self.planned_trajectory = np.zeros((2, 21))
        self.current_mpc_control = np.zeros(2)

        self.node = None
        self.model = None
        self.collision_np = None

    def _sample_terrain_pose(self, heightmap):
        if heightmap is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        heading_x = math.cos(self.angle)
        heading_y = math.sin(self.angle)
        left_x = -heading_y
        left_y = heading_x
        right_x = -left_x
        right_y = -left_y

        half_wb = 0.5 * WHEELBASE_M
        half_track = 0.5 * TRACK_WIDTH_M

        front_x = self.x_m + heading_x * half_wb
        front_y = self.y_m + heading_y * half_wb
        rear_x = self.x_m - heading_x * half_wb
        rear_y = self.y_m - heading_y * half_wb
        left_wheel_x = self.x_m + left_x * half_track
        left_wheel_y = self.y_m + left_y * half_track
        right_wheel_x = self.x_m + right_x * half_track
        right_wheel_y = self.y_m + right_y * half_track

        z_center = float(heightmap.get_height_at_world(self.x_m, self.y_m))
        z_front = float(heightmap.get_height_at_world(front_x, front_y))
        z_rear = float(heightmap.get_height_at_world(rear_x, rear_y))
        z_left = float(heightmap.get_height_at_world(left_wheel_x, left_wheel_y))
        z_right = float(heightmap.get_height_at_world(right_wheel_x, right_wheel_y))

        pitch = math.atan2(z_front - z_rear, WHEELBASE_M)
        roll = math.atan2(z_right - z_left, TRACK_WIDTH_M)
        return z_center, pitch, roll, z_front, z_rear

    def attach_visual(self, parent_np):
        from direct.showbase.ShowBaseGlobal import base

        self.node = parent_np.attachNewNode(f"truck_{self.id}")
        self.model = base.loader.loadModel("models/misc/rgbCube")
        self.model.reparentTo(self.node)
        self.model.setScale(CAR_WIDTH_M * 0.5, CAR_LENGTH_M * 0.5, CAR_HEIGHT_M * 0.5)
        self.model.setPos(0.0, 0.0, CAR_HEIGHT_M * 0.5)

        cnode = CollisionNode(f"truck_col_{self.id}")
        cnode.addSolid(CollisionBox(Point3(0.0, 0.0, CAR_HEIGHT_M * 0.5), CAR_LENGTH_M * 0.5, CAR_WIDTH_M * 0.5, CAR_HEIGHT_M * 0.5))
        cnode.setIntoCollideMask(BitMask32.bit(TRUCK_COLLIDE_MASK_BIT))
        cnode.setFromCollideMask(BitMask32.allOff())
        self.collision_np = self.node.attachNewNode(cnode)
        self.collision_np.setTag("truck_id", str(self.id))

    def _state_color(self, is_selected=False):
        if is_selected:
            return COLOR_SELECTED
        if self.op_state in ("LOADING", "UNLOADING", "TURNING_AROUND"):
            return COLOR_LOADING
        cargo_kg = max(0.0, self.current_mass_kg - MASS_KG)
        if cargo_kg > 1.0:
            return COLOR_LOADED
        return COLOR_EMPTY

    def update_visual(self, heightmap, is_selected=False):
        if not self.node:
            return
        self.heightmap = heightmap
        z_center, self.pitch_rad, self.roll_rad, z_front, z_rear = self._sample_terrain_pose(heightmap)
        
        # Calculate anti-clipping Z (average of front and rear wheels + 0.2m tire/suspension clearance)
        z_actual = ((z_front + z_rear) / 2.0) + 0.2
        
        self.node.setPos(float(self.x_m), float(self.y_m), z_actual)
        
        # Offset heading by -90 degrees to align Y-Forward model with 2D +X movement vector
        heading_deg = math.degrees(self.angle) - 90.0
        
        self.node.setHpr(
            heading_deg,
            math.degrees(self.pitch_rad),
            math.degrees(self.roll_rad),
        )
        if self.model:
            self.model.setColor(*self._state_color(is_selected=is_selected))

    def get_reference_trajectory(self, N, dt, other_cars=None, heightmap=None):
        ref = np.zeros((4, N + 1))
        active_heightmap = heightmap if heightmap is not None else self.heightmap

        if not self.path:
            ref[0, :] = self.x_m; ref[1, :] = self.y_m; ref[2, :] = self.angle; ref[3, :] = 0.0
            return ref

        follow_speed_cap = self.desired_speed_ms
        if other_cars:
            follow_speed_cap = self.cyberpunk_driver.calculate_target_speed(self, other_cars, self.desired_speed_ms)

        self.last_traffic_target_ms = follow_speed_cap

        SCAN_DIST = 40.0
        STEP_SIZE = 1.0
        num_steps = int(SCAN_DIST / STEP_SIZE)

        base_speed = min(self.desired_speed_ms, follow_speed_cap)
        v_profile = np.full(num_steps + 1, base_speed)
        swing_offsets = np.zeros(num_steps + 1)
        accel_cap_profile = np.full(num_steps + 1, ACCEL_LIMIT)
        brake_cap_profile = np.full(num_steps + 1, MAX_BRAKE_DECEL)

        current_s = self.s_path_m

        for i in range(num_steps + 1):
            s_check = current_s + i * STEP_SIZE
            if s_check > self.path.length:
                v_profile[i:] = 0.0
                break

            curvature = self.path.get_curvature_at(s_check)
            safe_curvature = max(curvature, 1e-4)

            PLANNING_LAT_ACCEL = 0.8
            v_limit = math.sqrt(PLANNING_LAT_ACCEL / safe_curvature)

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

            pitch = 0.0
            if active_heightmap is not None:
                s_prev = max(0.0, s_check - 0.5)
                s_next = min(self.path.length, s_check + 0.5)
                p_prev = self.path.point_at(s_prev)
                p_next_slope = self.path.point_at(s_next)
                dz = float(active_heightmap.get_height_at_world(p_next_slope[0], p_next_slope[1])) - float(
                    active_heightmap.get_height_at_world(p_prev[0], p_prev[1])
                )
                ds_xy = max(np.linalg.norm(p_next_slope - p_prev), 1e-3)
                pitch = math.atan2(dz, ds_xy)

            if pitch > 0.0:
                uphill_cap = estimate_max_uphill_speed_ms(
                    pitch,
                    self.current_mass_kg,
                    power_w=P_MAX_W,
                    v_max=max(base_speed * UPHILL_SPEED_SEARCH_MULTIPLIER, MIN_UPHILL_SEARCH_SPEED_MS),
                )
                v_limit = min(v_limit, uphill_cap)

            gravity_force = gravity_force_along_slope(self.current_mass_kg, pitch)
            rolling_force = rolling_resistance_force(self.current_mass_kg, pitch)
            drag_force = aerodynamic_drag_force(max(v_profile[i], 0.5))
            traction_limit = max_traction_force(max(v_profile[i], 0.5), power_w=P_MAX_W)
            brake_limit = max_brake_force(self.current_mass_kg)

            accel_phys = max(0.0, (traction_limit - gravity_force - rolling_force - drag_force) / self.current_mass_kg)
            brake_phys = max(
                MIN_BRAKE_PROFILE_DECEL,
                (brake_limit + gravity_force + rolling_force + drag_force) / self.current_mass_kg,
            )
            accel_cap_profile[i] = min(ACCEL_LIMIT, accel_phys)
            brake_cap_profile[i] = min(MAX_BRAKE_DECEL, brake_phys)

            if pitch < 0.0:
                downhill_factor = float(
                    np.clip(
                        brake_cap_profile[i] / max(MAX_BRAKE_DECEL, 1e-6),
                        MIN_DOWNHILL_SPEED_FACTOR,
                        1.0,
                    )
                )
                v_limit = min(v_limit, base_speed * downhill_factor)

            v_profile[i] = min(v_profile[i], v_limit)

        for i in range(1, num_steps + 1):
            v_prev = v_profile[i-1]
            v_allowable = math.sqrt(max(0.0, v_prev**2 + 2 * accel_cap_profile[i-1] * STEP_SIZE))
            v_profile[i] = min(v_profile[i], v_allowable)

        for i in range(num_steps - 1, -1, -1):
            v_next = v_profile[i+1]
            v_allowable = math.sqrt(max(0.0, v_next**2 + 2 * brake_cap_profile[i] * STEP_SIZE))
            v_profile[i] = min(v_profile[i], v_allowable)

        s_sim = self.s_path_m

        for i in range(N + 1):
            dist_from_start = s_sim - self.s_path_m
            idx_float = np.clip(dist_from_start / STEP_SIZE, 0, num_steps - 1)
            idx_int = int(idx_float)
            alpha = idx_float - idx_int
            target_v = v_profile[idx_int] * (1-alpha) + v_profile[idx_int+1] * alpha

            swing_val = swing_offsets[idx_int] * (1-alpha) + swing_offsets[idx_int+1] * alpha

            if not (v_profile[-1] < 0.1 and dist_from_start > SCAN_DIST - 5.0):
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

    def run_mpc(self, other_cars_trajectories, other_cars=None):
        if not self.path:
            self.current_mpc_control = np.zeros(2)
            self.planned_trajectory = np.tile(np.array([[self.x_m], [self.y_m]]), (1, self.mpc.N + 1))
            return

        state = [self.x_m, self.y_m, self.angle, self.speed_ms]
        ref = self.get_reference_trajectory(
            self.mpc.N,
            self.mpc.dt,
            other_cars=other_cars,
            heightmap=self.heightmap,
        )

        u_opt, x_opt = self.mpc.solve(state, ref, other_cars_trajectories)
        self.current_mpc_control = u_opt[0]
        self.planned_trajectory = x_opt.T[:2, :]

    def move(self, dt, heightmap=None):
        if heightmap is not None:
            self.heightmap = heightmap

        accel_cmd = self.current_mpc_control[0]
        steer_input = self.current_mpc_control[1]

        if self.last_traffic_target_ms < 0.1 and self.speed_ms > 0.1:
            accel_cmd = -MAX_BRAKE_DECEL

        accel_cmd = np.clip(accel_cmd, -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
        steer_input = np.clip(steer_input, -STEER_MAX_RAD, STEER_MAX_RAD)

        max_steer_change = STEER_RATE_RADPS * dt
        steer_diff = np.clip(steer_input - self.steer_angle, -max_steer_change, max_steer_change)
        self.steer_angle += steer_diff
        self.steer_angle = np.clip(self.steer_angle, -STEER_MAX_RAD, STEER_MAX_RAD)

        accel_diff = np.clip(accel_cmd - self.a_cmd_prev, -JERK_LIMIT * dt, JERK_LIMIT * dt)
        accel_cmd = self.a_cmd_prev + accel_diff
        self.a_cmd_prev = accel_cmd

        # Sample terrain to get pitch and roll angles
        _, self.pitch_rad, self.roll_rad, _, _ = self._sample_terrain_pose(self.heightmap)

        # Calculate resistive forces FIRST
        gravity_force = gravity_force_along_slope(self.current_mass_kg, self.pitch_rad)
        # Only apply tire friction if moving, otherwise it acts as a phantom pulling force
        rolling_force = rolling_resistance_force(self.current_mass_kg, self.pitch_rad) if self.speed_ms > 0.05 else 0.0
        drag_force = aerodynamic_drag_force(self.speed_ms)

        # What net force is the MPC actually asking for?
        desired_net_force = self.current_mass_kg * accel_cmd

        # Calculate how much engine/brake force is required to achieve that net force
        # F_engine_net - Resistances = desired_net_force
        # F_engine_net = desired_net_force + Resistances
        required_engine_force = desired_net_force + gravity_force + rolling_force + drag_force

        if required_engine_force > 0:
            traction_request = required_engine_force
            brake_request = 0.0
        else:
            traction_request = 0.0
            brake_request = -required_engine_force

        # Clamp to physical limits
        traction_limit = max_traction_force(self.speed_ms, power_w=P_MAX_W)
        brake_limit = max_brake_force(self.current_mass_kg)

        traction_force = min(traction_request, traction_limit)
        brake_force = min(brake_request, brake_limit)
        self.power_limited = traction_request > traction_limit + 1e-6

        # Apply final net acceleration
        net_force = traction_force - brake_force - gravity_force - rolling_force - drag_force
        self.accel_ms2 = net_force / max(self.current_mass_kg, 1e-6)
        self.speed_ms = max(0.0, self.speed_ms + self.accel_ms2 * dt)
        self.speed_ms = min(self.speed_ms, SPEED_MS_EMPTY)

        self.throttle_pct = 100.0 * (traction_force / max(traction_limit, 1e-6)) if traction_force > 0.0 else 0.0
        self.brake_pct = 100.0 * (brake_force / max(brake_limit, 1e-6)) if brake_force > 0.0 else 0.0

        if abs(self.steer_angle) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(self.steer_angle)
            angular_velocity = self.speed_ms / turn_radius
            self.angle += angular_velocity * dt

        self.x_m += self.speed_ms * math.cos(self.angle) * dt
        self.y_m += self.speed_ms * math.sin(self.angle) * dt
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
            if self.op_timer > 0:
                self.op_timer -= dt
            else:
                self.op_state = "TURNING_AROUND"
                self.turn_next_state = "RETURNING_TO_START"
                self.needs_new_path = True
                self.turn_phase = 0
                self.turn_phase_timer = 0.0
                max_cargo = CARGO_TON * 1000
                if dispatcher:
                    actual_coal = dispatcher.consume_coal(self.target_node_name, max_cargo)
                    self.current_mass_kg = MASS_KG + actual_coal
                else:
                    self.current_mass_kg = MASS_KG + max_cargo
                if dispatcher:
                    dispatcher.release_reservation(self.target_node_name)
                return +1, SPEED_MS_LOADED
            return 0, 0.0
        elif self.op_state == "TURNING_AROUND":
            return 0, 0.0
        elif self.op_state == "RETURNING_TO_START":
            if current_s_m >= path_length_m - 5.0 and self.speed_ms < 1.0:
                self.op_state = "UNLOADING"
                self.op_timer = LOAD_UNLOAD_TIME_S
                return 0, 0.0
            return +1, SPEED_MS_LOADED
        elif self.op_state == "UNLOADING":
            if self.op_timer > 0:
                self.op_timer -= dt
            else:
                self.op_state = "TURNING_AROUND"
                self.turn_next_state = "GOING_TO_ENDPOINT"
                self.needs_new_path = True
                self.turn_phase = 0
                self.turn_phase_timer = 0.0
                coal_to_dump = self.current_mass_kg - MASS_KG
                if dispatcher and coal_to_dump > 0:
                    dispatcher.record_coal_dumped(self.target_node_name, coal_to_dump)
                self.current_mass_kg = MASS_KG
                if dispatcher:
                    dispatcher.release_reservation(self.target_node_name)
                return +1, SPEED_MS_EMPTY
            return 0, 0.0
        return 0, 0.0

    def execute_turn_step(self, dt):
        TURN_FWD_SPEED = 2.0
        TURN_REV_SPEED = 1.5
        FWD_DURATION = 1.5
        REV_DURATION = 1.0
        HEADING_TOLERANCE = math.radians(15)

        angle_error = (self.turn_target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_error) < HEADING_TOLERANCE:
            self.angle = self.turn_target_angle
            self.speed_ms = 0.0
            self.steer_angle = 0.0
            self.accel_ms2 = 0.0
            self.op_state = self.turn_next_state
            return True

        turn_sign = 1.0 if angle_error > 0 else -1.0

        if self.turn_phase == 0:
            steer = turn_sign * STEER_MAX_RAD
            speed = TURN_FWD_SPEED
            self.turn_phase_timer += dt
            if self.turn_phase_timer >= FWD_DURATION:
                self.turn_phase = 1
                self.turn_phase_timer = 0.0
        else:
            steer = -turn_sign * STEER_MAX_RAD
            speed = -TURN_REV_SPEED
            self.turn_phase_timer += dt
            if self.turn_phase_timer >= REV_DURATION:
                self.turn_phase = 0
                self.turn_phase_timer = 0.0

        self.steer_angle = steer
        if abs(steer) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(steer)
            angular_velocity = speed / turn_radius
            self.angle += angular_velocity * dt

        self.x_m += speed * math.cos(self.angle) * dt
        self.y_m += speed * math.sin(self.angle) * dt
        self.speed_ms = abs(speed)
        self.accel_ms2 = 0.0
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

        return False

    def get_noisy_measurement(self):
        return np.array([
            self.x_m + np.random.normal(0, SENSOR_NOISE_STD_DEV),
            self.y_m + np.random.normal(0, SENSOR_NOISE_STD_DEV),
        ])

    def destroy(self):
        if self.node is not None:
            self.node.removeNode()
            self.node = None
