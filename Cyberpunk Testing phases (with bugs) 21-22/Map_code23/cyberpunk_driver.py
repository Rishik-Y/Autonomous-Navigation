import numpy as np
import math
from config import *

class CyberpunkDriver:
    """
    A hybrid Adaptive Cruise Control (ACC) system.
    Combines Cyberpunk 2077's 'Auto Drive' collision logic with 
    Proportional Control for smoother deceleration.
    """
    def __init__(self):
        # --- Tuning Parameters ---
        
        # 1. Safety Gap Calculation
        # Gap = Base + (Speed * TimeHeadway) + (MassPenalty)
        self.base_gap_m = SAFE_DISTANCE_M  # From config.py (8.0m)
        self.time_headway_s = 1.2          # Seconds to look ahead
        self.mass_penalty_factor = 0.5     # Extra gap per ton over base mass
        
        # 2. Stopping Logic
        self.min_standstill_distance_m = 5.0 # Distance to force 0 speed
        
        # 3. Panic / Cornering Logic
        self.cornering_steer_threshold_rad = math.radians(5.0) # If steering > 5 deg
        self.cornering_slowdown_speed_ms = 4.0 # Crawl speed when tight cornering near traffic

    def calculate_target_speed(self, my_car, other_cars, desired_speed):
        """
        Determines the safe target speed based on traffic.
        """
        # 1. Dynamic Safety Gap Calculation
        # Heavier trucks need more room to stop.
        mass_ratio = my_car.current_mass_kg / MASS_KG
        mass_penalty = max(0, (mass_ratio - 1.0) * self.mass_penalty_factor * 10.0)
        
        # Speed dependent gap (Time Headway)
        speed_gap = my_car.speed_ms * self.time_headway_s
        
        # Total required safety buffer
        required_gap = self.base_gap_m + speed_gap + mass_penalty
        
        # Scan range is slightly larger to smooth the transition
        scan_range = required_gap + 10.0
        
        # 2. Find critical vehicle
        target_speed = desired_speed
        critical_vehicle = None
        min_dist = float('inf')

        my_pos = np.array([my_car.x_m, my_car.y_m])
        my_heading = np.array([math.cos(my_car.angle), math.sin(my_car.angle)])
        my_normal = np.array([-math.sin(my_car.angle), math.cos(my_car.angle)]) # Left Normal

        vehicles_in_range = self.get_vehicles_in_range(my_car, other_cars, scan_range)

        for other in vehicles_in_range:
            # Vector to other
            other_pos = np.array([other.x_m, other.y_m])
            diff = other_pos - my_pos
            dist = np.linalg.norm(diff)
            
            # --- Standard Coordinate Projection ---
            # Forward Distance (relative to my front bumper approx)
            longitudinal_dist = np.dot(diff, my_heading) - (CAR_LENGTH_M / 2.0)
            
            # Lateral Distance (relative to centerline)
            lateral_dist = np.dot(diff, my_normal)
            
            # Filter 1: Is it in front?
            if longitudinal_dist < -2.0: 
                continue # Behind us
                
            # Filter 2: Is it in my lane?
            # Standard lane width is ~3.5m. 
            if abs(lateral_dist) > 3.0: 
                continue # In other lane
                
            # Filter 3: Advanced Collision Check (SAT)
            # Useful for curves where dot product lane check fails
            if not self.check_collision_trajectory(my_car, other, scan_range):
                 # If SAT says no collision, trust it (unless very close)
                 if dist > self.base_gap_m:
                     continue

            # Found a candidate
            if longitudinal_dist < min_dist:
                min_dist = longitudinal_dist
                critical_vehicle = other

        # 3. Apply Control Logic if Vehicle Found
        if critical_vehicle:
            # Calculate Proportional Factor
            # 1.0 = Far away (Safe), 0.0 = At Minimum Distance
            
            # Effective distance window
            # [min_standstill] <-------- dist --------> [required_gap]
            
            if min_dist <= self.min_standstill_distance_m:
                # Too close! Stop.
                return 0.0
            
            elif min_dist < required_gap:
                # In the "Braking Zone" -> Proportional Fade
                
                # Normalize distance 0..1 in the window
                window = required_gap - self.min_standstill_distance_m
                alpha = (min_dist - self.min_standstill_distance_m) / window
                alpha = np.clip(alpha, 0.0, 1.0)
                
                # Target speed blends between 0 and Lead Vehicle Speed
                # We want to match their speed, but if we are closing in, we must go SLOWER than them.
                
                lead_speed = critical_vehicle.speed_ms
                
                # If we are far in the window, match speed.
                # If we are close, go to 0.
                
                # Smooth blend
                safe_speed = lead_speed * alpha
                
                # But don't exceed our original desire
                target_speed = min(target_speed, safe_speed)
                
                # Ensure we don't just stop dead if they are moving fast and we have space
                # If alpha is 0.5 (halfway), and they are doing 20m/s, we do 10m/s. This closes the gap.
                # Eventually alpha becomes stable where V_us == V_them.
                
            else:
                # Just outside gap, maybe match speed to be polite?
                pass

        # 4. Cornering "Panic" Rule (Cyberpunk Logic)
        # If we are steering hard AND there are cars nearby (even in other lanes), slow down.
        # This prevents swinging tail into traffic.
        if abs(my_car.steer_angle) > self.cornering_steer_threshold_rad:
             # Check if ANY car is somewhat close (even if ignored by lane filter above)
             if len(vehicles_in_range) > 0:
                 target_speed = min(target_speed, self.cornering_slowdown_speed_ms)

        return max(0.0, target_speed)

    def get_vehicles_in_range(self, my_car, other_cars, range_val):
        result = []
        my_pos = np.array([my_car.x_m, my_car.y_m])
        for other in other_cars:
            if other.id == my_car.id: continue
            
            dist = np.linalg.norm(np.array([other.x_m, other.y_m]) - my_pos)
            if dist < range_val:
                result.append(other)
        return result

    def check_collision_trajectory(self, my_car, other_car, range_val):
        """
        Uses Separating Axis Theorem (SAT) on projected trajectories 
        to detect future overlaps on curves.
        """
        # Project My Box forward
        # Length extends by Lookahead Distance
        rect_a = self.get_oriented_box(my_car, length_extension=range_val)
        
        # Project Other Box forward (Assume constant velocity)
        # Prediction time approx range/speed
        dist_ext = other_car.speed_ms * 2.0  # Constant 2s lookahead
        rect_b = self.get_oriented_box(other_car, length_extension=dist_ext)
        
        return self.sat_collision(rect_a, rect_b)

    def get_oriented_box(self, car, length_extension=0.0):
        """
        Returns 4 corners of the car's bounding box in World Coordinates.
        Extends the 'Front' of the box by length_extension.
        """
        cx, cy, theta = car.x_m, car.y_m, car.angle
        
        # Dimensions from config
        w = CAR_WIDTH_M / 2.0
        l_front = (CAR_LENGTH_M / 2.0) + length_extension
        l_back = -(CAR_LENGTH_M / 2.0)
        
        # Local Corners (X=Forward, Y=Left)
        # FL, FR, BR, BL
        corners_local = [
            (l_front, w),   # Front-Left
            (l_front, -w),  # Front-Right
            (l_back, -w),   # Back-Right
            (l_back, w)     # Back-Left
        ]
        
        # Rotate and Translate to World
        c_cos = math.cos(theta)
        c_sin = math.sin(theta)
        
        world_corners = []
        for (lx, ly) in corners_local:
            # Rotate
            wx = lx * c_cos - ly * c_sin
            wy = lx * c_sin + ly * c_cos
            # Translate
            world_corners.append(np.array([cx + wx, cy + wy]))
            
        return world_corners

    def sat_collision(self, rect_a, rect_b):
        """Standard SAT intersection test."""
        polygons = [rect_a, rect_b]
        for polygon in polygons:
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                
                if np.linalg.norm(normal) == 0: continue
                normal = normal / np.linalg.norm(normal)
                
                min_a, max_a = self.project_polygon(rect_a, normal)
                min_b, max_b = self.project_polygon(rect_b, normal)
                
                if max_a < min_b or max_b < min_a:
                    return False
        return True

    def project_polygon(self, polygon, axis):
        min_proj = float('inf')
        max_proj = float('-inf')
        for p in polygon:
            proj = np.dot(p, axis)
            min_proj = min(min_proj, proj)
            max_proj = max(max_proj, proj)
        return min_proj, max_proj
