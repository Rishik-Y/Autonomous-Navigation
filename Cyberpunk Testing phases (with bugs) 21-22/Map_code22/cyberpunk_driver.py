import numpy as np
import math

class CyberpunkDriver:
    """
    A port of the Cyberpunk 2077 'Auto Drive' mod logic (SyncSpeed class).
    Implements Adaptive Cruise Control and Collision Avoidance rules.
    """
    def __init__(self):
        # Settings from auto_drive_settings.reds
        self.syncMaxSpeed = True
        self.syncMaxSpeedVehicleInFrontBaseDistance = 10.0 # Reduced from 20.0 to reduce ghost braking
        self.syncMaxSpeedVehicleInFrontDistanceSpeedFactor = 1.0
        self.syncMaxSpeedVehicleInFrontDistanceMassFactor = 1.0
        self.syncMaxSpeedVehicleInFrontDistanceBrakingTorqueFactor = 1.0
        self.syncMaxSpeedDistanceToStopApproaching = 6.0 # Reduced from 15.0 to prevent leaving huge gaps
        self.syncMaxSpeedApproachSpeed = 5.0
        
        self.forceBrakesBaseTime = 0.5
        self.forceBrakesSpeedFactor = 1.0
        self.forceBrakesMassFactor = 1.0
        self.forceBrakesBrakingTorqueFactor = 1.0

        # Internal state
        self.prev_transform = None

    def calculate_target_speed(self, my_car, other_cars, desired_speed):
        """
        Main entry point. Mimics SyncSpeed.GetSpeed().
        """
        if not self.syncMaxSpeed:
            return desired_speed

        # Calculate Braking Torque (Simplified approximation)
        # In CP2077: (Front + Back) / 2.
        # Here we assume a standard truck value if not present.
        braking_torque = getattr(my_car, 'braking_torque', 4000.0) 
        
        # Calculate Factors
        speed_factor = my_car.speed_ms * self.syncMaxSpeedVehicleInFrontDistanceSpeedFactor
        mass_factor = max((my_car.current_mass_kg / 1000.0) * self.syncMaxSpeedVehicleInFrontDistanceMassFactor, 1.0)
        
        braking_factor = self.syncMaxSpeedVehicleInFrontDistanceBrakingTorqueFactor
        if braking_torque > 0.0:
            # CP2077 logic: brakingFactor * ClampF(400.0 / brakingTorque, 0.70, 2.0)
            # Since trucks have huge torque, 400/4000 = 0.1. Clamped to 0.7.
            val = 400.0 / braking_torque
            val = max(0.7, min(val, 2.0))
            braking_factor = braking_factor * val
        else:
             braking_factor = 1.5

        # Calculate dynamic range (safety bubble)
        range_val = self.syncMaxSpeedVehicleInFrontBaseDistance + (speed_factor * mass_factor * braking_factor)
        
        # Filter vehicles in range
        # CP2077 adds +10.0 buffer to finding range
        vehicles_in_range = self.get_vehicles_in_range(my_car, other_cars, range_val + 10.0)

        max_speed = desired_speed

        for veh in vehicles_in_range:
            sync_speed_candidate = max_speed
            should_sync, calculated_speed = self.calc_sync_speed(my_car, veh, range_val, sync_speed_candidate)
            
            if should_sync:
                max_speed = min(max_speed, calculated_speed)

        return max(0.0, max_speed)

    def get_vehicles_in_range(self, my_car, other_cars, range_val):
        result = []
        my_pos = np.array([my_car.x_m, my_car.y_m])
        for other in other_cars:
            if other.id == my_car.id:
                continue
            dist = np.linalg.norm(np.array([other.x_m, other.y_m]) - my_pos)
            if dist < range_val:
                result.append(other)
        return result

    def calc_sync_speed(self, my_car, other_car, range_val, current_target_speed):
        """
        Mimics SyncSpeed.CalcSyncSpeed
        Returns: (should_sync: bool, new_speed: float)
        """
        # Transform Logic
        # We need local coordinates of other_car relative to my_car
        
        # My Transform
        cx, cy, ctheta = my_car.x_m, my_car.y_m, my_car.angle
        
        # Other Pos
        ox, oy = other_car.x_m, other_car.y_m
        
        # World to Local
        dx = ox - cx
        dy = oy - cy
        
        # Rotate by -theta
        c_cos = math.cos(-ctheta)
        c_sin = math.sin(-ctheta)
        
        local_x = dx * c_cos - dy * c_sin # Right
        local_y = dx * c_sin + dy * c_cos # Forward (standard convention: X=forward? No, usually Y=forward in games, but CP2077 REDScript might differ)
        
        # CP2077 Vector4: X, Y, Z, W. 
        # Typically in game engines: Y is forward or Z is forward.
        # Let's verify REDScript: 
        # "Transform.GetForward" usually returns the forward vector.
        # In the script: 
        # localPos = WorldTransform.TransformInvPoint(this.GetVehicle().GetWorldTransform(), vehicle.GetWorldPosition());
        # Checks: "localPos.Y < 0.0" -> Returns false. This implies Y is Forward in local space?
        # WAIT. "localPos.Y < 0.0" means "Behind me". 
        # Let's assume Standard Engineering: X=Forward, Y=Left.
        # But REDengine might be Y=Forward. 
        # Let's look at `CollisionTest2D`: 
        # rectA points: (l, b), (l, f), (r, f), (r, b). 
        # l = -1.5, r = 1.5. f = 2.5+forward. b = -2.5-backward.
        # Vectors are (l, b, 0, 0). 
        # If l/r are X/Y coordinates... usually L/R is Width (Y or X) and F/B is Length (X or Y).
        # Given -1.5 to 1.5, that looks like Width.
        # Given -2.5 to 2.5, that looks like Length.
        # If (l, f) is a point, and l=-1.5, f=2.5...
        # If Y was forward, Width would be X. 
        # Let's assume Y is Forward, X is Right. 
        # Then (X, Y): (-1.5, 2.5) is Front-Left?
        # SyncSpeed.InTraffic check: "localPos.Y < 0.0" => Behind. Yes, Y is Forward.
        
        # My coordinate system in Simulation (from car.py):
        # x += speed * cos(angle)
        # y += speed * sin(angle)
        # So Angle 0 is +X direction. +X is Forward.
        # So I need to map my (X=Forward, Y=Left) to REDScript (Y=Forward, X=Right).
        
        # My Local Space (X=Forward, Y=Left):
        # Local X = Dot(Diff, Heading)
        # Local Y = Dot(Diff, Normal)
        
        heading = np.array([math.cos(ctheta), math.sin(ctheta)])
        normal = np.array([-math.sin(ctheta), math.cos(ctheta)]) # Left normal
        
        diff = np.array([dx, dy])
        
        my_local_forward = np.dot(diff, heading) # My Forward distance (My X)
        my_local_right = -np.dot(diff, normal)   # My Right distance (My -Y) -> CP2077 X?
        
        # CP2077 Local Space: Y is Forward, X is Right.
        cp_local_y = my_local_forward
        cp_local_x = my_local_right
        
        # 1. Basic Filters
        # if localPos.Y < 0.0 (Behind) -> Return False
        if cp_local_y < 0.0:
            return False, current_target_speed
            
        dist_sq = dx*dx + dy*dy
        dist = math.sqrt(dist_sq)
        
        if dist > range_val:
            return False, current_target_speed

        # 2. Collision Test
        collided = self.velocity_collision_test_2d(my_car, other_car, range_val)
        
        final_speed = current_target_speed
        
        if collided:
            # Check orientation (Oncoming?)
            # vehicleForwardVelocity = Dot(MyForward, OtherVelocity)
            other_vel_vec = np.array([
                other_car.speed_ms * math.cos(other_car.angle),
                other_car.speed_ms * math.sin(other_car.angle)
            ])
            
            fwd_dot_vel = np.dot(heading, other_vel_vec)
            
            if fwd_dot_vel < -10.0:
                # Oncoming, ignore (Cyberpunk logic)
                return False, current_target_speed
            else:
                # Sync Speed
                # speed = ClampF(Floor(vehicleForwardVelocity, 0.5) - 0.5, approachSpeed, speed)
                snapped_speed = math.floor(fwd_dot_vel / 0.5) * 0.5 - 0.5
                final_speed = min(final_speed, max(self.syncMaxSpeedApproachSpeed, snapped_speed))
            
            # Stop if too close
            # if distance < stopApproachingDistance ...
            
            # IsInSameLane logic
            # Abs(localPos.X) < 3.5 && Dot(MyFwd, OtherFwd) > 0.75
            other_heading = np.array([math.cos(other_car.angle), math.sin(other_car.angle)])
            dot_headings = np.dot(heading, other_heading)
            
            is_same_lane = (abs(cp_local_x) < 3.5) and (dot_headings > 0.75)
            
            if dist < self.syncMaxSpeedDistanceToStopApproaching:
                 # "stop approaching if in same lane OR other is moving"
                 # Only stop if we are dangerously close and in the same lane
                 if is_same_lane:
                     final_speed = 0.0
            
            return True, final_speed

        # 3. Cornering Check (The "Panic" Rule)
        # if MySpeed > approachSpeed && Abs(turning) > 0.1 && distance < 20.0 && angle > 0.70
        
        # Turning: angular velocity? Or lateral slip?
        # REDScript: AngleDotXY(LinearVelocity, Orientation.Right)
        # Basically: Dot(VelocityDir, RightDir). This is ~Sin(SlipAngle).
        # Wait, usually "turning" implies angular velocity. 
        # But here it compares Linear Velocity direction vs Right Vector.
        # This detects SLIP (Drifting) or just Lateral Movement?
        # If car is moving Forward, Vel is aligned with Fwd. Dot(Fwd, Right) = 0.
        # If car is drifting, Vel has Right component. 
        
        # Let's calculate simple angular velocity check instead as a proxy, 
        # OR stick to the script literal:
        my_vel_vec = np.array([
            my_car.speed_ms * math.cos(my_car.angle),
            my_car.speed_ms * math.sin(my_car.angle)
        ])
        my_right_vec = np.array([
            math.cos(my_car.angle - math.pi/2), # Right is -90 deg from Fwd
            math.sin(my_car.angle - math.pi/2)
        ])
        
        # To match CP2077 "AngleDotXY", we normalize vectors first.
        if my_car.speed_ms > 0.1:
            v_norm = my_vel_vec / my_car.speed_ms
            turning_val = np.dot(v_norm, my_right_vec)
        else:
            turning_val = 0.0

        # "angle" in script: Dot(Forward, DiffToOther) (Cosine of angle to target)
        angle_to_other = np.dot(heading, diff / (dist + 1e-6))
        
        if (my_car.speed_ms > self.syncMaxSpeedApproachSpeed and 
            abs(turning_val) > 0.1 and 
            dist < 20.0 and 
            angle_to_other > 0.70):
            
            # Slow down
            final_speed = self.syncMaxSpeedApproachSpeed
            return True, final_speed
            
        return False, current_target_speed

    def velocity_collision_test_2d(self, my_car, other_car, range_val):
        """
        Projects rectangles based on speed and checks overlap.
        """
        # My Rect
        # forward = range, backward = -speed/6.0
        fwd_extra = range_val
        bwd_extra = -my_car.speed_ms / 6.0
        rect_a = self.get_2d_rect(my_car, fwd_extra, bwd_extra)
        
        # Other Rect
        # forward = speed + 5.0, backward = Max(5.0 - speed/2.0, 0)
        o_fwd = other_car.speed_ms + 5.0
        o_bwd = max(5.0 - other_car.speed_ms / 2.0, 0.0)
        rect_b = self.get_2d_rect(other_car, o_fwd, o_bwd)
        
        return self.sat_collision(rect_a, rect_b)

    def get_2d_rect(self, car, forward_ext, backward_ext):
        """
        Returns 4 corners of the collision box.
        """
        # Car dims (Approximate standard truck/car)
        # REDScript uses hardcoded: l=-1.5, r=1.5 (Width 3m), f=2.5, b=-2.5 (Length 5m)
        w_half = 1.5
        l_front = 2.5
        l_back = -2.5
        
        f = l_front + forward_ext
        b = l_back - backward_ext
        l = -w_half
        r = w_half
        
        # Local points (Y=Forward, X=Right in CP2077 logic)
        # But we must respect the transform logic used in SAT.
        # Let's stick to our Simulation Space: X=Forward, Y=Left.
        # Then width is Y, Length is X.
        
        # Corners in Car Local Space (X=Forward)
        # FL, FR, BR, BL
        # FL: x=f, y=l (Left is positive Y)
        # FR: x=f, y=r (Right is negative Y)
        # ...
        
        # Wait, REDScript (l, b), (l, f), (r, f), (r, b).
        # If Y=Fwd, X=Right.
        # (l, b) = (Left, Back) ? If l=-1.5, r=1.5. 
        # Usually Left is -X, Right is +X. 
        # Let's assume CP2077: X=Right, Y=Forward.
        # l=-1.5 (Left), r=1.5 (Right).
        # f=2.5 (Front), b=-2.5 (Back).
        
        # We need to map this to Global World Space.
        cx, cy, ctheta = car.x_m, car.y_m, car.angle
        cos_t = math.cos(ctheta)
        sin_t = math.sin(ctheta)
        
        # My Local: X=Fwd, Y=Left.
        # CP (X=Right, Y=Fwd) -> My (Y=-Right, X=Fwd).
        # So CP_X maps to -My_Y. CP_Y maps to My_X.
        
        # Points in CP Local: (x,y)
        points_local_cp = [
            (-w_half, b), # Left-Back
            (-w_half, f), # Left-Front
            (w_half, f),  # Right-Front
            (w_half, b)   # Right-Back
        ]
        
        world_points = []
        for (lx, ly) in points_local_cp:
            # Map CP Local (Right, Fwd) to World
            # Global = Center + Fwd*ly + Right*lx
            
            # Fwd Vector = (cos, sin)
            # Right Vector = (sin, -cos)  (Rotated -90 deg)
            
            gx = cx + (cos_t * ly) + (sin_t * lx)
            gy = cy + (sin_t * ly) + (-cos_t * lx)
            world_points.append(np.array([gx, gy]))
            
        return world_points

    def sat_collision(self, rect_a, rect_b):
        """
        Separating Axis Theorem for two convex polygons (rectangles).
        """
        polygons = [rect_a, rect_b]
        
        for polygon in polygons:
            # Check normals of this polygon
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % len(polygon)]
                
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]]) # Normal
                # Normalize? Not strictly needed for boolean overlap, but good for projection
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-6: continue
                normal /= norm_len
                
                # Project both polygons
                min_a, max_a = self.project_polygon(rect_a, normal)
                min_b, max_b = self.project_polygon(rect_b, normal)
                
                if max_a < min_b or max_b < min_a:
                    return False # Separating axis found
        
        return True # No separating axis found

    def project_polygon(self, polygon, axis):
        min_proj = float('inf')
        max_proj = float('-inf')
        for p in polygon:
            proj = np.dot(p, axis)
            min_proj = min(min_proj, proj)
            max_proj = max(max_proj, proj)
        return min_proj, max_proj
