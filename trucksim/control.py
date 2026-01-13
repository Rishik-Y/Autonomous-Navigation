# trucksim/control.py

from . import config as C
from .physics import resist_forces, traction_force_from_power
import math


def controller(dt, s, v, mass_kg, target_s, vcap_ms, ctrl):
    """
    Controller function to compute throttle and brake commands for a truck agent.

    Parameters:
    - dt: timestep in seconds
    - s: current position along path (m)
    - v: current velocity (m/s)
    - mass_kg: mass of the truck in kg
    - target_s: target position along path (m)
    - vcap_ms: speed cap in m/s
    - ctrl: controller state object holding previous command info

    Returns:
    - throttle: throttle command (0 to 1)
    - brake: brake command (0 to 1)
    """

    # Remaining distance to stop (always positive)
    dist = max(0.0, abs(target_s - s))

    # Braking-distance cap: v_stop_cap = sqrt(2 * a_comf * dist)
    v_stop_cap = math.sqrt(max(0.0, 2.0 * C.A_BRAKE_COMF * dist))

    # Reference speed: cruise cap limited by braking-distance cap
    v_ref = min(vcap_ms, v_stop_cap)

    # Decide comfort vs emergency deceleration if current v exceeds comfort-capable stop speed
    v_margin = 0.5  # m/s
    need_emergency = v > (v_stop_cap + v_margin)
    a_brake_limit = C.A_BRAKE_MAX if need_emergency else C.A_BRAKE_COMF

    # First-order speed tracking for smoothness (simpler and stable near v=0)
    tau = 0.6  # response time in seconds
    a_des = (v_ref - v) / max(0.1, tau)

    # Clamp acceleration command considering braking limits and max acceleration from config
    # Also enforce min acceleration limit when accelerating positively (unloaded)
    if a_des > 0:
        # Clamp between min and max acceleration
        a_des = max(C.A_ACCEL_MIN, min(C.A_ACCEL_MAX, a_des))
    else:
        # Clamp braking deceleration
        a_des = max(-a_brake_limit, min(0.0, a_des))

    # Jerk limit (rate of change of acceleration)
    da_max = C.JERK_LIMIT * dt

    # Apply jerk limit smoothing between previous command and current desired command
    a_cmd = max(ctrl.a_cmd_prev - da_max, min(a_des, ctrl.a_cmd_prev + da_max))
    ctrl.a_cmd_prev = a_cmd

    # Compute throttle and brake commands based on acceleration command
    if a_cmd >= 0.0:
        throttle = a_cmd / C.A_ACCEL_MAX
        brake = 0.0
    else:
        throttle = 0.0
        # normalize brake command (negative acceleration)
        brake = -a_cmd / C.A_BRAKE_MAX

    # Clamp throttle and brake to valid range [0, 1]
    throttle = max(0.0, min(throttle, 1.0))
    brake = max(0.0, min(brake, 1.0))

    return throttle, brake
