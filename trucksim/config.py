# Add near the top or under existing constants

# Logistics
TRUCK_CAPACITY_TON = 5.0
MINE_INITIAL_TON   = 10.0
LOAD_RATE_TPS      = 1.0      # tons per second (tune as desired)
UNLOAD_RATE_TPS    = 1.0

# Speed caps (km/h)
VCAP_EMPTY_KMH     = 65   # unchanged for empty trips
VCAP_LOADED_KMH    = 40.0     # per requirement for loaded trips

# Cycle timing and stop geometry
UTURN_TIME_S       = 2.0
STOP_ZONE_M        = 5.0      # stop this far from each end
STOP_EPS_M         = 0.5      # stop tolerance in meters

# Simulation parameters
ROAD_LEN_M      = 1_000.0
LANE_WIDTH_M    = 3.5
LANES           = 4
BASE_SCALE      = 10.0
TARGET_KMH      = 120.0
AUTO_MODE       = True
FOLLOW_TRUCK    = False

# Acceleration (m/s²)
A_ACCEL_MAX = 0.4          # Max acceleration, 0.1–0.4 m/s²
A_ACCEL_MIN = 0.1          # Min practical acceleration for heavy vehicles

# Truck physical parameters
MASS_KG         = 20_000.0
CD              = 0.8
FRONTAL_AREA    = 8.0
CRR             = 0.006
AIR_DENS        = 1.225
MU_TIRE         = 0.8
P_MAX_W         = 300_000.0
A_BRAKE_COMF    = 0.5
A_BRAKE_MAX     = 1

# Truck drawing parameters
TRUCK_LEN_M     = 8.0
TRUCK_WID_M     = 2.5

# Window and drawing
WIN_W           = 1000
WIN_H           = 800
FPS             = 60

# Controls
PAN_SPEED_MPS   = 80.0
PAN_SPEED_FAST  = 200.0
ZOOM_STEP       = 1.1

# Controller gains and limits
Kp          = 0.8
Ki          = 0.2
JERK_LIMIT  = 2.0  # m/s^3 (tune 1.5–2.5 for smoothness)
# A_BRAKE_COMF ~3 m/s^2; A_BRAKE_MAX ~4.5 m/s^2 already defined

# trucksim/config.py
# ...
DRIVE_SIDE      = "left"   # "left" or "right"; affects default lane choice
# ...
