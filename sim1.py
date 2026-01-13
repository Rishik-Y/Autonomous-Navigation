import pybullet as p
import pybullet_data
import time
import numpy as np
import csv

# Connect to PyBullet in DIRECT mode (no GUI)
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)

# Load plane (ground)
plane_id = p.loadURDF("road_layout.urdf")

# Load your simple truck URDF at start position (near point A at x = -20)
truck_start_pos = [-20, 0, 0.5]  # Slight elevation for ground clearance
truck_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
truck_id = p.loadURDF("road_layout.urdf", truck_start_pos, truck_start_orientation)

# Get joint indices by name
num_joints = p.getNumJoints(truck_id)
joint_name_to_id = {}
for i in range(num_joints):
    info = p.getJointInfo(truck_id, i)
    joint_name = info[1].decode('utf-8')
    joint_name_to_id[joint_name] = i

left_wheel_joints = [joint_name_to_id['front_left_joint'], joint_name_to_id['rear_left_joint']]
right_wheel_joints = [joint_name_to_id['front_right_joint'], joint_name_to_id['rear_right_joint']]

# Simulation parameters
dt = 0.1
max_speed = 11.11  # 40 km/h in m/s
circle_radius = 10.0
circle_angle = np.pi
straight_length = 20.0

# Prepare logging
log_filename = "truck_log.csv"
log_file = open(log_filename, "w", newline="")
logger = csv.writer(log_file)
logger.writerow(['time', 'left_speed', 'right_speed', 'straight_speed', 'pos_x', 'pos_y'])

# Helper function to set wheel speeds (same speed per side)
def set_wheel_speeds(left_speed, right_speed):
    for j in left_wheel_joints:
        p.setJointMotorControl2(truck_id, j, p.VELOCITY_CONTROL, targetVelocity=left_speed)
    for j in right_wheel_joints:
        p.setJointMotorControl2(truck_id, j, p.VELOCITY_CONTROL, targetVelocity=right_speed)

# Move straight from A to circle start (x: -20 to 0)
t = 0.0
while True:
    pos, _ = p.getBasePositionAndOrientation(truck_id)
    pos_x, pos_y, _ = pos
    if pos_x >= 0:
        break
    set_wheel_speeds(max_speed, max_speed)
    p.stepSimulation()
    logger.writerow([t, max_speed, max_speed, max_speed, pos_x, pos_y])
    t += dt
    time.sleep(dt)

# Move through semicircle (counter-clockwise)
arc_length = circle_radius * circle_angle
traveled_arc = 0.0

# Initial angle at circle start (point (10,0)), at 0 radians on circle in XY plane
# We will track truck position manually for circle part
theta = 0.0

while traveled_arc < arc_length:
    left_speed = max_speed * (circle_radius - 1.25) / circle_radius
    right_speed = max_speed * (circle_radius + 1.25) / circle_radius
    straight_speed = (left_speed + right_speed) / 2
    set_wheel_speeds(left_speed, right_speed)
    
    # Update position of truck base manually on circle path for logging accuracy
    theta += straight_speed * dt / circle_radius
    pos_x = circle_radius * np.cos(np.pi/2 - theta)  # Shift angle so circle is centered at origin shifted - starting at (10,0)
    pos_y = circle_radius * np.sin(np.pi/2 - theta)

    p.resetBasePositionAndOrientation(truck_id, [pos_x, pos_y, 0.5], p.getQuaternionFromEuler([0, 0, -theta]))
    
    p.stepSimulation()
    logger.writerow([t, left_speed, right_speed, straight_speed, pos_x, pos_y])
    t += dt
    traveled_arc += straight_speed * dt
    time.sleep(dt)

# Move straight from circle end to B (x: 0 to 20)
while True:
    pos, _ = p.getBasePositionAndOrientation(truck_id)
    pos_x, pos_y, _ = pos
    if pos_x >= 20:
        break
    set_wheel_speeds(max_speed, max_speed)
    p.stepSimulation()
    logger.writerow([t, max_speed, max_speed, max_speed, pos_x, pos_y])
    t += dt
    time.sleep(dt)

# Stop motors
set_wheel_speeds(0, 0)

log_file.close()
p.disconnect()

print(f"Simulation finished. Log saved as {log_filename}")
