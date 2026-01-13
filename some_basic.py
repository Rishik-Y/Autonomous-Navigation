


import pybullet as p
import pybullet_data
import time
import os
import math

def load_road(urdf_path):
    return p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

def load_truck(urdf_path, start_pos, start_ori_euler=[0, 0, 0]):
    quat = p.getQuaternionFromEuler(start_ori_euler)
    return p.loadURDF(urdf_path, basePosition=start_pos, baseOrientation=quat, useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE)

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    road = load_road("road_layout.urdf")
    truck = load_truck("truck.urdf", start_pos=[-10, 0, 0.2])

    path_length = 80
    steps = 240 * 10  # ~10 seconds at 240 Hz
    for i in range(steps):
        t = i / steps
        x = -10 + path_length * t
        y = 0
        yaw = 0  # adjust yaw based on curve sections if needed
        pos = [x, y, 0.2]
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(truck, pos, quat)
        p.stepSimulation()
        time.sleep(1/240)

    p.disconnect()

if __name__ == "__main__":
    main()
