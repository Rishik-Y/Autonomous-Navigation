#! /usr/bin/env python

from __future__ import annotations

import signal
import time

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import numpy.typing as npt
from cvxpy_mpc import MPC
from cvxpy_mpc.utils import (
    compute_path_from_wp,
    detect_obstacle_camera,
    ego_to_global,
    get_ref_trajectory,
)
from scipy.integrate import odeint

# Robot Starting position
SIM_START_X = 0.0
SIM_START_Y = 0.5
SIM_START_V = 0.0
SIM_START_H = 0.0


# Params
USE_OBS_AVOIDANCE = True

TARGET_VEL = 1.0  # m/s

OBS = (
    [
        (4.0, 2.2, 0.5),
        (12.5, 0.5, 0.35),
        (7.0, -5.5, 0.45),
    ]
    if USE_OBS_AVOIDANCE
    else []
)
SENSOR_MAX_RANGE = 4.0
SENSOR_FOV_DEG = 90.0


# Classes
class MPCSim:
    def __init__(self) -> None:
        # State of the robot [x,y,v, heading]
        self.state: npt.NDArray[np.float64] = np.array(
            [SIM_START_X, SIM_START_Y, SIM_START_V, SIM_START_H]
        )

        # helper variable to keep track of mpc output
        self.control: npt.NDArray[np.float64] = np.zeros(2)

        self.mpc: MPC = MPC("config/mpc.yaml")
        self.K: int = self.mpc.control_horizon
        self.detected_obs: tuple[float, float, float] | None = None

        # Path from waypoint interpolation
        self.path: npt.NDArray[np.float64] = compute_path_from_wp(
            [0, 3, 4, 6, 10, 12, 13, 13, 6, 1, 0],
            [0, 0, 2, 4, 3, 3, -1, -2, -6, -2, -2],
            0.05,
        )

        # Helper variables to keep track of the sim
        self.sim_time: float = 0.0
        self.x_history: list[float] = [SIM_START_X]
        self.y_history: list[float] = [SIM_START_Y]
        self.v_history: list[float] = [SIM_START_V]
        self.h_history: list[float] = [SIM_START_H]
        self.a_history: list[float] = [0.0]
        self.d_history: list[float] = [0.0]
        self.optimized_trajectory: npt.NDArray[np.float64] | None = None
        self.mpc_solve_time: float = 0.0

        # Persistent plot (no clf flickering)
        plt.style.use("ggplot")
        self.fig: plt.Figure = plt.figure()
        gs = plt.GridSpec(3, 3)

        self.ax_main = plt.subplot(gs[0:3, 0:2])
        self.ax_main.set_xlabel("map x")
        self.ax_main.set_ylabel("map y")
        self.ax_main.set_aspect("equal")
        self.ax_main.plot(
            self.path[0, :],
            self.path[1, :],
            c="tab:orange",
            marker=".",
            label="reference track",
        )
        x_pad, y_pad = 1.0, 1.0
        self.ax_main.set_xlim(
            self.path[0, :].min() - x_pad, self.path[0, :].max() + x_pad
        )
        self.ax_main.set_ylim(
            self.path[1, :].min() - y_pad, self.path[1, :].max() + y_pad
        )

        # Obstacle visualization
        self.obs_circles: list[plt.Circle] = []
        if USE_OBS_AVOIDANCE:
            for ox, oy, rad in OBS:
                c = plt.Circle(
                    (ox, oy),
                    rad,
                    color="red",
                    alpha=0.4,
                    label="obstacle",
                )
                self.ax_main.add_patch(c)
                self.obs_circles.append(c)

            # Sensor FOV wedge
            self.fov_patch = Wedge(
                (0, 0),
                SENSOR_MAX_RANGE,
                0,
                0,
                color="gold",
                alpha=0.15,
            )
            self.ax_main.add_patch(self.fov_patch)

        (self.traj_line,) = self.ax_main.plot(
            [],
            [],
            c="tab:blue",
            marker=".",
            alpha=0.5,
            label="vehicle trajectory",
        )
        (self.mpc_line,) = self.ax_main.plot(
            [],
            [],
            c="tab:green",
            marker="+",
            alpha=0.5,
            label="mpc opt trajectory",
        )
        self.mpc_line.set_visible(False)

        self.car_line: plt.Line2D | None = None

        # HUD overlay
        self.hud = self.ax_main.text(
            0.02,
            0.98,
            "",
            transform=self.ax_main.transAxes,
            va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Acceleration subplot
        self.ax_accel = plt.subplot(gs[0, 2])
        self.ax_accel.set_ylabel("a(t) [m/ss]")
        self.ax_accel.set_xlabel("t [s]")
        self.ax_accel.axhline(y=self.mpc.max_acc, c="gray", ls="--", lw=0.8)
        self.ax_accel.axhline(y=-self.mpc.max_acc, c="gray", ls="--", lw=0.8)
        self.ax_accel.set_ylim(
            -self.mpc.max_acc * 1.5, self.mpc.max_acc * 1.5
        )
        (self.accel_line,) = self.ax_accel.plot([], [], c="tab:orange")

        # Steering subplot
        self.ax_steer = plt.subplot(gs[1, 2])
        self.ax_steer.set_ylabel("gamma(t) [deg]")
        self.ax_steer.set_xlabel("t [s]")
        max_steer_deg = np.degrees(self.mpc.max_steer)
        self.ax_steer.axhline(y=max_steer_deg, c="gray", ls="--", lw=0.8)
        self.ax_steer.axhline(y=-max_steer_deg, c="gray", ls="--", lw=0.8)
        self.ax_steer.set_ylim(-max_steer_deg * 1.5, max_steer_deg * 1.5)
        (self.steer_line,) = self.ax_steer.plot([], [], c="tab:orange")

        # Velocity subplot
        self.ax_vel = plt.subplot(gs[2, 2])
        self.ax_vel.set_ylabel("v(t) [m/s]")
        self.ax_vel.set_xlabel("t [s]")
        self.ax_vel.axhline(y=TARGET_VEL, c="tab:orange", ls="--", label="target speed")
        self.ax_vel.set_ylim(0, self.mpc.max_speed * 1.2)
        (self.vel_line,) = self.ax_vel.plot([], [], c="tab:blue", label="vehicle speed")
        self.ax_vel.legend(loc="lower right")

        plt.tight_layout()
        plt.ion()
        plt.show()

    def run(self) -> None:
        self.plot_sim()
        try:
            while 1:
                if (
                    np.sqrt(
                        (self.state[0] - self.path[0, -1]) ** 2
                        + (self.state[1] - self.path[1, -1]) ** 2
                    )
                    < 0.5
                ):
                    print(
                        "Success! Goal Reached\nClose the plot window or press CTRL-C to exit."
                    )
                    while plt.fignum_exists(self.fig.number):
                        plt.pause(0.1)
                    return
                # External obstacle detection pipeline
                if USE_OBS_AVOIDANCE:
                    self.detected_obs = detect_obstacle_camera(
                        OBS,
                        self.state[0],
                        self.state[1],
                        self.state[3],
                        SENSOR_MAX_RANGE,
                        SENSOR_FOV_DEG,
                    )
                else:
                    self.detected_obs = None
                # Get Reference_traj -> inputs are in worldframe
                target = get_ref_trajectory(self.state, self.path, TARGET_VEL, self.mpc.control_horizon * self.mpc.dt, self.mpc.dt)

                # dynamycs w.r.t robot frame
                curr_state = np.array([0, 0, self.state[2], 0])

                # Transform global obstacle to ego frame
                if self.detected_obs is not None:
                    gx, gy, r = self.detected_obs
                    dx = gx - self.state[0]
                    dy = gy - self.state[1]
                    ct, st = np.cos(-self.state[3]), np.sin(-self.state[3])
                    obs_ego = (dx * ct - dy * st, dy * ct + dx * st, r)
                else:
                    obs_ego = None

                t0 = time.perf_counter()
                x_mpc, u_mpc = self.mpc.solve(
                    curr_state,
                    target,
                    verbose=False,
                    obstacle=obs_ego,
                )
                self.mpc_solve_time = time.perf_counter() - t0
                # only the first one is used to advance the simulation

                self.control[:] = [u_mpc[0, 0], u_mpc[1, 0]]

                # Convert MPC preview from ego->world BEFORE advancing state,
                # so it's anchored to the state it was computed for
                self.optimized_trajectory = ego_to_global(self.state, x_mpc)

                self.state = self.predict_next_state(
                    self.state, [self.control[0], self.control[1]], self.mpc.dt
                )

                self.sim_time += self.mpc.dt
                self.x_history.append(self.state[0])
                self.y_history.append(self.state[1])
                self.v_history.append(self.state[2])
                self.h_history.append(self.state[3])
                self.a_history.append(self.control[0])
                self.d_history.append(self.control[1])
                self.plot_sim()
        except KeyboardInterrupt:
            print("\nInterrupted by user (CTRL-C). Exiting...")

    def predict_next_state(
        self,
        state: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64] | list[float],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        L = self.mpc.wheelbase

        def kinematics_model(x, t, u):
            dxdt = x[2] * np.cos(x[3])
            dydt = x[2] * np.sin(x[3])
            dvdt = u[0]
            dthetadt = x[2] * np.tan(u[1]) / L
            dqdt = [dxdt, dydt, dvdt, dthetadt]
            return dqdt

        # solve ODE
        tspan = [0, dt]
        new_state = odeint(kinematics_model, state, tspan, args=(u[:],))[1]
        return new_state

    def plot_sim(self) -> None:
        # Title
        self.ax_main.set_title(
            f"MPC Simulation\nSimulation elapsed time {self.sim_time:.1f}s"
        )

        # Trajectory history
        self.traj_line.set_data(self.x_history, self.y_history)

        # MPC preview
        if self.optimized_trajectory is not None:
            self.mpc_line.set_data(
                self.optimized_trajectory[0, :], self.optimized_trajectory[1, :]
            )
            self.mpc_line.set_visible(True)
        else:
            self.mpc_line.set_visible(False)

        if self.car_line is not None:
            self.car_line.remove()
        self.car_line = plot_car(
            self.ax_main, self.x_history[-1], self.y_history[-1], self.h_history[-1]
        )

        # Sensor FOV wedge
        if USE_OBS_AVOIDANCE:
            half_fov = SENSOR_FOV_DEG / 2
            theta1 = np.degrees(self.h_history[-1]) - half_fov
            theta2 = np.degrees(self.h_history[-1]) + half_fov
            self.fov_patch.set_center((self.x_history[-1], self.y_history[-1]))
            self.fov_patch.set_theta1(theta1)
            self.fov_patch.set_theta2(theta2)

        # HUD
        goal_dist = np.sqrt(
            (self.state[0] - self.path[0, -1]) ** 2
            + (self.state[1] - self.path[1, -1]) ** 2
        )
        avoiding = (
            "YES"
            if self.detected_obs is not None
            else "no" if USE_OBS_AVOIDANCE else "off"
        )
        self.hud.set_text(
            f"v: {self.state[2]:.2f} m/s  |  goal: {goal_dist:.2f} m  |  avoid: {avoiding}  |  MPC: {self.mpc_solve_time*1000:.0f} ms"
        )

        # Subplot data: plot against time
        t = np.arange(len(self.a_history)) * self.mpc.dt
        self.accel_line.set_data(t, self.a_history)
        self.ax_accel.relim()
        self.ax_accel.autoscale_view(scalex=True, scaley=False)

        self.steer_line.set_data(t, np.degrees(self.d_history))
        self.ax_steer.relim()
        self.ax_steer.autoscale_view(scalex=True, scaley=False)

        self.vel_line.set_data(t, self.v_history)
        self.ax_vel.relim()
        self.ax_vel.autoscale_view(scalex=True, scaley=False)

        plt.draw()
        plt.pause(0.001)


def plot_car(ax: plt.Axes, x: float, y: float, yaw: float) -> plt.Line2D:
    CAR_LENGTH = 0.5
    CAR_WIDTH = 0.25
    CAR_OFFSET = CAR_LENGTH

    outline = np.array(
        [
            [
                -CAR_OFFSET,
                CAR_LENGTH - CAR_OFFSET,
                CAR_LENGTH - CAR_OFFSET,
                -CAR_OFFSET,
                -CAR_OFFSET,
            ],
            [
                CAR_WIDTH / 2,
                CAR_WIDTH / 2,
                -CAR_WIDTH / 2,
                -CAR_WIDTH / 2,
                CAR_WIDTH / 2,
            ],
        ]
    )

    Rotm = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    outline = (outline.T @ Rotm).T
    outline[0, :] += x
    outline[1, :] += y

    return ax.plot(outline[0, :].flatten(), outline[1, :].flatten(), "tab:blue")[0]


def do_sim() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    sim = MPCSim()
    try:
        sim.run()
    except Exception as e:
        sys.exit(e)


if __name__ == "__main__":
    do_sim()
