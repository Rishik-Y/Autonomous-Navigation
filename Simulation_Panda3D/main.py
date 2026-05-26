import json
import math
import os
import random
import sys

import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from panda3d.core import BitMask32, CollisionHandlerQueue, CollisionNode, CollisionRay, CollisionTraverser, Point3, loadPrcFileData

from Map import map_loader as map_data
from Algorithm.planner_registry import load_local_planner, DEFAULT_GLOBAL_PLANNER, DEFAULT_LOCAL_PLANNER
from car import Car
from config import *
from dispatcher import Dispatcher
from graphics import SimulationGraphics
from tooltip_overlay import TooltipOverlay, build_entity_tooltip, nearest_zone
from utils import KalmanFilter, Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_MAP_PANDA_DIR = os.path.join(_REPO_ROOT, "MAP_Panda3D")
if _MAP_PANDA_DIR not in sys.path:
    sys.path.insert(0, _MAP_PANDA_DIR)

from panda_common import CameraController, Picker  # noqa: E402
from panda_elevation import Heightmap, TerrainMesh  # noqa: E402
from panda_common import SceneRenderer  # noqa: E402


def _load_json_file(config_file, label):
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {label}: {e}")
    return None


def load_mine_config():
    config = {"truck_count": 5, "coal_capacities": {}}
    config_file = map_data.resolve_saved_map_path("mine_config.json")
    loaded = _load_json_file(config_file, "mine config")
    if loaded is not None:
        config["truck_count"] = loaded.get("truck_count", 5)
        config["coal_capacities"] = loaded.get("coal_capacities", {})
        return config

    fallback = os.path.join(_THIS_DIR, "truck.txt")
    if os.path.exists(fallback):
        with open(fallback, "r", encoding="utf-8") as f:
            try:
                config["truck_count"] = int(f.read().strip())
            except ValueError:
                pass
    return config


def load_algorithm_config():
    config = {
        "global_planner": DEFAULT_GLOBAL_PLANNER,
        "local_planner": DEFAULT_LOCAL_PLANNER,
    }
    config_file = os.path.join(_THIS_DIR, "algorithm_config.json")
    loaded = _load_json_file(config_file, "algorithm config")
    if loaded is not None:
        config["global_planner"] = loaded.get("global_planner", DEFAULT_GLOBAL_PLANNER)
        config["local_planner"] = loaded.get("local_planner", DEFAULT_LOCAL_PLANNER)
    return config


def get_path_from_nodes(route_node_names, waypoints_map):
    final_waypoints = []
    if not route_node_names:
        return []

    for i in range(len(route_node_names) - 1):
        seg_start, seg_end = route_node_names[i], route_node_names[i + 1]
        found_segment = False
        for chain_tuple, waypoints in waypoints_map.items():
            try:
                idx = chain_tuple.index(seg_start)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == seg_end:
                    start_wp_idx = idx * POINTS_PER_SEGMENT
                    end_wp_idx = (idx + 1) * POINTS_PER_SEGMENT
                    final_waypoints.extend(waypoints[start_wp_idx:end_wp_idx])
                    found_segment = True
                    break

                idx = chain_tuple.index(seg_end)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == seg_start:
                    start_wp_idx = idx * POINTS_PER_SEGMENT
                    end_wp_idx = (idx + 1) * POINTS_PER_SEGMENT
                    segment = waypoints[start_wp_idx:end_wp_idx + 1]
                    final_waypoints.extend(segment[::-1][:-1])
                    found_segment = True
                    break
            except ValueError:
                continue
        if not found_segment:
            pass

    if final_waypoints and route_node_names[-1] in map_data.NODES:
        final_waypoints.append(map_data.NODES[route_node_names[-1]])
    return final_waypoints


class PandaSimulationApp(ShowBase):
    def __init__(self):
        loadPrcFileData("", "window-title Autonomous Navigation Simulation Panda3D")
        loadPrcFileData("", f"win-size {SIM_WINDOW_W} {SIM_WINDOW_H}")
        super().__init__()
        self.disableMouse()
        self.render.setPythonTag("loader", self.loader)

        self.mine_config = load_mine_config()
        self.algorithm_config = load_algorithm_config()

        self.heightmap = Heightmap()
        self.heightmap.load_json()
        self.terrain = TerrainMesh(self.heightmap)
        self.renderer = SceneRenderer(self, self.heightmap, self.terrain)
        self.graphics = SimulationGraphics(self.renderer)
        self.graphics.draw_static_scene()

        cx = self.heightmap.origin_x + self.heightmap.cols * self.heightmap.cell_size * 0.5
        cy = self.heightmap.origin_y + self.heightmap.rows * self.heightmap.cell_size * 0.5
        cz = self.heightmap.get_height_at_world(cx, cy)
        self.camera_controller = CameraController(self, center=Point3(cx, cy, cz), dist=1400)
        self._bind_wasd_camera()

        self.picker = Picker(self, heightmap=self.heightmap, terrain_np_getter=self.renderer.get_terrain_np)
        self._init_truck_picker()

        self.tooltip_overlay = TooltipOverlay(self)
        self.hud = OnscreenText(
            text="",
            pos=(-1.3, 0.92),
            scale=0.045,
            fg=(0, 0, 0, 1),
            mayChange=True,
            align=0,
            parent=self.aspect2d,
        )

        self.paused = False
        self.sim_speed = 1.0
        self.selected_car_idx = 0

        self._load_sim_data()
        self._spawn_cars()

        self.mpc_timer = 0.0
        self.MPC_INTERVAL = 0.1
        self.traffic_update_timer = 0.0
        self.TRAFFIC_UPDATE_INTERVAL = 1.0
        self.global_opt_timer = 0.0
        self.GLOBAL_OPT_INTERVAL = 30.0

        self.accept("escape", self._exit)
        self.accept("space", self._toggle_pause)
        self.accept("tab", self._next_truck)
        self.accept("shift-1", self._set_speed, [1.0])
        self.accept("shift-2", self._set_speed, [2.0])
        self.accept("shift-3", self._set_speed, [3.0])
        self.accept("shift-4", self._set_speed, [4.0])
        self.accept("shift-5", self._set_speed, [5.0])
        self.accept("shift-0", self._set_speed, [0.5])

        self.taskMgr.add(self.sim_tick, "sim_tick")

    def _bind_wasd_camera(self):
        self.accept("w", self.camera_controller._set_key, ["arrow_up", True])
        self.accept("w-up", self.camera_controller._set_key, ["arrow_up", False])
        self.accept("s", self.camera_controller._set_key, ["arrow_down", True])
        self.accept("s-up", self.camera_controller._set_key, ["arrow_down", False])
        self.accept("a", self.camera_controller._set_key, ["arrow_left", True])
        self.accept("a-up", self.camera_controller._set_key, ["arrow_left", False])
        self.accept("d", self.camera_controller._set_key, ["arrow_right", True])
        self.accept("d-up", self.camera_controller._set_key, ["arrow_right", False])

    def _init_truck_picker(self):
        self.truck_traverser = CollisionTraverser()
        self.truck_queue = CollisionHandlerQueue()

        cnode = CollisionNode("truck_picker_ray")
        cnode.setFromCollideMask(BitMask32.bit(TRUCK_COLLIDE_MASK_BIT))
        cnode.setIntoCollideMask(BitMask32.allOff())
        self.truck_ray = CollisionRay()
        cnode.addSolid(self.truck_ray)
        self.truck_picker_np = self.camera.attachNewNode(cnode)
        self.truck_traverser.addCollider(self.truck_picker_np, self.truck_queue)

    def _load_sim_data(self):
        self.waypoints_map = map_data.load_pickle("waypoints.pkl")
        cache_data = map_data.load_pickle("map_cache.pkl")
        self.road_graph = cache_data.get("road_graph", {})
        self.route_cache = cache_data.get("route_cache", {})

        all_terminals = set(map_data.LOAD_ZONES + map_data.DUMP_ZONES)
        incoming_map = {}
        outgoing_counts = {}
        for start_node, edges in self.road_graph.items():
            outgoing_counts[start_node] = len(edges)
            for target, _ in edges:
                incoming_map.setdefault(target, []).append(start_node)

        def trace_back_and_patch(current_node, visited):
            if current_node in visited:
                return
            visited.add(current_node)
            parents = incoming_map.get(current_node, [])
            if not parents:
                return
            parent = parents[0]
            if current_node not in self.road_graph:
                self.road_graph[current_node] = []
            has_edge = any(target == parent for target, _ in self.road_graph[current_node])
            if not has_edge:
                p1 = map_data.NODES[current_node]
                p2 = map_data.NODES[parent]
                dist = float(np.linalg.norm(p1 - p2))
                self.road_graph[current_node].append((parent, dist))

            parent_out_degree = outgoing_counts.get(parent, 0)
            parent_in_degree = len(incoming_map.get(parent, []))
            is_hub = "hub" in parent or (parent_in_degree > 1 and parent_out_degree > 1)
            if not is_hub:
                trace_back_and_patch(parent, visited)

        for zone in all_terminals:
            trace_back_and_patch(zone, set())

        planner_name = self.algorithm_config["local_planner"]
        try:
            self.local_planner = load_local_planner(planner_name)
        except Exception:
            self.local_planner = load_local_planner(DEFAULT_LOCAL_PLANNER)

        self.dispatcher = Dispatcher(
            self.road_graph,
            coal_capacities=self.mine_config["coal_capacities"],
            global_planner_name=self.algorithm_config["global_planner"],
        )

    def _spawn_cars(self):
        self.cars = []
        self.kfs = []
        truck_count = self.mine_config["truck_count"]

        start_nodes = list(map_data.DUMP_ZONES)
        random.shuffle(start_nodes)

        for i in range(truck_count):
            start_node = start_nodes[i % len(start_nodes)]
            car = Car(i + 1, 0.0, 0.0, 0.0)
            car.current_node_name = start_node
            if start_node in map_data.NODES:
                car.x_m = map_data.NODES[start_node][0]
                car.y_m = map_data.NODES[start_node][1]
            car.attach_visual(self.renderer.root)
            car.update_visual(self.heightmap)
            self.cars.append(car)
            self.kfs.append(KalmanFilter(dt=1.0 / 60.0, start_x=car.x_m, start_y=car.y_m))

        self.dispatcher.update_global_plan(self.cars)
        for idx, car in enumerate(self.cars):
            target_node = self.dispatcher.assign_task(car)
            car.target_node_name = target_node

            route = self.local_planner.compute_route(
                self.road_graph,
                car.current_node_name,
                target_node,
                cache=self.route_cache,
            )
            if not route:
                continue

            wp = get_path_from_nodes(route, self.waypoints_map)
            if not wp or len(wp) < 2:
                continue

            pos = wp[0]
            angle = math.atan2(wp[1][1] - wp[0][1], wp[1][0] - wp[0][0])
            car.x_m, car.y_m, car.angle = pos[0], pos[1], angle
            car.path = Path(wp)
            car.op_state = "GOING_TO_ENDPOINT"
            car.desired_speed_ms = SPEED_MS_EMPTY
            self.kfs[idx] = KalmanFilter(dt=1.0 / 60.0, start_x=pos[0], start_y=pos[1])
            car.run_mpc([])
            car.update_visual(self.heightmap, is_selected=(idx == self.selected_car_idx))

    def _toggle_pause(self):
        self.paused = not self.paused

    def _set_speed(self, value):
        self.sim_speed = float(value)

    def _next_truck(self):
        if self.cars:
            self.selected_car_idx = (self.selected_car_idx + 1) % len(self.cars)

    def _exit(self):
        for car in self.cars:
            car.destroy()
        self.userExit()

    def _pick_hover(self):
        if not self.mouseWatcherNode.hasMouse():
            return None

        mp = self.mouseWatcherNode.getMouse()
        self.truck_ray.setFromLens(self.camNode, mp.getX(), mp.getY())
        self.truck_traverser.traverse(self.render)
        if self.truck_queue.getNumEntries() > 0:
            self.truck_queue.sortEntries()
            entry = self.truck_queue.getEntry(0)
            into_np = entry.getIntoNodePath()
            if not into_np.isEmpty() and into_np.hasNetTag("truck_id"):
                truck_id = int(into_np.getNetTag("truck_id"))
                for car in self.cars:
                    if car.id == truck_id:
                        z = self.heightmap.get_height_at_world(car.x_m, car.y_m) + CAR_HEIGHT_M + 1.2
                        return {"type": "truck", "car": car, "world": np.array([car.x_m, car.y_m, z])}

        pick = self.picker.pick_ground()
        if pick.world_xy is not None:
            zone = nearest_zone(pick.world_xy, self.dispatcher)
            if zone:
                zx, zy = zone["world"][0], zone["world"][1]
                zz = self.heightmap.get_height_at_world(zx, zy) + 2.5
                zone["world"] = np.array([zx, zy, zz])
                return zone
        return None

    def _update_hud(self):
        if not self.cars:
            self.hud.setText("No trucks")
            return
        sel = self.cars[self.selected_car_idx]
        self.hud.setText(
            f"Truck {sel.id} | Speed {sel.speed_ms * 3.6:.1f} km/h | State {sel.op_state}\n"
            f"Target {sel.target_node_name or '-'} | Sim {self.sim_speed}x | {'Paused' if self.paused else 'Running'}\n"
            "TAB next truck | SPACE pause | SHIFT+0..5 speed | WASD/Arrows pan | RMB orbit | Wheel zoom"
        )

    def sim_tick(self, task):
        frame_dt = globalClock.getDt()
        if frame_dt <= 0.0:
            return Task.cont

        sim_dt = 0.0 if self.paused else frame_dt * self.sim_speed
        if sim_dt > 0:
            self.mpc_timer += sim_dt
            self.traffic_update_timer += sim_dt
            self.global_opt_timer += sim_dt

            if self.traffic_update_timer >= self.TRAFFIC_UPDATE_INTERVAL:
                self.traffic_update_timer = 0.0
                self.dispatcher.update_traffic_weights(self.cars)

            if self.global_opt_timer >= self.GLOBAL_OPT_INTERVAL:
                self.global_opt_timer = 0.0
                self.dispatcher.update_global_plan(self.cars)

            if self.mpc_timer >= self.MPC_INTERVAL:
                self.mpc_timer = 0.0
                all_trajectories = [c.planned_trajectory for c in self.cars]
                for i, car in enumerate(self.cars):
                    if car.op_state == "TURNING_AROUND":
                        continue
                    other = [all_trajectories[j] for j in range(len(self.cars)) if j != i]
                    car.run_mpc(other, other_cars=self.cars)

            for idx, car in enumerate(self.cars):
                kf = self.kfs[idx]
                est_pos_m = np.array([kf.x[0], kf.x[2]])

                if car.path and car.op_state != "TURNING_AROUND":
                    car.s_path_m = car.path.project(est_pos_m, car.s_path_m)

                _, base_speed_ms = car.update_op_state(sim_dt, self.dispatcher)
                car.desired_speed_ms = base_speed_ms

                if car.needs_new_path:
                    car.needs_new_path = False
                    car.current_node_name = car.target_node_name
                    new_target = self.dispatcher.assign_task(car)
                    car.target_node_name = new_target
                    route_node_names = self.local_planner.compute_route(
                        self.dispatcher.get_graph(),
                        car.current_node_name,
                        new_target,
                        cache=self.route_cache,
                    )
                    if route_node_names:
                        waypoints_m = get_path_from_nodes(route_node_names, self.waypoints_map)
                        if waypoints_m and len(waypoints_m) >= 2:
                            car.path = Path(waypoints_m)
                            car.s_path_m = 0.0
                            car.turn_target_angle = math.atan2(
                                waypoints_m[1][1] - waypoints_m[0][1],
                                waypoints_m[1][0] - waypoints_m[0][0],
                            )
                            car.current_mpc_control = np.zeros(2)
                            car.mpc.prev_u = np.zeros_like(car.mpc.prev_u)

                if car.op_state == "TURNING_AROUND":
                    turn_done = car.execute_turn_step(sim_dt)
                    if turn_done:
                        if car.path:
                            car.s_path_m = car.path.project(np.array([car.x_m, car.y_m]), 0.0)
                        kf.x[0] = car.x_m
                        kf.x[1] = 0.0
                        kf.x[2] = car.y_m
                        kf.x[3] = 0.0
                        car.run_mpc([], other_cars=self.cars)
                else:
                    car.move(sim_dt)

                accel_vec_m = np.array([
                    car.accel_ms2 * math.cos(car.angle),
                    car.accel_ms2 * math.sin(car.angle),
                ])
                kf.predict(u=accel_vec_m)
                kf.update(z=car.get_noisy_measurement())

        selected = self.cars[self.selected_car_idx] if self.cars else None
        self.graphics.draw_active_paths(selected)

        for idx, car in enumerate(self.cars):
            car.update_visual(self.heightmap, is_selected=(idx == self.selected_car_idx))

        hover = self._pick_hover()
        if hover:
            text = build_entity_tooltip(hover["type"], hover)
            self.tooltip_overlay.show_entity(text, hover["world"])
        else:
            self.tooltip_overlay.hide()

        self._update_hud()
        return Task.cont


def run_simulation():
    app = PandaSimulationApp()
    app.run()


if __name__ == "__main__":
    run_simulation()
