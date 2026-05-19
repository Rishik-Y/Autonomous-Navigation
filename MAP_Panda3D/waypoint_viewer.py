import heapq
import os
import pickle
import random

import numpy as np

import map_data
import map_storage

POINTS_PER_SEGMENT = 20


def a_star_pathfinding(graph: dict, start_name: str, goal_name: str):
    open_set = [(0, start_name)]
    came_from = {}
    g_score = {name: float("inf") for name in graph}
    g_score[start_name] = 0
    while open_set:
        _, current_name = heapq.heappop(open_set)
        if current_name == goal_name:
            path_names = []
            temp = current_name
            while temp in came_from:
                path_names.append(temp)
                temp = came_from[temp]
            path_names.append(start_name)
            return list(reversed(path_names))
        for neighbor_name, weight in graph[current_name]:
            tentative = g_score[current_name] + weight
            if tentative < g_score[neighbor_name]:
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative
                heapq.heappush(open_set, (tentative, neighbor_name))
    return []


def get_path_from_nodes(route_node_names, waypoints_map):
    final_waypoints = []
    if not route_node_names:
        return []
    for i in range(len(route_node_names) - 1):
        seg_start, seg_end = route_node_names[i], route_node_names[i + 1]
        found = False
        for chain_tuple, waypoints in waypoints_map.items():
            try:
                idx = chain_tuple.index(seg_start)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == seg_end:
                    start = idx * POINTS_PER_SEGMENT
                    end = (idx + 1) * POINTS_PER_SEGMENT
                    final_waypoints.extend(waypoints[start:end])
                    found = True
                    break
                idx = chain_tuple.index(seg_end)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == seg_start:
                    start = idx * POINTS_PER_SEGMENT
                    end = (idx + 1) * POINTS_PER_SEGMENT
                    # Include end+1 here so reversing preserves the terminal waypoint
                    # for this segment before dropping the duplicated join point.
                    segment = waypoints[start : end + 1]
                    final_waypoints.extend(segment[::-1][:-1])
                    found = True
                    break
            except ValueError:
                continue
        if not found:
            # Keep behavior aligned with MAP/Simulation: silently skip missing segments.
            continue
    if final_waypoints and route_node_names:
        final_waypoints.append(map_data.NODES[route_node_names[-1]])
    return final_waypoints


class WaypointViewerMode:
    label = "Waypoint Viewer"

    def __init__(self, app):
        self.app = app
        self.waypoints_map = {}
        self.route_node_names = []
        self.waypoints_m = []
        self.status_text = "Waypoint viewer"

    def activate(self):
        waypoints_path = map_storage.resolve_input_path(
            "waypoints.pkl",
            [map_storage.legacy_path("waypoints.pkl"), map_storage.simulation_path("waypoints.pkl")],
        )
        cache_path = map_storage.resolve_input_path(
            "map_cache.pkl",
            [map_storage.legacy_path("map_cache.pkl"), map_storage.simulation_path("map_cache.pkl")],
        )
        if not os.path.exists(waypoints_path) or not os.path.exists(cache_path):
            self.status_text = "Missing waypoints.pkl or map_cache.pkl"
            self.waypoints_map = {}
            self.route_node_names = []
            self.waypoints_m = []
            self.redraw()
            return

        with open(waypoints_path, "rb") as f:
            self.waypoints_map = pickle.load(f)
        with open(cache_path, "rb") as f:
            road_graph = pickle.load(f)["road_graph"]

        if not map_data.LOAD_ZONES or not map_data.DUMP_ZONES:
            self.status_text = "No load/dump zones"
            self.redraw()
            return

        start = random.choice(map_data.DUMP_ZONES)
        goal = random.choice(map_data.LOAD_ZONES)
        self.route_node_names = a_star_pathfinding(road_graph, start, goal)
        self.waypoints_m = get_path_from_nodes(self.route_node_names, self.waypoints_map)
        self.status_text = f"Route: {start} -> {goal} | A* nodes: {len(self.route_node_names)} | spline points: {len(self.waypoints_m)}"
        self.redraw()

    def deactivate(self):
        pass

    def redraw(self):
        self.app.renderer.draw_grid()
        self.app.renderer.draw_roads(self.waypoints_map.values(), color=(0.4, 0.4, 0.4, 1), width=2.0)
        self.app.renderer.draw_nodes(map_data.NODES, map_data.LOAD_ZONES, map_data.DUMP_ZONES, map_data.FUEL_ZONES, False)

        if len(self.route_node_names) > 1:
            orange_path = [map_data.NODES[name] for name in self.route_node_names if name in map_data.NODES]
            self.app.renderer.draw_roads([orange_path], color=(1.0, 0.55, 0.0, 1), width=4.0, z=1.1)

        if self.waypoints_m:
            colors = [(0.0, 0.6, 1.0, 1.0)] * len(self.waypoints_m)
            self.app.renderer._point_cloud(self.waypoints_m, colors, size=4, z=1.3)

    def on_key(self, key):
        if key == "space":
            self.activate()

    def on_mouse1(self, down=True):
        pass

    def on_mouse_move(self):
        pass

    def tick(self):
        pass

    @property
    def controls_text(self):
        return "[Space] New random route"


def run_viewer():
    from direct.showbase.ShowBase import ShowBase
    from panda_common import CameraController, Picker, SceneRenderer

    class _WpViewApp(ShowBase):
        def __init__(self):
            super().__init__()
            self.disableMouse()
            self.camera_controller = CameraController(self)
            self.picker = Picker(self)
            self.renderer = SceneRenderer(self)
            self.mode = WaypointViewerMode(self)
            self.mode.activate()
            self.accept("space", self.mode.on_key, ["space"])

    app = _WpViewApp()
    app.run()


if __name__ == "__main__":
    run_viewer()
