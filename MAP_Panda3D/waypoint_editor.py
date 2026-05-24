import pickle

import numpy as np

import map_data
import map_storage
import session_tracker
from panda_common import POINTS_PER_SEGMENT, generate_curvy_path_from_nodes


class WaypointEditorMode:
    label = "Waypoint Editor"

    def __init__(self, app):
        self.app = app
        self.generated_waypoints_map = {}
        self.background_splines_map = {}
        self.is_dirty = False
        self.status_text = "Press A Generate, S Save, L Load"
        self._saved_files = []

    def activate(self):
        self.background_splines_map = self.generate_all_waypoints()
        self.redraw()

    def deactivate(self):
        pass

    def generate_all_waypoints(self):
        waypoints_data = {}
        for chain in map_data.VISUAL_ROAD_CHAINS:
            chain_key = tuple(chain)
            node_coords = [map_data.NODES[node_name] for node_name in chain_key if node_name in map_data.NODES]
            if len(node_coords) < 2:
                continue
            waypoints_data[chain_key] = generate_curvy_path_from_nodes(node_coords)
        return waypoints_data

    def save_waypoints_data(self):
        filepath = "waypoints.pkl"
        data = pickle.dumps(self.generated_waypoints_map)
        map_storage.write_binary_file(
            filepath,
            data,
            copy_targets=[map_storage.legacy_path(filepath), map_storage.simulation_path(filepath)],
        )
        self._saved_files.append(filepath)
        self.is_dirty = False
        session_tracker.mark_save_occurred()
        self.status_text = "Saved waypoints to Saved_Map/waypoints.pkl"

    def load_waypoints_data(self):
        path = map_storage.resolve_input_path(
            "waypoints.pkl",
            [map_storage.legacy_path("waypoints.pkl"), map_storage.simulation_path("waypoints.pkl")],
        )
        with open(path, "rb") as f:
            self.generated_waypoints_map = pickle.load(f)
        self.status_text = f"Loaded {len(self.generated_waypoints_map)} waypoint chains"
        self.is_dirty = False

    def redraw(self):
        self.app.renderer.draw_grid()
        self.app.renderer.draw_roads(self.background_splines_map.values(), color=(0.45, 0.45, 0.45, 1), width=2.0)
        self.app.renderer.draw_nodes(map_data.NODES, map_data.LOAD_ZONES, map_data.DUMP_ZONES, map_data.FUEL_ZONES)

        points = []
        colors = []
        for waypoints in self.generated_waypoints_map.values():
            for p in waypoints:
                points.append(p)
                colors.append((0.1, 0.6, 1.0, 1.0))
        if points:
            self.app.renderer._point_cloud(points, colors, size=4, z=1.2)

    def on_key(self, key):
        if key == "a":
            self.generated_waypoints_map = self.generate_all_waypoints()
            self.status_text = f"Generated {len(self.generated_waypoints_map)} waypoint chains"
            self.is_dirty = True
            self.redraw()
        elif key == "s":
            if self.generated_waypoints_map:
                self.save_waypoints_data()
        elif key == "l":
            self.load_waypoints_data()
            self.redraw()

    def on_mouse1(self, down=True):
        pass

    def on_mouse_move(self):
        pass

    def tick(self):
        pass

    @property
    def controls_text(self):
        return "[A] Generate All | [S] Save | [L] Load"


def run_waypoint_editor():
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import Point3
    from panda_common import CameraController, Picker, SceneRenderer

    class _WpEditorApp(ShowBase):
        def __init__(self):
            super().__init__()
            self.disableMouse()
            self.renderer = SceneRenderer(self)
            hm = self.renderer.heightmap
            cx = hm.origin_x + hm.cols * hm.cell_size * 0.5
            cy = hm.origin_y + hm.rows * hm.cell_size * 0.5
            cz = hm.get_height_at_world(cx, cy)
            self.camera_controller = CameraController(self, center=Point3(cx, cy, cz), dist=1400)
            self.picker = Picker(self, heightmap=self.renderer.heightmap, terrain_np_getter=self.renderer.get_terrain_np)
            self.mode = WaypointEditorMode(self)
            self.mode.activate()
            for k in ["a", "s", "l"]:
                self.accept(k, self.mode.on_key, [k])

    app = _WpEditorApp()
    app.run()


if __name__ == "__main__":
    run_waypoint_editor()
