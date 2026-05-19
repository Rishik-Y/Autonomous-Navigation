import json
import os

import numpy as np

import map_storage
import session_tracker
from panda_common import generate_curvy_path_from_nodes

DEFAULT_COAL_CAPACITY = 100
DEFAULT_TRUCK_COUNT = 5
CLICK_THRESHOLD_M = 8.0


class CoalMineEditorMode:
    label = "Coal Mine Editor"

    def __init__(self, app):
        self.app = app
        self.NODES = {}
        self.LOAD_ZONES = []
        self.DUMP_ZONES = []
        self.FUEL_ZONES = []
        self.VISUAL_ROAD_CHAINS = []
        self.PRE_CALCULATED_SPLINES = []
        self.is_dirty = False
        self.status_text = "Click mine node to increase capacity (+100)."
        self._saved_files = []
        self.config = {"truck_count": DEFAULT_TRUCK_COUNT, "coal_capacities": {}}

    def activate(self):
        self.load_map_data()
        self.load_config()
        self.sync_config_with_map()
        self.rebuild_splines()
        self.redraw()

    def deactivate(self):
        pass

    def load_map_data(self):
        map_file = map_storage.resolve_input_path("map_data.py", [map_storage.legacy_path("map_data.py")])
        sandbox = {"np": np}
        with open(map_file, "r", encoding="utf-8") as f:
            exec(f.read(), sandbox)
        self.NODES = sandbox.get("NODES", {})
        self.LOAD_ZONES = list(sandbox.get("LOAD_ZONES", []))
        self.DUMP_ZONES = list(sandbox.get("DUMP_ZONES", []))
        self.FUEL_ZONES = list(sandbox.get("FUEL_ZONES", []))
        self.VISUAL_ROAD_CHAINS = [list(c) for c in sandbox.get("VISUAL_ROAD_CHAINS", [])]

    def load_config(self):
        config_path = map_storage.resolve_input_path(
            "mine_config.json",
            [map_storage.legacy_path("mine_config.json"), map_storage.simulation_path("mine_config.json")],
        )
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.config = {
                "truck_count": loaded.get("truck_count", DEFAULT_TRUCK_COUNT),
                "coal_capacities": loaded.get("coal_capacities", {}),
            }

    def sync_config_with_map(self):
        current = set(self.LOAD_ZONES)
        cap = self.config.setdefault("coal_capacities", {})
        for key in list(cap.keys()):
            if key not in current:
                del cap[key]
        for mine in current:
            cap.setdefault(mine, DEFAULT_COAL_CAPACITY)

    def save_config(self):
        content = json.dumps(self.config, indent=4, sort_keys=True)
        map_storage.write_text_file(
            "mine_config.json",
            content,
            copy_targets=[map_storage.legacy_path("mine_config.json"), map_storage.simulation_path("mine_config.json")],
        )
        self._saved_files.append("mine_config.json")
        self.is_dirty = False
        self.status_text = "Configuration SAVED to Saved_Map/mine_config.json"
        session_tracker.mark_save_occurred()

    def rebuild_splines(self):
        self.PRE_CALCULATED_SPLINES = []
        for chain in self.VISUAL_ROAD_CHAINS:
            node_coords = [self.NODES[node_name] for node_name in chain if node_name in self.NODES]
            if len(node_coords) >= 2:
                self.PRE_CALCULATED_SPLINES.append(generate_curvy_path_from_nodes(node_coords))

    def redraw(self):
        self.app.renderer.draw_grid()
        self.app.renderer.draw_roads(self.PRE_CALCULATED_SPLINES, color=(0.5, 0.5, 0.5, 1), width=2.0)
        self.app.renderer.draw_nodes(self.NODES, self.LOAD_ZONES, self.DUMP_ZONES, self.FUEL_ZONES, False)

    def _get_mine_at(self, pos):
        if pos is None:
            return None
        for mine in self.LOAD_ZONES:
            if mine not in self.NODES:
                continue
            if float(np.linalg.norm(np.array(pos) - self.NODES[mine])) < CLICK_THRESHOLD_M:
                return mine
        return None

    def on_key(self, key):
        if key == "s":
            self.save_config()
        elif key in ("+", "="):
            self.config["truck_count"] = min(50, self.config["truck_count"] + 1)
            self.is_dirty = True
            self.status_text = f"Truck count: {self.config['truck_count']}"
        elif key == "-":
            self.config["truck_count"] = max(1, self.config["truck_count"] - 1)
            self.is_dirty = True
            self.status_text = f"Truck count: {self.config['truck_count']}"
        elif key == "r":
            self.activate()
            self.status_text = "Reloaded map data and synced config"

    def on_mouse1(self, down=True):
        if not down:
            return
        pos = self.app.picker.pick_surface().world_xy
        mine = self._get_mine_at(pos)
        if not mine:
            return
        current = self.config["coal_capacities"].get(mine, DEFAULT_COAL_CAPACITY)
        self.config["coal_capacities"][mine] = max(0, current + 100)
        self.is_dirty = True
        self.status_text = f"{mine} capacity: {self.config['coal_capacities'][mine]} kg"

    def on_mouse_move(self):
        pass

    def tick(self):
        pass

    @property
    def controls_text(self):
        return "[S] Save | [+/-] Truck Count | [R] Reload | Left Click load zone node: +100 coal"


def run_editor():
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import Point3
    from panda_common import CameraController, Picker, SceneRenderer

    class _CoalApp(ShowBase):
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
            self.mode = CoalMineEditorMode(self)
            self.mode.activate()
            self.accept("mouse1", self.mode.on_mouse1, [True])
            for key in list("s+r") + ["-", "+", "="]:
                self.accept(key, self.mode.on_key, [key])
            self.taskMgr.add(self._tick, "coal_editor_tick")

        def _tick(self, task):
            self.mode.tick()
            return task.cont

    app = _CoalApp()
    app.run()


if __name__ == "__main__":
    run_editor()
