import json
import os

import numpy as np

from direct.gui.DirectEntry import DirectEntry
from direct.gui.DirectFrame import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

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
        self.base_status = "Click on a mine node to edit coal capacity."
        self.status_text = self.base_status
        self._saved_files = []
        self.config = {"truck_count": DEFAULT_TRUCK_COUNT, "coal_capacities": {}}
        self.hovered_mine = None
        self.dialog_frame = None
        self.dialog_entry = None
        self.dialog_title = None
        self.dialog_hint = None

    def activate(self):
        self.load_map_data()
        self.load_config()
        self.sync_config_with_map()
        self.rebuild_splines()
        self.redraw()

    def deactivate(self):
        self._close_capacity_dialog()

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
        self.base_status = "Configuration SAVED to Saved_Map/mine_config.json"
        self.status_text = self.base_status
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
        self.app.renderer.draw_nodes(self.NODES, self.LOAD_ZONES, self.DUMP_ZONES, self.FUEL_ZONES, self.hovered_mine)

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
        if self.dialog_frame is not None:
            return
        if key == "s":
            self.save_config()
        elif key in ("+", "="):
            self.config["truck_count"] = min(50, self.config["truck_count"] + 1)
            self.is_dirty = True
            self.base_status = f"Truck count: {self.config['truck_count']}"
            self.status_text = self.base_status
        elif key == "-":
            self.config["truck_count"] = max(1, self.config["truck_count"] - 1)
            self.is_dirty = True
            self.base_status = f"Truck count: {self.config['truck_count']}"
            self.status_text = self.base_status
        elif key == "r":
            self.activate()
            self.base_status = "Reloaded map data and synced config"
            self.status_text = self.base_status

    def on_mouse1(self, down=True):
        if not down:
            return
        if self.dialog_frame is not None:
            return
        pos = self.app.picker.pick_surface().world_xy
        mine = self._get_mine_at(pos)
        if not mine:
            return
        self._open_capacity_dialog(mine)

    def on_mouse_move(self):
        if self.dialog_frame is not None:
            return
        pos = self.app.picker.pick_surface().world_xy
        mine = self._get_mine_at(pos)
        if mine != self.hovered_mine:
            self.hovered_mine = mine
            self.redraw()

    def tick(self):
        hover_text = ""
        if self.hovered_mine and self.hovered_mine in self.config["coal_capacities"]:
            capacity = self.config["coal_capacities"][self.hovered_mine]
            hover_text = f" | Hover: {self.hovered_mine} = {capacity} kg"
        self.status_text = f"{self.base_status}{hover_text}"

    @property
    def controls_text(self):
        return (
            "[S] Save | [+/-] Truck Count | [R] Reload | Left Click mine: set exact coal\n"
            f"Trucks: {self.config['truck_count']} | Total Mines: {len(self.LOAD_ZONES)}"
        )

    def _open_capacity_dialog(self, mine):
        current = self.config["coal_capacities"].get(mine, DEFAULT_COAL_CAPACITY)
        self._close_capacity_dialog()
        self.app.camera_controller.set_pan_enabled(False)
        self.dialog_frame = DirectFrame(
            parent=self.app.aspect2d,
            frameColor=(1, 1, 1, 0.95),
            frameSize=(-0.62, 0.62, -0.2, 0.2),
            pos=(0, 0, 0),
        )
        self.dialog_title = OnscreenText(
            text=f"Coal capacity for {mine}",
            pos=(0, 0.09),
            scale=0.055,
            fg=(0, 0, 0, 1),
            align=TextNode.ACenter,
            mayChange=False,
            parent=self.dialog_frame,
        )
        self.dialog_hint = OnscreenText(
            text="Type exact value and press Enter",
            pos=(0, -0.12),
            scale=0.04,
            fg=(0.2, 0.2, 0.2, 1),
            align=TextNode.ACenter,
            mayChange=False,
            parent=self.dialog_frame,
        )
        self.dialog_entry = DirectEntry(
            parent=self.dialog_frame,
            scale=0.06,
            width=10,
            numLines=1,
            initialText=str(current),
            command=self._submit_capacity_dialog,
            extraArgs=[mine],
            focus=1,
            pos=(-0.3, 0, -0.03),
        )
        self.dialog_entry.enterText(str(current))

    def _submit_capacity_dialog(self, _text, mine):
        raw = self.dialog_entry.get().strip() if self.dialog_entry is not None else ""
        if not raw.isdigit():
            self.base_status = "Capacity must be a non-negative integer"
            self.status_text = self.base_status
            return
        value = max(0, int(raw))
        self.config["coal_capacities"][mine] = value
        self.is_dirty = True
        self.base_status = f"{mine} capacity set to {value} kg"
        self.status_text = self.base_status
        self._close_capacity_dialog()

    def _close_capacity_dialog(self):
        if self.dialog_entry is not None:
            self.dialog_entry.destroy()
            self.dialog_entry = None
        if self.dialog_title is not None:
            self.dialog_title.destroy()
            self.dialog_title = None
        if self.dialog_hint is not None:
            self.dialog_hint.destroy()
            self.dialog_hint = None
        if self.dialog_frame is not None:
            self.dialog_frame.destroy()
            self.dialog_frame = None
        self.app.camera_controller.set_pan_enabled(True)


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
