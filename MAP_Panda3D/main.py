"""Unified Panda3D MAP launcher with single persistent window and TAB mode switching."""

import sys

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3, loadPrcFileData

import map_storage
import session_tracker
from coal_mine_editor import CoalMineEditorMode
from elevation_editor_mode import ElevationEditorMode
from map_editor import MapEditorMode
from map_ui import ModeOverlay, SavePrompt, StatusHud
from map_viewer import MapViewerMode
from panda_elevation import Heightmap, TerrainMesh
from panda_common import CameraController, Picker, SceneRenderer
from waypoint_editor import WaypointEditorMode
from waypoint_viewer import WaypointViewerMode


class UnifiedPandaMapLauncher(ShowBase):
    def __init__(self):
        loadPrcFileData("", "window-title MAP Panda3D")
        loadPrcFileData("", "win-size 1400 900")
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.95, 0.95, 0.95, 1.0)

        session_tracker.reset_save_tracker()
        self.files_saved = []
        self.pending_switch = None

        self.heightmap = Heightmap()
        self.heightmap.load_json()
        self.terrain = TerrainMesh(self.heightmap)
        self.renderer = SceneRenderer(self, self.heightmap, self.terrain)

        cx = self.heightmap.origin_x + self.heightmap.cols * self.heightmap.cell_size * 0.5
        cy = self.heightmap.origin_y + self.heightmap.rows * self.heightmap.cell_size * 0.5
        cz = self.heightmap.get_height_at_world(cx, cy)
        self.camera_controller = CameraController(self, center=Point3(cx, cy, cz), dist=1400)
        self.picker = Picker(self, heightmap=self.heightmap, terrain_np_getter=self.renderer.get_terrain_np)

        self.mode_overlay = ModeOverlay(self)
        self.status_hud = StatusHud(self)
        self.save_prompt = SavePrompt(self)

        self.modes = [
            MapEditorMode(self),
            CoalMineEditorMode(self),
            WaypointEditorMode(self),
            MapViewerMode(self),
            WaypointViewerMode(self),
            ElevationEditorMode(self),
        ]
        self.mode_index = 0
        self.current_mode = self.modes[self.mode_index]
        self.current_mode.activate()

        self.accept("escape", self.request_exit)
        self.accept("tab", self.request_next_mode)
        self.accept("shift-tab", self.request_prev_mode)

        for key in "abcdefghijklmnopqrstuvwxyz":
            self.accept(key, self.handle_key, [key])
            self.accept(key.upper(), self.handle_key, [key.upper()])
        for key in ["+", "=", "-", "space"]:
            self.accept(key, self.handle_key, [key])

        self.accept("mouse1", self.on_mouse1_down)
        self.accept("mouse1-up", self.on_mouse1_up)

        self.taskMgr.add(self._tick, "unified_tick")

    def handle_key(self, key):
        if self.save_prompt.handle_key(key):
            return
        self.current_mode.on_key(key)

    def request_next_mode(self):
        self._request_mode_switch(1)

    def request_prev_mode(self):
        self._request_mode_switch(-1)

    def _request_mode_switch(self, delta):
        if getattr(self.current_mode, "is_dirty", False):
            self.pending_switch = delta
            self.save_prompt.show(self.current_mode.label, self._save_prompt_result)
            return
        self.switch_mode(delta)

    def _save_prompt_result(self, choice):
        if choice == "save":
            if hasattr(self.current_mode, "save_map_data"):
                self.current_mode.save_map_data()
            elif hasattr(self.current_mode, "save_config"):
                self.current_mode.save_config()
            elif hasattr(self.current_mode, "save_waypoints_data"):
                self.current_mode.save_waypoints_data()
            elif hasattr(self.current_mode, "save_elevation_data"):
                self.current_mode.save_elevation_data()
        elif choice == "cancel":
            self.pending_switch = None
            return

        if self.pending_switch is not None:
            self.switch_mode(self.pending_switch)
            self.pending_switch = None

    def switch_mode(self, delta):
        self.collect_saved_files(self.current_mode)
        self.current_mode.deactivate()
        self.mode_index = (self.mode_index + delta) % len(self.modes)
        self.current_mode = self.modes[self.mode_index]
        self.current_mode.activate()

    def collect_saved_files(self, mode):
        saved = getattr(mode, "_saved_files", [])
        if saved:
            self.files_saved.extend(saved)

    def on_mouse1_down(self):
        self.current_mode.on_mouse1(True)

    def on_mouse1_up(self):
        self.current_mode.on_mouse1(False)

    def _tick(self, task):
        self.current_mode.on_mouse_move()
        self.current_mode.tick()
        self.mode_overlay.update(self.current_mode.label, self.mode_index + 1, len(self.modes), getattr(self.current_mode, "is_dirty", False))
        self.status_hud.update(self.current_mode.label, self.current_mode.controls_text, getattr(self.current_mode, "status_text", ""))
        return task.cont

    def request_exit(self):
        self.collect_saved_files(self.current_mode)
        self.current_mode.deactivate()
        self._finalize_and_exit()

    def _finalize_and_exit(self):
        if self.files_saved:
            snapshot_path = map_storage.create_snapshot()
            print(f"✓ Snapshot saved to: {snapshot_path}")
            print(f"  Files saved: {', '.join(sorted(set(self.files_saved)))}")
        else:
            print("No changes were saved. No snapshot created.")
        session_tracker.reset_save_tracker()
        self.mode_overlay.destroy()
        self.status_hud.destroy()
        self.save_prompt.destroy()
        self.renderer.destroy()
        self.destroy()
        sys.exit(0)


def main():
    app = UnifiedPandaMapLauncher()
    app.run()


if __name__ == "__main__":
    main()
