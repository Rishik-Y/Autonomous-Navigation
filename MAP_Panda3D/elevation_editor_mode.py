import map_data
import session_tracker
from panda_common import generate_curvy_path_from_nodes


class ElevationEditorMode:
    label = "Elevation Editor"

    def __init__(self, app):
        self.app = app
        self.edit_mode = "+"
        self.brush_size = 1
        self.hovered_cell = None
        self.clicking = False
        self.last_edit = None
        self.is_dirty = False
        self.status_text = "Mode: Raise | Brush: 1x1"
        self._saved_files = []
        self._splines = []

    def activate(self):
        self._rebuild_splines()
        self.redraw()
        self.status_text = f"Mode: {self._mode_label()} | Brush: {self.brush_size}x{self.brush_size}"

    def deactivate(self):
        self.clicking = False
        self.last_edit = None
        self.hovered_cell = None

    def _rebuild_splines(self):
        self._splines = []
        for chain in map_data.VISUAL_ROAD_CHAINS:
            node_coords = [map_data.NODES[node_name] for node_name in chain if node_name in map_data.NODES]
            if len(node_coords) >= 2:
                self._splines.append(generate_curvy_path_from_nodes(node_coords))

    def redraw(self):
        self.app.renderer.draw_grid()
        self.app.renderer.draw_roads(self._splines, color=(0.45, 0.45, 0.45, 1.0), width=2.0)
        self.app.renderer.draw_nodes(map_data.NODES, map_data.LOAD_ZONES, map_data.DUMP_ZONES, map_data.FUEL_ZONES)

    def _mode_label(self):
        return {"+": "Raise", "-": "Lower", "0": "Level"}.get(self.edit_mode, "Raise")

    def _set_status(self, prefix=None):
        status = prefix or f"Mode: {self._mode_label()}"
        if self.hovered_cell is not None:
            col, row = self.hovered_cell
            h = self.app.heightmap.get(col, row)
            status = f"{status} | Cell: ({col},{row}) Height: {h}"
        self.status_text = f"{status} | Brush: {self.brush_size}x{self.brush_size}"

    def _update_hover(self):
        pick_result = self.app.picker.pick_surface()
        if pick_result.world_xy is None:
            self.hovered_cell = None
            return
        col, row = self.app.heightmap.world_to_cell(*pick_result.world_xy)
        if 0 <= col < self.app.heightmap.cols and 0 <= row < self.app.heightmap.rows:
            self.hovered_cell = (col, row)
        else:
            self.hovered_cell = None

    def _apply_mode(self, current_height):
        if self.edit_mode == "+":
            return current_height + 1
        if self.edit_mode == "-":
            return current_height - 1
        if current_height > 0:
            return current_height - 1
        if current_height < 0:
            return current_height + 1
        return current_height

    def _apply_brush_edit(self, col, row):
        start_col = col - (self.brush_size // 2)
        start_row = row - (self.brush_size // 2)
        changed = False
        for dc in range(self.brush_size):
            for dr in range(self.brush_size):
                c = start_col + dc
                r = start_row + dr
                if not (0 <= c < self.app.heightmap.cols and 0 <= r < self.app.heightmap.rows):
                    continue
                current = self.app.heightmap.get(c, r)
                updated = self._apply_mode(current)
                if updated != current:
                    self.app.heightmap.set(c, r, updated)
                    changed = True
        if changed:
            self.app.renderer.draw_terrain()
            self.redraw()
            self.is_dirty = True
        self.last_edit = (col, row)

    def save_elevation_data(self):
        path = self.app.heightmap.save_json()
        self.is_dirty = False
        self._saved_files.append(path)
        session_tracker.mark_save_occurred()
        self._set_status(f"Saved elevation: {path.split('/')[-1]}")

    def on_key(self, key):
        if key in ("=",):
            self.edit_mode = "+"
            self._set_status("Mode: Raise")
        elif key in ("-",):
            self.edit_mode = "-"
            self._set_status("Mode: Lower")
        elif key in ("0",):
            self.edit_mode = "0"
            self._set_status("Mode: Level")
        elif key in ("+",):
            self.brush_size = min(15, self.brush_size + 1)
            self._set_status(f"Brush size: {self.brush_size}x{self.brush_size}")
        elif key in ("_",):
            self.brush_size = max(1, self.brush_size - 1)
            self._set_status(f"Brush size: {self.brush_size}x{self.brush_size}")
        elif key in ("l", "L"):
            if self.app.heightmap.load_json():
                self.app.renderer.draw_terrain()
                self.redraw()
                self.is_dirty = False
                self._set_status("Loaded elevation data")
            else:
                self._set_status("No saved elevation data found")
        elif key in ("s", "S"):
            self.save_elevation_data()

    def on_mouse1(self, down=True):
        self.clicking = down
        if down:
            self.last_edit = None
            self._update_hover()
            if self.hovered_cell is not None:
                self._apply_brush_edit(*self.hovered_cell)
        else:
            self.last_edit = None

    def on_mouse_move(self):
        self._update_hover()
        if self.clicking and self.hovered_cell is not None and self.hovered_cell != self.last_edit:
            self._apply_brush_edit(*self.hovered_cell)
        self._set_status()

    def tick(self):
        pass

    @property
    def controls_text(self):
        return "[=] Raise [-] Lower [Shift+=] Brush+ [Shift+-] Brush- [S] Save [L] Load"
