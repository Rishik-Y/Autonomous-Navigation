import os

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    TransparencyAttrib,
)

import map_data
import session_tracker
from panda_common import generate_curvy_path_from_nodes

HIGHLIGHT_Z_OFFSET = 0.4
MAX_BRUSH_SIZE = 9


class BrushHighlight:
    """Translucent quad showing the active brush footprint."""

    def __init__(self, parent, cell_size):
        self.cell_size = float(cell_size)
        self.np = self._make(parent)
        self.np.hide()

    def _make(self, parent):
        fmt = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("brush_hl", fmt, Geom.UHStatic)
        vdata.setNumRows(4)
        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")
        for x, y in ((0, 0), (self.cell_size, 0), (self.cell_size, self.cell_size), (0, self.cell_size)):
            vw.addData3f(x, y, 0.0)
            cw.addData4f(0.25, 0.75, 1.0, 0.3)
        tris = GeomTriangles(Geom.UHStatic)
        tris.addVertices(0, 1, 2)
        tris.addVertices(0, 2, 3)
        tris.closePrimitive()
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        gnode = GeomNode("brush_hl")
        gnode.addGeom(geom)
        np = parent.attachNewNode(gnode)
        np.setTransparency(TransparencyAttrib.MAlpha)
        np.setTwoSided(True)
        np.setBin("fixed", 40)
        np.setDepthWrite(False)
        return np

    def show_at(self, col, row, hm, brush_size):
        size = max(1, int(brush_size))
        max_col = max(0, hm.cols - size)
        max_row = max(0, hm.rows - size)
        start_col = min(max(col - (size - 1) // 2, 0), max_col)
        start_row = min(max(row - (size - 1) // 2, 0), max_row)
        wx = hm.origin_x + start_col * hm.cell_size
        wy = hm.origin_y + start_row * hm.cell_size
        center_x = hm.origin_x + (col + 0.5) * hm.cell_size
        center_y = hm.origin_y + (row + 0.5) * hm.cell_size
        wz = hm.get_height_at_world(center_x, center_y) + HIGHLIGHT_Z_OFFSET
        self.np.setPos(wx, wy, wz)
        self.np.setScale(size, size, 1.0)
        self.np.show()

    def hide(self):
        self.np.hide()


class ElevationEditorMode:
    label = "Elevation Editor"

    def __init__(self, app):
        self.app = app
        self.heightmap = app.heightmap
        self.edit_mode = "+"
        self.brush_size = 1
        self.hovered_cell = None
        self.clicking = False
        self.last_edit = None
        self.is_dirty = False
        self.status_text = ""
        self.base_status = ""
        self._saved_files = []
        self.undo_stack = []
        self.road_splines = []
        self.highlight = BrushHighlight(self.app.renderer.root, self.heightmap.cell_size)
        self._update_base_status()

    def activate(self):
        self.hovered_cell = None
        self.clicking = False
        self.last_edit = None
        self.highlight.hide()
        self._update_base_status()
        self.road_splines = self._build_road_splines()
        self.redraw()
        self.app.accept("control-s", self._save)
        self.app.accept("control-z", self._undo)
        self.app.accept("shift-=", self._increase_brush)
        self.app.accept("shift--", self._decrease_brush)

    def deactivate(self):
        self.app.ignore("control-s")
        self.app.ignore("control-z")
        self.app.ignore("shift-=")
        self.app.ignore("shift--")
        self.highlight.hide()

    def redraw(self):
        self.app.renderer.draw_grid()
        self._draw_roads()
        self.app.renderer.draw_nodes({}, [], [], [])

    def _update_base_status(self, extra=None):
        label = {"+": "Raise", "-": "Lower", "0": "Level"}[self.edit_mode]
        base = f"Mode: {label} | Brush: {self.brush_size}x{self.brush_size}"
        if extra:
            base = f"{base} | {extra}"
        self.base_status = base
        self._update_status()

    def _update_status(self):
        if self.hovered_cell:
            col, row = self.hovered_cell
            h = self.heightmap.get(col, row)
            sign = "+" if h > 0 else ""
            hover = f" | Cell ({col}, {row}) Height: {sign}{h}"
        else:
            hover = ""
        self.status_text = f"{self.base_status}{hover}"

    def _increase_brush(self):
        self.brush_size = min(MAX_BRUSH_SIZE, self.brush_size + 1)
        self._update_base_status()
        if self.hovered_cell:
            self.highlight.show_at(*self.hovered_cell, self.heightmap, self.brush_size)

    def _decrease_brush(self):
        self.brush_size = max(1, self.brush_size - 1)
        self._update_base_status()
        if self.hovered_cell:
            self.highlight.show_at(*self.hovered_cell, self.heightmap, self.brush_size)

    def on_key(self, key):
        if key in ("+", "=", "1"):
            self.edit_mode = "+"
            self._update_base_status()
        elif key in ("-", "2"):
            self.edit_mode = "-"
            self._update_base_status()
        elif key in ("0", "3"):
            self.edit_mode = "0"
            self._update_base_status()
        elif key in ("z", "Z"):
            self._undo()
        elif key in ("s", "S"):
            self._save()

    def on_mouse1(self, down=True):
        if down:
            if not self.clicking and self.hovered_cell:
                self.undo_stack.append(self.app.heightmap.data.copy())
            self.clicking = True
            self.last_edit = None
            if self.hovered_cell:
                self._apply_edit(*self.hovered_cell)
        else:
            self.clicking = False
            self.last_edit = None

    def on_mouse_move(self):
        pick = self.app.picker.pick_surface().world_xyz
        if pick is None:
            self.hovered_cell = None
            self.highlight.hide()
            self._update_status()
            return
        col, row = self.heightmap.world_to_cell(pick[0], pick[1])
        if 0 <= col < self.heightmap.cols and 0 <= row < self.heightmap.rows:
            self.hovered_cell = (col, row)
            self.highlight.show_at(col, row, self.heightmap, self.brush_size)
            if self.clicking and self.last_edit != (col, row):
                self._apply_edit(col, row)
        else:
            self.hovered_cell = None
            self.highlight.hide()
        self._update_status()

    def tick(self):
        pass

    def _apply_edit(self, col, row):
        self.last_edit = (col, row)
        col_start = col - (self.brush_size - 1) // 2
        row_start = row - (self.brush_size - 1) // 2
        for c in range(col_start, col_start + self.brush_size):
            for r in range(row_start, row_start + self.brush_size):
                self._modify_cell(c, r)
        self.is_dirty = True
        self._rebuild_terrain()
        self._update_status()

    def _modify_cell(self, col, row):
        if not (0 <= col < self.heightmap.cols and 0 <= row < self.heightmap.rows):
            return
        h = self.heightmap.get(col, row)
        if self.edit_mode == "+":
            self.heightmap.set(col, row, h + 1)
        elif self.edit_mode == "-":
            self.heightmap.set(col, row, h - 1)
        elif self.edit_mode == "0":
            total = 0
            count = 0
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nc = col + dc
                    nr = row + dr
                    if 0 <= nc < self.heightmap.cols and 0 <= nr < self.heightmap.rows:
                        total += self.heightmap.get(nc, nr)
                        count += 1
            if count == 0:
                return
            avg = total / float(count)
            if h < avg:
                self.heightmap.set(col, row, h + 1)
            elif h > avg:
                self.heightmap.set(col, row, h - 1)

    def _rebuild_terrain(self):
        self.app.renderer.draw_terrain()
        self.app.renderer.draw_grid()
        self._draw_roads()

    def _build_road_splines(self):
        splines = []
        for chain in map_data.VISUAL_ROAD_CHAINS:
            node_coords = [map_data.NODES[node_name] for node_name in chain if node_name in map_data.NODES]
            if len(node_coords) >= 2:
                splines.append(generate_curvy_path_from_nodes(node_coords))
        return splines

    def _draw_roads(self):
        self.app.renderer.draw_roads(self.road_splines, color=(0.4, 0.4, 0.4, 1), width=2.0)

    def _undo(self):
        if not self.undo_stack:
            return
        self.app.heightmap.data = self.undo_stack.pop()
        self.is_dirty = bool(self.undo_stack)
        self._rebuild_terrain()
        self._update_status()

    def _save(self):
        path = self.heightmap.save_json()
        self._saved_files.append(os.path.basename(path))
        self.is_dirty = False
        self.undo_stack.clear()
        session_tracker.mark_save_occurred()
        self._update_base_status(extra=f"Saved to {os.path.basename(path)}")

    def save_elevation_data(self):
        self._save()

    @property
    def controls_text(self):
        return (
            "[+/1] Raise | [-/2] Lower | [0/3] Level | [Shift +/-] Brush Size | "
            "[Click] Edit | [S] Save | [Z/Ctrl+Z] Undo\nArrow keys pan | RMB orbit | Scroll zoom"
        )
