"""
Elevation Editor - 2.5D Terrain Editor using Panda3D
=====================================================
Standalone application for creating and editing terrain elevation maps.
Uses a smooth heightmap mesh (shared vertices, interpolated corners) for
natural-looking terrain — NOT blocky cubes like dorfdelf.

Designed with the same coordinate system as MAP/map_data.py so the
elevation data can later be merged with the existing 2D map.

Controls:
    +/= or 1    : Switch to Raise mode
    -   or 2    : Switch to Lower mode
    0   or 3    : Switch to Level mode (toward zero)
    Left Click  : Apply current mode to hovered cell
    WASD        : Pan camera
    Right Drag  : Orbit camera
    Scroll      : Zoom in/out
    Ctrl+S      : Save elevation data
    L           : Load elevation data
    ESC         : Quit
"""

import os
import sys
import json
import math
import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.task.Task import Task

from panda3d.core import (
    Point3, Vec3, Vec4, LColor, LVector3f,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode,
    CollisionTraverser, CollisionRay, CollisionNode,
    CollisionHandlerQueue, BitMask32, GeomEnums,
    DirectionalLight, AmbientLight,
    AntialiasAttrib, TextNode,
    WindowProperties, loadPrcFileData,
    NodePath, TransparencyAttrib,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Map coordinate system (matches MAP/map_data.py node positions)
MAP_ORIGIN_X = -650.0     # Slightly beyond min X (-600) for margin
MAP_ORIGIN_Y = -450.0     # Slightly beyond min Y (-390) for margin
CELL_SIZE    = 10.0       # Meters per grid cell
GRID_COLS    = 150        # Cells in X → covers -650 to 850
GRID_ROWS    = 140        # Cells in Y → covers -450 to 950

# Rendering
BG_COLOR            = (0.42, 0.42, 0.42, 1.0)
HEIGHT_COLOR_SCALE  = 0.035   # Grey shift per unit of height
BASE_GREY           = 0.50    # Grey at height 0

# Paths
ELEVATION_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR     = os.path.join(ELEVATION_DIR, "Saved_Elevation")
SAVE_FILE     = os.path.join(SAVED_DIR, "elevation_data.json")


# ─────────────────────────────────────────────────────────────────────────────
#  Heightmap
# ─────────────────────────────────────────────────────────────────────────────
class Heightmap:
    """
    Integer heightmap aligned with the MAP coordinate system.
    Stores one height value per grid cell.  Vertex heights (at cell corners)
    are interpolated from neighbouring cells for smooth rendering.
    """

    def __init__(self, cols=GRID_COLS, rows=GRID_ROWS,
                 cell_size=CELL_SIZE,
                 origin_x=MAP_ORIGIN_X, origin_y=MAP_ORIGIN_Y):
        self.cols = cols
        self.rows = rows
        self.cell_size = cell_size
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.data = np.zeros((rows, cols), dtype=np.int32)

    # --- accessors ---
    def get(self, col, row):
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return int(self.data[row, col])
        return 0

    def set(self, col, row, val):
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.data[row, col] = val

    def modify(self, col, row, mode):
        """Apply edit mode ('+', '-', '0') to a cell."""
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            return
        h = int(self.data[row, col])
        if mode == '+':
            self.data[row, col] = h + 1
        elif mode == '-':
            self.data[row, col] = h - 1
        elif mode == '0':
            if h > 0:
                self.data[row, col] = h - 1
            elif h < 0:
                self.data[row, col] = h + 1

    # --- coordinate helpers ---
    def world_to_cell(self, wx, wy):
        c = int(math.floor((wx - self.origin_x) / self.cell_size))
        r = int(math.floor((wy - self.origin_y) / self.cell_size))
        return c, r

    def cell_center_world(self, col, row):
        wx = self.origin_x + (col + 0.5) * self.cell_size
        wy = self.origin_y + (row + 0.5) * self.cell_size
        return wx, wy

    def vertex_height(self, vi, vj):
        """
        Smooth vertex height at corner (vi, vj).
        A corner is shared by up to 4 cells; average their heights.
        """
        total, count = 0.0, 0
        for dc, dr in ((-1, -1), (0, -1), (-1, 0), (0, 0)):
            c, r = vi + dc, vj + dr
            if 0 <= c < self.cols and 0 <= r < self.rows:
                total += float(self.data[r, c])
                count += 1
        return total / count if count else 0.0

    # --- persistence ---
    def save(self, path=None):
        path = path or SAVE_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blob = {
            "grid_cols": self.cols,
            "grid_rows": self.rows,
            "cell_size": self.cell_size,
            "origin": [self.origin_x, self.origin_y],
            "heights": self.data.tolist(),
        }
        with open(path, "w") as f:
            json.dump(blob, f)
        return path

    def load(self, path=None):
        path = path or SAVE_FILE
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            blob = json.load(f)
        self.cols = blob["grid_cols"]
        self.rows = blob["grid_rows"]
        self.cell_size = blob["cell_size"]
        self.origin_x, self.origin_y = blob["origin"]
        self.data = np.array(blob["heights"], dtype=np.int32)
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  Terrain Mesh
# ─────────────────────────────────────────────────────────────────────────────
class TerrainMesh:
    """
    Builds a smooth 3D mesh from a Heightmap using Panda3D geometry.
    Vertices sit at cell corners and share heights for smooth interpolation
    (the key difference from dorfdelf's blocky per-cell cubes).
    """

    def __init__(self, heightmap: Heightmap):
        self.hm = heightmap
        self.np = None          # NodePath of the mesh

    def build(self, parent: NodePath) -> NodePath:
        if self.np:
            self.np.removeNode()

        hm   = self.hm
        vcol = hm.cols + 1     # vertices in X
        vrow = hm.rows + 1     # vertices in Y

        fmt   = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData("terrain", fmt, Geom.UHDynamic)
        vdata.setNumRows(vcol * vrow)

        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")
        cw = GeomVertexWriter(vdata, "color")

        cs = hm.cell_size

        for vj in range(vrow):
            for vi in range(vcol):
                wx = hm.origin_x + vi * cs
                wy = hm.origin_y + vj * cs
                wz = hm.vertex_height(vi, vj) * cs
                vw.addData3f(wx, wy, wz)

                # Finite-difference normal
                hl = hm.vertex_height(vi - 1, vj) * cs
                hr = hm.vertex_height(vi + 1, vj) * cs
                hb = hm.vertex_height(vi, vj - 1) * cs
                hf = hm.vertex_height(vi, vj + 1) * cs
                nx = (hl - hr) / (2.0 * cs)
                ny = (hb - hf) / (2.0 * cs)
                nz = 1.0
                ln = math.sqrt(nx * nx + ny * ny + nz * nz)
                nw.addData3f(nx / ln, ny / ln, nz / ln)

                # Grey vertex colour
                h_val = hm.vertex_height(vi, vj)
                grey  = BASE_GREY + h_val * HEIGHT_COLOR_SCALE
                grey  = max(0.08, min(0.97, grey))
                cw.addData4f(grey, grey, grey, 1.0)

        tris = GeomTriangles(Geom.UHDynamic)
        for cj in range(hm.rows):
            for ci in range(hm.cols):
                v00 = cj * vcol + ci
                v10 = v00 + 1
                v01 = v00 + vcol
                v11 = v01 + 1
                tris.addVertices(v00, v10, v11)
                tris.addVertices(v00, v11, v01)
        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        gnode = GeomNode("terrain_mesh")
        gnode.addGeom(geom)

        self.np = parent.attachNewNode(gnode)
        self.np.setTwoSided(True)
        return self.np


# ─────────────────────────────────────────────────────────────────────────────
#  Cell Highlight
# ─────────────────────────────────────────────────────────────────────────────
class CellHighlight:
    """Translucent coloured quad hovering over the picked cell."""

    def __init__(self, parent: NodePath, cell_size: float):
        self.cs = cell_size
        self.np = self._make(parent)
        self.np.hide()

    def _make(self, parent):
        fmt   = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("hl", fmt, Geom.UHDynamic)
        vdata.setNumRows(4)
        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")
        for x, y in ((0, 0), (self.cs, 0), (self.cs, self.cs), (0, self.cs)):
            vw.addData3f(x, y, 0.0)
            cw.addData4f(0.3, 0.6, 1.0, 0.30)
        tris = GeomTriangles(Geom.UHStatic)
        tris.addVertices(0, 1, 2)
        tris.addVertices(0, 2, 3)
        tris.closePrimitive()
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        gn = GeomNode("cell_hl")
        gn.addGeom(geom)
        np = parent.attachNewNode(gn)
        np.setTransparency(TransparencyAttrib.MAlpha)
        np.setTwoSided(True)
        np.setBin("fixed", 40)
        np.setDepthWrite(False)
        return np

    def show_at(self, col, row, hm: Heightmap):
        wx = hm.origin_x + col * hm.cell_size
        wy = hm.origin_y + row * hm.cell_size
        wz = hm.get(col, row) * hm.cell_size + 0.5
        self.np.setPos(wx, wy, wz)
        self.np.show()

    def hide(self):
        self.np.hide()


# ─────────────────────────────────────────────────────────────────────────────
#  Camera Controller  (pattern from dorfdelf camera.py)
# ─────────────────────────────────────────────────────────────────────────────
class CameraController:
    """WASD pan, right-drag orbit, scroll zoom."""

    def __init__(self, app: ShowBase, center: Point3, dist: float = 800):
        self.app      = app
        self.mouse    = app.mouseWatcherNode
        self.body     = app.render.attachNewNode("cam_body")
        self.body.setPos(center)
        self.dist     = dist
        self.heading  = 45.0
        self.pitch    = -40.0
        self.keys     = {k: False for k in ("w", "s", "a", "d")}
        self.drag     = None
        self.moving   = False

        app.camera.reparentTo(self.body)
        self._apply()

        for k in self.keys:
            app.accept(k, self._key, [k, True])
            app.accept(f"{k}-up", self._key, [k, False])
        app.accept("mouse3",    self._drag_start)
        app.accept("mouse3-up", self._drag_end)
        app.accept("wheel_up",  self._zoom, [-1])
        app.accept("wheel_down",self._zoom, [1])
        app.taskMgr.add(self._tick, "cam_tick")

    def _key(self, k, down):
        self.keys[k] = down

    def _drag_start(self):
        if self.mouse.hasMouse():
            m = self.mouse.getMouse()
            self.drag = (m.getX(), m.getY(), self.heading, self.pitch)

    def _drag_end(self):
        self.drag = None

    def _zoom(self, d):
        self.dist *= 1.12 ** d
        self.dist = max(40, min(6000, self.dist))
        self._apply()

    def _apply(self):
        rh = math.radians(self.heading)
        rp = math.radians(self.pitch)
        cp = math.cos(rp)
        x  = -self.dist * math.sin(rh) * cp
        y  = -self.dist * math.cos(rh) * cp
        z  = -self.dist * math.sin(rp)
        self.app.camera.setPos(x, y, z)
        self.app.camera.lookAt(self.body)

    def _tick(self, task):
        dt = globalClock.getDt()
        spd = self.dist * 0.7
        self.moving = any(self.keys.values()) or self.drag is not None
        rh = math.radians(self.heading)
        sin_h, cos_h = math.sin(rh), math.cos(rh)
        if self.keys["w"]:
            p = self.body.getPos()
            self.body.setPos(p.x + dt * spd * sin_h, p.y + dt * spd * cos_h, p.z)
        if self.keys["s"]:
            p = self.body.getPos()
            self.body.setPos(p.x - dt * spd * sin_h, p.y - dt * spd * cos_h, p.z)
        if self.keys["a"]:
            p = self.body.getPos()
            self.body.setPos(p.x - dt * spd * cos_h, p.y + dt * spd * sin_h, p.z)
        if self.keys["d"]:
            p = self.body.getPos()
            self.body.setPos(p.x + dt * spd * cos_h, p.y - dt * spd * sin_h, p.z)
        if self.drag and self.mouse.hasMouse():
            m = self.mouse.getMouse()
            sx, sy, sh, sp = self.drag
            self.heading = sh + (sx - m.getX()) * 150
            self.pitch   = max(-89, min(-5, sp - (sy - m.getY()) * 100))
            self._apply()
        return Task.cont


# ─────────────────────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────────────────────
class ElevationEditor(ShowBase):

    def __init__(self):
        loadPrcFileData("", "window-title Elevation Editor")
        loadPrcFileData("", "win-size 1400 900")

        super().__init__()
        self.setBackgroundColor(*BG_COLOR)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.disableMouse()

        # ── data ──
        self.heightmap = Heightmap()
        self.terrain   = TerrainMesh(self.heightmap)
        self.terrain.build(self.render)

        # ── lighting ──
        self._init_lights()

        # ── camera ──
        cx = self.heightmap.origin_x + self.heightmap.cols * self.heightmap.cell_size / 2
        cy = self.heightmap.origin_y + self.heightmap.rows * self.heightmap.cell_size / 2
        self.cam_ctrl = CameraController(self, Point3(cx, cy, 0), dist=900)

        # ── picker ──
        self._init_picker()

        # ── highlight ──
        self.highlight = CellHighlight(self.render, self.heightmap.cell_size)

        # ── HUD ──
        self._init_hud()

        # ── state ──
        self.edit_mode    = "+"
        self.hovered_cell = None
        self.clicking      = False
        self.last_edit     = None   # prevent repeat on hold

        # ── input ──
        self.accept("escape", sys.exit)
        for key in ("=", "+", "1"):
            self.accept(key, self._mode, ["+"])
        for key in ("-", "2"):
            self.accept(key, self._mode, ["-"])
        for key in ("0", "3"):
            self.accept(key, self._mode, ["0"])
        self.accept("mouse1",    self._click)
        self.accept("mouse1-up", self._release)
        self.accept("control-s", self._save)
        self.accept("l", self._load)

        self.taskMgr.add(self._pick, "pick")

    # ── lighting ─────────────────────────────────────────────────────────
    def _init_lights(self):
        dl = DirectionalLight("sun")
        dl.setColor(LColor(0.85, 0.85, 0.85, 1))
        dn = self.render.attachNewNode(dl)
        dn.setHpr(45, -45, 0)
        self.render.setLight(dn)

        dl2 = DirectionalLight("fill")
        dl2.setColor(LColor(0.3, 0.3, 0.3, 1))
        dn2 = self.render.attachNewNode(dl2)
        dn2.setHpr(-135, -30, 0)
        self.render.setLight(dn2)

        al = AmbientLight("amb")
        al.setColor(LColor(0.25, 0.25, 0.25, 1))
        self.render.setLight(self.render.attachNewNode(al))

    # ── picker ───────────────────────────────────────────────────────────
    def _init_picker(self):
        self.cTrav  = CollisionTraverser()
        self.pQueue = CollisionHandlerQueue()

        cn = CollisionNode("mray")
        cn.setFromCollideMask(BitMask32.bit(1))
        cn.setIntoCollideMask(BitMask32.allOff())
        self.pRay = CollisionRay()
        cn.addSolid(self.pRay)
        self.pNP = self.camera.attachNewNode(cn)
        self.cTrav.addCollider(self.pNP, self.pQueue)

        if self.terrain.np:
            self.terrain.np.node().setIntoCollideMask(BitMask32.bit(1))

    # ── HUD ──────────────────────────────────────────────────────────────
    def _init_hud(self):
        common = dict(
            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.8),
            mayChange=True, parent=self.aspect2d,
        )
        self.txt_mode = OnscreenText(
            text="Mode: + (Raise)", pos=(-1.3, 0.92), scale=0.06,
            align=TextNode.ALeft, **common,
        )
        self.txt_height = OnscreenText(
            text="Height: 0", pos=(1.3, 0.92), scale=0.07,
            align=TextNode.ARight, **common,
        )
        self.txt_cell = OnscreenText(
            text="", pos=(1.3, 0.84), scale=0.045,
            fg=(0.8, 0.8, 0.8, 1), shadow=(0, 0, 0, 0.6),
            align=TextNode.ARight, mayChange=True, parent=self.aspect2d,
        )
        self.txt_status = OnscreenText(
            text="", pos=(0, -0.95), scale=0.045,
            fg=(0.9, 0.9, 0.5, 1), shadow=(0, 0, 0, 0.6),
            align=TextNode.ACenter, mayChange=True, parent=self.aspect2d,
        )
        controls = (
            "[+/1] Raise  [-/2] Lower  [0/3] Level  "
            "[Click] Edit  [WASD] Pan  [RMB] Orbit  "
            "[Scroll] Zoom  [Ctrl+S] Save  [L] Load  [Esc] Quit"
        )
        OnscreenText(
            text=controls, pos=(0, -0.88), scale=0.038,
            fg=(0.7, 0.7, 0.7, 1), shadow=(0, 0, 0, 0.5),
            align=TextNode.ACenter, parent=self.aspect2d,
        )

    # ── mode ─────────────────────────────────────────────────────────────
    def _mode(self, m):
        self.edit_mode = m
        labels = {"+": "+ (Raise)", "-": "- (Lower)", "0": "0 (Level)"}
        self.txt_mode.setText(f"Mode: {labels[m]}")
        self._flash(f"Switched to {labels[m]} mode")

    def _flash(self, txt):
        self.txt_status.setText(txt)
        self.taskMgr.remove("clr_st")
        self.taskMgr.doMethodLater(
            3.0, lambda t: self.txt_status.setText(""), "clr_st"
        )

    # ── picking ──────────────────────────────────────────────────────────
    def _pick(self, task):
        if not self.mouseWatcherNode.hasMouse():
            self.highlight.hide()
            self.hovered_cell = None
            return Task.cont

        if self.cam_ctrl.moving and not self.cam_ctrl.drag:
            pass  # allow picking while WASD

        mp = self.mouseWatcherNode.getMouse()
        self.pRay.setFromLens(self.camNode, mp.getX(), mp.getY())
        self.cTrav.traverse(self.render)

        if self.pQueue.getNumEntries() > 0:
            self.pQueue.sortEntries()
            pt = self.pQueue.getEntry(0).getSurfacePoint(self.render)
            col, row = self.heightmap.world_to_cell(pt.getX(), pt.getY())
            if 0 <= col < self.heightmap.cols and 0 <= row < self.heightmap.rows:
                self.hovered_cell = (col, row)
                self.highlight.show_at(col, row, self.heightmap)
                h = self.heightmap.get(col, row)
                sign = "+" if h > 0 else ""
                self.txt_height.setText(f"Height: {sign}{h}")
                self.txt_cell.setText(f"Cell: ({col}, {row})")

                # continuous editing while holding click
                if self.clicking and (col, row) != self.last_edit:
                    self._apply_edit(col, row)
            else:
                self.highlight.hide()
                self.hovered_cell = None
        else:
            self.highlight.hide()
            self.hovered_cell = None

        return Task.cont

    # ── editing ──────────────────────────────────────────────────────────
    def _click(self):
        self.clicking = True
        self.last_edit = None
        if self.hovered_cell:
            self._apply_edit(*self.hovered_cell)

    def _release(self):
        self.clicking = False
        self.last_edit = None

    def _apply_edit(self, col, row):
        self.heightmap.modify(col, row, self.edit_mode)
        self.last_edit = (col, row)
        self._rebuild_terrain()
        h = self.heightmap.get(col, row)
        sign = "+" if h > 0 else ""
        self.txt_height.setText(f"Height: {sign}{h}")

    def _rebuild_terrain(self):
        self.terrain.build(self.render)
        if self.terrain.np:
            self.terrain.np.node().setIntoCollideMask(BitMask32.bit(1))

    # ── save / load ──────────────────────────────────────────────────────
    def _save(self):
        path = self.heightmap.save()
        self._flash(f"Saved to {os.path.basename(path)}")

    def _load(self):
        if self.heightmap.load():
            self._rebuild_terrain()
            self._flash("Loaded elevation data")
        else:
            self._flash("No saved elevation data found")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ElevationEditor()
    app.run()
