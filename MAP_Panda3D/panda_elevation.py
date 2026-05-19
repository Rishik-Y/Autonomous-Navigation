import json
import math
import os

import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)

MAP_ORIGIN_X = -650.0
MAP_ORIGIN_Y = -450.0
CELL_SIZE = 10.0
GRID_COLS = 150
GRID_ROWS = 140

HEIGHT_COLOR_SCALE = 0.035
BASE_GREY = 0.50

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))


class Heightmap:
    """Integer heightmap aligned with MAP coordinates."""

    ELEVATION_DIR = os.path.join(_REPO_ROOT, "Elevation", "Saved_Elevation")
    DEFAULT_SAVE = "elevation_data.json"

    def __init__(
        self,
        cols=GRID_COLS,
        rows=GRID_ROWS,
        cell_size=CELL_SIZE,
        origin_x=MAP_ORIGIN_X,
        origin_y=MAP_ORIGIN_Y,
    ):
        self.cols = int(cols)
        self.rows = int(rows)
        self.cell_size = float(cell_size)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.data = np.zeros((self.rows, self.cols), dtype=np.int32)

    def get(self, col, row):
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return int(self.data[row, col])
        return 0

    def set(self, col, row, val):
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.data[row, col] = int(val)

    def vertex_height(self, vi, vj):
        total = 0.0
        count = 0
        for dc, dr in ((-1, -1), (0, -1), (-1, 0), (0, 0)):
            c, r = vi + dc, vj + dr
            if 0 <= c < self.cols and 0 <= r < self.rows:
                total += float(self.data[r, c])
                count += 1
        return total / count if count else 0.0

    def world_to_cell(self, wx, wy):
        col = int(math.floor((wx - self.origin_x) / self.cell_size))
        row = int(math.floor((wy - self.origin_y) / self.cell_size))
        return col, row

    def cell_center_world(self, col, row):
        wx = self.origin_x + (col + 0.5) * self.cell_size
        wy = self.origin_y + (row + 0.5) * self.cell_size
        return wx, wy

    def get_height_at_world(self, wx, wy):
        gx = (wx - self.origin_x) / self.cell_size
        gy = (wy - self.origin_y) / self.cell_size
        gx = max(0.0, min(float(self.cols), gx))
        gy = max(0.0, min(float(self.rows), gy))

        x0 = int(math.floor(gx))
        y0 = int(math.floor(gy))
        x1 = min(self.cols, x0 + 1)
        y1 = min(self.rows, y0 + 1)
        tx = gx - x0
        ty = gy - y0

        h00 = self.vertex_height(x0, y0)
        h10 = self.vertex_height(x1, y0)
        h01 = self.vertex_height(x0, y1)
        h11 = self.vertex_height(x1, y1)

        h0 = (1.0 - tx) * h00 + tx * h10
        h1 = (1.0 - tx) * h01 + tx * h11
        h = (1.0 - ty) * h0 + ty * h1
        return h * self.cell_size

    @classmethod
    def _default_path(cls):
        return os.path.join(cls.ELEVATION_DIR, cls.DEFAULT_SAVE)

    def save_json(self, path=None):
        path = path or self._default_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blob = {
            "grid_cols": self.cols,
            "grid_rows": self.rows,
            "cell_size": self.cell_size,
            "origin": [self.origin_x, self.origin_y],
            "heights": self.data.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f)
        return path

    def load_json(self, path=None):
        path = path or self._default_path()
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        self.cols = int(blob["grid_cols"])
        self.rows = int(blob["grid_rows"])
        self.cell_size = float(blob["cell_size"])
        self.origin_x, self.origin_y = blob["origin"]
        self.data = np.array(blob["heights"], dtype=np.int32)
        return True


class TerrainMesh:
    """Builds smooth 3D terrain mesh from Heightmap."""

    def __init__(self, heightmap: Heightmap):
        self.hm = heightmap
        self.np = None

    def build(self, parent: NodePath):
        if self.np:
            self.np.removeNode()

        hm = self.hm
        vcol = hm.cols + 1
        vrow = hm.rows + 1
        cs = hm.cell_size

        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData("terrain", fmt, Geom.UHDynamic)
        vdata.setNumRows(vcol * vrow)

        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")
        cw = GeomVertexWriter(vdata, "color")

        for vj in range(vrow):
            for vi in range(vcol):
                wx = hm.origin_x + vi * cs
                wy = hm.origin_y + vj * cs
                wz = hm.vertex_height(vi, vj) * cs
                vw.addData3f(wx, wy, wz)

                hl = hm.vertex_height(vi - 1, vj) * cs
                hr = hm.vertex_height(vi + 1, vj) * cs
                hb = hm.vertex_height(vi, vj - 1) * cs
                hf = hm.vertex_height(vi, vj + 1) * cs
                nx = (hl - hr) / (2.0 * cs)
                ny = (hb - hf) / (2.0 * cs)
                nz = 1.0
                ln = math.sqrt(nx * nx + ny * ny + nz * nz)
                nw.addData3f(nx / ln, ny / ln, nz / ln)

                h_val = hm.vertex_height(vi, vj)
                grey = BASE_GREY + h_val * HEIGHT_COLOR_SCALE
                grey = max(0.08, min(0.97, grey))
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
