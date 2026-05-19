import math
from dataclasses import dataclass

import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.task.Task import Task
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
    Geom,
    GeomNode,
    GeomPoints,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    NodePath,
    Plane,
    Point2,
    Point3,
    TextNode,
    Vec3,
)

METERS_TO_PIXELS = 6.0
POINTS_PER_SEGMENT = 20
ROAD_WIDTH_M = 8.0


def catmull_rom_point(t, p0, p1, p2, p3):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t ** 2) + (-p0 + 3 * p1 - 3 * p2 + p3) * (t ** 3))


def generate_curvy_path_from_nodes(node_list, points_per_segment=POINTS_PER_SEGMENT):
    all_waypoints_m = []
    if not node_list or len(node_list) < 2:
        return []
    for i in range(len(node_list) - 1):
        p0 = node_list[0] if i == 0 else node_list[i - 1]
        p1 = node_list[i]
        p2 = node_list[i + 1]
        p3 = node_list[-1] if i >= len(node_list) - 2 else node_list[i + 2]
        if i == 0:
            all_waypoints_m.append(p1)
        for j in range(1, points_per_segment + 1):
            t = j / float(points_per_segment)
            point = catmull_rom_point(t, p0, p1, p2, p3)
            all_waypoints_m.append(point)
    return all_waypoints_m


@dataclass
class PickResult:
    world_xy: np.ndarray | None


class CameraController:
    """WASD pan, right-drag orbit, scroll zoom (Elevation pattern)."""

    def __init__(self, app, center=Point3(0, 0, 0), dist=1400):
        self.app = app
        self.mouse = app.mouseWatcherNode
        self.body = app.render.attachNewNode("cam_body")
        self.body.setPos(center)
        self.dist = dist
        self.heading = 45.0
        self.pitch = -55.0
        self.keys = {k: False for k in ("w", "s", "a", "d")}
        self.drag = None

        app.camera.reparentTo(self.body)
        self._apply()

        for k in self.keys:
            app.accept(k, self._set_key, [k, True])
            app.accept(f"{k}-up", self._set_key, [k, False])
        app.accept("mouse3", self._drag_start)
        app.accept("mouse3-up", self._drag_end)
        app.accept("wheel_up", self._zoom, [-1])
        app.accept("wheel_down", self._zoom, [1])
        app.taskMgr.add(self._tick, "cam_tick")

    def _set_key(self, key, down):
        self.keys[key] = down

    def _drag_start(self):
        if self.mouse.hasMouse():
            m = self.mouse.getMouse()
            self.drag = (m.getX(), m.getY(), self.heading, self.pitch)

    def _drag_end(self):
        self.drag = None

    def _zoom(self, direction):
        self.dist *= 1.12 ** direction
        self.dist = max(60, min(9000, self.dist))
        self._apply()

    def _apply(self):
        rh = math.radians(self.heading)
        rp = math.radians(self.pitch)
        cp = math.cos(rp)
        x = -self.dist * math.sin(rh) * cp
        y = -self.dist * math.cos(rh) * cp
        z = -self.dist * math.sin(rp)
        self.app.camera.setPos(x, y, z)
        self.app.camera.lookAt(self.body)

    def _tick(self, task):
        dt = globalClock.getDt()
        speed = self.dist * 0.7
        rh = math.radians(self.heading)
        sin_h = math.sin(rh)
        cos_h = math.cos(rh)
        if self.keys["w"]:
            p = self.body.getPos()
            self.body.setPos(p.x + dt * speed * sin_h, p.y + dt * speed * cos_h, p.z)
        if self.keys["s"]:
            p = self.body.getPos()
            self.body.setPos(p.x - dt * speed * sin_h, p.y - dt * speed * cos_h, p.z)
        if self.keys["a"]:
            p = self.body.getPos()
            self.body.setPos(p.x - dt * speed * cos_h, p.y + dt * speed * sin_h, p.z)
        if self.keys["d"]:
            p = self.body.getPos()
            self.body.setPos(p.x + dt * speed * cos_h, p.y - dt * speed * sin_h, p.z)
        if self.drag and self.mouse.hasMouse():
            m = self.mouse.getMouse()
            sx, sy, sh, sp = self.drag
            self.heading = sh + (sx - m.getX()) * 150
            self.pitch = max(-89, min(-5, sp - (sy - m.getY()) * 100))
            self._apply()
        return Task.cont


class Picker:
    def __init__(self, app):
        self.app = app
        self.plane = Plane(Vec3(0, 0, 1), Point3(0, 0, 0))

    def pick_ground(self):
        if not self.app.mouseWatcherNode.hasMouse():
            return PickResult(None)
        mp = self.app.mouseWatcherNode.getMouse()
        p_from = Point3()
        p_to = Point3()
        self.app.camLens.extrude(mp, p_from, p_to)
        p_from = self.app.render.getRelativePoint(self.app.cam, p_from)
        p_to = self.app.render.getRelativePoint(self.app.cam, p_to)
        direction = p_to - p_from
        hit = Point3()
        if self.plane.intersectsLine(hit, p_from, p_to):
            return PickResult(np.array([hit.x, hit.y]))
        return PickResult(None)


class SceneRenderer:
    def __init__(self, app):
        self.app = app
        self.root = app.render.attachNewNode("map_scene")
        self.grid_np = self.root.attachNewNode("grid")
        self.road_np = self.root.attachNewNode("roads")
        self.node_np = self.root.attachNewNode("nodes")
        self.overlay_np = self.root.attachNewNode("overlay")
        self._labels = []

    def clear_overlay(self):
        self.overlay_np.getChildren().detach()

    def draw_grid(self, min_v=-1000, max_v=1000, step=50):
        self.grid_np.removeNode()
        self.grid_np = self.root.attachNewNode("grid")
        segs = LineSegs("grid")
        segs.setColor(0.9, 0.9, 0.9, 1)
        segs.setThickness(1)
        for x in range(min_v, max_v + 1, step):
            segs.moveTo(x, min_v, 0)
            segs.drawTo(x, max_v, 0)
        for y in range(min_v, max_v + 1, step):
            segs.moveTo(min_v, y, 0)
            segs.drawTo(max_v, y, 0)
        self.grid_np.attachNewNode(segs.create())

    def draw_roads(self, splines, color=(0.4, 0.4, 0.4, 1), width=2.0, z=0.2):
        self.road_np.removeNode()
        self.road_np = self.root.attachNewNode("roads")
        segs = LineSegs("roads")
        segs.setColor(*color)
        segs.setThickness(width)
        for waypoints in splines:
            if len(waypoints) < 2:
                continue
            segs.moveTo(float(waypoints[0][0]), float(waypoints[0][1]), z)
            for p in waypoints[1:]:
                segs.drawTo(float(p[0]), float(p[1]), z)
        self.road_np.attachNewNode(segs.create())

    def _point_cloud(self, points, colors, size=7, z=1.0):
        fmt = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("points", fmt, Geom.UHStatic)
        vdata.setNumRows(len(points))
        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")
        for p, c in zip(points, colors):
            vw.addData3f(float(p[0]), float(p[1]), z)
            cw.addData4f(*c)
        prim = GeomPoints(Geom.UHStatic)
        for i in range(len(points)):
            prim.addVertex(i)
        prim.closePrimitive()
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode("points")
        node.addGeom(geom)
        np_node = self.node_np.attachNewNode(node)
        np_node.setRenderModeThickness(size)

    def draw_nodes(self, nodes, load_zones, dump_zones, fuel_zones, show_names=False):
        self.node_np.removeNode()
        self.node_np = self.root.attachNewNode("nodes")
        self._clear_labels()
        pts = []
        cols = []
        for name, pos in nodes.items():
            pts.append(pos)
            if name in load_zones:
                cols.append((0.0, 0.8, 0.0, 1.0))
            elif name in dump_zones:
                cols.append((0.9, 0.1, 0.1, 1.0))
            elif name in fuel_zones:
                cols.append((1.0, 0.6, 0.0, 1.0))
            else:
                cols.append((0.6, 0.0, 0.7, 1.0))
        self._point_cloud(pts, cols)
        if show_names:
            for name, pos in nodes.items():
                txt = OnscreenText(text=name, pos=(0, 0), scale=0.035, fg=(0, 0, 0, 1), mayChange=False, parent=self.app.aspect2d, align=TextNode.ALeft)
                txt.setBin("fixed", 120)
                txt.node_path = self.root.attachNewNode(name)
                txt.node_path.setPos(float(pos[0]), float(pos[1]), 3.0)
                self._labels.append(txt)

    def update_labels(self):
        for txt in self._labels:
            p3 = txt.node_path.getPos(self.app.render)
            p2 = Point2()
            if self.app.camLens.project(p3, p2):
                txt.setPos((p2.x, p2.y))
            else:
                txt.setPos((3, 3))

    def _clear_labels(self):
        for txt in self._labels:
            if hasattr(txt, "node_path"):
                txt.node_path.removeNode()
            txt.destroy()
        self._labels = []

    def destroy(self):
        self._clear_labels()
        self.root.removeNode()
