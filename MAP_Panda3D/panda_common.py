import math
from dataclasses import dataclass

import numpy as np

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
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    DirectionalLight,
    AmbientLight,
    LColor,
    LineSegs,
    Point3,
)

from panda_elevation import Heightmap, TerrainMesh

METERS_TO_PIXELS = 6.0
POINTS_PER_SEGMENT = 20
ROAD_WIDTH_M = 8.0

GRID_Z_OFFSET = -0.2
ROAD_Z_OFFSET = 0.7
NODE_Z_OFFSET = 0.6


def catmull_rom_point(t, p0, p1, p2, p3):
    # Standard uniform Catmull-Rom cubic interpolation between control points.
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
    world_xyz: np.ndarray | None

    @property
    def world_xy(self):
        if self.world_xyz is None:
            return None
        return self.world_xyz[:2]


class CameraController:
    """Arrow-key pan, right-drag orbit, scroll zoom."""

    def __init__(self, app, center=Point3(0, 0, 0), dist=1400):
        self.app = app
        self.mouse = app.mouseWatcherNode
        self.body = app.render.attachNewNode("cam_body")
        self.body.setPos(center)
        self.dist = dist
        self.heading = 45.0
        self.pitch = -55.0
        self.keys = {k: False for k in ("arrow_up", "arrow_down", "arrow_left", "arrow_right")}
        self.drag = None
        self.pan_enabled = True

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

    def set_pan_enabled(self, enabled):
        self.pan_enabled = bool(enabled)
        if not self.pan_enabled:
            for key in self.keys:
                self.keys[key] = False

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
        if self.pan_enabled:
            if self.keys["arrow_up"]:
                p = self.body.getPos()
                self.body.setPos(p.x + dt * speed * sin_h, p.y + dt * speed * cos_h, p.z)
            if self.keys["arrow_down"]:
                p = self.body.getPos()
                self.body.setPos(p.x - dt * speed * sin_h, p.y - dt * speed * cos_h, p.z)
            if self.keys["arrow_left"]:
                p = self.body.getPos()
                self.body.setPos(p.x - dt * speed * cos_h, p.y + dt * speed * sin_h, p.z)
            if self.keys["arrow_right"]:
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
    def __init__(self, app, heightmap=None, terrain_np_getter=None):
        self.app = app
        self.heightmap = heightmap
        self.terrain_np_getter = terrain_np_getter

        self.cTrav = CollisionTraverser()
        self.pQueue = CollisionHandlerQueue()

        cn = CollisionNode("picker_ray")
        cn.setFromCollideMask(BitMask32.bit(1))
        cn.setIntoCollideMask(BitMask32.allOff())
        self.pRay = CollisionRay()
        cn.addSolid(self.pRay)
        self.pNP = self.app.camera.attachNewNode(cn)
        self.cTrav.addCollider(self.pNP, self.pQueue)

    def pick_surface(self):
        if not self.app.mouseWatcherNode.hasMouse():
            return PickResult(None)
        mp = self.app.mouseWatcherNode.getMouse()
        self.pRay.setFromLens(self.app.camNode, mp.getX(), mp.getY())
        self.cTrav.traverse(self.app.render)
        if self.pQueue.getNumEntries() <= 0:
            return PickResult(None)
        self.pQueue.sortEntries()
        pt = self.pQueue.getEntry(0).getSurfacePoint(self.app.render)
        return PickResult(np.array([float(pt.x), float(pt.y), float(pt.z)]))

    def pick_ground(self):
        return self.pick_surface()

    def get_elevation_at(self, world_x, world_y):
        if self.heightmap is None:
            return 0.0
        return float(self.heightmap.get_height_at_world(world_x, world_y))


class SceneRenderer:
    def __init__(self, app, heightmap=None, terrain=None):
        self.app = app
        self.heightmap = heightmap or Heightmap()
        self.terrain = terrain or TerrainMesh(self.heightmap)

        self.root = app.render.attachNewNode("map_scene")
        self.terrain_np = None
        self.grid_np = self.root.attachNewNode("grid")
        self.road_np = self.root.attachNewNode("roads")
        self.node_np = self.root.attachNewNode("nodes")
        self.overlay_np = self.root.attachNewNode("overlay")
        self._sphere_model = self.app.loader.loadModel("models/misc/sphere")

        self._init_lights()
        self.draw_terrain()

    def _init_lights(self):
        dl = DirectionalLight("terrain_sun")
        dl.setColor(LColor(0.85, 0.85, 0.85, 1))
        dn = self.app.render.attachNewNode(dl)
        dn.setHpr(45, -45, 0)
        self.app.render.setLight(dn)

        al = AmbientLight("terrain_amb")
        al.setColor(LColor(0.25, 0.25, 0.25, 1))
        an = self.app.render.attachNewNode(al)
        self.app.render.setLight(an)

    def get_terrain_np(self):
        return self.terrain_np

    def terrain_elevation(self, x, y):
        return float(self.heightmap.get_height_at_world(float(x), float(y)))

    def draw_terrain(self):
        self.terrain_np = self.terrain.build(self.root)
        self.terrain_np.node().setIntoCollideMask(BitMask32.bit(1))
        self.terrain_np.setBin("background", 0)
        return self.terrain_np

    def clear_overlay(self):
        self.overlay_np.getChildren().detach()

    def draw_grid(self, min_v=None, max_v=None, step=50, sample_step=20):
        self.grid_np.removeNode()
        self.grid_np = self.root.attachNewNode("grid")

        segs = LineSegs("grid")
        segs.setColor(0.75, 0.75, 0.75, 0.9)
        segs.setThickness(1)

        if min_v is None or max_v is None:
            min_x = int(round(self.heightmap.origin_x))
            max_x = int(round(self.heightmap.origin_x + self.heightmap.cols * self.heightmap.cell_size))
            min_y = int(round(self.heightmap.origin_y))
            max_y = int(round(self.heightmap.origin_y + self.heightmap.rows * self.heightmap.cell_size))
        else:
            min_x = int(min_v)
            max_x = int(max_v)
            min_y = int(min_v)
            max_y = int(max_v)

        for x in range(min_x, max_x + 1, step):
            prev = None
            for y in range(min_y, max_y + 1, sample_step):
                z = self.terrain_elevation(x, y) + GRID_Z_OFFSET
                p = (float(x), float(y), float(z))
                if prev is None:
                    prev = p
                    continue
                segs.moveTo(*prev)
                segs.drawTo(*p)
                prev = p

        for y in range(min_y, max_y + 1, step):
            prev = None
            for x in range(min_x, max_x + 1, sample_step):
                z = self.terrain_elevation(x, y) + GRID_Z_OFFSET
                p = (float(x), float(y), float(z))
                if prev is None:
                    prev = p
                    continue
                segs.moveTo(*prev)
                segs.drawTo(*p)
                prev = p

        self.grid_np.attachNewNode(segs.create())

    def draw_roads(self, splines, color=(0.4, 0.4, 0.4, 1), width=2.0, z=ROAD_Z_OFFSET):
        self.road_np.removeNode()
        self.road_np = self.root.attachNewNode("roads")
        segs = LineSegs("roads")
        segs.setColor(*color)
        segs.setThickness(width)

        for waypoints in splines:
            if len(waypoints) < 2:
                continue
            x0 = float(waypoints[0][0])
            y0 = float(waypoints[0][1])
            z0 = self.terrain_elevation(x0, y0) + float(z)
            segs.moveTo(x0, y0, z0)
            for p in waypoints[1:]:
                x = float(p[0])
                y = float(p[1])
                zz = self.terrain_elevation(x, y) + float(z)
                segs.drawTo(x, y, zz)

        self.road_np.attachNewNode(segs.create())

    def _point_cloud(self, points, colors, size=7, z=NODE_Z_OFFSET):
        fmt = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("points", fmt, Geom.UHStatic)
        vdata.setNumRows(len(points))
        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")

        for p, c in zip(points, colors):
            x = float(p[0])
            y = float(p[1])
            zz = self.terrain_elevation(x, y) + float(z)
            vw.addData3f(x, y, zz)
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

    def draw_nodes(self, nodes, load_zones, dump_zones, fuel_zones, highlighted_node=None):
        self.node_np.removeNode()
        self.node_np = self.root.attachNewNode("nodes")
        base_scale = 4.0 / METERS_TO_PIXELS
        for name, pos in nodes.items():
            if name in load_zones:
                color = (0.0, 0.8, 0.0, 1.0)
            elif name in dump_zones:
                color = (0.9, 0.1, 0.1, 1.0)
            elif name in fuel_zones:
                color = (1.0, 0.6, 0.0, 1.0)
            else:
                color = (0.6, 0.0, 0.7, 1.0)
            if name == highlighted_node:
                color = (1.0, 0.9, 0.1, 1.0)
            px = float(pos[0])
            py = float(pos[1])
            pz = self.terrain_elevation(px, py) + NODE_Z_OFFSET
            sphere = self._sphere_model.copyTo(self.node_np)
            sphere.setPos(px, py, pz)
            sphere.setScale(base_scale * (1.2 if name == highlighted_node else 1.0))
            sphere.setColor(*color)

    def destroy(self):
        self.root.removeNode()
