from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3

from panda_common import CameraController, Picker, SceneRenderer
from waypoint_viewer import WaypointViewerMode


class TestWaypointViewerApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.95, 0.95, 0.95, 1)
        self.renderer = SceneRenderer(self)
        hm = self.renderer.heightmap
        cx = hm.origin_x + hm.cols * hm.cell_size * 0.5
        cy = hm.origin_y + hm.rows * hm.cell_size * 0.5
        cz = hm.get_height_at_world(cx, cy)
        self.camera_controller = CameraController(self, center=Point3(cx, cy, cz), dist=1400)
        self.picker = Picker(self, heightmap=self.renderer.heightmap, terrain_np_getter=self.renderer.get_terrain_np)
        self.mode = WaypointViewerMode(self)
        self.mode.activate()

        self.accept("space", self.mode.on_key, ["space"])
        self.taskMgr.add(self._tick, "test_tick")

    def _tick(self, task):
        self.mode.tick()
        return task.cont


if __name__ == "__main__":
    app = TestWaypointViewerApp()
    app.run()
