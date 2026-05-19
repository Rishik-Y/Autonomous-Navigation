from direct.showbase.ShowBase import ShowBase

from panda_common import CameraController, Picker, SceneRenderer
from waypoint_viewer import WaypointViewerMode


class TestWaypointViewerApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.95, 0.95, 0.95, 1)
        self.camera_controller = CameraController(self)
        self.picker = Picker(self)
        self.renderer = SceneRenderer(self)
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
