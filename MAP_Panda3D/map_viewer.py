import map_data
from panda_common import generate_curvy_path_from_nodes


class MapViewerMode:
    label = "Map Viewer"

    def __init__(self, app):
        self.app = app
        self.PRE_CALCULATED_SPLINES = []
        self.status_text = "Read-only map viewer"

    def activate(self):
        self.PRE_CALCULATED_SPLINES = []
        for chain in map_data.VISUAL_ROAD_CHAINS:
            node_coords = [map_data.NODES[node_name] for node_name in chain if node_name in map_data.NODES]
            if len(node_coords) >= 2:
                self.PRE_CALCULATED_SPLINES.append(generate_curvy_path_from_nodes(node_coords))
        self.redraw()

    def deactivate(self):
        pass

    def redraw(self):
        self.app.renderer.draw_grid()
        self.app.renderer.draw_roads(self.PRE_CALCULATED_SPLINES, color=(0.4, 0.4, 0.4, 1), width=2.0)
        self.app.renderer.draw_nodes(map_data.NODES, map_data.LOAD_ZONES, map_data.DUMP_ZONES, map_data.FUEL_ZONES)

    def on_key(self, key):
        pass

    def on_mouse1(self, down=True):
        pass

    def on_mouse_move(self):
        pass

    def tick(self):
        pass

    @property
    def controls_text(self):
        return "Arrow pan | RMB orbit | Scroll zoom"


def run_viewer():
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import Point3
    from panda_common import CameraController, Picker, SceneRenderer

    class _MapViewApp(ShowBase):
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
            self.mode = MapViewerMode(self)
            self.mode.activate()
            self.taskMgr.add(self._tick, "map_view_tick")

        def _tick(self, task):
            self.mode.tick()
            return task.cont

    app = _MapViewApp()
    app.run()


if __name__ == "__main__":
    run_viewer()
