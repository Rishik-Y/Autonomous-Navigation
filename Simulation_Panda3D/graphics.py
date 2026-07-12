import os
import sys
import numpy as np

from Map import map_loader as map_data
from config import ROAD_WIDTH_M

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_MAP_PANDA_DIR = os.path.join(_REPO_ROOT, "MAP_Panda3D")
if _MAP_PANDA_DIR not in sys.path:
    sys.path.insert(0, _MAP_PANDA_DIR)

from panda_common import generate_curvy_path_from_nodes  # noqa: E402


class SimulationGraphics:
    def __init__(self, renderer):
        self.renderer = renderer
        self.road_splines = []
        self._build_splines()

    def _build_splines(self):
        self.road_splines = []
        for chain in map_data.VISUAL_ROAD_CHAINS:
            node_coords = [map_data.NODES[node_name] for node_name in chain if node_name in map_data.NODES]
            if len(node_coords) >= 2:
                self.road_splines.append(generate_curvy_path_from_nodes(node_coords))

    def draw_static_scene(self):
        self.renderer.draw_grid()
        self.renderer.draw_roads(self.road_splines, color=(0.4, 0.4, 0.4, 1), width=ROAD_WIDTH_M / 4.0)
        self.renderer.draw_nodes(
            map_data.NODES,
            map_data.LOAD_ZONES,
            map_data.DUMP_ZONES,
            map_data.FUEL_ZONES,
        )

    def draw_active_paths(self, selected_car):
        if not selected_car:
            self.renderer.clear_path_lines()
            return

        splines = []
        if selected_car.path and len(selected_car.path.wp) >= 2:
            splines.append([np.array([float(p[0]), float(p[1])]) for p in selected_car.path.wp])

        if splines:
            path_color = (1.0, 0.75, 0.1, 1.0) if selected_car.op_state == "RETURNING_TO_START" else (1.0, 1.0, 0.0, 1.0)
            self.renderer.draw_path_lines(splines, color=path_color, width=3.0, z=1.1)
        else:
            self.renderer.clear_path_lines()
