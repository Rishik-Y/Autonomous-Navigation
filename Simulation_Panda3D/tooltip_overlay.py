import numpy as np
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import Point2, Point3

from Map import map_loader as map_data
from config import MASS_KG, CARGO_TON


class TooltipOverlay:
    def __init__(self, app):
        self.app = app
        self.label = OnscreenText(
            text="",
            pos=(0, 0),
            scale=0.045,
            fg=(1, 1, 1, 1),
            bg=(0.1, 0.1, 0.1, 0.75),
            mayChange=True,
            align=0,
            parent=app.aspect2d,
        )
        self.label.hide()

    def _world_to_aspect(self, world_xyz):
        p3 = self.app.cam.getRelativePoint(self.app.render, Point3(*world_xyz))
        p2 = Point2()
        if self.app.camLens.project(p3, p2):
            return float(p2.x), float(p2.y)
        return None

    def show_entity(self, entity_text, world_xyz):
        screen = self._world_to_aspect(world_xyz)
        if screen is None:
            self.label.hide()
            return
        self.label.setText(entity_text)
        self.label.setPos(screen[0], screen[1] + 0.08)
        self.label.show()

    def hide(self):
        self.label.hide()


def build_entity_tooltip(entity_type, data):
    if entity_type == "truck":
        car = data["car"]
        current_cargo = max(0.0, car.current_mass_kg - MASS_KG)
        return (
            f"Truck {car.id}\n"
            f"State: {car.op_state}\n"
            f"Target: {car.target_node_name or '-'}\n"
            f"Cargo: {int(current_cargo)}/{int(CARGO_TON * 1000)} kg"
        )

    if entity_type == "coal_mine":
        name = data["name"]
        state = data["state"]
        remaining = state.get("coal_remaining", float("inf"))
        en_route = state.get("en_route", 0)
        coal_text = "∞" if remaining == float("inf") else str(int(remaining))
        return f"Mine: {name}\nCoal remaining: {coal_text} kg\nTrucks en route: {en_route}"

    if entity_type == "dump_site":
        name = data["name"]
        state = data["state"]
        dumped = int(state.get("coal_dumped", 0))
        en_route = state.get("en_route", 0)
        return f"Dump: {name}\nCoal dumped: {dumped} kg\nTrucks en route: {en_route}"

    return ""


def nearest_zone(world_xy, dispatcher, threshold_m=8.0):
    if world_xy is None:
        return None
    p = np.array(world_xy, dtype=float)

    for zone_name in map_data.LOAD_ZONES:
        if zone_name in map_data.NODES:
            d = np.linalg.norm(p - map_data.NODES[zone_name])
            if d <= threshold_m:
                return {
                    "type": "coal_mine",
                    "name": zone_name,
                    "state": dispatcher.site_states.get(zone_name, {}),
                    "world": np.array([map_data.NODES[zone_name][0], map_data.NODES[zone_name][1], 0.0]),
                }

    for zone_name in map_data.DUMP_ZONES:
        if zone_name in map_data.NODES:
            d = np.linalg.norm(p - map_data.NODES[zone_name])
            if d <= threshold_m:
                return {
                    "type": "dump_site",
                    "name": zone_name,
                    "state": dispatcher.site_states.get(zone_name, {}),
                    "world": np.array([map_data.NODES[zone_name][0], map_data.NODES[zone_name][1], 0.0]),
                }

    return None
