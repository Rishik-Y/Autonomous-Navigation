from dataclasses import dataclass

@dataclass
class TruckState:
    s: float = 0.0
    v: float = 0.0
    a: float = 0.0
    lane_index: int = 1
    cargo_ton: float = 0.0
    heading: int = +1         # +1 up (to mine), -1 down (to dump)
    state: str = "DRIVE_TO_MINE"
    state_timer: float = 0.0

@dataclass
class ControllerState:
    int_err: float = 0.0
    a_cmd_prev: float = 0.0

@dataclass
class World:
    road_len_m: float
    lane_width_m: float
    lanes: int

@dataclass
class Camera:
    cx: float
    cy: float
    zoom: float

@dataclass
class World:
    road_points: list
    road_len_m: float
    lane_width_m: float
    lanes: int
    mine_ton: float = 0.0
    dump_ton: float = 0.0