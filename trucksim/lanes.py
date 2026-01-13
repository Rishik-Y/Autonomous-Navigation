# trucksim/lanes.py
from . import config as C

def clamp_lane_index(lane_index: int, lanes: int) -> int:
    return max(1, min(lanes, lane_index))

def default_lane(lanes: int, drive_side: str) -> int:
    side = (drive_side or "right").lower()
    # Keep-left: use leftmost lane; keep-right: use rightmost lane
    return 1 if side == "left" else lanes

def lane_center_x(lane_index: int, lane_width_m: float) -> float:
    # Lane i spans [(i-1)*W, i*W]; center at (i-0.5)*W
    return (lane_index - 0.5) * lane_width_m
