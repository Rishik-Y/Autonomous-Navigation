# --- CONFIGURATION & CONSTANTS ---
# (from config.py)
TRUCK_CAPACITY_TON = 5.0
MINE_INITIAL_TON   = 10.0
LOAD_RATE_TPS      = 1.0
UNLOAD_RATE_TPS    = 1.0
VCAP_EMPTY_KMH     = 120.0
VCAP_LOADED_KMH    = 80.0
UTURN_TIME_S       = 2.0
STOP_ZONE_M        = 5.0
STOP_EPS_M         = 1.0
ROAD_LEN_M      = 1_000.0
LANE_WIDTH_M    = 3.5
LANES           = 4
BASE_SCALE      = 10.0
TARGET_KMH      = 120.0
AUTO_MODE       = True
FOLLOW_TRUCK    = False
MASS_KG         = 20_000.0
CD              = 0.8
FRONTAL_AREA    = 8.0
CRR             = 0.006
AIR_DENS        = 1.225
MU_TIRE         = 0.8
P_MAX_W         = 300_000.0
A_BRAKE_COMF    = 3.0
A_BRAKE_MAX     = 4.5
TRUCK_LEN_M     = 8.0
TRUCK_WID_M     = 2.5
WIN_W           = 1000
WIN_H           = 800
FPS             = 60
PAN_SPEED_MPS   = 80.0
PAN_SPEED_FAST  = 200.0
ZOOM_STEP       = 1.1
Kp          = 0.8
Ki          = 0.2
JERK_LIMIT  = 2.0
DRIVE_SIDE      = "left"
WHEELBASE_M        = 4.0
STEER_MAX_DEG      = 28.0
STEER_RATE_DEGPS   = 180.0
STEER_MAX_RAD      = STEER_MAX_DEG * 3.1415926535 / 180.0
STEER_RATE_RADPS   = STEER_RATE_DEGPS * 3.1415926535 / 180.0
LOOKAHEAD_GAIN     = 1.2
LOOKAHEAD_MIN_M    = 5.0
LOOKAHEAD_MAX_M    = 40.0
ROAD_WIDTH_M       = 20.0
PASS_MARGIN_M      = 0.5

# --- MODEL CLASSES ---
from dataclasses import dataclass
@dataclass
class TruckState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 1.5707963268
    delta: float = 0.0
    v: float = 0.0
    a: float = 0.0
    s_path: float = 0.0
    cargo_ton: float = 0.0
    state: str = "DRIVE_TO_MINE"
    state_timer: float = 0.0
    width_m: float = 2.5
    length_m: float = 8.0
@dataclass
class ControllerState:
    int_err: float = 0.0
    a_cmd_prev: float = 0.0
@dataclass
class Camera:
    cx: float
    cy: float
    zoom: float
@dataclass
class World:
    road_len_m: float
    lane_width_m: float
    lanes: int
    mine_ton: float = 0.0
    dump_ton: float = 0.0
    road_width_m: float = 8.0

# --- LANES ---
def clamp_lane_index(lane_index: int, lanes: int) -> int:
    return max(1, min(lanes, lane_index))
def default_lane(lanes: int, drive_side: str) -> int:
    side = (drive_side or "right").lower()
    return 1 if side == "left" else lanes
def lane_center_x(lane_index: int, lane_width_m: float) -> float:
    return (lane_index - 0.5) * lane_width_m

# --- ROAD ---
def can_pass(road_width_m: float, truck_width_m: float, margin_m: float = 0.5) -> bool:
    return road_width_m >= (2.0 * truck_width_m + 2.0 * margin_m)
def default_lateral_offset(drive_side: str, road_width_m: float, truck_width_m: float) -> float:
    side = (drive_side or "left").lower()
    sign = -1.0 if side == "left" else +1.0
    max_off = 0.5 * (road_width_m - truck_width_m)
    return max(-max_off, min(max_off, sign * 0.25 * road_width_m))

# --- PHYSICS ---
import math
def kmh_to_ms(kmh): return kmh / 3.6
def ms_to_kmh(ms):  return ms * 3.6
TARGET_MS = kmh_to_ms(TARGET_KMH)
def current_mass_kg(cargo_ton: float) -> float:
    return MASS_KG + cargo_ton * 1000.0
def resist_forces(v_ms: float, mass_kg: float) -> float:
    F_rr = CRR * mass_kg * 9.81
    F_d  = 0.5 * AIR_DENS * CD * FRONTAL_AREA * v_ms * v_ms
    return F_rr + F_d
def traction_force_from_power(v_ms: float, throttle: float, mass_kg: float) -> float:
    v_eff = max(v_ms, 0.5)
    F_power = (P_MAX_W * max(0.0, min(1.0, throttle))) / v_eff
    F_mu = MU_TIRE * mass_kg * 9.81
    return min(F_power, F_mu)
def brake_force_from_command(brake_cmd: float, mass_kg: float) -> float:
    brake_cmd = max(0.0, min(1.0, brake_cmd))
    return brake_cmd * mass_kg * A_BRAKE_MAX
def stopping_distance(v_ms: float, a_dec: float) -> float:
    if a_dec <= 0.0:
        return float('inf')
    return v_ms * v_ms / (2.0 * a_dec)

# --- PATH ---
from typing import List, Tuple
Vec2 = Tuple[float, float]
def _dist(a: Vec2, b: Vec2) -> float:
    dx, dy = a[0]-b[0], a[1]-b[1]
    return math.hypot(dx, dy)
class Path:
    def __init__(self, waypoints: List[Vec2]):
        assert len(waypoints) >= 2
        self.wp = waypoints
        self.s = [0.0]
        for i in range(1, len(self.wp)):
            self.s.append(self.s[-1] + _dist(self.wp[i-1], self.wp[i]))
        self.length = self.s[-1]
    def project(self, p: Vec2) -> Tuple[float, Vec2, int]:
        best_d = 1e18
        best_s = 0.0
        best_pt = self.wp[0]
        best_i = 0
        for i in range(1, len(self.wp)):
            a = self.wp[i-1]; b = self.wp[i]
            ax, ay = a; bx, by = b
            apx, apy = p[0]-ax, p[1]-ay
            abx, aby = bx-ax, by-ay
            ab2 = abx*abx + aby*aby
            t = 0.0 if ab2 < 1e-9 else max(0.0, min(1.0, (apx*abx + apy*aby)/ab2))
            qx = ax + t*abx; qy = ay + t*aby
            d = math.hypot(p[0]-qx, p[1]-qy)
            if d < best_d:
                best_d = d
                best_i = i
                best_pt = (qx, qy)
                best_s = self.s[i-1] + math.hypot(qx-ax, qy-ay)
        return best_s, best_pt, best_i
    def point_at(self, s_query: float) -> Vec2:
        s = max(0.0, min(self.length, s_query))
        lo, hi = 0, len(self.wp)-1
        while lo < hi and not (self.s[lo] <= s <= self.s[lo+1]):
            mid = (lo + hi)//2
            if self.s[mid] < s:
                lo = mid
            else:
                hi = mid
        i = min(lo, len(self.wp)-2)
        a = self.wp[i]; b = self.wp[i+1]
        seg_len = self.s[i+1] - self.s[i]
        t = 0.0 if seg_len < 1e-9 else (s - self.s[i]) / seg_len
        return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))
    def tangent_at(self, s_query: float) -> Vec2:
        s = max(0.0, min(self.length, s_query))
        lo, hi = 0, len(self.wp)-1
        while lo < hi and not (self.s[lo] <= s <= self.s[lo+1]):
            mid = (lo + hi)//2
            if self.s[mid] < s:
                lo = mid
            else:
                hi = mid
        i = min(lo, len(self.wp)-2)
        a = self.wp[i]; b = self.wp[i+1]
        dx, dy = b[0]-a[0], b[1]-a[1]
        L = math.hypot(dx, dy)
        return (dx/L, dy/L) if L > 1e-9 else (1.0, 0.0)
    def lookahead_point(self, s_now: float, ld: float) -> Vec2:
        return self.point_at(s_now + ld)
    @staticmethod
    def build_straight_with_turnpads(length_m: float, radius_m: float, offset_x: float = 0.0, n_arc_pts: int = 24):
        wp = []
        wp.append((offset_x, 0.0))
        wp.append((offset_x, length_m - radius_m))
        cx, cy = offset_x + radius_m, length_m - radius_m
        for i in range(n_arc_pts+1):
            th = -math.pi/2 + i * (math.pi / n_arc_pts)
            wp.append((cx + radius_m * math.cos(th), cy + radius_m * math.sin(th)))
        wp.append((offset_x + 2*radius_m, radius_m))
        cxb, cyb = offset_x + radius_m, radius_m
        for i in range(n_arc_pts+1):
            th = math.pi/2 + i * (math.pi / n_arc_pts)
            wp.append((cxb + radius_m * math.cos(th), cyb + radius_m * math.sin(th)))
        wp.append((offset_x, 0.0))
        return Path(wp)

# --- STEERING ---
def dynamic_lookahead(v_ms: float) -> float:
    return max(LOOKAHEAD_MIN_M, min(LOOKAHEAD_MAX_M, LOOKAHEAD_GAIN * max(0.0, v_ms)))
def pure_pursuit_delta(x, y, yaw, path, s_on_path, v_ms, dt, delta_prev, dir_sign, remaining_dist):
    ld = dynamic_lookahead(v_ms)
    ld = min(ld, max(3.0, 0.7 * max(0.0, remaining_dist)))
    s_target = s_on_path + (dir_sign * ld)
    tx, ty = path.point_at(s_target)
    dx, dy = tx - x, ty - y
    ca, sa = math.cos(yaw), math.sin(yaw)
    x_v =  ca*dx + sa*dy
    y_v = -sa*dx + ca*dy
    alpha = math.atan2(y_v, x_v)
    delta = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), max(1e-3, ld))
    delta = max(-STEER_MAX_RAD, min(STEER_MAX_RAD, delta))
    dmax = STEER_RATE_RADPS * dt
    delta = max(delta_prev - dmax, min(delta, delta_prev + dmax))
    return delta

# --- CAMERA ---
def scale_px_per_m(zoom: float) -> float:
    return BASE_SCALE * zoom
def world_to_screen(xm, ym, cam, win_w, win_h):
    S = scale_px_per_m(cam.zoom)
    xs = win_w * 0.5 + (xm - cam.cx) * S
    ys = win_h * 0.5 + (cam.cy - ym) * S
    return int(xs), int(ys)
def screen_to_world(xs, ys, cam, win_w, win_h):
    S = scale_px_per_m(cam.zoom)
    xm = cam.cx + (xs - win_w * 0.5) / S
    ym = cam.cy - (ys - win_h * 0.5) / S
    return xm, ym
def zoom_at_cursor(cam, factor, mouse_pos, win_w, win_h):
    if factor <= 0.0:
        return
    mx, my = mouse_pos
    wx_before, wy_before = screen_to_world(mx, my, cam, win_w, win_h)
    cam.zoom *= factor
    cam.zoom = max(0.05, min(20.0, cam.zoom))
    wx_after, wy_after = screen_to_world(mx, my, cam, win_w, win_h)
    cam.cx += (wx_before - wx_after)
    cam.cy += (wy_before - wy_after)

# --- CYCLE ---
DRIVE_TO_MINE = "DRIVE_TO_MINE"
LOAD          = "LOAD"
UTURN         = "UTURN"
DRIVE_TO_DUMP = "DRIVE_TO_DUMP"
UNLOAD        = "UNLOAD"
IDLE          = "IDLE"
def mine_stop_s(world): return world.road_len_m - STOP_ZONE_M
def dump_stop_s(world): return 0.0 + STOP_ZONE_M
def init_cycle(world, truck):
    truck.state = DRIVE_TO_MINE
    truck.state_timer = 0.0
def update_cycle(dt, world, truck):
    loaded = truck.cargo_ton > 1e-6
    vcap_kmh = VCAP_LOADED_KMH if loaded else VCAP_EMPTY_KMH
    vcap_ms = vcap_kmh / 3.6
    if truck.state == DRIVE_TO_MINE:
        target_s = mine_stop_s(world)
        dir_sign = +1
        if abs(truck.s_path - target_s) < STOP_EPS_M and truck.v < 0.2:
            truck.state = LOAD
            truck.state_timer = 0.0
        return target_s, vcap_ms, dir_sign
    elif truck.state == LOAD:
        target_s = truck.s_path
        dir_sign = 0
        room_t = max(0.0, TRUCK_CAPACITY_TON - truck.cargo_ton)
        take_t = min(LOAD_RATE_TPS * dt, room_t, world.mine_ton)
        truck.cargo_ton += take_t
        world.mine_ton  -= take_t
        if room_t - take_t <= 1e-6 or world.mine_ton <= 1e-6:
            truck.state = UTURN
            truck.state_timer = UTURN_TIME_S
        return target_s, 0.0, dir_sign
    elif truck.state == UTURN:
        target_s = truck.s_path
        dir_sign = 0
        truck.state_timer -= dt
        if truck.state_timer <= 0.0:
            if truck.cargo_ton > 1e-6:
                truck.state = DRIVE_TO_DUMP
            elif world.mine_ton > 1e-6:
                truck.state = DRIVE_TO_MINE
            else:
                truck.state = IDLE
        return target_s, 0.0, dir_sign
    elif truck.state == DRIVE_TO_DUMP:
        target_s = dump_stop_s(world)
        dir_sign = -1
        if abs(truck.s_path - target_s) < STOP_EPS_M and truck.v < 0.2:
            truck.state = UNLOAD
            truck.state_timer = 0.0
        return target_s, vcap_ms, dir_sign
    elif truck.state == UNLOAD:
        target_s = truck.s_path
        dir_sign = 0
        drop_t = min(UNLOAD_RATE_TPS * dt, truck.cargo_ton)
        truck.cargo_ton -= drop_t
        world.dump_ton   += drop_t
        if truck.cargo_ton <= 1e-6:
            truck.state = UTURN
            truck.state_timer = UTURN_TIME_S
        return target_s, 0.0, dir_sign
    else:
        return truck.s_path, 0.0, 0

# --- CONTROL ---
def controller(dt, s, v, mass_kg, target_s, vcap_ms, ctrl, dir_sign):
    dist = max(0.0, dir_sign * (target_s - s))
    v_stop_cap = math.sqrt(max(0.0, 2.0 * A_BRAKE_COMF * dist))
    v_ref = min(vcap_ms, v_stop_cap)
    v_margin = 0.5
    need_emergency = v > (v_stop_cap + v_margin)
    a_brake_limit = A_BRAKE_MAX if need_emergency else A_BRAKE_COMF
    tau = 0.6
    a_des = (v_ref - v) / max(0.1, tau)
    a_raw = max(-a_brake_limit, min(1.5, a_des))
    da_max = JERK_LIMIT * dt
    a_cmd = max(ctrl.a_cmd_prev - da_max, min(a_raw, ctrl.a_cmd_prev + da_max))
    ctrl.a_cmd_prev = a_cmd
    F_res = resist_forces(v, mass_kg)
    if a_cmd >= 0.0:
        F_need = mass_kg * a_cmd + F_res
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if traction_force_from_power(v, mid, mass_kg) >= F_need:
                hi = mid
            else:
                lo = mid
        throttle = hi
        brake = 0.0
    else:
        F_need_brake = max(0.0, mass_kg * (-a_cmd) - F_res)
        brake = min(1.0, F_need_brake / (mass_kg * A_BRAKE_MAX))
        throttle = 0.0
    return throttle, brake

# --- AGENT ---
class AgentConfig:
    def __init__(self, id: str, keep_side: str = "left"):
        self.id = id
        self.keep_side = keep_side
class TruckAgent:
    def __init__(self, agent_id: str, world: World, path: Path, cfg: AgentConfig = None):
        self.id = agent_id
        self.state = TruckState()
        self.ctrl  = ControllerState()
        self.path  = path
        self.keep_side = (cfg.keep_side if cfg else "left")
        dump_xy = self.path.point_at(0.0)
        self.state.x, self.state.y = dump_xy
        self.state.yaw = 1.5707963268
        self.state.s_path = 0.0
        init_cycle(world, self.state)
    def step(self, dt: float, world: World, neighbors: List["TruckAgent"]):
        target_s, vcap_ms, dir_sign = update_cycle(dt, world, self.state)
        s_now, _, _ = self.path.project((self.state.x, self.state.y))
        self.state.s_path = s_now
        must_yield = False
        horizon_m = 60.0
        for other in neighbors:
            if other is self:
                continue
            s_other = other.state.s_path
            if abs(s_other - s_now) < horizon_m:
                going_up = ("MINE" in self.state.state) or (self.state.state == "DRIVE_TO_MINE")
                other_up = ("MINE" in other.state.state) or (other.state.state == "DRIVE_TO_MINE")
                if going_up != other_up:
                    if not can_pass(world.road_width_m, self.state.width_m, PASS_MARGIN_M):
                        must_yield = True
                        break
        mass_kg = current_mass_kg(self.state.cargo_ton)
        local_vcap = 0.0 if must_yield else vcap_ms
        throttle, brake = controller(dt, self.state.s_path, self.state.v, mass_kg,
                                     target_s, local_vcap, self.ctrl, dir_sign)
        remaining = max(0.0, dir_sign * (target_s - self.state.s_path))
        self.state.delta = pure_pursuit_delta(
            self.state.x, self.state.y, self.state.yaw,
            self.path, self.state.s_path, self.state.v, dt,
            self.state.delta, dir_sign, remaining
        )
        if remaining < 10.0:
            local_vcap = min(local_vcap, 5.0)
        F_res   = resist_forces(self.state.v, mass_kg)
        F_trac  = traction_force_from_power(self.state.v, throttle, mass_kg)
        F_brake = brake_force_from_command(brake, mass_kg)
        dv = (F_trac - F_res - F_brake) / mass_kg
        self.state.a = dv
        if local_vcap > 1e-6:
            self.state.v = max(0.0, min(self.state.v + dv * dt, local_vcap))
        else:
            self.state.v = max(0.0, self.state.v + dv * dt)
        self.state.yaw += (self.state.v * math.tan(self.state.delta) / WHEELBASE_M) * dt
        self.state.x   += self.state.v * math.cos(self.state.yaw) * dt
        self.state.y   += self.state.v * math.sin(self.state.yaw) * dt
        if self.state.s_path <= 0.0 and "DUMP" in self.state.state:
            self.state.s_path = 0.0
        if self.state.s_path >= self.path.length and "MINE" in self.state.state:
            self.state.s_path = self.path.length
        return {
            "id": self.id, "x": self.state.x, "y": self.state.y,
            "yaw": self.state.yaw, "v": self.state.v, "a": self.state.a,
            "cargo_ton": self.state.cargo_ton, "state": self.state.state,
        }

# --- FLEET ---
class Fleet:
    def __init__(self):
        self.agents: List[TruckAgent] = []
    def add(self, agent: TruckAgent):
        self.agents.append(agent)
    def update(self, dt: float, world: World):
        telemetry = []
        for i, agent in enumerate(self.agents):
            neighbors = self.agents
            telemetry.append(agent.step(dt, world, neighbors))
        return telemetry
    def render(self, screen, world, cam, win_w, win_h):
        for agent in self.agents:
            st = agent.state
            draw_truck(screen, st, world, cam, win_w, win_h)

# --- RENDER ---
import pygame
COLORS = {
    "GREY":   (85, 85, 85),
    "DARK":   (30, 30, 30),
    "WHITE":  (240, 240, 240),
    "YELLOW": (240, 220, 0),
    "RED":    (220, 40, 40),
    "GREEN":  (60, 200, 80),
    "LINE":   (210, 210, 210),
}
def draw_road(screen, world, cam, win_w, win_h, path=None):
    GREY = COLORS["GREY"]
    if path:
        pts = []
        S = scale_px_per_m(cam.zoom)
        half = 0.5 * world.road_width_m
        for i in range(1, len(path.wp)):
            ax, ay = path.wp[i-1]; bx, by = path.wp[i]
            axs, ays = world_to_screen(ax, ay, cam, win_w, win_h)
            bxs, bys = world_to_screen(bx, by, cam, win_w, win_h)
            pygame.draw.line(screen, GREY, (axs, ays), (bxs, bys), max(2, int(world.road_width_m * S)))
    else:
        pygame.draw.rect(screen, GREY, pygame.Rect(0, 0, win_w, win_h))
def draw_facilities(screen, world, cam, win_w, win_h, font):
    WHITE = COLORS["WHITE"]
    total_w_m = world.lanes * world.lane_width_m
    x0 = 0.0; x1 = total_w_m
    y_dump0 = 0.0
    y_dump1 = STOP_ZONE_M * 2.0
    xs0, ys0 = world_to_screen(x0, y_dump0, cam, win_w, win_h)
    xs1, ys1 = world_to_screen(x1, y_dump1, cam, win_w, win_h)
    dump_rect = pygame.Rect(min(xs0,xs1), min(ys0,ys1), abs(xs1-xs0), abs(ys1-ys0))
    pygame.draw.rect(screen, (60,60,60), dump_rect, border_radius=4)
    txt = font.render(f"Dump Site  |  Collected: {world.dump_ton:.2f} t", True, WHITE)
    screen.blit(txt, (dump_rect.x + 8, dump_rect.y + 6))
    y_mine1 = world.road_len_m
    y_mine0 = world.road_len_m - STOP_ZONE_M * 2.0
    xm0, ym0 = world_to_screen(x0, y_mine0, cam, win_w, win_h)
    xm1, ym1 = world_to_screen(x1, y_mine1, cam, win_w, win_h)
    mine_rect = pygame.Rect(min(xm0,xm1), min(ym0,ym1), abs(xm1-xm0), abs(ym1-ym0))
    pygame.draw.rect(screen, (60,60,60), mine_rect, border_radius=4)
    txt = font.render(f"Coal Mine  |  Remaining: {world.mine_ton:.2f} t", True, WHITE)
    screen.blit(txt, (mine_rect.x + 8, mine_rect.y + 6))
def draw_truck(screen, st, world, cam, win_w, win_h):
    S = scale_px_per_m(cam.zoom)
    w = max(2, int(st.width_m * S))
    h = max(4, int(st.length_m * S))
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(surf, COLORS["RED"], pygame.Rect(0, 0, w, h), border_radius=4)
    frac = 0.0 if TRUCK_CAPACITY_TON <= 1e-6 else min(1.0, st.cargo_ton / TRUCK_CAPACITY_TON)
    if frac > 0.0:
        fh = int(h * frac)
        pygame.draw.rect(surf, COLORS["GREEN"], pygame.Rect(0, h - fh, w, fh), border_radius=3)
    deg = -math.degrees(st.yaw) + 90.0
    rsurf = pygame.transform.rotozoom(surf, deg, 1.0)
    xs, ys = world_to_screen(st.x, st.y, cam, win_w, win_h)
    rect = rsurf.get_rect(center=(xs, ys))
    screen.blit(rsurf, rect)
def draw_hud(screen, font, v, a, s, world):
    WHITE = COLORS["WHITE"]; GREEN = COLORS["GREEN"]
    kmh = ms_to_kmh(v)
    text1 = font.render(f"Speed: {kmh:6.1f} km/h   Accel: {a:5.2f} m/s^2", True, WHITE)
    text2 = font.render(f"Distance: {s:7.1f} m / {world.road_len_m:.0f} m", True, WHITE)
    text3 = font.render("Controls: Wheel zoom, Right-drag pan, Arrows/WASD pan, +/- zoom, C follow, R reset", True, GREEN)
    screen.blit(text1, (16, 12))
    screen.blit(text2, (16, 36))
    screen.blit(text3, (16, 60))
def draw_scale_bar(screen, cam, font, win_w, win_h):
    WHITE = COLORS["WHITE"]
    S = scale_px_per_m(cam.zoom)
    nice = [1,2,5]; target_px = 220
    candidates = [k * (10 ** n) for n in range(-3, 6) for k in nice]
    best = min(candidates, key=lambda c: abs(c * S - target_px))
    length_m = max(1e-6, best)
    bar_px = length_m * S
    x0 = 20; y0 = win_h - 40
    pygame.draw.line(screen, WHITE, (x0, y0), (x0 + int(bar_px), y0), 3)
    pygame.draw.line(screen, WHITE, (x0, y0 - 8), (x0, y0 + 8), 2)
    pygame.draw.line(screen, WHITE, (x0 + int(bar_px), y0 - 8), (x0 + int(bar_px), y0 + 8), 2)
    sub_m = 1.0 if S >= 15 else (5.0 if S >= 3.0 else 10.0)
    n_sub = int(length_m // sub_m)
    for i in range(1, n_sub):
        xi = x0 + int(i * sub_m * S)
        pygame.draw.line(screen, WHITE, (xi, y0 - 5), (xi, y0 + 5), 1)
    label = f"{length_m:g} m   |   zoom {cam.zoom:.2f}x   |   {S:.1f} px/m"
    text = font.render(label, True, WHITE)
    screen.blit(text, (x0, y0 + 10))

# --- MAIN APP ---
def main():
    import sys
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("consolas", 18)
    world = World(
        road_len_m=ROAD_LEN_M,
        lane_width_m=LANE_WIDTH_M,
        lanes=LANES,
        mine_ton=MINE_INITIAL_TON,
        dump_ton=0.0,
        road_width_m=ROAD_WIDTH_M
    )
    center_x = 0.0
    path = Path([(center_x, 0.0), (center_x, world.road_len_m)])
    turn_radius = WHEELBASE_M / math.tan(STEER_MAX_RAD)
    min_u_width = 2.0 * turn_radius + TRUCK_WID_M + 1.0
    if world.road_width_m < min_u_width:
        print(f"[WARN] Road width {world.road_width_m:.1f} m < ~{min_u_width:.1f} m for one-sweep U-turn; "
              f"consider widening ROAD_WIDTH_M or adding turn pads.", flush=True)
    fleet = Fleet()
    agent = TruckAgent("truck-001", world, path, AgentConfig(id="truck-001", keep_side=DRIVE_SIDE))
    fleet.add(agent)
    cam = Camera(cx=center_x, cy=world.road_len_m * 0.15, zoom=1.0)
    running = True
    dragging = False
    last_mouse = (0, 0)
    follow = FOLLOW_TRUCK
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                factor = ZOOM_STEP if event.y > 0 else (1.0 / ZOOM_STEP)
                zoom_at_cursor(cam, factor, pygame.mouse.get_pos(), WIN_W, WIN_H)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    zoom_at_cursor(cam, ZOOM_STEP, event.pos, WIN_W, WIN_H)
                elif event.button == 5:
                    zoom_at_cursor(cam, 1.0 / ZOOM_STEP, event.pos, WIN_W, WIN_H)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                dx = mx - last_mouse[0]
                dy = my - last_mouse[1]
                last_mouse = (mx, my)
                S = scale_px_per_m(cam.zoom)
                cam.cx -= dx / S
                cam.cy += dy / S
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    follow = not follow
                elif event.key == pygame.K_r:
                    cam.zoom = 1.0
                    cam.cx = center_x
                    cam.cy = world.road_len_m * 0.15
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    zoom_at_cursor(cam, ZOOM_STEP, (WIN_W // 2, WIN_H // 2), WIN_W, WIN_H)
                elif event.key == pygame.K_MINUS:
                    zoom_at_cursor(cam, 1.0 / ZOOM_STEP, (WIN_W // 2, WIN_H // 2), WIN_W, WIN_H)
        keys = pygame.key.get_pressed()
        pan = PAN_SPEED_MPS * dt
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            pan = PAN_SPEED_FAST * dt
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            cam.cx -= pan
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            cam.cx += pan
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            cam.cy += pan
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            cam.cy -= pan
        if follow and fleet.agents:
            lead = fleet.agents[0]
            cam.cx = lead.state.x
            cam.cy = lead.state.y - (WIN_H * 0.3) / scale_px_per_m(cam.zoom)
        telemetry = fleet.update(dt, world)
        screen.fill(COLORS["DARK"])
        draw_road(screen, world, cam, WIN_W, WIN_H, path)
        draw_facilities(screen, world, cam, WIN_W, WIN_H, font)
        fleet.render(screen, world, cam, WIN_W, WIN_H)
        draw_scale_bar(screen, cam, font, WIN_W, WIN_H)
        lead = fleet.agents[0]
        draw_hud(screen, font, lead.state.v, lead.state.a, lead.state.s_path, world)
        pygame.display.flip()
    pygame.quit()
    sys.exit()

# --- ENTRY POINT ---
if __name__ == "__main__":
    main()

