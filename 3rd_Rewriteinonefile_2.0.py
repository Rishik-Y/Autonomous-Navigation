
import sys
import math
import pygame
from dataclasses import dataclass
from typing import List

# ==========================================
# CONFIGURATION
# ==========================================

# Logistics
TRUCK_CAPACITY_TON = 5.0
MINE_INITIAL_TON   = 10.0
LOAD_RATE_TPS      = 1.0      # tons per second (tune as desired)
UNLOAD_RATE_TPS    = 1.0

# Speed caps (km/h)
VCAP_EMPTY_KMH     = 65   # unchanged for empty trips
VCAP_LOADED_KMH    = 40.0     # per requirement for loaded trips

# Cycle timing and stop geometry
UTURN_TIME_S       = 2.0
STOP_ZONE_M        = 5.0      # stop this far from each end
STOP_EPS_M         = 0.5      # stop tolerance in meters

# Simulation parameters
ROAD_LEN_M      = 1_000.0
LANE_WIDTH_M    = 3.5
LANES           = 4
BASE_SCALE      = 10.0
TARGET_KMH      = 120.0
AUTO_MODE       = True
FOLLOW_TRUCK    = False

# Acceleration (m/s²)
A_ACCEL_MAX = 0.4          # Max acceleration, 0.1–0.4 m/s²
A_ACCEL_MIN = 0.1          # Min practical acceleration for heavy vehicles

# Truck physical parameters
MASS_KG         = 20_000.0
CD              = 0.8
FRONTAL_AREA    = 8.0
CRR             = 0.006
AIR_DENS        = 1.225
MU_TIRE         = 0.8
P_MAX_W         = 300_000.0
A_BRAKE_COMF    = 0.5
A_BRAKE_MAX     = 1

# Truck drawing parameters
TRUCK_LEN_M     = 8.0
TRUCK_WID_M     = 2.5

# Window and drawing
WIN_W           = 1000
WIN_H           = 800
FPS             = 60

# Controls
PAN_SPEED_MPS   = 80.0
PAN_SPEED_FAST  = 200.0
ZOOM_STEP       = 1.1

# Controller gains and limits
Kp          = 0.8
Ki          = 0.2
JERK_LIMIT  = 2.0  # m/s^3 (tune 1.5–2.5 for smoothness)

DRIVE_SIDE      = "left"   # "left" or "right"; affects default lane choice

# ==========================================
# PHYSICS UTILS
# ==========================================

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
    v_eff = max(v_ms, 0.5)  # avoid singularity near 0
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

# ==========================================
# LANE UTILS
# ==========================================

def clamp_lane_index(lane_index: int, lanes: int) -> int:
    return max(1, min(lanes, lane_index))

def default_lane(lanes: int, drive_side: str) -> int:
    side = (drive_side or "right").lower()
    # Keep-left: use leftmost lane; keep-right: use rightmost lane
    return 1 if side == "left" else lanes

def lane_center_x(lane_index: int, lane_width_m: float) -> float:
    # Lane i spans [(i-1)*W, i*W]; center at (i-0.5)*W
    return (lane_index - 0.5) * lane_width_m

# ==========================================
# ROAD UTILS
# ==========================================

def generate_c_shape_road(radius=21, arc_deg=210, center_x=0, center_y=0, resolution=1.0):
    """
    Generate C-shape waypoints centered at (center_x, center_y),
    with specified radius and arc length in degrees.
    """
    arc_rad = math.radians(arc_deg)
    total_length = radius * arc_rad
    num_points = max(2, int(total_length / resolution))
    
    # Starting angle so that ends open: for C shape opening horizontally
    # e.g. start at 75 degrees, span 210 degrees
    start_angle = math.radians(75)  # Adjust so that opening aligns with straight parts
    
    waypoints = []
    for i in range(num_points + 1):
        angle = start_angle + (arc_rad * i / num_points)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        waypoints.append((x,y))
    return waypoints

# ==========================================
# MODEL / DATA STRUCTURES
# ==========================================

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
    road_points: list
    road_len_m: float
    lane_width_m: float
    lanes: int
    mine_ton: float = 0.0
    dump_ton: float = 0.0

@dataclass
class Camera:
    cx: float
    cy: float
    zoom: float

# ==========================================
# CAMERA
# ==========================================

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

# ==========================================
# CONTROLLER
# ==========================================

def controller(dt, s, v, mass_kg, target_s, vcap_ms, ctrl):
    """
    Controller function to compute throttle and brake commands for a truck agent.
    """

    # Remaining distance to stop (always positive)
    dist = max(0.0, abs(target_s - s))

    # Braking-distance cap: v_stop_cap = sqrt(2 * a_comf * dist)
    v_stop_cap = math.sqrt(max(0.0, 2.0 * A_BRAKE_COMF * dist))

    # Reference speed: cruise cap limited by braking-distance cap
    v_ref = min(vcap_ms, v_stop_cap)

    # Decide comfort vs emergency deceleration if current v exceeds comfort-capable stop speed
    v_margin = 0.5  # m/s
    need_emergency = v > (v_stop_cap + v_margin)
    a_brake_limit = A_BRAKE_MAX if need_emergency else A_BRAKE_COMF

    # First-order speed tracking for smoothness (simpler and stable near v=0)
    tau = 0.6  # response time in seconds
    a_des = (v_ref - v) / max(0.1, tau)

    # Clamp acceleration command considering braking limits and max acceleration from config
    # Also enforce min acceleration limit when accelerating positively (unloaded)
    if a_des > 0:
        # Clamp between min and max acceleration
        a_des = max(A_ACCEL_MIN, min(A_ACCEL_MAX, a_des))
    else:
        # Clamp braking deceleration
        a_des = max(-a_brake_limit, min(0.0, a_des))

    # Jerk limit (rate of change of acceleration)
    da_max = JERK_LIMIT * dt

    # Apply jerk limit smoothing between previous command and current desired command
    a_cmd = max(ctrl.a_cmd_prev - da_max, min(a_des, ctrl.a_cmd_prev + da_max))
    ctrl.a_cmd_prev = a_cmd

    # Compute throttle and brake commands based on acceleration command
    if a_cmd >= 0.0:
        throttle = a_cmd / A_ACCEL_MAX
        brake = 0.0
    else:
        throttle = 0.0
        # normalize brake command (negative acceleration)
        brake = -a_cmd / A_BRAKE_MAX

    # Clamp throttle and brake to valid range [0, 1]
    throttle = max(0.0, min(throttle, 1.0))
    brake = max(0.0, min(brake, 1.0))

    return throttle, brake

# ==========================================
# CYCLE / LOGIC
# ==========================================

# States
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
    truck.heading = +1  # +1 upwards towards mine, -1 downwards to dump

def update_cycle(dt, world, truck):
    # Returns (target_s, vcap_ms)
    # Loaded vs empty speed caps
    loaded = truck.cargo_ton > 1e-6
    vcap_kmh = VCAP_LOADED_KMH if loaded else VCAP_EMPTY_KMH
    vcap_ms = kmh_to_ms(vcap_kmh)

    if truck.state == DRIVE_TO_MINE:
        truck.heading = +1
        target_s = mine_stop_s(world)
        # Arrival condition handled outside via controller stop; transition when stopped:
        if abs(truck.s - target_s) < STOP_EPS_M and truck.v < 0.2:  # was 0.05
            truck.state = LOAD
            truck.state_timer = 0.0
        return target_s, vcap_ms

    elif truck.state == LOAD:
        target_s = truck.s
        # Pause while loading, progress at LOAD_RATE_TPS up to capacity or remaining mine
        room_t = max(0.0, TRUCK_CAPACITY_TON - truck.cargo_ton)
        take_t = min(LOAD_RATE_TPS * dt, room_t, world.mine_ton)
        truck.cargo_ton += take_t
        world.mine_ton  -= take_t
        if room_t - take_t <= 1e-6 or world.mine_ton <= 1e-6:
            truck.state = UTURN
            truck.state_timer = UTURN_TIME_S
        return target_s, 0.0  # hold still

    elif truck.state == UTURN:
        target_s = truck.s
        truck.state_timer -= dt
        if truck.state_timer <= 0.0:
            truck.heading *= -1
            # Decide next drive state from heading and load
            if truck.heading < 0 and truck.cargo_ton > 1e-6:
                truck.state = DRIVE_TO_DUMP
            elif truck.heading > 0 and truck.cargo_ton <= 1e-6 and world.mine_ton > 1e-6:
                truck.state = DRIVE_TO_MINE
            else:
                # If nothing to do, idle
                truck.state = IDLE
        return target_s, 0.0  # hold still during U-turn

    elif truck.state == DRIVE_TO_DUMP:
        truck.heading = -1
        target_s = dump_stop_s(world)
        if abs(truck.s - target_s) < STOP_EPS_M and truck.v < 0.2:
            truck.state = UNLOAD
            truck.state_timer = 0.0
        return target_s, vcap_ms

    elif truck.state == UNLOAD:
        target_s = truck.s
        drop_t = min(UNLOAD_RATE_TPS * dt, truck.cargo_ton)
        truck.cargo_ton -= drop_t
        world.dump_ton   += drop_t
        if truck.cargo_ton <= 1e-6:
            truck.state = UTURN
            truck.state_timer = UTURN_TIME_S
        return target_s, 0.0

    else:  # IDLE
        return truck.s, 0.0

# ==========================================
# RENDER
# ==========================================

COLORS = {
    "GREY":   (85, 85, 85),
    "DARK":   (30, 30, 30),
    "WHITE":  (240, 240, 240),
    "YELLOW": (240, 220, 0),
    "RED":    (220, 40, 40),
    "GREEN":  (60, 200, 80),
    "LINE":   (210, 210, 210),
}

def draw_road(screen, world, cam, win_w, win_h):
    GREY = COLORS["GREY"]; WHITE = COLORS["WHITE"]; LINE = COLORS["LINE"]
    total_w_m = world.lanes * world.lane_width_m
    left_x = 0.0; right_x = total_w_m; top_y = world.road_len_m; bot_y = 0.0

    x0s, y0s = world_to_screen(left_x, bot_y, cam, win_w, win_h)
    x1s, y1s = world_to_screen(right_x, top_y, cam, win_w, win_h)
    rect = pygame.Rect(min(x0s, x1s), min(y0s, y1s), abs(x1s - x0s), abs(y1s - y0s))
    pygame.draw.rect(screen, GREY, rect)

    xls, _ = world_to_screen(left_x, 0.0, cam, win_w, win_h)
    xrs, _ = world_to_screen(right_x, 0.0, cam, win_w, win_h)
    _, yts = world_to_screen(0.0, top_y, cam, win_w, win_h)
    _, ybs = world_to_screen(0.0, bot_y, cam, win_w, win_h)
    pygame.draw.line(screen, WHITE, (xls, yts), (xls, ybs), max(1, int(2 * cam.zoom)))
    pygame.draw.line(screen, WHITE, (xrs, yts), (xrs, ybs), max(1, int(2 * cam.zoom)))

    # Removed the loop that draws dashed lines for lanes
    # to simulate a coal mine road without lane markings.

def draw_facilities(screen, world, cam, win_w, win_h, font):
    WHITE = COLORS["WHITE"]
    # Draw dump at bottom zone and mine at top zone with labels
    total_w_m = world.lanes * world.lane_width_m
    x0 = 0.0; x1 = total_w_m
    # Dump site box
    y_dump0 = 0.0
    y_dump1 = STOP_ZONE_M * 2.0
    xs0, ys0 = world_to_screen(x0, y_dump0, cam, win_w, win_h)
    xs1, ys1 = world_to_screen(x1, y_dump1, cam, win_w, win_h)
    dump_rect = pygame.Rect(min(xs0,xs1), min(ys0,ys1), abs(xs1-xs0), abs(ys1-ys0))
    pygame.draw.rect(screen, (60,60,60), dump_rect, border_radius=4)
    txt = font.render(f"Dump Site  |  Collected: {world.dump_ton:.2f} t", True, WHITE)
    screen.blit(txt, (dump_rect.x + 8, dump_rect.y + 6))

    # Coal mine box
    y_mine1 = world.road_len_m
    y_mine0 = world.road_len_m - STOP_ZONE_M * 2.0
    xm0, ym0 = world_to_screen(x0, y_mine0, cam, win_w, win_h)
    xm1, ym1 = world_to_screen(x1, y_mine1, cam, win_w, win_h)
    mine_rect = pygame.Rect(min(xm0,xm1), min(ym0,ym1), abs(xm1-xm0), abs(ym1-ym0))
    pygame.draw.rect(screen, (60,60,60), mine_rect, border_radius=4)
    txt = font.render(f"Coal Mine  |  Remaining: {world.mine_ton:.2f} t", True, WHITE)
    screen.blit(txt, (mine_rect.x + 8, mine_rect.y + 6))

def draw_truck(screen, s, world, cam, win_w, win_h, lane_index, cargo_ton, state):
    RED = COLORS["RED"]; GREEN = (60,200,80); WHITE = COLORS["WHITE"]
    x_center = lane_center_x(lane_index, world.lane_width_m)
    y_center = s
    S = scale_px_per_m(cam.zoom)
    truck_w_px = int(TRUCK_WID_M * S)
    truck_h_px = int(TRUCK_LEN_M * S)
    xs, ys = world_to_screen(x_center, y_center, cam, win_w, win_h)
    rect = pygame.Rect(xs - truck_w_px // 2, ys - truck_h_px // 2, truck_w_px, truck_h_px)
    pygame.draw.rect(screen, RED, rect, border_radius=3)
    # Cargo fill overlay proportional to load
    fill_frac = min(1.0, max(0.0, cargo_ton / max(1e-6, TRUCK_CAPACITY_TON)))
    if fill_frac > 1e-3:
        fill_h = int(truck_h_px * fill_frac)
        fill_rect = pygame.Rect(rect.x, rect.bottom - fill_h, truck_w_px, fill_h)
        pygame.draw.rect(screen, GREEN, fill_rect, border_radius=2)
    # Optional state label above truck
    lbl = f"{state.replace('_',' ').title()} | {cargo_ton:.2f} t"
    text = pygame.font.SysFont("consolas", 14).render(lbl, True, WHITE)
    screen.blit(text, (rect.x, rect.y - 18))

def draw_hud(screen, font, v, a, s, world, lane_index):
    WHITE = COLORS["WHITE"]; GREEN = COLORS["GREEN"]
    kmh = ms_to_kmh(v)
    text1 = font.render(f"Speed: {kmh:6.1f} km/h   Accel: {a:5.2f} m/s^2", True, WHITE)
    text2 = font.render(f"Distance: {s:7.1f} m / {world.road_len_m:.0f} m   Lane: {lane_index}", True, WHITE)
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

# ==========================================
# AGENT
# ==========================================

@dataclass
class AgentConfig:
    id: str
    lane_index: int


class TruckAgent:
    def __init__(self, agent_id: str, world: World, lane_index: int | None = None):
        self.id = agent_id
        self.state = TruckState()
        self.ctrl  = ControllerState()
        self.state.lane_index = lane_index if lane_index is not None else default_lane(world.lanes, DRIVE_SIDE)
        init_cycle(world, self.state)

    def step(self, dt: float, world: World):
        # Compute target stop and speed cap per trip state, and update loading/unloading via cycle
        target_s, vcap_ms = update_cycle(dt, world, self.state)

        mass_kg = current_mass_kg(self.state.cargo_ton)
        throttle, brake = controller(dt, self.state.s, self.state.v, mass_kg, target_s, vcap_ms, self.ctrl)

        # Forces
        F_res   = resist_forces(self.state.v, mass_kg)
        F_trac  = traction_force_from_power(self.state.v, throttle, mass_kg)
        F_brake = brake_force_from_command(brake, mass_kg)

        # Longitudinal dynamics: dv/dt = (F_trac - F_res - F_brake)/m
        dv = (F_trac - F_res - F_brake) / mass_kg
        self.state.a = dv
        self.state.v = max(0.0, min(self.state.v + dv * dt, vcap_ms))

        # Position integrates with heading
        self.state.s = self.state.s + self.state.heading * self.state.v * dt

        # Clamp position to the target if slight overshoot; do NOT force v=0 here
        if self.state.heading > 0 and self.state.s > target_s:
            self.state.s = target_s
        elif self.state.heading < 0 and self.state.s < target_s:
            self.state.s = target_s

        # Return a minimal telemetry dict for logging or multi-agent coordination
        return {
            "id": self.id,
            "s": self.state.s,
            "v": self.state.v,
            "a": self.state.a,
            "cargo_ton": self.state.cargo_ton,
            "state": self.state.state,
            "lane_center_x": lane_center_x(self.state.lane_index, world.lane_width_m),
        }

# ==========================================
# FLEET
# ==========================================

class Fleet:
    def __init__(self):
        self.agents: List[TruckAgent] = []

    def add(self, agent: TruckAgent):
        self.agents.append(agent)

    def update(self, dt: float, world: World):
        telemetry = []
        # Simple scheduler: sequential update; replace with round-robin/time-sliced if needed
        for agent in self.agents:
            telemetry.append(agent.step(dt, world))
        return telemetry

    def render(self, screen, world, cam, win_w, win_h):
        # Draw each truck; rendering stays outside the agent class to keep agents UI-agnostic
        for agent in self.agents:
            st = agent.state
            draw_truck(screen, st.s, world, cam, win_w, win_h, st.lane_index, st.cargo_ton, st.state)

# ==========================================
# MAIN APP
# ==========================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("consolas", 18)

    #here is the place where road is initalised & its stored in World dataclass of model.py

    road_points = generate_c_shape_road(radius=100, arc_deg=210, center_x=0, center_y=0, resolution=2.0)
    world = World(road_points=road_points, road_len_m=ROAD_LEN_M, lane_width_m=LANE_WIDTH_M, lanes=LANES,
                  mine_ton=MINE_INITIAL_TON, dump_ton=0.0)

    # Create fleet and one agent for now
    fleet = Fleet()
    agent = TruckAgent(agent_id="truck-001", world=world)
    fleet.add(agent)

    # Camera follows first agent by default
    cam = Camera(cx=lane_center_x(agent.state.lane_index, world.lane_width_m),
                 cy=world.road_len_m * 0.15, zoom=1.0)

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
                    zoom_at_cursor(cam, 1.0/ZOOM_STEP, event.pos, WIN_W, WIN_H)
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
                    cam.cx = lane_center_x(agent.state.lane_index, world.lane_width_m)
                    cam.cy = world.road_len_m * 0.15
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    zoom_at_cursor(cam, ZOOM_STEP, (WIN_W//2, WIN_H//2), WIN_W, WIN_H)
                elif event.key == pygame.K_MINUS:
                    zoom_at_cursor(cam, 1.0/ZOOM_STEP, (WIN_W//2, WIN_H//2), WIN_W, WIN_H)

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
            cam.cx = lane_center_x(lead.state.lane_index, world.lane_width_m)
            cam.cy = lead.state.s - (WIN_H * 0.3) / scale_px_per_m(cam.zoom)

        # Update all agents
        telemetry = fleet.update(dt, world)

        # Render
        screen.fill(COLORS["DARK"])
        draw_road(screen, world, cam, WIN_W, WIN_H)
        draw_facilities(screen, world, cam, WIN_W, WIN_H, font)
        fleet.render(screen, world, cam, WIN_W, WIN_H)
        draw_scale_bar(screen, cam, font, WIN_W, WIN_H)
        # HUD uses lead agent for readout
        lead = fleet.agents[0]
        draw_hud(screen, font, lead.state.v, lead.state.a, lead.state.s, world, lead.state.lane_index)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
