import pygame
from . import config as C
from .camera import world_to_screen, screen_to_world, scale_px_per_m
from .lanes import lane_center_x  # ADD THIS IMPORT

COLORS = {
    "GREY":   (85, 85, 85),
    "DARK":   (30, 30, 30),
    "WHITE":  (240, 240, 240),
    "YELLOW": (240, 220, 0),
    "RED":    (220, 40, 40),
    "GREEN":  (60, 200, 80),
    "LINE":   (210, 210, 210),
}

# The actual straight road is drawn in draw_road():

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

    dash_m = 10.0; gap_m  = 10.0
    for li in range(1, world.lanes):
        x_m = li * world.lane_width_m
        xs, _ = world_to_screen(x_m, 0.0, cam, win_w, win_h)
        _, y_world_top = screen_to_world(0, 0, cam, win_w, win_h)
        _, y_world_bot = screen_to_world(0, win_h, cam, win_w, win_h)
        y_min = max(0.0, min(y_world_top, y_world_bot))
        y_max = min(world.road_len_m, max(y_world_top, y_world_bot))
        start = y_min - ((y_min) % (dash_m + gap_m))
        y = start
        while y < y_max:
            y2 = min(y + dash_m, y_max)
            _, ys1 = world_to_screen(0.0, y, cam, win_w, win_h)
            _, ys2 = world_to_screen(0.0, y2, cam, win_w, win_h)
            pygame.draw.line(screen, LINE, (xs, ys1), (xs, ys2), max(1, int(2 * cam.zoom)))
            y += dash_m + gap_m

def draw_facilities(screen, world, cam, win_w, win_h, font):
    WHITE = COLORS["WHITE"]
    # Draw dump at bottom zone and mine at top zone with labels
    total_w_m = world.lanes * world.lane_width_m
    x0 = 0.0; x1 = total_w_m
    # Dump site box
    y_dump0 = 0.0
    y_dump1 = C.STOP_ZONE_M * 2.0
    xs0, ys0 = world_to_screen(x0, y_dump0, cam, win_w, win_h)
    xs1, ys1 = world_to_screen(x1, y_dump1, cam, win_w, win_h)
    dump_rect = pygame.Rect(min(xs0,xs1), min(ys0,ys1), abs(xs1-xs0), abs(ys1-ys0))
    pygame.draw.rect(screen, (60,60,60), dump_rect, border_radius=4)
    txt = font.render(f"Dump Site  |  Collected: {world.dump_ton:.2f} t", True, WHITE)
    screen.blit(txt, (dump_rect.x + 8, dump_rect.y + 6))

    # Coal mine box
    y_mine1 = world.road_len_m
    y_mine0 = world.road_len_m - C.STOP_ZONE_M * 2.0
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
    truck_w_px = int(C.TRUCK_WID_M * S)
    truck_h_px = int(C.TRUCK_LEN_M * S)
    xs, ys = world_to_screen(x_center, y_center, cam, win_w, win_h)
    rect = pygame.Rect(xs - truck_w_px // 2, ys - truck_h_px // 2, truck_w_px, truck_h_px)
    pygame.draw.rect(screen, RED, rect, border_radius=3)
    # Cargo fill overlay proportional to load
    fill_frac = min(1.0, max(0.0, cargo_ton / max(1e-6, C.TRUCK_CAPACITY_TON)))
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
    from .physics import ms_to_kmh
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

def draw_hud(screen, font, v, a, s, world):
    WHITE = COLORS["WHITE"]; GREEN = COLORS["GREEN"]
    from .physics import ms_to_kmh
    kmh = ms_to_kmh(v)
    text1 = font.render(f"Speed: {kmh:6.1f} km/h   Accel: {a:5.2f} m/s^2", True, WHITE)
    text2 = font.render(f"Distance: {s:7.1f} m / {world.road_len_m:.0f} m", True, WHITE)
    text3 = font.render("Controls: Wheel zoom, Right-drag pan, Arrows/WASD pan, +/- zoom, C follow, R reset", True, GREEN)
    screen.blit(text1, (16, 12))
    screen.blit(text2, (16, 36))
    screen.blit(text3, (16, 60))

def draw_curved_road(screen, waypoints, cam, win_w, win_h):
    """
    Draw a curved road on the screen based on a list of (x,y) waypoints.
    """
    if len(waypoints) < 2:
        return  # Need at least 2 points to draw

    # Convert waypoints to screen coordinates
    screen_points = [world_to_screen(x, y, cam, win_w, win_h) for (x, y) in waypoints]

    # Flatten (x,y) tuples into list of points for pygame draw
    pygame_points = [(int(xs), int(ys)) for xs, ys in screen_points]

    # Draw road centerline or edge lines in white or another color
    pygame.draw.lines(screen, C.COLORS["WHITE"], False, pygame_points, max(2, int(2 * cam.zoom)))

    # Optionally you can draw multiple lines to represent lane edges,
    # or use polygons/fill for road surface if desired.