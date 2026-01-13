import sys
import pygame
from . import config as C
from .model import World, Camera
from .camera import world_to_screen, screen_to_world, zoom_at_cursor, scale_px_per_m
from .render import draw_road, draw_scale_bar, draw_hud, draw_facilities, COLORS
from .lanes import lane_center_x
from .fleet import Fleet
from .agent import TruckAgent
from .road_utils import generate_c_shape_road

def main():
    pygame.init()
    screen = pygame.display.set_mode((C.WIN_W, C.WIN_H))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("consolas", 18)

    #here is the place where road is initalised & its stored in World dataclass of model.py

    road_points = generate_c_shape_road(radius=100, arc_deg=210, center_x=0, center_y=0, resolution=2.0)
    world = World(road_points=road_points, road_len_m=C.ROAD_LEN_M, lane_width_m=C.LANE_WIDTH_M, lanes=C.LANES,
                  mine_ton=C.MINE_INITIAL_TON, dump_ton=0.0)

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
    follow = C.FOLLOW_TRUCK

    while running:
        dt = clock.tick(C.FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                factor = C.ZOOM_STEP if event.y > 0 else (1.0 / C.ZOOM_STEP)
                zoom_at_cursor(cam, factor, pygame.mouse.get_pos(), C.WIN_W, C.WIN_H)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    zoom_at_cursor(cam, C.ZOOM_STEP, event.pos, C.WIN_W, C.WIN_H)
                elif event.button == 5:
                    zoom_at_cursor(cam, 1.0/C.ZOOM_STEP, event.pos, C.WIN_W, C.WIN_H)
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
                    zoom_at_cursor(cam, C.ZOOM_STEP, (C.WIN_W//2, C.WIN_H//2), C.WIN_W, C.WIN_H)
                elif event.key == pygame.K_MINUS:
                    zoom_at_cursor(cam, 1.0/C.ZOOM_STEP, (C.WIN_W//2, C.WIN_H//2), C.WIN_W, C.WIN_H)

        keys = pygame.key.get_pressed()
        pan = C.PAN_SPEED_MPS * dt
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            pan = C.PAN_SPEED_FAST * dt
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
            cam.cy = lead.state.s - (C.WIN_H * 0.3) / scale_px_per_m(cam.zoom)

        # Update all agents
        telemetry = fleet.update(dt, world)

        # Render
        screen.fill(COLORS["DARK"])
        draw_road(screen, world, cam, C.WIN_W, C.WIN_H)
        draw_facilities(screen, world, cam, C.WIN_W, C.WIN_H, font)
        fleet.render(screen, world, cam, C.WIN_W, C.WIN_H)
        draw_scale_bar(screen, cam, font, C.WIN_W, C.WIN_H)
        # HUD uses lead agent for readout
        lead = fleet.agents[0]
        draw_hud(screen, font, lead.state.v, lead.state.a, lead.state.s, world)
        pygame.display.flip()

    pygame.quit()
    sys.exit()
