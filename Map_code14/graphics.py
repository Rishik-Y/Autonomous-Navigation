import pygame
import numpy as np
import map_data
from config import *

def grid_to_screen(pos_m, scale, pan):
    pos_m_np = np.array(pos_m)
    pos_px = pos_m_np * METERS_TO_PIXELS
    return (int(pos_px[0] * scale + pan[0]), int(pos_px[1] * scale + pan[1]))

def screen_to_grid(pos_px, scale, pan):
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return (grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS)

def draw_road_network(screen, g_to_s, scale, waypoints_map):
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    for waypoints in waypoints_map.values():
        if len(waypoints) < 2: continue
        road_px = [g_to_s(p) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    
    for node_name, pos_m in map_data.NODES.items():
        if node_name in map_data.LOAD_ZONES: color = (0, 200, 0)
        elif node_name in map_data.DUMP_ZONES: color = (200, 0, 0)
        elif node_name in map_data.FUEL_ZONES: color = ORANGE
        else: color = PURPLE_NODE
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(2, int(scale * 4)))

def draw_active_path(screen, path, g_to_s, scale):
    if len(path.wp) < 2: return
    road_px = [g_to_s(p) for p in path.wp]
    if len(road_px) > 1:
        road_width_px = max(2, int((ROAD_WIDTH_M - 1.0) * METERS_TO_PIXELS * scale))
        pygame.draw.lines(screen, BLUE_ACTIVE, False, road_px, road_width_px)