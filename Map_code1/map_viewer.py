import pygame
import numpy as np
import math
import map_data

# --- VISUAL & GAME SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, GRAY, PURPLE_NODE = (255, 255, 255), (100, 100, 100), (150, 0, 150)
ROAD_WIDTH_M = 4.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS

# --- GLOBAL CACHE for drawing ---
PRE_CALCULATED_SPLINES = []

# --- HELPER FUNCTIONS (from main.py) ---
def catmull_rom_point(t, p0, p1, p2, p3):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t**2) + (-p0 + 3 * p1 - 3 * p2 + p3) * (t**3))

def generate_curvy_path_from_nodes(node_list: list[np.ndarray], points_per_segment=20) -> list[np.ndarray]:
    all_waypoints_m = []
    if not node_list or len(node_list) < 2: return []
    for i in range(len(node_list) - 1):
        p0 = node_list[0] if i == 0 else node_list[i - 1]
        p1 = node_list[i]
        p2 = node_list[i + 1]
        p3 = node_list[-1] if i >= len(node_list) - 2 else node_list[i + 2]
        if i == 0: all_waypoints_m.append(p1)
        for j in range(1, points_per_segment + 1):
            t = j / float(points_per_segment)
            point = catmull_rom_point(t, p0, p1, p2, p3)
            all_waypoints_m.append(point)
    return all_waypoints_m

# --- Drawing Functions ---
def grid_to_screen(pos_m, scale, pan):
    pos_m_np = np.array(pos_m)
    pos_px = pos_m_np * METERS_TO_PIXELS
    return (int(pos_px[0] * scale + pan[0]), int(pos_px[1] * scale + pan[1]))

def screen_to_grid(pos_px, scale, pan):
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return (grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS)

def draw_road_network(screen, g_to_s, scale):
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    # Draw road segments
    for waypoints in PRE_CALCULATED_SPLINES:
        if len(waypoints) < 2: continue
        road_px = [g_to_s(p) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    
    # Draw nodes
    for node_name, pos_m in map_data.NODES.items():
        if node_name in map_data.LOAD_ZONES: color = (0, 200, 0) # Green for load
        elif node_name in map_data.DUMP_ZONES: color = (200, 0, 0) # Red for dump
        else: color = PURPLE_NODE
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(2, int(scale * 4)))
        
def run_viewer():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Map Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    # --- Pre-calculate road visuals ---
    print("Pre-calculating road visuals...")
    for chain in map_data.VISUAL_ROAD_CHAINS:
        node_coords = [map_data.NODES[node_name] for node_name in chain if node_name in map_data.NODES]
        if len(node_coords) < 2: continue
        PRE_CALCULATED_SPLINES.append(generate_curvy_path_from_nodes(node_coords))

    # --- Setup View ---
    all_nodes_m = list(map_data.NODES.values())
    min_x_m = min(p[0] for p in all_nodes_m)
    max_x_m = max(p[0] for p in all_nodes_m)
    min_y_m = min(p[1] for p in all_nodes_m)
    max_y_m = max(p[1] for p in all_nodes_m)
    map_w_m = max(1.0, max_x_m - min_x_m)
    map_h_m = max(1.0, max_y_m - min_y_m)
    
    scale = min((WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS))
    pan = [PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale)]
    mouse_dragging, last_mouse_pos = False, None
    show_node_names = False

    # --- Main Loop ---
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    show_node_names = not show_node_names
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: mouse_dragging, last_mouse_pos = True, event.pos
                elif event.button in (4, 5):
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos_m = screen_to_grid(event.pos, scale, pan)
                    scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos_m, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]
                    pan[1] += event.pos[1] - new_screen_pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0] += dx
                pan[1] += dy
                last_mouse_pos = event.pos

        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)
        
        draw_road_network(screen, g_to_s, scale)

        if show_node_names:
            for name, pos_m in map_data.NODES.items():
                text_surface = font.render(name, True, (0,0,0))
                screen.blit(text_surface, (g_to_s(pos_m)[0] + 8, g_to_s(pos_m)[1]))

        # --- HUD ---
        hud_text = font.render("Map Viewer | Pan: Left-Click+Drag | Zoom: Mouse Wheel | Toggle Node Names: N", True, (0,0,0))
        screen.blit(hud_text, (10, 10))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_viewer()
