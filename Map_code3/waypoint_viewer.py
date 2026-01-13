import pygame
import numpy as np
import math
import heapq
import random
import map_data

# --- VISUAL & GAME SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, GRAY, BLUE_ACTIVE, RED, PURPLE_NODE = (255, 255, 255), (100, 100, 100), (0, 100, 200, 150), (255, 0, 0), (150, 0, 150)
WAYPOINT_COLOR = (0, 150, 255)
NODE_PATH_COLOR = (255, 165, 0) # Orange
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

def build_weighted_graph(nodes: dict, edges: list) -> dict:
    graph = {name: [] for name in nodes}
    for n1_name, n2_name in edges:
        p1 = nodes[n1_name]
        p2 = nodes[n2_name]
        distance = np.linalg.norm(p1 - p2)
        graph[n1_name].append((n2_name, distance))
        graph[n2_name].append((n1_name, distance))
    return graph

def a_star_pathfinding(graph: dict, nodes_coords: dict, start_name: str, goal_name: str) -> list[np.ndarray]:
    def h(node_name): return np.linalg.norm(nodes_coords[node_name] - nodes_coords[goal_name])
    open_set = [(h(start_name), start_name)]
    came_from = {}
    g_score = {name: float('inf') for name in nodes_coords}; g_score[start_name] = 0
    f_score = {name: float('inf') for name in nodes_coords}; f_score[start_name] = h(start_name)
    while open_set:
        _, current_name = heapq.heappop(open_set)
        if current_name == goal_name:
            path_nodes = []
            temp = current_name
            while temp in came_from:
                path_nodes.append(nodes_coords[temp])
                temp = came_from[temp]
            path_nodes.append(nodes_coords[start_name])
            return list(reversed(path_nodes))
        for neighbor_name, weight in graph[current_name]:
            tentative_g_score = g_score[current_name] + weight
            if tentative_g_score < g_score[neighbor_name]:
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative_g_score
                f_score[neighbor_name] = tentative_g_score + h(neighbor_name)
                if (f_score[neighbor_name], neighbor_name) not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor_name], neighbor_name))
    return []

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
    for waypoints in PRE_CALCULATED_SPLINES:
        if len(waypoints) < 2: continue
        road_px = [g_to_s(p) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    for node_name, pos_m in map_data.NODES.items():
        if node_name in map_data.LOAD_ZONES: color = (0, 200, 0)
        elif node_name in map_data.DUMP_ZONES: color = (200, 0, 0)
        else: color = PURPLE_NODE
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(2, int(scale * 4)))

def draw_waypoints(screen, waypoints_m, g_to_s, scale):
    if len(waypoints_m) < 2: return
    # Draw the fine-grained waypoints
    for point_m in waypoints_m:
        pygame.draw.circle(screen, WAYPOINT_COLOR, g_to_s(point_m), max(1, int(scale * 1.5)))

def draw_node_path(screen, node_path, g_to_s, scale):
    if len(node_path) < 2: return
    # Draw the coarse A* path
    path_px = [g_to_s(p) for p in node_path]
    pygame.draw.lines(screen, NODE_PATH_COLOR, False, path_px, max(2, int(scale * 3)))


def run_viewer():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Waypoint Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    # --- Build Graph and Roads ---
    print("Building road graph...")
    road_graph = build_weighted_graph(map_data.NODES, map_data.EDGES)
    print("Pre-calculating road visuals...")
    for chain in map_data.VISUAL_ROAD_CHAINS:
        node_coords = [map_data.NODES[node_name] for node_name in chain if node_name in map_data.NODES]
        if len(node_coords) < 2: continue
        PRE_CALCULATED_SPLINES.append(generate_curvy_path_from_nodes(node_coords))

    # --- Generate a single, random path ---
    start_node_name = random.choice(map_data.DUMP_ZONES)
    goal_node_name = random.choice(map_data.LOAD_ZONES)
    
    print(f"Generating path from '{start_node_name}' to '{goal_node_name}'...")
    # This is the coarse path from A* (the "plan")
    route_nodes = a_star_pathfinding(road_graph, map_data.NODES, start_node_name, goal_node_name)
    # This is the fine-grained path from the spline (the "execution")
    waypoints_m = generate_curvy_path_from_nodes(route_nodes)
    print(f"Generated {len(waypoints_m)} waypoints for the truck to follow.")

    # --- Setup View ---
    all_nodes_m = list(map_data.NODES.values())
    min_x_m, max_x_m = min(p[0] for p in all_nodes_m), max(p[0] for p in all_nodes_m)
    min_y_m, max_y_m = min(p[1] for p in all_nodes_m), max(p[1] for p in all_nodes_m)
    map_w_m, map_h_m = max(1.0, max_x_m - min_x_m), max(1.0, max_y_m - min_y_m)
    scale = min((WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS))
    pan = [PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale)]
    mouse_dragging, last_mouse_pos = False, None

    # --- Main Loop ---
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
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
        
        # 1. Draw the full map in the background
        draw_road_network(screen, g_to_s, scale)
        
        # 2. Draw the A* path (the high-level node-to-node plan)
        draw_node_path(screen, route_nodes, g_to_s, scale)

        # 3. Draw the detailed waypoints the truck actually follows
        draw_waypoints(screen, waypoints_m, g_to_s, scale)

        # --- HUD ---
        hud_texts = [
            "Waypoint Viewer | Pan: Left-Click+Drag | Zoom: Mouse Wheel",
            f"Route: {start_node_name} -> {goal_node_name}",
            f"A* Path Nodes: {len(route_nodes)} (Orange Line)",
            f"Spline Waypoints: {len(waypoints_m)} (Blue Dots)"
        ]
        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, (0,0,0))
            screen.blit(text_surface, (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_viewer()
