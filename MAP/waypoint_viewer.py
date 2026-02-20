import pygame
import numpy as np
import math
import heapq
import random
import pickle # Added for loading waypoints.pkl
import os     # Added for checking file existence
import map_data

# --- VISUAL & GAME SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, GRAY, BLUE_ACTIVE, RED, PURPLE_NODE, ORANGE = (255, 255, 255), (100, 100, 100), (0, 100, 200, 150), (255, 0, 0), (150, 0, 150), (255, 165, 0)
WAYPOINT_COLOR = (0, 150, 255)
NODE_PATH_COLOR = (255, 165, 0) # Orange
ROAD_WIDTH_M = 4.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS
POINTS_PER_SEGMENT = 20 # Must match the value from Waypoint_Editor.py

# --- HELPER FUNCTIONS ---
def build_weighted_graph(nodes: dict, edges: list) -> dict:
    graph = {name: [] for name in nodes}
    for n1_name, n2_name in edges:
        p1 = nodes[n1_name]
        p2 = nodes[n2_name]
        distance = np.linalg.norm(p1 - p2)
        graph[n1_name].append((n2_name, distance))
        graph[n2_name].append((n1_name, distance))
    return graph

def a_star_pathfinding(graph: dict, start_name: str, goal_name: str) -> list[str]:
    """Finds the shortest path between two nodes using A* and returns a list of node names."""
    open_set = [(0, start_name)] # (f_score, node_name)
    came_from = {}
    g_score = {name: float('inf') for name in graph}; g_score[start_name] = 0
    
    while open_set:
        _, current_name = heapq.heappop(open_set)

        if current_name == goal_name:
            path_names = []
            temp = current_name
            while temp in came_from:
                path_names.append(temp)
                temp = came_from[temp]
            path_names.append(start_name)
            return list(reversed(path_names))

        for neighbor_name, weight in graph[current_name]:
            tentative_g_score = g_score[current_name] + weight
            if tentative_g_score < g_score[neighbor_name]:
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative_g_score
                # Heuristic (h) is not needed for f_score since we are just looking for the cheapest path
                heapq.heappush(open_set, (tentative_g_score, neighbor_name))
    return []

def get_path_from_nodes(route_node_names, waypoints_map):
    """Stitches together pre-calculated waypoints to form a complete path."""
    final_waypoints = []
    if not route_node_names:
        return []

    for i in range(len(route_node_names) - 1):
        seg_start, seg_end = route_node_names[i], route_node_names[i+1]
        
        found_segment = False
        for chain_tuple, waypoints in waypoints_map.items():
            try:
                # Find segment in forward direction
                idx = chain_tuple.index(seg_start)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx+1] == seg_end:
                    start_wp_idx = idx * POINTS_PER_SEGMENT
                    end_wp_idx = (idx + 1) * POINTS_PER_SEGMENT
                    # Add segment, excluding the last point if it's not the final segment
                    final_waypoints.extend(waypoints[start_wp_idx:end_wp_idx])
                    found_segment = True
                    break
                
                # Find segment in reverse direction
                idx = chain_tuple.index(seg_end)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx+1] == seg_start:
                    start_wp_idx = idx * POINTS_PER_SEGMENT
                    end_wp_idx = (idx + 1) * POINTS_PER_SEGMENT
                    # Get segment and reverse it
                    segment = waypoints[start_wp_idx:end_wp_idx+1]
                    final_waypoints.extend(segment[::-1][:-1]) # Exclude last point after reversing
                    found_segment = True
                    break

            except ValueError:
                continue # Node not in this chain
        
        if not found_segment:
            print(f"!!! Warning: Could not find waypoint segment for {seg_start} -> {seg_end}")

    # Add the very last point of the entire path
    if final_waypoints and route_node_names:
        final_waypoints.append(map_data.NODES[route_node_names[-1]])

    return final_waypoints

# --- Drawing Functions ---
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
        if node_name in map_data.LOAD_ZONES: color = (0, 200, 0) # Green for load
        elif node_name in map_data.DUMP_ZONES: color = (200, 0, 0) # Red for dump
        elif node_name in map_data.FUEL_ZONES: color = ORANGE # Orange for fuel
        else: color = PURPLE_NODE
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(2, int(scale * 4)))
        
def draw_waypoints(screen, waypoints_m, g_to_s, scale):
    if len(waypoints_m) < 2: return
    # Draw the fine-grained waypoints
    for point_m in waypoints_m:
        pygame.draw.circle(screen, WAYPOINT_COLOR, g_to_s(point_m), max(1, int(scale * 1.5)))

def draw_node_path(screen, node_path_names, g_to_s, scale):
    if len(node_path_names) < 2: return
    # Draw the coarse A* path
    path_px = [g_to_s(map_data.NODES[name]) for name in node_path_names]
    pygame.draw.lines(screen, NODE_PATH_COLOR, False, path_px, max(2, int(scale * 3)))


def run_viewer():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Waypoint Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    # --- Load Pre-computed Data ---
    waypoints_filepath = 'waypoints.pkl'
    if not os.path.exists(waypoints_filepath):
        print(f"Error: Waypoint file '{waypoints_filepath}' not found.")
        print(f"Please run 'python Waypoint_Editor.py', press 'A' then 'S' to generate the file.")
        return
        
    with open(waypoints_filepath, 'rb') as f:
        waypoints_map = pickle.load(f)
    print(f"Loaded {len(waypoints_map)} pre-calculated road paths.")

    cache_filename = 'map_cache.pkl'
    if not os.path.exists(cache_filename):
        print(f"Error: Map cache file '{cache_filename}' not found.")
        print(f"Please run 'python map_data.py' to generate the cache file first.")
        return

    with open(cache_filename, 'rb') as f:
        road_graph = pickle.load(f)['road_graph']

    # --- Generate a single, random path ---
    start_node_name = random.choice(map_data.DUMP_ZONES)
    goal_node_name = random.choice(map_data.LOAD_ZONES)
    
    print(f"Generating path from '{start_node_name}' to '{goal_node_name}'...")
    # This is the coarse path from A* (the "plan")
    route_node_names = a_star_pathfinding(road_graph, start_node_name, goal_node_name)
    # This is the fine-grained path from the spline (the "execution")
    waypoints_m = get_path_from_nodes(route_node_names, waypoints_map)
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
        draw_road_network(screen, g_to_s, scale, waypoints_map)
        
        # 2. Draw the A* path (the high-level node-to-node plan)
        draw_node_path(screen, route_node_names, g_to_s, scale)

        # 3. Draw the detailed waypoints the truck actually follows
        draw_waypoints(screen, waypoints_m, g_to_s, scale)

        # --- HUD ---
        hud_texts = [
            "Waypoint Viewer | Pan: Left-Click+Drag | Zoom: Mouse Wheel",
            f"Route: {start_node_name} -> {goal_node_name}",
            f"A* Path Nodes: {len(route_node_names)} (Orange Line)",
            f"Spline Waypoints: {len(waypoints_m)} (Blue Dots)"
        ]
        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, (0,0,0))
            screen.blit(text_surface, (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_viewer()
