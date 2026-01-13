import pygame
import cv2
import numpy as np
import math
import networkx as nx
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
import os

# --- Constants ---
# --- Main Settings ---
IMAGE_FILE = r'C:\Users\DAIICT B\Downloads\202301258_Rishik\MMP\MMP_Sub\smtgimg_to_road\g.png'

ROAD_WIDTH_PX = 40  # Visual width of the road in the simulation

# --- Pygame Display ---
WIDTH, HEIGHT = 1200, 900
BACKGROUND_COLOR = (240, 240, 240)
EDGE_COLOR = (0, 100, 200)
PATH_COLOR = (0, 200, 50)
CAR_COLOR = (200, 100, 0)
ZOOM_FACTOR = 1.1
PADDING = 50

# --- Car Simulation (in PIXEL units) ---
SPEED_STRAIGHT_PXPS = 50.0
SPEED_CURVE_PXPS = 15.0
MAX_ACCELERATION_PXPS = 10.0
MAX_STEER_ANGLE = math.pi / 4
CAR_LENGTH_PX = 20.0
CAR_WIDTH_PX = 10.0
WAYPOINT_THRESHOLD_PX = 30.0

# --- Controller Look-ahead ---
LOOK_AHEAD_STEERING_IDX = 4
LOOK_AHEAD_SPEED_IDX = 12
CURVE_THRESHOLD_DOT = 0.97

# --- Car Class ---
class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed_pxps = 0.0
        self.steer_angle = 0.0
        self.length = CAR_LENGTH_PX
        self.width = CAR_WIDTH_PX
        self.path_waypoints = []
        self.current_waypoint_idx = 0

    def move(self, throttle, steer_input, dt):
        self.speed_pxps += throttle * dt
        self.steer_angle = np.clip(steer_input, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)

        if self.steer_angle != 0:
            turn_radius = self.length / math.tan(self.steer_angle)
            angular_velocity = self.speed_pxps / turn_radius
            self.angle += angular_velocity * dt
        
        self.x += self.speed_pxps * math.cos(self.angle) * dt
        self.y += self.speed_pxps * math.sin(self.angle) * dt

    def draw(self, screen, g_to_s):
        car_center_px = g_to_s((self.x, self.y))
        car_surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        car_surface.fill(CAR_COLOR)
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        rect = rotated_surface.get_rect(center=car_center_px)
        screen.blit(rotated_surface, rect.topleft)

# --- Graph and Pathing Functions ---
def build_graph_from_skeleton(skeleton):
    """
    Builds a NetworkX graph from a skeletonized image.
    Nodes are junctions and endpoints. Edges are the paths between them.
    """
    # Use a 3x3 kernel to count neighbors
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    
    # Convolve the skeleton with the kernel to count neighbors for each pixel
    # We only care about skeleton pixels, so we multiply by the skeleton mask
    neighbor_counts = convolve2d(skeleton, kernel, mode='same', boundary='fill', fillvalue=0)
    neighbor_counts = neighbor_counts * (skeleton > 0)

    # **FIXED**: Endpoints have 1 neighbor. Junctions have more than 2.
    endpoints = np.argwhere((neighbor_counts == 1))
    junctions = np.argwhere((neighbor_counts > 2))
    
    nodes = np.concatenate([endpoints, junctions], axis=0)
    # Map pixel coordinates (x,y) to a unique node ID
    node_map = {tuple(node[::-1]): i for i, node in enumerate(nodes)}
    
    G = nx.Graph()
    for i, node_pos in enumerate(nodes):
        G.add_node(i, pos=tuple(node_pos[::-1])) # Store position as (x,y)

    # Trace paths between nodes to create edges
    visited_pixels = set()
    for start_node_idx, start_node_pos_yx in enumerate(nodes):
        y, x = start_node_pos_yx
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                
                neighbor_y, neighbor_x = y + dy, x + dx
                
                if 0 <= neighbor_y < skeleton.shape[0] and 0 <= neighbor_x < skeleton.shape[1] and \
                   skeleton[neighbor_y, neighbor_x] > 0 and (neighbor_y, neighbor_x) not in visited_pixels:
                    
                    path = [(neighbor_x, neighbor_y)]
                    visited_pixels.add((neighbor_y, neighbor_x))
                    current_pos_yx = (neighbor_y, neighbor_x)
                    
                    while tuple(current_pos_yx[::-1]) not in node_map:
                        found_next = False
                        cy, cx = current_pos_yx
                        for ddy in [-1, 0, 1]:
                            for ddx in [-1, 0, 1]:
                                if ddy == 0 and ddx == 0: continue
                                next_y, next_x = cy + ddy, cx + ddx
                                next_pixel = (next_y, next_x)
                                if 0 <= next_y < skeleton.shape[0] and 0 <= next_x < skeleton.shape[1] and \
                                   skeleton[next_y, next_x] > 0 and next_pixel not in visited_pixels:
                                    path.append((next_x, next_y))
                                    visited_pixels.add(next_pixel)
                                    current_pos_yx = (next_y, next_x)
                                    found_next = True
                                    break
                            if found_next: break
                        if not found_next: break
                    
                    end_node_pos_xy = tuple(current_pos_yx[::-1])
                    if end_node_pos_xy in node_map:
                        end_node_idx = node_map[end_node_pos_xy]
                        G.add_edge(start_node_idx, end_node_idx, path=path, weight=len(path))
    return G

def find_nearest_point_on_graph(graph, point):
    """Finds the closest point on any edge in the graph to a given (x,y) point."""
    min_dist = float('inf')
    closest_point_info = {}
    
    for u, v, data in graph.edges(data=True):
        path = data.get('path', [])
        for i, p in enumerate(path):
            dist = np.linalg.norm(np.array(p) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                closest_point_info = {
                    'point': p,
                    'edge': (u, v),
                    'index_on_path': i,
                }
    return closest_point_info

def get_path_segment(path_list, start_idx, end_idx):
    """Helper to get a sub-path, handling forward or reverse traversal."""
    if start_idx <= end_idx:
        return path_list[start_idx : end_idx + 1]
    else:
        return path_list[start_idx : end_idx - 1 : -1]

def calculate_path(graph, start_info, end_info):
    """
    **NEW/IMPROVED**: Calculates the optimal pixel path between two points on the graph.
    This version is more robust by checking all 4 possible node-to-node routes.
    """
    start_edge = start_info['edge']
    end_edge = end_info['edge']

    # Simple case: start and end are on the same road segment
    if start_edge == end_edge:
        path_segment = graph.edges[start_edge]['path']
        waypoint_tuples = get_path_segment(path_segment, start_info['index_on_path'], end_info['index_on_path'])
        # **THE FIX IS HERE**: Convert the list of tuples to a list of NumPy arrays
        return [np.array(p) for p in waypoint_tuples]

    # Complex case: start and end are on different edges
    # (The rest of this function is correct and does not need to be changed)
    su1, su2 = start_edge
    eu1, eu2 = end_edge
    
    paths_to_test = [
        (su1, eu1), (su1, eu2),
        (su2, eu1), (su2, eu2)
    ]

    best_path = []
    min_total_dist = float('inf')

    for start_node, end_node in paths_to_test:
        try:
            # 1. Get the path of nodes from NetworkX
            node_path = nx.shortest_path(graph, source=start_node, target=end_node, weight='weight')
            
            # 2. Get the pixel path from the car to the first node
            start_edge_path = graph.edges[start_edge]['path']
            start_node_pos = graph.nodes[start_node]['pos']
            start_node_idx_on_path = start_edge_path.index(start_node_pos)
            path_to_first_node = get_path_segment(start_edge_path, start_info['index_on_path'], start_node_idx_on_path)

            # 3. Get the pixel path from the last node to the destination
            end_edge_path = graph.edges[end_edge]['path']
            end_node_pos = graph.nodes[end_node]['pos']
            end_node_idx_on_path = end_edge_path.index(end_node_pos)
            path_from_last_node = get_path_segment(end_edge_path, end_node_idx_on_path, end_info['index_on_path'])

            # 4. Stitch together the path between the intermediate nodes
            intermediate_path = []
            for i in range(len(node_path) - 1):
                u, v = node_path[i], node_path[i+1]
                edge_path = graph.edges[(u,v)]['path']
                if graph.nodes[u]['pos'] != edge_path[0]:
                    edge_path = edge_path[::-1]
                intermediate_path.extend(edge_path)
            
            # 5. Assemble and calculate total distance
            full_path = path_to_first_node + intermediate_path + path_from_last_node
            total_dist = len(full_path)
            
            if total_dist < min_total_dist:
                min_total_dist = total_dist
                best_path = full_path

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
            
    return [np.array(p) for p in best_path]

# --- Coordinate and Drawing Functions ---
def grid_to_screen(pos, scale, pan): return (int(pos[0] * scale + pan[0]), int(pos[1] * scale + pan[1]))
def screen_to_grid(pos, scale, pan): return ((pos[0] - pan[0]) / scale, (pos[1] - pan[1]) / scale)

# --- Main Simulation ---
def run_simulation(graph, img_shape):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Interactive Road Network Simulation")
    clock = pygame.time.Clock()
    
    car = None # No car initially
    
    # --- View Setup ---
    scale = min((WIDTH - PADDING * 2) / img_shape[1], (HEIGHT - PADDING * 2) / img_shape[0])
    pan = [PADDING, PADDING]
    mouse_dragging = False
    last_mouse_pos = None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    grid_pos = screen_to_grid(event.pos, scale, pan)
                    if car is None:
                        # SPAWN CAR
                        spawn_info = find_nearest_point_on_graph(graph, grid_pos)
                        if spawn_info:
                            pos = spawn_info['point']
                            car = Car(pos[0], pos[1])
                            print(f"Car spawned at {pos}")
                    else:
                        # SET DESTINATION
                        start_info = find_nearest_point_on_graph(graph, (car.x, car.y))
                        end_info = find_nearest_point_on_graph(graph, grid_pos)
                        if start_info and end_info:
                            print("Calculating new path...")
                            path = calculate_path(graph, start_info, end_info)
                            if path:
                                car.path_waypoints = path
                                car.current_waypoint_idx = 0
                                print(f"New path found with {len(path)} waypoints.")
                elif event.button == 3: # Right Click for dragging
                    mouse_dragging = True; last_mouse_pos = event.pos
                elif event.button in (4, 5): # Zoom
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos = screen_to_grid(event.pos, scale, pan)
                    scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]
                    pan[1] += event.pos[1] - new_screen_pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3: mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0] += dx; pan[1] += dy
                last_mouse_pos = event.pos

        # --- Car AI Logic ---
        if car and car.path_waypoints:
            if car.current_waypoint_idx >= len(car.path_waypoints) - 1:
                car.path_waypoints = []; throttle = -MAX_ACCELERATION_PXPS; steer_input = 0
            else:
                car_pos = np.array([car.x, car.y])
                target_waypoint = car.path_waypoints[car.current_waypoint_idx]
                if np.linalg.norm(target_waypoint - car_pos) < WAYPOINT_THRESHOLD_PX:
                    car.current_waypoint_idx += 1
                
                # Speed Control
                speed_idx = min(car.current_waypoint_idx + LOOK_AHEAD_SPEED_IDX, len(car.path_waypoints) - 1)
                next_idx = min(car.current_waypoint_idx + 1, len(car.path_waypoints) - 1)
                v1 = car.path_waypoints[next_idx] - car_pos
                v2 = car.path_waypoints[speed_idx] - car.path_waypoints[next_idx]
                is_straight = True
                if np.linalg.norm(v1) > 1 and np.linalg.norm(v2) > 1:
                    dot = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
                    is_straight = dot > CURVE_THRESHOLD_DOT
                desired_speed = SPEED_STRAIGHT_PXPS if is_straight else SPEED_CURVE_PXPS
                throttle = np.clip(desired_speed - car.speed_pxps, -MAX_ACCELERATION_PXPS, MAX_ACCELERATION_PXPS)

                # Steering Control
                steer_idx = min(car.current_waypoint_idx + LOOK_AHEAD_STEERING_IDX, len(car.path_waypoints) - 1)
                vec_to_target = car.path_waypoints[steer_idx] - car_pos
                angle_to_target = math.atan2(vec_to_target[1], vec_to_target[0])
                angle_err = (angle_to_target - car.angle + math.pi) % (2 * math.pi) - math.pi
                steer_input = np.clip(angle_err * 2.5, -MAX_STEER_ANGLE, MAX_STEER_ANGLE)
            
            car.move(throttle, steer_input, dt)

        # --- Drawing ---
        screen.fill(BACKGROUND_COLOR)
        g_to_s = lambda pos: grid_to_screen(pos, scale, pan)
        
        # Draw road network
        for u, v, data in graph.edges(data=True):
            path_px = [g_to_s(p) for p in data['path']]
            if len(path_px) > 1:
                pygame.draw.lines(screen, EDGE_COLOR, False, path_px, max(1, int(ROAD_WIDTH_PX * scale)))
        
        # Draw car's current path
        if car and car.path_waypoints:
            path_px = [g_to_s(p) for p in car.path_waypoints[car.current_waypoint_idx:]]
            if len(path_px) > 1:
                pygame.draw.lines(screen, PATH_COLOR, False, path_px, 3)

        if car:
            car.draw(screen, g_to_s)
        
        pygame.display.flip()

    pygame.quit()

# --- Main Execution Block ---
if __name__ == '__main__':
    if not os.path.exists(IMAGE_FILE):
        print(f"ERROR: Image file not found at '{IMAGE_FILE}'.")
    else:
        print("Processing image to build road graph...")
        img = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Convert mask to 0s and 1s for skeletonization
        skeleton_input = (binary_mask / 255).astype(np.uint8)
        skeleton = skeletonize(skeleton_input).astype(np.uint8)
        
        road_graph = build_graph_from_skeleton(skeleton)
        print(f"Graph built with {road_graph.number_of_nodes()} nodes and {road_graph.number_of_edges()} edges.")
        
        print("\n--- Starting Simulation ---")
        print("Left-click to spawn car.")
        print("Left-click again to set a destination.")
        print("Right-click and drag to pan.")
        print("Scroll wheel to zoom.")
        
        run_simulation(road_graph, img.shape)