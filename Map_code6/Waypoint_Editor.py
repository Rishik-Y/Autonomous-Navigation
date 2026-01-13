import pygame
import numpy as np
import map_data
import pickle

# --- SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (100, 100, 100)
PURPLE_NODE, WAYPOINT_COLOR = (150, 0, 150), (0, 150, 255)
ROAD_WIDTH_M = 4.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS
POINTS_PER_SEGMENT = 20

# --- WAYPOINT GENERATION ---
def catmull_rom_point(t, p0, p1, p2, p3):
    """Calculates a single point on a Catmull-Rom spline."""
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t**2) + (-p0 + 3 * p1 - 3 * p2 + p3) * (t**3))

def generate_curvy_path_from_nodes(node_list: list[np.ndarray], points_per_segment=POINTS_PER_SEGMENT) -> list[np.ndarray]:
    """Generates a smooth list of waypoint coordinates for a given list of nodes."""
    all_waypoints_m = []
    if not node_list or len(node_list) < 2: return []
    
    # Pad the node list to provide context for the spline at the start and end of the chain
    node_list_padded = [node_list[0]] + node_list + [node_list[-1]]

    for i in range(len(node_list_padded) - 3):
        p0, p1, p2, p3 = node_list_padded[i:i+4]
        
        # The first point of the very first segment is the first node itself
        if i == 0:
            all_waypoints_m.append(p1)

        # Generate the intermediate points for the segment between p1 and p2
        for j in range(1, points_per_segment + 1):
            t = j / float(points_per_segment)
            point = catmull_rom_point(t, p0, p1, p2, p3)
            all_waypoints_m.append(point)
    return all_waypoints_m

def generate_all_waypoints():
    """
    Generates waypoint data for all road chains defined in VISUAL_ROAD_CHAINS.
    Returns a dictionary mapping a chain (tuple of node names) to its waypoints (list of np.arrays).
    """
    print("Generating waypoints for all visual road chains...")
    waypoints_data = {}
    for chain in map_data.VISUAL_ROAD_CHAINS:
        # Ensure chain is a tuple for dictionary key compatibility
        chain_key = tuple(chain)
        node_coords = [map_data.NODES[node_name] for node_name in chain_key if node_name in map_data.NODES]
        
        if len(node_coords) < 2:
            continue
        
        waypoints = generate_curvy_path_from_nodes(node_coords)
        waypoints_data[chain_key] = waypoints
        
    print(f"Generated waypoints for {len(waypoints_data)} chains.")
    return waypoints_data

def save_waypoints_data(waypoints_data):
    """Saves the generated waypoint data to a pickle file."""
    filepath = 'waypoints.pkl'
    print(f"Saving {len(waypoints_data)} waypoint chains to {filepath}...")
    with open(filepath, 'wb') as f:
        pickle.dump(waypoints_data, f)
    print("Save successful!")

# --- DRAWING & COORDINATE FUNCTIONS ---
def grid_to_screen(pos_m, scale, pan):
    pos_m_np = np.array(pos_m)
    pos_px = pos_m_np * METERS_TO_PIXELS
    return (int(pos_px[0] * scale + pan[0]), int(pos_px[1] * scale + pan[1]))

def screen_to_grid(pos_px, scale, pan):
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return (grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS)

def draw_road_network(screen, g_to_s, scale, splines):
    """Draws the base road network and nodes."""
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    for waypoints in splines:
        if len(waypoints) < 2: continue
        road_px = [g_to_s(p) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    
    for node_name, pos_m in map_data.NODES.items():
        if node_name in map_data.LOAD_ZONES: color = (0, 200, 0)
        elif node_name in map_data.DUMP_ZONES: color = (200, 0, 0)
        else: color = PURPLE_NODE
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(2, int(scale * 4)))

def draw_waypoints(screen, g_to_s, scale, waypoints_map):
    """Draws the generated waypoints."""
    for waypoints_list in waypoints_map.values():
        for point_m in waypoints_list:
            pygame.draw.circle(screen, WAYPOINT_COLOR, g_to_s(point_m), max(1, int(scale * 1.5)))

# --- MAIN EDITOR LOOP ---
def run_waypoint_editor():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Waypoint Editor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    # Generate background visuals from the same function to ensure they match
    background_splines_map = generate_all_waypoints()
    
    generated_waypoints_map = {}
    status_text = "Press [A] to Generate, [S] to Save, [L] to Load existing"

    # --- View State ---
    all_nodes_m = list(map_data.NODES.values()) if map_data.NODES else [np.array([0,0])]
    min_x_m, max_x_m = min(p[0] for p in all_nodes_m), max(p[0] for p in all_nodes_m)
    min_y_m, max_y_m = min(p[1] for p in all_nodes_m), max(p[1] for p in all_nodes_m)
    map_w_m, map_h_m = max(1.0, max_x_m - min_x_m), max(1.0, max_y_m - min_y_m)
    scale = min((WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS))
    pan = [PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale)]
    mouse_dragging, last_mouse_pos = False, None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    generated_waypoints_map = generate_all_waypoints()
                    status_text = f"Generated waypoints for {len(generated_waypoints_map)} chains. Press [S] to save."
                elif event.key == pygame.K_s:
                    if generated_waypoints_map:
                        save_waypoints_data(generated_waypoints_map)
                        status_text = f"Saved waypoints to waypoints.pkl"
                    else:
                        status_text = "No waypoints generated. Press [A] first."
                elif event.key == pygame.K_l:
                    try:
                        with open('waypoints.pkl', 'rb') as f:
                            generated_waypoints_map = pickle.load(f)
                        status_text = f"Loaded {len(generated_waypoints_map)} waypoint chains from waypoints.pkl"
                    except FileNotFoundError:
                        status_text = "waypoints.pkl not found."
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3: # Right-click
                    mouse_dragging, last_mouse_pos = True, event.pos
                elif event.button in (4, 5): # Mouse wheel
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos_m = screen_to_grid(mouse_pos, scale, pan)
                    scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos_m, scale, pan)
                    pan[0] += mouse_pos[0] - new_screen_pos[0]
                    pan[1] += mouse_pos[1] - new_screen_pos[1]

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    dx, dy = mouse_pos[0] - last_mouse_pos[0], mouse_pos[1] - last_mouse_pos[1]
                    pan[0] += dx
                    pan[1] += dy
                    last_mouse_pos = mouse_pos
        
        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)
        
        # Draw the base map roads for context
        draw_road_network(screen, g_to_s, scale, background_splines_map.values())
        
        # Draw the generated waypoints that will be saved
        if generated_waypoints_map:
            draw_waypoints(screen, g_to_s, scale, generated_waypoints_map)

        # --- HUD ---
        hud_texts = [
            "Waypoint Editor",
            "CONTROLS: [A] Generate All | [S] Save All | [L] Load from file",
            "PAN/ZOOM: Right-Click+Drag / Mouse Wheel",
            status_text
        ]
        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_waypoint_editor()
