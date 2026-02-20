"""
Coal Mine Editor - Configure coal capacities for load zones
Matches Map_Editor.py style and auto-syncs with map_data.py
"""
import pygame
import numpy as np
import json
import os

# --- EDITOR SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (100, 100, 100)
GREEN, RED, PURPLE, ORANGE = (0, 200, 0), (200, 0, 0), (150, 0, 150), (255, 165, 0)
COAL_COLOR_BASE = (139, 69, 19)  # Saddle brown for coal mines
HIGHLIGHT_COLOR = (255, 255, 0)

ROAD_WIDTH_M = 8.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS
CLICK_THRESHOLD_PX = 20

MAP_DATA_FILE = 'map_data.py'
CONFIG_FILE = 'mine_config.json'
DEFAULT_COAL_CAPACITY = 100
DEFAULT_TRUCK_COUNT = 5

# --- DATA STORAGE ---
NODES = {}
EDGES = []
LOAD_ZONES = []
DUMP_ZONES = []
VISUAL_ROAD_CHAINS = []

# Configuration data
config = {
    "truck_count": DEFAULT_TRUCK_COUNT,
    "coal_capacities": {}
}

# --- HELPER FUNCTIONS ---
def load_map_data():
    """Load map data from map_data.py"""
    global NODES, EDGES, LOAD_ZONES, DUMP_ZONES, VISUAL_ROAD_CHAINS
    print(f"Loading map data from {MAP_DATA_FILE}...")
    try:
        with open(MAP_DATA_FILE, 'r') as f:
            content = f.read()
        
        sandbox = {'np': np}
        exec(content, sandbox)

        NODES = sandbox.get('NODES', {})
        EDGES = sandbox.get('EDGES', [])
        LOAD_ZONES = sandbox.get('LOAD_ZONES', [])
        DUMP_ZONES = sandbox.get('DUMP_ZONES', [])
        VISUAL_ROAD_CHAINS = sandbox.get('VISUAL_ROAD_CHAINS', [])

        print(f"Loaded {len(NODES)} nodes, {len(LOAD_ZONES)} load zones (coal mines).")
    except Exception as e:
        print(f"Error loading map data: {e}")
        NODES, EDGES, LOAD_ZONES, DUMP_ZONES, VISUAL_ROAD_CHAINS = {}, [], [], [], []

def load_config():
    """Load configuration from mine_config.json"""
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            print(f"Loaded config: {config['truck_count']} trucks, {len(config['coal_capacities'])} mines configured.")
        except Exception as e:
            print(f"Error loading config: {e}")
            config = {"truck_count": DEFAULT_TRUCK_COUNT, "coal_capacities": {}}
    else:
        print(f"Config file not found, will create new one.")
        config = {"truck_count": DEFAULT_TRUCK_COUNT, "coal_capacities": {}}

def sync_config_with_map():
    """
    Synchronize config with current LOAD_ZONES from map_data.py.
    - Remove entries for mines that no longer exist
    - Add default capacity for new mines
    """
    global config
    
    current_mines = set(LOAD_ZONES)
    config_mines = set(config.get("coal_capacities", {}).keys())
    
    # Remove deleted mines from config
    removed = config_mines - current_mines
    for mine in removed:
        del config["coal_capacities"][mine]
        print(f"Removed deleted mine from config: {mine}")
    
    # Add new mines with default capacity
    added = current_mines - config_mines
    for mine in added:
        config["coal_capacities"][mine] = DEFAULT_COAL_CAPACITY
        print(f"Added new mine to config: {mine} (default: {DEFAULT_COAL_CAPACITY} kg)")
    
    if removed or added:
        save_config()
        print("Config synchronized with map data.")

def save_config():
    """Save configuration to mine_config.json"""
    global config
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4, sort_keys=True)
        print(f"Config saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def get_coal_color(capacity, max_capacity=500):
    """Get color intensity based on coal capacity (darker = more coal)"""
    ratio = min(capacity / max_capacity, 1.0)
    # Interpolate from light tan to dark brown
    r = int(222 - ratio * 83)   # 222 -> 139
    g = int(184 - ratio * 115)  # 184 -> 69
    b = int(135 - ratio * 116)  # 135 -> 19
    return (max(0, r), max(0, g), max(0, b))

# --- Drawing & Coordinate Functions ---
PRE_CALCULATED_SPLINES = []

def catmull_rom_point(t, p0, p1, p2, p3):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t**2) + (-p0 + 3 * p1 - 3 * p2 + p3) * (t**3))

def generate_curvy_path_from_nodes(node_list, points_per_segment=20):
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

def rebuild_splines():
    global PRE_CALCULATED_SPLINES
    PRE_CALCULATED_SPLINES = []
    for chain in VISUAL_ROAD_CHAINS:
        if not isinstance(chain, (list, tuple)): continue
        node_coords = [NODES[node_name] for node_name in chain if node_name in NODES]
        if len(node_coords) < 2: continue
        PRE_CALCULATED_SPLINES.append(generate_curvy_path_from_nodes(node_coords))
    print(f"Rebuilt {len(PRE_CALCULATED_SPLINES)} visual road splines.")

def grid_to_screen(pos_m, scale, pan):
    pos_m_np = np.array(pos_m)
    pos_px = pos_m_np * METERS_TO_PIXELS
    return (int(pos_px[0] * scale + pan[0]), int(pos_px[1] * scale + pan[1]))

def screen_to_grid(pos_px, scale, pan):
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return np.array([grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS])

def draw_road_network(screen, g_to_s, scale):
    """Draw road network (similar to Map_Editor)"""
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    
    # Draw road splines
    for waypoints in PRE_CALCULATED_SPLINES:
        if len(waypoints) < 2: continue
        road_px = [g_to_s(p) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    
    # Draw non-coal mine nodes (smaller, muted colors)
    for node_name, pos_m in NODES.items():
        if node_name in LOAD_ZONES:
            continue  # We draw these separately
        elif node_name in DUMP_ZONES:
            color = RED
        else:
            color = PURPLE
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(2, int(scale * 3)))

def draw_coal_mines(screen, g_to_s, scale, font, hovered_mine=None):
    """Draw coal mines with capacity visualization"""
    for mine_name in LOAD_ZONES:
        if mine_name not in NODES:
            continue
        
        pos_m = NODES[mine_name]
        pos_px = g_to_s(pos_m)
        capacity = config["coal_capacities"].get(mine_name, DEFAULT_COAL_CAPACITY)
        
        # Calculate radius based on capacity (min 8, max 25)
        base_radius = max(8, min(25, 8 + capacity / 20))
        radius = int(base_radius * scale)
        radius = max(6, min(30, radius))
        
        # Get color based on capacity
        color = get_coal_color(capacity)
        
        # Draw filled circle
        pygame.draw.circle(screen, color, pos_px, radius)
        
        # Draw border (highlight if hovered)
        border_color = HIGHLIGHT_COLOR if mine_name == hovered_mine else BLACK
        border_width = 3 if mine_name == hovered_mine else 2
        pygame.draw.circle(screen, border_color, pos_px, radius, border_width)
        
        # Draw capacity text on mine (if zoomed in enough)
        if scale > 0.5:
            cap_text = font.render(str(capacity), True, WHITE)
            text_rect = cap_text.get_rect(center=pos_px)
            screen.blit(cap_text, text_rect)

def get_mine_at_pos(pos_px, scale, pan):
    """Get the coal mine at the given screen position"""
    pos_m = screen_to_grid(pos_px, scale, pan)
    threshold_m = (CLICK_THRESHOLD_PX / scale) * PIXELS_TO_METERS
    
    for mine_name in LOAD_ZONES:
        if mine_name not in NODES:
            continue
        mine_pos_m = NODES[mine_name]
        if np.linalg.norm(pos_m - mine_pos_m) < threshold_m:
            return mine_name
    return None

def show_input_dialog(screen, font, title, current_value):
    """Show a simple text input dialog for editing values"""
    dialog_width, dialog_height = 300, 120
    dialog_x = (WIDTH - dialog_width) // 2
    dialog_y = (HEIGHT - dialog_height) // 2
    
    input_text = str(current_value)
    active = True
    
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    try:
                        return int(input_text) if input_text else current_value
                    except ValueError:
                        return current_value
                elif event.key == pygame.K_ESCAPE:
                    return None
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode.isdigit():
                    input_text += event.unicode
        
        # Draw dialog box
        pygame.draw.rect(screen, WHITE, (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 3)
        
        # Title
        title_surface = font.render(title, True, BLACK)
        screen.blit(title_surface, (dialog_x + 10, dialog_y + 10))
        
        # Input box
        input_rect = pygame.Rect(dialog_x + 10, dialog_y + 45, dialog_width - 20, 30)
        pygame.draw.rect(screen, (240, 240, 240), input_rect)
        pygame.draw.rect(screen, BLACK, input_rect, 2)
        
        input_surface = font.render(input_text + "|", True, BLACK)
        screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
        
        # Instructions
        hint_surface = font.render("Enter to confirm, Esc to cancel", True, GRAY)
        screen.blit(hint_surface, (dialog_x + 10, dialog_y + 85))
        
        pygame.display.flip()
    
    return None

# --- Main Editor Loop ---
def run_editor():
    global config
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Coal Mine Editor - Configure Coal Capacities")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)
    small_font = pygame.font.SysFont("Consolas", 12)

    # Load data
    load_map_data()
    load_config()
    sync_config_with_map()
    rebuild_splines()

    # Status text
    status_text = "Click on a mine to edit its coal capacity"
    
    # --- View State ---
    all_nodes_m = list(NODES.values()) if NODES else [np.array([0, 0])]
    min_x_m = min(p[0] for p in all_nodes_m)
    max_x_m = max(p[0] for p in all_nodes_m)
    min_y_m = min(p[1] for p in all_nodes_m)
    max_y_m = max(p[1] for p in all_nodes_m)
    map_w_m = max(1.0, max_x_m - min_x_m)
    map_h_m = max(1.0, max_y_m - min_y_m)
    
    scale = min(
        (WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS),
        (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS)
    ) if map_w_m > 0 and map_h_m > 0 else 1.0
    
    pan = [
        PADDING - (min_x_m * METERS_TO_PIXELS * scale),
        PADDING - (min_y_m * METERS_TO_PIXELS * scale)
    ]
    
    mouse_dragging = False
    last_mouse_pos = None
    hovered_mine = None

    running = True
    while running:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                # Save config
                if event.key == pygame.K_s:
                    if save_config():
                        status_text = "Configuration SAVED to mine_config.json"
                    else:
                        status_text = "ERROR saving configuration!"
                
                # Adjust truck count with +/- keys
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    config["truck_count"] = min(20, config["truck_count"] + 1)
                    status_text = f"Truck count: {config['truck_count']}"
                elif event.key == pygame.K_MINUS:
                    config["truck_count"] = max(1, config["truck_count"] - 1)
                    status_text = f"Truck count: {config['truck_count']}"
                
                # Edit truck count directly with T
                elif event.key == pygame.K_t:
                    new_count = show_input_dialog(screen, font, "Enter truck count:", config["truck_count"])
                    if new_count is not None:
                        config["truck_count"] = max(1, min(50, new_count))
                        status_text = f"Truck count set to: {config['truck_count']}"
                
                # Reload map data with R
                elif event.key == pygame.K_r:
                    load_map_data()
                    sync_config_with_map()
                    rebuild_splines()
                    status_text = "Reloaded map data and synced config"
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    clicked_mine = get_mine_at_pos(mouse_pos, scale, pan)
                    if clicked_mine:
                        current_cap = config["coal_capacities"].get(clicked_mine, DEFAULT_COAL_CAPACITY)
                        new_cap = show_input_dialog(screen, font, f"Coal capacity for {clicked_mine}:", current_cap)
                        if new_cap is not None:
                            config["coal_capacities"][clicked_mine] = max(0, new_cap)
                            status_text = f"Set {clicked_mine} capacity to {new_cap} kg"
                
                elif event.button == 3:  # Right click - start panning
                    mouse_dragging = True
                    last_mouse_pos = event.pos
                
                elif event.button in (4, 5):  # Scroll wheel - zoom
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos_m = screen_to_grid(mouse_pos, scale, pan)
                    scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos_m, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]
                    pan[1] += event.pos[1] - new_screen_pos[1]
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    dx = mouse_pos[0] - last_mouse_pos[0]
                    dy = mouse_pos[1] - last_mouse_pos[1]
                    pan[0] += dx
                    pan[1] += dy
                    last_mouse_pos = mouse_pos
                else:
                    # Update hovered mine
                    hovered_mine = get_mine_at_pos(mouse_pos, scale, pan)

        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)
        
        # Draw road network (faded)
        draw_road_network(screen, g_to_s, scale)
        
        # Draw coal mines with capacity visualization
        draw_coal_mines(screen, g_to_s, scale, small_font, hovered_mine)

        # --- HUD ---
        hud_texts = [
            f"CONTROLS: [S]ave | [T]ruck count | [+/-] Adjust trucks | [R]eload map",
            f"PAN/ZOOM: Right-Click+Drag / Mouse Wheel | CLICK on mine to edit",
            f"Trucks: {config['truck_count']} | Total Mines: {len(LOAD_ZONES)}",
            status_text
        ]
        
        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10 + i * 22))
        
        # Draw hover info
        if hovered_mine and hovered_mine in config["coal_capacities"]:
            capacity = config["coal_capacities"][hovered_mine]
            info_text = f"{hovered_mine}: {capacity} kg coal"
            info_surface = font.render(info_text, True, BLACK)
            # Draw near mouse cursor
            screen.blit(info_surface, (mouse_pos[0] + 15, mouse_pos[1] + 15))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_editor()
