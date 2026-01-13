import pygame
import numpy as np
import math
import os
import re
from ast import literal_eval

# --- EDITOR SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (100, 100, 100)
GREEN, RED, PURPLE = (0, 200, 0), (200, 0, 0), (150, 0, 150)
HIGHLIGHT_COLOR = (255, 255, 0, 150)

ROAD_WIDTH_M = 4.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS
CLICK_THRESHOLD_PX = 15

# --- DATA STORAGE ---
# We will manage the map data in memory and save it back to the file.
NODES = {}
EDGES = []
LOAD_ZONES = []
DUMP_ZONES = []
VISUAL_ROAD_CHAINS = []
MAP_DATA_FILE = 'map_data.py'

# --- HELPER FUNCTIONS ---
def load_map_data():
    global NODES, EDGES, LOAD_ZONES, DUMP_ZONES, VISUAL_ROAD_CHAINS
    print(f"Loading data from {MAP_DATA_FILE}...")
    try:
        with open(MAP_DATA_FILE, 'r') as f:
            content = f.read()
        
        # Create a sandbox to exec the file in and get the variables
        sandbox = {'np': np}
        exec(content, sandbox)

        NODES = sandbox.get('NODES', {})
        EDGES = sandbox.get('EDGES', [])
        LOAD_ZONES = sandbox.get('LOAD_ZONES', [])
        DUMP_ZONES = sandbox.get('DUMP_ZONES', [])
        VISUAL_ROAD_CHAINS = sandbox.get('VISUAL_ROAD_CHAINS', [])

        print(f"Loaded {len(NODES)} nodes, {len(EDGES)} edges, and {len(VISUAL_ROAD_CHAINS)} visual chains.")
    except Exception as e:
        print(f"An error occurred while loading map data: {e}")
        # Initialize with empty data if loading fails
        NODES, EDGES, LOAD_ZONES, DUMP_ZONES, VISUAL_ROAD_CHAINS = {}, [], [], [], []

def save_map_data():
    print(f"Saving data to {MAP_DATA_FILE}...")
    try:
        with open(MAP_DATA_FILE, 'w') as f:
            f.write("import numpy as np\n\n")
            f.write("# --- MAP DATA ---\n\n")

            # Write NODES
            f.write("NODES = {\n")
            for name, pos in sorted(NODES.items()):
                f.write(f'    "{name}": np.array([{pos[0]:.1f}, {pos[1]:.1f}]),\n')
            f.write("}\n\n")

            # Write EDGES
            f.write("EDGES = [\n")
            for edge in sorted(EDGES):
                f.write(f'    {edge},\n')
            f.write("]\n\n")

            # Write LOAD_ZONES
            f.write("LOAD_ZONES = [\n")
            for zone in sorted(LOAD_ZONES):
                f.write(f'    "{zone}",\n')
            f.write("]\n\n")

            # Write DUMP_ZONES
            f.write("DUMP_ZONES = [\n")
            for zone in sorted(DUMP_ZONES):
                f.write(f'    "{zone}",\n')
            f.write("]\n\n")

            # Write VISUAL_ROAD_CHAINS
            f.write("VISUAL_ROAD_CHAINS = [\n")
            # Do not sort a list of lists, as it's not comparable and will crash.
            for chain in VISUAL_ROAD_CHAINS:
                f.write(f'    {chain},\n')
            f.write("]\n")
        print("Save successful!")
    except Exception as e:
        print(f"Error saving map data: {e}")

# --- Drawing & Coordinate Functions ---
PRE_CALCULATED_SPLINES = []

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

def draw_grid(screen, scale, pan):
    # Simple grid for better placement
    for x in range(-1000, 1000, 50):
        start = grid_to_screen((x, -1000), scale, pan)
        end = grid_to_screen((x, 1000), scale, pan)
        pygame.draw.line(screen, (230, 230, 230), start, end, 1)
    for y in range(-1000, 1000, 50):
        start = grid_to_screen((-1000, y), scale, pan)
        end = grid_to_screen((1000, y), scale, pan)
        pygame.draw.line(screen, (230, 230, 230), start, end, 1)

def draw_road_network(screen, g_to_s, scale):
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    # Draw road splines
    for waypoints in PRE_CALCULATED_SPLINES:
        if len(waypoints) < 2: continue
        road_px = [g_to_s(p) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    
    # Draw Nodes
    for node_name, pos_m in NODES.items():
        color = PURPLE
        if node_name in LOAD_ZONES: color = GREEN
        elif node_name in DUMP_ZONES: color = RED
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(3, int(scale * 4)))

def get_node_at_pos(pos_px, scale, pan):
    pos_m = screen_to_grid(pos_px, scale, pan)
    threshold_m = (CLICK_THRESHOLD_PX / scale) * PIXELS_TO_METERS
    for name, node_pos_m in NODES.items():
        if np.linalg.norm(pos_m - node_pos_m) < threshold_m:
            return name
    return None

def find_line_segment_intersection(p1, p2, p3, p4):
    """
    Finds the intersection of two line segments.
    Returns the intersection point as a numpy array, or None if they don't intersect within their segments.
    """
    v1 = p2 - p1
    v2 = p4 - p3
    
    # Using numpy's cross product for 2D vectors
    cross_prod = np.cross(v1, v2)
    
    # Check for parallel or collinear lines
    if abs(cross_prod) < 1e-9:
        return None

    p1_p3 = p3 - p1
    
    t = np.cross(p1_p3, v2) / cross_prod
    u = np.cross(p1_p3, v1) / cross_prod

    # Check if the intersection is within the segments (using a small margin to avoid endpoints)
    if 0.001 < t < 0.999 and 0.001 < u < 0.999:
        return p1 + t * v1
    
    return None

def fix_intersections():
    """
    Finds intersections between road segments that don't have a node and adds one.
    This helps clean up the map where roads cross without a junction.
    """
    global NODES, EDGES, VISUAL_ROAD_CHAINS
    print("Attempting to fix intersections...")
    
    # Create a list of all segments from VISUAL_ROAD_CHAINS, defined by node-to-node connections
    all_segments = []
    for i, chain in enumerate(VISUAL_ROAD_CHAINS):
        for k in range(len(chain) - 1):
            n1_name, n2_name = chain[k], chain[k+1]
            if n1_name in NODES and n2_name in NODES:
                all_segments.append({
                    "p1": NODES[n1_name], "p2": NODES[n2_name],
                    "n1": n1_name, "n2": n2_name,
                    "chain_idx": i, "chain_pos": k
                })

    intersections_found = []
    for i in range(len(all_segments)):
        for j in range(i + 1, len(all_segments)):
            seg1 = all_segments[i]
            seg2 = all_segments[j]

            # Skip if segments share a node, as they are connected, not crossing
            if seg1['n1'] in (seg2['n1'], seg2['n2']) or seg1['n2'] in (seg2['n1'], seg2['n2']):
                continue

            intersection_point = find_line_segment_intersection(seg1['p1'], seg1['p2'], seg2['p1'], seg2['p2'])
            
            if intersection_point is not None:
                # Check if this intersection is already near an existing node
                threshold_m = ROAD_WIDTH_M / 2.0
                is_near_existing = False
                for node_pos in NODES.values():
                    if np.linalg.norm(intersection_point - node_pos) < threshold_m:
                        is_near_existing = True
                        break
                # Also check against other newly found intersections to avoid duplicates
                if not is_near_existing:
                    for existing_int_pt, _, _ in intersections_found:
                        if np.linalg.norm(intersection_point - existing_int_pt) < threshold_m:
                            is_near_existing = True
                            break
                
                if not is_near_existing:
                    intersections_found.append((intersection_point, seg1, seg2))

    if not intersections_found:
        print("No new intersections found to fix.")
        return 0

    print(f"Found {len(intersections_found)} intersections. Preparing modifications...")

    modifications = {}
    new_nodes = {}
    edges_to_add = set()
    edges_to_remove = set()

    for int_point, seg1, seg2 in intersections_found:
        num = len(new_nodes) + 1
        while f"purple_auto_fix_{num}" in NODES or f"purple_auto_fix_{num}" in new_nodes: num += 1
        new_name = f"purple_auto_fix_{num}"
        new_nodes[new_name] = int_point

        # Process both segments involved in the intersection
        for seg_info in [seg1, seg2]:
            c_idx, c_pos, n1, n2 = seg_info['chain_idx'], seg_info['chain_pos'], seg_info['n1'], seg_info['n2']
            
            if c_idx not in modifications: modifications[c_idx] = []
            modifications[c_idx].append((c_pos + 1, new_name))
            
            edges_to_remove.add(tuple(sorted((n1, n2))))
            edges_to_add.add(tuple(sorted((n1, new_name))))
            edges_to_add.add(tuple(sorted((new_name, n2))))

    print("Applying modifications...")
    # 1. Add new nodes to the main dictionary
    NODES.update(new_nodes)
    
    # 2. Update EDGES by removing old ones and adding new ones
    current_edges = set(EDGES)
    current_edges.difference_update(edges_to_remove)
    current_edges.update(edges_to_add)
    EDGES = sorted(list(current_edges)) # Keep it sorted

    # 3. Update VISUAL_ROAD_CHAINS by inserting the new nodes
    for chain_idx, mods in sorted(modifications.items(), key=lambda item: item[0]):
        # Sort modifications by position in reverse to not mess up indices during insertion
        mods.sort(key=lambda item: item[0], reverse=True)
        for pos, name in mods:
            VISUAL_ROAD_CHAINS[chain_idx].insert(pos, name)

    print(f"Successfully fixed {len(intersections_found)} intersections.")
    return len(intersections_found)

# --- Main Editor Loop ---
def run_editor():
    global NODES, EDGES, LOAD_ZONES, DUMP_ZONES, VISUAL_ROAD_CHAINS
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Map Editor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    load_map_data()
    rebuild_splines() # Initial spline generation

    # --- Editor State ---
    mode = 'add_purple'
    brush_color = PURPLE
    status_text = "Mode: ADD PURPLE"
    selection_start_node = None
    is_drawing_manual = False
    manual_path_px = []
    
    # --- View State ---
    all_nodes_m = list(NODES.values()) if NODES else [np.array([0,0])]
    min_x_m, max_x_m = min(p[0] for p in all_nodes_m), max(p[0] for p in all_nodes_m)
    min_y_m, max_y_m = min(p[1] for p in all_nodes_m), max(p[1] for p in all_nodes_m)
    map_w_m, map_h_m = max(1.0, max_x_m - min_x_m), max(1.0, max_y_m - min_y_m)
    scale = min((WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS)) if map_w_m > 0 and map_h_m > 0 else 1.0
    pan = [PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale)]
    mouse_dragging, last_mouse_pos = False, None

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        mouse_pos = pygame.mouse.get_pos()
        needs_rebuild = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            # --- Keyboard Input for Mode Change ---
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_map_data()
                    status_text = "SAVED to map_data.py"
                elif event.key == pygame.K_g:
                    mode, brush_color, status_text = 'add_green', GREEN, "Mode: ADD GREEN (Load Zone)"
                    selection_start_node = None
                elif event.key == pygame.K_r:
                    mode, brush_color, status_text = 'add_red', RED, "Mode: ADD RED (Dump Zone)"
                    selection_start_node = None
                elif event.key == pygame.K_p:
                    mode, brush_color, status_text = 'add_purple', PURPLE, "Mode: ADD PURPLE (Intermediate)"
                    selection_start_node = None
                elif event.key == pygame.K_w:
                    mode, brush_color, status_text = 'delete', WHITE, "Mode: DELETE"
                    selection_start_node = None
                elif event.key == pygame.K_c:
                    mode, brush_color, status_text = 'connect_start', HIGHLIGHT_COLOR, "Mode: CONNECT (Click first node)"
                    selection_start_node = None
                elif event.key == pygame.K_d:
                    mode, brush_color, status_text = 'disconnect_start', HIGHLIGHT_COLOR, "Mode: DISCONNECT (Click first node)"
                    selection_start_node = None
                elif event.key == pygame.K_m:
                    mode, brush_color, status_text = 'manual', PURPLE, "Mode: MANUAL DRAW (Click and drag)"
                    selection_start_node = None
                elif event.key == pygame.K_f:
                    num_fixed = fix_intersections()
                    if num_fixed > 0:
                        status_text = f"Fixed {num_fixed} intersections. SAVE your changes."
                        needs_rebuild = True
                    else:
                        status_text = "No intersections needed fixing."

            # --- Mouse Input for Editing ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    clicked_node = get_node_at_pos(mouse_pos, scale, pan)
                    
                    if mode == 'manual':
                        is_drawing_manual = True
                        manual_path_px = [mouse_pos]

                    elif mode.startswith('add'):
                        if not clicked_node:
                            pos_m = screen_to_grid(mouse_pos, scale, pan)
                            node_type = mode.split('_')[1]
                            num = 1
                            while f"{node_type}_auto_{num}" in NODES: num += 1
                            new_name = f"{node_type}_auto_{num}"
                            NODES[new_name] = pos_m
                            if mode == 'add_green': LOAD_ZONES.append(new_name)
                            elif mode == 'add_red': DUMP_ZONES.append(new_name)
                            status_text = f"Added node: {new_name}"
                            needs_rebuild = True

                    elif mode == 'delete':
                        if clicked_node:
                            del NODES[clicked_node]
                            if clicked_node in LOAD_ZONES: LOAD_ZONES.remove(clicked_node)
                            if clicked_node in DUMP_ZONES: DUMP_ZONES.remove(clicked_node)
                            EDGES = [e for e in EDGES if clicked_node not in e]
                            
                            # Split any visual chains containing the deleted node
                            new_visual_chains = []
                            for chain in VISUAL_ROAD_CHAINS:
                                if clicked_node in chain:
                                    try:
                                        idx = chain.index(clicked_node)
                                        part1 = chain[:idx]
                                        part2 = chain[idx+1:]
                                        if len(part1) > 1: new_visual_chains.append(part1)
                                        if len(part2) > 1: new_visual_chains.append(part2)
                                    except ValueError:
                                        pass # Should not happen
                                else:
                                    new_visual_chains.append(chain)
                            VISUAL_ROAD_CHAINS = new_visual_chains

                            status_text = f"Deleted node: {clicked_node}"
                            needs_rebuild = True

                    elif mode == 'connect_start':
                        if clicked_node:
                            selection_start_node = clicked_node
                            mode = 'connect_end'
                            status_text = f"Connecting from '{clicked_node}'. Click second node."
                    
                    elif mode == 'connect_end':
                        if clicked_node and clicked_node != selection_start_node:
                            new_edge = tuple(sorted((selection_start_node, clicked_node)))
                            if new_edge not in EDGES:
                                EDGES.append(new_edge)
                                VISUAL_ROAD_CHAINS.append(list(new_edge))
                                status_text = f"Connected {selection_start_node} to {clicked_node}"
                                needs_rebuild = True
                            else:
                                status_text = "Edge already exists."
                            selection_start_node = None
                            mode = 'connect_start'
                        else:
                            selection_start_node = None
                            mode = 'connect_start'
                            status_text = "Connection cancelled. Click first node."
                            
                    elif mode == 'disconnect_start':
                        if clicked_node:
                            selection_start_node = clicked_node
                            mode = 'disconnect_end'
                            status_text = f"Disconnecting from '{clicked_node}'. Click second node."

                    elif mode == 'disconnect_end':
                        if clicked_node and clicked_node != selection_start_node:
                            edge_to_remove = tuple(sorted((selection_start_node, clicked_node)))
                            if edge_to_remove in EDGES:
                                EDGES.remove(edge_to_remove)

                                # Handle splitting of long visual chains
                                new_visual_chains = []
                                chain_was_split = False
                                for chain in VISUAL_ROAD_CHAINS:
                                    # Try to find adjacent nodes in the chain and split
                                    try:
                                        # Search for start -> clicked
                                        idx = chain.index(selection_start_node)
                                        if idx + 1 < len(chain) and chain[idx+1] == clicked_node:
                                            part1 = chain[:idx+1]
                                            part2 = chain[idx+1:]
                                            if len(part1) > 1: new_visual_chains.append(part1)
                                            if len(part2) > 1: new_visual_chains.append(part2)
                                            chain_was_split = True
                                            continue
                                        
                                        # Search for clicked -> start
                                        idx = chain.index(clicked_node)
                                        if idx + 1 < len(chain) and chain[idx+1] == selection_start_node:
                                            part1 = chain[:idx+1]
                                            part2 = chain[idx+1:]
                                            if len(part1) > 1: new_visual_chains.append(part1)
                                            if len(part2) > 1: new_visual_chains.append(part2)
                                            chain_was_split = True
                                            continue
                                    except ValueError:
                                        pass # Node not in chain, or not in sequence
                                    
                                    new_visual_chains.append(chain)
                                
                                VISUAL_ROAD_CHAINS = new_visual_chains

                                # Fallback for simple 2-point chains that might have been missed
                                if not chain_was_split:
                                    chain_to_remove_1 = [selection_start_node, clicked_node]
                                    chain_to_remove_2 = [clicked_node, selection_start_node]
                                    if chain_to_remove_1 in VISUAL_ROAD_CHAINS:
                                        VISUAL_ROAD_CHAINS.remove(chain_to_remove_1)
                                    elif chain_to_remove_2 in VISUAL_ROAD_CHAINS:
                                        VISUAL_ROAD_CHAINS.remove(chain_to_remove_2)

                                status_text = f"Disconnected {selection_start_node} from {clicked_node}"
                                needs_rebuild = True
                            else:
                                status_text = "No direct edge exists to disconnect."
                            selection_start_node = None
                            mode = 'disconnect_start'
                        else:
                            selection_start_node = None
                            mode = 'disconnect_start'
                            status_text = "Disconnection cancelled. Click first node."


                elif event.button == 3:
                    mouse_dragging, last_mouse_pos = True, event.pos
                elif event.button in (4, 5):
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos_m = screen_to_grid(mouse_pos, scale, pan)
                    scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos_m, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]
                    pan[1] += event.pos[1] - new_screen_pos[1]

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and mode == 'manual' and is_drawing_manual:
                    is_drawing_manual = False
                    
                    if len(manual_path_px) > 1:
                        MIN_NODE_DISTANCE_M = 20.0
                        
                        new_chain_node_names = []
                        
                        manual_path_m = [screen_to_grid(p, scale, pan) for p in manual_path_px]
                        
                        last_added_node_pos_m = None

                        start_node_snap = get_node_at_pos(manual_path_px[0], scale, pan)
                        if start_node_snap:
                            new_chain_node_names.append(start_node_snap)
                            last_added_node_pos_m = NODES[start_node_snap]
                        else:
                            pos_m = manual_path_m[0]
                            num = 1
                            while f"purple_auto_{num}" in NODES: num += 1
                            new_name = f"purple_auto_{num}"
                            NODES[new_name] = pos_m
                            new_chain_node_names.append(new_name)
                            last_added_node_pos_m = pos_m

                        for i in range(1, len(manual_path_m)):
                            current_pos_m = manual_path_m[i]
                            if np.linalg.norm(current_pos_m - last_added_node_pos_m) > MIN_NODE_DISTANCE_M:
                                is_last_point = (i == len(manual_path_m) - 1)
                                if is_last_point:
                                    end_node_snap = get_node_at_pos(manual_path_px[i], scale, pan)
                                    if end_node_snap and end_node_snap != new_chain_node_names[-1]:
                                        new_chain_node_names.append(end_node_snap)
                                        last_added_node_pos_m = NODES[end_node_snap]
                                        continue 

                                num = 1
                                while f"purple_auto_{num}" in NODES: num += 1
                                new_name = f"purple_auto_{num}"
                                NODES[new_name] = current_pos_m
                                new_chain_node_names.append(new_name)
                                last_added_node_pos_m = current_pos_m

                        if len(new_chain_node_names) > 1:
                            deduped_chain = [new_chain_node_names[0]]
                            for i in range(1, len(new_chain_node_names)):
                                if new_chain_node_names[i] != new_chain_node_names[i-1]:
                                    deduped_chain.append(new_chain_node_names[i])
                            
                            if len(deduped_chain) > 1:
                                VISUAL_ROAD_CHAINS.append(deduped_chain)
                                for i in range(len(deduped_chain) - 1):
                                    node1, node2 = deduped_chain[i], deduped_chain[i+1]
                                    new_edge = tuple(sorted((node1, node2)))
                                    if new_edge not in EDGES:
                                        EDGES.append(new_edge)
                                status_text = f"Created manual road with {len(deduped_chain)} nodes."
                                needs_rebuild = True
                            else:
                                status_text = "Manual road too short after processing."
                        else:
                            status_text = "Manual road too short, cancelled."

                    manual_path_px = []

                if event.button == 3:
                    mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if is_drawing_manual:
                    if np.linalg.norm(np.array(mouse_pos) - np.array(manual_path_px[-1])) > 5:
                         manual_path_px.append(mouse_pos)
                elif mouse_dragging:
                    dx, dy = mouse_pos[0] - last_mouse_pos[0], mouse_pos[1] - last_mouse_pos[1]
                    pan[0] += dx
                    pan[1] += dy
                    last_mouse_pos = mouse_pos
        
        if needs_rebuild:
            rebuild_splines()

        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)
        
        draw_grid(screen, scale, pan)
        draw_road_network(screen, g_to_s, scale)

        if is_drawing_manual and len(manual_path_px) > 1:
            pygame.draw.lines(screen, PURPLE, False, manual_path_px, 3)

        # Draw brush/cursor
        if mode.startswith('add'):
            pygame.draw.circle(screen, brush_color, mouse_pos, 10, 2)
        elif mode == 'delete':
            pygame.draw.line(screen, brush_color, (mouse_pos[0]-10, mouse_pos[1]-10), (mouse_pos[0]+10, mouse_pos[1]+10), 3)
            pygame.draw.line(screen, brush_color, (mouse_pos[0]-10, mouse_pos[1]+10), (mouse_pos[0]+10, mouse_pos[1]-10), 3)
        elif mode.startswith('connect') or mode.startswith('disconnect'):
            hover_node = get_node_at_pos(mouse_pos, scale, pan)
            if hover_node:
                pygame.draw.circle(screen, HIGHLIGHT_COLOR, g_to_s(NODES[hover_node]), int(CLICK_THRESHOLD_PX / scale * 2), 4)
            if selection_start_node:
                start_pos_px = g_to_s(NODES[selection_start_node])
                pygame.draw.line(screen, HIGHLIGHT_COLOR, start_pos_px, mouse_pos, 2)


        # --- HUD ---
        hud_texts = [
            "CONTROLS: [G]reen | [R]ed | [P]urple | [W]hite/Delete | [C]onnect | [D]isconnect | [M]anual | [F]ix | [S]ave",
            "PAN/ZOOM: Right-Click+Drag / Mouse Wheel",
            status_text
        ]
        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_editor()
