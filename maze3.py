import pygame
import networkx as nx
import math
import random
from collections import deque, defaultdict

# --- Constants ---
# Screen dimensions
WIDTH, HEIGHT = 900, 900

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100) # Road color
DARK_GRAY = (50, 50, 50) # Intersection color
RED = (255, 0, 0)       # Load sites
GREEN = (0, 200, 0)     # Store sites
BLUE = (0, 0, 255)      # Fuel sites
PURPLE = (128, 0, 128)  # Park sites
HUD_BG = (240, 240, 240)
HUD_TEXT = (10, 10, 10)

# --- NEW: Road and View Constants ---
ROAD_WIDTH_PX = 50      # <<< Much wider roads
SITE_RADIUS_PX = 25     # <<< Scaled up site markers
ZOOM_FACTOR = 1.1       # How fast to zoom
PADDING = 50            # Initial padding

# --- Constants from Maze Code ---
GRID_N, REMOVE_RATIO        = 25, 0.60
DEAD_END_PRUNE_RATIO        = 0.50
EDGE_MIN_KM, EDGE_MAX_KM    = 1.0, 2.0
LOAD_SITES, MAX_LOAD_CAP    = 5, 4
FUEL_SITES, STORE_SITES     = 3, 5
PARK_BAYS                   = 3

# --- Helper functions from Maze Code ---
def _perp(dx,dy): 
    ux,uy = -dy,dx
    n = math.hypot(ux,uy)
    return (0,0) if n==0 else (ux/n,uy/n)

def _ctrl(p0,p2,s=.35):
    (x0,y0),(x2,y2)=p0,p2
    mx,my=((x0+x2)/2,(y0+y2)/2)
    dx,dy=x2-x0,y2-y0
    px,py=_perp(dx,dy)
    return mx+random.choice((-1,1))*s*px, my+random.choice((-1,1))*s*py

def _qbez(p0,p1,p2,t):
    """Gets a point on a quadratic Bezier curve."""
    u=1-t
    return (u*u*p0[0]+2*u*t*p1[0]+t*t*p2[0],
            u*u*p0[1]+2*u*t*p1[1]+t*t*p2[1])

# --- Maze Generation Functions from Maze Code ---
def build_maze(n,drop):
    G = nx.grid_2d_graph(n,n)
    POS = {(x,y):(float(x),float(y)) for x,y in G}
    for u,v in G.edges():
        wt  = random.uniform(EDGE_MIN_KM, EDGE_MAX_KM)
        G.edges[u,v]['weight'] = wt
        G.edges[u,v]['ctrl']   = _ctrl(POS[u], POS[v])

    original_edges = list(G.edges(data=True))
    edges_to_remove = list(G.edges()); random.shuffle(edges_to_remove)
    for u,v in edges_to_remove[:int(drop*len(edges_to_remove))]:
        G.remove_edge(u,v)
        if not nx.is_connected(G):
            d = next((d for ou,ov,d in original_edges if (ou,ov)==(u,v) or (ou,ov)==(v,u)), None)
            if d: G.add_edge(u, v, **d)

    for u,v in [((8,18),(9,18)),((11,14),(12,14)),((19,21),(19,22))]:
        if (u in G and v in G) and not G.has_edge(u,v):
            d = next((d for ou,ov,d in original_edges if (ou,ov)==(u,v) or (ou,ov)==(v,u)), None)
            if d: G.add_edge(u, v, **d)
            else:
                wt = random.uniform(EDGE_MIN_KM, EDGE_MAX_KM)
                G.add_edge(u, v, weight=wt, ctrl=_ctrl(POS[u], POS[v]))
    return G, POS

def prune_leaves(G,keep):
    leaves=[n for n in G if G.degree[n]==1 and n not in keep]
    random.shuffle(leaves)
    for n in leaves[:int(len(leaves)*DEAD_END_PRUNE_RATIO)]:
        if n in G: G.remove_node(n)
    return G

def find_closest_node(target_coord, available_nodes):
    return min(available_nodes, key=lambda node: math.dist(node, target_coord))

# --- NEW Coordinate and Drawing Functions ---

def grid_to_screen(grid_pos, scale, pan_offset):
    """Converts (0-24) grid coordinates to screen pixels."""
    px = (grid_pos[0] * scale) + pan_offset[0]
    py = (grid_pos[1] * scale) + pan_offset[1]
    return (int(px), int(py))

def screen_to_grid(screen_pos, scale, pan_offset):
    """Converts screen pixels back to (0-24) grid coordinates."""
    gx = (screen_pos[0] - pan_offset[0]) / scale
    gy = (screen_pos[1] - pan_offset[1]) / scale
    return (gx, gy)

def draw_maze_network(screen, G, POS, SPECIAL, grid_to_screen_func, font):
    """Draws the entire networkx maze using curved roads and sites."""
    
    # 1. Draw Roads
    for u, v, data in G.edges(data=True):
        p0_grid = POS[u]
        p2_grid = POS[v]
        p1_grid = data['ctrl'] # The curve control point
        
        curve_points_grid = []
        num_segments = 20
        for i in range(num_segments + 1):
            t = i / num_segments
            point_grid = _qbez(p0_grid, p1_grid, p2_grid, t)
            curve_points_grid.append(point_grid)
        
        curve_points_px = [grid_to_screen_func(p) for p in curve_points_grid]
        
        # Draw the wide road
        pygame.draw.lines(screen, GRAY, False, curve_points_px, ROAD_WIDTH_PX)

    # 2. Draw Intersections (to cover road ends smoothly)
    for node in G.nodes():
        pos_px = grid_to_screen_func(POS[node])
        # Draw a hub with a radius slightly larger than the road's half-width
        pygame.draw.circle(screen, DARK_GRAY, pos_px, ROAD_WIDTH_PX // 2 + 2)

    # 3. Draw Special Sites
    site_colors = {'L': RED, 'S': GREEN, 'F': BLUE, 'P': PURPLE}
    
    for label, node in SPECIAL.items():
        if node not in G: continue
        pos_grid = POS[node]
        pos_px = grid_to_screen_func(pos_grid)
        color = site_colors.get(label[0], WHITE)
        
        pygame.draw.circle(screen, color, pos_px, SITE_RADIUS_PX)
        pygame.draw.circle(screen, BLACK, pos_px, SITE_RADIUS_PX, 3) # Thicker outline
        
        # Draw label
        label_text = font.render(label, True, BLACK)
        text_rect = label_text.get_rect(center=(pos_px[0], pos_px[1] - SITE_RADIUS_PX - 10))
        screen.blit(label_text, text_rect)

# --- Main simulation ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pannable Maze Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16, bold=True)
    hud_font = pygame.font.SysFont("Arial", 18, bold=True)

    # --- Generate the maze ---
    random.seed(7)
    G, POS = build_maze(GRID_N, REMOVE_RATIO)
    
    # --- Place Sites ---
    available_nodes = list(G.nodes())
    target_coords = {
        'L1': (5,20), 'L2': (20,23), 'L4': (10,17), 'S4': (15,19), 'L5': (18,10),
        'S3': (21,15), 'F1': (12,13), 'S5': (14,11), 'S1': (22,6), 'F2': (8,5),
        'S2': (4,6),  'P1': (3,22),  'P2': (23,2),  'P3': (2,4),  'F3': (18,21),
        'L3': (3,16),
    }
    site_count_var_map = {'L':'LOAD_SITES','S':'STORE_SITES','F':'FUEL_SITES','P':'PARK_BAYS'}
    site_nodes = {}
    
    sim_globals = globals() 
    
    for code, target in target_coords.items():
        if not available_nodes: break
        node_type = code[0]
        limit_var_name = site_count_var_map.get(node_type)
        if limit_var_name and len(site_nodes.get(node_type, [])) < sim_globals[limit_var_name]:
            closest = find_closest_node(target, available_nodes)
            if closest in available_nodes:
                site_nodes[node_type] = site_nodes.get(node_type, []) + [closest]
                available_nodes.remove(closest)

    park_nodes_list  = site_nodes.get('P', [])
    if not park_nodes_list:
        park_nodes_list = [random.choice(list(G.nodes()))]
        site_nodes['P'] = park_nodes_list

    SPECIAL = {**{f'L{i+1}':n for i,n in enumerate(site_nodes.get('L', []))},
               **{f'S{i+1}':n for i,n in enumerate(site_nodes.get('S', []))},
               **{f'F{i+1}':n for i,n in enumerate(site_nodes.get('F', []))},
               **{f'P{i+1}':n for i,n in enumerate(park_nodes_list)}}
    
    G = prune_leaves(G, set(SPECIAL.values()))

    # --- NEW: View Control Variables ---
    scale = (min(WIDTH, HEIGHT) - PADDING * 2) / GRID_N # Initial scale
    pan_offset_px = [PADDING, PADDING]                  # Initial offset
    mouse_dragging = False
    last_mouse_pos = None

    # --- Main Loop ---
    running = True
    
    while running:
        clock.tick(60)
        
        # --- Event Handling for Panning and Zooming ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # --- Panning ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    mouse_dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
                
                # --- Zooming ---
                elif event.button == 4: # Scroll up
                    mouse_pos_screen = pygame.mouse.get_pos()
                    grid_pos_before = screen_to_grid(mouse_pos_screen, scale, pan_offset_px)
                    
                    scale *= ZOOM_FACTOR
                    
                    screen_pos_after_scale = grid_to_screen(grid_pos_before, scale, pan_offset_px)
                    
                    pan_offset_px[0] += mouse_pos_screen[0] - screen_pos_after_scale[0]
                    pan_offset_px[1] += mouse_pos_screen[1] - screen_pos_after_scale[1]

                elif event.button == 5: # Scroll down
                    mouse_pos_screen = pygame.mouse.get_pos()
                    grid_pos_before = screen_to_grid(mouse_pos_screen, scale, pan_offset_px)
                    
                    scale /= ZOOM_FACTOR
                    
                    screen_pos_after_scale = grid_to_screen(grid_pos_before, scale, pan_offset_px)
                    
                    pan_offset_px[0] += mouse_pos_screen[0] - screen_pos_after_scale[0]
                    pan_offset_px[1] += mouse_pos_screen[1] - screen_pos_after_scale[1]

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: # Left click release
                    mouse_dragging = False
                    last_mouse_pos = None
            
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    current_mouse_pos = pygame.mouse.get_pos()
                    dx = current_mouse_pos[0] - last_mouse_pos[0]
                    dy = current_mouse_pos[1] - last_mouse_pos[1]
                    
                    pan_offset_px[0] += dx
                    pan_offset_px[1] += dy
                    last_mouse_pos = current_mouse_pos

        # --- Drawing ---
        screen.fill(WHITE)
        
        # Create a "lambda" function to pass the current scale and pan
        # to the drawing function.
        g_to_s_func = lambda pos: grid_to_screen(pos, scale, pan_offset_px)
        
        draw_maze_network(screen, G, POS, SPECIAL, g_to_s_func, font)
        
        # --- Draw HUD ---
        hud_surface = pygame.Surface((250, 70), pygame.SRCALPHA)
        hud_surface.fill((*HUD_BG, 200)) # Semi-transparent background
        pygame.draw.rect(hud_surface, BLACK, hud_surface.get_rect(), 1) # Border

        text1 = hud_font.render("Left Click + Drag to Pan", True, HUD_TEXT)
        text2 = hud_font.render("Mouse Wheel to Zoom", True, HUD_TEXT)
        text3 = hud_font.render(f"Zoom: {scale:.1f}x", True, HUD_TEXT)
        
        hud_surface.blit(text1, (10, 5))
        hud_surface.blit(text2, (10, 25))
        hud_surface.blit(text3, (10, 45))
        
        screen.blit(hud_surface, (10, 10))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()

