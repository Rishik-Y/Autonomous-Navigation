import pygame
import networkx as nx
import math
import random
from collections import deque, defaultdict

# --- Constants ---
# Screen dimensions
WIDTH, HEIGHT = 800, 800

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100) # Road color
DARK_GRAY = (50, 50, 50) # Intersection color
RED = (255, 0, 0)       # Load sites
GREEN = (0, 200, 0)     # Store sites
BLUE = (0, 0, 255)      # Fuel sites
PURPLE = (128, 0, 128)  # Park sites

# Road drawing
ROAD_WIDTH_PX = 8
SITE_RADIUS_PX = 10

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
    # POS is the position in (0-24, 0-24) grid coordinates
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

    # Manual connectivity fixes (optional, but in original)
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

# --- NEW Pygame Drawing Functions ---

def draw_maze_network(screen, G, POS, SPECIAL, grid_to_screen_func, font):
    """Draws the entire networkx maze using curved roads and sites."""
    
    # 1. Draw Roads
    for u, v, data in G.edges(data=True):
        p0_grid = POS[u]
        p2_grid = POS[v]
        p1_grid = data['ctrl'] # The curve control point
        
        # Sample the Bezier curve to create many small line segments
        curve_points_grid = []
        num_segments = 10 # More segments = smoother curve
        for i in range(num_segments + 1):
            t = i / num_segments
            point_grid = _qbez(p0_grid, p1_grid, p2_grid, t)
            curve_points_grid.append(point_grid)
        
        # Convert all points from grid coordinates to screen pixels
        curve_points_px = [grid_to_screen_func(p) for p in curve_points_grid]
        
        # Draw the road as a series of connected lines
        pygame.draw.lines(screen, GRAY, False, curve_points_px, ROAD_WIDTH_PX)

    # 2. Draw Intersections (so roads connect smoothly)
    for node in G.nodes():
        pos_px = grid_to_screen_func(POS[node])
        pygame.draw.circle(screen, DARK_GRAY, pos_px, ROAD_WIDTH_PX // 2 + 2)

    # 3. Draw Special Sites
    site_colors = {'L': RED, 'S': GREEN, 'F': BLUE, 'P': PURPLE}
    
    for label, node in SPECIAL.items():
        if node not in G: continue
        pos_grid = POS[node]
        pos_px = grid_to_screen_func(pos_grid)
        color = site_colors.get(label[0], WHITE)
        
        pygame.draw.circle(screen, color, pos_px, SITE_RADIUS_PX)
        pygame.draw.circle(screen, BLACK, pos_px, SITE_RADIUS_PX, 2) # Outline
        
        # Draw label
        label_text = font.render(label, True, BLACK)
        screen.blit(label_text, (pos_px[0] + 10, pos_px[1] - 10))

# --- Main simulation ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Networkx Maze Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14, bold=True)

    # --- Generate the maze ---
    random.seed(7); # Use a fixed seed for consistent maze
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
    
    # Use globals() to access the constants defined at the top
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
    
    # --- Prune leaves, keeping sites safe ---
    G = prune_leaves(G, set(SPECIAL.values()))

    # --- Setup Scaling ---
    # Fit the 25x25 grid into the screen with padding
    PADDING = 50
    # Calculate scale to fit grid (0-24) inside the padded area
    PIXELS_PER_GRID_UNIT = (min(WIDTH, HEIGHT) - PADDING * 2) / GRID_N
    
    # This function converts (0-24, 0-24) grid coords to screen pixels
    def grid_to_screen(grid_pos):
        px = grid_pos[0] * PIXELS_PER_GRID_UNIT + PADDING
        py = grid_pos[1] * PIXELS_PER_GRID_UNIT + PADDING
        return (int(px), int(py))

    # --- Main Loop ---
    running = True
    drawn = False # Only draw the static maze once
    
    while running:
        clock.tick(30) # No need to run at 60 FPS for a static image
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Only draw the maze one time, not every frame
        if not drawn:
            screen.fill(WHITE)
            draw_maze_network(screen, G, POS, SPECIAL, grid_to_screen, font)
            pygame.display.flip()
            drawn = True

    pygame.quit()

if __name__ == '__main__':
    main()
