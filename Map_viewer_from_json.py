import pygame
import sys
import numpy as np
import json

# --- 1. DATA HANDLING (UPDATED to handle 'sites') ---
def load_map_from_json(filename="map_data.json"):
    """Loads map data including junctions, edges, and special sites."""
    print(f"Loading map data from {filename}...")
    try:
        with open(filename, 'r') as f:
            map_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error! The file '{filename}' was not found.")
        return None, None, None
    except json.JSONDecodeError:
        print(f"❌ Error! The file '{filename}' is not a valid JSON file.")
        return None, None, None

    junctions = {jid: {'pos': np.array(data['pos'])} for jid, data in map_data.get('junctions', {}).items()}
    edges = [{'shape': [np.array(p) for p in edge['shape']], 'width': edge['width']} for edge in map_data.get('edges', [])]
    # --- ADDED: Load sites ---
    sites = {sid: {'type': data['type'], 'pos': np.array(data['pos'])} for sid, data in map_data.get('sites', {}).items()}

    # Re-calculate fill radius for drawing junctions seamlessly
    max_road_width = 0
    if edges:
        max_road_width = max(edge['width'] for edge in edges)
    for j in junctions.values():
        j['fill_radius'] = max_road_width * 0.7

    print(f"✅ Success! Loaded {len(junctions)} junctions, {len(edges)} edges, and {len(sites)} sites.")
    return junctions, edges, sites

# --- 2. THE PYGAME VIEWER (UPDATED) ---
class MapViewer:
    def __init__(self, junctions, edges, sites): # ADDED sites
        pygame.init()
        self.junctions, self.edges, self.sites = junctions, edges, sites # ADDED sites
        self.width, self.height = 1200, 900
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("JSON Map Viewer | Right-Drag to Pan, Scroll to Zoom")

        # Colors
        self.BG_COLOR, self.ROAD_COLOR = (240, 240, 240), (100, 100, 100)
        self.DUMP_COLOR, self.COAL_COLOR = (80, 80, 90), (139, 69, 19) # Grey, Brown

        # Camera controls
        self.zoom, self.offset = 1.0, np.array([0.0, 0.0])
        self.panning, self.pan_start_pos = False, np.array([0, 0])
        self.center_map()

    def center_map(self):
        # --- MODIFIED to include all map elements in centering ---
        all_points = []
        if self.junctions: all_points.extend([j['pos'] for j in self.junctions.values()])
        if self.edges: all_points.extend(p for edge in self.edges for p in edge['shape'])
        if self.sites: all_points.extend([s['pos'] for s in self.sites.values()])

        if not all_points:
            self.offset = np.array([self.width / 2.0, self.height / 2.0])
            return

        all_points = np.array(all_points)
        min_coords, max_coords = np.min(all_points, axis=0), np.max(all_points, axis=0)
        map_center = (min_coords + max_coords) / 2.0
        map_size = np.where(max_coords - min_coords == 0, 1, max_coords - min_coords)
        zoom_x = self.width / map_size[0] * 0.9 if map_size[0] > 0 else 1
        zoom_y = self.height / map_size[1] * 0.9 if map_size[1] > 0 else 1
        self.zoom = min(zoom_x, zoom_y)
        self.offset = np.array([self.width / 2.0, self.height / 2.0]) - map_center * self.zoom

    def world_to_screen(self, world_pos):
        return np.array([world_pos[0], -world_pos[1]]) * self.zoom + self.offset

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.VIDEORESIZE: self.width, self.height = event.size
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Use right-click for panning to avoid accidental interaction
                if event.button == 3: self.panning, self.pan_start_pos = True, np.array(event.pos)
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3: self.panning = False
            if event.type == pygame.MOUSEMOTION and self.panning:
                self.offset += np.array(event.pos) - self.pan_start_pos
                self.pan_start_pos = np.array(event.pos)
            if event.type == pygame.MOUSEWHEEL:
                self.zoom *= (1.1 if event.y > 0 else 0.9)
        return True

    def draw(self):
        self.screen.fill(self.BG_COLOR)
        # Draw junctions to fill gaps
        for j_data in self.junctions.values():
            fill_radius = j_data['fill_radius'] * self.zoom
            if fill_radius > 1:
                pygame.draw.circle(self.screen, self.ROAD_COLOR, self.world_to_screen(j_data['pos']), fill_radius)
        # Draw roads
        for edge in self.edges:
            if len(edge['shape']) > 1:
                scaled_width = int(edge['width'] * self.zoom); scaled_width = max(2, scaled_width)
                screen_points = [self.world_to_screen(p) for p in edge['shape']]
                pygame.draw.lines(self.screen, self.ROAD_COLOR, False, screen_points, scaled_width)
                for p in screen_points:
                    pygame.draw.circle(self.screen, self.ROAD_COLOR, p, scaled_width / 2)

        # --- ADDED: Draw sites ---
        site_radius = int(10 * self.zoom)
        for site_id, data in self.sites.items():
            color = self.DUMP_COLOR if data['type'] == 'dump_site' else self.COAL_COLOR
            pos = self.world_to_screen(data['pos'])
            pygame.draw.rect(self.screen, color, (pos[0]-site_radius, pos[1]-site_radius, site_radius*2, site_radius*2))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    MAP_FILE = 'map_data_cleaned.json'
    # --- MODIFIED to load sites ---
    junctions, edges, sites = load_map_from_json(MAP_FILE)

    if junctions is not None and edges is not None and sites is not None:
        viewer = MapViewer(junctions, edges, sites)
        viewer.run()
    else:
        print("Could not start viewer. Please check the JSON file.")