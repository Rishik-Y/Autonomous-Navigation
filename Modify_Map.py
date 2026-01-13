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
        print(f"Info: '{filename}' not found. Starting with a blank canvas.")
        return {}, [], {} # Return empty sites dict
    except json.JSONDecodeError:
        print(f"Error! Could not parse '{filename}'. Starting with a blank canvas.")
        return {}, [], {} # Return empty sites dict

    junctions = {jid: {'pos': np.array(data['pos'])} for jid, data in map_data.get('junctions', {}).items()}
    edges = [{'shape': [np.array(p) for p in edge['shape']], 'width': edge['width']} for edge in map_data.get('edges', [])]
    # --- ADDED: Load sites ---
    sites = {sid: {'type': data['type'], 'pos': np.array(data['pos'])} for sid, data in map_data.get('sites', {}).items()}

    print(f"Loaded {len(junctions)} junctions, {len(edges)} edges, and {len(sites)} sites.")
    return junctions, edges, sites

# --- 2. THE MAP EDITOR CLASS (Features Re-implemented) ---
class MapEditor:
    def __init__(self, junctions, edges, sites): # ADDED sites
        pygame.init()
        self.junctions, self.edges, self.sites = junctions, edges, sites # ADDED sites
        self.width, self.height = 1200, 900
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Map Editor | Draw/Erase/Place Sites | Press 'S' to Save")

        # Colors and settings
        self.BG_COLOR, self.ROAD_COLOR = (240, 240, 240), (100, 100, 100)
        self.UI_BG_COLOR, self.UI_ICON_COLOR = (200, 200, 200), (50, 50, 50)
        self.UI_HIGHLIGHT_COLOR, self.WARNING_COLOR = (0, 200, 0), (255, 0, 0)
        self.DUMP_COLOR, self.COAL_COLOR = (80, 80, 90), (139, 69, 19)
        self.ERASER_COLOR = (255, 100, 100, 150)
        # --- ADDED: Connectivity check setting ---
        self.CONNECTIVITY_THRESHOLD = 15.0 # How close a site must be to a road to be "connected"

        # Camera controls
        self.zoom, self.offset = 1.0, np.array([0.0, 0.0])
        self.panning, self.pan_start_pos = False, np.array([0, 0])

        # Editor state
        self.tool = "draw" # "draw", "erase", "place_dump", "place_coal"
        self.is_drawing = False
        self.new_road_points = []
        self.DEFAULT_ROAD_WIDTH = 4.0
        self.ERASER_RADIUS = 15
        # --- ADDED: State for warnings ---
        self.unreachable_sites = set()
        self.warning_flash_timer = 0

        # UI element positions (ADDED new buttons)
        self.ui_buttons = {
            "draw": pygame.Rect(10, 10, 40, 40), "erase": pygame.Rect(60, 10, 40, 40),
            "place_dump": pygame.Rect(10, 60, 40, 40), "place_coal": pygame.Rect(60, 60, 40, 40)
        }
        self.center_map()

    # --- ADDED: Connectivity Check ---
    def check_site_connectivity(self):
        """Checks if all sites are connected to a road. Returns True if all are connected."""
        self.unreachable_sites.clear()
        for site_id, site_data in self.sites.items():
            is_reachable = False
            for edge in self.edges:
                for point in edge['shape']:
                    # Check distance from site to every point on every road
                    if np.linalg.norm(site_data['pos'] - point) < self.CONNECTIVITY_THRESHOLD:
                        is_reachable = True
                        break
                if is_reachable:
                    break
            if not is_reachable:
                self.unreachable_sites.add(site_id)

        return not self.unreachable_sites # Return True if the unreachable set is empty

    def save_map_to_json(self, filename="map_data.json"):
        # --- MODIFIED: Added connectivity check before saving ---
        print("\nChecking site connectivity before saving...")
        if not self.check_site_connectivity():
            print(f"❌ SAVE CANCELLED. The following sites are not connected to a road: {', '.join(self.unreachable_sites)}")
            return # Abort the save

        print(f"Saving current map state to {filename}...")
        serializable_junctions = {jid: {'pos': data['pos'].tolist()} for jid, data in self.junctions.items()}
        serializable_edges = [{'shape': [p.tolist() for p in edge['shape']], 'width': edge['width']} for edge in self.edges]
        # --- ADDED: Save sites data ---
        serializable_sites = {sid: {'type': data['type'], 'pos': data['pos'].tolist()} for sid, data in self.sites.items()}

        map_data = {'junctions': serializable_junctions, 'edges': serializable_edges, 'sites': serializable_sites}
        try:
            with open(filename, 'w') as f:
                json.dump(map_data, f, indent=2)
            print(f"✅ Success! Map saved.")
        except Exception as e:
            print(f"❌ Error! Could not save map. Reason: {e}")

    def center_map(self):
        # --- MODIFIED to include sites in centering ---
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

    def screen_to_world(self, screen_pos):
        p = (np.array(screen_pos) - self.offset) / self.zoom
        return np.array([p[0], -p[1]])

    def world_to_screen(self, world_pos):
        p = np.array([world_pos[0], -world_pos[1]])
        return p * self.zoom + self.offset

    # --- ADDED: Site placement logic ---
    def place_site(self, site_type, pos):
        if site_type == "dump_site":
            # Only one dump site allowed, overwrite if it exists
            self.sites["DUMP_0"] = {'type': 'dump_site', 'pos': pos}
        elif site_type == "coal_mine":
            i = 0
            while f"COAL_{i}" in self.sites:
                i += 1
            self.sites[f"COAL_{i}"] = {'type': 'coal_mine', 'pos': pos}

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.VIDEORESIZE: self.width, self.height = event.size

            # --- Tool Selection (MODIFIED for 4 tools) ---
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_ui = False
                for tool_name, rect in self.ui_buttons.items():
                    if rect.collidepoint(event.pos):
                        self.tool = tool_name
                        clicked_ui = True
                        break
                if clicked_ui: continue # Skip map interaction if a UI button was clicked

                # --- Map Interaction (MODIFIED for site placement) ---
                self.is_drawing = True # For erase/draw dragging
                if self.tool == "draw":
                    self.new_road_points = [self.screen_to_world(event.pos)]
                elif self.tool == "erase":
                    self.erase_at_pos(event.pos)
                elif self.tool == "place_dump":
                    self.place_site("dump_site", self.screen_to_world(event.pos))
                elif self.tool == "place_coal":
                    self.place_site("coal_mine", self.screen_to_world(event.pos))

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3: self.panning, self.pan_start_pos = True, np.array(event.pos)
                if event.button == 4: self.zoom *= 1.1
                if event.button == 5: self.zoom *= 0.9

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if self.tool == "draw" and len(self.new_road_points) > 1:
                        self.edges.append({'shape': self.new_road_points, 'width': self.DEFAULT_ROAD_WIDTH})
                    self.is_drawing = False
                    self.new_road_points = []
                if event.button == 3: self.panning = False

            if event.type == pygame.MOUSEMOTION:
                if self.panning:
                    self.offset += np.array(event.pos) - self.pan_start_pos
                    self.pan_start_pos = np.array(event.pos)
                elif self.is_drawing:
                    if self.tool == "draw": self.new_road_points.append(self.screen_to_world(event.pos))
                    elif self.tool == "erase": self.erase_at_pos(event.pos)

            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.save_map_to_json()
        return True

    def erase_at_pos(self, screen_pos):
        world_pos = self.screen_to_world(screen_pos)
        erase_radius_world = self.ERASER_RADIUS / self.zoom
        # Erase edges
        self.edges[:] = [edge for edge in self.edges if not any(np.linalg.norm(p - world_pos) < erase_radius_world for p in edge['shape'])]
        # --- ADDED: Erase sites ---
        self.sites = {sid: data for sid, data in self.sites.items() if np.linalg.norm(data['pos'] - world_pos) > erase_radius_world}

    def draw_ui(self):
        # --- MODIFIED to draw 4 buttons ---
        for tool_name, rect in self.ui_buttons.items():
            pygame.draw.circle(self.screen, self.UI_BG_COLOR, rect.center, 20)
            if self.tool == tool_name:
                pygame.draw.circle(self.screen, self.UI_HIGHLIGHT_COLOR, rect.center, 22, 3)
        # Icons
        p1 = self.ui_buttons["draw"].center + np.array([-8, 8]); p2 = self.ui_buttons["draw"].center + np.array([8, -8])
        pygame.draw.line(self.screen, self.UI_ICON_COLOR, p1, p2, 4)
        pygame.draw.polygon(self.screen, self.UI_ICON_COLOR, [p1, p1+np.array([4,-4]), p1+np.array([0,4])])
        eraser_icon_rect = pygame.Rect(0,0, 20, 15); eraser_icon_rect.center = self.ui_buttons["erase"].center
        pygame.draw.rect(self.screen, self.UI_ICON_COLOR, eraser_icon_rect, 0, 3)
        font = pygame.font.SysFont(None, 30)
        dump_text = font.render('D', True, self.UI_ICON_COLOR); self.screen.blit(dump_text, self.ui_buttons["place_dump"].center - np.array([7,10]))
        coal_text = font.render('C', True, self.UI_ICON_COLOR); self.screen.blit(coal_text, self.ui_buttons["place_coal"].center - np.array([7,10]))

    def draw(self):
        self.screen.fill(self.BG_COLOR)
        # Draw existing roads
        for edge in self.edges:
            if len(edge['shape']) > 1:
                scaled_width = int(edge['width'] * self.zoom); scaled_width = max(2, scaled_width)
                screen_points = [self.world_to_screen(p) for p in edge['shape']]
                pygame.draw.lines(self.screen, self.ROAD_COLOR, False, screen_points, scaled_width)
                for p in screen_points:
                    pygame.draw.circle(self.screen, self.ROAD_COLOR, p, scaled_width / 2)
        # Draw new road preview
        if self.tool == "draw" and len(self.new_road_points) > 1:
            scaled_width = int(self.DEFAULT_ROAD_WIDTH * self.zoom); scaled_width = max(2, scaled_width)
            screen_points = [self.world_to_screen(p) for p in self.new_road_points]
            pygame.draw.lines(self.screen, (0,150,0), False, screen_points, scaled_width)
        # --- ADDED: Draw sites and warnings ---
        site_radius = int(10 * self.zoom)
        for site_id, data in self.sites.items():
            color = self.DUMP_COLOR if data['type'] == 'dump_site' else self.COAL_COLOR
            pos = self.world_to_screen(data['pos'])
            pygame.draw.rect(self.screen, color, (pos[0]-site_radius, pos[1]-site_radius, site_radius*2, site_radius*2))
            # Flash a warning border if unconnected
            if site_id in self.unreachable_sites and (self.warning_flash_timer % 60 < 30):
                warning_rect = pygame.Rect(pos[0]-site_radius-2, pos[1]-site_radius-2, site_radius*2+4, site_radius*2+4)
                pygame.draw.rect(self.screen, self.WARNING_COLOR, warning_rect, 3)

        # Draw eraser cursor
        if self.tool == "erase":
            eraser_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.circle(eraser_surface, self.ERASER_COLOR, pygame.mouse.get_pos(), self.ERASER_RADIUS)
            self.screen.blit(eraser_surface, (0,0))

        self.draw_ui()
        self.warning_flash_timer += 1
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    MAP_FILE = 'map_data.json'
    # --- MODIFIED to load sites ---
    junctions, edges, sites = load_map_from_json(MAP_FILE)
    editor = MapEditor(junctions, edges, sites) # Pass sites to editor
    editor.run()