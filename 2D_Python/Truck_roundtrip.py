import pygame
import sys
import numpy as np
import json
from collections import deque

def load_cleaned_map(filename="map_data_cleaned.json"):
    """Loads the clean, processed map data."""
    print(f"Loading cleaned map data from {filename}...")
    try:
        with open(filename, 'r') as f: map_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: The file '{filename}' was not found."); return None, None, None

    junctions = {jid: {'pos': np.array(data['pos'])} for jid, data in map_data.get('junctions', {}).items()}
    edges = map_data.get('edges', [])
    sites = {sid: {'type': data['type'], 'pos': np.array(data['pos'])} for sid, data in map_data.get('sites', {}).items()}
    return junctions, edges, sites

def analyze_connectivity(junctions, edges):
    """Analyzes the road network and assigns a color to each disconnected component."""
    if not junctions: return {}

    graph = {jid: [] for jid in junctions}
    for edge in edges:
        graph[edge['from']].append(edge['to'])
        graph[edge['to']].append(edge['from'])

    visited = set()
    component_colors = {}
    component_id = 0
    colors = [(34, 139, 34), (0, 0, 205), (255, 140, 0), (220, 20, 60), (148, 0, 211)] # Green, Blue, Orange, Red, Violet

    print("\n--- Network Connectivity Analysis ---")
    for jid in junctions:
        if jid not in visited:
            component_id += 1
            color = colors[(component_id - 1) % len(colors)]
            print(f"Found network component #{component_id} (Color: {color})")

            queue = deque([jid])
            visited.add(jid)
            component_colors[jid] = color

            while queue:
                node = queue.popleft()
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component_colors[neighbor] = color
                        queue.append(neighbor)

    if component_id == 1:
        print("✅ Good News! Your entire road network is fully connected.")
    else:
        print(f"⚠️ WARNING! Your map is split into {component_id} disconnected road networks.")
        print("Use the Map Editor to draw roads connecting the different colored sections.")

    return component_colors

class MapDebugger:
    def __init__(self, junctions, edges, sites, component_colors):
        pygame.init()
        self.junctions, self.edges, self.sites = junctions, edges, sites
        self.component_colors = component_colors
        self.width, self.height = 1200, 900
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Map Connectivity Debugger")
        self.BG_COLOR, self.ROAD_COLOR = (240, 240, 240), (100, 100, 100)
        self.DUMP_COLOR, self.COAL_COLOR = (80, 80, 90), (139, 69, 19)
        self.zoom, self.offset = 1.0, np.array([0.0, 0.0])
        self.center_map()

    def center_map(self):
        all_points = np.vstack([j['pos'] for j in self.junctions.values()])
        min_coords, max_coords = np.min(all_points, axis=0), np.max(all_points, axis=0)
        map_center = (min_coords + max_coords) / 2.0; map_size = np.where(max_coords - min_coords == 0, 1, max_coords - min_coords)
        zoom_x = self.width / map_size[0] * 0.9; zoom_y = self.height / map_size[1] * 0.9
        self.zoom = min(zoom_x, zoom_y)
        self.offset = np.array([self.width / 2.0, self.height / 2.0]) - map_center * self.zoom

    def world_to_screen(self, world_pos):
        return (np.array([world_pos[0], -world_pos[1]]) * self.zoom + self.offset).astype(int)

    def draw(self):
        self.screen.fill(self.BG_COLOR)
        for edge in self.edges:
            if len(edge['shape']) > 1:
                screen_points = [self.world_to_screen(np.array(p)) for p in edge['shape']]
                pygame.draw.lines(self.screen, self.ROAD_COLOR, False, screen_points, 3)

        for jid, jdata in self.junctions.items():
            color = self.component_colors.get(jid, (0,0,0)) # Black if something is wrong
            pygame.draw.circle(self.screen, color, self.world_to_screen(jdata['pos']), 8)

        site_radius = int(10 * self.zoom)
        for data in self.sites.values():
            color = self.DUMP_COLOR if data['type'] == 'dump_site' else self.COAL_COLOR
            pos = self.world_to_screen(data['pos'])
            pygame.draw.rect(self.screen, color, (pos[0]-site_radius, pos[1]-site_radius, site_radius*2, site_radius*2))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            self.draw()
        pygame.quit()

if __name__ == "__main__":
    junctions, edges, sites = load_cleaned_map()
    if junctions:
        component_colors = analyze_connectivity(junctions, edges)
        debugger = MapDebugger(junctions, edges, sites, component_colors)
        debugger.run()