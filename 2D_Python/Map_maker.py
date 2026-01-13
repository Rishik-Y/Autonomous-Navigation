import xml.etree.ElementTree as ET
import pygame
import sys
import numpy as np
import json

# --- 1. NEW: High-Resolution Path Interpolation ---
def interpolate_shape(points, step_distance=1.0):
    """
    Takes a list of points and adds new points between them, ensuring the
    distance between any two consecutive points is no more than step_distance.
    This creates high-resolution paths perfect for simulation.
    """
    high_res_points = []
    if not points:
        return high_res_points

    high_res_points.append(points[0]) # Always add the first point

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        distance = np.linalg.norm(p2 - p1)

        if distance > step_distance:
            # Calculate how many new points to add
            num_steps = int(distance / step_distance)
            for j in range(1, num_steps + 1):
                alpha = j / num_steps
                interpolated_point = p1 * (1 - alpha) + p2 * alpha
                high_res_points.append(interpolated_point)

        high_res_points.append(p2) # Always add the last point of the segment

    return high_res_points

# --- 2. PARSING FUNCTION (Now with Interpolation) ---
def parse_and_process_net(net_file):
    print(f"Parsing and processing network file: {net_file}...")
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
    except ET.ParseError as e: return None, None

    junctions = {}
    for junction in root.iter('junction'):
        x_str, y_str = junction.get('x'), junction.get('y')
        if x_str is not None and y_str is not None:
            junctions[junction.get('id')] = {'pos': np.array([float(x_str), float(y_str)])}

    edges = []
    max_road_width = 0
    for edge in root.iter('edge'):
        lane = edge.find('lane')
        if lane is not None:
            shape_str, width_str = lane.get('shape'), lane.get('width')
            if shape_str and width_str:
                original_points = [np.array([float(p.split(',')[0]), float(p.split(',')[1])]) for p in shape_str.split(' ')]

                # --- THIS IS THE KEY STEP ---
                # Create a high-resolution version of the path
                high_res_shape = interpolate_shape(original_points)

                width = float(width_str)
                edges.append({'shape': high_res_shape, 'width': width})
                if width > max_road_width: max_road_width = width

    for j in junctions.values():
        j['fill_radius'] = max_road_width * 0.7
    print(f"Successfully parsed and processed {len(junctions)} junctions and {len(edges)} edges.")
    return junctions, edges

# --- 3. THE PYGAME VIEWER (With High-Quality Drawing and Saving) ---
class MapViewer:
    def __init__(self, junctions, edges):
        pygame.init()
        self.junctions, self.edges = junctions, edges
        self.width, self.height = 1200, 900
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("High-Fidelity Map | Press 'S' to Save for Simulation")
        self.BG_COLOR, self.ROAD_COLOR = (240, 240, 240), (100, 100, 100)
        self.zoom, self.offset = 1.0, np.array([0.0, 0.0])
        self.panning, self.pan_start_pos = False, np.array([0, 0])
        self.center_map()

    def save_map_to_json(self, filename="map_data.json"):
        print(f"\nSaving high-resolution map data to {filename}...")
        serializable_junctions = {
            jid: {'pos': data['pos'].tolist()} for jid, data in self.junctions.items()
        }
        serializable_edges = [
            {'shape': [point.tolist() for point in edge['shape']], 'width': edge['width']}
            for edge in self.edges
        ]
        map_data = {'junctions': serializable_junctions, 'edges': serializable_edges}
        try:
            with open(filename, 'w') as f:
                json.dump(map_data, f, indent=2) # Using indent=2 for smaller file size
            print(f"✅ Success! High-fidelity map data saved to {filename}")
        except Exception as e:
            print(f"❌ Error! Could not save map data. Reason: {e}")

    def center_map(self):
        if not self.junctions: return
        all_node_coords = np.array([j['pos'] for j in self.junctions.values()])
        min_coords, max_coords = np.min(all_node_coords, axis=0), np.max(all_node_coords, axis=0)
        map_center = (min_coords + max_coords) / 2.0
        map_size = np.where(max_coords - min_coords == 0, 1, max_coords - min_coords)
        zoom_x, zoom_y = self.width / map_size[0] * 0.9, self.height / map_size[1] * 0.9
        self.zoom = min(zoom_x, zoom_y)
        self.offset = np.array([self.width / 2.0, self.height / 2.0]) - map_center * self.zoom

    def world_to_screen(self, world_pos):
        return np.array([world_pos[0], -world_pos[1]]) * self.zoom + self.offset

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.VIDEORESIZE: self.width, self.height = event.size
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: self.panning, self.pan_start_pos = True, np.array(event.pos)
                if event.button == 4: self.zoom *= 1.1
                if event.button == 5: self.zoom *= 0.9
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: self.panning = False
            if event.type == pygame.MOUSEMOTION and self.panning:
                self.offset += np.array(event.pos) - self.pan_start_pos
                self.pan_start_pos = np.array(event.pos)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.save_map_to_json()
        return True

    def draw(self):
        self.screen.fill(self.BG_COLOR)
        for j_data in self.junctions.values():
            fill_radius = j_data['fill_radius'] * self.zoom
            if fill_radius > 1:
                pygame.draw.circle(self.screen, self.ROAD_COLOR, self.world_to_screen(j_data['pos']), fill_radius)
        for edge in self.edges:
            shape, width = edge['shape'], edge['width']
            scaled_width = int(width * self.zoom)
            if scaled_width < 2: scaled_width = 2
            screen_points = [self.world_to_screen(p) for p in shape]
            if len(screen_points) > 1:
                pygame.draw.lines(self.screen, self.ROAD_COLOR, False, screen_points, scaled_width)
            for p in screen_points:
                pygame.draw.circle(self.screen, self.ROAD_COLOR, p, scaled_width / 2)
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    SUMO_NET_FILE = 'sumo.net.xml'
    try:
        junction_data, edge_data = parse_and_process_net(SUMO_NET_FILE)
        if junction_data and edge_data:
            viewer = MapViewer(junction_data, edge_data)
            viewer.run()
        else:
            print("Could not start viewer due to parsing issues.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
