import json
import numpy as np
import os

# --- MAP DATA LOADER ---
# This module replaces the direct import of map_data.py
# It loads the map structure from map_data.json and converts coordinates back to numpy arrays.

NODES = {}
EDGES = []
LOAD_ZONES = []
DUMP_ZONES = []
FUEL_ZONES = []
VISUAL_ROAD_CHAINS = []

def load_map_data(json_file='map_data.json'):
    global NODES, EDGES, LOAD_ZONES, DUMP_ZONES, FUEL_ZONES, VISUAL_ROAD_CHAINS
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found. Map data could not be loaded.")
        return

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # 1. Load NODES and convert to numpy arrays
        raw_nodes = data.get('NODES', {})
        for name, pos in raw_nodes.items():
            NODES[name] = np.array(pos)

        # 2. Load EDGES
        EDGES = data.get('EDGES', [])
        # Ensure tuples if needed, but lists are usually fine for iteration
        # If code expects tuples for keys or hashable items, we might need conversion.
        # map_data.py had EDGES as a list of tuples/lists. JSON gives lists.
        # Let's convert to tuples to be safe and match original python behavior exactly.
        EDGES = [tuple(e) for e in EDGES]

        # 3. Load ZONES
        LOAD_ZONES = data.get('LOAD_ZONES', [])
        DUMP_ZONES = data.get('DUMP_ZONES', [])
        FUEL_ZONES = data.get('FUEL_ZONES', [])

        # 4. Load VISUAL_ROAD_CHAINS
        VISUAL_ROAD_CHAINS = data.get('VISUAL_ROAD_CHAINS', [])
        
        # print(f"MapLoader: Loaded {len(NODES)} nodes from {json_file}")

    except Exception as e:
        print(f"Error loading map data from JSON: {e}")

# Load data on module import
load_map_data()
