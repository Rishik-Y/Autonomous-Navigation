import json
import numpy as np
import map_data
import os
import map_storage

def generate_json():
    print("Generating map_data.json from map_data.py...")
    
    data = {}
    
    # 1. NODES: Convert numpy arrays to lists [x, y]
    nodes_dict = {}
    if hasattr(map_data, 'NODES'):
        for name, pos in map_data.NODES.items():
            if isinstance(pos, np.ndarray):
                nodes_dict[name] = pos.tolist()
            else:
                nodes_dict[name] = list(pos)
    data['NODES'] = nodes_dict

    # 2. EDGES: List of lists/tuples
    if hasattr(map_data, 'EDGES'):
        data['EDGES'] = map_data.EDGES
    else:
        data['EDGES'] = []

    # 3. ZONES
    for zone_type in ['LOAD_ZONES', 'DUMP_ZONES', 'FUEL_ZONES']:
        if hasattr(map_data, zone_type):
            data[zone_type] = getattr(map_data, zone_type)
        else:
            data[zone_type] = []

    # 4. VISUAL_ROAD_CHAINS
    if hasattr(map_data, 'VISUAL_ROAD_CHAINS'):
        data['VISUAL_ROAD_CHAINS'] = map_data.VISUAL_ROAD_CHAINS
    else:
        data['VISUAL_ROAD_CHAINS'] = []

    output_file = 'map_data.json'
    try:
        content = json.dumps(data, indent=4)
        map_storage.write_text_file(
            output_file,
            content,
            copy_targets=[map_storage.simulation_path(output_file)]
        )
        print(f"Successfully generated Saved_Map/{output_file} with {len(nodes_dict)} nodes.")
    except Exception as e:
        print(f"Error generating JSON: {e}")

if __name__ == "__main__":
    generate_json()
