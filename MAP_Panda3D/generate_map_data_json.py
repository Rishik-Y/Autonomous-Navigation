import json

import numpy as np

import map_data
import map_storage


def generate_json():
    data = {}
    nodes_dict = {}
    for name, pos in getattr(map_data, "NODES", {}).items():
        nodes_dict[name] = pos.tolist() if isinstance(pos, np.ndarray) else list(pos)
    data["NODES"] = nodes_dict

    data["EDGES"] = getattr(map_data, "EDGES", [])
    data["LOAD_ZONES"] = getattr(map_data, "LOAD_ZONES", [])
    data["DUMP_ZONES"] = getattr(map_data, "DUMP_ZONES", [])
    data["FUEL_ZONES"] = getattr(map_data, "FUEL_ZONES", [])
    data["VISUAL_ROAD_CHAINS"] = getattr(map_data, "VISUAL_ROAD_CHAINS", [])

    output_file = "map_data.json"
    map_storage.write_text_file(
        output_file,
        json.dumps(data, indent=4),
        copy_targets=[map_storage.simulation_path(output_file)],
    )


if __name__ == "__main__":
    generate_json()
