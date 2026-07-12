import json
import os
import pickle
import numpy as np

NODES = {}
EDGES = []
LOAD_ZONES = []
DUMP_ZONES = []
FUEL_ZONES = []
VISUAL_ROAD_CHAINS = []

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))


def _candidate_json_paths():
    return [
        os.path.join(_REPO_ROOT, "MAP_Panda3D", "Saved_Map", "map_data.json"),
        os.path.join(_REPO_ROOT, "Simulation", "Map", "map_data.json"),
        os.path.join(_REPO_ROOT, "MAP", "Saved_Map", "map_data.json"),
    ]


def resolve_saved_map_path(filename):
    candidates = [
        os.path.join(_REPO_ROOT, "MAP_Panda3D", "Saved_Map", filename),
        os.path.join(_REPO_ROOT, "Simulation", "Map", filename),
        os.path.join(_REPO_ROOT, "MAP", "Saved_Map", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_map_data(json_file=None):
    global NODES, EDGES, LOAD_ZONES, DUMP_ZONES, FUEL_ZONES, VISUAL_ROAD_CHAINS

    selected = json_file
    if selected is None:
        for candidate in _candidate_json_paths():
            if os.path.exists(candidate):
                selected = candidate
                break

    if not selected or not os.path.exists(selected):
        raise FileNotFoundError("No map_data.json found in MAP_Panda3D/Saved_Map or fallbacks")

    with open(selected, "r", encoding="utf-8") as f:
        data = json.load(f)

    NODES = {name: np.array(pos, dtype=float) for name, pos in data.get("NODES", {}).items()}
    EDGES = [tuple(e) for e in data.get("EDGES", [])]
    LOAD_ZONES = list(data.get("LOAD_ZONES", []))
    DUMP_ZONES = list(data.get("DUMP_ZONES", []))
    FUEL_ZONES = list(data.get("FUEL_ZONES", []))
    VISUAL_ROAD_CHAINS = [tuple(chain) for chain in data.get("VISUAL_ROAD_CHAINS", [])]


load_map_data()


def load_pickle(filename):
    path = resolve_saved_map_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found in MAP_Panda3D/Saved_Map fallbacks")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(filename):
    path = resolve_saved_map_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{filename} not found in MAP_Panda3D/Saved_Map fallbacks")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
