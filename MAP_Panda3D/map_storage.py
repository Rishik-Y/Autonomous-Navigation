import os
import shutil
from datetime import datetime

MAP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(MAP_DIR, os.pardir))
SIMULATION_DIR = os.path.join(REPO_ROOT, "Simulation")
SAVED_DIR = os.path.join(MAP_DIR, "Saved_Map")
HISTORY_DIR = os.path.join(SAVED_DIR, "History")


def ensure_dirs():
    os.makedirs(HISTORY_DIR, exist_ok=True)


def saved_path(filename: str) -> str:
    return os.path.join(SAVED_DIR, filename)


def legacy_path(filename: str) -> str:
    return os.path.join(MAP_DIR, filename)


def simulation_path(filename: str) -> str:
    target_dir = os.path.join(SIMULATION_DIR, "Map")
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)


def resolve_input_path(filename: str, fallback_paths=None) -> str:
    preferred = saved_path(filename)
    if os.path.exists(preferred):
        return preferred
    for path in fallback_paths or []:
        if path and os.path.exists(path):
            return path
    return preferred


def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_text_file(filename: str, content: str, copy_targets=None) -> str:
    ensure_dirs()
    dest = saved_path(filename)
    _ensure_parent(dest)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(content)
    for target in copy_targets or []:
        _ensure_parent(target)
        shutil.copy2(dest, target)
    return dest


def write_binary_file(filename: str, data: bytes, copy_targets=None) -> str:
    ensure_dirs()
    dest = saved_path(filename)
    _ensure_parent(dest)
    with open(dest, "wb") as f:
        f.write(data)
    for target in copy_targets or []:
        _ensure_parent(target)
        shutil.copy2(dest, target)
    return dest


def create_snapshot():
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_folder = os.path.join(HISTORY_DIR, timestamp)
    os.makedirs(snapshot_folder, exist_ok=True)

    files_to_copy = [
        "map_data.py",
        "mine_config.json",
        "waypoints.pkl",
        "map_cache.pkl",
        "map_data.json",
    ]

    for filename in files_to_copy:
        source = saved_path(filename)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(snapshot_folder, filename))

    return snapshot_folder
