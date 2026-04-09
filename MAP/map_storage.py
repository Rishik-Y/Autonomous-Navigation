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


def _history_path(filename: str) -> str:
    base, ext = os.path.splitext(os.path.basename(filename))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(HISTORY_DIR, f"{base}_{timestamp}{ext}")


def _backup_existing(saved_file: str):
    """
    DISABLED: This function previously backed up individual files.
    Now only used internally. Snapshot system handles all backups.
    """
    pass  # No longer creates individual file backups


def write_text_file(filename: str, content: str, copy_targets=None) -> str:
    ensure_dirs()
    dest = saved_path(filename)
    _backup_existing(dest)
    _ensure_parent(dest)
    with open(dest, "w") as f:
        f.write(content)
    for target in copy_targets or []:
        _ensure_parent(target)
        shutil.copy2(dest, target)
    return dest


def write_binary_file(filename: str, data: bytes, copy_targets=None) -> str:
    ensure_dirs()
    dest = saved_path(filename)
    _backup_existing(dest)
    _ensure_parent(dest)
    with open(dest, "wb") as f:
        f.write(data)
    for target in copy_targets or []:
        _ensure_parent(target)
        shutil.copy2(dest, target)
    return dest


def create_snapshot():
    """
    Create a timestamp folder in History/ and copy all map files.
    Stops launcher if any file copy fails.
    Returns: path to created snapshot folder
    """
    # Generate timestamp folder name: 2026-04-07_16-10-30
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_folder = os.path.join(HISTORY_DIR, timestamp)
    
    # Create folder
    os.makedirs(snapshot_folder, exist_ok=True)
    
    # Files to copy (all 4-5 files regardless of which was modified)
    files_to_copy = [
        'map_data.py',
        'mine_config.json',
        'waypoints.pkl',
        'map_cache.pkl',
        'map_data.json'  # optional, will skip if not exists
    ]
    
    # Copy each file if it exists
    for filename in files_to_copy:
        source = saved_path(filename)
        if os.path.exists(source):
            dest = os.path.join(snapshot_folder, filename)
            try:
                shutil.copy2(source, dest)
            except Exception as e:
                print(f"ERROR: Failed to copy {filename} to snapshot: {e}")
                print("Stopping launcher due to snapshot creation failure.")
                input("Press Enter to exit...")
                os._exit(1)  # Force stop launcher
    
    print(f"Snapshot created: {snapshot_folder}")
    return snapshot_folder
