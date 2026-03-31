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
    return os.path.join(SIMULATION_DIR, filename)


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
    if os.path.exists(saved_file):
        ensure_dirs()
        history_file = _history_path(saved_file)
        shutil.copy2(saved_file, history_file)


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
