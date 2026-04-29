import Map_Editor
import Coal_Mine_Editor
import Waypoint_Editor
import map_viewer
import waypoint_viewer
import session_tracker
import map_storage


MODES = [
    ("Map Editor", Map_Editor.run_editor),
    ("Coal Mine Editor", Coal_Mine_Editor.run_editor),
    ("Waypoint Editor", Waypoint_Editor.run_waypoint_editor),
    ("Map Viewer", map_viewer.run_viewer),
    ("Waypoint Viewer", waypoint_viewer.run_viewer),
]


def run_launcher():
    session_tracker.reset_save_tracker()
    mode_index = 0
    files_saved = []
    
    while True:
        mode_label, runner = MODES[mode_index]
        try:
            result = runner(
                mode_label=mode_label,
                allow_tab_switch=True,
                mode_index=mode_index + 1,
                total_modes=len(MODES)
            )
        except Exception as exc:
            print(f"Launcher exiting due to error in {mode_label}: {exc}")
            raise
        
        # Track which files were saved in each editor
        if mode_label == "Map Editor":
            files_saved.extend(getattr(Map_Editor, '_saved_files', []))
        elif mode_label == "Coal Mine Editor":
            files_saved.extend(getattr(Coal_Mine_Editor, '_saved_files', []))
        elif mode_label == "Waypoint Editor":
            files_saved.extend(getattr(Waypoint_Editor, '_saved_files', []))
        
        if result == "next":
            mode_index = (mode_index + 1) % len(MODES)
            continue
        if result == "prev":
            mode_index = (mode_index - 1) % len(MODES)
            continue
        break
    
    # Create snapshot if user saved actual changes
    if files_saved:
        snapshot_path = map_storage.create_snapshot()
        unique_files = list(set(files_saved))
        print(f"✓ Snapshot saved to: {snapshot_path}")
        print(f"  Files saved: {', '.join(unique_files)}")
    else:
        print("No changes were saved. No snapshot created.")
    
    session_tracker.reset_save_tracker()


if __name__ == "__main__":
    run_launcher()
