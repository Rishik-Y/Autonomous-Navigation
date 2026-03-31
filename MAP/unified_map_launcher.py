import Map_Editor
import Coal_Mine_Editor
import Waypoint_Editor
import map_viewer
import waypoint_viewer


MODES = [
    ("Map Editor", Map_Editor.run_editor),
    ("Coal Mine Editor", Coal_Mine_Editor.run_editor),
    ("Waypoint Editor", Waypoint_Editor.run_waypoint_editor),
    ("Map Viewer", map_viewer.run_viewer),
    ("Waypoint Viewer", waypoint_viewer.run_viewer),
]


def run_launcher():
    mode_index = 0
    while True:
        mode_label, runner = MODES[mode_index]
        try:
            result = runner(mode_label=mode_label, allow_tab_switch=True)
        except Exception as exc:
            print(f"Launcher exiting due to error in {mode_label}: {exc}")
            break
        if result == "next":
            mode_index = (mode_index + 1) % len(MODES)
            continue
        break


if __name__ == "__main__":
    run_launcher()
