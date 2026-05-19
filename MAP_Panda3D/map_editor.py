import io
import importlib
from ast import literal_eval

import numpy as np

import generate_map_cache
import map_data
import map_storage
import session_tracker
from panda_common import POINTS_PER_SEGMENT, ROAD_WIDTH_M, generate_curvy_path_from_nodes

CLICK_THRESHOLD_M = 6.0


class MapEditorMode:
    label = "Map Editor"

    def __init__(self, app):
        self.app = app
        self.mode = "add_purple"
        self.status_text = "Mode: ADD PURPLE"
        self.selection_start_node = None
        self.is_drawing_manual = False
        self.manual_path = []
        self.is_dirty = False
        self.cache_needs_regen = False
        self.show_node_names = False

        self.NODES = {}
        self.EDGES = []
        self.LOAD_ZONES = []
        self.DUMP_ZONES = []
        self.FUEL_ZONES = []
        self.VISUAL_ROAD_CHAINS = []
        self.PRE_CALCULATED_SPLINES = []
        self._saved_files = []

    def activate(self):
        self.load_map_data()
        self.rebuild_splines()
        self.redraw()

    def deactivate(self):
        if self.cache_needs_regen:
            self.regenerate_map_cache()

    def load_map_data(self):
        map_file = map_storage.resolve_input_path("map_data.py", [map_storage.legacy_path("map_data.py")])
        sandbox = {"np": np}
        with open(map_file, "r", encoding="utf-8") as f:
            exec(f.read(), sandbox)
        self.NODES = sandbox.get("NODES", {})
        self.EDGES = [tuple(e) for e in sandbox.get("EDGES", [])]
        self.LOAD_ZONES = list(sandbox.get("LOAD_ZONES", []))
        self.DUMP_ZONES = list(sandbox.get("DUMP_ZONES", []))
        self.FUEL_ZONES = list(sandbox.get("FUEL_ZONES", []))
        self.VISUAL_ROAD_CHAINS = [list(c) for c in sandbox.get("VISUAL_ROAD_CHAINS", [])]

    def save_map_data(self):
        output = io.StringIO()
        output.write("import numpy as np\n\n")
        output.write("# --- MAP DATA ---\n\n")
        output.write("NODES = {\n")
        for name, pos in sorted(self.NODES.items()):
            output.write(f'    "{name}": np.array([{pos[0]:.1f}, {pos[1]:.1f}]),\n')
        output.write("}\n\n")

        output.write("EDGES = [\n")
        for edge in sorted(set(tuple(sorted(e)) for e in self.EDGES)):
            output.write(f"    {edge},\n")
        output.write("]\n\n")

        for key, values in (
            ("LOAD_ZONES", self.LOAD_ZONES),
            ("DUMP_ZONES", self.DUMP_ZONES),
            ("FUEL_ZONES", self.FUEL_ZONES),
        ):
            output.write(f"{key} = [\n")
            for zone in sorted(set(values)):
                output.write(f'    "{zone}",\n')
            output.write("]\n\n")

        output.write("VISUAL_ROAD_CHAINS = [\n")
        for chain in self.VISUAL_ROAD_CHAINS:
            output.write(f"    {chain},\n")
        output.write("]\n")

        map_storage.write_text_file(
            "map_data.py",
            output.getvalue(),
            copy_targets=[map_storage.legacy_path("map_data.py")],
        )
        self._saved_files.append("map_data.py")
        self.is_dirty = False
        self.cache_needs_regen = True
        self.status_text = "SAVED to Saved_Map/map_data.py"
        session_tracker.mark_save_occurred()

    def regenerate_map_cache(self):
        importlib.reload(map_data)
        generate_map_cache.main()

    def rebuild_splines(self):
        self.PRE_CALCULATED_SPLINES = []
        for chain in self.VISUAL_ROAD_CHAINS:
            node_coords = [self.NODES[node_name] for node_name in chain if node_name in self.NODES]
            if len(node_coords) < 2:
                continue
            self.PRE_CALCULATED_SPLINES.append(generate_curvy_path_from_nodes(node_coords))

    def redraw(self):
        self.app.renderer.draw_grid()
        self.app.renderer.draw_roads(self.PRE_CALCULATED_SPLINES)
        self.app.renderer.draw_nodes(self.NODES, self.LOAD_ZONES, self.DUMP_ZONES, self.FUEL_ZONES, self.show_node_names)

    def _distance(self, p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def get_node_at_pos(self, pos_m):
        if pos_m is None:
            return None
        for name, node_pos in self.NODES.items():
            if self._distance(pos_m, node_pos) <= CLICK_THRESHOLD_M:
                return name
        return None

    def _next_auto_name(self, prefix):
        num = 1
        while f"{prefix}_{num}" in self.NODES:
            num += 1
        return f"{prefix}_{num}"

    def _insert_node_type(self, name, mode):
        if mode == "add_green":
            self.LOAD_ZONES.append(name)
        elif mode == "add_red":
            self.DUMP_ZONES.append(name)
        elif mode == "add_orange":
            self.FUEL_ZONES.append(name)

    def on_key(self, key):
        if key == "s":
            self.save_map_data()
            self.redraw()
            return
        if key == "n":
            self.show_node_names = not self.show_node_names
            self.redraw()
            return
        mode_map = {
            "g": ("add_green", "Mode: ADD GREEN (Load Zone)"),
            "r": ("add_red", "Mode: ADD RED (Dump Zone)"),
            "o": ("add_orange", "Mode: ADD ORANGE (Fuel Zone)"),
            "p": ("add_purple", "Mode: ADD PURPLE (Intermediate)"),
            "w": ("delete", "Mode: DELETE"),
            "c": ("connect_start", "Mode: CONNECT (Click first node)"),
            "d": ("disconnect_start", "Mode: DISCONNECT (Click first node)"),
            "m": ("manual", "Mode: MANUAL DRAW (Hold left mouse)")
        }
        if key in mode_map:
            self.mode, self.status_text = mode_map[key]
            self.selection_start_node = None
            return
        if key == "f":
            fixed = self.fix_intersections()
            self.status_text = f"Fixed {fixed} intersections" if fixed else "No intersections needed fixing"
            if fixed:
                self.is_dirty = True
                self.rebuild_splines()
                self.redraw()

    def on_mouse1(self, down=True):
        pick = self.app.picker.pick_ground().world_xy
        if self.mode == "manual":
            if down:
                self.is_drawing_manual = True
                self.manual_path = [] if pick is None else [np.array(pick)]
            else:
                self.is_drawing_manual = False
                self.finish_manual_path()
            return

        if not down:
            return
        clicked_node = self.get_node_at_pos(pick)

        if self.mode.startswith("add"):
            if not clicked_node and pick is not None:
                prefix = self.mode.split("_")[1]
                if prefix == "orange":
                    prefix = "fuel"
                name = self._next_auto_name(f"{prefix}_auto")
                self.NODES[name] = np.array(pick)
                self._insert_node_type(name, self.mode)
                self.status_text = f"Added node: {name}"
                self.is_dirty = True
                self.rebuild_splines()
                self.redraw()

        elif self.mode == "delete" and clicked_node:
            self.delete_node(clicked_node)
            self.status_text = f"Deleted node: {clicked_node}"
            self.is_dirty = True
            self.rebuild_splines()
            self.redraw()

        elif self.mode == "connect_start" and clicked_node:
            self.selection_start_node = clicked_node
            self.mode = "connect_end"
            self.status_text = f"Connecting from '{clicked_node}'. Click second node."

        elif self.mode == "connect_end":
            if clicked_node and clicked_node != self.selection_start_node:
                new_edge = tuple(sorted((self.selection_start_node, clicked_node)))
                if new_edge not in self.EDGES:
                    self.EDGES.append(new_edge)
                    self.VISUAL_ROAD_CHAINS.append(list(new_edge))
                    self.is_dirty = True
                    self.rebuild_splines()
                    self.redraw()
                    self.status_text = f"Connected {self.selection_start_node} to {clicked_node}"
            self.selection_start_node = None
            self.mode = "connect_start"

        elif self.mode == "disconnect_start" and clicked_node:
            self.selection_start_node = clicked_node
            self.mode = "disconnect_end"
            self.status_text = f"Disconnecting from '{clicked_node}'. Click second node."

        elif self.mode == "disconnect_end":
            if clicked_node and clicked_node != self.selection_start_node:
                edge = tuple(sorted((self.selection_start_node, clicked_node)))
                if edge in self.EDGES:
                    self.EDGES.remove(edge)
                    self.split_chains_for_removed_edge(*edge)
                    self.is_dirty = True
                    self.rebuild_splines()
                    self.redraw()
                    self.status_text = f"Disconnected {self.selection_start_node} and {clicked_node}"
            self.selection_start_node = None
            self.mode = "disconnect_start"

    def on_mouse_move(self):
        if self.is_drawing_manual:
            pick = self.app.picker.pick_ground().world_xy
            if pick is not None:
                p = np.array(pick)
                if not self.manual_path or self._distance(self.manual_path[-1], p) > 2.0:
                    self.manual_path.append(p)

    def finish_manual_path(self):
        if len(self.manual_path) <= 1:
            self.manual_path = []
            return

        min_node_dist = 20.0
        chain = []
        start_snap = self.get_node_at_pos(self.manual_path[0])
        if start_snap:
            chain.append(start_snap)
            last_pos = self.NODES[start_snap]
        else:
            name = self._next_auto_name("purple_auto")
            self.NODES[name] = np.array(self.manual_path[0])
            chain.append(name)
            last_pos = self.NODES[name]

        for idx, p in enumerate(self.manual_path[1:], start=1):
            if self._distance(p, last_pos) <= min_node_dist:
                continue
            is_last = idx == len(self.manual_path) - 1
            if is_last:
                end_snap = self.get_node_at_pos(p)
                if end_snap and end_snap != chain[-1]:
                    chain.append(end_snap)
                    last_pos = self.NODES[end_snap]
                    continue
            name = self._next_auto_name("purple_auto")
            self.NODES[name] = np.array(p)
            chain.append(name)
            last_pos = self.NODES[name]

        deduped = [chain[0]]
        for n in chain[1:]:
            if n != deduped[-1]:
                deduped.append(n)
        if len(deduped) > 1:
            self.VISUAL_ROAD_CHAINS.append(deduped)
            for i in range(len(deduped) - 1):
                edge = tuple(sorted((deduped[i], deduped[i + 1])))
                if edge not in self.EDGES:
                    self.EDGES.append(edge)
            self.is_dirty = True
            self.rebuild_splines()
            self.redraw()
            self.status_text = f"Created manual road with {len(deduped)} nodes"
        self.manual_path = []

    def delete_node(self, clicked_node):
        del self.NODES[clicked_node]
        if clicked_node in self.LOAD_ZONES:
            self.LOAD_ZONES.remove(clicked_node)
        if clicked_node in self.DUMP_ZONES:
            self.DUMP_ZONES.remove(clicked_node)
        if clicked_node in self.FUEL_ZONES:
            self.FUEL_ZONES.remove(clicked_node)
        self.EDGES = [e for e in self.EDGES if clicked_node not in e]

        new_chains = []
        for chain in self.VISUAL_ROAD_CHAINS:
            if clicked_node not in chain:
                new_chains.append(chain)
                continue
            idx = chain.index(clicked_node)
            p1 = chain[:idx]
            p2 = chain[idx + 1 :]
            if len(p1) > 1:
                new_chains.append(p1)
            if len(p2) > 1:
                new_chains.append(p2)
        self.VISUAL_ROAD_CHAINS = new_chains

    def split_chains_for_removed_edge(self, n1, n2):
        new_chains = []
        for chain in self.VISUAL_ROAD_CHAINS:
            split = False
            for i in range(len(chain) - 1):
                a, b = chain[i], chain[i + 1]
                if set((a, b)) == set((n1, n2)):
                    left = chain[: i + 1]
                    right = chain[i + 1 :]
                    if len(left) > 1:
                        new_chains.append(left)
                    if len(right) > 1:
                        new_chains.append(right)
                    split = True
                    break
            if not split:
                new_chains.append(chain)
        self.VISUAL_ROAD_CHAINS = new_chains

    def find_line_segment_intersection(self, p1, p2, p3, p4):
        v1 = p2 - p1
        v2 = p4 - p3
        cross = np.cross(v1, v2)
        if abs(cross) < 1e-9:
            return None
        p1_p3 = p3 - p1
        t = np.cross(p1_p3, v2) / cross
        u = np.cross(p1_p3, v1) / cross
        if 0.001 < t < 0.999 and 0.001 < u < 0.999:
            return p1 + t * v1
        return None

    def fix_intersections(self):
        all_segments = []
        for i, chain in enumerate(self.VISUAL_ROAD_CHAINS):
            for k in range(len(chain) - 1):
                n1, n2 = chain[k], chain[k + 1]
                if n1 in self.NODES and n2 in self.NODES:
                    all_segments.append({"p1": self.NODES[n1], "p2": self.NODES[n2], "n1": n1, "n2": n2, "chain_idx": i, "chain_pos": k})

        found = []
        for i in range(len(all_segments)):
            for j in range(i + 1, len(all_segments)):
                s1 = all_segments[i]
                s2 = all_segments[j]
                if s1["n1"] in (s2["n1"], s2["n2"]) or s1["n2"] in (s2["n1"], s2["n2"]):
                    continue
                p = self.find_line_segment_intersection(s1["p1"], s1["p2"], s2["p1"], s2["p2"])
                if p is None:
                    continue
                if any(self._distance(p, npos) < ROAD_WIDTH_M / 2.0 for npos in self.NODES.values()):
                    continue
                if any(self._distance(p, ep[0]) < ROAD_WIDTH_M / 2.0 for ep in found):
                    continue
                found.append((p, s1, s2))

        if not found:
            return 0

        modifications = {}
        new_nodes = {}
        edges_add = set()
        edges_remove = set()
        for p, s1, s2 in found:
            idx = len(new_nodes) + 1
            while f"purple_auto_fix_{idx}" in self.NODES or f"purple_auto_fix_{idx}" in new_nodes:
                idx += 1
            name = f"purple_auto_fix_{idx}"
            new_nodes[name] = p
            for s in (s1, s2):
                c_idx, c_pos = s["chain_idx"], s["chain_pos"]
                modifications.setdefault(c_idx, []).append((c_pos + 1, name))
                edges_remove.add(tuple(sorted((s["n1"], s["n2"]))))
                edges_add.add(tuple(sorted((s["n1"], name))))
                edges_add.add(tuple(sorted((name, s["n2"]))))

        self.NODES.update(new_nodes)
        current = set(tuple(sorted(e)) for e in self.EDGES)
        current.difference_update(edges_remove)
        current.update(edges_add)
        self.EDGES = sorted(current)

        for chain_idx, mods in sorted(modifications.items()):
            mods.sort(key=lambda x: x[0], reverse=True)
            for pos, name in mods:
                self.VISUAL_ROAD_CHAINS[chain_idx].insert(pos, name)

        return len(found)

    def tick(self):
        self.app.renderer.update_labels()

    @property
    def controls_text(self):
        return "[G] Load [R] Dump [O] Fuel [P] Purple [W] Delete [C] Connect [D] Disconnect [M] Manual [F] Fix [S] Save [N] Names"


def run_editor():
    from direct.showbase.ShowBase import ShowBase
    from panda_common import CameraController, Picker, SceneRenderer

    class _EditorApp(ShowBase):
        def __init__(self):
            super().__init__()
            self.disableMouse()
            self.camera_controller = CameraController(self)
            self.picker = Picker(self)
            self.renderer = SceneRenderer(self)
            self.mode = MapEditorMode(self)
            self.mode.activate()
            self.accept("mouse1", self.mode.on_mouse1, [True])
            self.accept("mouse1-up", self.mode.on_mouse1, [False])
            for k in "abcdefghijklmnopqrstuvwxyz":
                self.accept(k, self.mode.on_key, [k])
            self.accept("s", self.mode.on_key, ["s"])
            self.taskMgr.add(self._tick, "map_editor_tick")

        def _tick(self, task):
            self.mode.on_mouse_move()
            self.mode.tick()
            return task.cont

    app = _EditorApp()
    app.run()


if __name__ == "__main__":
    run_editor()
