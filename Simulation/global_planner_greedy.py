import numpy as np

import map_loader as map_data

from config import LOAD_UNLOAD_TIME_S, SPEED_MS_EMPTY, SPEED_MS_LOADED
from graph_adapter import GraphAdapter
from planner_interface import GlobalPlannerInterface

ROAD_CURVATURE_FACTOR = 1.5
DEFAULT_TRAVEL_TIME_S = 60.0


class Planner(GlobalPlannerInterface):
    def __init__(self):
        self.adapter = GraphAdapter()
        self.load_zones = map_data.LOAD_ZONES
        self.dump_zones = map_data.DUMP_ZONES
        self.static_dist_matrix = self.adapter.get_distance_matrix(
            self.load_zones + self.dump_zones,
            self.load_zones + self.dump_zones,
        )
        print("GlobalGreedyPlanner: initialized with static distance matrix.")

    def get_travel_time(self, start, end):
        if start == end:
            return 0.0

        dist = self.static_dist_matrix.get((start, end))
        if dist is None:
            p1 = map_data.NODES.get(start)
            p2 = map_data.NODES.get(end)
            if p1 is not None and p2 is not None:
                dist = np.linalg.norm(p1 - p2) * ROAD_CURVATURE_FACTOR
            else:
                return DEFAULT_TRAVEL_TIME_S

        speed = SPEED_MS_LOADED if end in map_data.DUMP_ZONES else SPEED_MS_EMPTY
        return max(1.0, dist / speed)

    def optimize_assignments(self, trucks, site_states):
        active_mines = [
            name
            for name, state in site_states.items()
            if name in map_data.LOAD_ZONES and state.get("coal_remaining", 0) > 0
        ]
        if not active_mines:
            return {}

        free_trucks = [
            t
            for t in trucks
            if t.op_state in ("IDLE", "UNLOADING", "RETURNING_TO_START")
            or (t.op_state == "GOING_TO_ENDPOINT" and not t.target_node_name)
        ]
        if not free_trucks:
            return {}

        assignments = {}
        for truck in free_trucks:
            start = truck.current_node_name if truck.current_node_name in map_data.NODES else "main_hub"
            best_mine = None
            best_time = float("inf")
            for mine in active_mines:
                travel_time = self.get_travel_time(start, mine)
                wait_time = site_states.get(mine, {}).get("en_route", 0) * LOAD_UNLOAD_TIME_S
                total_time = travel_time + wait_time
                if total_time < best_time:
                    best_time = total_time
                    best_mine = mine
            if best_mine:
                assignments[truck.id] = [best_mine]

        return assignments
