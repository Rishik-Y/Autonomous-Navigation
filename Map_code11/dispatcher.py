import numpy as np
import random
import map_data  # Uses local map_data.py
from config import MASS_KG

class Dispatcher:
    def __init__(self, road_graph):
        self.road_graph = road_graph
        self.mine_capacities = {zone: random.randint(2000, 10000) for zone in map_data.LOAD_ZONES}
        print("--- Mine Capacities Initialized ---")
        for k, v in list(self.mine_capacities.items())[:5]:
            print(f"{k}: {v} kg")

    def get_nearest_site(self, current_node, target_list):
        if current_node not in map_data.NODES: return random.choice(target_list)
        curr_pos = map_data.NODES[current_node]
        best_node = None
        min_dist = float('inf')
        for target in target_list:
            if target not in map_data.NODES: continue
            target_pos = map_data.NODES[target]
            dist = np.linalg.norm(target_pos - curr_pos)
            if dist < min_dist:
                min_dist = dist
                best_node = target
        return best_node

    def assign_task(self, car):
        # If just finished loading (or starting full), go to Dump
        if car.op_state == "RETURNING_TO_START" or (car.op_state == "GOING_TO_ENDPOINT" and car.current_mass_kg > MASS_KG + 100):
             target = self.get_nearest_site(car.current_node_name, map_data.DUMP_ZONES)
             return target
        # If just finished unloading (or starting empty), go to Mine
        elif car.op_state == "GOING_TO_ENDPOINT" or (car.op_state == "RETURNING_TO_START" and car.current_mass_kg < MASS_KG + 100):
            active_mines = [m for m, cap in self.mine_capacities.items() if cap > 0]
            if not active_mines:
                print("All mines empty! Returning to parking.")
                return "parking_1"
            target = self.get_nearest_site(car.current_node_name, active_mines)
            return target
        return random.choice(map_data.LOAD_ZONES)

    def record_load(self, mine_node):
        if mine_node in self.mine_capacities:
            taken = 1000 # 1 ton
            self.mine_capacities[mine_node] = max(0, self.mine_capacities[mine_node] - taken)
            print(f"Mine {mine_node} level: {self.mine_capacities[mine_node]} kg")