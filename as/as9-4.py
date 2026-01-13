import pygame
import numpy as np
import math
import heapq  # For A* Priority Queue
import random # For picking random destinations

# --- Constants ---

# --- "MEGA-MAP" NODES (5x Scale) [FIXED] ---
# --- *** EXPANDED "GIGA-MAP" (v3) *** ---
NODES = {
    # --- ORIGINAL "CENTRAL HUB" NODES (Unchanged) ---
    "start_zone":   np.array([150.0, 300.0]),"service_exit_1": np.array([150.0, 280.0]),"service_exit_2": np.array([160.0, 270.0]),"main_hub":       np.array([150.0, 250.0]),
    "ix_west_1":    np.array([120.0, 250.0]),"ix_west_2":    np.array([100.0, 250.0]),"ix_east_1":    np.array([180.0, 250.0]),"ix_east_2":    np.array([200.0, 250.0]),"ix_north_1":   np.array([150.0, 220.0]),"ix_north_2":   np.array([150.0, 200.0]),
    "load_hub_1a":    np.array([120.0, 200.0]),"load_hub_1b":    np.array([100.0, 200.0]),"load_spur_1a_1": np.array([90.0, 190.0]),"load_spur_1a_2": np.array([80.0, 180.0]),"load_zone_1":  np.array([80.0, 160.0]),
    "load_spur_1b_1": np.array([110.0, 190.0]),"load_spur_1b_2": np.array([120.0, 180.0]),"load_zone_2":  np.array([120.0, 160.0]),
    "load_hub_2a":    np.array([180.0, 200.0]),"load_hub_2b":    np.array([200.0, 200.0]),"load_spur_2a_1": np.array([190.0, 170.0]),"load_spur_2a_2": np.array([180.0, 150.0]),"load_spur_2b_1": np.array([210.0, 170.0]),"load_spur_2b_2": np.array([220.0, 150.0]),"load_zone_3":  np.array([200.0, 120.0]),
    "dump_hub_1a":    np.array([180.0, 280.0]),"dump_hub_1b":    np.array([200.0, 280.0]),"dump_spur_1a_1": np.array([190.0, 310.0]),"dump_spur_1a_2": np.array([180.0, 320.0]),"dump_zone_1":  np.array([180.0, 340.0]),
    "dump_spur_1b_1": np.array([210.0, 310.0]),"dump_spur_1b_2": np.array([220.0, 320.0]),"dump_zone_2":  np.array([220.0, 340.0]),
    "connector_1_1":  np.array([150.0, 140.0]),"connector_1_2":  np.array([100.0, 100.0]),"connector_1_3":  np.array([70.0, 150.0]),
    "connector_2_1":  np.array([230.0, 260.0]),"connector_2_2":  np.array([250.0, 250.0]),"connector_2_3":  np.array([260.0, 280.0]),
    
    # --- REGION B: FAR WESTERN PIT (LOAD) (Unchanged) ---
    "fw_haul_1": np.array([-50.0, 250.0]),"fw_haul_2": np.array([-150.0, 250.0]),"fw_haul_3": np.array([-250.0, 250.0]),"fw_hub": np.array([-350.0, 250.0]),
    "fw_pit_1_a": np.array([-350.0, 150.0]),"fw_pit_1_b": np.array([-450.0, 150.0]),"fw_pit_1_c": np.array([-500.0, 250.0]),"fw_pit_1_d": np.array([-450.0, 350.0]),"fw_pit_1_e": np.array([-350.0, 350.0]),"fw_pit_1_f": np.array([-300.0, 300.0]),"fw_pit_1_g": np.array([-300.0, 200.0]),
    "fw_load_spur_1": np.array([-470.0, 150.0]),"load_zone_4": np.array([-490.0, 150.0]),
    "fw_load_spur_2": np.array([-500.0, 270.0]),"load_zone_5": np.array([-500.0, 290.0]),
    "fw_load_spur_3": np.array([-450.0, 370.0]),"load_zone_6": np.array([-450.0, 390.0]),
    "fw_load_spur_4": np.array([-300.0, 320.0]),"load_zone_7": np.array([-300.0, 340.0]),
    "fw_pit_2_a": np.array([-350.0, 100.0]),"fw_pit_2_b": np.array([-500.0, 100.0]),"fw_pit_2_c": np.array([-550.0, 250.0]),"fw_pit_2_d": np.array([-500.0, 400.0]),"fw_pit_2_e": np.array([-350.0, 400.0]),"fw_pit_2_f": np.array([-250.0, 350.0]),"fw_pit_2_g": np.array([-250.0, 150.0]),
    "fw_load_spur_5": np.array([-520.0, 100.0]),"load_zone_8": np.array([-540.0, 100.0]),
    "fw_load_spur_6": np.array([-550.0, 270.0]),"load_zone_9": np.array([-550.0, 290.0]),
    "fw_load_spur_7": np.array([-500.0, 420.0]),"load_zone_10": np.array([-500.0, 440.0]),
    "fw_load_spur_8": np.array([-250.0, 370.0]),"load_zone_11": np.array([-250.0, 390.0]),
    "fw_pit_3_a": np.array([-350.0, 50.0]),"fw_pit_3_b": np.array([-550.0, 50.0]),"fw_pit_3_c": np.array([-600.0, 250.0]),"fw_pit_3_d": np.array([-550.0, 450.0]),"fw_pit_3_e": np.array([-350.0, 450.0]),"fw_pit_3_f": np.array([-200.0, 400.0]),"fw_pit_3_g": np.array([-200.0, 100.0]),
    "fw_load_spur_9": np.array([-570.0, 50.0]), "load_zone_12": np.array([-590.0, 50.0]),
    "fw_load_spur_10": np.array([-600.0, 270.0]),"load_zone_13": np.array([-600.0, 290.0]),
    "fw_load_spur_11": np.array([-550.0, 470.0]),"load_zone_14": np.array([-550.0, 490.0]),
    "fw_load_spur_12": np.array([-200.0, 420.0]),"load_zone_15": np.array([-200.0, 440.0]),

    # --- REGION C: NORTHERN QUARRY (LOAD) (Unchanged) ---
    "n_haul_1": np.array([150.0, 100.0]),"n_haul_2": np.array([150.0, 0.0]),"n_haul_3": np.array([150.0, -100.0]),"n_hub": np.array([150.0, -200.0]),
    "n_q_1_a": np.array([100.0, -200.0]),"n_q_1_b": np.array([50.0, -250.0]),"n_q_1_c": np.array([100.0, -300.0]),"n_q_1_d": np.array([200.0, -300.0]),"n_q_1_e": np.array([250.0, -250.0]),"n_q_1_f": np.array([200.0, -200.0]),
    "n_load_spur_1": np.array([50.0, -270.0]), "load_zone_16": np.array([50.0, -290.0]),
    "n_load_spur_2": np.array([150.0, -320.0]),"load_zone_17": np.array([150.0, -340.0]),
    "n_load_spur_3": np.array([250.0, -270.0]),"load_zone_18": np.array([250.0, -290.0]),
    "n_q_2_a": np.array([50.0, -200.0]),"n_q_2_b": np.array([0.0, -250.0]),"n_q_2_c": np.array([50.0, -350.0]),"n_q_2_d": np.array([250.0, -350.0]),"n_q_2_e": np.array([300.0, -250.0]),"n_q_2_f": np.array([250.0, -200.0]),
    "n_load_spur_4": np.array([0.0, -270.0]),  "load_zone_19": np.array([0.0, -290.0]),
    "n_load_spur_5": np.array([150.0, -370.0]),"load_zone_20": np.array([150.0, -390.0]),
    "n_load_spur_6": np.array([300.0, -270.0]),"load_zone_21": np.array([300.0, -290.0]),

    # --- REGION D: EASTERN DUMP GRIDS (DUMP) (Unchanged) ---
    "e_haul_1": np.array([300.0, 250.0]),"e_haul_2": np.array([400.0, 250.0]),"e_haul_3": np.array([500.0, 250.0]),"e_hub": np.array([600.0, 250.0]),
    "e_grid_1_a": np.array([600.0, 200.0]),"e_grid_1_b": np.array([650.0, 200.0]),"e_grid_1_c": np.array([700.0, 200.0]),"e_grid_1_d": np.array([750.0, 200.0]),
    "dump_zone_3": np.array([650.0, 180.0]),"dump_zone_4": np.array([700.0, 180.0]),"dump_zone_5": np.array([750.0, 180.0]),
    "e_grid_2_a": np.array([600.0, 150.0]),"e_grid_2_b": np.array([650.0, 150.0]),"e_grid_2_c": np.array([700.0, 150.0]),"e_grid_2_d": np.array([750.0, 150.0]),
    "dump_zone_6": np.array([650.0, 130.0]),"dump_zone_7": np.array([700.0, 130.0]),"dump_zone_8": np.array([750.0, 130.0]),
    "e_grid_3_a": np.array([600.0, 300.0]),"e_grid_3_b": np.array([650.0, 300.0]),"e_grid_3_c": np.array([700.0, 300.0]),"e_grid_3_d": np.array([750.0, 300.0]),
    "dump_zone_9": np.array([650.0, 320.0]),"dump_zone_10": np.array([700.0, 320.0]),"dump_zone_11": np.array([750.0, 320.0]),
    "e_grid_4_a": np.array([600.0, 350.0]),"e_grid_4_b": np.array([650.0, 350.0]),"e_grid_4_c": np.array([700.0, 350.0]),"e_grid_4_d": np.array([750.0, 350.0]),
    "dump_zone_12": np.array([650.0, 370.0]),"dump_zone_13": np.array([700.0, 370.0]),"dump_zone_14": np.array([750.0, 370.0]),
    "e_connector_n": np.array([750.0, 225.0]),"e_connector_s": np.array([750.0, 275.0]),

    # --- REGION E: SOUTHERN DUMP SPIRALS (DUMP) (Unchanged) ---
    "s_haul_1": np.array([150.0, 350.0]),"s_haul_2": np.array([150.0, 450.0]),"s_haul_3": np.array([150.0, 550.0]),"s_hub": np.array([150.0, 650.0]),
    "s_sp_1_a": np.array([100.0, 650.0]),"s_sp_1_b": np.array([50.0, 700.0]),"s_sp_1_c": np.array([100.0, 750.0]),"s_sp_1_d": np.array([200.0, 750.0]),"s_sp_1_e": np.array([250.0, 700.0]),"s_sp_1_f": np.array([200.0, 650.0]),
    "s_dump_spur_1": np.array([50.0, 720.0]), "dump_zone_15": np.array([50.0, 740.0]),
    "s_dump_spur_2": np.array([150.0, 770.0]),"dump_zone_16": np.array([150.0, 790.0]),
    "s_dump_spur_3": np.array([250.0, 720.0]),"dump_zone_17": np.array([250.0, 740.0]),
    "s_sp_2_a": np.array([0.0, 650.0]),"s_sp_2_b": np.array([-50.0, 700.0]),"s_sp_2_c": np.array([0.0, 750.0]),"s_sp_2_d": np.array([100.0, 800.0]),"s_sp_2_e": np.array([200.0, 800.0]),"s_sp_2_f": np.array([300.0, 750.0]),"s_sp_2_g": np.array([350.0, 700.0]),"s_sp_2_h": np.array([300.0, 650.0]),
    "s_dump_spur_4": np.array([-50.0, 720.0]),"dump_zone_18": np.array([-50.0, 740.0]),
    "s_dump_spur_5": np.array([50.0, 820.0]), "dump_zone_19": np.array([50.0, 840.0]),
    "s_dump_spur_6": np.array([250.0, 820.0]),"dump_zone_20": np.array([250.0, 840.0]),
    "s_dump_spur_7": np.array([350.0, 720.0]),"dump_zone_21": np.array([350.0, 740.0]),
    "s_connector_1": np.array([0.0, 550.0]),"s_connector_2": np.array([-50.0, 450.0]),
    
    # --- REGION F: SOUTH-WEST SERVICE & DUMPS (DUMP/LOAD) (Unchanged) ---
    "sw_haul_1": np.array([100.0, 300.0]),"sw_haul_2": np.array([50.0, 350.0]),"sw_haul_3": np.array([0.0, 400.0]),"sw_hub": np.array([-100.0, 450.0]),
    "sw_dump_spur_1": np.array([-150.0, 450.0]),"dump_zone_22": np.array([-170.0, 450.0]),
    "sw_dump_spur_2": np.array([-100.0, 500.0]),"dump_zone_23": np.array([-100.0, 520.0]),
    "sw_dump_spur_3": np.array([-50.0, 450.0]), "dump_zone_24": np.array([-30.0, 450.0]),
    "sw_load_spur_1": np.array([-150.0, 400.0]),"load_zone_22": np.array([-170.0, 400.0]),
    "sw_load_spur_2": np.array([-100.0, 350.0]),"load_zone_23": np.array([-100.0, 330.0]),
    "sw_connector_1": np.array([-100.0, 400.0]),"sw_connector_2": np.array([-50.0, 350.0]),
    
    # --- *** NEW REGION G: NORTH-EAST QUARRY (LOAD) *** ---
    "ne_haul_1": np.array([600.0, 100.0]), "ne_haul_2": np.array([600.0, 0.0]), "ne_hub": np.array([600.0, -100.0]),
    # NE Loop 1
    "ne_q_1_a": np.array([550.0, -100.0]), "ne_q_1_b": np.array([500.0, -150.0]), "ne_q_1_c": np.array([550.0, -200.0]), "ne_q_1_d": np.array([650.0, -200.0]), "ne_q_1_e": np.array([700.0, -150.0]), "ne_q_1_f": np.array([650.0, -100.0]),
    "ne_load_spur_1": np.array([500.0, -170.0]), "load_zone_24": np.array([500.0, -190.0]),
    "ne_load_spur_2": np.array([600.0, -220.0]), "load_zone_25": np.array([600.0, -240.0]),
    "ne_load_spur_3": np.array([700.0, -170.0]), "load_zone_26": np.array([700.0, -190.0]),
    # NE Loop 2 (Inner)
    "ne_q_2_a": np.array([500.0, -100.0]), "ne_q_2_b": np.array([450.0, -150.0]), "ne_q_2_c": np.array([500.0, -250.0]), "ne_q_2_d": np.array([700.0, -250.0]), "ne_q_2_e": np.array([750.0, -150.0]), "ne_q_2_f": np.array([700.0, -100.0]),
    "ne_load_spur_4": np.array([450.0, -170.0]), "load_zone_27": np.array([450.0, -190.0]),
    "ne_load_spur_5": np.array([600.0, -270.0]), "load_zone_28": np.array([600.0, -290.0]),
    "ne_load_spur_6": np.array([750.0, -170.0]), "load_zone_29": np.array([750.0, -190.0]),
    
    # --- *** NEW REGION H: SOUTHERN SERVICE BAY (PARK/FUEL) *** ---
    "service_haul_1": np.array([600.0, 400.0]), "service_haul_2": np.array([600.0, 500.0]), "service_hub": np.array([600.0, 600.0]),
    "service_loop_1": np.array([550.0, 600.0]), "service_loop_2": np.array([500.0, 650.0]), "service_loop_3": np.array([550.0, 700.0]),
    "service_loop_4": np.array([650.0, 700.0]), "service_loop_5": np.array([700.0, 650.0]), "service_loop_6": np.array([650.0, 600.0]),
    "parking_1": np.array([500.0, 630.0]),
    "parking_2": np.array([500.0, 670.0]),
    "fuel_1": np.array([700.0, 630.0]),
    "fuel_2": np.array([700.0, 670.0]),
}

# --- EDGES for A* LOGIC (Mega-Map) ---
EDGES = [
    # --- ORIGINAL "CENTRAL HUB" EDGES ---
    ("start_zone", "service_exit_1"),("service_exit_1", "service_exit_2"), ("service_exit_2", "main_hub"),("main_hub", "ix_west_1"),("ix_west_1", "ix_west_2"), ("main_hub", "ix_east_1"),("ix_east_1", "ix_east_2"), ("main_hub", "ix_north_1"),
    ("ix_north_1", "ix_north_2"), ("main_hub", "dump_hub_1a"), ("ix_west_2", "load_hub_1b"),("ix_west_2", "connector_1_3"), ("ix_north_2", "load_hub_1a"),("ix_north_2", "load_hub_2a"),("ix_east_2", "load_hub_2b"),
    ("ix_east_2", "dump_hub_1b"),("ix_east_2", "connector_2_1"), ("load_hub_1a", "load_hub_1b"),("load_hub_1b", "load_spur_1a_1"),("load_spur_1a_1", "load_spur_1a_2"),("load_spur_1a_2", "load_zone_1"),
    ("load_hub_1b", "load_spur_1b_1"),("load_spur_1b_1", "load_spur_1b_2"),("load_spur_1b_2", "load_zone_2"),("load_hub_2a", "load_hub_2b"),("load_hub_2b", "load_spur_2a_1"),("load_spur_2a_1", "load_spur_2a_2"),
    ("load_spur_2a_2", "load_zone_3"),("load_hub_2b", "load_spur_2b_1"),("load_spur_2b_1", "load_spur_2b_2"),("load_spur_2b_2", "load_zone_3"), ("dump_hub_1a", "dump_hub_1b"),("dump_hub_1b", "dump_spur_1a_1"),
    ("dump_spur_1a_1", "dump_spur_1a_2"),("dump_spur_1a_2", "dump_zone_1"),("dump_hub_1b", "dump_spur_1b_1"),("dump_spur_1b_1", "dump_spur_1b_2"),("dump_spur_1b_2", "dump_zone_2"),("load_zone_3", "connector_1_1"),
    ("connector_1_1", "connector_1_2"),("connector_1_2", "connector_1_3"),("connector_1_3", "load_hub_1b"), ("connector_2_1", "connector_2_2"),("connector_2_2", "connector_2_3"),("connector_2_3", "dump_zone_2"),  

    # --- REGION B: FAR WESTERN PIT (LOAD) ---
    ("ix_west_2", "fw_haul_1"), ("fw_haul_1", "fw_haul_2"), ("fw_haul_2", "fw_haul_3"), ("fw_haul_3", "fw_hub"),
    ("fw_hub", "fw_pit_1_a"), ("fw_pit_1_a", "fw_pit_1_b"), ("fw_pit_1_b", "fw_pit_1_c"), ("fw_pit_1_c", "fw_pit_1_d"), ("fw_pit_1_d", "fw_pit_1_e"), ("fw_pit_1_e", "fw_pit_1_f"), ("fw_pit_1_f", "fw_pit_1_g"), ("fw_pit_1_g", "fw_pit_1_a"), # Loop
    ("fw_pit_1_b", "fw_load_spur_1"), ("fw_load_spur_1", "load_zone_4"),("fw_pit_1_c", "fw_load_spur_2"), ("fw_load_spur_2", "load_zone_5"),("fw_pit_1_d", "fw_load_spur_3"), ("fw_load_spur_3", "load_zone_6"),("fw_pit_1_f", "fw_load_spur_4"), ("fw_load_spur_4", "load_zone_7"),
    ("fw_pit_1_a", "fw_pit_2_a"), ("fw_pit_2_a", "fw_pit_2_b"), ("fw_pit_2_b", "fw_pit_2_c"), ("fw_pit_2_c", "fw_pit_2_d"), ("fw_pit_2_d", "fw_pit_2_e"), ("fw_pit_2_e", "fw_pit_2_f"), ("fw_pit_2_f", "fw_pit_2_g"), ("fw_pit_2_g", "fw_pit_2_a"), # Loop
    ("fw_pit_2_b", "fw_load_spur_5"), ("fw_load_spur_5", "load_zone_8"),("fw_pit_2_c", "fw_load_spur_6"), ("fw_load_spur_6", "load_zone_9"),("fw_pit_2_d", "fw_load_spur_7"), ("fw_load_spur_7", "load_zone_10"),("fw_pit_2_f", "fw_load_spur_8"), ("fw_load_spur_8", "load_zone_11"),
    ("fw_pit_2_a", "fw_pit_3_a"), ("fw_pit_3_a", "fw_pit_3_b"), ("fw_pit_3_b", "fw_pit_3_c"), ("fw_pit_3_c", "fw_pit_3_d"), ("fw_pit_3_d", "fw_pit_3_e"), ("fw_pit_3_e", "fw_pit_3_f"), ("fw_pit_3_f", "fw_pit_3_g"), ("fw_pit_3_g", "fw_pit_3_a"), # Loop
    ("fw_pit_3_b", "fw_load_spur_9"), ("fw_load_spur_9", "load_zone_12"),("fw_pit_3_c", "fw_load_spur_10"), ("fw_load_spur_10", "load_zone_13"),("fw_pit_3_d", "fw_load_spur_11"), ("fw_load_spur_11", "load_zone_14"),("fw_pit_3_f", "fw_load_spur_12"), ("fw_load_spur_12", "load_zone_15"),
    ("fw_pit_1_e", "fw_pit_2_e"), ("fw_pit_1_g", "fw_pit_2_g"), ("fw_pit_2_e", "fw_pit_3_e"), ("fw_pit_2_g", "fw_pit_3_g"),

    # --- REGION C: NORTHERN QUARRY (LOAD) ---
    ("ix_north_2", "n_haul_1"), ("n_haul_1", "n_haul_2"), ("n_haul_2", "n_haul_3"), ("n_haul_3", "n_hub"),
    ("n_hub", "n_q_1_a"), ("n_q_1_a", "n_q_1_b"), ("n_q_1_b", "n_q_1_c"), ("n_q_1_c", "n_q_1_d"), ("n_q_1_d", "n_q_1_e"), ("n_q_1_e", "n_q_1_f"), ("n_q_1_f", "n_hub"),
    ("n_q_1_b", "n_load_spur_1"), ("n_load_spur_1", "load_zone_16"),("n_q_1_c", "n_load_spur_2"), ("n_load_spur_2", "load_zone_17"),("n_q_1_e", "n_load_spur_3"), ("n_load_spur_3", "load_zone_18"),
    ("n_q_1_a", "n_q_2_a"), ("n_q_2_a", "n_q_2_b"), ("n_q_2_b", "n_q_2_c"), ("n_q_2_c", "n_q_2_d"), ("n_q_2_d", "n_q_2_e"), ("n_q_2_e", "n_q_2_f"), ("n_q_2_f", "n_q_1_f"),
    ("n_q_2_b", "n_load_spur_4"), ("n_load_spur_4", "load_zone_19"),("n_q_2_c", "n_load_spur_5"), ("n_load_spur_5", "load_zone_20"),("n_q_2_e", "n_load_spur_6"), ("n_load_spur_6", "load_zone_21"),
    ("connector_1_1", "n_haul_1"), # Connects to original map

    # --- REGION D: EASTERN DUMP GRIDS (DUMP) ---
    ("ix_east_2", "e_haul_1"), ("e_haul_1", "e_haul_2"), ("e_haul_2", "e_haul_3"), ("e_haul_3", "e_hub"),
    ("e_hub", "e_grid_1_a"), ("e_grid_1_a", "e_grid_1_b"), ("e_grid_1_b", "e_grid_1_c"), ("e_grid_1_c", "e_grid_1_d"),
    ("e_grid_1_b", "dump_zone_3"), ("e_grid_1_c", "dump_zone_4"), ("e_grid_1_d", "dump_zone_5"),
    ("e_grid_1_a", "e_grid_2_a"), ("e_grid_1_b", "e_grid_2_b"), ("e_grid_1_c", "e_grid_2_c"), ("e_grid_1_d", "e_grid_2_d"),
    ("e_grid_2_a", "e_grid_2_b"), ("e_grid_2_b", "e_grid_2_c"), ("e_grid_2_c", "e_grid_2_d"),
    ("e_grid_2_b", "dump_zone_6"), ("e_grid_2_c", "dump_zone_7"), ("e_grid_2_d", "dump_zone_8"),
    ("e_hub", "e_grid_3_a"), ("e_grid_3_a", "e_grid_3_b"), ("e_grid_3_b", "e_grid_3_c"), ("e_grid_3_c", "e_grid_3_d"),
    ("e_grid_3_b", "dump_zone_9"), ("e_grid_3_c", "dump_zone_10"), ("e_grid_3_d", "dump_zone_11"),
    ("e_grid_3_a", "e_grid_4_a"), ("e_grid_3_b", "e_grid_4_b"), ("e_grid_3_c", "e_grid_4_c"), ("e_grid_3_d", "e_grid_4_d"),
    ("e_grid_4_a", "e_grid_4_b"), ("e_grid_4_b", "e_grid_4_c"), ("e_grid_4_c", "e_grid_4_d"),
    ("e_grid_4_b", "dump_zone_12"), ("e_grid_4_c", "dump_zone_13"), ("e_grid_4_d", "dump_zone_14"),
    ("e_grid_1_d", "e_connector_n"), ("e_grid_2_d", "e_connector_n"),("e_grid_3_d", "e_connector_s"), ("e_grid_4_d", "e_connector_s"),
    ("e_connector_n", "e_connector_s"), ("connector_2_2", "e_haul_1"), # Connects to original map

    # --- REGION E: SOUTHERN DUMP SPIRALS (DUMP) ---
    ("dump_hub_1a", "s_haul_1"), ("s_haul_1", "s_haul_2"), ("s_haul_2", "s_haul_3"), ("s_haul_3", "s_hub"),
    ("s_hub", "s_sp_1_a"), ("s_sp_1_a", "s_sp_1_b"), ("s_sp_1_b", "s_sp_1_c"), ("s_sp_1_c", "s_sp_1_d"), ("s_sp_1_d", "s_sp_1_e"), ("s_sp_1_e", "s_sp_1_f"), ("s_sp_1_f", "s_hub"),
    ("s_sp_1_b", "s_dump_spur_1"), ("s_dump_spur_1", "dump_zone_15"),("s_sp_1_c", "s_dump_spur_2"), ("s_dump_spur_2", "dump_zone_16"),("s_sp_1_e", "s_dump_spur_3"), ("s_dump_spur_3", "dump_zone_17"),
    ("s_sp_1_a", "s_sp_2_a"), ("s_sp_2_a", "s_sp_2_b"), ("s_sp_2_b", "s_sp_2_c"), ("s_sp_2_c", "s_sp_2_d"), ("s_sp_2_d", "s_sp_2_e"), ("s_sp_2_e", "s_sp_2_f"), ("s_sp_2_f", "s_sp_2_g"), ("s_sp_2_g", "s_sp_2_h"), ("s_sp_2_h", "s_sp_1_f"),
    ("s_sp_2_b", "s_dump_spur_4"), ("s_dump_spur_4", "dump_zone_18"),("s_sp_2_d", "s_dump_spur_5"), ("s_dump_spur_5", "dump_zone_19"),
    ("s_sp_2_e", "s_dump_spur_6"), ("s_dump_spur_6", "dump_zone_20"),("s_sp_2_g", "s_dump_spur_7"), ("s_dump_spur_7", "dump_zone_21"),
    ("s_haul_3", "s_connector_1"), ("s_connector_1", "s_connector_2"),

    # --- REGION F: SOUTH-WEST SERVICE & DUMPS (DUMP/LOAD) ---
    ("main_hub", "sw_haul_1"), ("sw_haul_1", "sw_haul_2"), ("sw_haul_2", "sw_haul_3"), ("sw_haul_3", "sw_hub"),
    ("sw_hub", "sw_dump_spur_1"), ("sw_dump_spur_1", "dump_zone_22"),("sw_hub", "sw_dump_spur_2"), ("sw_dump_spur_2", "dump_zone_23"),("sw_hub", "sw_dump_spur_3"), ("sw_dump_spur_3", "dump_zone_24"),
    ("sw_hub", "sw_load_spur_1"), ("sw_load_spur_1", "load_zone_22"),("sw_hub", "sw_load_spur_2"), ("sw_load_spur_2", "load_zone_23"),
    ("sw_haul_2", "sw_connector_2"), ("sw_connector_2", "sw_haul_3"),("sw_hub", "sw_connector_1"), ("sw_connector_1", "sw_haul_3"),
    ("s_connector_2", "sw_hub"), # Connects Region E and F
    
    # --- *** NEW REGION G: NORTH-EAST QUARRY (LOAD) *** ---
    ("e_grid_1_a", "ne_haul_1"), # Connects from Region D
    ("ne_haul_1", "ne_haul_2"), ("ne_haul_2", "ne_hub"),
    # Loop 1
    ("ne_hub", "ne_q_1_a"), ("ne_q_1_a", "ne_q_1_b"), ("ne_q_1_b", "ne_q_1_c"), ("ne_q_1_c", "ne_q_1_d"), ("ne_q_1_d", "ne_q_1_e"), ("ne_q_1_e", "ne_q_1_f"), ("ne_q_1_f", "ne_hub"),
    ("ne_q_1_b", "ne_load_spur_1"), ("ne_load_spur_1", "load_zone_24"),
    ("ne_q_1_c", "ne_load_spur_2"), ("ne_load_spur_2", "load_zone_25"),
    ("ne_q_1_e", "ne_load_spur_3"), ("ne_load_spur_3", "load_zone_26"),
    # Loop 2
    ("ne_q_1_a", "ne_q_2_a"), ("ne_q_2_a", "ne_q_2_b"), ("ne_q_2_b", "ne_q_2_c"), ("ne_q_2_c", "ne_q_2_d"), ("ne_q_2_d", "ne_q_2_e"), ("ne_q_2_e", "ne_q_2_f"), ("ne_q_2_f", "ne_q_1_f"),
    ("ne_q_2_b", "ne_load_spur_4"), ("ne_load_spur_4", "load_zone_27"),
    ("ne_q_2_c", "ne_load_spur_5"), ("ne_load_spur_5", "load_zone_28"),
    ("ne_q_2_e", "ne_load_spur_6"), ("ne_load_spur_6", "load_zone_29"),
    
    # --- *** NEW REGION H: SOUTHERN SERVICE BAY (PARK/FUEL) *** ---
    ("e_grid_4_a", "service_haul_1"), # Connects from Region D
    ("service_haul_1", "service_haul_2"), ("service_haul_2", "service_hub"),
    # Service Loop
    ("service_hub", "service_loop_1"), ("service_loop_1", "service_loop_2"), ("service_loop_2", "service_loop_3"),
    ("service_hub", "service_loop_6"), ("service_loop_6", "service_loop_5"), ("service_loop_5", "service_loop_4"),
    ("service_loop_3", "service_loop_4"), # Connects loop
    # Spurs
    ("service_loop_2", "parking_1"),
    ("service_loop_2", "parking_2"),
    ("service_loop_5", "fuel_1"),
    ("service_loop_5", "fuel_2"),
]

# --- Zone Definitions (Unchanged) ---
# --- *** MODIFIED: Zone Definitions (Giga-Map) *** ---
LOAD_ZONES = [
    "load_zone_1", "load_zone_2", "load_zone_3", # Original
    "load_zone_4", "load_zone_5", "load_zone_6", "load_zone_7", "load_zone_8", "load_zone_9", # Region B
    "load_zone_10", "load_zone_11", "load_zone_12", "load_zone_13", "load_zone_14", "load_zone_15", # Region B
    "load_zone_16", "load_zone_17", "load_zone_18", "load_zone_19", "load_zone_20", "load_zone_21", # Region C
    "load_zone_22", "load_zone_23", # Region F
    # --- NEW ZONES ---
    "load_zone_24", "load_zone_25", "load_zone_26", "load_zone_27", "load_zone_28", "load_zone_29", # Region G
]
# --- *** MODIFIED: Zone Definitions (Giga-Map) *** ---
DUMP_ZONES = [
    "dump_zone_1", "dump_zone_2", 
    # "start_zone", # <-- REMOVED this single start point
    "dump_zone_3", "dump_zone_4", "dump_zone_5", "dump_zone_6", "dump_zone_7", "dump_zone_8", # Region D
    "dump_zone_9", "dump_zone_10", "dump_zone_11", "dump_zone_12", "dump_zone_13", "dump_zone_14", # Region D
    "dump_zone_15", "dump_zone_16", "dump_zone_17", "dump_zone_18", "dump_zone_19", "dump_zone_20", "dump_zone_21", # Region E
    "dump_zone_22", "dump_zone_23", "dump_zone_24", # Region F
    # --- NEW "RESTING" SPOTS ---
    "parking_1", "parking_2", "fuel_1", "fuel_2" # Region H
]

# --- Visual Road Chains for Spline Drawing (Unchanged) ---
# --- *** MODIFIED: Visual Road Chains (Giga-Map) *** ---
VISUAL_ROAD_CHAINS = [
    # --- ORIGINAL "CENTRAL HUB" CHAINS ---
    ["start_zone", "service_exit_1", "service_exit_2", "main_hub"],["main_hub", "ix_west_1", "ix_west_2"],["main_hub", "ix_east_1", "ix_east_2"],["main_hub", "ix_north_1", "ix_north_2"],["main_hub", "dump_hub_1a", "dump_hub_1b"],
    ["ix_west_2", "load_hub_1b"],["ix_north_2", "load_hub_1a", "load_hub_1b"],["ix_north_2", "load_hub_2a", "load_hub_2b"],["ix_east_2", "load_hub_2b"],["ix_east_2", "dump_hub_1b"],["ix_east_2", "connector_2_1", "connector_2_2", "connector_2_3", "dump_zone_2"],
    ["load_hub_1b", "load_spur_1a_1", "load_spur_1a_2", "load_zone_1"],["load_hub_1b", "load_spur_1b_1", "load_spur_1b_2", "load_zone_2"],["load_hub_2b", "load_spur_2a_1", "load_spur_2a_2", "load_zone_3"],
    ["load_hub_2b", "load_spur_2b_1", "load_spur_2b_2", "load_zone_3"],["dump_hub_1b", "dump_spur_1a_1", "dump_spur_1a_2", "dump_zone_1"],["dump_hub_1b", "dump_spur_1b_1", "dump_spur_1b_2", "dump_zone_2"],
    ["load_zone_3", "connector_1_1", "connector_1_2", "connector_1_3", "load_hub_1b"],

    # --- REGION B: FAR WESTERN PIT (LOAD) ---
    ["ix_west_2", "fw_haul_1", "fw_haul_2", "fw_haul_3", "fw_hub"],
    ["fw_hub", "fw_pit_1_a", "fw_pit_1_b", "fw_pit_1_c", "fw_pit_1_d", "fw_pit_1_e", "fw_pit_1_f", "fw_pit_1_g", "fw_pit_1_a"],
    ["fw_pit_1_b", "fw_load_spur_1", "load_zone_4"], ["fw_pit_1_c", "fw_load_spur_2", "load_zone_5"],
    ["fw_pit_1_d", "fw_load_spur_3", "load_zone_6"], ["fw_pit_1_f", "fw_load_spur_4", "load_zone_7"],
    ["fw_pit_1_a", "fw_pit_2_a", "fw_pit_2_b", "fw_pit_2_c", "fw_pit_2_d", "fw_pit_2_e", "fw_pit_1_e"],
    ["fw_pit_1_g", "fw_pit_2_g", "fw_pit_2_f", "fw_pit_2_e"],
    ["fw_pit_2_b", "fw_load_spur_5", "load_zone_8"], ["fw_pit_2_c", "fw_load_spur_6", "load_zone_9"],
    ["fw_pit_2_d", "fw_load_spur_7", "load_zone_10"], ["fw_pit_2_f", "fw_load_spur_8", "load_zone_11"],
    ["fw_pit_2_a", "fw_pit_3_a", "fw_pit_3_b", "fw_pit_3_c", "fw_pit_3_d", "fw_pit_3_e", "fw_pit_2_e"],
    ["fw_pit_2_g", "fw_pit_3_g", "fw_pit_3_f", "fw_pit_3_e"],
    ["fw_pit_3_b", "fw_load_spur_9", "load_zone_12"], ["fw_pit_3_c", "fw_load_spur_10", "load_zone_13"],
    ["fw_pit_3_d", "fw_load_spur_11", "load_zone_14"], ["fw_pit_3_f", "fw_load_spur_12", "load_zone_15"],

    # --- REGION C: NORTHERN QUARRY (LOAD) ---
    ["ix_north_2", "connector_1_1", "n_haul_1", "n_haul_2", "n_haul_3", "n_hub"],
    ["n_hub", "n_q_1_a", "n_q_1_b", "n_q_1_c", "n_q_1_d", "n_q_1_e", "n_q_1_f", "n_hub"],
    ["n_q_1_b", "n_load_spur_1", "load_zone_16"], ["n_q_1_c", "n_load_spur_2", "load_zone_17"], ["n_q_1_e", "n_load_spur_3", "load_zone_18"],
    ["n_q_1_a", "n_q_2_a", "n_q_2_b", "n_q_2_c", "n_q_2_d", "n_q_2_e", "n_q_2_f", "n_q_1_f"],
    ["n_q_2_b", "n_load_spur_4", "load_zone_19"], ["n_q_2_c", "n_load_spur_5", "load_zone_20"], ["n_q_2_e", "n_load_spur_6", "load_zone_21"],

    # --- REGION D: EASTERN DUMP GRIDS (DUMP) ---
    ["ix_east_2", "connector_2_1", "connector_2_2", "e_haul_1", "e_haul_2", "e_haul_3", "e_hub"],
    ["e_hub", "e_grid_1_a", "e_grid_1_b", "e_grid_1_c", "e_grid_1_d"],
    ["e_grid_1_a", "e_grid_2_a"], ["e_grid_1_b", "e_grid_2_b"], ["e_grid_1_c", "e_grid_2_c"], ["e_grid_1_d", "e_grid_2_d"],
    ["e_grid_2_a", "e_grid_2_b", "e_grid_2_c", "e_grid_2_d"],
    ["e_grid_1_b", "dump_zone_3"], ["e_grid_1_c", "dump_zone_4"], ["e_grid_1_d", "dump_zone_5"],
    ["e_grid_2_b", "dump_zone_6"], ["e_grid_2_c", "dump_zone_7"], ["e_grid_2_d", "dump_zone_8"],
    ["e_hub", "e_grid_3_a", "e_grid_3_b", "e_grid_3_c", "e_grid_3_d"],
    ["e_grid_3_a", "e_grid_4_a"], ["e_grid_3_b", "e_grid_4_b"], ["e_grid_3_c", "e_grid_4_c"], ["e_grid_3_d", "e_grid_4_d"],
    ["e_grid_4_a", "e_grid_4_b", "e_grid_4_c", "e_grid_4_d"],
    ["e_grid_3_b", "dump_zone_9"], ["e_grid_3_c", "dump_zone_10"], ["e_grid_3_d", "dump_zone_11"],
    ["e_grid_4_b", "dump_zone_12"], ["e_grid_4_c", "dump_zone_13"], ["e_grid_4_d", "dump_zone_14"],
    ["e_grid_1_d", "e_connector_n", "e_connector_s", "e_grid_3_d"], ["e_grid_2_d", "e_connector_n"], ["e_grid_4_d", "e_connector_s"],

    # --- REGION E: SOUTHERN DUMP SPIRALS (DUMP) ---
    ["dump_hub_1a", "s_haul_1", "s_haul_2", "s_haul_3", "s_hub"],
    ["s_hub", "s_sp_1_a", "s_sp_1_b", "s_sp_1_c", "s_sp_1_d", "s_sp_1_e", "s_sp_1_f", "s_hub"],
    ["s_sp_1_b", "s_dump_spur_1", "dump_zone_15"], ["s_sp_1_c", "s_dump_spur_2", "dump_zone_16"], ["s_sp_1_e", "s_dump_spur_3", "dump_zone_17"],
    ["s_sp_1_a", "s_sp_2_a", "s_sp_2_b", "s_sp_2_c", "s_sp_2_d", "s_sp_2_e", "s_sp_2_f", "s_sp_2_g", "s_sp_2_h", "s_sp_1_f"],
    ["s_sp_2_b", "s_dump_spur_4", "dump_zone_18"], ["s_sp_2_d", "s_dump_spur_5", "dump_zone_19"],
    ["s_sp_2_e", "s_dump_spur_6", "dump_zone_20"], ["s_sp_2_g", "s_dump_spur_7", "dump_zone_21"],
    ["s_haul_3", "s_connector_1", "s_connector_2"],
    
    # --- REGION F: SOUTH-WEST SERVICE & DUMPS (DUMP/LOAD) ---
    ["main_hub", "sw_haul_1", "sw_haul_2", "sw_haul_3", "sw_hub"],
    ["sw_hub", "sw_dump_spur_1", "dump_zone_22"], ["sw_hub", "sw_dump_spur_2", "dump_zone_23"], ["sw_hub", "sw_dump_spur_3", "dump_zone_24"],
    ["sw_hub", "sw_load_spur_1", "load_zone_22"], ["sw_hub", "sw_load_spur_2", "load_zone_23"],
    ["sw_haul_2", "sw_connector_2", "sw_haul_3"], ["sw_hub", "sw_connector_1", "sw_haul_3"],
    ["s_connector_2", "sw_hub"], # Connects Region E and F
    
    # --- *** NEW REGION G: NORTH-EAST QUARRY (LOAD) *** ---
    ["e_grid_1_a", "ne_haul_1", "ne_haul_2", "ne_hub"],
    ["ne_hub", "ne_q_1_a", "ne_q_1_b", "ne_q_1_c", "ne_q_1_d", "ne_q_1_e", "ne_q_1_f", "ne_hub"],
    ["ne_q_1_b", "ne_load_spur_1", "load_zone_24"], ["ne_q_1_c", "ne_load_spur_2", "load_zone_25"], ["ne_q_1_e", "ne_load_spur_3", "load_zone_26"],
    ["ne_q_1_a", "ne_q_2_a", "ne_q_2_b", "ne_q_2_c", "ne_q_2_d", "ne_q_2_e", "ne_q_2_f", "ne_q_1_f"],
    ["ne_q_2_b", "ne_load_spur_4", "load_zone_27"], ["ne_q_2_c", "ne_load_spur_5", "load_zone_28"], ["ne_q_2_e", "ne_load_spur_6", "load_zone_29"],
    
    # --- *** NEW REGION H: SOUTHERN SERVICE BAY (PARK/FUEL) *** ---
    ["e_grid_4_a", "service_haul_1", "service_haul_2", "service_hub"],
    ["service_hub", "service_loop_1", "service_loop_2", "service_loop_3", "service_loop_4", "service_loop_5", "service_loop_6", "service_hub"],
    ["service_loop_2", "parking_1"], ["service_loop_2", "parking_2"],
    ["service_loop_5", "fuel_1"], ["service_loop_5", "fuel_2"],
]
PRE_CALCULATED_SPLINES = [] 

# --- Main Settings (Unchanged) ---
SMOOTH_ITERATIONS = 5 
WIDTH, HEIGHT = 1200, 900 
WHITE, GRAY, BLUE_ACTIVE, RED, PURPLE_NODE, CAR_COLOR = (255, 255, 255), (100, 100, 100), (0, 100, 200, 150), (255, 0, 0), (150, 0, 150), (0, 80, 200)
ROAD_WIDTH_M, ZOOM_FACTOR, PADDING = 4.0, 1.1, 50
METERS_TO_PIXELS, PIXELS_TO_METERS = 3.0, 1.0 / 3.0

# --- Car Physics Parameters (Unchanged) ---
CAR_LENGTH_M, CAR_WIDTH_M, WHEELBASE_M, MASS_KG, CARGO_TON = 4.5, 2.0, 2.8, 1500.0, 1.0
P_MAX_W, CD, FRONTAL_AREA, CRR = 80_000.0, 0.35, 2.2, 0.01
MAX_ACCEL_CMD, MAX_BRAKE_DECEL = 1.5, 1.5
SPEED_KMPH_EMPTY, SPEED_KMPH_LOADED = 35.0, 25.0
SPEED_MS_EMPTY, SPEED_MS_LOADED = SPEED_KMPH_EMPTY / 3.6, SPEED_KMPH_LOADED / 3.6
STEER_MAX_DEG = 35.0

# --- "PRO DRIVER" TUNING PARAMETERS ---
STEER_RATE_DEGPS = 720.0  # <-- Fast steering
STEER_MAX_RAD = math.radians(STEER_MAX_DEG)
STEER_RATE_RADPS = math.radians(STEER_RATE_DEGPS)
JERK_LIMIT = 1.0
LOAD_UNLOAD_TIME_S = 3.0 

# --- "PRO DRIVER" CONTROLLER PARAMETERS ---
LOOKAHEAD_GAIN = 1.2      # <-- Look further ahead
LOOKAHEAD_MIN_M = 4.0   
LOOKAHEAD_MAX_M = 35.0    # <-- Use your new 35.0 value
MAX_LAT_ACCEL = 4.5       # <-- High "tire grip"
SENSOR_NOISE_STD_DEV = 0.5

# --- Physics Functions (Unchanged) ---
def resist_forces(v_ms: float, mass_kg: float) -> float:
    return CRR * mass_kg * 9.81 + 0.5 * 1.225 * CD * FRONTAL_AREA * v_ms**2
def traction_force_from_power(v_ms: float, throttle: float) -> float:
    return (P_MAX_W * np.clip(throttle, 0, 1)) / max(v_ms, 0.5)
def brake_force_from_command(brake_cmd: float, mass_kg: float) -> float:
    return np.clip(brake_cmd, 0, 1) * mass_kg * MAX_BRAKE_DECEL

# --- A* Pathfinding Code (Unchanged) ---
def build_weighted_graph(nodes: dict, edges: list) -> dict:
    graph = {name: [] for name in nodes}
    for n1_name, n2_name in edges:
        p1 = nodes[n1_name]
        p2 = nodes[n2_name]
        distance = np.linalg.norm(p1 - p2)
        graph[n1_name].append((n2_name, distance))
        graph[n2_name].append((n1_name, distance))
    return graph
def a_star_pathfinding(graph: dict, nodes_coords: dict, start_name: str, goal_name: str) -> list[np.ndarray]:
    def h(node_name): return np.linalg.norm(nodes_coords[node_name] - nodes_coords[goal_name])
    open_set = [(h(start_name), start_name)]
    came_from = {}
    g_score = {name: float('inf') for name in nodes_coords}; g_score[start_name] = 0
    f_score = {name: float('inf') for name in nodes_coords}; f_score[start_name] = h(start_name)
    while open_set:
        current_f, current_name = heapq.heappop(open_set)
        if current_name == goal_name:
            path_nodes = []
            temp = current_name
            while temp in came_from:
                path_nodes.append(nodes_coords[temp])
                temp = came_from[temp]
            path_nodes.append(nodes_coords[start_name])
            return list(reversed(path_nodes))
        for neighbor_name, weight in graph[current_name]:
            tentative_g_score = g_score[current_name] + weight
            if tentative_g_score < g_score[neighbor_name]:
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative_g_score
                f_score[neighbor_name] = tentative_g_score + h(neighbor_name)
                if (f_score[neighbor_name], neighbor_name) not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor_name], neighbor_name))
    print(f"A* Error: Path not found from {start_name} to {goal_name}")
    return []

# --- Catmull-Rom Spline Path Generation (Unchanged) ---
def catmull_rom_point(t, p0, p1, p2, p3):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t**2) + (-p0 + 3 * p1 - 3 * p2 + p3) * (t**3))
def generate_curvy_path_from_nodes(node_list: list[np.ndarray], points_per_segment=20) -> list[np.ndarray]:
    all_waypoints_m = []
    if not node_list or len(node_list) < 2: return []
    for i in range(len(node_list) - 1):
        p0 = node_list[0] if i == 0 else node_list[i - 1]
        p1 = node_list[i]
        p2 = node_list[i + 1]
        p3 = node_list[-1] if i >= len(node_list) - 2 else node_list[i + 2]
        if i == 0: all_waypoints_m.append(p1)
        for j in range(1, points_per_segment + 1):
            t = j / float(points_per_segment)
            point = catmull_rom_point(t, p0, p1, p2, p3)
            all_waypoints_m.append(point)
    return all_waypoints_m

# --- Path Class (Unchanged) ---
class Path:
    def __init__(self, waypoints: list[np.ndarray]):
        if not waypoints or len(waypoints) < 2: self.wp = [np.array([0,0]), np.array([1,1])]
        else: self.wp = waypoints
        self.s = [0.0]
        for i in range(1, len(self.wp)): self.s.append(self.s[-1] + np.linalg.norm(self.wp[i] - self.wp[i-1]))
        self.length = self.s[-1]
        if self.length < 1e-6: self.length = 1e-6
    def get_segment_index(self, s: float) -> int:
        s = np.clip(s, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]: return i
        return len(self.s) - 2
    def point_at(self, s_query: float) -> np.ndarray:
        s = np.clip(s_query, 0.0, self.length)
        for i in range(len(self.s) - 1):
            if self.s[i] <= s <= self.s[i+1]:
                s_base, s_end = self.s[i], self.s[i+1]; p_base, p_end = self.wp[i], self.wp[i + 1]
                if s_end - s_base < 1e-6: return p_base
                t = (s - s_base) / (s_end - s_base)
                return p_base + t * (p_end - p_base)
        return self.wp[-1]
    def project(self, p: np.ndarray, s_hint: float) -> float:
        idx_hint = self.get_segment_index(s_hint)
        search_window = 20
        idx_start, idx_end = max(0, idx_hint - search_window), min(len(self.wp) - 1, idx_hint + search_window)
        min_dist_sq, best_s = float('inf'), s_hint
        for i in range(idx_start, idx_end):
            a, b = self.wp[i], self.wp[i + 1]; seg_vec = b - a; ap = p - a; dot_val = np.dot(seg_vec, seg_vec)
            if dot_val < 1e-9: t = 0.0
            else: t = np.clip(np.dot(ap, seg_vec) / dot_val, 0, 1)
            proj = a + t * seg_vec; dist_sq = np.sum((p - proj)**2)
            if dist_sq < min_dist_sq: min_dist_sq, best_s = dist_sq, self.s[i] + t * np.linalg.norm(seg_vec)
        return best_s
    def get_curvature_at(self, s: float) -> float:
        s_before, s_at, s_after = self.point_at(s - 1.0), self.point_at(s), self.point_at(s + 1.0)
        area = 0.5 * abs(s_before[0]*(s_at[1]-s_after[1]) + s_at[0]*(s_after[1]-s_before[1]) + s_after[0]*(s_before[1]-s_at[1]))
        d1, d2, d3 = np.linalg.norm(s_at - s_before), np.linalg.norm(s_after - s_at), np.linalg.norm(s_after - s_before)
        if d1*d2*d3 < 1e-6: return 0.0
        return (4 * area) / (d1 * d2 * d3)

# --- Kalman Filter Class (Unchanged) ---
class KalmanFilter:
    def __init__(self, dt):
        self.x = np.zeros(4); self.dt = dt
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt**2, 0], [dt, 0], [0, 0.5 * dt**2], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.eye(4) * 0.1; self.R = np.eye(2) * (SENSOR_NOISE_STD_DEV**2); self.P = np.eye(4)
    def predict(self, u):
        self.x = self.F @ self.x + self.B @ u; self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        y = z - self.H @ self.x; S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y; self.P = (np.eye(4) - K @ self.H) @ self.P
        except np.linalg.LinAlgError: pass

# --- Car Class (Restored to "Dynamic Task" version) ---
class Car:
    def __init__(self, x_m, y_m, angle=0):
        self.x_m, self.y_m, self.angle = x_m, y_m, angle
        self.speed_ms, self.accel_ms2, self.steer_angle = 0.0, 0.0, 0.0
        self.current_mass_kg = MASS_KG
        self.op_state = "GOING_TO_ENDPOINT" 
        self.op_timer = 0.0
        self.a_cmd_prev, self.s_path_m = 0.0, 0.0
        self.current_node_name = "start_zone" # This will be updated immediately
        self.target_node_name = ""          

    def move(self, accel_cmd, steer_input, dt):
        accel_cmd = np.clip(accel_cmd, self.a_cmd_prev - JERK_LIMIT * dt, self.a_cmd_prev + JERK_LIMIT * dt)
        self.a_cmd_prev = accel_cmd
        F_resist = resist_forces(self.speed_ms, self.current_mass_kg)
        F_needed = self.current_mass_kg * accel_cmd
        if F_needed >= 0:
            F_engine = F_needed + F_resist; throttle = (F_engine * max(self.speed_ms, 0.5)) / P_MAX_W; brake = 0.0
        else:
            F_brake_req = -F_needed + F_resist; brake = F_brake_req / (self.current_mass_kg * MAX_BRAKE_DECEL); throttle = 0.0
        self.steer_angle = np.clip(steer_input, self.steer_angle - STEER_RATE_RADPS * dt, self.steer_angle + STEER_RATE_RADPS * dt)
        F_trac = traction_force_from_power(self.speed_ms, throttle); F_brake = brake_force_from_command(brake, self.current_mass_kg)
        F_net = F_trac - F_resist - F_brake
        self.accel_ms2 = F_net / self.current_mass_kg
        self.speed_ms = max(0.0, self.speed_ms + self.accel_ms2 * dt)
        if abs(self.steer_angle) > 1e-6:
            turn_radius = WHEELBASE_M / math.tan(self.steer_angle); angular_velocity = self.speed_ms / turn_radius; self.angle += angular_velocity * dt
        self.x_m += self.speed_ms * math.cos(self.angle) * dt; self.y_m += self.speed_ms * math.sin(self.angle) * dt

    def update_op_state(self, dt, path_length_m, current_s_m):
        # --- RESTORED: Full State Machine Logic ---
        if self.op_state == "GOING_TO_ENDPOINT": # Going to Load
            if current_s_m >= path_length_m - 2.0 and self.speed_ms < 0.5:
                self.op_state = "LOADING"; self.op_timer = LOAD_UNLOAD_TIME_S
            return +1, SPEED_MS_EMPTY
        elif self.op_state == "LOADING":
            if self.op_timer > 0: self.op_timer -= dt
            else: self.op_state = "RETURNING_TO_START"; self.current_mass_kg = MASS_KG + CARGO_TON * 1000
            return 0, 0.0
        elif self.op_state == "RETURNING_TO_START": # Going to Dump/Park
            if current_s_m <= 2.0 and self.speed_ms < 0.5: 
                self.op_state = "UNLOADING"; self.op_timer = LOAD_UNLOAD_TIME_S
            return -1, SPEED_MS_LOADED
        elif self.op_state == "UNLOADING": # "Unloading" is also "Parking"
            if self.op_timer > 0: self.op_timer -= dt
            else: self.op_state = "GOING_TO_ENDPOINT"; self.current_mass_kg = MASS_KG
            return 0, 0.0
        return 0, 0.0

    def get_noisy_measurement(self):
        return np.array([self.x_m + np.random.normal(0, SENSOR_NOISE_STD_DEV),
                         self.y_m + np.random.normal(0, SENSOR_NOISE_STD_DEV)])

    def draw(self, screen, g_to_s):
        car_center_screen = g_to_s((self.x_m, self.y_m)); length_px = CAR_LENGTH_M * METERS_TO_PIXELS * g_to_s.scale; width_px = CAR_WIDTH_M * METERS_TO_PIXELS * g_to_s.scale
        if length_px < 1 or width_px < 1: return
        car_surface = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
        cargo_ratio = (self.current_mass_kg - MASS_KG) / (CARGO_TON * 1000)
        body_color = tuple(np.array(CAR_COLOR) * (1 - cargo_ratio) + np.array([120, 120, 120]) * cargo_ratio)
        car_surface.fill(body_color); rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(self.angle))
        rect = rotated_surface.get_rect(center=car_center_screen); screen.blit(rotated_surface, rect.topleft)

# --- Path Processing (Chaikin - Unchanged) ---
def chaikin_smoother(points, iterations):
    for _ in range(iterations):
        new_points = []; new_points.append(points[0])
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]; q = (1 - 0.25) * p1 + 0.25 * p2; r = 0.25 * p1 + (1 - 0.25) * p2
            new_points.extend([q, r])
        new_points.append(points[-1]); points = new_points
    return points

# --- Coordinate and Drawing Functions (Unchanged) ---
def grid_to_screen(pos_m, scale, pan):  
    pos_m_np = np.array(pos_m); pos_px = pos_m_np * METERS_TO_PIXELS
    return (int(pos_px[0] * scale + pan[0]), int(pos_px[1] * scale + pan[1]))
def screen_to_grid(pos_px, scale, pan):  
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return (grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS)
def draw_active_path(screen, path: Path, g_to_s, scale):
    if len(path.wp) < 2: return
    smoothed_points_m = chaikin_smoother(path.wp, SMOOTH_ITERATIONS); road_px = [g_to_s(p) for p in smoothed_points_m]
    if len(road_px) > 1:
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        road_width_px = max(2, int((ROAD_WIDTH_M + 0.5) * METERS_TO_PIXELS * scale))
        pygame.draw.lines(s, BLUE_ACTIVE, False, road_px, road_width_px); screen.blit(s, (0,0))
def draw_road_network(screen, g_to_s, scale):
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    for waypoints in PRE_CALCULATED_SPLINES:
        if len(waypoints) < 2: continue
        road_px = [g_to_s(p) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    for node_name, pos_m in NODES.items():
        if node_name in LOAD_ZONES: color = (0, 200, 0)
        elif node_name in DUMP_ZONES: color = (200, 0, 0) # Dumps and Parking spots
        elif node_name == "start_zone": color = (200, 200, 0) # Old start
        else: color = PURPLE_NODE
        pygame.draw.circle(screen, color, g_to_s(pos_m), max(2, int(scale)))

# --- "PRO DRIVER" Main Simulation ---
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE) 
    pygame.display.set_caption("Pro Driver: Dynamic A* with Smooth Braking & Speed Floor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # --- Step 1: Build Graph, Pre-Calculate Splines, Find First Path (Unchanged) ---
    print("Building weighted road network graph...")
    road_graph = build_weighted_graph(NODES, EDGES)
    print(f"Pre-calculating {len(VISUAL_ROAD_CHAINS)} visual road splines...")
    for chain in VISUAL_ROAD_CHAINS:
        node_coords_list = [NODES[node_name] for node_name in chain if node_name in NODES]
        if len(node_coords_list) < 2: continue
        spline_waypoints = generate_curvy_path_from_nodes(node_coords_list)
        PRE_CALCULATED_SPLINES.append(spline_waypoints)
    
    # --- *** MODIFIED: New Start Logic *** ---
    car_start_node = "parking_1" # <-- NEW: Start in the service bay
    car_target_node = random.choice(LOAD_ZONES) # Find a load zone
    
    print(f"A* calculating initial path from '{car_start_node}' to '{car_target_node}'...")
    route_nodes = a_star_pathfinding(road_graph, NODES, car_start_node, car_target_node)
    if not route_nodes: print(f"FATAL ERROR: A* could not find initial path. Exiting."); return
    waypoints_m = generate_curvy_path_from_nodes(route_nodes); current_path = Path(waypoints_m)
    
    # --- *** MODIFIED: Setup Car at new start *** ---
    initial_pos = NODES[car_start_node] # Get coordinates from the node
    initial_angle = math.atan2(waypoints_m[1][1] - waypoints_m[0][1], waypoints_m[1][0] - waypoints_m[0][0])
    car = Car(initial_pos[0], initial_pos[1], angle=initial_angle)
    car.current_node_name = car_start_node # IMPORTANT
    car.target_node_name = car_target_node 
    
    kf = KalmanFilter(dt=1.0/60.0); kf.x = np.array([car.x_m, 0, car.y_m, 0])
    all_nodes_m = list(NODES.values()); min_x_m, max_x_m = min(p[0] for p in all_nodes_m), max(p[0] for p in all_nodes_m)
    min_y_m, max_y_m = min(p[1] for p in all_nodes_m), max(p[1] for p in all_nodes_m); img_w_m, img_h_m = max(1.0, max_x_m - min_x_m), max(1.0, max_y_m - min_y_m)
    scale = min((WIDTH - PADDING * 2) / (img_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (img_h_m * METERS_TO_PIXELS))
    pan = [ PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale) ]
    mouse_dragging, last_mouse_pos = False, None

    # --- Main Loop ---
    running = True; desired_speed_ms = 0.0
    
    while running:
        dt = clock.tick(60) / 1000.0
        if dt == 0: continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: mouse_dragging, last_mouse_pos = True, event.pos
                elif event.button in (4, 5):
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    mouse_pos_m = screen_to_grid(event.pos, scale, pan); scale *= zoom_factor
                    new_screen_pos = grid_to_screen(mouse_pos_m, scale, pan)
                    pan[0] += event.pos[0] - new_screen_pos[0]; pan[1] += event.pos[1] - new_screen_pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0], pan[1] = pan[0] + dx, pan[1] + dy; last_mouse_pos = event.pos

        # --- Step 2: Get State (Unchanged) ---
        est_pos_m = np.array([kf.x[0], kf.x[2]]); est_vel_m = np.array([kf.x[1], kf.x[3]]); est_speed_ms = np.linalg.norm(est_vel_m)
        est_s_path_m = current_path.project(est_pos_m, car.s_path_m); car.s_path_m = est_s_path_m 

        # --- Step 3: Update State Machine & Dynamic Path Re-planning (Unchanged) ---
        prev_op_state = car.op_state
        direction, base_speed_ms = car.update_op_state(dt, current_path.length, est_s_path_m)
        new_op_state = car.op_state

        if new_op_state != prev_op_state:
            print(f"State change: {prev_op_state} -> {new_op_state}")
            if new_op_state == "RETURNING_TO_START": # Finished Loading
                car.current_node_name = car.target_node_name
                
                # --- *** MODIFIED: Now goes to Dumps OR Parking *** ---
                car_target_node = random.choice(DUMP_ZONES) 
                
                car.target_node_name = car_target_node
                print(f"A* re-planning: from '{car.current_node_name}' to '{car_target_node}'")
                route_nodes = a_star_pathfinding(road_graph, NODES, car.current_node_name, car_target_node)
                waypoints_m = generate_curvy_path_from_nodes(list(reversed(route_nodes))); current_path = Path(waypoints_m)
                car.s_path_m = current_path.project(est_pos_m, current_path.length)
            elif new_op_state == "GOING_TO_ENDPOINT": # Finished Dumping/Parking
                car.current_node_name = car.target_node_name
                
                # --- *** MODIFIED: Now goes from Dumps OR Parking *** ---
                car_target_node = random.choice(LOAD_ZONES) 
                
                car.target_node_name = car_target_node
                print(f"A* re-planning: from '{car.current_node_name}' to '{car_target_node}'")
                route_nodes = a_star_pathfinding(road_graph, NODES, car.current_node_name, car_target_node)
                waypoints_m = generate_curvy_path_from_nodes(route_nodes); current_path = Path(waypoints_m)
                car.s_path_m = current_path.project(est_pos_m, 0.0)
        path_length_m = current_path.length 

        # --- *** MODIFIED: "PRO DRIVER" CONTROLLER LOGIC *** ---
        if direction != 0:
            # === 1. STEERING (Unchanged, but uses new tuning) ===
            ld = np.clip(est_speed_ms * LOOKAHEAD_GAIN, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)
            s_target_steer = est_s_path_m + direction * ld
            p_target = current_path.point_at(s_target_steer)
            dx_local = (p_target[0] - est_pos_m[0]) * math.cos(car.angle) + (p_target[1] - est_pos_m[1]) * math.sin(car.angle)
            dy_local = -(p_target[0] - est_pos_m[0]) * math.sin(car.angle) + (p_target[1] - est_pos_m[1]) * math.cos(car.angle)
            alpha = math.atan2(dy_local, max(dx_local, 1e-3))
            steer_input = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), ld)
            steer_input = np.clip(steer_input, -STEER_MAX_RAD, STEER_MAX_RAD)

            # === 2. SPEED (Curvature limit + ***NEW SPEED FLOOR***) ===
            lookahead_dist_speed = 15.0; max_curvature = 0.0
            check_steps = np.linspace(0, lookahead_dist_speed, 10)
            for step in check_steps:
                s_check = est_s_path_m + direction * step
                curvature = current_path.get_curvature_at(s_check)
                max_curvature = max(max_curvature, abs(curvature))
                
            if max_curvature > 1e-4: 
                v_turn_cap_raw = math.sqrt(MAX_LAT_ACCEL / max_curvature)
                
                # --- *** THIS IS THE FIX *** ---
                # 9 km/h is ~2.5 m/s. This is our "speed floor" for turns.
                MIN_CURVE_SPEED_MS = 2.5 
                v_turn_cap = max(v_turn_cap_raw, MIN_CURVE_SPEED_MS)
                # --- *** END OF FIX *** ---
            else: 
                v_turn_cap = base_speed_ms
            
            # === 3. "PRO DRIVER" SMOOTH BRAKING LOGIC (Unchanged) ===
            target_s = 0.0 if direction == -1 else path_length_m
            dist_to_target = abs(target_s - est_s_path_m)
            BRAKING_START_DISTANCE_M = 30.0 
            if dist_to_target < BRAKING_START_DISTANCE_M:
                v_profile_cap = base_speed_ms * (dist_to_target / BRAKING_START_DISTANCE_M)
                v_profile_cap = min(v_profile_cap, base_speed_ms)
            else:
                v_profile_cap = base_speed_ms
            v_panic_cap = math.sqrt(2 * MAX_BRAKE_DECEL * max(0, dist_to_target - 0.5))
            
            # === 4. FINAL SPEED ===
            desired_speed_ms = min(base_speed_ms, v_turn_cap, v_profile_cap, v_panic_cap)
            
            speed_error_ms = desired_speed_ms - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / max(0.4, dt), -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)

        else: # (direction == 0) -> Loading/unloading
            steer_input = 0.0; speed_error_ms = 0.0 - est_speed_ms
            accel_cmd = np.clip(speed_error_ms / max(0.4, dt), -MAX_BRAKE_DECEL, MAX_ACCEL_CMD)
            desired_speed_ms = 0.0 

        # --- Step 5: Update Physics and KF (Unchanged) ---
        car.move(accel_cmd, steer_input, dt)
        accel_vec_m = np.array([car.accel_ms2 * math.cos(car.angle), car.accel_ms2 * math.sin(car.angle)])
        kf.predict(u=accel_vec_m)
        kf.update(z=car.get_noisy_measurement())

        # --- Step 6: Drawing (Unchanged) ---
        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan); g_to_s.scale = scale
        draw_road_network(screen, g_to_s, scale); draw_active_path(screen, current_path, g_to_s, scale)
        car.draw(screen, g_to_s)
        kf_pos_m = (kf.x[0], kf.x[2]); kf_screen = g_to_s(kf_pos_m)
        pygame.draw.circle(screen, RED, kf_screen, 6, 2)

        # HUD
        speed_kmph = car.speed_ms * 3.6
        hud_texts = [
            f"Speed: {speed_kmph:.1f} km/h (Target: {desired_speed_ms*3.6:.1f})",
            f"Mass: {car.current_mass_kg:.0f} kg", f"State: {car.op_state}", f"Target: {car.target_node_name}", 
            f"Tuning: (Grip: {MAX_LAT_ACCEL}m/s², Steer: {STEER_RATE_DEGPS}deg/s)",
            f"Path: {est_s_path_m:.1f}m / {path_length_m:.1f}m (Dir: {direction})"
        ]
        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 22))
        pygame.display.flip()

    pygame.quit()

# --- Main Block ---
if __name__ == '__main__':
    print("Starting 'Pro Driver' simulation with Dynamic A* and Smooth Braking...")
    run_simulation()