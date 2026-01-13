# Simulation Configuration

# Number of trucks to simulate
NUM_TRUCKS = 3

# Capacity of each truck in kg
TRUCK_CAPACITY = 100

# The main dump site node name (must match a node in map_data.NODES)
DUMP_SITE = 'dump_zone_1'

# List of active mines (load zones) to be used in the simulation.
# These must be valid keys in map_data.NODES.
# We select a subset of mines to keep the DP algorithm fast.
ACTIVE_MINES = [
    'load_zone_1',
    'load_zone_2',
    'load_zone_3',
    'load_zone_4',
    'load_zone_5'
]

# Initial coal capacity for each active mine in kg
# You can customize this per mine if needed
MINE_CAPACITIES = {
    'load_zone_1': 150,
    'load_zone_2': 80,
    'load_zone_3': 200,
    'load_zone_4': 120,
    'load_zone_5': 90
}

# Simulation speed multiplier (1.0 = real time, >1.0 = faster)
SIM_SPEED_MULTIPLIER = 1.0
