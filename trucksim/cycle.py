# trucksim/cycle.py
from . import config as C
from .physics import kmh_to_ms

# States
DRIVE_TO_MINE = "DRIVE_TO_MINE"
LOAD          = "LOAD"
UTURN         = "UTURN"
DRIVE_TO_DUMP = "DRIVE_TO_DUMP"
UNLOAD        = "UNLOAD"
IDLE          = "IDLE"

def mine_stop_s(world): return world.road_len_m - C.STOP_ZONE_M
def dump_stop_s(world): return 0.0 + C.STOP_ZONE_M

def init_cycle(world, truck):
    truck.state = DRIVE_TO_MINE
    truck.state_timer = 0.0
    truck.heading = +1  # +1 upwards towards mine, -1 downwards to dump

def update_cycle(dt, world, truck):
    # Returns (target_s, vcap_ms)
    # Loaded vs empty speed caps
    loaded = truck.cargo_ton > 1e-6
    vcap_kmh = C.VCAP_LOADED_KMH if loaded else C.VCAP_EMPTY_KMH
    vcap_ms = kmh_to_ms(vcap_kmh)

    if truck.state == DRIVE_TO_MINE:
        truck.heading = +1
        target_s = mine_stop_s(world)
        # Arrival condition handled outside via controller stop; transition when stopped:
        if abs(truck.s - target_s) < C.STOP_EPS_M and truck.v < 0.2:  # was 0.05
            truck.state = LOAD
            truck.state_timer = 0.0
        return target_s, vcap_ms

    elif truck.state == LOAD:
        target_s = truck.s
        # Pause while loading, progress at LOAD_RATE_TPS up to capacity or remaining mine
        room_t = max(0.0, C.TRUCK_CAPACITY_TON - truck.cargo_ton)
        take_t = min(C.LOAD_RATE_TPS * dt, room_t, world.mine_ton)
        truck.cargo_ton += take_t
        world.mine_ton  -= take_t
        if room_t - take_t <= 1e-6 or world.mine_ton <= 1e-6:
            truck.state = UTURN
            truck.state_timer = C.UTURN_TIME_S
        return target_s, 0.0  # hold still

    elif truck.state == UTURN:
        target_s = truck.s
        truck.state_timer -= dt
        if truck.state_timer <= 0.0:
            truck.heading *= -1
            # Decide next drive state from heading and load
            if truck.heading < 0 and truck.cargo_ton > 1e-6:
                truck.state = DRIVE_TO_DUMP
            elif truck.heading > 0 and truck.cargo_ton <= 1e-6 and world.mine_ton > 1e-6:
                truck.state = DRIVE_TO_MINE
            else:
                # If nothing to do, idle
                truck.state = IDLE
        return target_s, 0.0  # hold still during U-turn

    elif truck.state == DRIVE_TO_DUMP:
        truck.heading = -1
        target_s = dump_stop_s(world)
        if abs(truck.s - target_s) < C.STOP_EPS_M and truck.v < 0.2:
            truck.state = UNLOAD
            truck.state_timer = 0.0
        return target_s, vcap_ms

    elif truck.state == UNLOAD:
        target_s = truck.s
        drop_t = min(C.UNLOAD_RATE_TPS * dt, truck.cargo_ton)
        truck.cargo_ton -= drop_t
        world.dump_ton   += drop_t
        if truck.cargo_ton <= 1e-6:
            truck.state = UTURN
            truck.state_timer = C.UTURN_TIME_S
        return target_s, 0.0

    else:  # IDLE
        return truck.s, 0.0
