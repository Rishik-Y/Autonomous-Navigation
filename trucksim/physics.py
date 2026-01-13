# trucksim/physics.py
import math
from . import config as C

def kmh_to_ms(kmh): return kmh / 3.6
def ms_to_kmh(ms):  return ms * 3.6

TARGET_MS = kmh_to_ms(C.TARGET_KMH)

def current_mass_kg(cargo_ton: float) -> float:
    return C.MASS_KG + cargo_ton * 1000.0

def resist_forces(v_ms: float, mass_kg: float) -> float:
    F_rr = C.CRR * mass_kg * 9.81
    F_d  = 0.5 * C.AIR_DENS * C.CD * C.FRONTAL_AREA * v_ms * v_ms
    return F_rr + F_d

def traction_force_from_power(v_ms: float, throttle: float, mass_kg: float) -> float:
    v_eff = max(v_ms, 0.5)  # avoid singularity near 0
    F_power = (C.P_MAX_W * max(0.0, min(1.0, throttle))) / v_eff
    F_mu = C.MU_TIRE * mass_kg * 9.81
    return min(F_power, F_mu)

def brake_force_from_command(brake_cmd: float, mass_kg: float) -> float:
    brake_cmd = max(0.0, min(1.0, brake_cmd))
    return brake_cmd * mass_kg * C.A_BRAKE_MAX

def stopping_distance(v_ms: float, a_dec: float) -> float:
    if a_dec <= 0.0:
        return float('inf')
    return v_ms * v_ms / (2.0 * a_dec)
