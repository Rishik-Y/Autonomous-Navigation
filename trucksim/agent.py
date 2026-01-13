# trucksim/agent.py
from dataclasses import dataclass
from .model import TruckState, ControllerState, World
from .lanes import default_lane, lane_center_x
from .physics import current_mass_kg, resist_forces, traction_force_from_power, brake_force_from_command
from .control import controller
from .cycle import init_cycle, update_cycle
from . import config as C


@dataclass
class AgentConfig:
    id: str
    lane_index: int


class TruckAgent:
    def __init__(self, agent_id: str, world: World, lane_index: int | None = None):
        self.id = agent_id
        self.state = TruckState()
        self.ctrl  = ControllerState()
        self.state.lane_index = lane_index if lane_index is not None else default_lane(world.lanes, C.DRIVE_SIDE)
        init_cycle(world, self.state)

    def step(self, dt: float, world: World):
        # Compute target stop and speed cap per trip state, and update loading/unloading via cycle
        target_s, vcap_ms = update_cycle(dt, world, self.state)

        mass_kg = current_mass_kg(self.state.cargo_ton)
        throttle, brake = controller(dt, self.state.s, self.state.v, mass_kg, target_s, vcap_ms, self.ctrl)

        # Forces
        F_res   = resist_forces(self.state.v, mass_kg)
        F_trac  = traction_force_from_power(self.state.v, throttle, mass_kg)
        F_brake = brake_force_from_command(brake, mass_kg)

        # Longitudinal dynamics: dv/dt = (F_trac - F_res - F_brake)/m
        dv = (F_trac - F_res - F_brake) / mass_kg
        self.state.a = dv
        self.state.v = max(0.0, min(self.state.v + dv * dt, vcap_ms))

        # Position integrates with heading
        self.state.s = self.state.s + self.state.heading * self.state.v * dt

        # Clamp position to the target if slight overshoot; do NOT force v=0 here
        if self.state.heading > 0 and self.state.s > target_s:
            self.state.s = target_s
        elif self.state.heading < 0 and self.state.s < target_s:
            self.state.s = target_s

        # Return a minimal telemetry dict for logging or multi-agent coordination
        return {
            "id": self.id,
            "s": self.state.s,
            "v": self.state.v,
            "a": self.state.a,
            "cargo_ton": self.state.cargo_ton,
            "state": self.state.state,
            "lane_center_x": lane_center_x(self.state.lane_index, world.lane_width_m),
        }
