# trucksim/fleet.py
from typing import List
from .agent import TruckAgent
from .model import World
from .render import draw_truck
from .camera import scale_px_per_m

class Fleet:
    def __init__(self):
        self.agents: List[TruckAgent] = []

    def add(self, agent: TruckAgent):
        self.agents.append(agent)

    def update(self, dt: float, world: World):
        telemetry = []
        # Simple scheduler: sequential update; replace with round-robin/time-sliced if needed
        for agent in self.agents:
            telemetry.append(agent.step(dt, world))
        return telemetry

    def render(self, screen, world, cam, win_w, win_h):
        # Draw each truck; rendering stays outside the agent class to keep agents UI-agnostic
        for agent in self.agents:
            st = agent.state
            draw_truck(screen, st.s, world, cam, win_w, win_h, st.lane_index, st.cargo_ton, st.state)
