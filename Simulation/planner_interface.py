from abc import ABC, abstractmethod


class GlobalPlannerInterface(ABC):
    @abstractmethod
    def optimize_assignments(self, trucks, site_states):
        """Return {truck_id: [target_node_1, ...]} for the current planning window."""

    @abstractmethod
    def get_travel_time(self, start, end):
        """Return estimated travel time in seconds between two nodes."""


class LocalPlannerInterface(ABC):
    @abstractmethod
    def compute_route(self, graph, start_name, goal_name, cache=None):
        """Return a list of node names from start to goal."""
