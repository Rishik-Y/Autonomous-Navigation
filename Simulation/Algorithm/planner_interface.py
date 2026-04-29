from abc import ABC, abstractmethod


class GlobalPlannerInterface(ABC):
    @abstractmethod
    def optimize_assignments(self, trucks, site_states):
        """Return {truck_id: [target_node_1, ...]} for the current planning window.

        trucks: iterable of truck-like objects with id/op_state/current_node_name/target_node_name.
        site_states: dict keyed by node name with state dicts (e.g., coal_remaining, en_route).
        """

    @abstractmethod
    def get_travel_time(self, start, end):
        """Return estimated travel time in seconds between two node names (strings)."""


class LocalPlannerInterface(ABC):
    @abstractmethod
    def compute_route(self, graph, start_name, goal_name, cache=None):
        """Return a list of node names from start to goal, or [] when unreachable.

        graph: adjacency list dict {node_name: [(neighbor_name, weight), ...]}.
        cache: optional dict keyed by (start, goal) with cached node-name sequences.
        """
