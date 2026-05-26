from Algorithm.planner_interface import LocalPlannerInterface
from utils import a_star_pathfinding


class Planner(LocalPlannerInterface):
    def compute_route(self, graph, start_name, goal_name, cache=None):
        return a_star_pathfinding(graph, start_name, goal_name, cache=cache)
