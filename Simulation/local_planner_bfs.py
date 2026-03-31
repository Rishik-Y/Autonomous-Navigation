from collections import deque

from planner_interface import LocalPlannerInterface


class Planner(LocalPlannerInterface):
    def compute_route(self, graph, start_name, goal_name, cache=None):
        if cache and (start_name, goal_name) in cache:
            return list(cache[(start_name, goal_name)])

        if start_name not in graph or goal_name not in graph:
            return []

        queue = deque([start_name])
        came_from = {start_name: None}

        while queue:
            current = queue.popleft()
            if current == goal_name:
                break
            for neighbor, _ in graph[current]:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    queue.append(neighbor)

        if goal_name not in came_from:
            return []

        path = []
        node = goal_name
        while node is not None:
            path.append(node)
            node = came_from[node]
        return list(reversed(path))
