import pygame
from mazelib import Maze
from mazelib.generate.Prims import Prims
from collections import deque
import time

# --- 1. Pathfinding and Simplification Logic ---

def bfs_solver(grid, start, end):
    """
    A generator that performs a Breadth-First Search and yields its state
    at each step for visualization.
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])  # The queue stores (position, path_so_far)
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()

        # Yield the current state for drawing the exploration process
        yield {
            'visited': visited,
            'queue': {item[0] for item in queue},
            'current_path': path
        }
        
        if (r, c) == end:
            print("Solution Found!")
            # Yield the final solution path and stop
            yield {'solution': path}
            return

        # Explore neighbors (Up, Down, Left, Right)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            # Check if the neighbor is valid
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = path + [(nr, nc)]
                queue.append(((nr, nc), new_path))
    
    print("No solution found.")
    yield {'solution': None} # Indicate failure

def simplify_path(path):
    """Reduces a list of path coordinates to only the start, end, and turn points."""
    if not path or len(path) < 3:
        return path
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        p_prev, p_curr, p_next = path[i-1], path[i], path[i+1]
        dir1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        dir2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        if dir1 != dir2:
            simplified.append(p_curr)
    simplified.append(path[-1])
    return simplified

# --- 2. Pygame Drawing Functions ---

def draw_maze_with_lanes(screen, grid, cell_size, road_color, wall_color, lane_color):
    """Draws the maze with roads and dashed lane markings."""
    screen.fill(wall_color)
    lane_width = max(1, cell_size // 8)
    dash_length = cell_size // 3

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == 1:  # It's a road
                # Draw the base road tile
                pygame.draw.rect(screen, road_color, (c * cell_size, r * cell_size, cell_size, cell_size))
                
                # Center point of the cell
                cx, cy = c * cell_size + cell_size // 2, r * cell_size + cell_size // 2

                # Horizontal lane drawing
                if c + 1 < len(row) and grid[r][c + 1] == 1:
                    pygame.draw.line(screen, lane_color, (cx, cy), (cx + cell_size // 2, cy), lane_width)
                    # Erase part for dash effect
                    pygame.draw.line(screen, road_color, (cx + dash_length, cy), (cx + cell_size // 2, cy), lane_width)
                
                # Vertical lane drawing
                if r + 1 < len(grid) and grid[r + 1][c] == 1:
                    pygame.draw.line(screen, lane_color, (cx, cy), (cx, cy + cell_size // 2), lane_width)
                    # Erase part for dash effect
                    pygame.draw.line(screen, road_color, (cx, cy + dash_length), (cx, cy + cell_size // 2), lane_width)
    
    # Redraw a small circle at intersections to smooth out lane connections
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
             if cell == 1:
                cx, cy = c * cell_size + cell_size // 2, r * cell_size + cell_size // 2
                pygame.draw.circle(screen, lane_color, (cx, cy), lane_width)


def draw_solver_state(screen, state, cell_size):
    """Draws the exploration state of the BFS algorithm."""
    # Colors for visualization
    VISITED_COLOR = (50, 50, 100, 150)  # Semi-transparent dark blue
    QUEUE_COLOR = (50, 100, 50, 150)    # Semi-transparent dark green

    # Draw visited cells
    if 'visited' in state:
        for r, c in state['visited']:
            surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
            surface.fill(VISITED_COLOR)
            screen.blit(surface, (c * cell_size, r * cell_size))

    # Draw cells currently in the queue
    if 'queue' in state:
         for r, c in state['queue']:
            surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
            surface.fill(QUEUE_COLOR)
            screen.blit(surface, (c * cell_size, r * cell_size))

def draw_path(screen, path, cell_size, color, line_thickness=4, circle_radius=0):
    """Draws a path (either solution or simplified waypoints)."""
    if not path: return
    
    pixel_points = [(c * cell_size + cell_size // 2, r * cell_size + cell_size // 2) for r, c in path]
    if len(pixel_points) > 1:
        pygame.draw.lines(screen, color, False, pixel_points, line_thickness)
    if circle_radius > 0:
        for p in pixel_points:
            pygame.draw.circle(screen, color, p, circle_radius)


# --- 3. Main Application ---
def main():
    GRID_SIZE, CELL_SIZE = 25, 24
    
    # --- Maze Generation ---
    m = Maze()
    m.generator = Prims(GRID_SIZE, GRID_SIZE)
    m.generate()
    m.start = (1, 1)
    m.end = (len(m.grid) - 2, len(m.grid[0]) - 2)
    
    # --- Pygame Setup ---
    pygame.init()
    width, height = len(m.grid[0]) * CELL_SIZE, len(m.grid) * CELL_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Visualizing Maze Solver")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16, bold=True)

    # --- Solver Initialization ---
    solver = bfs_solver(m.grid, m.start, m.end)
    solver_state = {}
    final_solution_path = None
    simplified_waypoints = None
    solving_complete = False

    # --- Main Loop ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        # --- Update Solver State (if not complete) ---
        if not solving_complete:
            try:
                solver_state = next(solver)
                if 'solution' in solver_state:
                    solving_complete = True
                    final_solution_path = solver_state['solution']
                    if final_solution_path:
                        simplified_waypoints = simplify_path(final_solution_path)
            except StopIteration:
                solving_complete = True

        # --- Drawing ---
        # 1. Draw the base maze with road lanes
        draw_maze_with_lanes(screen, m.grid, CELL_SIZE, 
                             road_color=(128, 128, 128), 
                             wall_color=(20, 20, 20), 
                             lane_color=(255, 255, 0))

        # 2. Draw the solver's exploration progress
        if not solving_complete:
            draw_solver_state(screen, solver_state, CELL_SIZE)
        
        # 3. Once solved, draw the final path and waypoints
        else:
            if final_solution_path:
                # Draw the full solution path in a thin red line
                draw_path(screen, final_solution_path, CELL_SIZE, color=(200, 0, 0, 180), line_thickness=4)
                # Draw the simplified waypoints as thick orange circles/lines
                draw_path(screen, simplified_waypoints, CELL_SIZE, color=(255, 165, 0), line_thickness=5, circle_radius=7)

        pygame.display.flip()
        clock.tick(30) # Control the speed of the animation

    pygame.quit()

if __name__ == '__main__':
    main()