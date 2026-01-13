import pygame
from mazelib import Maze
from mazelib.generate.Prims import Prims

# ---------- Generate Maze ----------
m = Maze()
m.generator = Prims(40, 40)  # Maze size (rows, cols)
m.generate()

# The grid from mazelib includes the walls, so its dimensions are ~2x the input
rows, cols = len(m.grid), len(m.grid[0])

# ---------- Pygame Setup ----------
# NEW: Control the look of the maze with these two variables
CELL_SIZE = 12      # The total size of a cell block (road + wall) in pixels
WALL_THICKNESS = 2  # How many pixels thick the walls should be

# Calculate the width of the road itself
PATH_WIDTH = CELL_SIZE - WALL_THICKNESS

# Calculate the full pixel dimensions for the image
WIDTH, HEIGHT = cols * CELL_SIZE, rows * CELL_SIZE

pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))

# Colors
ROAD_COLOR = (255, 255, 255)  # white roads
WALL_COLOR = (0, 0, 0)        # black walls

# Fill the entire background with the wall color first
screen.fill(WALL_COLOR)

# ---------- Draw Roads (Updated Logic) ----------
# Instead of drawing small squares, we draw larger, overlapping rectangles
# to form continuous paths.
for r in range(rows):
    for c in range(cols):
        # Only draw if the current cell is a path (value is 1)
        if m.grid[r][c] == 1:
            # Draw the main square for this path cell
            pygame.draw.rect(
                screen,
                ROAD_COLOR,
                (c * CELL_SIZE, r * CELL_SIZE, PATH_WIDTH, PATH_WIDTH)
            )
            # If the cell to the right is also a path, draw a connection
            if c + 1 < cols and m.grid[r][c + 1] == 1:
                pygame.draw.rect(
                    screen,
                    ROAD_COLOR,
                    (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, PATH_WIDTH)
                )
            # If the cell below is also a path, draw a connection
            if r + 1 < rows and m.grid[r + 1][c] == 1:
                pygame.draw.rect(
                    screen,
                    ROAD_COLOR,
                    (c * CELL_SIZE, r * CELL_SIZE, PATH_WIDTH, CELL_SIZE)
                )

# ---------- Save Maze as Image ----------
pygame.image.save(screen, "maze_thick_roads.png")
print("Maze saved as maze_thick_roads.png!")

pygame.quit()