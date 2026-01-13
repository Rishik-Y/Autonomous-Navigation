import pygame
import math
import numpy as np
from scipy.spatial import KDTree
import os

# --- Constants ---
WIDTH, HEIGHT = 900, 900
WHITE = (255, 255, 255)
GRAY = (100, 100, 100) # Road color
BLACK = (0, 0, 0)
HUD_BG = (240, 240, 240)
HUD_TEXT = (10, 10, 10)
ROAD_WIDTH_PX = 50 # Road thickness
ZOOM_FACTOR = 1.1
PADDING = 50
SMOOTH_ITERATIONS = 3 # How many times to apply the smoother. More = smoother.
SMOOTH_FACTOR = 0.25  # How much to "cut" the corners (0.25 is standard)

# --- Image Processing Function (Unchanged) ---
def generate_waypoints_from_image(image_path, step_size=15):
    """Loads a PNG, finds a black path, and returns an ordered list of waypoints."""
    if not os.path.exists(image_path):
        print(f"--- ERROR ---")
        print(f"Image file not found at the specified path: '{image_path}'")
        print(f"---------------")
        return None

    print(f"Loading image from: {image_path}")
    try:
        image = pygame.image.load(image_path)
    except pygame.error as e:
        print(f"Pygame could not load the image. Error: {e}")
        return None
        
    image_w, image_h = image.get_size()
    
    path_pixels = []
    BLACK_THRESHOLD = 100
    for x in range(image_w):
        for y in range(image_h):
            color = image.get_at((x, y))
            r, g, b = color.r, color.g, color.b
            if r + g + b < BLACK_THRESHOLD:
                path_pixels.append((x, y))

    if not path_pixels:
        print(f"--- WARNING ---")
        print(f"No black pixels found in '{image_path}'.")
        print(f"---------------")
        return []
    
    print(f"Found {len(path_pixels)} path pixels. Tracing path...")
    
    pixel_tree = KDTree(path_pixels)
    start_pixel_idx = np.argmin([p[1] * image_w + p[0] for p in path_pixels])
    
    ordered_path = []
    visited_indices = set()
    current_idx = start_pixel_idx
    
    for _ in range(len(path_pixels)):
        if current_idx in visited_indices: break
        ordered_path.append(path_pixels[current_idx])
        visited_indices.add(current_idx)
        distances, indices = pixel_tree.query(path_pixels[current_idx], k=min(20, len(path_pixels)))
        next_idx = -1
        for i in indices:
            if i not in visited_indices:
                next_idx = i
                break
        if next_idx == -1: break
        current_idx = next_idx

    # Get simplified points
    simplified_path = [np.array(p) for p in ordered_path[::step_size]]
    if ordered_path and list(simplified_path[-1]) != ordered_path[-1]:
         simplified_path.append(np.array(ordered_path[-1]))

    print(f"Path traced. Simplified to {len(simplified_path)} raw waypoints.")
    return simplified_path

# --- +++ NEW: Path Smoothing Function +++ ---
def chaikin_smoother(points, iterations, factor):
    """
    Smooths a path using Chaikin's algorithm.
    'factor' controls how far to cut the corner (0.25 is standard).
    """
    for _ in range(iterations):
        new_points = []
        if not points:
            return []
            
        # Keep the first point
        new_points.append(points[0]) 
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            
            # Calculate the two new points
            q = (1 - factor) * p1 + factor * p2
            r = factor * p1 + (1 - factor) * p2
            
            new_points.append(q)
            new_points.append(r)
            
        # Keep the last point
        new_points.append(points[-1])
        points = new_points
        
    return points

# --- Coordinate and Drawing Functions ---
def grid_to_screen(pos, scale, pan_offset):
    px = (pos[0] * scale) + pan_offset[0]
    py = (pos[1] * scale) + pan_offset[1]
    return (int(px), int(py))

def screen_to_grid(screen_pos, scale, pan_offset):
    gx = (screen_pos[0] - pan_offset[0]) / scale
    gy = (screen_pos[1] - pan_offset[1]) / scale
    return (gx, gy)

# --- +++ MODIFIED: Drawing Function +++ ---
def draw_smoothed_road(screen, waypoints, grid_to_screen_func, scale):
    """
    Takes raw waypoints, smooths them, and draws a clean road.
    """
    if len(waypoints) < 3: # Need at least 3 points to smooth
        return
    
    # 1. Smooth the raw waypoints
    smoothed_points = chaikin_smoother(waypoints, SMOOTH_ITERATIONS, SMOOTH_FACTOR)
    
    # 2. Convert smoothed points to screen coordinates
    road_points_px = [grid_to_screen_func(p) for p in smoothed_points]
    
    # 3. Draw the final road as a single thick line
    if len(road_points_px) > 1:
        line_thickness = max(1, int(ROAD_WIDTH_PX * scale))
        pygame.draw.lines(screen, GRAY, False, road_points_px, line_thickness)

# --- Main ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Smoothed Road Generation Viewer")
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("Arial", 18, bold=True)

    IMAGE_FILE = os.path.join(os.path.dirname(__file__), 'gimp.png')
    raw_waypoints = generate_waypoints_from_image(IMAGE_FILE, step_size=20)

    if raw_waypoints is None or len(raw_waypoints) < 3:
        print("Could not generate a valid path (need at least 3 points for smoothing). Exiting.")
        pygame.time.wait(5000) 
        return

    # --- Setup View Controls (Unchanged) ---
    temp_img = pygame.image.load(IMAGE_FILE)
    img_width, img_height = temp_img.get_size()
    scale_x = (WIDTH - PADDING * 2) / img_width
    scale_y = (HEIGHT - PADDING * 2) / img_height
    scale = min(scale_x, scale_y) 
    pan_offset_px = [PADDING, PADDING]
    mouse_dragging = False
    last_mouse_pos = None

    # --- Main Loop ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    mouse_dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4: # Scroll up
                    mouse_pos_screen = pygame.mouse.get_pos()
                    grid_pos_before = screen_to_grid(mouse_pos_screen, scale, pan_offset_px)
                    scale *= ZOOM_FACTOR
                    screen_pos_after_scale = grid_to_screen(grid_pos_before, scale, pan_offset_px)
                    pan_offset_px[0] += mouse_pos_screen[0] - screen_pos_after_scale[0]
                    pan_offset_px[1] += mouse_pos_screen[1] - screen_pos_after_scale[1]
                elif event.button == 5: # Scroll down
                    mouse_pos_screen = pygame.mouse.get_pos()
                    grid_pos_before = screen_to_grid(mouse_pos_screen, scale, pan_offset_px)
                    scale /= ZOOM_FACTOR
                    screen_pos_after_scale = grid_to_screen(grid_pos_before, scale, pan_offset_px)
                    pan_offset_px[0] += mouse_pos_screen[0] - screen_pos_after_scale[0]
                    pan_offset_px[1] += mouse_pos_screen[1] - screen_pos_after_scale[1]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    current_mouse_pos = pygame.mouse.get_pos()
                    dx = current_mouse_pos[0] - last_mouse_pos[0]
                    dy = current_mouse_pos[1] - last_mouse_pos[1]
                    pan_offset_px[0] += dx; pan_offset_px[1] += dy
                    last_mouse_pos = current_mouse_pos
        
        # --- Drawing ---
        screen.fill(WHITE)
        g_to_s_func = lambda pos: grid_to_screen(pos, scale, pan_offset_px)
        
        # --- +++ CALL THE NEW DRAW FUNCTION +++ ---
        draw_smoothed_road(screen, raw_waypoints, g_to_s_func, scale)
        
        # (Removed the old yellow debug line as it's no longer accurate)

        # Draw simplified HUD
        hud_surface = pygame.Surface((250, 60), pygame.SRCALPHA)
        hud_surface.fill((*HUD_BG, 200))
        text1 = hud_font.render("Left Click + Drag to Pan", True, HUD_TEXT)
        text2 = hud_font.render("Mouse Wheel to Zoom", True, HUD_TEXT)
        hud_surface.blit(text1, (10, 5)); hud_surface.blit(text2, (10, 30))
        screen.blit(hud_surface, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()