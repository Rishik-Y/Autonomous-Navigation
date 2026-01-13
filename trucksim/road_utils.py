import math

def generate_c_shape_road(radius=21, arc_deg=210, center_x=0, center_y=0, resolution=1.0):
    """
    Generate C-shape waypoints centered at (center_x, center_y),
    with specified radius and arc length in degrees.
    
    Parameters
    ----------
    radius : float
        Radius of the circular arc (meters)
    arc_deg : float
        Angle of the arc in degrees (partial circle, 210 here)
    center_x : float
        X coordinate of circle center
    center_y : float
        Y coordinate of circle center
    resolution : float
        Approximate segment length between points (meters)
    
    Returns
    -------
    List of (x, y) tuples representing waypoints along C-shaped road
    """
    arc_rad = math.radians(arc_deg)
    total_length = radius * arc_rad
    num_points = max(2, int(total_length / resolution))
    
    # Starting angle so that ends open: for C shape opening horizontally
    # e.g. start at 75 degrees, span 210 degrees
    start_angle = math.radians(75)  # Adjust so that opening aligns with straight parts
    
    waypoints = []
    for i in range(num_points + 1):
        angle = start_angle + (arc_rad * i / num_points)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        waypoints.append((x,y))
    return waypoints
