import cv2
import cv2.ximgproc as ximgproc
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import os


# (Paste the extract_waypoints_from_mask function from above right here)
def extract_waypoints_from_mask(road_mask, step_size=25):
    """
    Takes a binary road mask (white road on black background),
    thins it, and traces it to extract an ordered list of waypoints.
    """
    if road_mask is None or road_mask.size == 0:
        return []
    thinned = ximgproc.thinning(road_mask)
    points = np.argwhere(thinned > 0)
    if len(points) < 2:
        print("Not enough points found after thinning.")
        return []
    points = points[:, ::-1]
    pixel_tree = KDTree(points)
    start_index = np.argmin(points[:, 1] * thinned.shape[1] + points[:, 0])
    ordered_points = []
    visited_indices = set()
    current_idx = start_index
    while len(visited_indices) < len(points):
        if current_idx in visited_indices: break
        ordered_points.append(points[current_idx])
        visited_indices.add(current_idx)
        distances, indices = pixel_tree.query(points[current_idx], k=10)
        next_idx = -1
        for i in indices:
            if i not in visited_indices:
                next_idx = i
                break
        if next_idx == -1: break
        current_idx = next_idx
    waypoints = ordered_points[::step_size]
    if ordered_points and list(waypoints[-1]) != list(ordered_points[-1]):
        waypoints.append(ordered_points[-1])
    print(f"Extracted {len(waypoints)} waypoints.")
    return [tuple(p) for p in waypoints]


# --- Main script ---
# Load the image
img_path = os.path.join(os.path.dirname(__file__), 'gimp1.png')  # Use image in same directory
img = cv2.imread(img_path)

if img is None:
    raise ValueError("Image not found. Please check the file path.")

# --- The image processing part of your code is perfect ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Use thresholding for simple drawings instead of Canny
_, binary_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# --- +++ NEW: Get waypoints from the mask +++ ---
# The mask from thresholding is already what we need (white road on black)
waypoints = extract_waypoints_from_mask(binary_mask, step_size=30)

# --- Visualization ---
# Create an image to draw the waypoints on
result_with_waypoints = img.copy()

# Draw the waypoint path as a line
if len(waypoints) > 1:
    for i in range(len(waypoints) - 1):
        cv2.line(result_with_waypoints, waypoints[i], waypoints[i+1], (255, 0, 0), 3) # Blue line

# Draw each waypoint as a circle
for point in waypoints:
    cv2.circle(result_with_waypoints, point, 7, (0, 0, 255), -1) # Red circles

# Display results using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Road Mask')
plt.imshow(binary_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Extracted Waypoints')
plt.imshow(cv2.cvtColor(result_with_waypoints, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the result
cv2.imwrite('result_with_waypoints.png', result_with_waypoints)

# You can now use the 'waypoints' list in your simulation!
print("\nFinal Waypoints:")
print(waypoints)