# date :1/11/25
import cv2
import numpy as np
import networkx as nx  # <-- ADD THIS LINE
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.signal import convolve2d
import os

def build_graph_from_skeleton(skeleton):
    """
    Builds a NetworkX graph from a skeletonized image.
    Nodes are junctions and endpoints. Edges are the paths between them.
    """
    # Find all skeleton pixels
    points = np.argwhere(skeleton > 0)
    
    # Define a kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    # Convolve to find neighbor counts
    neighbor_counts = convolve2d(skeleton, kernel, mode='same', boundary='fill', fillvalue=0)
    neighbor_counts = neighbor_counts * (skeleton > 0)

    # Find endpoints (1 neighbor) and junctions (> 2 neighbors)
    endpoints = np.argwhere((neighbor_counts == 1))
    junctions = np.argwhere((neighbor_counts > 2))
    
    nodes = np.concatenate([endpoints, junctions], axis=0)
    node_map = {tuple(node[::-1]): i for i, node in enumerate(nodes)}
    
    G = nx.Graph() # This will now correctly use the imported library
    for i, node_pos in enumerate(nodes):
        G.add_node(i, pos=tuple(node_pos[::-1]))

    visited_pixels = set()

    for start_node_idx, start_node_pos_yx in enumerate(nodes):
        y, x = start_node_pos_yx
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                
                # RENAMED VARIABLES HERE
                neighbor_y, neighbor_x = y + dy, x + dx
                
                if 0 <= neighbor_y < skeleton.shape[0] and 0 <= neighbor_x < skeleton.shape[1] and \
                   skeleton[neighbor_y, neighbor_x] > 0 and tuple((neighbor_y, neighbor_x)) not in visited_pixels:
                    
                    path = [(neighbor_x, neighbor_y)]
                    visited_pixels.add(tuple((neighbor_y, neighbor_x)))
                    
                    current_pos_yx = (neighbor_y, neighbor_x)
                    
                    while tuple(current_pos_yx[::-1]) not in node_map:
                        found_next = False
                        cy, cx = current_pos_yx
                        for ddy in [-1, 0, 1]:
                            for ddx in [-1, 0, 1]:
                                if ddy == 0 and ddx == 0: continue
                                
                                # AND RENAMED HERE
                                next_y, next_x = cy + ddy, cx + ddx
                                next_pixel = tuple((next_y, next_x))

                                if 0 <= next_y < skeleton.shape[0] and 0 <= next_x < skeleton.shape[1] and \
                                   skeleton[next_y, next_x] > 0 and next_pixel not in visited_pixels:
                                    
                                    path.append((next_x, next_y))
                                    visited_pixels.add(next_pixel)
                                    current_pos_yx = (next_y, next_x)
                                    found_next = True
                                    break
                            if found_next:
                                break
                        if not found_next:
                            break
                    
                    end_node_pos_xy = tuple(current_pos_yx[::-1])
                    if end_node_pos_xy in node_map:
                        end_node_idx = node_map[end_node_pos_xy]
                        G.add_edge(start_node_idx, end_node_idx, path=path, weight=len(path))

    return G

# --- Main script ---
# Load an image with intersections
img_path = os.path.join(os.path.dirname(__file__), 'gimp1.png') # Use image in same directory
img = cv2.imread(img_path)

if img is None:
    raise ValueError("Image not found. Please check the file path.")

# Convert to grayscale and create a binary mask
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# --- +++ NEW: Use skeletonization for complex roads +++ ---
# Convert mask to boolean for skeletonize function
skeleton = skeletonize(binary_mask > 0).astype(np.uint8) * 255

# --- +++ NEW: Build the graph +++ ---
road_graph = build_graph_from_skeleton(skeleton.copy()) # Pass a copy

print(f"Graph created with {road_graph.number_of_nodes()} nodes and {road_graph.number_of_edges()} edges.")

# --- Visualization ---
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Skeleton')
plt.imshow(skeleton, cmap='gray')
plt.axis('off')

# Draw the graph on a new image
graph_img = img.copy()
node_positions = nx.get_node_attributes(road_graph, 'pos')

# Draw edges
for (u, v, data) in road_graph.edges(data=True):
    pt1 = node_positions[u]
    pt2 = node_positions[v]
    cv2.line(graph_img, pt1, pt2, (255, 0, 0), 2) # Blue lines for edges

# Draw nodes
for node_id, pos in node_positions.items():
    cv2.circle(graph_img, pos, 6, (0, 0, 255), -1) # Red circles for nodes

plt.subplot(1, 3, 3)
plt.title('Road Network Graph')
plt.imshow(cv2.cvtColor(graph_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# --- Now you can use the graph for pathfinding! ---
# For example, find the shortest path between node 0 and node 5
try:
    path_nodes = nx.shortest_path(road_graph, source=0, target=5, weight='weight')
    print(f"\nShortest path between node 0 and 5: {path_nodes}")
    
    # To get the full pixel path:
    full_pixel_path = []
    for i in range(len(path_nodes) - 1):
        edge_data = road_graph.get_edge_data(path_nodes[i], path_nodes[i+1])
        # The path might be stored in reverse, check which end matches
        if road_graph.nodes[path_nodes[i]]['pos'] == edge_data['path'][0]:
            full_pixel_path.extend(edge_data['path'])
        else:
            full_pixel_path.extend(edge_data['path'][::-1])
    
    print(f"Total waypoints in path: {len(full_pixel_path)}")
except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
    print(f"\nCould not find a path: {e}")