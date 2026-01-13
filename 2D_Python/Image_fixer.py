import json
import numpy as np
import sys

# --- CONFIGURATION ---
# How close two road endings can be to be considered the same junction.
SNAP_THRESHOLD = 10.0

def process_map_data(input_filename="map_data.json", output_filename="map_data_cleaned.json"):
    """
    Loads a hand-drawn map, cleans connections by creating explicit, shared
    junctions, and saves a new, routable map file for simulation.
    """
    print(f"--- Map Data Processor ---")

    # 1. Load the messy map data
    try:
        with open(input_filename, 'r') as f:
            map_data = json.load(f)
        print(f"✅ Successfully loaded '{input_filename}'.")
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: The file '{input_filename}' was not found. Aborting."); sys.exit()
    except json.JSONDecodeError:
        print(f"❌ FATAL ERROR: Could not parse '{input_filename}'. Aborting."); sys.exit()

    edges = map_data.get('edges', [])
    sites = map_data.get('sites', {})
    if not edges:
        print("⚠️ Warning: No roads (edges) found. Nothing to process."); return

    print(f"Found {len(edges)} roads to process.")

    # 2. Collect all road endpoints
    all_endpoints = [tuple(p) for edge in edges if len(edge['shape']) >= 2 for p in (edge['shape'][0], edge['shape'][-1])]

    # 3. Group nearby endpoints into clean, canonical junctions
    print(f"Grouping endpoints with a snap threshold of {SNAP_THRESHOLD}...")
    junction_groups = []
    for point in all_endpoints:
        found_group = False
        for group in junction_groups:
            if np.linalg.norm(np.array(point) - np.mean(group, axis=0)) < SNAP_THRESHOLD:
                group.append(point)
                found_group = True
                break
        if not found_group:
            junction_groups.append([point])

    # Calculate the center of each group to create the final junction
    junction_centers = {f"J{i}": np.mean(group, axis=0) for i, group in enumerate(junction_groups)}
    print(f"Identified {len(junction_centers)} clean junctions.")

    def get_closest_junction_id(point_tuple):
        """Finds the ID of the canonical junction center for a given point."""
        point_np = np.array(point_tuple)
        return min(junction_centers.keys(), key=lambda jid: np.linalg.norm(point_np - junction_centers[jid]))

    # 4. Rebuild the road network, connecting them to the new junction IDs
    print("Rebuilding road network with explicit connections...")
    cleaned_edges = []
    for edge in edges:
        if len(edge['shape']) < 2: continue

        start_junction_id = get_closest_junction_id(tuple(edge['shape'][0]))
        end_junction_id = get_closest_junction_id(tuple(edge['shape'][-1]))

        # Don't create roads that loop back to the same junction
        if start_junction_id == end_junction_id: continue

        # The new shape starts and ends at the perfect junction centers
        new_shape = [junction_centers[start_junction_id].tolist()] + edge['shape'][1:-1] + [junction_centers[end_junction_id].tolist()]

        cleaned_edges.append({
            'from': start_junction_id,
            'to': end_junction_id,
            'shape': new_shape,
            'width': edge['width']
        })

    # 5. Prepare the final data structure with explicit junctions
    serializable_junctions = {jid: {'pos': pos.tolist()} for jid, pos in junction_centers.items()}
    cleaned_map_data = {
        'junctions': serializable_junctions,
        'edges': cleaned_edges,
        'sites': sites
    }

    # 6. Save the cleaned data to a new file
    try:
        with open(output_filename, 'w') as f:
            json.dump(cleaned_map_data, f, indent=2)
        print(f"✅ Success! Cleaned, routable map saved to '{output_filename}'.")
        print("\nYou can now use this new file in your simulation script.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not save the cleaned map file. Reason: {e}")


if __name__ == "__main__":
    process_map_data()

