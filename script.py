import json

json_file = 'Simulation/Map/map_data.json'
with open(json_file, 'r') as f:
    data = json.load(f)

# Write map_data.py content
with open('MAP/map_data.py', 'w') as f:
    f.write('import numpy as np\n\n')
    
    f.write('NODES = {\n')
    for k, v in data['NODES'].items():
        f.write(f'    "{k}": np.array([{v[0]}, {v[1]}]),\n')
    f.write('}\n\n')
    
    f.write('EDGES = [\n')
    for e in data['EDGES']:
        f.write(f'    ("{e[0]}", "{e[1]}"),\n')
    f.write(']\n\n')
    
    f.write(f'LOAD_ZONES = {data.get("LOAD_ZONES", [])}\n')
    f.write(f'DUMP_ZONES = {data.get("DUMP_ZONES", [])}\n')
    f.write(f'FUEL_ZONES = {data.get("FUEL_ZONES", [])}\n')
    f.write(f'VISUAL_ROAD_CHAINS = {data.get("VISUAL_ROAD_CHAINS", [])}\n')

print('Regenerated MAP/map_data.py')
