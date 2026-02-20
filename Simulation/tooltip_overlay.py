"""
Tooltip Overlay Module for Main Simulation
Provides hover detection and tooltip rendering for mines, trucks, and dump sites.
"""
import pygame
import numpy as np
import map_loader as map_data
from config import METERS_TO_PIXELS, PIXELS_TO_METERS, CAR_LENGTH_M

# Click/hover threshold in pixels
HOVER_THRESHOLD_PX = 20
HOVER_THRESHOLD_TRUCK_PX = 25

# Tooltip styling
TOOLTIP_BG_COLOR = (50, 50, 50, 220)
TOOLTIP_TEXT_COLOR = (255, 255, 255)
TOOLTIP_BORDER_COLOR = (100, 100, 100)
TOOLTIP_PADDING = 8


def screen_to_grid(pos_px, scale, pan):
    """Convert screen position to grid (world) coordinates."""
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return np.array([grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS])


def get_hovered_entity(mouse_pos, scale, pan, cars, dispatcher):
    """
    Detect what entity the mouse is hovering over.
    
    Returns: dict with 'type' and relevant data, or None if nothing hovered.
    Types: 'coal_mine', 'dump_site', 'truck'
    """
    mouse_m = screen_to_grid(mouse_pos, scale, pan)
    threshold_m = (HOVER_THRESHOLD_PX / scale) * PIXELS_TO_METERS
    threshold_truck_m = (HOVER_THRESHOLD_TRUCK_PX / scale) * PIXELS_TO_METERS
    
    # 1. Check trucks first (priority)
    for car in cars:
        car_pos = np.array([car.x_m, car.y_m])
        dist = np.linalg.norm(mouse_m - car_pos)
        if dist < threshold_truck_m:
            return {
                'type': 'truck',
                'car': car,
                'id': car.id,
                'current_mass': car.current_mass_kg,
                'state': car.op_state,
                'target': car.target_node_name
            }
    
    # 2. Check coal mines (load zones)
    for zone_name in map_data.LOAD_ZONES:
        if zone_name not in map_data.NODES:
            continue
        zone_pos = map_data.NODES[zone_name]
        dist = np.linalg.norm(mouse_m - zone_pos)
        if dist < threshold_m:
            # Get coal capacity from dispatcher
            state = dispatcher.site_states.get(zone_name, {})
            coal_remaining = state.get('coal_remaining', float('inf'))
            coal_initial = dispatcher.coal_capacities.get(zone_name, float('inf'))
            en_route = state.get('en_route', 0)
            
            return {
                'type': 'coal_mine',
                'name': zone_name,
                'coal_remaining': coal_remaining,
                'coal_initial': coal_initial,
                'en_route': en_route
            }
    
    # 3. Check dump sites
    for zone_name in map_data.DUMP_ZONES:
        if zone_name not in map_data.NODES:
            continue
        zone_pos = map_data.NODES[zone_name]
        dist = np.linalg.norm(mouse_m - zone_pos)
        if dist < threshold_m:
            state = dispatcher.site_states.get(zone_name, {})
            coal_dumped = state.get('coal_dumped', 0)
            en_route = state.get('en_route', 0)
            
            return {
                'type': 'dump_site',
                'name': zone_name,
                'coal_dumped': coal_dumped,
                'en_route': en_route
            }
    
    return None


def draw_tooltip(screen, mouse_pos, entity_info, font):
    """
    Draw a tooltip near the mouse cursor with entity information.
    """
    if entity_info is None:
        return
    
    # Build tooltip text lines based on entity type
    lines = []
    
    if entity_info['type'] == 'coal_mine':
        name = entity_info['name']
        remaining = entity_info['coal_remaining']
        initial = entity_info['coal_initial']
        en_route = entity_info['en_route']
        
        lines.append(f"⛏ {name}")
        if initial == float('inf'):
            lines.append(f"Coal: ∞ (unlimited)")
        else:
            lines.append(f"Coal: {int(remaining)}/{int(initial)} kg")
        if en_route > 0:
            lines.append(f"Trucks en route: {en_route}")
    
    elif entity_info['type'] == 'dump_site':
        name = entity_info['name']
        dumped = entity_info['coal_dumped']
        en_route = entity_info['en_route']
        
        lines.append(f"📦 {name}")
        lines.append(f"Coal dumped: {int(dumped)} kg")
        if en_route > 0:
            lines.append(f"Trucks en route: {en_route}")
    
    elif entity_info['type'] == 'truck':
        truck_id = entity_info['id']
        mass = entity_info['current_mass']
        state = entity_info['state']
        target = entity_info['target']
        
        # Calculate cargo (mass minus empty truck mass ~1500 kg)
        from config import MASS_KG, CARGO_TON
        empty_mass = MASS_KG
        max_cargo = CARGO_TON * 1000  # Convert tons to kg
        current_cargo = max(0, mass - empty_mass)
        
        lines.append(f"🚚 Truck {truck_id}")
        lines.append(f"Cargo: {int(current_cargo)}/{int(max_cargo)} kg")
        lines.append(f"State: {state}")
        if target:
            lines.append(f"Target: {target}")
    
    if not lines:
        return
    
    # Calculate tooltip size
    text_surfaces = [font.render(line, True, TOOLTIP_TEXT_COLOR) for line in lines]
    max_width = max(surf.get_width() for surf in text_surfaces)
    total_height = sum(surf.get_height() for surf in text_surfaces) + (len(text_surfaces) - 1) * 4
    
    tooltip_width = max_width + TOOLTIP_PADDING * 2
    tooltip_height = total_height + TOOLTIP_PADDING * 2
    
    # Position tooltip (offset from cursor, keep on screen)
    tooltip_x = mouse_pos[0] + 15
    tooltip_y = mouse_pos[1] + 15
    
    screen_w, screen_h = screen.get_size()
    if tooltip_x + tooltip_width > screen_w - 10:
        tooltip_x = mouse_pos[0] - tooltip_width - 10
    if tooltip_y + tooltip_height > screen_h - 10:
        tooltip_y = mouse_pos[1] - tooltip_height - 10
    
    # Draw tooltip background
    tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_width, tooltip_height)
    tooltip_surface = pygame.Surface((tooltip_width, tooltip_height), pygame.SRCALPHA)
    tooltip_surface.fill(TOOLTIP_BG_COLOR)
    screen.blit(tooltip_surface, tooltip_rect.topleft)
    
    # Draw border
    pygame.draw.rect(screen, TOOLTIP_BORDER_COLOR, tooltip_rect, 2)
    
    # Draw text
    y_offset = tooltip_y + TOOLTIP_PADDING
    for surf in text_surfaces:
        screen.blit(surf, (tooltip_x + TOOLTIP_PADDING, y_offset))
        y_offset += surf.get_height() + 4
