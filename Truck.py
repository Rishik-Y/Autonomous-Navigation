import pygame
import sys
import time
import math

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)

# Screen dimensions
WIDTH, HEIGHT = 1200, 700

def get_user_input():
    print("=== Multi-Truck Coal Mine Simulation Setup ===")
    try:
        distance_km = float(input("Enter distance between dump site and coal mine (km): "))
        coal_amount = float(input("Enter amount of coal in the coal mine (kg): "))
        truck_capacity = float(input("Enter capacity of each truck (kg): "))
        truck_speed = float(input("Enter speed of the trucks (km/h): "))
        load_time = float(input("Enter time to load each truck (seconds): "))
        num_trucks = int(input("Enter number of trucks: "))

        # Validate inputs
        if distance_km <= 0 or coal_amount <= 0 or truck_capacity <= 0 or truck_speed <= 0 or load_time <= 0 or num_trucks <= 0:
            print("Error: All values must be positive numbers!")
            return get_user_input()

        if num_trucks > 10:
            print("Warning: Maximum 10 trucks recommended for optimal visualization")
            num_trucks = min(num_trucks, 10)

        return distance_km, coal_amount, truck_capacity, truck_speed, load_time, num_trucks
    except ValueError:
        print("Error: Please enter valid numbers only!")
        return get_user_input()

class Truck:
    def __init__(self, truck_id, capacity, speed_pps, load_time, dump_site_pos, coal_mine_pos, color):
        self.id = truck_id
        self.capacity = capacity
        self.speed_pps = speed_pps
        self.load_time = load_time
        self.dump_site_pos = dump_site_pos
        self.coal_mine_pos = coal_mine_pos
        self.position = list(dump_site_pos)
        self.state = 'to_mine'  # to_mine, loading, to_dump, unloading, waiting
        self.load_timer = 0
        self.cargo = 0
        self.trips_completed = 0
        self.color = color
        self.road_offset = 30  # vertical offset for road separation

    def update(self, dt, coal_mine, loading_queue, trucks):
        min_spacing = 35  # pixels
        if self.state == 'to_mine':
            # Move horizontally on upper road
            target_x = self.coal_mine_pos[0]
            target_y = self.coal_mine_pos[1] - self.road_offset
            # Find truck ahead
            ahead = None
            for t in trucks:
                if t.id < self.id and t.state == 'to_mine':
                    ahead = t
            if ahead:
                dx = ahead.position[0] - self.position[0]
                dy = ahead.position[1] - self.position[1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_spacing:
                    return
            # Move horizontally until near coal mine
            if abs(self.position[0] - target_x) > 5:
                direction = (target_x - self.position[0], 0)
                move_dist = self.speed_pps * dt
                self.position[0] += math.copysign(min(move_dist, abs(direction[0])), direction[0])
                self.position[1] = target_y
            else:
                self.position[0] = target_x
                self.position[1] = target_y
                if coal_mine.coal_amount > 0:
                    if self.id not in loading_queue:
                        loading_queue.append(self.id)
                    if loading_queue and loading_queue[0] == self.id and not any(truck.state == 'loading' for truck in trucks if truck.id != self.id):
                        self.state = 'loading'
                        self.load_timer = 0
                    else:
                        self.state = 'waiting'
        elif self.state == 'to_dump':
            # Move horizontally on lower road
            target_x = self.dump_site_pos[0]
            target_y = self.dump_site_pos[1] + self.road_offset
            ahead = None
            for t in trucks:
                if t.id < self.id and t.state == 'to_dump':
                    ahead = t
            if ahead:
                dx = ahead.position[0] - self.position[0]
                dy = ahead.position[1] - self.position[1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_spacing:
                    return
            if abs(self.position[0] - target_x) > 5:
                direction = (target_x - self.position[0], 0)
                move_dist = self.speed_pps * dt
                self.position[0] += math.copysign(min(move_dist, abs(direction[0])), direction[0])
                self.position[1] = target_y
            else:
                self.position[0] = target_x
                self.position[1] = target_y
                self.state = 'unloading'
                self.load_timer = 0
        elif self.state == 'waiting':
            # Wait for turn to load
            if loading_queue and loading_queue[0] == self.id and coal_mine.coal_amount > 0 and not any(truck.state == 'loading' for truck in trucks if truck.id != self.id):
                self.state = 'loading'
                self.load_timer = 0

        elif self.state == 'loading':
            self.load_timer += dt
            if self.load_timer >= self.load_time:
                load_amount = min(self.capacity, coal_mine.coal_amount)
                self.cargo = load_amount
                coal_mine.coal_amount -= load_amount
                self.state = 'to_dump'
                # Remove from loading queue
                if self.id in loading_queue:
                    loading_queue.remove(self.id)

        elif self.state == 'unloading':
            self.load_timer += dt
            if self.load_timer >= self.load_time:
                coal_mine.dumped_coal += self.cargo
                self.cargo = 0
                self.trips_completed += 1
                if coal_mine.coal_amount > 0:
                    self.state = 'to_mine'
                else:
                    self.state = 'finished'

class CoalMine:
    def __init__(self, initial_coal):
        self.initial_coal = initial_coal
        self.coal_amount = initial_coal
        self.dumped_coal = 0

def run_simulation():
    # Get user input
    distance_km, coal_amount, truck_capacity, truck_speed, load_time, num_trucks = get_user_input()

    print(f"\n=== Simulation Starting ===")
    print(f"Distance: {distance_km} km")
    print(f"Travel time at {truck_speed} km/h: {(distance_km / truck_speed) * 60:.1f} minutes")
    print(f"Coal amount: {coal_amount} kg")
    print(f"Truck capacity: {truck_capacity} kg each")
    print(f"Number of trucks: {num_trucks}")
    print(f"Loading time: {load_time} seconds")
    print("Starting pygame window...")

    # Initialize screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Multi-Truck Coal Mine Simulation - {num_trucks} Trucks")
    clock = pygame.time.Clock()

    # Calculate positions and scaling
    PIXELS_PER_KM = min(400, (WIDTH - 300) / distance_km)
    dump_site_pos = (150, HEIGHT//2)
    coal_mine_pos = (150 + distance_km * PIXELS_PER_KM, HEIGHT//2)

    # Convert speed
    speed_pps = (truck_speed / 3600) * PIXELS_PER_KM

    # Create trucks with different colors
    truck_colors = [RED, BLUE, PURPLE, ORANGE, CYAN, GREEN, YELLOW, GRAY, BLACK, (255, 192, 203)]
    trucks = []
    for i in range(num_trucks):
        color = truck_colors[i % len(truck_colors)]
        truck = Truck(i, truck_capacity, speed_pps, load_time, dump_site_pos, coal_mine_pos, color)
        trucks.append(truck)

    # Initialize coal mine
    coal_mine = CoalMine(coal_amount)
    loading_queue = []  # Queue for trucks waiting to load

    # Font
    font = pygame.font.SysFont(None, 20)
    big_font = pygame.font.SysFont(None, 24)

    running = True
    last_time = time.time()
    simulation_time = 0

    print("Pygame window should now be visible!")

    while running:
        # Calculate delta time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        simulation_time += dt

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Update all trucks
        for truck in trucks:
            truck.update(dt, coal_mine, loading_queue, trucks)

        # Clear screen
        screen.fill(WHITE)

        # Draw roads (two parallel lines)
        upper_road_offset = 30
        lower_road_offset = 30
        # Upper road: dump site to coal mine (for going)
        pygame.draw.line(screen, GRAY, (dump_site_pos[0], dump_site_pos[1] - upper_road_offset), (coal_mine_pos[0], coal_mine_pos[1] - upper_road_offset), 8)
        # Lower road: coal mine to dump site (for returning)
        pygame.draw.line(screen, GRAY, (coal_mine_pos[0], coal_mine_pos[1] + lower_road_offset), (dump_site_pos[0], dump_site_pos[1] + lower_road_offset), 8)

        # Draw dump site
        pygame.draw.rect(screen, GREEN, (dump_site_pos[0]-35, dump_site_pos[1]-35, 70, 70))
        dump_text = big_font.render('DUMP SITE', True, BLACK)
        screen.blit(dump_text, (dump_site_pos[0]-45, dump_site_pos[1]-60))

        # Draw coal mine
        pygame.draw.rect(screen, BLACK, (coal_mine_pos[0]-35, coal_mine_pos[1]-35, 70, 70))
        mine_text = big_font.render('COAL MINE', True, WHITE)
        screen.blit(mine_text, (coal_mine_pos[0]-45, coal_mine_pos[1]-60))

        # Draw trucks
        for truck in trucks:
            # Set truck color: RED if empty, BLUE if loaded
            if truck.cargo == 0:
                truck_color = RED
            else:
                truck_color = BLUE
            pygame.draw.rect(screen, truck_color, (truck.position[0]-12, truck.position[1]-8, 24, 16))

            # Draw truck ID
            id_text = font.render(str(truck.id + 1), True, WHITE)
            screen.blit(id_text, (truck.position[0]-5, truck.position[1]-5))

            # Draw loading progress
            if truck.state == 'loading':
                progress = truck.load_timer / truck.load_time
                pygame.draw.rect(screen, GRAY, (truck.position[0]-15, truck.position[1]-20, 30, 6))
                pygame.draw.rect(screen, YELLOW, (truck.position[0]-15, truck.position[1]-20, 30*progress, 6))
            # Draw unloading progress (at dump site)
            if truck.state == 'unloading':
                progress = truck.load_timer / truck.load_time
                pygame.draw.rect(screen, GRAY, (truck.position[0]-15, truck.position[1]+20, 30, 6))
                pygame.draw.rect(screen, BLUE, (truck.position[0]-15, truck.position[1]+20, 30*progress, 6))

            # Draw waiting indicator
            if truck.state == 'waiting':
                pygame.draw.circle(screen, YELLOW, (int(truck.position[0]), int(truck.position[1]-15)), 5)

        # Display simulation information
        info_x = 10
        info_y = 10

        # General info
        general_info = [
            f"Simulation Time: {simulation_time:.1f}s",
            f"Coal Remaining: {coal_mine.coal_amount:.0f} kg",
            f"Coal Dumped: {coal_mine.dumped_coal:.0f} kg",
            f"Progress: {(coal_mine.dumped_coal / coal_mine.initial_coal * 100):.1f}%",
            f"Loading Queue: {len(loading_queue)} truck(s)",
            ""
        ]

        for line in general_info:
            if line:  # Skip empty lines
                text = font.render(line, True, BLACK)
                screen.blit(text, (info_x, info_y))
            info_y += 22

        # Truck status
        truck_header = big_font.render("TRUCK STATUS:", True, BLACK)
        screen.blit(truck_header, (info_x, info_y))
        info_y += 25

        total_trips = 0
        for truck in trucks:
            total_trips += truck.trips_completed
            truck_info = f"Truck {truck.id + 1}: {truck.state.replace('_', ' ').title()} | " \
                         f"Cargo: {truck.cargo:.0f} kg | Trips: {truck.trips_completed}"

            # Color code the text based on truck state
            text_color = BLACK
            if truck.state == 'loading':
                text_color = BLUE
            elif truck.state == 'waiting':
                text_color = ORANGE
            elif truck.cargo > 0:
                text_color = GREEN

            text = font.render(truck_info, True, text_color)
            screen.blit(text, (info_x, info_y))
            info_y += 20

        # Summary stats
        info_y += 10
        summary_text = font.render(f"Total Trips Completed: {total_trips}", True, BLACK)
        screen.blit(summary_text, (info_x, info_y))
        info_y += 20

        efficiency = (coal_mine.dumped_coal / (simulation_time * num_trucks)) if simulation_time > 0 else 0
        efficiency_text = font.render(f"Efficiency: {efficiency:.1f} kg/truck/second", True, BLACK)
        screen.blit(efficiency_text, (info_x, info_y))

        # Check if simulation is complete
        all_finished = all(truck.state == 'finished' for truck in trucks)
        if all_finished or coal_mine.coal_amount <= 0:
            completion_text = big_font.render("SIMULATION COMPLETE!", True, RED)
            screen.blit(completion_text, (WIDTH//2 - 100, HEIGHT - 80))
            time_text = font.render(f"Total Time: {simulation_time:.1f} seconds", True, RED)
            screen.blit(time_text, (WIDTH//2 - 80, HEIGHT - 60))

        # Instructions
        instruction = font.render("Press ESC to exit", True, BLACK)
        screen.blit(instruction, (WIDTH - 150, HEIGHT - 30))

        # Legend
        legend_y = HEIGHT - 150
        legend_text = font.render("Legend:", True, BLACK)
        screen.blit(legend_text, (WIDTH - 150, legend_y))
        legend_y += 20

        legend_items = [
            ("Empty truck", RED),
            ("Loaded truck", BLUE),
            ("Loading", YELLOW),
            ("Waiting", ORANGE)
        ]

        for item, color in legend_items:
            pygame.draw.rect(screen, color, (WIDTH - 150, legend_y, 15, 15))
            text = font.render(item, True, BLACK)
            screen.blit(text, (WIDTH - 130, legend_y))
            legend_y += 18

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_simulation()
