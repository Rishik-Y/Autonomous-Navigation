import pygame
import sys
import time
import math
import threading

# Physics parameters
MAX_SPEED = 30.0  # m/s
ACCELERATION = 5.0  # m/s^2
FRICTION = 0.2  # m/s^2
FUEL_CONSUMPTION_RATE = 0.05  # liters per meter
INITIAL_FUEL = 100.0  # liters
TRUCK_LENGTH = 80
TRUCK_WIDTH = 40
ROAD_WIDTH = 200
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400

class Truck:
    def __init__(self):
        self.x = WINDOW_WIDTH // 2 - TRUCK_LENGTH // 2
        self.y = WINDOW_HEIGHT // 2 - TRUCK_WIDTH // 2
        self.velocity = 0.0
        self.acceleration = 0.0
        self.distance = 0.0
        self.fuel = INITIAL_FUEL
        self.last_update = time.time()

    def update(self, dt, keys):
        # Acceleration control
        if keys[pygame.K_w]:
            self.acceleration = ACCELERATION
        elif keys[pygame.K_s]:
            self.acceleration = -ACCELERATION
        else:
            self.acceleration = 0

        # Friction
        if self.velocity > 0:
            self.velocity -= FRICTION * dt
            if self.velocity < 0:
                self.velocity = 0
        elif self.velocity < 0:
            self.velocity += FRICTION * dt
            if self.velocity > 0:
                self.velocity = 0

        # Update velocity
        self.velocity += self.acceleration * dt
        self.velocity = max(-MAX_SPEED, min(MAX_SPEED, self.velocity))

        # Update position
        self.x += self.velocity * dt
        self.x = max((WINDOW_WIDTH - ROAD_WIDTH) // 2, min(self.x, (WINDOW_WIDTH + ROAD_WIDTH) // 2 - TRUCK_LENGTH))
        self.distance += abs(self.velocity * dt)

        # Fuel consumption
        self.fuel -= abs(self.velocity * dt) * FUEL_CONSUMPTION_RATE
        if self.fuel < 0:
            self.fuel = 0
            self.velocity = 0

    def get_stats(self):
        return {
            'Velocity': self.velocity,
            'Acceleration': self.acceleration,
            'Distance': self.distance,
            'Fuel': self.fuel
        }

def print_stats(truck):
    while True:
        stats = truck.get_stats()
        sys.stdout.write(f"\rVelocity: {stats['Velocity']:.2f} m/s | Acceleration: {stats['Acceleration']:.2f} m/s^2 | Distance: {stats['Distance']:.2f} m | Fuel: {stats['Fuel']:.2f} L   ")
        sys.stdout.flush()
        time.sleep(0.1)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('2D Truck Physics Simulation')
    clock = pygame.time.Clock()
    truck = Truck()

    # Start stats printing thread
    stats_thread = threading.Thread(target=print_stats, args=(truck,), daemon=True)
    stats_thread.start()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        truck.update(dt, keys)

        # Drawing
        screen.fill((50, 50, 50))
        # Draw road
        pygame.draw.rect(screen, (100, 100, 100), ((WINDOW_WIDTH - ROAD_WIDTH) // 2, 0, ROAD_WIDTH, WINDOW_HEIGHT))
        # Draw truck
        pygame.draw.rect(screen, (0, 200, 0), (truck.x, truck.y, TRUCK_LENGTH, TRUCK_WIDTH))
        pygame.display.flip()

    pygame.quit()
    print("\nSimulation ended.")

if __name__ == "__main__":
    main()

