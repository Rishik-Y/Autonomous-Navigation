import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SIMULATION_DIR = os.path.join(REPO_ROOT, "Simulation")
MAP_DIR = os.path.join(REPO_ROOT, "MAP")


def run_simulation():
    subprocess.run([sys.executable, "main.py"], cwd=SIMULATION_DIR)


def run_map():
    subprocess.run([sys.executable, "unified_map_launcher.py"], cwd=MAP_DIR)


def main():
    while True:
        print("\nAutonomous Navigation Launcher")
        print("1) Simulation")
        print("2) Map Tools")
        print("Q) Quit")
        print("Tip: Press ESC inside Simulation or Map tools to return here.")
        choice = input("Select an option: ").strip().lower()

        if choice in {"1", "simulation", "sim", "s"}:
            run_simulation()
        elif choice in {"2", "map", "m"}:
            run_map()
        elif choice in {"q", "quit", "exit"}:
            break
        else:
            print("Invalid selection. Try again.")


if __name__ == "__main__":
    main()
