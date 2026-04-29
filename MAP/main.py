"""
Unified Single-Window MAP Editor Launcher
Creates one persistent window and cycles through all modes without flicker
"""
import pygame
import sys
import Map_Editor
import Coal_Mine_Editor
import Waypoint_Editor
import map_viewer
import waypoint_viewer
import map_storage
import session_tracker
import map_ui

MODES = [
    ("Map Editor", Map_Editor.run_editor),
    ("Coal Mine Editor", Coal_Mine_Editor.run_editor),
    ("Waypoint Editor", Waypoint_Editor.run_waypoint_editor),
    ("Map Viewer", map_viewer.run_viewer),
    ("Waypoint Viewer", waypoint_viewer.run_viewer),
]

class SingleWindowLauncher:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 900), pygame.RESIZABLE)
        pygame.display.set_caption("Unified MAP Editor")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)
        
        # Initialize mode state tracking
        self.mode_states = {
            'is_dirty': False,
            '_saved_files': []
        }
        self.current_mode_index = 0
        
        # Shared view state (zoom/pan) across all modes
        self.view_state = {
            'scale': 1.0,
            'pan': [50, 50]
        }
        
        self.running = True
        self.exit_requested = False
    
    def run(self):
        session_tracker.reset_save_tracker()
        files_saved = []
        
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            
            # Handle QUIT and ESC events first
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
                    self.exit_requested = True
                    break
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.exit_requested = True
                        break
            
            if not self.running:
                break
            
            # Process event with current mode using shared screen
            mode_label, mode_func = MODES[self.current_mode_index]
            
            result = mode_func(
                mode_label=mode_label,
                allow_tab_switch=True,
                mode_index=self.current_mode_index + 1,
                total_modes=len(MODES),
                _shared_screen=self.screen,
                _shared_font=self.font,
                _view_state=self.view_state
            )
            
            # Track saved files from this mode
            if mode_label == "Map Editor":
                files_saved.extend(getattr(Map_Editor, '_saved_files', []))
            elif mode_label == "Coal Mine Editor":
                files_saved.extend(getattr(Coal_Mine_Editor, '_saved_files', []))
            elif mode_label == "Waypoint Editor":
                files_saved.extend(getattr(Waypoint_Editor, '_saved_files', []))
            
            # Handle results
            if result == "next":
                self.current_mode_index = (self.current_mode_index + 1) % len(MODES)
            elif result == "prev":
                self.current_mode_index = (self.current_mode_index - 1) % len(MODES)
            elif result == "quit":
                self.exit_requested = True
                break
            # If result is None or mode wants to quit, exit launcher
            elif result is None:
                self.running = False
            
            # Draw mode overlay
            mode_label, _ = MODES[self.current_mode_index]
            map_ui.draw_mode_overlay(
                self.screen, self.font,
                mode_label,
                self.current_mode_index + 1,
                len(MODES),
                False  # Don't show dirty state in single-window mode
            )
            
            pygame.display.flip()
        
        pygame.quit()
        
        # Create snapshot if files were saved
        if files_saved:
            snapshot_path = map_storage.create_snapshot()
            unique_files = list(set(files_saved))
            print(f"✓ Snapshot saved to: {snapshot_path}")
            print(f"  Files saved: {', '.join(unique_files)}")
        else:
            print("No changes were saved. No snapshot created.")

def main():
    launcher = SingleWindowLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
