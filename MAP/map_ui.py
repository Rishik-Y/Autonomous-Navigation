import pygame


def confirm_save_dialog(screen, font, mode_label: str, timeout_seconds: int = 120):
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))

    dialog_width, dialog_height = 520, 200
    dialog_x = (screen.get_width() - dialog_width) // 2
    dialog_y = (screen.get_height() - dialog_height) // 2
    dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)

    title_text = f"Unsaved changes in {mode_label}"
    message_text = "Save before switching modes?"
    options_text = "Y: Save | N: Don't Save | Esc: Cancel"

    clock = pygame.time.Clock()
    start_ticks = pygame.time.get_ticks()
    while True:
        if timeout_seconds is not None:
            elapsed = (pygame.time.get_ticks() - start_ticks) / 1000.0
            if elapsed >= timeout_seconds:
                return "cancel"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    return "save"
                if event.key == pygame.K_n:
                    return "discard"
                if event.key == pygame.K_ESCAPE:
                    return "cancel"

        screen.blit(overlay, (0, 0))
        pygame.draw.rect(screen, (255, 255, 255), dialog_rect)
        pygame.draw.rect(screen, (0, 0, 0), dialog_rect, 3)

        title_surface = font.render(title_text, True, (0, 0, 0))
        message_surface = font.render(message_text, True, (0, 0, 0))
        options_surface = font.render(options_text, True, (0, 0, 0))

        screen.blit(title_surface, (dialog_x + 20, dialog_y + 30))
        screen.blit(message_surface, (dialog_x + 20, dialog_y + 80))
        screen.blit(options_surface, (dialog_x + 20, dialog_y + 130))

        pygame.display.flip()
        clock.tick(30)
