# Tracks if user saved at least once during launcher session
_save_occurred = False


def mark_save_occurred():
    """Mark that at least one save operation completed in this session."""
    global _save_occurred
    _save_occurred = True


def has_save_occurred():
    """Return whether a save operation has occurred in the current session."""
    return _save_occurred


def reset_save_tracker():
    """Reset save tracking for a new launcher session."""
    global _save_occurred
    _save_occurred = False
