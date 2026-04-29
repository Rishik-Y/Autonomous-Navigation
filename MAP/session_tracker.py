# Tracks if user saved at least once during launcher session
_save_occurred = False

def mark_save_occurred():
    """Mark that a save operation completed successfully"""
    global _save_occurred
    _save_occurred = True

def has_save_occurred():
    """Check if save occurred"""
    return _save_occurred

def reset_save_tracker():
    """Reset for new launcher session"""
    global _save_occurred
    _save_occurred = False
