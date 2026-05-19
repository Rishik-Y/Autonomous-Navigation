# Tracks if user saved at least once during launcher session
_save_occurred = False

def mark_save_occurred():
    global _save_occurred
    _save_occurred = True

def has_save_occurred():
    return _save_occurred

def reset_save_tracker():
    global _save_occurred
    _save_occurred = False
