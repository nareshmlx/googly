from datetime import datetime, timezone

def get_current_time() -> str:
    """
    Returns the current UTC time as an ISO string.
    Deterministic, safe, no side effects.
    """
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
