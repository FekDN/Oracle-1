"""Platform-independent timestamp conversion helpers."""
from __future__ import annotations

from ._shared import *

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _ts_to_year(ts: float) -> int:
    try:
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (epoch + timedelta(seconds=float(ts))).year
    except Exception:
        return 1970 + math.floor(float(ts) / 31556952.0)


def _year_to_ts(year: int) -> float:
    """Inverse of _ts_to_year: Jan 1 UTC of the given year as a unix
    timestamp. Used to turn a TemporalZone's LLM-supplied start_year/
    end_year into the start_timestamp/end_timestamp the geometric field
    test (TemporalZone.covers) actually compares against."""
    start_of_year = datetime(int(year), 1, 1, tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (start_of_year - epoch).total_seconds()

__all__ = ['_ts_to_year', '_year_to_ts']
