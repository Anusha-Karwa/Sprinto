"""Clock + business-day helpers. The clock is injectable so every branch can be tested with a fake time."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from agent.config import settings


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def add_business_days(start: datetime, days: int) -> datetime:
    """Counted in the Growth Lead's time zone (Sat/Sun skipped)."""
    tz = ZoneInfo(settings().growth_lead_tz)
    d = start.astimezone(tz)
    added = 0
    while added < days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d.astimezone(timezone.utc)
