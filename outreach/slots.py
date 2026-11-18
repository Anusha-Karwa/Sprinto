"""Finding free 30-minute slots (pure code). Working hours, buffers, minimum notice, held + rejected slots excluded.
Three slots on DIFFERENT days, inside both parties' working hours where possible, near the prospect's preference."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from agent.config import outreach_cfg, settings


def _hm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def overlaps(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def is_free(start: datetime, busy: list[tuple[datetime, datetime]], minutes: int | None = None) -> bool:
    cfg = outreach_cfg()["sequence2"]
    minutes = minutes or cfg["meeting_minutes"]
    buf = timedelta(minutes=cfg["buffer_minutes"])
    end = start + timedelta(minutes=minutes)
    return not any(overlaps(start - buf, end + buf, b0, b1) for b0, b1 in busy)


def in_hours(start: datetime, tz: str, hours: dict, minutes: int) -> bool:
    z = ZoneInfo(tz)
    s, e = start.astimezone(z), (start + timedelta(minutes=minutes)).astimezone(z)
    return s.weekday() < 5 and s.time() >= _hm(hours["start"]) and e.time() <= _hm(hours["end"]) and s.date() == e.date()


def bookable(start: datetime, now: datetime) -> tuple[bool, str]:
    cfg = outreach_cfg()["sequence2"]
    if start < now + timedelta(hours=cfg["min_notice_hours"]):
        return False, f"less than {cfg['min_notice_hours']}h notice"
    if start > now + timedelta(days=cfg["max_days_ahead"]):
        return False, f"more than {cfg['max_days_ahead']} days out"
    if not in_hours(start, settings().growth_lead_tz, cfg["working_hours"], cfg["meeting_minutes"]):
        return False, "outside the Growth Lead's working hours"
    return True, ""


def find_slots(now: datetime, busy, *, exclude: set[str], prospect_tz: str | None = None, near: datetime | None = None,
               preferred_dates: list[date] | None = None, n: int | None = None, days: int = 14) -> list[datetime]:
    cfg = outreach_cfg()["sequence2"]
    n = n or cfg["slots_to_propose"]
    lead_tz = ZoneInfo(settings().growth_lead_tz)
    minutes = cfg["meeting_minutes"]
    cands: list[datetime] = []
    d0 = now.astimezone(lead_tz).date()
    for i in range(days + 1):
        day = d0 + timedelta(days=i)
        t = datetime.combine(day, _hm(cfg["working_hours"]["start"]), lead_tz)
        end = datetime.combine(day, _hm(cfg["working_hours"]["end"]), lead_tz)
        while t + timedelta(minutes=minutes) <= end:
            s = t.astimezone(timezone.utc)
            ok, _ = bookable(s, now)
            if ok and s.isoformat() not in exclude and is_free(s, busy):
                cands.append(s)
            t += timedelta(minutes=30)

    def score(s: datetime):
        both = 0 if (not prospect_tz or in_hours(s, prospect_tz, cfg["prospect_working_hours"], minutes)) else 1
        pref = 0 if (not preferred_dates or s.astimezone(lead_tz).date() in preferred_dates) else 1
        dist = abs((s - near).total_seconds()) if near else (s - now).total_seconds()
        return (pref, both, dist)

    picked, days_used = [], set()
    for s in sorted(cands, key=score):
        d = s.astimezone(lead_tz).date()
        if d in days_used:
            continue
        picked.append(s)
        days_used.add(d)
        if len(picked) == n:
            break
    return sorted(picked)


def fmt(start: datetime, tz: str) -> str:
    z = ZoneInfo(tz)
    s = start.astimezone(z)
    e = (start + timedelta(minutes=outreach_cfg()["sequence2"]["meeting_minutes"])).astimezone(z)
    abbr = s.tzname() if tz != "Asia/Kolkata" else "IST"
    return f"{s:%a %d %b}, {s:%I:%M %p}".replace(" 0", " ") + f"–{e:%I:%M %p}".replace("–0", "–") + f" {abbr}"
