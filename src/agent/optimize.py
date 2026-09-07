"""Deterministic multi-day sketch from tool outputs. Not an LLM itinerary."""

from __future__ import annotations

from datetime import date, datetime, timedelta


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def daterange(start: date, end: date) -> list[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def optimize_trip(
    start_date: str | None,
    end_date: str | None,
    must_visit: list[str] | None,
    ranked_alternatives: list[dict] | None,
    crowd_levels: dict | None,
) -> list[dict]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if not start or not end or end < start:
        return []
    days = daterange(start, end)
    anchors = list(must_visit or [])
    orbits = [r["name"] for r in (ranked_alternatives or []) if r.get("name") not in anchors]
    plan = []
    for i, d in enumerate(days):
        if i < len(anchors):
            name = anchors[i]
            kind = "anchor"
            crowd = (crowd_levels or {}).get(name, {})
            level = crowd.get("level") if isinstance(crowd, dict) else None
            reason = (
                f"Must-visit kept in the itinerary. "
                f"Relative demand {level or 'unknown'} if ASI history exists; "
                "not an hourly prediction."
            )
        elif orbits:
            name = orbits[(i - len(anchors)) % len(orbits)]
            kind = "orbit"
            reason = (
                "Lower-pressure / discovery slot using a ranked relevant alternative "
                "(orbit), not a random least-visited place."
            )
        else:
            name = anchors[0] if anchors else "Free exploration"
            kind = "flexible"
            reason = "Not enough ranked orbits; keep time flexible rather than inventing sites."
        plan.append(
            {
                "date": d.isoformat(),
                "destination": name,
                "type": kind,
                "reason": reason,
            }
        )
    return plan
