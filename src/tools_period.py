"""Travel-window pressure. Does not invent a new visitor headcount."""

from __future__ import annotations

from datetime import date, datetime, timedelta

# Gazetted-style dates we treat as India-wide busy (heuristic, not a visitor model).
INDIA_BUSY_DATES = {
    date(2025, 1, 1),
    date(2025, 1, 26),
    date(2025, 8, 15),
    date(2025, 10, 2),
    date(2025, 12, 25),
    date(2026, 1, 1),
    date(2026, 1, 26),
    date(2026, 8, 15),
    date(2026, 10, 2),
    date(2026, 12, 25),
    date(2022, 1, 26),
    date(2022, 8, 15),
    date(2022, 10, 2),
    date(2022, 12, 25),
}

PEAK_WINTER_MONTHS = {10, 11, 12, 1, 2, 3}
MONSOON_MONTHS = {6, 7, 8, 9}


def _parse(iso: str | None) -> date | None:
    if not iso:
        return None
    try:
        return datetime.strptime(iso[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def iter_days(start: date, end: date) -> list[date]:
    if end < start:
        start, end = end, start
    out = []
    cur = start
    while cur <= end and len(out) < 31:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def season_for_months(
    months: list[int],
    dest_state: str | None = None,
    climate_zone: str | None = None,
    destination: str | None = None,
) -> str:
    if not months:
        return "unknown"
    monsoon = sum(1 for m in months if m in MONSOON_MONTHS)
    winter = sum(1 for m in months if m in PEAK_WINTER_MONTHS)
    if monsoon >= winter and monsoon > 0:
        return "monsoon"
    if winter <= 0:
        return "shoulder"
    blob = f"{destination or ''} {dest_state or ''} {climate_zone or ''}".lower()
    south = any(
        x in blob
        for x in (
            "karnataka",
            "maharashtra",
            "tamil",
            "kerala",
            "goa",
            "andhra",
            "telangana",
            "peninsular",
            "coastal",
            "belagavi",
            "belgaum",
            "belgaon",
            "hampi",
            "mumbai",
            "pune",
            "kolhapur",
            "sangli",
            "chennai",
            "bengaluru",
            "bangalore",
        )
    )
    north = any(
        x in blob
        for x in (
            "north_india",
            "delhi",
            "uttar pradesh",
            "rajasthan",
            "punjab",
            "haryana",
            "taj mahal",
            "agra",
            "qutub",
        )
    )
    if climate_zone in {"peninsular", "coastal"} or (south and not north):
        return "dry_cool_season"
    if climate_zone == "north_india" or north:
        return "peak_winter_tourism"
    return "winter"


def assess_period_pressure(
    destination: str | None,
    start_date: str | None,
    end_date: str | None,
    crowd_level: str | None = None,
    web_context: dict | None = None,
    dest_state: str | None = None,
    climate_zone: str | None = None,
) -> dict:
    """Direction of pressure vs the annual ASI figure. Never a new count."""
    start = _parse(start_date)
    end = _parse(end_date) or start
    web = web_context or {}
    if not start:
        return {
            "found": False,
            "footfall_direction": "unknown",
            "layer": "HEURISTIC",
            "note": "No travel dates, so no weekend/holiday/season window.",
            "does_not_change_annual_count": True,
        }

    days = iter_days(start, end)
    weekend_days = [d for d in days if d.weekday() >= 5]
    holiday_days = [d for d in days if d in INDIA_BUSY_DATES]
    months = [d.month for d in days]
    season = season_for_months(months, dest_state, climate_zone, destination)

    weather_hits = list(web.get("weather") or [])
    event_hits = list(web.get("events") or []) + list(web.get("holidays") or [])

    flags: list[str] = []
    if weekend_days:
        flags.append(f"{len(weekend_days)} weekend day(s) in a {len(days)}-day window")
    if holiday_days:
        flags.append("public-holiday date(s) in window: " + ", ".join(d.isoformat() for d in holiday_days))
    if season == "peak_winter_tourism":
        flags.append("window overlaps typical north-India heritage peak months (heuristic)")
    if season == "dry_cool_season":
        flags.append("window is the cooler dry season in peninsular India (heuristic)")
    if season == "winter":
        flags.append("window overlaps winter months (heuristic; region not treated as north-India peak)")
    if season == "monsoon":
        flags.append("window overlaps monsoon months (heuristic; outdoor sites often slower except festivals)")
    if event_hits:
        flags.append("web mentioned events/holidays (WEB FACT, not a visitor count)")
    if weather_hits:
        flags.append("web mentioned weather (WEB FACT, not a visitor count)")

    up = 0
    down = 0
    if len(weekend_days) >= max(1, len(days) // 3):
        up += 1
    if holiday_days:
        up += 1
    if season == "peak_winter_tourism":
        up += 1
    if season == "monsoon" and not holiday_days and not event_hits:
        down += 1
    if event_hits:
        up += 1
    if crowd_level in {"HIGH", "VERY HIGH"}:
        up += 1

    if up >= 2 and up > down:
        direction = "likely_higher"
    elif down >= 1 and down > up:
        direction = "likely_lower"
    elif up == 1 and down == 0:
        direction = "likely_higher"
    else:
        direction = "similar"

    return {
        "found": True,
        "destination": destination,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "n_days": len(days),
        "weekend_days": len(weekend_days),
        "weekday_days": len(days) - len(weekend_days),
        "holiday_dates": [d.isoformat() for d in holiday_days],
        "season": season,
        "crowd_level_annual": crowd_level,
        "footfall_direction": direction,
        "reasons": flags,
        "layer": "HEURISTIC",
        "web_used": bool(event_hits or weather_hits),
        "does_not_change_annual_count": True,
        "note": (
            "This is pressure on the travel window vs the annual ASI persistence figure. "
            "It is not a new daily/hourly visitor total."
        ),
    }
