"""Structured tourist intent. Keyword rules are authoritative for routing flags."""

from __future__ import annotations

import re
from datetime import datetime

from src.agent.state import IntentFlags

ALT_PATTERNS = (
    "instead",
    "what else",
    "alternative",
    "similar",
    "hidden gem",
    "less crowded",
    "lesser-known",
    "lesser known",
    "quieter",
    "where else",
    "other places",
    "what should i visit",
    "what should i see",
)
PRED_PATTERNS = (
    "crowd",
    "crowded",
    "busy",
    "busier",
    "forecast",
    "visitor",
    "demand",
    "how packed",
    "how many",
)
WEB_PATTERNS = (
    "this weekend",
    "weather",
    "event",
    "festival",
    "closure",
    "closed",
    "happening",
    "advisory",
    "current",
    "holiday",
)
TRIP_PATTERNS = (
    "visiting from",
    "i am visiting",
    "i'm visiting",
    "itinerary",
    "new to",
    "i like",
    "want to see",
    "i want to see",
)
TIME_PATTERNS = (
    "best time",
    "when should",
    "which month",
    "which season",
    "visiting period",
)

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

REGION_PLACES = [
    ("taj mahal", "Taj Mahal"),
    ("qutub minar", "Qutub Minar"),
    ("qutb minar", "Qutub Minar"),
    ("kutub minar", "Qutub Minar"),
    ("kutb minar", "Qutub Minar"),
    ("pratapgad", "Pratapgad"),
    ("pratap gad", "Pratapgad"),
    ("mumbai", "Mumbai"),
    ("bombay", "Mumbai"),
    ("kolhapur", "Kolhapur"),
    ("satara", "Satara"),
    ("pune", "Pune"),
    ("sangli", "Sangli"),
    ("belgaon", "Belgaon"),
    ("belgaum", "Belgaum"),
    ("belagavi", "Belagavi"),
    ("sindhudurg", "Sindhudurg"),
    ("delhi", "Delhi"),
    ("chennai", "Chennai"),
    ("bengaluru", "Bengaluru"),
    ("bangalore", "Bengaluru"),
    ("hampi", "Hampi"),
    ("sanchi", "Sanchi"),
    ("udupi", "Udupi"),
    ("goa", "Goa"),
    ("jaipur", "Jaipur"),
    ("varanasi", "Varanasi"),
]


def _has(text: str, patterns: tuple[str, ...]) -> bool:
    return any(p in text for p in patterns)


def _mon(token: str) -> int:
    t = token.lower()[:4]
    if t.startswith("sept"):
        return 9
    return MONTHS[token.lower()[:3]]


def _year(raw: str | None, default: str = "2026") -> int:
    if not raw:
        return int(default)
    y = raw.strip()
    if len(y) == 2:
        y = "20" + y
    return int(y)


def _dates(text: str) -> tuple[str | None, str | None]:
    iso = re.findall(r"20\d{2}-\d{2}-\d{2}", text)
    if len(iso) >= 2:
        return iso[0], iso[1]
    if len(iso) == 1:
        return iso[0], None

    range1 = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+"
        r"(\d{1,2})\s*(?:-|to|–)\s*(\d{1,2})"
        r"(?:,?\s*((?:20)?\d{2}))?",
        text,
        re.I,
    )
    if range1:
        y = _year(range1.group(4))
        mon = _mon(range1.group(1))
        try:
            start = datetime(y, mon, int(range1.group(2))).date().isoformat()
            end = datetime(y, mon, int(range1.group(3))).date().isoformat()
            return start, end
        except ValueError:
            pass

    range_mdy = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+"
        r"(\d{1,2})\s*(?:-|to|–)\s*"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+"
        r"(\d{1,2})(?:,?\s*((?:20)?\d{2}))?",
        text,
        re.I,
    )
    if range_mdy:
        y = _year(range_mdy.group(5))
        try:
            start = datetime(y, _mon(range_mdy.group(1)), int(range_mdy.group(2))).date().isoformat()
            end = datetime(y, _mon(range_mdy.group(3)), int(range_mdy.group(4))).date().isoformat()
            return start, end
        except ValueError:
            pass

    range_dmy = re.search(
        r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s*(?:-|to)\s*(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
        r"(?:\s+((?:20)?\d{2}))?",
        text,
        re.I,
    )
    if range_dmy:
        y = _year(range_dmy.group(4))
        mon = _mon(range_dmy.group(3))
        try:
            start = datetime(y, mon, int(range_dmy.group(1))).date()
            end = datetime(y, mon, int(range_dmy.group(2))).date()
            if end < start:
                raise ValueError("inverted range")
            return start.isoformat(), end.isoformat()
        except ValueError:
            pass

    range2 = re.search(
        r"(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
        r"(?:\s+((?:20)?\d{2}))?\s*(?:-|to|–)\s*"
        r"(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
        r"(?:\s+((?:20)?\d{2}))?",
        text,
        re.I,
    )
    if range2:
        y = _year(range2.group(6) or range2.group(3))
        try:
            start = datetime(y, _mon(range2.group(2)), int(range2.group(1))).date().isoformat()
            end = datetime(y, _mon(range2.group(5)), int(range2.group(4))).date().isoformat()
            return start, end
        except ValueError:
            pass

    single = re.search(
        r"(?:on\s+)?(?:(\d{1,2})\s+)?"
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*((?:20)?\d{2}))?",
        text,
        re.I,
    )
    if single and single.group(3):
        y = _year(single.group(4))
        mon = _mon(single.group(2))
        day = int(single.group(3) if single.group(1) is None else single.group(3))
        try:
            return datetime(y, mon, day).date().isoformat(), None
        except ValueError:
            pass

    dmy = re.search(
        r"(?:on\s+)?(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
        r"(?:\s+((?:20)?\d{2}))?",
        text,
        re.I,
    )
    if dmy:
        y = _year(dmy.group(3))
        try:
            return datetime(y, _mon(dmy.group(2)), int(dmy.group(1))).date().isoformat(), None
        except ValueError:
            pass
    return None, None


def _destination(raw: str, lower: str) -> str | None:
    skip = {"the", "a", "an"}
    skip_dest = skip | {
        "instead",
        "should",
        "visit",
        "what",
        "this",
        "that",
        "there",
        "crowded",
        "busy",
    }
    for needle, label in REGION_PLACES:
        if needle in lower:
            return label
    for name in ("Mahalakshmi", "Jyotiba", "Panhala"):
        if name.lower() in lower:
            return name
    who = re.match(r"who is\s+([^?]+)", lower)
    if who:
        return who.group(1).strip().title()
    new_to = re.search(r"\bnew to\s+([a-z][a-z\s]{1,30}?)(?:\.|,|$)", lower)
    if new_to:
        token = new_to.group(1).strip()
        if token and token.split()[0] not in skip_dest:
            return " ".join(w.capitalize() for w in token.split())
    visiting = re.search(
        r"\b(?:i(?:'m| am)?\s+)?(?:plan(?:ning)?\s+(?:to\s+visit|a\s+trip\s+to|to\s+travel\s+to|to\s+go\s+to|to\s+head\s+to|trip\s+to)|visit(?:ing)?|travel(?:ing)?\s+to|go(?:ing)?\s+to|head(?:ing)?\s+to)\s+"
        r"(?:to\s+)?([a-z][a-z\s]{1,50}?)(?:\s+(?:with|from|on|during|in|next|for)\b|[,.?!]|$)",
        lower,
    )
    if visiting:
        token = visiting.group(1).strip()
        if token and token.split()[0] not in skip_dest:
            return " ".join(w.capitalize() for w in token.split() if w not in skip)
    crowd_q = re.search(
        r"\b(?:how\s+(?:crowded|busy)\s+is|will|is)\s+([a-z][a-z\s]{1,40}?)\s+(?:be\s+crowded|be\s+busy|expected|crowded|busy|\?|$)",
        lower,
    )
    if crowd_q:
        token = crowd_q.group(1).strip()
        if token and token.split()[0] not in skip_dest:
            return " ".join(w.capitalize() for w in token.split() if w not in skip)
    what_about = re.search(
        r"\b(?:what|how)\s+about\s+([a-z][a-z\s]{1,40}?)(?:\s+next|\s+with|\s+in|\?|$)",
        lower,
    )
    if what_about:
        token = what_about.group(1).strip()
        if token and token.split()[0] not in skip_dest:
            return " ".join(w.capitalize() for w in token.split() if w not in skip)
    fort = re.search(r"\b([a-z][a-z]+(?:gad|gadh|pur)?)\s+fort\b", lower)
    if fort:
        return fort.group(1).title()
    m = re.search(
        r"(?:city of|at the|\bat\b|\bin\b|\baround\b|\bnear\b)\s+"
        r"([a-z][a-z\s]{1,40}?)(?:\s+expected|\s+on\s|\s+during|\?|$)",
        lower,
    )
    if m:
        token = m.group(1).strip()
        first = token.split()[0] if token else ""
        if first not in skip_dest and "instead" not in token:
            return " ".join(w.capitalize() for w in token.split() if w not in skip)
    return None


def extract_intent(text: str) -> dict:
    raw = text.strip()
    lower = raw.lower().replace("–", "-").replace("—", "-")
    start, end = _dates(lower)

    needs_prediction = _has(lower, PRED_PATTERNS)
    needs_alternatives = _has(lower, ALT_PATTERNS)
    needs_current_context = _has(lower, WEB_PATTERNS) or bool(start)
    needs_time_recommendation = _has(lower, TIME_PATTERNS)
    wants_hourly = bool(re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", lower)) or "hourly" in lower
    trip_language = _has(lower, TRIP_PATTERNS)
    needs_trip_plan = trip_language and bool(start)

    if lower.startswith("who is"):
        needs_prediction = False
        needs_alternatives = False
        needs_trip_plan = False
        needs_current_context = True

    dest = _destination(raw, lower)
    must = [n for n in ["Mahalakshmi", "Jyotiba", "Panhala"] if n.lower() in lower]
    interests = [w for w in ["temples", "forts", "nature", "heritage", "spiritual", "beaches"] if w in lower]

    party = None
    if any(w in lower for w in ("family", "parents", "kids", "children")):
        party = "family"
    elif any(w in lower for w in ("friends", "buddies", "group")):
        party = "friends"
    elif any(w in lower for w in ("solo", "alone", "myself")):
        party = "solo"
    elif any(w in lower for w in ("couple", "partner", "wife", "husband")):
        party = "couple"

    if needs_trip_plan:
        needs_alternatives = True
        needs_current_context = True

    requested_n = 3
    n_m = re.search(r"\b(\d+)\s+(quieter|similar|alternative)", lower)
    if n_m:
        requested_n = max(1, min(8, int(n_m.group(1))))

    flags: IntentFlags = {
        "needs_prediction": needs_prediction,
        "needs_alternatives": needs_alternatives,
        "needs_current_context": needs_current_context,
        "needs_time_recommendation": needs_time_recommendation,
        "needs_trip_plan": needs_trip_plan,
        "wants_hourly": wants_hourly,
        "fallback_discovery": False,
        "requested_n": requested_n,
    }
    return {
        "destination": dest,
        "party_type": party,
        "start_date": start,
        "end_date": end,
        "interests": interests,
        "must_visit": must,
        "intent": flags,
        "is_multi_day": bool(start and end),
        "wants_hourly": wants_hourly,
        "needs_web": needs_current_context,
        "wants_alternatives": needs_alternatives,
        "requested_n": requested_n,
    }


def _slot_bool(slots: dict, key: str) -> bool:
    val = slots.get(key)
    return bool(val) if val is not None else False


def resolve_intent(text: str, use_llm: bool = True) -> dict:
    """Keyword flags plus place/party slots."""
    return extract_intent(text)
