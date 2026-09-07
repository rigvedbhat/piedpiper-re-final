"""Geographic practicality for nearby (orbit) recommendations.

City/town coordinates are a small gazetteer of published centroids.
Monument-level GPS is not invented: unknown places get distance_km=None.
"""

from __future__ import annotations

import math

# Published city/town centroids (decimal degrees). Not monument GPS.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "kolhapur": (16.7050, 74.2433),
    "panhala": (16.8120, 74.1240),
    "satara": (17.6805, 74.0183),
    "solapur": (17.6599, 75.9064),
    "sholapur": (17.6599, 75.9064),
    "sangli": (16.8524, 74.5815),
    "miraj": (16.8302, 74.6470),
    "kopeshwar": (16.6920, 74.6860),
    "khidrapur": (16.6920, 74.6860),
    "sagareshwar": (17.1660, 74.3830),
    "ratnagiri": (16.9902, 73.3120),
    "sindhudurg": (16.1700, 73.7000),
    "malvan": (16.0598, 73.4700),
    "vengurla": (15.8610, 73.6320),
    "sawantwadi": (15.9050, 73.8210),
    "belgaum": (15.8497, 74.4977),
    "belagavi": (15.8497, 74.4977),
    "belgaon": (15.8497, 74.4977),
    "pune": (18.5204, 73.8567),
    "mahabaleshwar": (17.9300, 73.6470),
    "lonavala": (18.7481, 73.4072),
    "mumbai": (19.0760, 72.8777),
    "alibaug": (18.6411, 72.8722),
    "aurangabad": (19.8762, 75.3433),
    "chhatrapati sambhajinagar": (19.8762, 75.3433),
    "ajanta": (20.5519, 75.7033),
    "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090),
    "agra": (27.1767, 78.0081),
    "mathura": (27.4924, 77.6737),
    "fatehpur sikri": (27.0940, 77.6680),
    "hampi": (15.3350, 76.4600),
    "hospet": (15.2695, 76.3871),
    "hosapete": (15.2695, 76.3871),
    "bellary": (15.1394, 76.9214),
    "ballari": (15.1394, 76.9214),
    "sanchi": (23.4793, 77.7398),
    "chennai": (13.0827, 80.2707),
    "mamallapuram": (12.6208, 80.1920),
    "mahabalipuram": (12.6208, 80.1920),
    "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "mysore": (12.2958, 76.6394),
    "mysuru": (12.2958, 76.6394),
    "qutub minar": (28.5245, 77.1855),
    "taj mahal": (27.1751, 78.0421),
}

# Ordinary alternative radius (day-trip). Same-state is not "nearby".
MAX_NEARBY_KM = 160.0
MAX_RELAXED_KM = 220.0

NEARBY_CITIES: dict[str, tuple[str, ...]] = {
    "kolhapur": (
        "kolhapur",
        "panhala",
        "satara",
        "sangli",
        "ratnagiri",
        "sindhudurg",
        "belgaum",
        "belagavi",
    ),
    "satara": ("satara", "pune", "mahabaleshwar", "kolhapur", "lonavala"),
    "pune": ("pune", "lonavala", "mahabaleshwar", "satara", "mumbai"),
    "sindhudurg": (
        "sindhudurg",
        "ratnagiri",
        "malvan",
        "vengurla",
        "sawantwadi",
        "kolhapur",
    ),
    "hampi": ("hampi", "hospet", "hosapete", "bellary", "ballari"),
    "agra": ("agra", "mathura", "fatehpur sikri", "sikandra"),
    "taj mahal": ("agra", "mathura", "fatehpur sikri", "sikandra"),
    "delhi": ("delhi", "new delhi", "noida", "gurgaon", "gurugram", "faridabad"),
    "qutub minar": ("delhi", "new delhi"),
    "chennai": ("chennai", "mamallapuram", "mahabalipuram", "kanchipuram"),
    "bengaluru": ("bengaluru", "bangalore", "mysore", "mysuru"),
    "bangalore": ("bengaluru", "bangalore", "mysore", "mysuru"),
    "sanchi": ("sanchi", "bhopal", "vidisha"),
    "belgaum": (
        "belgaum",
        "belagavi",
        "belgaon",
        "kolhapur",
        "sangli",
        "dharwad",
    ),
    "belagavi": (
        "belgaum",
        "belagavi",
        "belgaon",
        "kolhapur",
        "sangli",
        "dharwad",
    ),
    "belgaon": (
        "belgaum",
        "belagavi",
        "belgaon",
        "kolhapur",
        "sangli",
        "dharwad",
    ),
    "sangli": (
        "sangli",
        "miraj",
        "kolhapur",
        "satara",
        "kopeshwar",
        "khidrapur",
        "sagareshwar",
        "belgaum",
        "belagavi",
    ),
}

STATE_ALIASES = {
    "maharastra": "maharashtra",
    "maharashtra": "maharashtra",
    "karnataka": "karnataka",
    "delhi": "delhi",
    "tamil nadu": "tamil nadu",
    "tamilnadu": "tamil nadu",
    "uttar pradesh": "uttar pradesh",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)), 1)


def lookup_coords(name: str | None) -> tuple[float, float] | None:
    if not name:
        return None
    key = name.strip().lower()
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    for token, coords in CITY_COORDS.items():
        if token in key or key in token:
            return coords
    return None


def normalize_state(value: str | None) -> str:
    if not value:
        return ""
    return STATE_ALIASES.get(value.strip().lower(), value.strip().lower())


def nearby_set(origin: str | None) -> set[str]:
    if not origin:
        return set()
    from src.place_aliases import spellings

    key = origin.strip().lower()
    extra = NEARBY_CITIES.get(key, ())
    out = {key, *extra}
    for token in list(out):
        out |= spellings(token)
    return {t for t in out if t}


def geo_tier(
    *,
    origin_name: str,
    origin_city: str | None,
    origin_state: str | None,
    cand_city: str | None,
    cand_state: str | None,
    cand_name: str | None = None,
    same_asi_circle: bool = False,
    distance_km: float | None = None,
) -> str:
    """Geography only. same_asi_circle is ignored for locality (similarity is separate)."""
    del same_asi_circle
    oc = (origin_city or origin_name or "").lower()
    cc = (cand_city or "").lower()
    name_l = (cand_name or "").lower()
    near = nearby_set(origin_name) | nearby_set(origin_city)
    blob = f"{cc} {name_l}".strip()
    if cc and (cc == oc or oc in cc or cc in oc):
        return "locality_city"
    if blob and any(n and n in blob for n in near if n != oc):
        return "nearby_region"
    if cc and cc in near:
        return "nearby_region"
    if distance_km is not None and distance_km <= 40:
        return "locality_district"
    if distance_km is not None and distance_km <= MAX_NEARBY_KM:
        return "nearby_region"
    os_ = normalize_state(origin_state)
    cs = normalize_state(cand_state)
    if os_ and cs and os_ == cs:
        return "same_state"
    return "unverified"


def geography_verified(tier: str, distance_km: float | None) -> bool:
    if distance_km is not None:
        return distance_km <= MAX_NEARBY_KM
    return tier in {"locality_city", "locality_district", "nearby_region"}


def distance_score(distance_km: float | None, tier: str) -> float:
    """1 = next door, 0 = too far. Missing coords: never assume nearby via ASI circle."""
    if distance_km is not None:
        if distance_km <= 25:
            return 1.0
        if distance_km <= 60:
            return 0.85
        if distance_km <= 100:
            return 0.55
        if distance_km <= MAX_NEARBY_KM:
            return 0.25
        return 0.0
    if tier in {"locality_city", "locality_district"}:
        return 0.75
    if tier == "nearby_region":
        return 0.55
    if tier == "same_state":
        return 0.1
    return 0.05


def is_practical_orbit(tier: str, distance_km: float | None, *, relax: bool = False) -> bool:
    """ASI-circle-only is not practical. Geography must be verified by place names or km."""
    cap = MAX_RELAXED_KM if relax else MAX_NEARBY_KM
    if distance_km is not None:
        return distance_km <= cap
    return tier in {"locality_city", "locality_district", "nearby_region"}


def pair_distance_km(origin_name: str, cand_name: str, cand_city: str | None = None) -> float | None:
    a = lookup_coords(origin_name)
    b = lookup_coords(cand_city) or lookup_coords(cand_name)
    if not a or not b:
        return None
    return haversine_km(a[0], a[1], b[0], b[1])
