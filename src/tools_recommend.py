"""Nearby orbit ranking: geography first, then crowd-backed vs discovery modes.

Does not change the persistence forecast. Does not invent visitor counts.
"""

from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np

from src.geo import (
    MAX_NEARBY_KM,
    distance_score,
    geo_tier,
    geography_verified,
    is_practical_orbit,
    pair_distance_km,
)
from src.tools_local import (
    forecast_crowd,
    get_destination_profile,
    kaggle_catalog,
    search_destinations,
)

# Documented weights. Crowd pressure is used only in crowd_backed mode.
CROWD_BACKED_WEIGHTS = {
    "similarity": 0.28,
    "season_fit": 0.10,
    "accessibility": 0.10,
    "discovery_bonus": 0.12,
    "crowd_pressure": 0.20,
    "distance_penalty": 0.20,
}

DISCOVERY_WEIGHTS = {
    "similarity": 0.32,
    "season_fit": 0.12,
    "accessibility": 0.14,
    "discovery_bonus": 0.16,
    "crowd_pressure": 0.0,
    "distance_penalty": 0.26,
}

DEFAULT_WEIGHTS = CROWD_BACKED_WEIGHTS


def _dedupe_candidates(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in items:
        key = (item.get("name") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def find_similar_destinations(
    destination: str,
    interests: list[str] | None = None,
    location: str | None = None,
    limit: int = 12,
    max_distance_km: float | None = MAX_NEARBY_KM,
    include_wider: bool = False,
) -> list[dict]:
    """Nearby relevant candidates. Same-state-only is not treated as nearby."""
    profile = get_destination_profile(destination)
    kaggle = profile.get("kaggle") or {}
    place_type = (kaggle.get("place_type") or "").lower()
    state = (kaggle.get("state") or location or "").lower()
    origin_city = (kaggle.get("city") or location or destination or "").strip()
    origin_name = profile.get("asi_name") or profile.get("kaggle_name") or destination

    cat = kaggle_catalog()
    origin_l = origin_name.lower()
    kaggle_candidates = []
    for _, row in cat.iterrows():
        name = str(row["name"])
        if name.lower() == origin_l or name.lower() == destination.lower():
            continue
        type_l = str(row["place_type"]).lower()
        same_type = bool(place_type and type_l == place_type)
        same_state = bool(state and str(row["state"]).lower() == state)
        name_sim = SequenceMatcher(None, origin_l, name.lower()).ratio()
        interest_hit = False
        for item in interests or []:
            blob = type_l + " " + str(row["significance"]).lower()
            if item.lower().rstrip("s") in blob or item.lower() in blob:
                interest_hit = True
        dist = pair_distance_km(origin_city or destination, name, str(row["city"]))
        tier = geo_tier(
            origin_name=destination,
            origin_city=origin_city,
            origin_state=state,
            cand_city=str(row["city"]),
            cand_state=str(row["state"]),
            cand_name=name,
            same_asi_circle=False,
            distance_km=dist,
        )
        geo_ok = geography_verified(tier, dist)
        local_ok = is_practical_orbit(tier, dist, relax=include_wider)
        type_local = (same_type or interest_hit or name_sim >= 0.72) and local_ok
        if not local_ok and not include_wider:
            continue
        if not (type_local or local_ok and (same_type or interest_hit or same_state)):
            continue
        if max_distance_km is not None and dist is not None and dist > max_distance_km and not include_wider:
            continue
        kaggle_candidates.append(
            {
                "name": name,
                "place_type": row["place_type"],
                "state": row["state"],
                "city": row["city"],
                "zone": row["zone"],
                "season": row["season"],
                "google_rating": row["google_rating"],
                "airport_within_50km": row["airport_within_50km"],
                "distance_km": dist,
                "geo_tier": tier,
                "geography_verified": geo_ok,
                "source": "kaggle",
                "role": "orbit",
                "relevance": {
                    "same_type": same_type,
                    "same_state": same_state,
                    "same_zone": False,
                    "interest_hit": interest_hit,
                    "same_asi_circle": False,
                },
            }
        )

    asi_candidates = []
    if profile.get("asi_circle"):
        extras = search_destinations(profile["asi_circle"], limit=20)
        for hit in extras:
            if hit["source"] != "asi":
                continue
            if hit["name"] == origin_name:
                continue
            dist = pair_distance_km(origin_name, hit["name"], None)
            if dist is None:
                dist = pair_distance_km(origin_city or destination, hit["name"], None)
            tier = geo_tier(
                origin_name=destination,
                origin_city=origin_city,
                origin_state=state,
                cand_city="",
                cand_state="",
                cand_name=hit["name"],
                same_asi_circle=True,
                distance_km=dist,
            )
            geo_ok = geography_verified(tier, dist)
            if not is_practical_orbit(tier, dist, relax=include_wider):
                continue
            if max_distance_km is not None and dist is not None and dist > max_distance_km and not include_wider:
                continue
            asi_candidates.append(
                {
                    "name": hit["name"],
                    "place_type": "ASI monument",
                    "state": "",
                    "city": "",
                    "zone": "",
                    "season": "",
                    "google_rating": None,
                    "airport_within_50km": "",
                    "distance_km": dist,
                    "geo_tier": tier,
                    "geography_verified": geo_ok,
                    "source": "asi",
                    "role": "orbit",
                    "relevance": {
                        "same_type": True,
                        "same_state": False,
                        "same_zone": False,
                        "interest_hit": False,
                        "same_asi_circle": True,
                    },
                }
            )

    merged = _dedupe_candidates(asi_candidates + kaggle_candidates)
    return merged[:limit]


def _crowd_pressure_score(anchor_visitors: float | None, candidate_visitors: float | None) -> float:
    if anchor_visitors is None or candidate_visitors is None:
        return 0.5
    if anchor_visitors <= 0:
        return 0.5
    return float(np.clip(candidate_visitors / anchor_visitors, 0, 1.5) / 1.5)


def _mode_for(cand_visitors, anchor_visitors) -> str:
    if cand_visitors is not None and anchor_visitors is not None:
        return "crowd_backed"
    return "discovery"


def rank_alternatives(
    anchor: str,
    candidates: list[dict] | None = None,
    forecast_context: dict | None = None,
    user_preferences: dict | None = None,
    weights: dict | None = None,
) -> list[dict]:
    prefs = user_preferences or {}
    if candidates is None:
        candidates = find_similar_destinations(
            destination=anchor,
            interests=prefs.get("interests"),
            location=prefs.get("location"),
        )
    candidates = _dedupe_candidates(candidates)
    ctx = forecast_context or {}
    anchor_visitors = ctx.get("predicted_visitors")
    if anchor_visitors is None:
        try:
            fc = forecast_crowd(anchor)
            if fc.get("found"):
                anchor_visitors = fc["predicted_visitors"]
        except Exception:
            anchor_visitors = None

    ranked = []
    for cand in candidates:
        name = cand["name"]
        rel = cand.get("relevance") or {}
        city = cand.get("city")
        dist = cand.get("distance_km")
        if dist is None:
            dist = pair_distance_km(prefs.get("location") or anchor, name, city)
        tier = cand.get("geo_tier") or geo_tier(
            origin_name=anchor,
            origin_city=prefs.get("location"),
            origin_state=None,
            cand_city=city,
            cand_state=cand.get("state"),
            cand_name=name,
            same_asi_circle=bool(rel.get("same_asi_circle")),
            distance_km=dist,
        )
        geo_ok = cand.get("geography_verified")
        if geo_ok is None:
            geo_ok = geography_verified(tier, dist)
        practical = bool(is_practical_orbit(tier, dist, relax=bool(prefs.get("include_wider"))))
        if not practical and not prefs.get("include_wider"):
            continue

        d_score = distance_score(dist, tier)
        distance_penalty = round(1.0 - d_score, 3)
        if not geo_ok:
            distance_penalty = min(1.0, distance_penalty + 0.45)

        similarity = 0.0
        if rel.get("same_type"):
            similarity += 0.45
        if rel.get("same_asi_circle"):
            similarity += 0.12
        if rel.get("interest_hit"):
            similarity += 0.20
        if tier in {"locality_city", "locality_district"}:
            similarity += 0.20
        elif tier == "nearby_region":
            similarity += 0.10
        similarity = float(min(similarity, 1.0))
        relevant = similarity >= 0.2 or bool(rel.get("interest_hit") or rel.get("same_type"))

        season_fit = 0.5
        wanted_season = (prefs.get("season") or "").lower()
        if wanted_season and wanted_season in str(cand.get("season", "")).lower():
            season_fit = 0.9

        airport = str(cand.get("airport_within_50km") or "")
        accessibility = 0.8 if airport.lower() == "yes" else 0.45

        cand_visitors = None
        try:
            from src.predict import load_panel

            official_names = {n.lower(): n for n in load_panel()["monument"].unique()}
            if name.lower() in official_names:
                fc = forecast_crowd(official_names[name.lower()])
                if fc.get("found"):
                    cand_visitors = fc["predicted_visitors"]
        except Exception:
            cand_visitors = None

        mode = "crowd_backed" if (cand_visitors is not None and relevant and practical) else "discovery"
        if not relevant:
            continue
        crowd_data_available = cand_visitors is not None
        if mode == "discovery":
            cand_visitors = None

        w = {**(DISCOVERY_WEIGHTS if mode == "discovery" else CROWD_BACKED_WEIGHTS), **(weights or {})}

        crowd_pressure = 0.0
        if mode == "crowd_backed":
            crowd_pressure = _crowd_pressure_score(anchor_visitors, cand_visitors)

        discovery = 0.0
        if mode == "crowd_backed" and cand_visitors is not None and anchor_visitors and cand_visitors < 0.5 * anchor_visitors:
            absorb_floor = min(200_000.0, 0.05 * float(anchor_visitors))
            discovery = 0.85 if cand_visitors >= absorb_floor else 0.15
        elif mode == "discovery":
            discovery = 0.45

        if mode == "crowd_backed":
            score = (
                w["similarity"] * similarity
                + w["season_fit"] * season_fit
                + w["accessibility"] * accessibility
                + w["discovery_bonus"] * discovery
                - w["crowd_pressure"] * crowd_pressure
                - w["distance_penalty"] * distance_penalty
            )
        else:
            score = (
                w["similarity"] * similarity
                + w["season_fit"] * season_fit
                + w["accessibility"] * accessibility
                + w["discovery_bonus"] * discovery
                - w["distance_penalty"] * distance_penalty
            )

        why = []
        why.append(f"recommendation_mode={mode}")
        why.append(f"geo_tier={tier}")
        if dist is not None:
            why.append(f"distance_km={dist}")
        else:
            why.append("No gazetteer coordinates for this pair; distance not invented")
        if rel.get("same_asi_circle"):
            why.append("Same ASI circle (heritage similarity only; not treated as nearby)")
        if rel.get("same_type"):
            why.append(f"Similar type ({cand.get('place_type')})")
        if cand_visitors is not None and mode == "crowd_backed":
            why.append(
                f"Lower predicted annual demand ({cand_visitors:,.0f} vs {anchor_visitors:,.0f})"
                if anchor_visitors and cand_visitors < anchor_visitors
                else f"Predicted annual demand {cand_visitors:,.0f} (model output)"
            )
        elif mode == "discovery":
            why.append("Discovery: no ASI annual series; no crowd number invented")
        why.append("Heuristic ranking, not occupancy")

        ranked.append(
            {
                "name": name,
                "final_score": round(float(score), 4),
                "recommendation_mode": mode,
                "recommendation_kind": (
                    "crowd-backed alternative" if mode == "crowd_backed" else "discovery recommendation"
                ),
                "relevant": relevant,
                "practical": practical,
                "crowd_data_available": crowd_data_available,
                "geography_verified": bool(geo_ok),
                "role": cand.get("role") or "orbit",
                "geo_tier": tier,
                "distance_km": dist,
                "distance_score": round(d_score, 3),
                "city": city,
                "state": cand.get("state"),
                "source": cand.get("source"),
                "url": cand.get("url"),
                "evidence_tier": cand.get("evidence_tier"),
                "components": {
                    "similarity": round(similarity, 3),
                    "season_fit": season_fit,
                    "accessibility": accessibility,
                    "discovery_bonus": round(discovery, 3),
                    "crowd_pressure": round(crowd_pressure, 3) if mode == "crowd_backed" else None,
                    "distance_penalty": distance_penalty,
                },
                "predicted_visitors": cand_visitors if mode == "crowd_backed" else None,
                "why": why,
                "layer": "RECOMMENDATION",
            }
        )
    ranked.sort(
        key=lambda r: (
            -r["final_score"],
            r.get("distance_km") if r.get("distance_km") is not None else 10_000,
        )
    )
    return ranked


def filter_practical_candidates(
    candidates: list[dict],
    *,
    relax: bool = False,
) -> list[dict]:
    kept = []
    for c in candidates:
        dist = c.get("distance_km")
        tier = c.get("geo_tier") or ""
        if c.get("geography_verified") is False and dist is None:
            if not relax:
                continue
        if is_practical_orbit(tier, dist, relax=relax):
            kept.append(c)
    return kept
