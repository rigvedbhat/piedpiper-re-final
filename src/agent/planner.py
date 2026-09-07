"""Deterministic tool plan from structured intent. Does not invent numbers."""

from __future__ import annotations

from src.agent.optimize import optimize_trip
from src.agent.state import TourismState
from src.geo import nearby_set
from src.tools_local import (
    classify_crowd,
    forecast_crowd,
    get_destination_profile,
    get_historical_footfall,
    search_destinations,
)
from src.tools_recommend import (
    filter_practical_candidates,
    find_similar_destinations,
    rank_alternatives,
)
from src.tools_period import assess_period_pressure
from src.tools_web import (
    OFFICIAL_HOST_MARKERS,
    structure_web_context,
    web_discovery_candidates,
    llm_nearby_attractions,
    web_search,
)

REGION_STATE = {
    "Mumbai": "Maharashtra",
    "Pratapgad": "Maharashtra",
    "Sangli": "Maharashtra",
    "Kolhapur": "Maharashtra",
    "Satara": "Maharashtra",
    "Pune": "Maharashtra",
    "Sindhudurg": "Maharashtra",
    "Hampi": "Karnataka",
    "Belagavi": "Karnataka",
    "Belgaum": "Karnataka",
    "Belgaon": "Karnataka",
    "Bengaluru": "Karnataka",
    "Chennai": "Tamil Nadu",
    "Delhi": "Delhi",
    "Taj Mahal": "Uttar Pradesh",
    "Qutub Minar": "Delhi",
    "Sanchi": "Madhya Pradesh",
}

KNOWN_FORECAST_SITES = {"Taj Mahal", "Hampi", "Sanchi", "Qutub Minar"}


def _uniq(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = (item or "").strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        out.append(key)
    return out


def _safe_classify(name: str, predicted: float | None) -> dict | None:
    try:
        return classify_crowd(destination=name, predicted_visitors=predicted)
    except Exception as exc:
        return {"destination": name, "found": False, "level": None, "note": str(exc)}


def _merge_web(
    raw_hits: list[dict],
    dest: str,
    queries: list[str],
    provider: str | None,
    nearby_tokens: list[str] | None = None,
    dest_state: str | None = None,
) -> dict:
    return structure_web_context(
        raw_hits,
        destination=dest,
        queries=queries,
        provider=provider,
        nearby_tokens=nearby_tokens or list(nearby_set(dest)),
        dest_state=dest_state or (REGION_STATE.get(dest) if dest else None),
    )


def execute_plan(state: TourismState) -> dict:
    intent = dict(state.get("intent") or {})
    dest = state.get("destination")
    must = list(state.get("must_visit") or [])
    interests = list(state.get("interests") or [])
    start = state.get("start_date")
    end = state.get("end_date")
    loc = dest
    want_n = int(intent.get("requested_n") or state.get("requested_n") or 3)
    place_names = [n for n in (state.get("place_names") or []) if n]
    if dest and dest.lower() not in {n.lower() for n in place_names}:
        place_names = [dest, *place_names]
    dest_state = state.get("dest_state") or (REGION_STATE.get(dest) if dest else None)
    climate_zone = state.get("climate_zone")
    state_label = dest_state or ""
    orbit_tokens = list({t.lower() for t in nearby_set(dest)} | {n.lower() for n in place_names})

    forecasts: dict = {}
    crowds: dict = {}
    profiles: dict = {}
    historical: dict = {}
    destinations: list = list(state.get("destinations") or [])
    candidates: list = []
    ranked: list = []
    sources: list = []
    web_context: dict = {}
    trace = list(state.get("tool_trace") or [])
    fallback = False
    anchors = _uniq(must)
    decisions: list[str] = []
    if intent.get("place_note"):
        decisions.append(intent["place_note"])

    region_query = dest or (must[0] if must else None)
    known_forecast_sites = KNOWN_FORECAST_SITES
    need_discovery = bool(
        intent.get("needs_trip_plan")
        or (dest and dest not in known_forecast_sites)
        or (not dest and must)
    )

    if need_discovery and region_query:
        merged: list[dict] = []
        for q in _uniq([region_query] + place_names + must):
            merged.extend(
                search_destinations(
                    q, location=loc if q == region_query else None, interests=None, limit=8
                )
            )
        trace.append("search_destinations")
        decisions.append(
            f"Searched local catalogs for {region_query}; kept {len(destinations)} name-or-location matches."
        )
        seen_hits = set()
        destinations = []
        for h in merged:
            name = h.get("name") or ""
            blob = " ".join(str(v) for v in h.values()).lower()
            local = bool(loc and loc.lower() in blob)
            named = any(tok.lower() in name.lower() for tok in [region_query] + place_names + must if tok)
            if not local and not named:
                continue
            key = (name.lower(), h.get("source"))
            if key in seen_hits:
                continue
            seen_hits.add(key)
            destinations.append(h)

    profile_names = _uniq(
        must + ([dest] if dest else []) + [h["name"] for h in destinations[:6] if h.get("name")]
    )
    if need_discovery or intent.get("needs_trip_plan"):
        for name in profile_names[:8]:
            profiles[name] = get_destination_profile(name)
            trace.append("get_destination_profile")
            historical[name] = get_historical_footfall(name)
            trace.append("get_historical_footfall")

    predict_names = []
    if intent.get("needs_prediction"):
        predict_names = _uniq(([dest] if dest else []) + must)
    elif intent.get("needs_trip_plan") or must:
        predict_names = _uniq(must + ([dest] if dest else []))

    for name in predict_names:
        if not name:
            continue
        fc = forecast_crowd(name)
        key = fc.get("destination") or name
        forecasts[key] = fc
        trace.append("forecast_crowd")
        if fc.get("found"):
            classified = _safe_classify(fc["destination"], fc.get("predicted_visitors"))
            if classified:
                crowds[fc["destination"]] = classified
                trace.append("classify_crowd")
                decisions.append(
                    f"Forecast {fc['destination']}: persistence {fc.get('predicted_visitors')} "
                    f"({fc.get('forecast_period')}); class {classified.get('level')}."
                )
        else:
            fallback = True
            decisions.append(f"No ASI series for '{name}'; refused to invent a visitor count.")

    if dest and forecasts and not any(v.get("found") for v in forecasts.values()):
        fallback = True

    catalog_empty = not destinations and dest not in KNOWN_FORECAST_SITES
    all_hits: list[dict] = []
    queries: list[str] = []
    provider = None
    official = list(OFFICIAL_HOST_MARKERS)
    unknown_local = bool(dest) and catalog_empty
    who_only = bool(
        intent.get("needs_current_context")
        and not intent.get("needs_prediction")
        and not intent.get("needs_trip_plan")
        and not intent.get("needs_alternatives")
    )

    def _search(query: str, domains=None) -> dict:
        nonlocal provider
        out = web_search(query, domains=domains, max_results=5)
        provider = out.get("provider") or provider
        queries.append(query)
        all_hits.extend(out.get("results") or [])
        return out

    need_web = bool(intent.get("needs_current_context") or intent.get("needs_trip_plan"))
    place = dest or " ".join(must)
    state_label = dest_state or REGION_STATE.get(place or "", "")
    if unknown_local and who_only:
        _search(f"Who is {dest}", None)
        trace.append("web_search")
        decisions.append(f"Unknown catalog name '{dest}'; searched the web without inventing tourism facts.")
    elif dest or must:
        if unknown_local:
            brain_queries = [q for q in (state.get("web_queries") or []) if q]
            if brain_queries:
                for q in brain_queries[:3]:
                    _search(q, official)
            else:
                alias_q = " ".join(_uniq(place_names)[:4]) or place
                _search(
                    f"{alias_q} {state_label} famous tourist attractions official tourism".strip(),
                    official,
                )
                _search(f"{alias_q} tourist places to visit official tourism", official)
            intent["needs_alternatives"] = True
            decisions.append(
                f"'{place}' is not in the ASI/Kaggle visitor tables; "
                "searched official web using brain-resolved names "
                f"{', '.join(place_names[:4]) or place} (no invented crowd numbers)."
            )
        if need_web or unknown_local:
            month_year = start[:7] if start else ""
            search_name = place_names[0] if place_names else place
            if start or need_web:
                _search(f"{search_name} tourism {month_year} official".strip(), official)
            if start and end:
                _search(f"{search_name} events {start} {end} official tourism", official)
            for site in must[:2]:
                _search(f"{site} {place} current timings official", official)
            if not all_hits:
                _search(f"{' '.join(_uniq(place_names)[:3]) or place} tourism attractions India".strip())
            if "web_search" not in trace:
                trace.append("web_search")
            decisions.append("Ran official-first web search for destination / current context.")

    if dest == "Zorblax" or (catalog_empty and dest and lower_unknown(dest)):
        needles = [n.lower() for n in place_names] or [(dest or "").lower()]
        all_hits = [
            h
            for h in all_hits
            if any(
                n in f"{h.get('title')} {h.get('snippet')} {h.get('url')}".lower()
                for n in needles
                if n
            )
        ]

    web_context = _merge_web(
        all_hits, dest or "", queries, provider, orbit_tokens, dest_state
    )
    for hit in web_context.get("results") or []:
        if hit.get("url"):
            sources.append(
                {
                    "title": hit.get("title"),
                    "url": hit.get("url"),
                    "domain": hit.get("source"),
                    "evidence_tier": hit.get("evidence_tier"),
                    "snippet": hit.get("snippet"),
                }
            )

    period_pressure: dict = {}
    if start:
        crowd_lv = None
        for c in crowds.values():
            if c.get("level"):
                crowd_lv = c.get("level")
                break
        period_pressure = assess_period_pressure(
            dest,
            start,
            end or start,
            crowd_level=crowd_lv,
            web_context=web_context,
            dest_state=dest_state,
            climate_zone=climate_zone,
        )
        trace.append("assess_period_pressure")
        decisions.append(
            f"Travel window {start}→{end or start}: direction="
            f"{period_pressure.get('footfall_direction')} (heuristic; annual count unchanged)."
        )
        from src.agent.research import nearby_from_research

        research = nearby_from_research(
            {
                "period_pressure": period_pressure,
                "web_context": web_context,
                "crowd_levels": crowds,
                "historical_demand": historical,
            }
        )
        if research.get("suggest_nearby"):
            intent["needs_alternatives"] = True
            decisions.append(
                "Research bundle suggests nearby places ("
                + "; ".join(research.get("reasons") or [])
                + "). Annual HIGH is not the only trigger."
            )

    if intent.get("needs_alternatives") and (dest or must):
        anchor = dest or must[0]
        found_fc = next((v for v in forecasts.values() if v.get("found")), None)
        if found_fc:
            anchor = found_fc["destination"]
        elif destinations:
            kaggle_hit = next((h["name"] for h in destinations if h.get("source") == "kaggle"), None)
            if kaggle_hit:
                anchor = kaggle_hit
        if not anchors:
            anchors = [anchor]
        candidates = find_similar_destinations(
            anchor,
            interests=interests or None,
            location=loc,
            limit=20,
        )
        trace.append("find_similar_destinations")
        practical = filter_practical_candidates(candidates, relax=False)
        decisions.append(
            f"Similarity search around '{anchor}' produced {len(candidates)} raw hits; "
            f"{len(practical)} passed verified geography (ASI circle alone is not nearby)."
        )
        candidates = practical
        web_cands = web_discovery_candidates(
            all_hits,
            dest or anchor,
            orbit_tokens,
            interests=interests or None,
            dest_state=dest_state or REGION_STATE.get(dest or "", ""),
        )
        if len(candidates) + len(web_cands) < want_n:
            _search(
                f"lesser known tourist places near {place_names[0] if place_names else dest} {state_label} official tourism",
                official,
            )
            if "web_search" not in trace:
                trace.append("web_search")
            elif trace[-1] != "web_search":
                trace.append("web_search")
            web_cands = web_discovery_candidates(
                all_hits,
                dest or anchor,
                orbit_tokens,
                interests=interests or None,
                dest_state=dest_state or REGION_STATE.get(dest or "", ""),
            )
            web_context = _merge_web(
                all_hits, dest or "", queries, provider, orbit_tokens, dest_state
            )
        llm_cands = llm_nearby_attractions(
            dest or anchor,
            all_hits,
            dest_state or REGION_STATE.get(dest or "", ""),
            name_tokens=orbit_tokens,
        )
        names = {c["name"].lower() for c in candidates}
        anchor_l = [a.lower() for a in (anchors + must) if a]
        for w in list(llm_cands) + list(web_cands):
            nl = w["name"].lower()
            if nl in names:
                continue
            if any(a in nl or nl in a for a in anchor_l):
                continue
            candidates.append(w)
            names.add(nl)
        decisions.append(
            f"Promoted {len(web_cands)} official-web names after dropping duplicates and must-visit aliases."
        )
        if not candidates:
            practical = filter_practical_candidates(
                find_similar_destinations(
                    anchor, interests=interests or None, location=loc, limit=20, include_wider=True
                ),
                relax=True,
            )
            candidates = [c for c in practical if (c.get("distance_km") or 9999) <= 220]

        fc_ctx = found_fc or {}
        ranked = rank_alternatives(
            anchor,
            candidates=candidates,
            forecast_context=fc_ctx if fc_ctx.get("found") else {},
            user_preferences={"interests": interests or None, "location": loc},
        )
        trace.append("rank_alternatives")
        ranked = [r for r in ranked if r.get("geo_tier") != "same_state" or (r.get("distance_km") or 999) <= 160]
        if not ranked:
            ranked = rank_alternatives(
                anchor,
                candidates=candidates,
                forecast_context=fc_ctx if fc_ctx.get("found") else {},
                user_preferences={"interests": interests or None, "location": loc},
            )
        ranked = ranked[:want_n]
        modes = {r.get("recommendation_mode") for r in ranked}
        decisions.append(
            f"Ranked top {len(ranked)} orbits; modes present: {', '.join(sorted(m for m in modes if m)) or 'none'}."
        )
        if len(ranked) < want_n:
            intent["insufficient_candidates"] = True
            intent["candidate_note"] = (
                f"Only {len(ranked)} practical nearby candidate(s) met the locality bar; "
                "did not pad with distant same-state sites."
            )

    trip_plan: list = []
    if intent.get("needs_trip_plan"):
        trip_plan = optimize_trip(start, end, must or anchors, ranked, crowds)
        trace.append("optimize_trip")

    intent["fallback_discovery"] = fallback
    intent["anchors"] = anchors
    return {
        "intent": intent,
        "destinations": destinations,
        "profiles": profiles,
        "historical_demand": historical,
        "forecasts": forecasts,
        "crowd_levels": crowds,
        "web_context": web_context,
        "sources": sources,
        "candidate_alternatives": candidates,
        "ranked_alternatives": ranked,
        "trip_plan": trip_plan,
        "period_pressure": period_pressure,
        "tool_trace": trace,
        "data_granularity": {"supported": "annual", "hourly": False, "daily": False},
        "requested_n": want_n,
        "decisions": decisions,
    }


def lower_unknown(dest: str) -> bool:
    return dest.lower() not in {k.lower() for k in REGION_STATE} and dest not in {
        "Taj Mahal",
        "Hampi",
        "Sanchi",
        "Qutub Minar",
    }
