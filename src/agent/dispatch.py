"""Run one brain-selected tool against Python implementations. No invented counts."""

from __future__ import annotations

import json
from typing import Any

from src.agent.optimize import optimize_trip
from src.geo import nearby_set
from src.tools_local import (
    classify_crowd,
    forecast_crowd,
    get_destination_profile,
    get_historical_footfall,
    search_destinations,
)
from src.tools_period import assess_period_pressure
from src.tools_recommend import filter_practical_candidates, find_similar_destinations, rank_alternatives
from src.tools_web import (
    OFFICIAL_HOST_MARKERS,
    llm_nearby_attractions,
    structure_web_context,
    web_discovery_candidates,
    web_search,
)


def _orbit_tokens(state: dict) -> list[str]:
    dest = state.get("destination")
    names = [n for n in (state.get("place_names") or []) if n]
    return list({t.lower() for t in nearby_set(dest)} | {n.lower() for n in names} | ({dest.lower()} if dest else set()))


def _clip(payload: Any, limit: int = 8000) -> str:
    text = json.dumps(payload, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def apply_set_trip_context(state: dict, args: dict) -> dict:
    patch: dict[str, Any] = {}
    dest = (args.get("destination") or "").strip()
    if dest:
        patch["destination"] = dest
    aliases = [a.strip() for a in str(args.get("also_known_as") or "").split(",") if a.strip()]
    names = []
    seen = set()
    for n in [args.get("destination"), *(state.get("place_names") or []), *aliases]:
        if not n:
            continue
        key = str(n).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(str(n).strip())
    if names:
        patch["place_names"] = names
    if (args.get("state") or "").strip():
        patch["dest_state"] = args["state"].strip()
    if (args.get("climate_zone") or "").strip():
        patch["climate_zone"] = args["climate_zone"].strip().lower()
    if (args.get("start_date") or "").strip():
        patch["start_date"] = args["start_date"].strip()[:10]
    if (args.get("end_date") or "").strip():
        patch["end_date"] = args["end_date"].strip()[:10]
    if (args.get("party_type") or "").strip():
        patch["party_type"] = args["party_type"].strip().lower()
    must = [a.strip() for a in str(args.get("must_visit") or "").split(",") if a.strip()]
    if must:
        patch["must_visit"] = list(dict.fromkeys(list(state.get("must_visit") or []) + must))
    interests = [a.strip() for a in str(args.get("interests") or "").split(",") if a.strip()]
    if interests:
        patch["interests"] = list(dict.fromkeys(list(state.get("interests") or []) + interests))
    intent = dict(state.get("intent") or {})
    for flag in ("needs_prediction", "needs_alternatives", "needs_trip_plan", "needs_current_context"):
        if flag in args and args[flag] is not None:
            intent[flag] = bool(args[flag])
    if dest or aliases:
        intent["needs_current_context"] = True
        intent["place_note"] = (
            f"Interpreted destination as {patch.get('destination') or dest}"
            + (f" / {', '.join(aliases)}" if aliases else "")
            + (f" ({patch.get('dest_state')})" if patch.get("dest_state") else "")
            + "."
        )
    patch["intent"] = intent
    return patch


def run_named_tool(name: str, args: dict, state: dict) -> tuple[Any, dict]:
    """Execute a whitelist tool. Returns (observation, state_patch)."""
    args = args or {}
    dest = (args.get("destination") or args.get("anchor") or args.get("query") or state.get("destination") or "")
    dest = str(dest).strip() if dest else ""
    location = (args.get("location") or state.get("destination") or "") or None
    interests = args.get("interests")
    if isinstance(interests, str):
        interest_list = [x.strip() for x in interests.split(",") if x.strip()] or None
    else:
        interest_list = list(state.get("interests") or []) or None
    dest_state = args.get("region_state") or args.get("state") or state.get("dest_state")
    climate_zone = args.get("climate_zone") or state.get("climate_zone")
    start = args.get("start_date") or state.get("start_date")
    end = args.get("end_date") or state.get("end_date")
    patch: dict[str, Any] = {"tool_trace": list(state.get("tool_trace") or []) + [name]}
    decisions = list(state.get("decisions") or [])

    if name == "clarify_with_user":
        question = (args.get("question") or "").strip()
        missing = (args.get("missing") or "").strip()
        return (
            {"awaiting_user": True, "question": question, "missing": missing},
            {
                "awaiting_user": True,
                "pending_question": question,
                "missing_slots": [p.strip() for p in missing.split(",") if p.strip()],
                "tool_trace": patch["tool_trace"],
                "decisions": decisions + ["Asked the traveler for missing trip details."],
            },
        )

    if name == "set_trip_context":
        ctx = apply_set_trip_context(state, args)
        ctx["awaiting_user"] = False
        decisions.append(ctx.get("intent", {}).get("place_note") or "Updated trip context from the model.")
        ctx["decisions"] = decisions
        return {"ok": True, **{k: v for k, v in ctx.items() if k != "intent"}}, {**ctx, "decisions": decisions}

    if name == "search_destinations":
        hits = search_destinations(
            args.get("query") or dest,
            location=location if location else None,
            interests=interest_list,
            limit=8,
        )
        merged = list(state.get("destinations") or []) + list(hits or [])
        patch["destinations"] = merged
        decisions.append(f"Catalog search '{args.get('query') or dest}': {len(hits or [])} hit(s).")
        patch["decisions"] = decisions
        return hits, patch

    if name == "get_destination_profile":
        profile = get_destination_profile(dest)
        profiles = dict(state.get("profiles") or {})
        profiles[dest or profile.get("query") or ""] = profile
        patch["profiles"] = profiles
        return profile, patch

    if name == "get_historical_footfall":
        hist = get_historical_footfall(dest)
        historical = dict(state.get("historical_demand") or {})
        historical[dest] = hist
        patch["historical_demand"] = historical
        return hist, patch

    if name == "forecast_crowd":
        fc = forecast_crowd(dest, args.get("forecast_period"))
        forecasts = dict(state.get("forecasts") or {})
        key = fc.get("destination") or dest
        forecasts[key] = fc
        patch["forecasts"] = forecasts
        if fc.get("found"):
            decisions.append(
                f"Forecast {key}: persistence {fc.get('predicted_visitors')} ({fc.get('forecast_period')})."
            )
        else:
            decisions.append(f"No ASI series for '{dest}'; refused to invent a visitor count.")
        patch["decisions"] = decisions
        return fc, patch

    if name == "classify_crowd":
        pred = args.get("predicted_visitors")
        if pred is None:
            fc = (state.get("forecasts") or {}).get(dest) or {}
            pred = fc.get("predicted_visitors")
        classified = classify_crowd(destination=dest, predicted_visitors=pred)
        crowds = dict(state.get("crowd_levels") or {})
        if classified.get("destination"):
            crowds[classified["destination"]] = classified
        patch["crowd_levels"] = crowds
        if classified.get("level"):
            decisions.append(f"Classified {classified.get('destination')} as {classified.get('level')}.")
        patch["decisions"] = decisions
        return classified, patch

    if name == "web_search":
        official = bool(args.get("official_only", True))
        domains = None
        if args.get("domains"):
            domains = [d.strip() for d in str(args["domains"]).split(",") if d.strip()]
        elif official:
            domains = list(OFFICIAL_HOST_MARKERS)
        out = web_search(str(args.get("query") or dest), domains=domains, max_results=5)
        raw_hits = list((state.get("web_raw_hits") or [])) + list(out.get("results") or [])
        queries = list(state.get("web_queries") or []) + [str(args.get("query") or dest)]
        web_context = structure_web_context(
            raw_hits,
            destination=state.get("destination") or dest,
            queries=queries,
            provider=out.get("provider"),
            nearby_tokens=_orbit_tokens({**state, **patch, "destination": state.get("destination") or dest}),
            dest_state=dest_state,
        )
        sources = list(state.get("sources") or [])
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
        patch.update(
            {
                "web_raw_hits": raw_hits,
                "web_queries": queries,
                "web_context": web_context,
                "sources": sources,
                "decisions": decisions + [f"Web search: {args.get('query') or dest}"],
            }
        )
        return out, patch

    if name == "extract_nearby_places":
        hits = state.get("web_raw_hits") or (state.get("web_context") or {}).get("results") or []
        origin = dest or state.get("destination") or ""
        tokens = _orbit_tokens({**state, "destination": origin})
        web_cands = web_discovery_candidates(
            hits,
            origin,
            tokens,
            interests=interest_list,
            dest_state=dest_state,
        )
        llm_cands = llm_nearby_attractions(origin, hits, dest_state or "", name_tokens=tokens)
        names = {c["name"].lower() for c in (state.get("candidate_alternatives") or [])}
        added = []
        for w in list(llm_cands) + list(web_cands):
            nl = (w.get("name") or "").lower()
            if not nl or nl in names:
                continue
            added.append(w)
            names.add(nl)
        candidates = list(state.get("candidate_alternatives") or []) + added
        patch["candidate_alternatives"] = candidates
        patch["decisions"] = decisions + [f"Extracted {len(added)} nearby names from web (no crowd numbers)."]
        return added, patch

    if name == "assess_period_pressure":
        crowd_lv = args.get("crowd_level")
        if not crowd_lv:
            for c in (state.get("crowd_levels") or {}).values():
                if (c or {}).get("level"):
                    crowd_lv = c.get("level")
                    break
        pressure = assess_period_pressure(
            dest or state.get("destination"),
            start,
            end or start,
            crowd_level=crowd_lv,
            web_context=state.get("web_context") or {},
            dest_state=dest_state,
            climate_zone=climate_zone,
        )
        patch["period_pressure"] = pressure
        patch["decisions"] = decisions + [
            f"Travel window {start}→{end or start}: direction={pressure.get('footfall_direction')} "
            "(heuristic; annual count unchanged)."
        ]
        return pressure, patch

    if name == "find_similar_destinations":
        raw = find_similar_destinations(
            dest or state.get("destination") or "",
            interests=interest_list,
            location=location,
            limit=20,
        )
        practical = filter_practical_candidates(raw, relax=False)
        patch["candidate_alternatives"] = practical
        patch["decisions"] = decisions + [
            f"Similarity around '{dest}': {len(raw)} raw, {len(practical)} practical nearby."
        ]
        return practical, patch

    if name == "rank_alternatives":
        anchor = args.get("anchor") or dest or state.get("destination") or ""
        cands = state.get("candidate_alternatives") or None
        fc_ctx = {}
        for v in (state.get("forecasts") or {}).values():
            if v.get("found"):
                fc_ctx = v
                break
        ranked = rank_alternatives(
            anchor,
            candidates=cands,
            forecast_context=fc_ctx if fc_ctx.get("found") else {},
            user_preferences={"interests": interest_list, "location": location or anchor},
        )
        ranked = [r for r in ranked if r.get("geo_tier") != "same_state" or (r.get("distance_km") or 999) <= 160]
        want_n = int((state.get("intent") or {}).get("requested_n") or 3)
        ranked = ranked[:want_n]
        patch["ranked_alternatives"] = ranked
        patch["decisions"] = decisions + [f"Ranked {len(ranked)} orbit(s) with existing scorecard."]
        return ranked, patch

    if name == "optimize_trip":
        must = list(state.get("must_visit") or [])
        if dest and dest not in must:
            must = [dest] + must
        plan = optimize_trip(
            start,
            end or start,
            must,
            state.get("ranked_alternatives") or [],
            state.get("crowd_levels") or {},
        )
        patch["trip_plan"] = plan
        patch["decisions"] = decisions + [f"Built {len(plan)}-day sketch from anchors + ranked orbits."]
        return plan, patch

    return {"error": f"unknown tool {name}"}, patch


def observation_text(name: str, payload: Any) -> str:
    return f"{name} → {_clip(payload)}"
