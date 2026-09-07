"""Web-search abstraction. No fake results. Optional Tavily if TAVILY_API_KEY is set."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date

from src.geo import CITY_COORDS, nearby_set

USER_AGENT = "tourism-crowd-dss/0.1 (SIH prototype; educational)"

OFFICIAL_HOST_MARKERS = (
    "asi.nic.in",
    "tourism.gov.in",
    "incredibleindia.gov.in",
    "maharashtratourism.gov.in",
    "karnatakaholidays.net",
    "karnatakatourism.org",
    "delhitourism.gov.in",
    "tamilnadutourism.tn.gov.in",
    "karnataka.gov.in",
    "maharashtra.gov.in",
)

# Official portals that belong to one state. Hits from another state's portal
# are dropped unless the user's destination string is in the URL.
HOST_STATE = {
    "karnatakatourism.org": "karnataka",
    "karnatakaholidays.net": "karnataka",
    "karnataka.gov.in": "karnataka",
    "maharashtratourism.gov.in": "maharashtra",
    "maharashtra.gov.in": "maharashtra",
    "delhitourism.gov.in": "delhi",
    "tamilnadutourism.tn.gov.in": "tamil nadu",
}

CREDIBLE_HOST_MARKERS = (
    "wikipedia.org",
    "unesco.org",
    "lonelyplanet.com",
)

JUNK_TITLE_MARKERS = (
    "recruitment",
    "advertisement",
    "eoi",
    "usq ",
    "usq%",
    "piddc",
    "tender",
    "vacancy",
    "revised advertisement",
    "newsletter",
    "facebook",
    "instagram",
    "book my trip",
    "home - department of tourism",
)
JUNK_URL_MARKERS = ("usq", "/recruitment", "eoi", "piddc", "facebook.com", "instagram.com")

TOPIC_KEYS = {
    "weather": ("weather", "temperature", "rainfall", "climate"),
    "events": ("festival", "event", "mahotsav", "celebration", "mela"),
    "holidays": ("holiday", "public holiday", "bank holiday"),
    "closures": ("closed", "closure", "shutdown", "strike"),
    "timings": ("timing", "timings", "darshan", "visiting hours", "opening"),
    "advisories": ("advisory", "advisory", "alert", "warning"),
}


def _blob(hit: dict) -> str:
    return f"{hit.get('title') or ''} {hit.get('snippet') or ''} {hit.get('url') or ''}".lower()


def mentions_token(blob: str, token: str) -> bool:
    tok = (token or "").strip().lower()
    if len(tok) < 3:
        return False
    import re

    if len(tok) <= 4:
        return re.search(rf"\b{re.escape(tok)}\b", blob) is not None
    return tok in blob


def mentions_any_token(blob: str, tokens: set[str]) -> bool:
    return any(mentions_token(blob, t) for t in tokens if t)


def is_portal_homepage(url: str, dest_l: str) -> bool:
    path = urllib.parse.urlparse(url or "").path.strip("/").lower()
    if dest_l and dest_l in (url or "").lower():
        return False
    return path in {"", "en", "en-in", "home", "index.php", "index.html"}


def wrong_state_portal(url: str, dest_state: str | None, dest_l: str) -> bool:
    if not dest_state:
        return False
    host = urllib.parse.urlparse(url or "").netloc.lower()
    want = dest_state.strip().lower()
    for marker, st in HOST_STATE.items():
        if marker in host and st != want:
            if dest_l and dest_l in (url or "").lower():
                return False
            return True
    return False


def allowed_place_tokens(dest: str | None, nearby_tokens: list[str] | None) -> set[str]:
    dest_l = (dest or "").strip().lower()
    allowed = nearby_set(dest) if dest else set()
    allowed |= {t.lower() for t in (nearby_tokens or []) if t}
    if dest_l:
        allowed.add(dest_l)
    return {t for t in allowed if t}


def is_off_orbit_gazetteer_name(name: str, dest: str | None, nearby_tokens: list[str] | None) -> bool:
    """Reject a candidate that is itself another known city/district not in this orbit."""
    allowed = allowed_place_tokens(dest, nearby_tokens)
    key = (name or "").strip().lower()
    if not key:
        return True
    if key in CITY_COORDS and key not in allowed:
        return True
    first = key.split()[0]
    if first in CITY_COORDS and first not in allowed:
        return True
    return False


def classify_evidence_tier(url: str | None, source: str | None = None) -> str:
    blob = f"{url or ''} {source or ''}".lower()
    if any(h in blob for h in OFFICIAL_HOST_MARKERS):
        return "official"
    if any(h in blob for h in CREDIBLE_HOST_MARKERS):
        return "credible_secondary"
    return "general_web"


def is_junk_result(hit: dict) -> bool:
    title = (hit.get("title") or "").lower()
    url = (hit.get("url") or "").lower()
    if any(m in title for m in JUNK_TITLE_MARKERS):
        return True
    if any(m in url for m in JUNK_URL_MARKERS):
        return True
    if url.endswith(".pdf") and any(x in url for x in ("usq", "question", "loksabha", "rajyasabha")):
        return True
    return False


def classify_topics(hit: dict) -> list[str]:
    text = f"{hit.get('title') or ''} {hit.get('snippet') or ''}".lower()
    topics = [topic for topic, keys in TOPIC_KEYS.items() if any(k in text for k in keys)]
    return topics


def annotate_results(results: list[dict]) -> list[dict]:
    out = []
    for hit in results:
        item = dict(hit)
        item["evidence_tier"] = classify_evidence_tier(item.get("url"), item.get("source"))
        item["junk"] = is_junk_result(item)
        item["topics"] = classify_topics(item)
        out.append(item)
    return out


def structure_web_context(
    results: list[dict],
    *,
    destination: str | None,
    queries: list[str],
    provider: str | None,
    nearby_tokens: list[str] | None = None,
    dest_state: str | None = None,
) -> dict:
    annotated = annotate_results(results)
    kept = [h for h in annotated if not h.get("junk")]
    discarded = [h for h in annotated if h.get("junk")]
    dest_l = (destination or "").lower()
    tokens = allowed_place_tokens(destination, nearby_tokens)
    filtered = []
    for h in kept:
        url = h.get("url") or ""
        blob = _blob(h)
        if is_portal_homepage(url, dest_l):
            continue
        if wrong_state_portal(url, dest_state, dest_l):
            continue
        if tokens and not mentions_any_token(blob, tokens):
            continue
        filtered.append(h)
    kept = filtered
    buckets = {k: [] for k in TOPIC_KEYS}
    discovery = []
    for h in kept:
        placed = False
        for topic in h.get("topics") or []:
            if topic in buckets:
                buckets[topic].append(h)
                placed = True
        if not placed:
            discovery.append(h)
    return {
        "provider": provider,
        "retrieved_on": date.today().isoformat(),
        "queries": queries,
        "layer": "FACTS",
        "weather": buckets["weather"],
        "events": buckets["events"],
        "holidays": buckets["holidays"],
        "closures": buckets["closures"],
        "timings": buckets["timings"],
        "advisories": buckets["advisories"],
        "discovery": discovery,
        "results": kept,
        "discarded": discarded,
        "warnings": [
            f"Dropped {len(discarded)} irrelevant/junk hits (recruitment, parliamentary PDFs, etc.)."
        ]
        if discarded
        else [],
    }


JUNK_NAME_MARKERS = (
    "facebook",
    "zoom",
    "instagram",
    "whatsapp",
    "newsletter",
    "download",
    "preview",
    "book my trip",
)


def _normalize_attraction_name(name: str) -> str:
    import re

    n = re.sub(r"\s+", " ", name or "").strip(" .-|")
    n = re.sub(r"^(india|the)\s+", "", n, flags=re.I)
    n = re.sub(r"\b(facebook|zoom|instagram)\b", "", n, flags=re.I)
    n = re.sub(r"\s+", " ", n).strip()
    words = n.split()
    if len(words) >= 4 and len(words) % 2 == 0:
        half = len(words) // 2
        if [w.lower() for w in words[:half]] == [w.lower() for w in words[half:]]:
            n = " ".join(words[:half])
    return n.strip(" .-")


def _name_is_junk(name: str, dest_l: str, nearby_tokens: list[str] | None = None) -> bool:
    nl = (name or "").lower()
    if len(nl) < 4:
        return True
    if any(m in nl for m in JUNK_NAME_MARKERS):
        return True
    if nl in {"home"}:
        return True
    if dest_l and nl == dest_l:
        return True
    if is_off_orbit_gazetteer_name(name, dest_l, nearby_tokens):
        return True
    return False


def llm_nearby_attractions(
    dest: str,
    hits: list[dict],
    region_state: str = "",
    name_tokens: list[str] | None = None,
) -> list[dict]:
    """Lightweight pass-through: web_discovery_candidates extracts attractions directly
    without making a redundant nested LLM call."""
    return []
    dest_l = (dest or "").lower()
    tokens = [t.lower() for t in (name_tokens or []) if t] or ([dest_l] if dest_l else [])
    usable = []
    for h in hits[:12]:
        blob = f"{h.get('title')} {h.get('snippet')} {h.get('url')}"
        blob_l = blob.lower()
        if tokens and not any(t in blob_l for t in tokens if len(t) >= 4):
            continue
        if is_portal_homepage(h.get('url') or "", dest_l):
            continue
        if wrong_state_portal(h.get('url') or "", region_state, dest_l):
            continue
        usable.append(
            {
                "title": h.get("title"),
                "url": h.get("url"),
                "snippet": (h.get("snippet") or "")[:400],
            }
        )
    if not usable:
        return []
    try:
        import json
        import re

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        prompt = (
            f"Destination: {dest} ({region_state or 'India'}).\n"
            "From these search hits, list up to 5 tourist attractions IN or NEAR that destination only. "
            "Reject other states, other districts, and any city that is not in the same local orbit. "
            "Reject Facebook/Zoom/newsletter/statewide homepage junk. Use each clean name once. "
            "Do not invent visitor counts.\n"
            'JSON: {"places": [{"name": "...", "kind": "Temple"}]}\n'
            f"Hits: {json.dumps(usable)[:5000]}"
        )
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-3.6-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
            max_output_tokens=400,
        )
        msg = llm.invoke(
            [
                SystemMessage(content="Extract nearby attractions JSON only. No visitor counts."),
                HumanMessage(content=prompt),
            ]
        )
        raw = (msg.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        data = json.loads(raw)
        places = data.get("places") if isinstance(data, dict) else None
        if not isinstance(places, list):
            return []
    except Exception:
        return []
    out = []
    for p in places:
        if not isinstance(p, dict):
            continue
        name = _normalize_attraction_name(str(p.get("name") or ""))
        if _name_is_junk(name, dest_l, list(tokens) if tokens else list(nearby_set(dest) if dest else [])):
            continue
        out.append(
            {
                "name": name,
                "place_type": p.get("kind") or "Attraction",
                "state": region_state,
                "city": dest if dest_l and dest_l in name.lower() else "",
                "source": "official_web",
                "role": "orbit",
                "geo_tier": "nearby_region",
                "distance_km": None,
                "geography_verified": True,
                "crowd_data_available": False,
                "predicted_visitors": None,
                "evidence_tier": "official",
                "relevance": {
                    "same_type": False,
                    "same_state": True,
                    "interest_hit": True,
                    "same_asi_circle": False,
                },
            }
        )
    return _dedupe_web(out)


def web_discovery_candidates(
    results: list[dict],
    dest: str,
    nearby_tokens: list[str] | None = None,
    interests: list[str] | None = None,
    dest_state: str | None = None,
) -> list[dict]:
    """Turn official tourism hits into discovery candidates. Never invents crowd counts."""
    import re

    dest_l = (dest or "").strip().lower()
    tokens = {t.lower() for t in (nearby_tokens or []) if t} | ({dest_l} if dest_l else set())
    interest_l = [i.lower() for i in (interests or [])]
    type_map = {
        "fort": "Fort",
        "beach": "Beach",
        "temple": "Temple",
        "lake": "Lake",
        "sanctuary": "Wildlife Sanctuary",
        "wildlife-sanctuary": "Wildlife Sanctuary",
        "waterfall": "Waterfall",
        "palace": "Palace",
        "caves": "Caves",
        "cave": "Caves",
    }
    skip_slugs = {
        "districts",
        "directorate-of-tourism-dot",
        "festivals",
        "news",
        "recruitment",
        "tourist-intrests",
        "tourist-interests",
        "best-tourism-village",
        "hill-stations",
    }
    skip_names = {
        dest_l,
        "home",
        "forts",
        "beaches",
        "temples",
        "holi",
        "ganeshotsav",
        "department of tourism",
        "maharashtra",
        "karnataka",
        "tamil nadu",
        "delhi",
    }

    def _title_case(slug: str) -> str:
        return " ".join(p.capitalize() for p in slug.replace("_", "-").split("-") if p)

    found: list[dict] = []

    for hit in annotate_results(results):
        if hit.get("junk"):
            continue
        if hit.get("evidence_tier") not in {"official", "credible_secondary"}:
            continue
        title = hit.get("title") or ""
        url = hit.get("url") or ""
        snippet = hit.get("snippet") or ""
        blob = f"{title} {snippet} {url}".lower()
        if is_portal_homepage(url, dest_l):
            continue
        if wrong_state_portal(url, dest_state, dest_l):
            continue
        if tokens and not mentions_any_token(blob, tokens):
            continue
        path = urllib.parse.urlparse(url).path.lower()
        if any(f"/{s}" in path or path.rstrip("/").endswith(s) for s in skip_slugs):
            # still scan snippet for named attractions on district pages
            pass

        names: list[tuple[str, str]] = []

        m_url = re.search(
            r"/(fort|beach|temple|lake|sanctuary|wildlife-sanctuary|waterfall|palace|caves?)/([a-z0-9-]+)",
            path,
        )
        if m_url:
            kind, slug = m_url.group(1), m_url.group(2)
            if slug not in skip_slugs and slug != dest_l:
                label = type_map.get(kind, kind.replace("-", " ").title())
                pretty = _title_case(slug)
                if kind in {"fort", "beach", "temple"} and label.lower() not in pretty.lower():
                    pretty = f"{pretty} {label}"
                names.append((pretty, label))
            elif slug == dest_l and kind in {"fort", "beach", "temple"}:
                names.append((f"{dest} {type_map.get(kind, kind.title())}", type_map.get(kind, kind)))

        m_explore = re.search(
            r"(?:explore|visit|discover)\s+(?:the\s+)?([A-Z][A-Za-z0-9 .'-]{3,60}?)(?:\s+near|\s+in\s|\s+\||$)",
            title,
            re.I,
        )
        if m_explore:
            label = m_explore.group(1).split(" near")[0].strip(" .")
            if not is_off_orbit_gazetteer_name(label, dest, list(tokens)):
                names.append((label, "Attraction"))

        m_dash = re.match(
            r"^([A-Z][A-Za-z0-9 .']{2,40}?)\s*[-–|]\s*(?:Department of Tourism|Incredible India)",
            title,
        )
        if m_dash:
            label = m_dash.group(1).strip()
            if label.lower() in tokens and label.lower() != dest_l:
                names.append((label, "Attraction"))

        for m in re.finditer(
            r"\b([A-Z][A-Za-z][A-Za-z0-9']*(?:\s+[A-Z][A-Za-z0-9']*){0,3})\s+"
            r"(Fort|Beach|Temple|Lake|Sanctuary|Palace|Caves|Falls|Waterfall)\b",
            f"{title} {snippet}",
        ):
            names.append((f"{m.group(1)} {m.group(2)}", m.group(2)))

        for raw_name, ptype in names:
            name = _normalize_attraction_name(raw_name)
            if _name_is_junk(name, dest_l, list(tokens)):
                continue
            if not name or len(name) < 4:
                continue
            nl = name.lower()
            if nl in skip_names:
                continue
            if "department" in nl or "tourism" in nl or "constituency" in nl:
                continue
            interest_hit = True
            if interest_l:
                interest_hit = any(
                    i.rstrip("s") in (nl + " " + ptype.lower()) or i in (nl + " " + ptype.lower())
                    for i in interest_l
                ) or any(k in ptype.lower() for k in ("fort", "beach", "temple", "sanctuary", "lake", "nature"))
            tied = dest_l in nl or any(t in nl for t in tokens if t and t != dest_l)
            found.append(
                {
                    "name": name,
                    "place_type": ptype,
                    "state": "",
                    "city": dest if dest_l in nl else "",
                    "source": "official_web" if hit.get("evidence_tier") == "official" else "web",
                    "role": "orbit",
                    "geo_tier": "nearby_region" if tied else "unverified",
                    "distance_km": None,
                    "geography_verified": bool(tied),
                    "crowd_data_available": False,
                    "predicted_visitors": None,
                    "evidence_tier": hit.get("evidence_tier"),
                    "url": url,
                    "relevance": {
                        "same_type": False,
                        "same_state": False,
                        "interest_hit": interest_hit,
                        "same_asi_circle": False,
                    },
                }
            )

    return _dedupe_web(found)


def _dedupe_web(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in items:
        key = (item.get("name") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def web_search(query: str, domains: list[str] | None = None, max_results: int = 5) -> dict:
    """Search the public web. Returns structured hits or an explicit failure.

    Priority: Tavily (if key) → Wikipedia OpenSearch. Never invent URLs.
    """
    q = query.strip()
    if domains:
        scoped = " ".join(f"site:{d}" for d in domains)
        q = f"{q} {scoped}"

    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_key:
        try:
            return _tavily_search(q, tavily_key, max_results)
        except Exception as exc:
            tavily_error = str(exc)
    else:
        tavily_error = None

    wiki = _wikipedia_search(query.strip(), max_results)
    results = annotate_results(wiki["results"])
    if domains:
        results = [
            r for r in results if any(d.lower() in (r.get("url") or "").lower() for d in domains)
        ]
    return {
        "query": query,
        "provider": "wikipedia_opensearch" if results else "none",
        "retrieved_on": date.today().isoformat(),
        "layer": "FACTS",
        "results": results[:max_results],
        "warnings": [
            w
            for w in [
                "No TAVILY_API_KEY; used Wikipedia OpenSearch only."
                if not tavily_key
                else None,
                f"Tavily failed: {tavily_error}" if tavily_error else None,
                "Wikipedia is a general source (credible_secondary), not an official tourism authority."
                if results
                else "No web results returned.",
            ]
            if w
        ],
    }


def _tavily_search(query: str, key: str, max_results: int) -> dict:
    payload = json.dumps(
        {"api_key": key, "query": query, "max_results": max_results}
    ).encode()
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode())
    results = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "source": urllib.parse.urlparse(item.get("url") or "").netloc,
                "published": item.get("published_date"),
                "snippet": item.get("content"),
            }
        )
    results = annotate_results(results)
    return {
        "query": query,
        "provider": "tavily",
        "retrieved_on": date.today().isoformat(),
        "layer": "FACTS",
        "results": results,
        "warnings": [],
    }


def _wikipedia_search(query: str, max_results: int) -> dict:
    params = urllib.parse.urlencode(
        {
            "action": "opensearch",
            "search": query,
            "limit": max_results,
            "namespace": 0,
            "format": "json",
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode())
    titles = payload[1] if len(payload) > 1 else []
    descs = payload[2] if len(payload) > 2 else []
    urls = payload[3] if len(payload) > 3 else []
    results = []
    for i, title in enumerate(titles):
        results.append(
            {
                "title": title,
                "url": urls[i] if i < len(urls) else None,
                "source": "en.wikipedia.org",
                "published": None,
                "snippet": descs[i] if i < len(descs) else None,
            }
        )
    return {"results": results}
