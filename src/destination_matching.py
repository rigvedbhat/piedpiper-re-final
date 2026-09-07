"""Conservative ASI ↔ Kaggle destination matching. No forced matches."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# Hand-checked aliases only where the entity is unambiguous.
MANUAL_ALIASES = {
    "taj mahal": "Taj Mahal",
    "agra fort": "Agra Fort",
    "fatehpur sikri": "Fatehpur Sikri",
    "qutub minar": "Qutub Minar",
    "qutb minar": "Qutub Minar",
    "red fort": "Red Fort",
    "humayuns tomb": "Humayuns Tomb",
    "humayun tomb": "Humayuns Tomb",
    "hampi": "Group of Monuments, Hampi",
    "sanchi": "Buddhist Monuments, Sanchi",
    "mahabalipuram": "Group of Monuments, Mamallapuram",
    "mamallapuram": "Group of Monuments, Mamallapuram",
    "konark sun temple": "Sun Temple, Konarak",
    "sun temple konark": "Sun Temple, Konarak",
    "ajanta caves": "Ajanta Caves",
    "ellora caves": "Ellora Caves",
    "elephanta caves": "Elephanta Caves",
    "khajuraho": "Western Group of Temples, Khajuraho",
    "golconda fort": "Golkonda Fort",
    "golkonda fort": "Golkonda Fort",
    "charminar": "Charminar",
    "sarnath": "Excavated Remains at Sarnath",
    "nalanda": "Excavated Site, Nalanda",
    "chittorgarh fort": "Chittaurgarh Fort",
    "kumbhalgarh fort": "Kumbhalgarh Fort",
    "rani ki vav": "Rani Ki- Vav, Patan",
}


def normalize_name(text: str) -> str:
    s = str(text).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(
        r"\b(group of monuments|group of|the|wh|india|fort)\b",
        " ",
        s,
    )
    return " ".join(s.split())


def unique_kaggle_places(kaggle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, grp in kaggle.groupby("Place_Name", dropna=True):
        rows.append(
            {
                "kaggle_name": name,
                "kaggle_norm": normalize_name(name),
                "state": grp["Location_State"].mode().iloc[0]
                if grp["Location_State"].notna().any()
                else "",
                "zone": grp["Zone"].mode().iloc[0] if grp["Zone"].notna().any() else "",
                "place_type": grp["Place_Type"].mode().iloc[0]
                if grp["Place_Type"].notna().any()
                else "",
                "significance": grp["Significance"].mode().iloc[0]
                if grp["Significance"].notna().any()
                else "",
                "google_rating": float(grp["Google_Rating"].median()),
                "ticket_price": float(grp["Ticket_Price"].median()),
                "airport_within_50km": grp["Airport_Within_50km"].mode().iloc[0]
                if grp["Airport_Within_50km"].notna().any()
                else "Unknown",
            }
        )
    return pd.DataFrame(rows)


def match_destinations(asi_names: pd.DataFrame, kaggle_places: pd.DataFrame) -> pd.DataFrame:
    """asi_names columns: monument, asi_circle."""
    by_norm = kaggle_places.drop_duplicates("kaggle_norm")
    kaggle_norm_map = dict(zip(by_norm["kaggle_norm"], by_norm["kaggle_name"]))
    used_kaggle = set()
    rows = []
    ambiguous = []

    for _, rec in asi_names.iterrows():
        official = rec["monument"]
        n = normalize_name(official)
        kaggle_hit = None
        method = "unmatched"
        confidence = 0.0
        kaggle_name = ""
        state = ""
        validated = False

        for knorm, official_target in MANUAL_ALIASES.items():
            if official_target == official and knorm in kaggle_norm_map:
                kaggle_hit = kaggle_norm_map[knorm]
                method = "manual_alias"
                confidence = 1.0
                validated = True
                break

        if kaggle_hit is None and n in kaggle_norm_map:
            kaggle_hit = kaggle_norm_map[n]
            method = "normalized_exact"
            confidence = 0.95
            validated = True

        if kaggle_hit is None:
            candidates = []
            for _, krow in kaggle_places.iterrows():
                ratio = SequenceMatcher(None, n, krow["kaggle_norm"]).ratio()
                if n and (n in krow["kaggle_norm"] or krow["kaggle_norm"] in n):
                    if min(len(n), len(krow["kaggle_norm"])) >= 6:
                        ratio = max(ratio, 0.86)
                if ratio >= 0.88:
                    candidates.append((ratio, krow))
            candidates.sort(key=lambda x: x[0], reverse=True)
            if len(candidates) == 1:
                kaggle_hit = candidates[0][1]["kaggle_name"]
                method = "conservative_fuzzy"
                confidence = float(candidates[0][0])
                validated = False
            elif len(candidates) > 1 and candidates[0][0] >= candidates[1][0] + 0.05:
                kaggle_hit = candidates[0][1]["kaggle_name"]
                method = "conservative_fuzzy"
                confidence = float(candidates[0][0])
                validated = False
            elif len(candidates) > 1:
                ambiguous.append(
                    {
                        "official_name": official,
                        "candidates": [c[1]["kaggle_name"] for c in candidates[:3]],
                    }
                )

        if kaggle_hit:
            meta = kaggle_places.loc[kaggle_places["kaggle_name"] == kaggle_hit].iloc[0]
            kaggle_name = kaggle_hit
            state = meta["state"]
            used_kaggle.add(kaggle_hit)
        rows.append(
            {
                "official_name": official,
                "kaggle_name": kaggle_name,
                "state": state,
                "asi_circle": rec["asi_circle"],
                "match_method": method if kaggle_name else "unmatched",
                "match_confidence": confidence,
                "validated": validated,
            }
        )

    mapping = pd.DataFrame(rows)
    mapping.attrs["ambiguous"] = ambiguous
    return mapping


def write_mapping(path: Path, mapping: pd.DataFrame) -> None:
    mapping.to_csv(path, index=False)
