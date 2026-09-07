"""Spelling variants for Indian places (Belgaon/Belgaum/Belagavi, etc.)."""

from __future__ import annotations

GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"belgaon", "belgaum", "belagavi"}),
    frozenset({"bangalore", "bengaluru"}),
    frozenset({"bombay", "mumbai"}),
    frozenset({"calcutta", "kolkata"}),
    frozenset({"madras", "chennai"}),
    frozenset({"poona", "pune"}),
    frozenset({"mysore", "mysuru"}),
    frozenset({"gurgaon", "gurugram"}),
    frozenset({"qutub minar", "qutb minar", "kutub minar", "kutb minar"}),
    frozenset({"mamallapuram", "mahabalipuram"}),
    frozenset({"sholapur", "solapur"}),
    frozenset({"trichy", "tiruchirappalli", "tiruchi"}),
)

CANONICAL_LABEL = {
    "belgaon": "Belagavi",
    "belgaum": "Belagavi",
    "belagavi": "Belagavi",
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "poona": "Pune",
    "pune": "Pune",
    "madras": "Chennai",
    "chennai": "Chennai",
    "mysore": "Mysuru",
    "mysuru": "Mysuru",
    "sholapur": "Solapur",
    "solapur": "Solapur",
    "qutub minar": "Qutub Minar",
    "qutb minar": "Qutub Minar",
    "kutub minar": "Qutub Minar",
    "kutb minar": "Qutub Minar",
    "mamallapuram": "Mamallapuram",
    "mahabalipuram": "Mamallapuram",
}


def spellings(name: str | None) -> set[str]:
    key = (name or "").strip().lower()
    if not key:
        return set()
    out = {key}
    for group in GROUPS:
        if key in group or any(g in key or key in g for g in group if len(g) >= 4):
            out |= set(group)
    return out


def canonical_label(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    if key in CANONICAL_LABEL:
        return CANONICAL_LABEL[key]
    for needle, label in CANONICAL_LABEL.items():
        if needle in key:
            return label
    return name.strip()
