# SPDX-FileCopyrightText: 2026 ANULUM / Fortis Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic data generators for temporal training components C3, C4, C5."""

from __future__ import annotations

import calendar
import json
import random
from datetime import date, timedelta
from pathlib import Path

_RNG = random.Random(42)

# ---------------------------------------------------------------------------
# C4: Date normalisation — synthetic (expr, ref_date, iso_target) triples
# ---------------------------------------------------------------------------

_QUANTIFIED_TEMPLATES = [
    "{n} day ago",
    "{n} days ago",
    "about {n} days ago",
    "{n} week ago",
    "{n} weeks ago",
    "about {n} weeks ago",
    "roughly {n} weeks ago",
    "{n} month ago",
    "{n} months ago",
    "about {n} months ago",
    "around {n} months ago",
    "{n} year ago",
    "{n} years ago",
    "about {n} years ago",
]

_VAGUE_TEMPLATES = [
    ("a few days ago", lambda r: r - timedelta(days=_RNG.randint(2, 5))),
    ("a couple of days ago", lambda r: r - timedelta(days=2)),
    ("several days ago", lambda r: r - timedelta(days=_RNG.randint(4, 8))),
    ("a few weeks ago", lambda r: r - timedelta(weeks=_RNG.randint(2, 4))),
    ("a couple of weeks ago", lambda r: r - timedelta(weeks=2)),
    ("several weeks ago", lambda r: r - timedelta(weeks=_RNG.randint(4, 7))),
    ("a few months ago", lambda r: _month_delta(r, -_RNG.randint(2, 4))),
    ("a couple of months ago", lambda r: _month_delta(r, -2)),
    ("several months ago", lambda r: _month_delta(r, -_RNG.randint(4, 8))),
    ("some time ago", lambda r: r - timedelta(days=_RNG.randint(14, 90))),
    ("some time back", lambda r: r - timedelta(days=_RNG.randint(14, 90))),
    ("not long ago", lambda r: r - timedelta(days=_RNG.randint(2, 14))),
    ("recently", lambda r: r - timedelta(days=_RNG.randint(1, 10))),
    ("the other day", lambda r: r - timedelta(days=_RNG.randint(2, 5))),
    ("earlier this week", lambda r: r - timedelta(days=_RNG.randint(1, min(r.weekday(), 3) + 1))),
    ("earlier this month", lambda r: r.replace(day=max(1, r.day - _RNG.randint(5, min(r.day - 1, 15))))),
    ("late last month", lambda r: _month_delta(r, -1).replace(day=_RNG.randint(20, 28))),
    ("early last month", lambda r: _month_delta(r, -1).replace(day=_RNG.randint(1, 10))),
]

_WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

_SEASONS = {
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
    "fall": (9, 11),
    "winter": (12, 2),
}

_HEDGING_PREFIXES = [
    "I think it was ",
    "I believe it was ",
    "maybe ",
    "probably ",
    "if I remember correctly, ",
    "I'm pretty sure it was ",
]


def _month_delta(d: date, months: int) -> date:
    """Shift date by N months, clamping day."""
    m = d.month + months
    y = d.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    max_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, max_day))


def _random_ref_date() -> date:
    """Random reference date between 2020-01-15 and 2025-12-15."""
    start = date(2020, 1, 15)
    end = date(2025, 12, 15)
    delta = (end - start).days
    return start + timedelta(days=_RNG.randint(0, delta))


def generate_date_normalisation(n: int = 50000) -> list[dict]:
    """Generate synthetic training data for the C4 date normaliser.

    Produces *n* ``(expr, ref_date, target_date)`` triples covering
    quantified relative dates, vague expressions, weekday references,
    seasonal phrases, and compound date arithmetic.

    Args:
        n: Number of samples to generate (default 50 000).

    Returns:
        List of dicts with keys ``expr``, ``ref_date``, ``target_date``
        (all strings, dates in ISO 8601 format).
    """
    samples: list[dict] = []

    # --- Quantified relative ("N days/weeks/months/years ago") ---
    for _ in range(n // 5):
        ref = _random_ref_date()
        tmpl = _RNG.choice(_QUANTIFIED_TEMPLATES)
        if "day" in tmpl:
            num = _RNG.randint(1, 60)
            target = ref - timedelta(days=num)
        elif "week" in tmpl:
            num = _RNG.randint(1, 20)
            target = ref - timedelta(weeks=num)
        elif "month" in tmpl:
            num = _RNG.randint(1, 18)
            target = _month_delta(ref, -num)
        else:  # year
            num = _RNG.randint(1, 5)
            target = _month_delta(ref, -num * 12)
        expr = tmpl.format(n=num)
        if _RNG.random() < 0.15:
            expr = _RNG.choice(_HEDGING_PREFIXES) + expr
        samples.append({
            "expr": expr,
            "ref_date": ref.isoformat(),
            "target_date": target.isoformat(),
        })

    # --- Vague relative ("a few days ago", "recently", etc.) ---
    for _ in range(n // 10):
        ref = _random_ref_date()
        expr_text, resolver = _RNG.choice(_VAGUE_TEMPLATES)
        try:
            target = resolver(ref)
        except ValueError:
            continue
        if _RNG.random() < 0.1:
            expr_text = _RNG.choice(_HEDGING_PREFIXES) + expr_text
        samples.append({
            "expr": expr_text,
            "ref_date": ref.isoformat(),
            "target_date": target.isoformat(),
        })

    # --- "last Monday/Tuesday/..." ---
    for _ in range(n // 10):
        ref = _random_ref_date()
        day_name = _RNG.choice(_WEEKDAYS)
        day_idx = _WEEKDAYS.index(day_name)
        # iso weekday: Monday=0
        days_back = (ref.weekday() - day_idx) % 7
        if days_back == 0:
            days_back = 7
        target = ref - timedelta(days=days_back)
        expr = f"last {day_name}"
        samples.append({
            "expr": expr,
            "ref_date": ref.isoformat(),
            "target_date": target.isoformat(),
        })

    # --- "this past Friday", "this Monday" ---
    for _ in range(n // 10):
        ref = _random_ref_date()
        day_name = _RNG.choice(_WEEKDAYS)
        day_idx = _WEEKDAYS.index(day_name)
        days_back = (ref.weekday() - day_idx) % 7
        if days_back == 0:
            days_back = 7
        target = ref - timedelta(days=days_back)
        expr = _RNG.choice([f"this past {day_name}", f"this {day_name}"])
        samples.append({
            "expr": expr,
            "ref_date": ref.isoformat(),
            "target_date": target.isoformat(),
        })

    # --- Seasonal ("earlier this spring", "late last summer") ---
    for _ in range(n // 25):
        ref = _random_ref_date()
        season = _RNG.choice(list(_SEASONS.keys()))
        start_m, end_m = _SEASONS[season]
        prefix = _RNG.choice(["earlier this", "late last", "early last", "last"])
        if "last" in prefix:
            year = ref.year - 1
        else:
            year = ref.year
        if start_m > end_m:  # winter wraps
            month = _RNG.choice([start_m, 1, end_m])
        else:
            month = _RNG.randint(start_m, end_m)
        if "early" in prefix or "earlier" in prefix:
            day = _RNG.randint(1, 15)
        elif "late" in prefix:
            day = _RNG.randint(15, 28)
        else:
            day = _RNG.randint(1, 28)
        try:
            target = date(year, month, day)
        except ValueError:
            continue
        samples.append({
            "expr": f"{prefix} {season}",
            "ref_date": ref.isoformat(),
            "target_date": target.isoformat(),
        })

    # --- Compound: "N weeks/months before/after <event>" ---
    # For these, the target is relative to a synthetic anchor date
    for _ in range(n // 10):
        ref = _random_ref_date()
        anchor = ref - timedelta(days=_RNG.randint(10, 200))
        unit = _RNG.choice(["days", "weeks", "months"])
        num = _RNG.randint(1, 12)
        direction = _RNG.choice(["before", "after"])
        if unit == "days":
            delta = timedelta(days=num)
        elif unit == "weeks":
            delta = timedelta(weeks=num)
        else:
            delta = timedelta(days=num * 30)  # approximate
        if direction == "before":
            target = anchor - delta
        else:
            target = anchor + delta
        expr = f"{num} {unit} {direction} {anchor.isoformat()}"
        samples.append({
            "expr": expr,
            "ref_date": ref.isoformat(),
            "target_date": target.isoformat(),
        })

    # Pad to exactly n if short
    while len(samples) < n:
        ref = _random_ref_date()
        num = _RNG.randint(1, 30)
        target = ref - timedelta(days=num)
        samples.append({
            "expr": f"{num} days ago",
            "ref_date": ref.isoformat(),
            "target_date": target.isoformat(),
        })

    _RNG.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# C3: Temporal relation classification — synthetic event pairs
# ---------------------------------------------------------------------------

_EVENT_TEMPLATES = [
    "I started {activity}",
    "I finished {activity}",
    "I went to {place}",
    "I bought a {item}",
    "I visited {place}",
    "I had a meeting about {topic}",
    "I attended the {event} event",
    "We discussed {topic} at work",
    "I signed up for {activity}",
    "I completed the {activity} project",
    "I moved to {place}",
    "I received the {item}",
    "I booked {activity}",
    "My {item} broke down",
    "I repaired the {item}",
]

_FILLERS = {
    "activity": [
        "yoga classes", "cooking course", "language lessons",
        "marathon training", "swimming practice", "guitar lessons",
        "painting workshop", "coding bootcamp", "dance classes",
        "meditation retreat", "photography course", "writing workshop",
    ],
    "place": [
        "the dentist", "New York", "the gym", "the library",
        "Paris", "the clinic", "downtown", "the park",
        "San Francisco", "the museum", "Tokyo", "the office",
    ],
    "item": [
        "new laptop", "coffee maker", "bicycle", "car",
        "phone", "camera", "watch", "stand mixer",
        "printer", "headphones", "tablet", "vacuum cleaner",
    ],
    "topic": [
        "budget planning", "project timeline", "team restructuring",
        "product launch", "marketing strategy", "performance review",
        "hiring plan", "office relocation", "client feedback",
    ],
    "event": [
        "charity run", "company picnic", "team building",
        "fundraiser gala", "science fair", "book club",
        "volunteer day", "hackathon", "open house",
    ],
}

_RELATION_LABELS = ["before", "after", "same_day", "overlaps", "contains", "unknown"]


def _random_event_text() -> str:
    """Return a random event description with filled-in placeholders."""
    tmpl = _RNG.choice(_EVENT_TEMPLATES)
    for key, values in _FILLERS.items():
        if "{" + key + "}" in tmpl:
            tmpl = tmpl.replace("{" + key + "}", _RNG.choice(values))
    return tmpl


def generate_temporal_relations(n: int = 15000) -> list[dict]:
    """Generate synthetic training data for the C3 temporal relation classifier.

    Produces *n* event pairs labelled with one of six Allen-interval-inspired
    relations: before, after, same_day, overlaps, contains, unknown.

    Args:
        n: Number of samples (default 15 000).

    Returns:
        List of dicts with keys ``event_a``, ``event_b``, ``date_a``,
        ``date_b``, ``relation``.
    """
    samples: list[dict] = []

    for _ in range(n):
        event_a = _random_event_text()
        event_b = _random_event_text()
        while event_b == event_a:
            event_b = _random_event_text()

        base = _random_ref_date()
        relation_type = _RNG.choice(_RELATION_LABELS)

        if relation_type == "before":
            date_a = base
            date_b = base + timedelta(days=_RNG.randint(1, 180))
        elif relation_type == "after":
            date_a = base + timedelta(days=_RNG.randint(1, 180))
            date_b = base
        elif relation_type == "same_day":
            date_a = base
            date_b = base
        elif relation_type == "overlaps":
            # A starts before B, A ends after B starts
            date_a = base
            date_b = base + timedelta(days=_RNG.randint(1, 5))
        elif relation_type == "contains":
            # A spans a range that includes B
            date_a = base - timedelta(days=_RNG.randint(5, 30))
            date_b = base
        else:  # unknown
            date_a = base
            date_b = base + timedelta(days=_RNG.randint(-365, 365))

        samples.append({
            "event_a": event_a,
            "event_b": event_b,
            "date_a": date_a.isoformat(),
            "date_b": date_b.isoformat(),
            "relation": relation_type,
        })

    _RNG.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# C5: Fact validity — synthetic (fact, type, supersedes, has_boundary)
# ---------------------------------------------------------------------------

_STATE_TEMPLATES = [
    "I live in {place}",
    "I work at {company}",
    "My job title is {title}",
    "I drive a {car}",
    "My phone is a {phone}",
    "I'm dating {person}",
    "I weigh {weight} kg",
    "My salary is {amount} per year",
]

_EVENT_FACT_TEMPLATES = [
    "I went to {place} on {date}",
    "I attended {event} last {day}",
    "I bought a {item} {time}",
    "I visited the {place} {time}",
    "I had {event} yesterday",
]

_PREFERENCE_TEMPLATES = [
    "I love {thing}",
    "I prefer {thing} over {alt}",
    "My favourite {category} is {thing}",
    "I always choose {thing}",
    "I hate {thing}",
    "I really enjoy {thing}",
]

_PLAN_TEMPLATES = [
    "I plan to visit {place} next month",
    "I'm going to start {activity} soon",
    "I will buy a {item} next week",
    "I intend to change {thing}",
    "I'm thinking of moving to {place}",
    "I might start {activity}",
]

_CHANGE_TEMPLATES = [
    ("I started working at {company}", True),
    ("I moved to {place}", True),
    ("I switched to {thing}", True),
    ("I changed my {category}", True),
    ("I quit {activity}", True),
    ("I joined {company}", True),
    ("I bought a new {item}", True),
    ("I upgraded to {thing}", True),
    ("I sold my {item}", True),
    ("I left {company}", True),
    ("I graduated from {place}", True),
    ("I retired from {company}", True),
]

_FACT_FILLERS = {
    "place": ["Berlin", "London", "Tokyo", "New York", "Paris", "Zurich"],
    "company": ["Google", "Microsoft", "ANULUM", "Meta", "Apple", "Tesla"],
    "title": ["engineer", "manager", "designer", "analyst", "director", "CTO"],
    "car": ["Tesla Model 3", "BMW X3", "Toyota Camry", "Honda Civic", "Audi A4"],
    "phone": ["iPhone 15", "Pixel 8", "Galaxy S24", "OnePlus 12"],
    "person": ["Alex", "Jordan", "Sam", "Robin", "Taylor"],
    "weight": ["70", "80", "65", "90", "75"],
    "amount": ["80000", "95000", "120000", "60000"],
    "item": ["laptop", "camera", "bicycle", "watch", "headphones"],
    "thing": ["Python", "running", "coffee", "sushi", "hiking", "photography"],
    "alt": ["Java", "swimming", "tea", "pizza", "cycling", "painting"],
    "category": ["food", "hobby", "music", "colour", "sport"],
    "event": ["a concert", "a meeting", "a workshop", "a conference"],
    "activity": ["yoga", "coding", "painting", "cooking", "gym"],
    "date": ["March 15", "January 3", "December 20", "July 8", "October 1"],
    "day": ["Monday", "Friday", "Saturday", "Wednesday"],
    "time": ["last week", "yesterday", "two days ago", "last month", "recently"],
}


def _fill_template(tmpl: str) -> str:
    """Replace ``{key}`` placeholders in *tmpl* with random filler values."""
    for key, values in _FACT_FILLERS.items():
        if "{" + key + "}" in tmpl:
            tmpl = tmpl.replace("{" + key + "}", _RNG.choice(values))
    return tmpl


def generate_fact_validity(n: int = 6000) -> list[dict]:
    """Generate synthetic training data for the C5 fact validity model.

    Produces *n* labelled facts balanced across state, event, preference,
    plan, and state-change (superseding) categories.

    Args:
        n: Number of samples (default 6 000).

    Returns:
        List of dicts with keys ``text``, ``fact_type``, ``supersedes``
        (bool), ``has_boundary`` (bool).
    """
    samples: list[dict] = []
    per_type = n // 5

    # State facts (non-superseding)
    for _ in range(per_type):
        text = _fill_template(_RNG.choice(_STATE_TEMPLATES))
        samples.append({
            "text": text,
            "fact_type": "state",
            "supersedes": False,
            "has_boundary": False,
        })

    # Event facts
    for _ in range(per_type):
        text = _fill_template(_RNG.choice(_EVENT_FACT_TEMPLATES))
        samples.append({
            "text": text,
            "fact_type": "event",
            "supersedes": False,
            "has_boundary": True,
        })

    # Preference facts
    for _ in range(per_type):
        text = _fill_template(_RNG.choice(_PREFERENCE_TEMPLATES))
        samples.append({
            "text": text,
            "fact_type": "preference",
            "supersedes": False,
            "has_boundary": False,
        })

    # Plan facts
    for _ in range(per_type):
        text = _fill_template(_RNG.choice(_PLAN_TEMPLATES))
        samples.append({
            "text": text,
            "fact_type": "plan",
            "supersedes": False,
            "has_boundary": True,
        })

    # State-change facts (superseding)
    for _ in range(per_type):
        tmpl, _ = _RNG.choice(_CHANGE_TEMPLATES)
        text = _fill_template(tmpl)
        samples.append({
            "text": text,
            "fact_type": "state",
            "supersedes": True,
            "has_boundary": True,
        })

    # Pad to n
    while len(samples) < n:
        text = _fill_template(_RNG.choice(_STATE_TEMPLATES))
        samples.append({
            "text": text,
            "fact_type": "state",
            "supersedes": False,
            "has_boundary": False,
        })

    _RNG.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# CLI entry point: generate all synthetic data to JSONL files
# ---------------------------------------------------------------------------

def save_jsonl(data: list[dict], path: Path) -> None:
    """Write *data* as newline-delimited JSON to *path*, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(data)} samples")


def main() -> None:
    """Generate all synthetic training datasets to training/datasets/."""
    out_dir = Path(__file__).parent / "datasets"

    print("Generating C4 date normalisation data...")
    date_data = generate_date_normalisation(50000)
    save_jsonl(date_data, out_dir / "date_normalisation_synth.jsonl")

    print("Generating C3 temporal relation data...")
    rel_data = generate_temporal_relations(15000)
    save_jsonl(rel_data, out_dir / "temporal_relations_synth.jsonl")

    print("Generating C5 fact validity data...")
    fv_data = generate_fact_validity(6000)
    save_jsonl(fv_data, out_dir / "fact_validity_synth.jsonl")

    print("Done.")


if __name__ == "__main__":
    main()
