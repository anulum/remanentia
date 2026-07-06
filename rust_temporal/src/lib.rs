// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust temporal engine (date parsing + vague normalisation)
// Imported crate; idiomatic clippy cleanup tracked in TODO.md (Rust WIP).
#![allow(clippy::manual_pattern_char_comparison, clippy::unnecessary_sort_by)]

use chrono::{Datelike, Duration, NaiveDate};
use pyo3::prelude::*;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

// ── Compiled regexes (initialised once) ─────────────────────────

static DATE_ISO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").unwrap());

static DATE_ENGLISH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:,?\s+(\d{4}))?\b"
    ).unwrap()
});

static DATE_RELATIVE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(yesterday|today|last\s+(?:week|month|year)|this\s+(?:week|month|year))\b")
        .unwrap()
});

static QUANTIFIED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:about\s+|around\s+|roughly\s+|approximately\s+)?(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b").unwrap()
});

static COUPLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\ba\s+couple\s+of\s+(days?|weeks?|months?)\s+ago\b").unwrap()
});

static FEW: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\ba\s+few\s+(days?|weeks?|months?)\s+ago\b").unwrap());

static SEVERAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bseveral\s+(days?|weeks?|months?)\s+ago\b").unwrap());

static WEEKDAY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:last|this\s+past)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b").unwrap()
});

static SIMPLE_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (Regex::new(r"(?i)\byesterday\b").unwrap(), "yesterday"),
        (Regex::new(r"(?i)\btoday\b").unwrap(), "today"),
        (
            Regex::new(r"(?i)\bthe\s+other\s+day\b").unwrap(),
            "other_day",
        ),
        (
            Regex::new(r"(?i)\bnot\s+long\s+ago\b").unwrap(),
            "not_long_ago",
        ),
        (Regex::new(r"(?i)\brecently\b").unwrap(), "recently"),
        (
            Regex::new(r"(?i)\bearlier\s+this\s+week\b").unwrap(),
            "earlier_week",
        ),
        (
            Regex::new(r"(?i)\bearlier\s+this\s+month\b").unwrap(),
            "earlier_month",
        ),
        (Regex::new(r"(?i)\blast\s+week\b").unwrap(), "last_week"),
        (Regex::new(r"(?i)\blast\s+month\b").unwrap(), "last_month"),
        (Regex::new(r"(?i)\blast\s+year\b").unwrap(), "last_year"),
        (Regex::new(r"(?i)\bthis\s+week\b").unwrap(), "this_week"),
        (Regex::new(r"(?i)\bthis\s+month\b").unwrap(), "this_month"),
        (Regex::new(r"(?i)\bthis\s+year\b").unwrap(), "this_year"),
    ]
});

// ── Helpers ─────────────────────────────────────────────────────

fn month_num(name: &str) -> Option<u32> {
    match name.to_lowercase().as_str() {
        "january" | "jan" => Some(1),
        "february" | "feb" => Some(2),
        "march" | "mar" => Some(3),
        "april" | "apr" => Some(4),
        "may" => Some(5),
        "june" | "jun" => Some(6),
        "july" | "jul" => Some(7),
        "august" | "aug" => Some(8),
        "september" | "sep" => Some(9),
        "october" | "oct" => Some(10),
        "november" | "nov" => Some(11),
        "december" | "dec" => Some(12),
        _ => None,
    }
}

fn weekday_num(name: &str) -> Option<u32> {
    match name.to_lowercase().as_str() {
        "monday" => Some(0),
        "tuesday" => Some(1),
        "wednesday" => Some(2),
        "thursday" => Some(3),
        "friday" => Some(4),
        "saturday" => Some(5),
        "sunday" => Some(6),
        _ => None,
    }
}

fn month_delta(d: NaiveDate, months: i32) -> NaiveDate {
    let total_months = d.year() * 12 + d.month() as i32 - 1 + months;
    let y = total_months.div_euclid(12);
    let m = (total_months.rem_euclid(12) + 1) as u32;
    let max_day = days_in_month(y, m);
    NaiveDate::from_ymd_opt(y, m, d.day().min(max_day)).unwrap_or(d)
}

fn days_in_month(year: i32, month: u32) -> u32 {
    NaiveDate::from_ymd_opt(year, month + 1, 1)
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(year + 1, 1, 1).unwrap())
        .pred_opt()
        .unwrap()
        .day()
}

fn resolve_simple(tag: &str, r: NaiveDate) -> Option<NaiveDate> {
    match tag {
        "yesterday" => Some(r - Duration::days(1)),
        "today" => Some(r),
        "other_day" => Some(r - Duration::days(3)),
        "not_long_ago" => Some(r - Duration::days(7)),
        "recently" => Some(r - Duration::days(5)),
        "earlier_week" => {
            Some(r - Duration::days(r.weekday().num_days_from_monday().max(1) as i64))
        }
        "earlier_month" => NaiveDate::from_ymd_opt(r.year(), r.month(), (r.day() / 2).max(1)),
        "last_week" => Some(r - Duration::weeks(1)),
        "last_month" => Some(month_delta(r, -1)),
        "last_year" => NaiveDate::from_ymd_opt(r.year() - 1, r.month(), r.day().min(28)),
        "this_week" => Some(r - Duration::days(r.weekday().num_days_from_monday() as i64)),
        "this_month" => NaiveDate::from_ymd_opt(r.year(), r.month(), 1),
        "this_year" => NaiveDate::from_ymd_opt(r.year(), 1, 1),
        _ => None,
    }
}

// ── Public API ──────────────────────────────────────────────────

/// Parse all dates from text, return sorted unique ISO strings.
/// `reference_date` is ISO format (YYYY-MM-DD) for resolving relative dates.
#[pyfunction]
fn parse_dates(text: &str, reference_date: Option<&str>) -> Vec<String> {
    let ref_date = reference_date
        .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
        .unwrap_or_else(|| chrono::Local::now().date_naive());

    let mut dates = HashSet::new();

    // ISO dates: 2026-03-15
    for cap in DATE_ISO.captures_iter(text) {
        dates.insert(cap[0].to_string());
    }

    // English dates: March 15, 2026
    for cap in DATE_ENGLISH.captures_iter(text) {
        if let Some(month) = month_num(&cap[1]) {
            if let Ok(day) = cap[2].parse::<u32>() {
                let year = cap
                    .get(3)
                    .and_then(|m| m.as_str().parse::<i32>().ok())
                    .unwrap_or(ref_date.year());
                if (1..=31).contains(&day) {
                    if let Some(d) = NaiveDate::from_ymd_opt(year, month, day) {
                        dates.insert(d.format("%Y-%m-%d").to_string());
                    }
                }
            }
        }
    }

    // Relative: yesterday, last week, etc.
    for cap in DATE_RELATIVE.captures_iter(text) {
        let expr = cap[1].to_lowercase();
        if let Some(resolved) = resolve_relative(&expr, ref_date) {
            dates.insert(resolved);
        }
    }

    // Vague expressions: "N days/weeks ago", "a couple of", "a few", "several", "last Monday", simple patterns
    let _ref_str = ref_date.format("%Y-%m-%d").to_string();
    for regex in [&*QUANTIFIED, &*COUPLE, &*FEW, &*SEVERAL, &*WEEKDAY] {
        for cap in regex.captures_iter(text) {
            if let Some(result) = normalise_vague_date_inner(cap.get(0).unwrap().as_str(), ref_date)
            {
                dates.insert(result.0);
            }
        }
    }
    for (pat, _tag) in SIMPLE_PATTERNS.iter() {
        for m in pat.find_iter(text) {
            if let Some(result) = normalise_vague_date_inner(m.as_str(), ref_date) {
                dates.insert(result.0);
            }
        }
    }

    let mut result: Vec<String> = dates.into_iter().collect();
    result.sort();
    result
}

fn resolve_relative(expr: &str, r: NaiveDate) -> Option<String> {
    let d = match expr {
        "yesterday" => r - Duration::days(1),
        "today" => r,
        "last week" => r - Duration::weeks(1),
        "last month" => month_delta(r, -1),
        "last year" => NaiveDate::from_ymd_opt(r.year() - 1, r.month(), r.day().min(28))?,
        "this week" => r - Duration::days(r.weekday().num_days_from_monday() as i64),
        "this month" => NaiveDate::from_ymd_opt(r.year(), r.month(), 1)?,
        "this year" => NaiveDate::from_ymd_opt(r.year(), 1, 1)?,
        _ => return None,
    };
    Some(d.format("%Y-%m-%d").to_string())
}

fn normalise_vague_date_inner(expr: &str, ref_date: NaiveDate) -> Option<(String, f64, String)> {
    let e = expr.trim();

    // "N days/weeks/months/years ago"
    if let Some(cap) = QUANTIFIED.captures(e) {
        let n: i64 = cap[1].parse().ok()?;
        let unit = cap[2].to_lowercase();
        let unit = unit.trim_end_matches('s');
        let target = match unit {
            "day" => ref_date - Duration::days(n),
            "week" => ref_date - Duration::weeks(n),
            "month" => month_delta(ref_date, -(n as i32)),
            "year" => month_delta(ref_date, -(n as i32) * 12),
            _ => return None,
        };
        return Some((target.format("%Y-%m-%d").to_string(), 0.95, "rule".into()));
    }

    // "a couple of ..."
    if let Some(cap) = COUPLE.captures(e) {
        let unit = cap[1].to_lowercase();
        let unit = unit.trim_end_matches('s');
        let target = match unit {
            "day" => ref_date - Duration::days(2),
            "week" => ref_date - Duration::weeks(2),
            "month" => month_delta(ref_date, -2),
            _ => return None,
        };
        return Some((target.format("%Y-%m-%d").to_string(), 0.9, "rule".into()));
    }

    // "a few ..."
    if let Some(cap) = FEW.captures(e) {
        let unit = cap[1].to_lowercase();
        let unit = unit.trim_end_matches('s');
        let target = match unit {
            "day" => ref_date - Duration::days(3),
            "week" => ref_date - Duration::weeks(3),
            "month" => month_delta(ref_date, -3),
            _ => return None,
        };
        return Some((target.format("%Y-%m-%d").to_string(), 0.8, "rule".into()));
    }

    // "several ..."
    if let Some(cap) = SEVERAL.captures(e) {
        let unit = cap[1].to_lowercase();
        let unit = unit.trim_end_matches('s');
        let target = match unit {
            "day" => ref_date - Duration::days(5),
            "week" => ref_date - Duration::weeks(5),
            "month" => month_delta(ref_date, -5),
            _ => return None,
        };
        return Some((target.format("%Y-%m-%d").to_string(), 0.7, "rule".into()));
    }

    // "last Monday"
    if let Some(cap) = WEEKDAY.captures(e) {
        if let Some(day_idx) = weekday_num(&cap[1]) {
            let current = ref_date.weekday().num_days_from_monday();
            let mut days_back = (current as i64 - day_idx as i64).rem_euclid(7);
            if days_back == 0 {
                days_back = 7;
            }
            let target = ref_date - Duration::days(days_back);
            return Some((target.format("%Y-%m-%d").to_string(), 0.95, "rule".into()));
        }
    }

    // Simple fixed patterns
    for (pat, tag) in SIMPLE_PATTERNS.iter() {
        if pat.is_match(e) {
            if let Some(target) = resolve_simple(tag, ref_date) {
                return Some((target.format("%Y-%m-%d").to_string(), 0.85, "rule".into()));
            }
        }
    }

    None
}

/// Rule-based vague date normalisation. Returns (iso_date, confidence, method)
/// or None if no rule matches.
#[pyfunction]
fn normalise_vague_date(expr: &str, reference_date: &str) -> Option<(String, f64, String)> {
    let ref_date = NaiveDate::parse_from_str(reference_date, "%Y-%m-%d").ok()?;
    normalise_vague_date_inner(expr, ref_date)
}

/// Extract temporal events from text: returns list of (date, sentence) tuples.
#[pyfunction]
fn extract_temporal_events(text: &str, reference_date: Option<&str>) -> Vec<(String, String)> {
    let ref_str = reference_date.unwrap_or("2026-01-01");
    let sentences: Vec<&str> = text
        .split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
        .filter(|s| s.len() > 10)
        .collect();

    let mut results = Vec::new();
    for sent in sentences {
        let dates = parse_dates(sent, Some(ref_str));
        for d in dates {
            let trimmed = sent.trim();
            let display = if trimmed.len() > 200 {
                &trimmed[..200]
            } else {
                trimmed
            };
            results.push((d, display.to_string()));
        }
    }
    results
}

// ── TemporalGraph.add_events() kernel ─────────────────────────

/// Build temporal edges for new events added to a graph.
///
/// Args:
///   by_date: {date_str: [event_idx, ...]} — existing date buckets
///   new_events: [(date, text_80)] — events being added
///   start_idx: first index of new events in the global list
///   old_event_texts: {event_idx: text_80} — text for pre-existing events
///
/// Returns: [(source_text, target_text, relation, source_date, target_date), ...]
#[pyfunction]
fn build_temporal_edges(
    by_date: HashMap<String, Vec<usize>>,
    new_events: Vec<(String, String)>,
    start_idx: usize,
    old_event_texts: HashMap<usize, String>,
) -> Vec<(String, String, String, String, String)> {
    let mut edges = Vec::new();
    let mut updated_by_date = by_date;

    // Insert new events into buckets
    let mut new_dates: HashSet<String> = HashSet::new();
    for (i, (date, _text)) in new_events.iter().enumerate() {
        let idx = start_idx + i;
        updated_by_date.entry(date.clone()).or_default().push(idx);
        new_dates.insert(date.clone());
    }

    let get_text = |idx: usize| -> String {
        if idx >= start_idx {
            let local = idx - start_idx;
            if local < new_events.len() {
                return new_events[local].1.clone();
            }
        }
        old_event_texts.get(&idx).cloned().unwrap_or_default()
    };

    // same_day edges within each bucket that received new events
    for d in &new_dates {
        let bucket: &Vec<usize> = match updated_by_date.get(d) {
            Some(b) => b,
            None => continue,
        };
        let new_in_bucket: Vec<usize> = bucket
            .iter()
            .filter(|&&i| i >= start_idx)
            .copied()
            .collect();
        let old_in_bucket: Vec<usize> =
            bucket.iter().filter(|&&i| i < start_idx).copied().collect();

        for &ni in &new_in_bucket {
            for &oi in &old_in_bucket {
                edges.push((
                    get_text(oi),
                    get_text(ni),
                    "same_day".to_string(),
                    d.clone(),
                    d.clone(),
                ));
            }
            for &nj in &new_in_bucket {
                if nj > ni {
                    edges.push((
                        get_text(ni),
                        get_text(nj),
                        "same_day".to_string(),
                        d.clone(),
                        d.clone(),
                    ));
                }
            }
        }
    }

    // before/after edges between adjacent dates
    let mut all_dates: Vec<String> = updated_by_date.keys().cloned().collect();
    all_dates.sort();
    for (idx_d, d) in all_dates.iter().enumerate() {
        if !new_dates.contains(d.as_str()) {
            continue;
        }
        if idx_d + 1 < all_dates.len() {
            let next_d = &all_dates[idx_d + 1];
            let new_here: Vec<usize> = updated_by_date
                .get(d)
                .unwrap()
                .iter()
                .filter(|&&i| i >= start_idx)
                .copied()
                .collect();
            let next_bucket = updated_by_date.get(next_d).unwrap();
            for &ni in new_here.iter().take(3) {
                for &nj in next_bucket.iter().take(3) {
                    edges.push((
                        get_text(ni),
                        get_text(nj),
                        "before".to_string(),
                        d.clone(),
                        next_d.clone(),
                    ));
                }
            }
        }
    }

    edges
}

// ── TemporalGraph.query_temporal() kernel ─────────────────────

static RE_QUERY_TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z0-9]{3,}").unwrap());

/// Score and filter temporal events for query_temporal().
///
/// Args:
///   events: [(date, text, source, paragraph_idx), ...]
///   query: raw query string
///   dates_in_query: parsed dates from the query
///   top_k: max results
///
/// Returns: indices of matching events, sorted by relevance/date
#[pyfunction]
fn score_temporal_query(
    events: Vec<(String, String, String, usize)>,
    query: &str,
    dates_in_query: Vec<String>,
    top_k: usize,
) -> Vec<usize> {
    let q = query.to_lowercase();

    // Filter by date constraints
    let mut filtered_indices: Vec<usize> = Vec::new();
    if !dates_in_query.is_empty() {
        let has_after = q.contains("after") || q.contains("since");
        let has_before = q.contains("before") || q.contains("until");
        for (i, (date, _, _, _)) in events.iter().enumerate() {
            let keep = if has_after {
                date.as_str() >= dates_in_query[0].as_str()
            } else if has_before {
                date.as_str() <= dates_in_query[0].as_str()
            } else if dates_in_query.len() >= 2 {
                let (d1, d2) = if dates_in_query[0] <= dates_in_query[1] {
                    (&dates_in_query[0], &dates_in_query[1])
                } else {
                    (&dates_in_query[1], &dates_in_query[0])
                };
                date.as_str() >= d1.as_str() && date.as_str() <= d2.as_str()
            } else {
                true
            };
            if keep {
                filtered_indices.push(i);
            }
        }
    } else {
        filtered_indices = (0..events.len()).collect();
    }

    // Score by token overlap
    let q_tokens: HashSet<String> = RE_QUERY_TOKEN
        .find_iter(&q)
        .map(|m| m.as_str().to_string())
        .collect();
    let mut scored: Vec<(usize, usize)> = Vec::new();
    for &i in &filtered_indices {
        let ev_lower = events[i].1.to_lowercase();
        let ev_tokens: HashSet<String> = RE_QUERY_TOKEN
            .find_iter(&ev_lower)
            .map(|m| m.as_str().to_string())
            .collect();
        let overlap = q_tokens.intersection(&ev_tokens).count();
        if overlap > 0 || !dates_in_query.is_empty() {
            scored.push((i, overlap));
        }
    }

    // Sort by query intent
    let is_latest = ["latest", "recent", "last", "newest"]
        .iter()
        .any(|w| q.contains(w));
    let is_earliest = ["first", "earliest", "oldest", "original"]
        .iter()
        .any(|w| q.contains(w));

    if is_latest {
        scored.sort_by(|a, b| {
            events[b.0]
                .0
                .cmp(&events[a.0].0)
                .then_with(|| b.1.cmp(&a.1))
        });
    } else if is_earliest {
        scored.sort_by(|a, b| {
            events[a.0]
                .0
                .cmp(&events[b.0].0)
                .then_with(|| b.1.cmp(&a.1))
        });
    } else {
        scored.sort_by(|a, b| b.1.cmp(&a.1));
    }

    scored.iter().take(top_k).map(|(i, _)| *i).collect()
}

// ── UPDE Phase Engine ──────────────────────────────────────────

/// Map an ISO date string to cyclic phase θ ∈ [0, 2π).
/// θ(date) = 2π · day_of_year / 365.25
#[pyfunction]
fn date_to_phase(date_str: &str) -> f64 {
    let trimmed = if date_str.len() >= 10 {
        &date_str[..10]
    } else {
        date_str
    };
    match NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        Ok(d) => 2.0 * std::f64::consts::PI * d.ordinal() as f64 / 365.25,
        Err(_) => 0.0,
    }
}

/// Find events whose cyclic phase resonates with the query date.
/// Returns Vec<(index, resonance_score)> for events where
/// cos(θ_event - θ_query) >= (1 - tolerance).
#[pyfunction]
fn resonance_search(
    event_dates: Vec<String>,
    query_date: &str,
    tolerance: f64,
) -> Vec<(usize, f64)> {
    let query_phase = date_to_phase(query_date);
    let threshold = 1.0 - tolerance;
    let mut results: Vec<(usize, f64)> = Vec::new();
    for (i, date_str) in event_dates.iter().enumerate() {
        let phase = date_to_phase(date_str);
        let resonance = (phase - query_phase).cos();
        if resonance >= threshold {
            results.push((i, resonance));
        }
    }
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ── Module ──────────────────────────────────────────────────────

#[pymodule]
fn remanentia_temporal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_dates, m)?)?;
    m.add_function(wrap_pyfunction!(normalise_vague_date, m)?)?;
    m.add_function(wrap_pyfunction!(extract_temporal_events, m)?)?;
    m.add_function(wrap_pyfunction!(build_temporal_edges, m)?)?;
    m.add_function(wrap_pyfunction!(score_temporal_query, m)?)?;
    m.add_function(wrap_pyfunction!(date_to_phase, m)?)?;
    m.add_function(wrap_pyfunction!(resonance_search, m)?)?;
    Ok(())
}
