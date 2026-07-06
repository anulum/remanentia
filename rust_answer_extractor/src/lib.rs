// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Remanentia — Rust answer extractor (regex-based)

use pyo3::prelude::*;
use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

static RE_DATE_ISO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap());
static RE_DATE_WRITTEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s*\d{4})?").unwrap()
});
static RE_PERCENTAGE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+\.?\d*%").unwrap());
static RE_VERSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"v\d+\.\d+(?:\.\d+)?").unwrap());
static RE_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(\d[\d,]*(?:\.\d+)?)\b").unwrap());
static RE_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+").unwrap());
static RE_SINGLE_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z][a-z]{2,}\b").unwrap());
static RE_WORD3: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w{3,}").unwrap());
static RE_WHEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bwhen\b|\bwhat date\b|\bwhat time\b").unwrap());
static RE_HOW_MANY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bhow many\b|\bhow much\b|\bcount\b|\bnumber of\b").unwrap());
static RE_VERSION_Q: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bversion\b|\brelease\b|\bv\d").unwrap());
static RE_WHO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bwho\b|\bwhose\b|\bwhom\b").unwrap());
static RE_YESNO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(is|are|was|were|did|does|do|has|have|can|will|should)\b").unwrap()
});
static RE_PERCENT_Q: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\bpercent\b|\baccuracy\b|\bscore\b|\brate\b|\b%").unwrap());
static RE_YEAR_ONLY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^20\d{2}$").unwrap());
// Note: Rust regex doesn't support lookbehind; we split manually instead

fn tokenise(q: &str) -> HashSet<String> {
    RE_WORD3
        .find_iter(&q.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect()
}

// UTF-8 safe slice: snap byte indices to nearest char boundary.
// Required because proximity windows (±N bytes) can land mid-character in
// multi-byte text (¥, €, CJK, emoji) and &s[a..b] panics on non-boundaries.
fn snap_floor(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn snap_ceil(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

fn safe_window(s: &str, start: usize, end: usize) -> &str {
    &s[snap_floor(s, start)..snap_ceil(s, end)]
}

fn proximity(text: &str, pos: usize, len: usize, qt: &HashSet<String>) -> usize {
    let s = pos.saturating_sub(80);
    let e = (pos + len + 80).min(text.len());
    let w = safe_window(text, s, e).to_lowercase();
    qt.iter().filter(|t| w.contains(t.as_str())).count()
}

fn best_prox(cands: &[(String, usize)], text: &str, query: &str) -> String {
    if cands.is_empty() {
        return String::new();
    }
    if cands.len() == 1 || query.is_empty() {
        return cands[0].0.clone();
    }
    let qt = tokenise(query);
    cands
        .iter()
        .max_by_key(|(v, p)| proximity(text, *p, v.len(), &qt))
        .unwrap()
        .0
        .clone()
}

fn extract_date(text: &str, query: &str) -> Option<String> {
    let mut c: Vec<(String, usize)> = Vec::new();
    for m in RE_DATE_ISO.find_iter(text) {
        c.push((m.as_str().into(), m.start()));
    }
    for m in RE_DATE_WRITTEN.find_iter(text) {
        c.push((m.as_str().into(), m.start()));
    }
    if c.is_empty() {
        None
    } else {
        Some(best_prox(&c, text, query))
    }
}

fn extract_number(text: &str, query: &str) -> Option<String> {
    let c: Vec<(String, usize)> = RE_NUMBER
        .find_iter(text)
        .filter(|m| !RE_YEAR_ONLY.is_match(m.as_str()) && !m.as_str().is_empty())
        .map(|m| (m.as_str().into(), m.start()))
        .collect();
    if c.is_empty() {
        None
    } else {
        Some(best_prox(&c, text, query))
    }
}

fn extract_version(t: &str) -> Option<String> {
    RE_VERSION.find(t).map(|m| m.as_str().into())
}
fn extract_pct(t: &str) -> Option<String> {
    RE_PERCENTAGE.find(t).map(|m| m.as_str().into())
}

fn extract_name(text: &str, query: &str) -> Option<String> {
    let mut c: Vec<(String, usize)> = RE_NAME
        .find_iter(text)
        .map(|m| (m.as_str().into(), m.start()))
        .collect();
    if c.is_empty() {
        let qt = tokenise(query);
        for m in RE_SINGLE_NAME.find_iter(text) {
            let s = m.start().saturating_sub(50);
            let e = (m.start() + 50).min(text.len());
            let w = safe_window(text, s, e).to_lowercase();
            if qt.iter().any(|t| w.contains(t.as_str())) {
                c.push((m.as_str().into(), m.start()));
            }
        }
    }
    if c.is_empty() {
        None
    } else {
        Some(best_prox(&c, text, query))
    }
}

fn extract_yesno(text: &str, query: &str) -> String {
    let t = text.to_lowercase();
    let neg: &[&str] = &[
        "not ",
        "no ",
        "never ",
        "doesn't ",
        "didn't ",
        "isn't ",
        "wasn't ",
        "weren't ",
        "won't ",
        "wouldn't ",
        "couldn't ",
        "shouldn't ",
        "haven't ",
        "hasn't ",
        "can't ",
        "cannot ",
        "unable ",
        "failed to ",
        "stopped ",
        "quit ",
        "gave up ",
    ];
    let qt = tokenise(query);
    let (mut nh, mut ph) = (0i32, 0i32);
    for tok in &qt {
        if let Some(pos) = t.find(tok.as_str()) {
            let s = pos.saturating_sub(40);
            let e = (pos + 40).min(t.len());
            let w = safe_window(&t, s, e);
            if neg.iter().any(|n| w.contains(n)) {
                nh += 1;
            } else {
                ph += 1;
            }
        }
    }
    if nh > ph {
        "No".into()
    } else {
        "Yes".into()
    }
}

#[pyfunction]
fn extract_answer(query: &str, paragraph: &str) -> Option<String> {
    let q = query.to_lowercase();
    if RE_WHEN.is_match(&q) {
        return extract_date(paragraph, query);
    }
    if RE_HOW_MANY.is_match(&q) {
        return extract_number(paragraph, query);
    }
    if RE_VERSION_Q.is_match(&q) {
        return extract_version(paragraph);
    }
    if RE_WHO.is_match(&q) {
        return extract_name(paragraph, &q);
    }
    if RE_YESNO.is_match(&q) {
        return Some(extract_yesno(paragraph, query));
    }
    if RE_PERCENT_Q.is_match(&q) {
        return extract_pct(paragraph);
    }
    extract_pct(paragraph)
        .or_else(|| extract_version(paragraph))
        .or_else(|| extract_date(paragraph, ""))
        .or_else(|| extract_number(paragraph, ""))
}

fn split_sents(text: &str) -> Vec<&str> {
    let mut sents = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        if (bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?')
            && i + 1 < bytes.len()
            && bytes[i + 1] == b' '
        {
            sents.push(&text[start..=i]);
            start = i + 2;
        }
    }
    if start < text.len() {
        sents.push(&text[start..]);
    }
    sents
}

#[pyfunction]
fn extract_best_sentence(query: &str, paragraph: &str) -> Option<String> {
    let sents = split_sents(paragraph);
    let qt = tokenise(query);
    sents
        .iter()
        .max_by_key(|s| {
            let st: HashSet<String> = RE_WORD3
                .find_iter(&s.to_lowercase())
                .map(|m| m.as_str().into())
                .collect();
            qt.intersection(&st).count()
        })
        .map(|s| s.to_string())
}

fn lcs_ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut prev = vec![0u32; b.len() + 1];
    let mut curr = vec![0u32; b.len() + 1];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }
    2.0 * prev[b.len()] as f64 / (a.len() + b.len()) as f64
}

#[pyfunction]
fn fuzzy_match(candidate: &str, gold: &str, threshold: f64) -> bool {
    let (c, g) = (candidate.trim().to_lowercase(), gold.trim().to_lowercase());
    if c.is_empty() || g.is_empty() {
        return false;
    }
    if c == g || c.contains(&g) || g.contains(&c) {
        return true;
    }
    lcs_ratio(&c, &g) >= threshold
}

#[pyfunction]
fn normalize_number(text: &str) -> Option<String> {
    static RE_NUM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^([\d,]+\.?\d*)%?$").unwrap());
    let t = text.trim().to_lowercase();
    if let Some(c) = RE_NUM.captures(&t) {
        return Some(c[1].replace(',', ""));
    }
    let words: std::collections::HashMap<&str, u64> = [
        ("zero", 0),
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
        ("eleven", 11),
        ("twelve", 12),
        ("thirteen", 13),
        ("fourteen", 14),
        ("fifteen", 15),
        ("sixteen", 16),
        ("seventeen", 17),
        ("eighteen", 18),
        ("nineteen", 19),
        ("twenty", 20),
        ("thirty", 30),
        ("forty", 40),
        ("fifty", 50),
        ("sixty", 60),
        ("seventy", 70),
        ("eighty", 80),
        ("ninety", 90),
        ("hundred", 100),
        ("thousand", 1000),
    ]
    .into_iter()
    .collect();
    let mut cur = 0u64;
    let mut found = false;
    for p in t
        .split(|c: char| c == '-' || c.is_whitespace())
        .filter(|s| !s.is_empty())
    {
        if let Some(&v) = words.get(p) {
            if v >= 100 {
                cur = cur.max(1) * v;
            } else {
                cur += v;
            }
            found = true;
        } else if p == "and" {
            continue;
        } else {
            break;
        }
    }
    if found {
        Some(cur.to_string())
    } else {
        None
    }
}

#[pymodule]
fn remanentia_answer_extractor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_answer, m)?)?;
    m.add_function(wrap_pyfunction!(extract_best_sentence, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy_match, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_number, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_floor_on_boundary_is_identity() {
        let s = "abc";
        assert_eq!(snap_floor(s, 0), 0);
        assert_eq!(snap_floor(s, 1), 1);
        assert_eq!(snap_floor(s, 3), 3);
    }

    #[test]
    fn snap_floor_inside_multibyte_walks_back() {
        // ¥ = 0xC2 0xA5 in UTF-8 (2 bytes) — actually U+00A5 = 0xC2 0xA5
        let s = "a¥b"; // bytes: 61 C2 A5 62
        assert_eq!(s.len(), 4);
        assert!(!s.is_char_boundary(2));
        assert_eq!(snap_floor(s, 2), 1); // walk back to 'a' end
    }

    #[test]
    fn snap_ceil_inside_multibyte_walks_forward() {
        let s = "a¥b";
        assert_eq!(snap_ceil(s, 2), 3); // walk forward past '¥'
    }

    #[test]
    fn snap_handles_4byte_emoji() {
        // 🎯 = U+1F3AF = F0 9F 8E AF (4 bytes)
        let s = "x🎯y";
        assert_eq!(s.len(), 6);
        for i in 2..=4 {
            assert!(!s.is_char_boundary(i));
            let f = snap_floor(s, i);
            let c = snap_ceil(s, i);
            assert!(s.is_char_boundary(f));
            assert!(s.is_char_boundary(c));
        }
    }

    #[test]
    fn safe_window_never_panics_on_yen() {
        let s = "Tokyo costs ¥500 per ride on the Yamanote line around the city";
        for start in 0..s.len() {
            for end in start..=s.len() {
                let _ = safe_window(s, start, end); // must not panic
            }
        }
    }

    #[test]
    fn proximity_handles_cjk_and_yen() {
        let text = "Tokyo 東京 ticket ¥500 subway convenient transport";
        let qt: HashSet<String> = ["tokyo", "ticket", "subway"]
            .into_iter()
            .map(String::from)
            .collect();
        // match at arbitrary positions — must not panic
        let p = proximity(text, 10, 6, &qt);
        assert!(p <= qt.len());
    }

    #[test]
    fn extract_answer_on_tokyo_paragraph_no_panic() {
        // Reproduces the LongMemEval Tokyo panic at byte 777 inside '¥'.
        let para = "Navigating Tokyo's transportation system can be intimidating, but don't worry, I'm here to help! \
        While taxis are convenient, they can be expensive, especially during peak hours. Fortunately, Tokyo has a \
        comprehensive and efficient public transportation system. Subway tickets start at ¥170 and day passes \
        cost around ¥800. The JR Yamanote line is a convenient circular route through the city. Taxis typically \
        charge ¥730 for the first 2 km and then ¥90 per additional 280 m. For longer journeys consider an IC card \
        like Suica or Pasmo which gives a small fare discount on most trains and buses across the metropolitan area \
        so that you do not need to buy individual tickets each time you board a vehicle in Tokyo on your trip.";
        assert!(para.len() > 500);
        // Try several query types — none must panic
        for q in &["how much", "when", "who", "where", "why", "what", "is it"] {
            let _ = extract_answer(q, para);
        }
    }

    #[test]
    fn extract_yesno_handles_multibyte() {
        let text =
            "The answer is no, ¥500 tickets are not available on the Tokyo 東京 subway today";
        let r = extract_yesno(text, "are ¥500 tickets available");
        assert!(r == "Yes" || r == "No"); // must produce an answer, not panic
    }

    #[test]
    fn extract_name_single_fallback_multibyte_safe() {
        let text = "¥ ¥ ¥ Einstein ¥ ¥ ¥ physics genius";
        let r = extract_name(text, "who is the genius");
        assert!(r.is_some() || r.is_none()); // just ensure no panic
    }
}
