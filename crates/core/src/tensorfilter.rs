//! Rich, shared tensor filtering — the one matcher behind the web filter bar and
//! the TUI `--filter` flag / palette. A filter is a compact text query of
//! whitespace-separated **terms** that AND together; each term is a facet
//! predicate that may be **negated** with a leading `!`, and a facet's
//! comma-separated values OR. The text is the canonical form, so it round-trips
//! through the web URL, the `--filter` flag, and the `y` reopen command.
//!
//! Grammar (whitespace-separated; `(…)` groups and `"…"` keep their spaces):
//! ```text
//!   dtype:F16,BF16     dtype is one of these (case-insensitive)
//!   shape:(6,_,42)     shape matches: exact dim, `_` any one dim, `..` any run of dims
//!   dim:4096           *some* axis satisfies this (position-agnostic; ranges too)
//!   rank:>=3           number of dimensions (integer range)
//!   size:1MiB..1GiB    logical byte size (B/KiB/MiB/GiB/TiB, or KB/MB/GB = ×1000)
//!   params:>1M         element count (K/M/B/T = ×1000)
//!   name:re:^model\.   name: bare = substring, `re:` regex, `glob:` glob
//!   shard:00001        source file basename contains this
//!   !size:<4KiB        `!` negates any term
//! ```
//! A bare word (no `facet:` prefix) is a name substring, so `q_proj` just works.

use anyhow::{Result, anyhow, bail};
use regex::Regex;

use crate::tree::TensorInfo;

/// An inclusive-or-exclusive numeric bound.
#[derive(Clone, Copy, Debug)]
struct Bound {
    value: f64,
    inclusive: bool,
}

/// A numeric range with either end open — for `size` / `params` / `rank`.
#[derive(Clone, Debug, Default)]
struct NumRange {
    lo: Option<Bound>,
    hi: Option<Bound>,
}

impl NumRange {
    // The compared values are dims / byte sizes / element counts — integers below 2^53,
    // so they are exact in f64 and an exclusive bound must test equality precisely.
    #[allow(clippy::float_cmp)]
    fn contains(&self, x: f64) -> bool {
        if let Some(b) = self.lo
            && (x < b.value || (!b.inclusive && x == b.value))
        {
            return false;
        }
        if let Some(b) = self.hi
            && (x > b.value || (!b.inclusive && x == b.value))
        {
            return false;
        }
        true
    }
}

/// One element of a shape pattern.
#[derive(Clone, Debug, PartialEq)]
enum DimPat {
    Exact(usize),
    /// `_` — exactly one dimension of any size.
    Any,
    /// `..` — zero or more dimensions of any size (at most one per pattern).
    Rest,
}

#[derive(Clone, Debug)]
struct ShapePattern(Vec<DimPat>);

impl ShapePattern {
    fn matches(&self, shape: &[usize]) -> bool {
        match_dims(&self.0, shape)
    }
}

/// Glob-style dimension match with at most one `Rest` (validated at parse). With
/// no `Rest`, lengths must be equal and every dim matches; a `Rest` soaks up the
/// dimensions between the pattern before it and the suffix after it.
fn match_dims(pats: &[DimPat], shape: &[usize]) -> bool {
    match pats.split_first() {
        None => shape.is_empty(),
        Some((DimPat::Rest, tail)) => {
            // `tail` has no further Rest, so it must match the trailing dims.
            // `Rest` matches the trailing dims: take exactly `tail.len()` from the end.
            shape
                .len()
                .checked_sub(tail.len())
                .and_then(|from| shape.get(from..))
                .is_some_and(|trailing| match_dims(tail, trailing))
        }
        Some((p, tail)) => match shape.split_first() {
            Some((d, rest)) => {
                let ok = match p {
                    DimPat::Exact(n) => n == d,
                    DimPat::Any => true,
                    DimPat::Rest => unreachable!(),
                };
                ok && match_dims(tail, rest)
            }
            None => false,
        },
    }
}

/// How a `name:` term matches.
#[derive(Clone, Debug)]
enum NameMatch {
    /// Case-insensitive substring (the default, and what a bare word uses).
    Substr(String),
    /// One or more globs — brace alternation `{a,b,c}` expands to several, matched
    /// as alternatives (any).
    Glob(Vec<glob::Pattern>),
    Regex(Regex),
}

impl NameMatch {
    fn matches(&self, name: &str) -> bool {
        match self {
            Self::Substr(s) => name.to_lowercase().contains(&s.to_lowercase()),
            Self::Glob(ps) => ps.iter().any(|p| p.matches(name)),
            Self::Regex(r) => r.is_match(name),
        }
    }
}

/// A single facet predicate over a tensor.
#[derive(Clone, Debug)]
enum Predicate {
    Dtype(Vec<String>),
    Shape(Vec<ShapePattern>),
    /// `dim:` — *some* axis satisfies the range (position-agnostic), e.g. any 4096.
    Dim(NumRange),
    Rank(NumRange),
    Size(NumRange),
    Params(NumRange),
    Name(NameMatch),
    Shard(Vec<String>),
}

impl Predicate {
    fn matches(&self, t: &TensorInfo) -> bool {
        match self {
            Self::Dtype(vs) => vs.iter().any(|v| v.eq_ignore_ascii_case(&t.dtype)),
            Self::Shape(ps) => ps.iter().any(|p| p.matches(&t.shape)),
            Self::Dim(r) => t.shape.iter().any(|&d| r.contains(d as f64)),
            Self::Rank(r) => r.contains(t.shape.len() as f64),
            Self::Size(r) => r.contains(t.size_bytes as f64),
            Self::Params(r) => r.contains(t.num_elements as f64),
            Self::Name(m) => m.matches(&t.name),
            Self::Shard(vs) => {
                let base = t
                    .source_path
                    .rsplit(['/', '\\'])
                    .next()
                    .unwrap_or(&t.source_path)
                    .to_lowercase();
                vs.iter().any(|v| base.contains(&v.to_lowercase()))
            }
        }
    }
}

/// One AND-ed term: a predicate, optionally negated.
#[derive(Clone, Debug)]
struct Term {
    negate: bool,
    pred: Predicate,
}

/// A parsed tensor filter: terms AND together (a tensor passes only if every term
/// does). Parse from the text grammar; match against each [`TensorInfo`].
#[derive(Clone, Debug, Default)]
pub struct TensorFilter {
    query: String,
    terms: Vec<Term>,
}

impl TensorFilter {
    /// Parse the text query. An empty / whitespace-only query is the inactive
    /// filter (matches everything). A malformed term is an error (the UI surfaces
    /// it so the user can fix the query).
    pub fn parse(query: &str) -> Result<Self> {
        let mut terms = Vec::new();
        for tok in tokenize(query) {
            terms.push(parse_term(&tok)?);
        }
        Ok(Self {
            query: query.to_string(),
            terms,
        })
    }

    /// The canonical text (for round-tripping through the URL / `y` / `--filter`).
    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    #[must_use]
    pub fn is_active(&self) -> bool {
        !self.terms.is_empty()
    }

    /// Whether `t` passes every term (a negated term passes when its predicate
    /// does *not* match).
    #[must_use]
    pub fn matches(&self, t: &TensorInfo) -> bool {
        self.terms
            .iter()
            .all(|term| term.pred.matches(t) != term.negate)
    }
}

/// Split a query into terms on whitespace, keeping `(…)` groups and `"…"` strings
/// intact (so `shape:(6, _, 42)` and quoted names with spaces survive).
fn tokenize(q: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    let mut quote = false;
    for c in q.chars() {
        match c {
            '"' => quote = !quote, // the quotes group spaces but aren't kept
            '(' if !quote => {
                depth += 1;
                cur.push(c);
            }
            ')' if !quote => {
                depth = (depth - 1).max(0);
                cur.push(c);
            }
            c if c.is_whitespace() && depth == 0 && !quote => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn parse_term(tok: &str) -> Result<Term> {
    let (negate, body) = tok
        .strip_prefix('!')
        .map_or((false, tok), |rest| (true, rest));
    let pred = match body.split_once(':') {
        // A bare word (no facet) is a name substring — `q_proj` just works.
        None => Predicate::Name(NameMatch::Substr(body.to_string())),
        Some((facet, val)) => parse_facet(facet, val)?,
    };
    Ok(Term { negate, pred })
}

fn parse_facet(facet: &str, val: &str) -> Result<Predicate> {
    Ok(match facet {
        "dtype" | "dt" => {
            let vs = split_or(val);
            if vs.is_empty() {
                bail!("dtype: needs a value, e.g. dtype:F16");
            }
            Predicate::Dtype(vs)
        }
        "shape" => {
            let mut pats = Vec::new();
            for group in split_or(val) {
                pats.push(parse_shape(&group)?);
            }
            if pats.is_empty() {
                bail!("shape: needs a pattern, e.g. shape:(6,_,42)");
            }
            Predicate::Shape(pats)
        }
        "dim" => Predicate::Dim(parse_range(val, &[])?),
        "rank" | "ndim" => Predicate::Rank(parse_range(val, &[])?),
        "size" => Predicate::Size(parse_range(val, SIZE_UNITS)?),
        "params" | "param" => Predicate::Params(parse_range(val, COUNT_UNITS)?),
        "shard" | "file" => {
            let vs = split_or(val);
            if vs.is_empty() {
                bail!("shard: needs a value, e.g. shard:00001");
            }
            Predicate::Shard(vs)
        }
        "name" => Predicate::Name(match val.split_once(':') {
            Some(("re", pat)) => {
                NameMatch::Regex(Regex::new(pat).map_err(|e| anyhow!("bad name regex: {e}"))?)
            }
            Some(("glob", pat)) => NameMatch::Glob(
                crate::filter::expand_braces(pat)
                    .iter()
                    .map(|p| glob::Pattern::new(p).map_err(|e| anyhow!("bad name glob: {e}")))
                    .collect::<Result<Vec<_>>>()?,
            ),
            _ => NameMatch::Substr(val.to_string()),
        }),
        other => {
            bail!("unknown filter facet {other:?} (dtype/shape/dim/rank/size/params/name/shard)")
        }
    })
}

/// Split on commas at paren-depth 0 (so `dtype:F16,BF16` and `shape:(2,3),(4,5)`
/// both split into their OR alternatives while dims inside `(…)` stay put).
fn split_or(val: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    for c in val.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth = (depth - 1).max(0);
                cur.push(c);
            }
            ',' if depth == 0 => {
                let t = cur.trim();
                if !t.is_empty() {
                    out.push(t.to_string());
                }
                cur.clear();
            }
            c => cur.push(c),
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

fn parse_shape(group: &str) -> Result<ShapePattern> {
    let inner = group
        .trim()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| anyhow!("shape must be parenthesized, e.g. (6,_,42): got {group:?}"))?;
    let mut dims = Vec::new();
    let mut rests = 0;
    for tok in inner.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue; // tolerate a trailing comma / `()` scalar
        }
        dims.push(match tok {
            "_" => DimPat::Any,
            ".." | "..." | "*" => {
                rests += 1;
                DimPat::Rest
            }
            n => DimPat::Exact(
                n.parse::<usize>()
                    .map_err(|_| anyhow!("shape dim must be a number, `_`, or `..`: {tok:?}"))?,
            ),
        });
    }
    if rests > 1 {
        bail!("a shape pattern may use `..` at most once: {group:?}");
    }
    Ok(ShapePattern(dims))
}

const SIZE_UNITS: &[(&str, f64)] = &[
    ("TiB", 1024.0 * 1024.0 * 1024.0 * 1024.0),
    ("GiB", 1024.0 * 1024.0 * 1024.0),
    ("MiB", 1024.0 * 1024.0),
    ("KiB", 1024.0),
    ("TB", 1e12),
    ("GB", 1e9),
    ("MB", 1e6),
    ("KB", 1e3),
    ("B", 1.0),
];
const COUNT_UNITS: &[(&str, f64)] = &[("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)];

/// Parse a range term: `>N`, `>=N`, `<N`, `<=N`, `A..B`, `A..`, `..B`, or exact `N`
/// (bounds inclusive unless the strict `>`/`<` form is used). `units` maps a suffix
/// to a multiplier (empty for a plain integer like `rank`).
fn parse_range(val: &str, units: &[(&str, f64)]) -> Result<NumRange> {
    let val = val.trim();
    if let Some((a, b)) = val.split_once("..") {
        let lo = a.trim();
        let hi = b.trim();
        return Ok(NumRange {
            lo: if lo.is_empty() {
                None
            } else {
                Some(Bound {
                    value: parse_num(lo, units)?,
                    inclusive: true,
                })
            },
            hi: if hi.is_empty() {
                None
            } else {
                Some(Bound {
                    value: parse_num(hi, units)?,
                    inclusive: true,
                })
            },
        });
    }
    if let Some(rest) = val.strip_prefix(">=") {
        return Ok(NumRange {
            lo: Some(Bound {
                value: parse_num(rest, units)?,
                inclusive: true,
            }),
            hi: None,
        });
    }
    if let Some(rest) = val.strip_prefix("<=") {
        return Ok(NumRange {
            lo: None,
            hi: Some(Bound {
                value: parse_num(rest, units)?,
                inclusive: true,
            }),
        });
    }
    if let Some(rest) = val.strip_prefix('>') {
        return Ok(NumRange {
            lo: Some(Bound {
                value: parse_num(rest, units)?,
                inclusive: false,
            }),
            hi: None,
        });
    }
    if let Some(rest) = val.strip_prefix('<') {
        return Ok(NumRange {
            lo: None,
            hi: Some(Bound {
                value: parse_num(rest, units)?,
                inclusive: false,
            }),
        });
    }
    // Bare number → exact match.
    let n = parse_num(val, units)?;
    Ok(NumRange {
        lo: Some(Bound {
            value: n,
            inclusive: true,
        }),
        hi: Some(Bound {
            value: n,
            inclusive: true,
        }),
    })
}

/// Parse a number with an optional (case-insensitive) unit suffix. `units` is
/// tried longest-first, so `KiB` wins over the bare `B`.
fn parse_num(s: &str, units: &[(&str, f64)]) -> Result<f64> {
    let s = s.trim();
    let up = s.to_ascii_uppercase();
    for (suf, mul) in units {
        let sufu = suf.to_ascii_uppercase();
        if up.len() > sufu.len() && up.ends_with(&sufu) {
            let num = s[..s.len() - suf.len()].trim();
            return num
                .parse::<f64>()
                .map(|n| n * mul)
                .map_err(|_| anyhow!("bad number in {s:?}"));
        }
    }
    s.parse::<f64>().map_err(|_| anyhow!("bad number in {s:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Layout, Storage};

    fn t(name: &str, dtype: &str, shape: &[usize], src: &str) -> TensorInfo {
        let num: usize = shape.iter().product();
        TensorInfo {
            name: name.into(),
            dtype: dtype.into(),
            shape: shape.to_vec(),
            size_bytes: num * 2, // pretend 2 bytes/elem
            num_elements: num,
            storage: Storage::Unknown,
            source_path: src.into(),
            layout: Layout::None,
        }
    }

    fn passes(q: &str, ti: &TensorInfo) -> bool {
        TensorFilter::parse(q).unwrap().matches(ti)
    }

    #[test]
    fn empty_is_inactive_and_matches_all() {
        let f = TensorFilter::parse("   ").unwrap();
        assert!(!f.is_active());
        assert!(f.matches(&t("x", "F16", &[2, 2], "a.safetensors")));
    }

    #[test]
    fn dtype_or_and_case_insensitive() {
        let x = t("w", "BF16", &[4], "s");
        assert!(passes("dtype:f16,bf16", &x));
        assert!(!passes("dtype:f32", &x));
    }

    #[test]
    fn shape_patterns() {
        let x = t("w", "F16", &[6, 128, 42], "s");
        assert!(passes("shape:(6,_,42)", &x));
        assert!(passes("shape:(6,..)", &x)); // starts with 6
        assert!(passes("shape:(..,42)", &x)); // ends with 42
        assert!(passes("shape:(6,_,42),(9,9)", &x)); // OR
        assert!(!passes("shape:(6,_,43)", &x));
        assert!(!passes("shape:(_,_)", &x)); // wrong rank (needs exactly 2)
        assert!(passes("shape:(_,_,_)", &x));
    }

    #[test]
    fn size_and_params_ranges_with_units() {
        // 1000 elems × 2 bytes = 2000 bytes.
        let x = t("w", "F16", &[1000], "s");
        assert!(passes("size:>1KiB", &x)); // 2000 > 1024
        assert!(passes("size:1KiB..4KiB", &x));
        assert!(!passes("size:>4KiB", &x));
        assert!(passes("params:>=1K", &x)); // exactly 1000
        assert!(passes("params:1K..2K", &x));
        assert!(!passes("params:>1K", &x)); // strict, 1000 not > 1000
    }

    #[test]
    fn dim_matches_any_axis() {
        let x = t("w", "F16", &[6, 4096, 42], "s");
        assert!(passes("dim:4096", &x)); // some axis is 4096
        assert!(passes("dim:>=1000", &x));
        assert!(!passes("dim:7", &x));
        assert!(!passes("dim:>5000", &x));
    }

    #[test]
    fn rank_range() {
        let x = t("w", "F16", &[2, 3, 4], "s");
        assert!(passes("rank:3", &x));
        assert!(passes("rank:>=3", &x));
        assert!(!passes("rank:2", &x));
        assert!(!passes("rank:<3", &x));
    }

    #[test]
    fn name_modes_and_bare_word() {
        let x = t("model.layers.0.self_attn.q_proj.weight", "F16", &[4], "s");
        assert!(passes("q_proj", &x)); // bare = substring
        assert!(passes("name:Q_PROJ", &x)); // substring, case-insensitive
        assert!(passes(r"name:re:layers\.\d+\.self_attn", &x));
        assert!(passes("name:glob:*.q_proj.weight", &x));
        assert!(!passes("name:re:^lm_head", &x));
    }

    #[test]
    fn shard_and_negation_and_and() {
        let x = t(
            "w",
            "F16",
            &[6, 42],
            "/ckpt/model-00001-of-00016.safetensors",
        );
        assert!(passes("shard:00001", &x));
        assert!(!passes("shard:00002", &x));
        assert!(passes("!dtype:f32", &x)); // negation
        // facets AND: dtype matches but shard doesn't → overall fail.
        assert!(!passes("dtype:f16 shard:99999", &x));
        assert!(passes("dtype:f16 shape:(6,42) !name:bias", &x));
    }

    #[test]
    fn spaces_in_shape_are_tolerated() {
        let x = t("w", "F16", &[6, 128, 42], "s");
        assert!(passes("shape:(6, _, 42)", &x));
    }

    #[test]
    fn bad_terms_error() {
        assert!(TensorFilter::parse("nope:1").is_err());
        assert!(TensorFilter::parse("size:notanumber").is_err());
        assert!(TensorFilter::parse("shape:6,42").is_err()); // needs parens
        assert!(TensorFilter::parse("name:re:(unclosed").is_err());
        assert!(TensorFilter::parse("shape:(..,..)").is_err()); // two rests
    }
}
