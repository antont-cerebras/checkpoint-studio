//! The `diff` subcommand: compare two checkpoints' *structure* and summarize the
//! differences. "Structure" means the tensors (by name, dtype, and shape) and the
//! metadata (by name, value, and value type) — **not** the tensor data/values,
//! which a structural diff never reads (so it stays fast even on multi-GB files).
//!
//! The comparison ([`compare`]) is a pure function over two [`CheckpointSummary`]s
//! and produces a [`DiffReport`]; rendering ([`DiffReport::render`]) and the
//! `diff`-style exit code ([`DiffReport::has_differences`]) are separate so the
//! logic is testable without any I/O.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;

use anyhow::{Context, Result};
use glob::{MatchOptions, Pattern};
use regex::Regex;
use serde_json::Value;

use crate::filter::NameFilter;
use crate::remote::{S3Meta, S3Object};
use crate::sample::{HistBins, HistogramDiff, ValueDiff};
use crate::tree::{MetadataInfo, TensorInfo};
use crate::utils::{format_parameters, format_shape, format_size};

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Rendering options for the diff output.
///
/// Six flags, over clippy's threshold of three. They stay flags because they are genuinely
/// independent switches rather than a state with named alternatives: any combination is
/// valid and each maps 1:1 to a CLI flag. Grouping them into sub-structs to satisfy the
/// lint would add a layer that means nothing — the alternative the lint is really asking
/// for (an enum) only helps when the options are mutually exclusive, and these are not.
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct DiffOpts {
    /// Colorize with ANSI escapes (removed in red, added in green; for a changed
    /// tensor only the dtype/shape token that differs).
    pub color: bool,
    /// Include the metadata section (off under `--only-tensors`).
    pub metadata: bool,
    /// Collapse entries sharing a name template + the same change into one line
    /// with a count and index range (off under `--full`).
    pub group: bool,
    /// Element values were compared (`--values`): show per-change value stats and
    /// note when a change's values weren't compared.
    pub values: bool,
    /// Value distributions were compared (`--histogram`): show a per-change
    /// total-variation-distance summary.
    pub histogram: bool,
    /// A [`TensorFilter`] scoped the diff to a subset of tensors — the metadata
    /// section's "not compared" note names this (rather than `--only-tensors`) so
    /// it's clear why the whole checkpoint wasn't diffed.
    pub filtered: bool,
}

/// A tensor's distribution shift for `diff --histogram`: total variation distance
/// (`0` = same shape, `1` = disjoint) and the bin count it was measured over.
#[derive(Clone, Copy, serde::Serialize)]
pub struct HistShift {
    pub tvd: f64,
    pub bins: usize,
}

/// Per-tensor element / distribution comparison attached to a change — filled by
/// `--values` / `--histogram`, empty for a pure structural diff.
#[derive(Default)]
pub struct TensorExtras {
    pub values: Option<ValueDiff>,
    pub histogram: Option<HistShift>,
}

impl TensorExtras {
    /// Whether the extras themselves indicate a difference (so a structurally
    /// identical tensor still counts as changed).
    fn differ(&self) -> bool {
        self.values.is_some_and(|v| v.differing > 0) || self.histogram.is_some_and(|h| h.tvd > 0.0)
    }
}

/// Wrap `text` in an ANSI colour `code` when `on`, else return it unchanged.
fn paint(text: &str, on: bool, code: &str) -> String {
    if on {
        format!("{code}{text}{RESET}")
    } else {
        text.to_string()
    }
}

/// What to call the two overall-total lines: the checkpoints' totals, or the matched subset's.
///
/// A filter narrows the totals with the tensors ([`TensorFilter::apply`]), which is what makes them
/// describe the report — and makes the bare label wrong, because `size:` above nineteen tensors reads as
/// the checkpoint's size. Both UIs ask this, so neither can label a scoped comparison as a whole one.
/// The wording matches the metadata section's `not compared (filtered subset)`.
#[must_use]
pub fn totals_labels(filtered: bool) -> (&'static str, &'static str) {
    if filtered {
        ("size (filtered subset)", "params (filtered subset)")
    } else {
        ("size", "params")
    }
}

/// A `label: old → new (±abs, ±rel%)` line summarizing an overall total's change
/// (checkpoint size or parameter count), formatting values with `fmt`. Shows
/// "(unchanged)" when equal, and omits the percentage when the old side is zero
/// (no baseline). Coloured like the tensor diff — the old value red, the new value
/// green — while the parenthetical delta is dimmed (a convenience; its sign
/// already shows the direction).
///
/// Public because the web UI's report shows the same line, and showed it without the delta:
/// `451.8 GiB → 32 B` left the reader to work out a change the terminal states. The browser cannot
/// call this, so the agreement is the cross-language parity fixture (`shared/parity/format.json`,
/// generated here by `cargo test --test parity` and checked against the TypeScript in
/// `web/src/lib/parity.test.ts`) — including the `{:.1}` tie rule, which is the part two languages
/// round differently.
#[must_use]
pub fn totals_line(
    label: &str,
    old: usize,
    new: usize,
    color: bool,
    fmt: fn(usize) -> String,
) -> String {
    let parts = totals_parts(old, new, fmt);
    let Some(change) = &parts.change else {
        return format!("{label}: {} (unchanged)", parts.new);
    };
    let old_s = paint(&parts.old, color, RED);
    let new_s = paint(&parts.new, color, GREEN);
    let delta_s = paint(&format!("{}{}", change.delta, change.percent), color, DIM);
    format!("{label}: {old_s} → {new_s} ({delta_s})")
}

/// The pieces [`totals_line`] assembles: the two values, and how they differ.
///
/// Split out so a *screen* can lay them out its own way — the diff screen's header puts a size and a
/// parameter count on one line with no percentage — while the arithmetic and the wording of the delta
/// stay in one place. Two functions computing one delta is the drift the parity fixture exists to
/// stop, and the browser's `totalsParts` is deliberately the same shape.
#[derive(Debug, Clone)]
pub struct TotalsParts {
    /// The old value, formatted.
    pub old: String,
    /// The new value, formatted.
    pub new: String,
    /// `None` when the two sides are equal — there is no change to describe.
    pub change: Option<TotalsChange>,
}

/// How two totals differ.
#[derive(Debug, Clone)]
pub struct TotalsChange {
    /// The signed change, formatted: `-72 B`, `+20`.
    pub delta: String,
    /// `, -43.9%` — with its leading separator, so a renderer that wants it can concatenate and one
    /// that does not can drop it. Empty when the old side is zero: there is no baseline to be a
    /// percentage of.
    pub percent: String,
    /// Whether the new side is the larger one, for colour.
    pub grew: bool,
}

/// The two totals and their difference, as parts.
#[must_use]
pub fn totals_parts(old: usize, new: usize, fmt: fn(usize) -> String) -> TotalsParts {
    let parts = TotalsParts {
        old: fmt(old),
        new: fmt(new),
        change: None,
    };
    if old == new {
        return parts;
    }
    let delta = new as i128 - old as i128;
    let sign = if delta >= 0 { "+" } else { "-" };
    let magnitude = delta.unsigned_abs() as usize;
    let percent = if old == 0 {
        String::new()
    } else {
        format!(", {sign}{:.1}%", magnitude as f64 / old as f64 * 100.0)
    };
    TotalsParts {
        change: Some(TotalsChange {
            delta: format!("{sign}{}", fmt(magnitude)),
            percent,
            grew: delta >= 0,
        }),
        ..parts
    }
}

/// Format an ISO-8601 timestamp (`2026-06-26T14:32:01+00:00`) as
/// `2026-06-26 14:32:01 UTC` for display; an unrecognised format passes through.
fn fmt_timestamp(iso: &str) -> String {
    let Some((date, rest)) = iso.split_once('T') else {
        return iso.to_string();
    };
    let utc = rest.ends_with('Z') || rest.contains("+00:00") || rest.contains("+0000");
    // The time is `HH:MM:SS(.fff)?` before the timezone; strip the tz (`+`/`-`/`Z`)
    // and any fractional seconds.
    let core = rest
        .split(['+', '-', 'Z'])
        .next()
        .unwrap_or(rest)
        .split('.')
        .next()
        .unwrap_or(rest);
    format!("{date} {core}{}", if utc { " UTC" } else { "" })
}

/// Render a changed tensor's `old` and `new` signatures, colouring only what
/// actually differs — the dtype (if it changed) and the shape dimensions that
/// changed — old side red, new green, so the eye lands on the change.
///
/// The shape diff is **squeeze-aware**: `size-1` dimensions are "impedance" (an
/// export/packing artefact — e.g. `(7168, 8192)` vs `(7168, 1, 36864)`), so they
/// don't count as a shape change. The real dimensions are aligned ignoring the
/// singletons — so here only `8192 → 36864` is coloured, `7168` stays plain, and
/// the inserted `1` is dimmed — instead of the whole shape reading as changed just
/// because the rank differs. Only when the *squeezed* ranks differ (a genuine
/// rank change) is the whole shape coloured. No colour when `color` is off.
fn render_change(old: &TensorSig, new: &TensorSig, color: bool) -> (String, String) {
    let dtype_changed = old.dtype != new.dtype;
    let squeeze =
        |shape: &[usize]| -> Vec<usize> { shape.iter().copied().filter(|&d| d != 1).collect() };
    let old_sq = squeeze(&old.shape);
    let new_sq = squeeze(&new.shape);
    // Real dims line up only when the same number survive the squeeze.
    let aligned = old_sq.len() == new_sq.len();
    let one = |sig: &TensorSig, other_sq: &[usize], code: &str| {
        let dtype = paint(&sig.dtype, color && dtype_changed, code);
        let shape = if !color {
            format_shape(&sig.shape)
        } else if !aligned {
            // Genuine rank change (beyond singletons) — dims don't line up.
            paint(&format_shape(&sig.shape), true, code)
        } else {
            // Walk the real (non-1) dims against the other side's real dims; a
            // size-1 dim is dimmed impedance, never a change.
            let mut i = 0;
            let dims: Vec<String> = sig
                .shape
                .iter()
                .map(|&d| {
                    if d == 1 {
                        paint("1", true, DIM)
                    } else {
                        let differs = other_sq.get(i) != Some(&d);
                        i += 1;
                        paint(&d.to_string(), differs, code)
                    }
                })
                .collect();
            format!("({})", dims.join(", "))
        };
        format!("{dtype} {shape}")
    };
    (one(old, &new_sq, RED), one(new, &old_sq, GREEN))
}

/// Split a name into a template (each run of digits → a `{}` placeholder) and the
/// digit-run values, so entries differing only by an index — a layer number, an
/// expert id — share a template and can be collapsed.
#[must_use]
pub fn templatize(name: &str) -> (String, Vec<String>) {
    let mut template = String::new();
    let mut indices = Vec::new();
    let mut digits = String::new();
    for ch in name.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            if !digits.is_empty() {
                template.push_str("{}");
                indices.push(std::mem::take(&mut digits));
            }
            template.push(ch);
        }
    }
    if !digits.is_empty() {
        template.push_str("{}");
        indices.push(digits);
    }
    (template, indices)
}

/// One collapsed run of entries: the shared `template`, the index values seen at
/// each placeholder, the member `count`, and the (identical) change `key`.
struct Group<K> {
    template: String,
    indices: Vec<Vec<String>>,
    count: usize,
    key: K,
}

/// Group `(name, change-key)` entries by `(template, key)` in first-seen order, so
/// only entries with the same structure *and* the same change merge.
fn group_entries<K: Clone + Eq + std::hash::Hash>(items: &[(String, K)]) -> Vec<Group<K>> {
    use std::collections::HashMap;
    let mut index: HashMap<(String, K), usize> = HashMap::new();
    let mut groups: Vec<Group<K>> = Vec::new();
    for (name, key) in items {
        let (template, idx) = templatize(name);
        let gi = if let Some(&i) = index.get(&(template.clone(), key.clone())) {
            i
        } else {
            index.insert((template.clone(), key.clone()), groups.len());
            groups.push(Group {
                template,
                indices: vec![Vec::new(); idx.len()],
                count: 0,
                key: key.clone(),
            });
            groups.len() - 1
        };
        // `gi` indexes the entry just found or pushed, and `indices` has one bucket per
        // placeholder — the same count `idx` was built from.
        let Some(g) = groups.get_mut(gi) else {
            continue;
        };
        g.count += 1;
        for (bucket, v) in g.indices.iter_mut().zip(idx) {
            bucket.push(v);
        }
    }
    groups
}

/// Collapse tensor `names` into their index-templated schema: names sharing a
/// template (each run of digits — a layer number, an expert id — becomes a range
/// placeholder) merge into one `(display_name, count)`, e.g.
/// `model.layers.{0-47}.…experts.{0-3}.down_proj.weight` → count 192. Ordered by
/// first appearance (alphabetical when `names` is sorted). Used to summarize which
/// tensors a `diff` filter matched.
#[must_use]
pub fn name_schema(names: &[&str]) -> Vec<(String, usize)> {
    let items: Vec<(String, ())> = names.iter().map(|n| ((*n).to_string(), ())).collect();
    group_entries(&items)
        .into_iter()
        .map(|g| (display_name(&g.template, &g.indices), g.count))
        .collect()
}

/// A family of tensors sharing an index template (layer / expert numbers → range
/// placeholders; see [`name_schema`]), with rolled-up stats — for a compact
/// per-layer / per-expert listing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorFamily {
    /// Templated display name, e.g. `model.layers.{0-47}.…experts.{0-3}.down_proj.weight`.
    pub name: String,
    /// How many tensors collapsed into this family.
    pub count: usize,
    /// The dtype, when uniform across the family (else `None` — "varies").
    pub dtype: Option<String>,
    /// The shape, when uniform across the family (else `None`).
    pub shape: Option<Vec<usize>>,
    /// Total parameters across the family.
    pub params: usize,
    /// Total logical bytes across the family.
    pub size_bytes: usize,
}

/// Collapse `tensors` into families by index template (each digit run — a layer
/// number, an expert id — becomes a range), rolling up the member count, total
/// params / bytes, and the dtype + shape when uniform. First-appearance order
/// (alphabetical when the input is sorted). A compact "what's in here, per layer /
/// per expert" summary, mirroring how `diff` collapses its entries.
#[must_use]
pub fn tensor_families(tensors: &[TensorInfo]) -> Vec<TensorFamily> {
    use std::collections::HashMap;
    struct Agg {
        template: String,
        indices: Vec<Vec<String>>,
        count: usize,
        params: usize,
        size: usize,
        dtype: Option<String>,
        shape: Option<Vec<usize>>,
    }
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut aggs: Vec<Agg> = Vec::new();
    for t in tensors {
        let (template, idx) = templatize(&t.name);
        let np = idx.len();
        let gi = *index.entry(template.clone()).or_insert_with(|| {
            aggs.push(Agg {
                template,
                indices: vec![Vec::new(); np],
                count: 0,
                params: 0,
                size: 0,
                dtype: Some(t.dtype.clone()),
                shape: Some(t.shape.clone()),
            });
            aggs.len() - 1
        });
        // `gi` indexes the entry just found or pushed.
        let Some(a) = aggs.get_mut(gi) else {
            continue;
        };
        a.count += 1;
        a.params += t.num_elements;
        a.size += t.size_bytes;
        for (p, v) in idx.into_iter().enumerate() {
            if let Some(slot) = a.indices.get_mut(p) {
                slot.push(v);
            }
        }
        if a.dtype.as_deref() != Some(t.dtype.as_str()) {
            a.dtype = None;
        }
        if a.shape.as_deref() != Some(t.shape.as_slice()) {
            a.shape = None;
        }
    }
    aggs.into_iter()
        .map(|a| TensorFamily {
            name: display_name(&a.template, &a.indices),
            count: a.count,
            dtype: a.dtype,
            shape: a.shape,
            params: a.params,
            size_bytes: a.size,
        })
        .collect()
}

/// Render each entry as its own group (no collapsing) — for `--full`. Each
/// placeholder gets its single value back, so the displayed name is the original.
fn singletons<K: Clone>(items: &[(String, K)]) -> Vec<Group<K>> {
    items
        .iter()
        .map(|(name, key)| {
            let (template, idx) = templatize(name);
            Group {
                template,
                indices: idx.into_iter().map(|v| vec![v]).collect(),
                count: 1,
                key: key.clone(),
            }
        })
        .collect()
}

/// Reconstruct a group's display name: fill each `{}` with its index — the single
/// value when constant across the group, else `{lo-hi,…}` for the range.
fn display_name(template: &str, indices: &[Vec<String>]) -> String {
    let mut out = String::new();
    for (i, part) in template.split("{}").enumerate() {
        out.push_str(part);
        if let Some(vals) = indices.get(i) {
            out.push_str(&summarize_indices(vals));
        }
    }
    out
}

/// One placeholder's index values as a compact string: the lone value when they're
/// all equal, else `{0-47}` / `{0-3,5}` (integer ranges) or `{a,b}` (sorted list).
///
/// Shared with [`crate::difftree::fold_families`], which labels a folded run of layers the same way —
/// one wording for "these indices", wherever a family is collapsed.
pub(crate) fn summarize_indices(values: &[String]) -> String {
    use std::collections::BTreeSet;
    let distinct: BTreeSet<&str> = values.iter().map(String::as_str).collect();
    // One distinct value means at least one value.
    if distinct.len() == 1 {
        return values.first().cloned().unwrap_or_default();
    }
    distinct
        .iter()
        .map(|s| s.parse::<i64>().ok())
        .collect::<Option<Vec<i64>>>()
        .map_or_else(
            || format!("{{{}}}", distinct.into_iter().collect::<Vec<_>>().join(",")),
            |mut nums| {
                nums.sort_unstable();
                format!("{{{}}}", compact_int_ranges(&nums))
            },
        )
}

/// Collapse a sorted integer list into comma-separated runs: `[0,1,2,5]` → `0-2,5`.
fn compact_int_ranges(sorted: &[i64]) -> String {
    let mut out = String::new();
    let mut rest = sorted;
    while let Some((&start, mut tail)) = rest.split_first() {
        // Extend the run while each next value is one more than the last.
        let mut end = start;
        while let Some((&next, more)) = tail.split_first() {
            if next != end + 1 {
                break;
            }
            end = next;
            tail = more;
        }
        if !out.is_empty() {
            out.push(',');
        }
        if end == start {
            let _ = write!(out, "{start}");
        } else {
            let _ = write!(out, "{start}-{end}");
        }
        rest = tail;
    }
    out
}

/// The `  (×N)` suffix for a collapsed group (empty for a single entry).
fn count_suffix(count: usize) -> String {
    if count > 1 {
        format!("  (×{count})")
    } else {
        String::new()
    }
}

/// A collapsed run of changed tensors sharing a template and the same dtype/shape
/// change, with their value comparisons aggregated across the run.
struct ChangedGroup {
    template: String,
    indices: Vec<Vec<String>>,
    /// The names this row stands for. Kept so a per-name fact — the fold count — can be reported for a
    /// grouped row only when every member agrees about it (see `DiffReport::fold_note`).
    names: Vec<String>,
    count: usize,
    old: TensorSig,
    new: TensorSig,
    values: Option<ValueDiff>,
    /// Each member's histogram TVD (empty when `--histogram` wasn't run), plus the
    /// shared bin count — so the group can report max & mean shift.
    hist_tvds: Vec<f64>,
    hist_bins: usize,
}

/// Combine two value comparisons: counts sum, `max_abs` is the max, `mean_abs` is
/// the element-weighted mean — so a group's aggregate reads like one comparison.
fn merge_values(acc: Option<ValueDiff>, next: Option<ValueDiff>) -> Option<ValueDiff> {
    match (acc, next) {
        (None, x) | (x, None) => x,
        (Some(a), Some(b)) => {
            let elements = a.elements + b.elements;
            let mean_abs = if elements > 0 {
                b.mean_abs
                    .mul_add(b.elements as f64, a.mean_abs * a.elements as f64)
                    / elements as f64
            } else {
                0.0
            };
            Some(ValueDiff {
                elements,
                differing: a.differing + b.differing,
                max_abs: a.max_abs.max(b.max_abs),
                mean_abs,
                nonfinite_mismatch: a.nonfinite_mismatch + b.nonfinite_mismatch,
            })
        }
    }
}

/// Group changed tensors by `(template, old_sig, new_sig)` in first-seen order
/// (aggregating their value comparisons), or one group per tensor when `!group`.
fn group_changed(items: &[TensorChange], group: bool) -> Vec<ChangedGroup> {
    use std::collections::HashMap;
    let mut index: HashMap<(String, TensorSig, TensorSig), usize> = HashMap::new();
    let mut groups: Vec<ChangedGroup> = Vec::new();
    for c in items {
        let (template, idx) = templatize(&c.name);
        // `!group` keeps every entry distinct: key on the unique name too.
        let bucket = if group {
            (template.clone(), c.old.clone(), c.new.clone())
        } else {
            (c.name.clone(), c.old.clone(), c.new.clone())
        };
        let gi = if let Some(&i) = index.get(&bucket) {
            i
        } else {
            index.insert(bucket, groups.len());
            groups.push(ChangedGroup {
                template,
                indices: vec![Vec::new(); idx.len()],
                names: Vec::new(),
                count: 0,
                old: c.old.clone(),
                new: c.new.clone(),
                values: None,
                hist_tvds: Vec::new(),
                hist_bins: 0,
            });
            groups.len() - 1
        };
        // `gi` indexes the entry just found or pushed, and `indices` has one bucket per
        // placeholder — the same count `idx` was built from.
        let Some(g) = groups.get_mut(gi) else {
            continue;
        };
        g.count += 1;
        g.names.push(c.name.clone());
        for (bucket, v) in g.indices.iter_mut().zip(idx) {
            bucket.push(v);
        }
        g.values = merge_values(g.values, c.values);
        if let Some(h) = c.histogram {
            g.hist_tvds.push(h.tvd);
            g.hist_bins = h.bins;
        }
    }
    groups
}

/// A tensor's compared identity: dtype + shape. Two tensors with the same name
/// are "changed" when these differ (data bytes are not part of the comparison).
#[derive(Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub struct TensorSig {
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl TensorSig {
    /// The signature of a loaded tensor.
    #[must_use]
    pub fn of(t: &TensorInfo) -> Self {
        Self {
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
        }
    }

    fn render(&self) -> String {
        format!("{} {}", self.dtype, format_shape(&self.shape))
    }
}

/// The element-value comparison outcome for the focused (`--tensor`) diff, when
/// the tensor exists on both sides.
pub enum ValueCmp {
    /// All elements are equal (bit-equal, or NaN in the same slots).
    Identical,
    /// Some elements differ; carries the diff statistics.
    Differ(ValueDiff),
    /// Values weren't compared — the reason (e.g. "shapes differ", an unreadable
    /// dtype, or an I/O error).
    Skipped(String),
}

/// A metadata entry's compared value: its string value + declared type.
#[derive(Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub struct MetaVal {
    pub value: String,
    pub value_type: String,
}

/// What one entry of a summary contributes to its totals, and how many tensors it stands for.
#[derive(Clone, Copy, serde::Serialize)]
pub struct Footprint {
    /// Its stored size, as the source reports it — not re-derived from dtype × shape, which is wrong
    /// for a packed weight.
    pub bytes: usize,
    /// Its element count.
    pub params: usize,
    /// How many tensors this entry stands for: `1` ordinarily, more when an alignment **folded** several
    /// onto one name — 256 per-expert tensors onto the one fused tensor that holds them
    /// ([`OnCollision::Fold`]). The point of counting rather than collapsing silently: a row reading
    /// `×256 → ×1` is the answer to "did the conversion keep everything", and a row reading `×255` is a
    /// missing expert.
    pub parts: usize,
}

impl Default for Footprint {
    /// One tensor, contributing nothing — the identity for a fold.
    fn default() -> Self {
        Self {
            bytes: 0,
            params: 0,
            parts: 1,
        }
    }
}

/// One checkpoint reduced to what the structural diff compares. Every map is
/// keyed by name and ordered, so the diff output is deterministic and alphabetical.
#[derive(serde::Serialize)]
pub struct CheckpointSummary {
    pub tensors: BTreeMap<String, TensorSig>,
    pub metadata: BTreeMap<String, MetaVal>,
    /// Per tensor, keyed exactly like `tensors`.
    ///
    /// **Kept per name rather than pre-summed, so narrowing the comparison narrows the totals.** A
    /// `--name` filter retains a subset of `tensors`; a stored sum stayed behind and described the whole
    /// checkpoints, so a report about nineteen tensors was headed `size: 1966.5 GiB → 451.8 GiB`. The
    /// totals are now [`Self::total_bytes`] / [`Self::total_params`], derived from this map, and
    /// [`Self::retain_tensors`] is the one way to drop tensors — which is what keeps the two in step.
    pub footprints: BTreeMap<String, Footprint>,
}

impl CheckpointSummary {
    /// The summed stored size of the tensors **in scope**.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.footprints.values().map(|f| f.bytes).sum()
    }

    /// The summed element count of the tensors **in scope**.
    #[must_use]
    pub fn total_params(&self) -> usize {
        self.footprints.values().map(|f| f.params).sum()
    }

    /// Names that stand for more than one tensor, and how many — see [`Footprint::parts`].
    #[must_use]
    pub fn folds(&self) -> BTreeMap<String, usize> {
        self.footprints
            .iter()
            .filter(|(_, f)| f.parts > 1)
            .map(|(n, f)| (n.clone(), f.parts))
            .collect()
    }

    /// Keep only the tensors whose name `keep` accepts — signatures and footprints together.
    ///
    /// The only way to narrow a summary, deliberately: `tensors.retain(…)` on its own would leave the
    /// footprints (and so the totals) describing tensors the report no longer mentions.
    pub fn retain_tensors(&mut self, keep: impl Fn(&str) -> bool) {
        self.tensors.retain(|name, _| keep(name));
        self.footprints.retain(|name, _| keep(name));
    }

    /// Re-root this summary at `prefix`: drop the tensors outside that subtree and key the rest by
    /// their sub-path. Answers how many were kept, so a prefix matching nothing can be reported as the
    /// typo it probably is.
    ///
    /// This is what `OLD#language_model` means — a **scope change, not a rename**: the names move so
    /// that `language_model.model.layers.0.w` lines up with the other side's `model.layers.0.w`, and
    /// the totals then describe the subtree rather than the checkpoint. Siblings (`vision_tower.…`)
    /// are out of scope, not "removed".
    ///
    /// Metadata is left alone: an entry's key is not a tensor path, so no prefix applies to it.
    pub fn reroot(&mut self, prefix: &str) -> usize {
        let prefix = format!("{}.", prefix.trim_end_matches('.'));
        let inside = |name: &str| name.strip_prefix(&prefix).map(str::to_string);
        self.tensors = std::mem::take(&mut self.tensors)
            .into_iter()
            .filter_map(|(name, sig)| inside(&name).map(|sub| (sub, sig)))
            .collect();
        self.footprints = std::mem::take(&mut self.footprints)
            .into_iter()
            .filter_map(|(name, f)| inside(&name).map(|sub| (sub, f)))
            .collect();
        self.tensors.len()
    }

    /// Reduce a freshly-loaded checkpoint to its comparable structure. A sharded
    /// checkpoint can list a name in more than one file; the last one wins (the
    /// same name+shape is expected across shards, so this only matters if they
    /// genuinely disagree, which a diff can't meaningfully represent anyway).
    #[must_use]
    pub fn from_loaded(tensors: &[TensorInfo], metadata: &[MetadataInfo]) -> Self {
        let mut t = BTreeMap::new();
        // Footprints are keyed and de-duplicated exactly like `t` (last-wins), so totals are over the
        // deduped set rather than counting a shared name once per shard.
        let mut footprints: BTreeMap<String, Footprint> = BTreeMap::new();
        for ti in tensors {
            t.insert(ti.name.clone(), TensorSig::of(ti));
            footprints.insert(
                ti.name.clone(),
                Footprint {
                    bytes: ti.size_bytes,
                    params: ti.num_elements,
                    parts: 1,
                },
            );
        }
        let mut m = BTreeMap::new();
        for mi in metadata {
            m.insert(
                mi.name.clone(),
                MetaVal {
                    value: mi.value.clone(),
                    value_type: mi.value_type.clone(),
                },
            );
        }
        Self {
            tensors: t,
            metadata: m,
            footprints,
        }
    }
}

/// The rules that line an **unfused** checkpoint up with its **fused** counterpart.
///
/// Two checkpoints can hold the same model in two layouts, and then a structural diff of them is
/// useless in a specific way: they share no tensor name, so every tensor of both sides is one-sided and
/// the difference count is their sum. 80,107 against 933, "nothing lines up" — a true statement that
/// answers nothing. What the reader wants to know is whether the conversion kept everything.
///
/// Two kinds of difference are in the way, and these rules are exactly those two:
///
/// * **The expert index.** An unfused checkpoint stores one tensor per expert
///   (`…experts.37.w2.weight`); a fused one stores all of them in a single tensor
///   (`…experts.down_proj.weight`, whose leading dimension is the expert count). Dropping the index is
///   what makes those correspond — as a *fold*, so the row reads `×256 → ×1` rather than 255 removals.
/// * **The naming conventions.** `w1`/`w2`/`w3` (Mixtral-style) against `gate_proj`/`down_proj`/
///   `up_proj` (HF-style) against the fused `gate_up_proj`/`down_proj`; a `.weight.qscale` suffix
///   against `.qscale`; three attention projections against one `qkv_proj`;
///   `e_score_correction_bias` against `gate.bias`; a `language_model.` prefix, or none.
///
/// **Applied to both sides**, because each rule is a no-op on a checkpoint already in the fused
/// layout — `\.experts\.\d+\.` does not match it, and it has no `w2` to rename. So "align these two"
/// needs no answer to "which one is which".
///
/// Returned as data, and printed by the `diff` subcommand when it applies them, because a
/// transformation you cannot see is one you cannot check: the pairs below are exactly what `--map`
/// takes, so a checkpoint these rules mis-align can be aligned by hand from here.
#[must_use]
pub fn fused_layout_rules() -> Vec<(String, String)> {
    // Order matters: the more specific name goes first, since each rule rewrites the running name.
    [
        // A quantized weight's sidecars, whose suffix moves: `…w2.weight.qscale` → `…w2.qscale`.
        (r"\.weight\.qscale$", ".qscale"),
        (r"\.weight\.codebook$", ".codebook"),
        // The expert index. Folded: several tensors, one fused counterpart.
        (r"\.experts\.\d+\.", ".experts."),
        // The two projections a fused checkpoint concatenates, in both naming schemes.
        (r"\.w1__w3\.", ".gate_up_proj."),
        (r"\.gate_proj__up_proj\.", ".gate_up_proj."),
        (r"\.w1\.", ".gate_up_proj."),
        (r"\.w3\.", ".gate_up_proj."),
        (r"\.gate_proj\.", ".gate_up_proj."),
        (r"\.up_proj\.", ".gate_up_proj."),
        // …and the one it keeps to itself.
        (r"\.w2\.", ".down_proj."),
        // The MoE container, which the two schemes name differently.
        (r"\.mlp\.experts\.", ".block_sparse_moe.experts."),
        (r"\.mlp\.gate\.", ".block_sparse_moe.gate."),
        // The router's bias, ditto.
        (
            r"\.block_sparse_moe\.e_score_correction_bias$",
            ".block_sparse_moe.gate.bias",
        ),
        // Attention: three projections against one.
        (r"\.self_attn\.q_proj\.", ".self_attn.qkv_proj."),
        (r"\.self_attn\.k_proj\.", ".self_attn.qkv_proj."),
        (r"\.self_attn\.v_proj\.", ".self_attn.qkv_proj."),
        // A multimodal wrapper prefixes the language model; the fused side does not.
        (r"^language_model\.model\.", "model."),
        (r"^language_model\.lm_head\.", "lm_head."),
    ]
    .into_iter()
    .map(|(pat, rep)| (pat.to_string(), rep.to_string()))
    .collect()
}

/// What it means for several names to land on one, when renaming.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OnCollision {
    /// A rule was too broad: report it, and keep one of the tensors. The `--map` default, because a
    /// tensor silently leaving a comparison is the failure worth naming.
    Warn,
    /// Several tensors *are* one tensor on the other side — the unfused-to-fused case. Merge them into
    /// one entry that counts its parts and sums their footprints.
    Fold,
}

/// An ordered list of regex rewrite rules that rename one checkpoint's tensor
/// names into the other's naming scheme *before* the structural diff, so a tensor
/// that is "the same tensor" under a different name lines up (and shows as changed
/// or unchanged) instead of appearing as a removed/added pair. Rules apply in
/// order, each a `replace_all` on the running name (sed-style), with `$1` / `$name`
/// capture references in the replacement — so one rule can rewrite a shared
/// substring across every layer at once (e.g. `\.mlp\.experts\.` for all layers).
#[derive(Default)]
pub struct NameMap {
    rules: Vec<(Regex, String)>,
}

impl NameMap {
    /// No rules — [`NameMap::map`] returns names unchanged and [`remap_summary`]
    /// is a no-op.
    ///
    /// [`remap_summary`]: NameMap::remap_summary
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The number of rewrite rules (for a "applied N rule(s)" note).
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Compile `(pattern, replacement)` pairs into rules — shared by the CLI /
    /// plain-text form ([`parse_rules`]) and the JSON form. An invalid regex is an
    /// error naming the offending pattern.
    ///
    /// [`parse_rules`]: NameMap::parse_rules
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, String)>) -> Result<Self> {
        let mut rules = Vec::new();
        for (pat, rep) in pairs {
            let re = Regex::new(&pat).with_context(|| format!("invalid --map regex {pat:?}"))?;
            rules.push((re, rep));
        }
        Ok(Self { rules })
    }

    /// Parse `PATTERN=>REPLACEMENT` lines (the CLI `--map` value and the plain-text
    /// `--map-from` file): split on the first `=>`, trim both sides (tensor names
    /// carry no whitespace, so rules can be column-aligned), and skip blank lines
    /// and `#` comments — mirroring the `--names-from` file convention.
    pub fn parse_rules<'a>(
        lines: impl IntoIterator<Item = &'a str>,
    ) -> Result<Vec<(String, String)>> {
        let mut pairs = Vec::new();
        for line in lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (pat, rep) = line
                .split_once("=>")
                .with_context(|| format!("map rule missing `=>` separator: {line:?}"))?;
            pairs.push((pat.trim().to_string(), rep.trim().to_string()));
        }
        Ok(pairs)
    }

    /// Rewrite `name` through every rule in order. `replace_all` only allocates on
    /// a match, so a name no rule touches passes through cheaply.
    #[must_use]
    pub fn map<'n>(&self, name: &'n str) -> Cow<'n, str> {
        let mut cur = Cow::Borrowed(name);
        for (re, rep) in &self.rules {
            if let Cow::Owned(s) = re.replace_all(&cur, rep.as_str()) {
                cur = Cow::Owned(s);
            }
        }
        cur
    }

    /// Rewrite `sum`'s tensor names through the rules in place. Metadata names are
    /// left untouched (renames are a tensor-naming concern). Returns any target
    /// names that two distinct source names collided onto — the map keeps the last
    /// (a `BTreeMap` insert), so the caller can warn that a rule is too broad.
    pub fn remap_summary(&self, sum: &mut CheckpointSummary) -> Vec<String> {
        self.remap_summary_with(sum, OnCollision::Warn)
    }

    /// [`Self::remap_summary`], choosing what several names landing on one means.
    ///
    /// For a hand-written `--map` it is a mistake ([`OnCollision::Warn`]): two unrelated tensors on one
    /// name means a rule was too broad, and a tensor has quietly left the comparison. For an alignment
    /// between an unfused and a fused checkpoint it is the *point* ([`OnCollision::Fold`]): 256
    /// per-expert tensors correspond to the one fused tensor that holds them, and the useful row says
    /// so — `×256 → ×1` — rather than dropping 255 of them and calling it a collision.
    pub fn remap_summary_with(
        &self,
        sum: &mut CheckpointSummary,
        on_collision: OnCollision,
    ) -> Vec<String> {
        if self.rules.is_empty() {
            return Vec::new();
        }
        // Classify from the original names *before* they collapse into `mapped`,
        // reusing the same collision detection the in-place rename relies on.
        let collisions = match on_collision {
            OnCollision::Warn => {
                self.plan_renames(sum.tensors.keys().map(String::as_str))
                    .collisions
            }
            // Folding is what was asked for, so there is nothing to warn about.
            OnCollision::Fold => Vec::new(),
        };
        let mut mapped: BTreeMap<String, TensorSig> = BTreeMap::new();
        for (name, sig) in std::mem::take(&mut sum.tensors) {
            // The first signature wins for a fold: the parts of one fused tensor are the same shape as
            // each other, and which of 256 identical shapes is shown does not matter. (Where they are
            // *not* identical the count still tells you how many there were.)
            mapped.entry(self.map(&name).into_owned()).or_insert(sig);
        }
        sum.tensors = mapped;
        // The footprints move with the names, or the totals would be keyed by names the summary no
        // longer has. Merged rather than overwritten under `Fold`: the folded entry stands for all of
        // them, so it carries their summed bytes and parameters and counts the parts.
        let mut footprints: BTreeMap<String, Footprint> = BTreeMap::new();
        for (name, f) in std::mem::take(&mut sum.footprints) {
            let target = self.map(&name).into_owned();
            match (footprints.get_mut(&target), on_collision) {
                (Some(into), OnCollision::Fold) => {
                    into.bytes += f.bytes;
                    into.params += f.params;
                    into.parts += f.parts;
                }
                // `Warn` keeps one entry, matching the signature map above — the collision it reports is
                // the thing to act on.
                (Some(_), OnCollision::Warn) => {}
                (None, _) => {
                    footprints.insert(target, f);
                }
            }
        }
        sum.footprints = footprints;
        collisions
    }

    /// Classify how the rules rewrite `names` (see [`RenamePlan`]). This is the
    /// shared basis for both the `diff` collision *warning* and the `convert --map`
    /// in-place rename's (fatal) validity checks — one engine, two severities.
    ///
    /// The collision set counts *every* resulting name, including names no rule
    /// changed, so a rule that renames one tensor onto another, untouched one is
    /// caught too (not just two sources colliding onto a fresh name).
    pub fn plan_renames<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> RenamePlan {
        let mut renames = Vec::new();
        let mut counts: HashMap<String, u32> = HashMap::new();
        for name in names {
            let to = self.map(name).into_owned();
            *counts.entry(to.clone()).or_insert(0) += 1;
            if to != name {
                renames.push((name.to_string(), to));
            }
        }
        let mut collisions: Vec<String> = counts
            .into_iter()
            .filter_map(|(name, c)| (c > 1).then_some(name))
            .collect();
        collisions.sort_unstable();
        RenamePlan {
            renames,
            collisions,
        }
    }

    /// The indices of rules that never change *any* of `names`. Rules are applied
    /// cumulatively (a later rule sees earlier rules' output, exactly as [`map`]
    /// does), so this reports a genuinely dead rule — a typo'd pattern the caller
    /// can warn about without failing the whole operation.
    ///
    /// [`map`]: NameMap::map
    pub fn unmatched_rules<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Vec<usize> {
        if self.rules.is_empty() {
            return Vec::new();
        }
        let mut matched = vec![false; self.rules.len()];
        for name in names {
            let mut cur = Cow::Borrowed(name);
            for ((re, rep), hit) in self.rules.iter().zip(matched.iter_mut()) {
                // `replace_all` returns `Owned` iff it matched at least once.
                if let Cow::Owned(s) = re.replace_all(&cur, rep.as_str()) {
                    *hit = true;
                    cur = Cow::Owned(s);
                }
            }
        }
        matched
            .iter()
            .enumerate()
            .filter_map(|(i, &hit)| (!hit).then_some(i))
            .collect()
    }

    /// How many of `names` at least one rule matches — counting a name even when
    /// the rewrite leaves it unchanged (the replacement re-inserts the same
    /// capture). Lets a caller tell "the pattern matches N tensors but the new
    /// name is identical" apart from "the pattern matches nothing", which
    /// [`plan_renames`](Self::plan_renames) (changed-only) can't.
    pub fn match_count<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> usize {
        names
            .into_iter()
            .filter(|name| {
                let mut cur = Cow::Borrowed(*name);
                let mut matched = false;
                for (re, rep) in &self.rules {
                    if let Cow::Owned(s) = re.replace_all(&cur, rep.as_str()) {
                        matched = true;
                        cur = Cow::Owned(s);
                    }
                }
                matched
            })
            .count()
    }
}

/// How a [`NameMap`]'s rules rewrite a set of tensor names — the shared basis for
/// the `diff` collision warning and the in-place `rename` validity checks.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RenamePlan {
    /// `(old, new)` for every name a rule changes, in the input order.
    pub renames: Vec<(String, String)>,
    /// Target names that two or more distinct sources map onto (rules too broad).
    /// Sorted. A rename is unsafe while any collision remains.
    pub collisions: Vec<String>,
}

/// A tensor's shape as a glob-matchable path, `dim/dim/…` (empty for a scalar) —
/// so a shape pattern can wildcard one dimension with `*` and any number with
/// `**`, matched with [`shape_match_opts`] (a literal `/` separates dims).
fn shape_key(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("/")
}

/// Glob options for matching a [`shape_key`]: `/` is a real separator, so `*`
/// matches within one dimension and only `**` spans several — mirroring how a
/// filesystem glob treats path components.
fn shape_match_opts() -> MatchOptions {
    MatchOptions {
        require_literal_separator: true,
        ..MatchOptions::new()
    }
}

/// A CLI-driven selection of which tensors to diff (`--name` / `--names` /
/// `--names-from` / `--dtype-is` / `--shape-is`). The constraints compose with
/// **AND** — a tensor is kept only if it satisfies every constraint that was
/// given; an unset constraint always passes. Names, dtypes and shapes are matched
/// with the same [`glob`] engine, so `*`/`**`/`?`/`[…]` work everywhere (shapes
/// via [`shape_key`], dtypes case-insensitively).
#[derive(Default)]
pub struct TensorFilter {
    /// Name globs (with `!`-negation) — the shared [`NameFilter`].
    pub names: NameFilter,
    /// Exact names (union of `--names` and `--names-from`); `None` = unconstrained.
    pub names_exact: Option<HashSet<String>>,
    /// A dtype glob, matched against the UPPERCASED dtype; `None` = unconstrained.
    pub dtype: Option<Pattern>,
    /// A shape glob, matched against the [`shape_key`]; `None` = unconstrained.
    pub shape: Option<Pattern>,
}

impl TensorFilter {
    /// Whether any constraint is set (so the diff is scoped to a subset).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.names.is_active()
            || self.names_exact.is_some()
            || self.dtype.is_some()
            || self.shape.is_some()
    }

    /// Whether `name` — with its old and/or new signature (either may be absent
    /// when the tensor is only on one side) — passes every constraint. A dtype /
    /// shape constraint matches if **either** side matches, so a tensor whose
    /// dtype or shape changed is still selected.
    fn matches(&self, name: &str, old: Option<&TensorSig>, new: Option<&TensorSig>) -> bool {
        if !self.names.matches(name) {
            return false;
        }
        if self
            .names_exact
            .as_ref()
            .is_some_and(|set| !set.contains(name))
        {
            return false;
        }
        if let Some(pat) = &self.dtype {
            let hit = |s: &TensorSig| pat.matches(&s.dtype.to_uppercase());
            if !old.is_some_and(hit) && !new.is_some_and(hit) {
                return false;
            }
        }
        if let Some(pat) = &self.shape {
            let opts = shape_match_opts();
            let hit = |s: &TensorSig| pat.matches_with(&shape_key(&s.shape), opts);
            if !old.is_some_and(hit) && !new.is_some_and(hit) {
                return false;
            }
        }
        true
    }

    /// Restrict both summaries to the tensors that pass the filter. The union of
    /// names is tested, so a tensor present on only one side is kept iff it
    /// matches (and still shows as added/removed). No-op when inactive.
    ///
    /// Narrows the **totals** with the tensors ([`CheckpointSummary::retain_tensors`]): a report about
    /// nineteen of 117,664 tensors used to be headed by the two checkpoints' whole sizes, which is a
    /// true statement about something the reader is not looking at.
    pub fn apply(&self, old: &mut CheckpointSummary, new: &mut CheckpointSummary) {
        if !self.is_active() {
            return;
        }
        let keep: HashSet<String> = old
            .tensors
            .keys()
            .chain(new.tensors.keys())
            .filter(|n| self.matches(n, old.tensors.get(*n), new.tensors.get(*n)))
            .cloned()
            .collect();
        old.retain_tensors(|n| keep.contains(n));
        new.retain_tensors(|n| keep.contains(n));
    }

    /// A one-line, human-readable summary of the active constraints (for the
    /// "diff: …" context line), or `None` when inactive.
    pub fn describe(&self) -> Option<String> {
        if !self.is_active() {
            return None;
        }
        let mut parts = Vec::new();
        if !self.names.include.is_empty() {
            let globs: Vec<&str> = self.names.include.iter().map(Pattern::as_str).collect();
            parts.push(format!("name~{}", globs.join("|")));
        }
        if !self.names.exclude.is_empty() {
            let globs: Vec<&str> = self.names.exclude.iter().map(Pattern::as_str).collect();
            parts.push(format!("name!~{}", globs.join("|")));
        }
        if let Some(set) = &self.names_exact {
            parts.push(format!("names({})", set.len()));
        }
        if let Some(p) = &self.dtype {
            parts.push(format!("dtype~{}", p.as_str()));
        }
        if let Some(p) = &self.shape {
            // Show dims comma-separated, as the user wrote them.
            parts.push(format!("shape~{}", p.as_str().replace('/', ",")));
        }
        Some(parts.join(", "))
    }
}

/// A tensor present in both checkpoints that differs — by dtype/shape, or (with
/// `--values`) by element values even when the signature is unchanged.
#[derive(serde::Serialize)]
pub struct TensorChange {
    pub name: String,
    pub old: TensorSig,
    pub new: TensorSig,
    /// The element-value comparison, when `--values` ran it (`None` otherwise, or
    /// when the shapes differ so an element-wise comparison isn't defined).
    pub values: Option<ValueDiff>,
    /// The distribution shift, when `--histogram` ran it.
    pub histogram: Option<HistShift>,
}

/// A metadata entry present in both checkpoints whose value and/or type differ.
#[derive(serde::Serialize)]
pub struct MetaChange {
    pub name: String,
    pub old: MetaVal,
    pub new: MetaVal,
}

/// The structural difference between two checkpoints (old → new). "Removed" is in
/// the old but not the new; "added" is in the new but not the old; "changed" is in
/// both with a differing signature.
#[derive(serde::Serialize)]
pub struct DiffReport {
    pub tensors_removed: Vec<(String, TensorSig)>,
    pub tensors_added: Vec<(String, TensorSig)>,
    pub tensors_changed: Vec<TensorChange>,
    pub tensors_unchanged: usize,
    pub meta_removed: Vec<(String, MetaVal)>,
    pub meta_added: Vec<(String, MetaVal)>,
    pub meta_changed: Vec<MetaChange>,
    pub meta_unchanged: usize,
    /// Overall size (bytes) and parameter count of each side, for the size/params
    /// comparison in the summary.
    pub old_bytes: usize,
    pub new_bytes: usize,
    pub old_params: usize,
    pub new_params: usize,
    /// The S3 object-metadata diff, set only for an s3-vs-s3 comparison (attached
    /// after [`compare_with`], which has no S3 data). `None` otherwise.
    pub s3: Option<S3Diff>,
    /// Each side's last-modified timestamp (the newest S3 object under the prefix),
    /// shown in the size/params summary. `Some` only for an s3-vs-s3 diff.
    pub old_modified: Option<String>,
    pub new_modified: Option<String>,
    /// Names that stand for several tensors after an alignment folded them: `name → (old parts, new
    /// parts)`. Empty unless something folded.
    ///
    /// This is what makes an unfused-to-fused comparison readable: the row for
    /// `…experts.down_proj.weight` reports `×256 → ×1`, so "did the conversion keep every expert" is
    /// answered on the row rather than inferred from a count of removals.
    pub folded: BTreeMap<String, (usize, usize)>,
}

impl DiffReport {
    /// True when anything was added, removed, or changed — drives the exit code
    /// (`1` like `diff`, vs `0` when the two checkpoints are structurally identical).
    /// `count_s3` folds in whole-prefix S3 object-metadata material changes (a
    /// re-uploaded object, a differing `ETag`, …); pass `false` when a `--name` filter
    /// scoped the diff to a subset of tensors, since the S3 comparison is
    /// whole-prefix and thus out of that scope — exactly like the metadata section,
    /// which is likewise "not compared (filtered subset)". Timestamp-only S3 deltas
    /// never count either way.
    pub fn has_differences_with(&self, count_s3: bool) -> bool {
        !self.tensors_removed.is_empty()
            || !self.tensors_added.is_empty()
            || !self.tensors_changed.is_empty()
            || !self.meta_removed.is_empty()
            || !self.meta_added.is_empty()
            || !self.meta_changed.is_empty()
            || (count_s3 && self.s3.as_ref().is_some_and(S3Diff::has_material_changes))
    }

    /// [`Self::has_differences_with`] counting S3 material changes — the default
    /// whole-checkpoint comparison (no `--name` filter).
    #[must_use]
    pub fn has_differences(&self) -> bool {
        self.has_differences_with(true)
    }

    /// `  (×256 → ×1)` for a row an alignment folded, or `""` for the ordinary one-to-one row.
    ///
    /// Takes the group's member names because a grouped line can stand for many; a group whose members
    /// folded identically reports the fold once, and a group whose members disagree reports nothing
    /// rather than one member's count as if it spoke for all of them.
    fn fold_note(&self, names: &[String]) -> String {
        match self.fold_of(names) {
            Some([old, new]) => format!("  (×{old} → ×{new})"),
            None => String::new(),
        }
    }

    /// The fold every one of `names` agrees about, or `None`.
    ///
    /// `None` for a group whose members disagree, deliberately: one member's count presented for a row
    /// standing for sixty would be a claim about the other fifty-nine.
    fn fold_of(&self, names: &[String]) -> Option<[usize; 2]> {
        if self.folded.is_empty() {
            return None;
        }
        let mut counts = names.iter().filter_map(|n| self.folded.get(n));
        let first = counts.next()?;
        if counts.any(|c| c != first) {
            return None;
        }
        let (old, new) = *first;
        (old != new).then_some([old, new])
    }

    /// The `modified: OLD → NEW` line for an s3-vs-s3 pair, or `None` when the sides carry no
    /// timestamps (every other kind of source).
    ///
    /// Its own function because the browser shows this line too, and the timestamps are *humanised*
    /// (`2026-06-26T14:32:01Z` → `2026-06-26 14:32:01 UTC`) — a rule worth having once rather than
    /// mirrored in TypeScript for one line.
    #[must_use]
    pub fn modified_line(&self, color: bool) -> Option<String> {
        let (o, n) = (self.old_modified.as_ref()?, self.new_modified.as_ref()?);
        let (os, ns) = (fmt_timestamp(o), fmt_timestamp(n));
        Some(if os == ns {
            format!("modified: {os} (unchanged)")
        } else {
            format!(
                "modified: {} → {}",
                paint(&os, color, RED),
                paint(&ns, color, GREEN),
            )
        })
    }

    /// Render the report as plain text: a `---`/`+++` header naming the two sides,
    /// then a counts line and a `- removed / + added / ~ changed` list for tensors,
    /// then the same for metadata (unless `opts.metadata` is false). Entries are
    /// collapsed by name template + change when `opts.group`; colourised per
    /// `opts.color`. The counts lines always report raw entry totals.
    pub fn render(&self, old_label: &str, new_label: &str, opts: DiffOpts) -> String {
        // `--full` (no grouping) renders each entry as its own singleton group.
        let grouped = |items: &[(String, TensorSig)]| {
            if opts.group {
                group_entries(items)
            } else {
                singletons(items)
            }
        };

        let mut s = String::new();
        // Old side red, new side green — the same convention as the entries/totals.
        let _ = writeln!(s, "{}", paint(&format!("--- {old_label}"), opts.color, RED));
        let _ = writeln!(
            s,
            "{}",
            paint(&format!("+++ {new_label}"), opts.color, GREEN)
        );

        // Spell out what was (and wasn't) compared, and what the -/+/~ markers on
        // the summary and the entries below mean.
        let scope = if opts.values {
            "scope: tensor structure (name, dtype, shape) + element values"
        } else {
            "scope: tensor structure (name, dtype, shape) — element values not compared"
        };
        let _ = writeln!(s, "{}", paint(scope, opts.color, DIM));
        let _ = writeln!(
            s,
            "{}",
            paint("legend: - removed, + added, ~ changed", opts.color, DIM)
        );

        // Overall change: total on-disk size and parameter count (absolute +
        // relative %); the per-tensor breakdown follows.
        //
        // Under a filter these describe the **matched tensors**, not the checkpoints, so they say so —
        // the same words the metadata section uses for the same reason. `1966.5 GiB → 451.8 GiB` above
        // nineteen of 117,664 tensors is true of something the reader is not looking at.
        let _ = writeln!(s);
        let (size_label, params_label) = totals_labels(opts.filtered);
        let _ = writeln!(
            s,
            "{}",
            totals_line(
                size_label,
                self.old_bytes,
                self.new_bytes,
                opts.color,
                format_size
            )
        );
        let _ = writeln!(
            s,
            "{}",
            totals_line(
                params_label,
                self.old_params,
                self.new_params,
                opts.color,
                format_parameters
            )
        );
        // For an s3-vs-s3 diff, the checkpoints' last-modified (newest object under
        // each prefix) — old red, new green, like the size/params values.
        if let Some(line) = self.modified_line(opts.color) {
            let _ = writeln!(s, "{line}");
        }

        let _ = writeln!(
            s,
            "\ntensors: -{} +{} ~{} ({} unchanged)",
            self.tensors_removed.len(),
            self.tensors_added.len(),
            self.tensors_changed.len(),
            self.tensors_unchanged,
        );
        for g in grouped(&self.tensors_removed) {
            let line = format!(
                "- {}  [{}]",
                display_name(&g.template, &g.indices),
                g.key.render()
            );
            let _ = writeln!(
                s,
                "  {}{}",
                paint(&line, opts.color, RED),
                count_suffix(g.count)
            );
        }
        for g in grouped(&self.tensors_added) {
            let line = format!(
                "+ {}  [{}]",
                display_name(&g.template, &g.indices),
                g.key.render()
            );
            let _ = writeln!(
                s,
                "  {}{}",
                paint(&line, opts.color, GREEN),
                count_suffix(g.count)
            );
        }
        for g in group_changed(&self.tensors_changed, opts.group) {
            let name = display_name(&g.template, &g.indices);
            let suffix = count_suffix(g.count);
            if g.old == g.new {
                // Same dtype & shape — only the values / distribution changed.
                let reason = if g.values.is_some_and(|v| v.differing > 0) {
                    "values differ"
                } else {
                    "distribution differs"
                };
                let _ = writeln!(s, "  ~ {name}  [{}]  ({reason}){suffix}", g.old.render());
            } else {
                let (old, new) = render_change(&g.old, &g.new, opts.color);
                // `×256 → ×1` when an alignment folded one side: what the fused tensor stands for, on
                // the row that compares it. Without it the shapes alone look like an unexplained
                // change of rank.
                let fold = self.fold_note(&g.names);
                let _ = writeln!(s, "  ~ {name}  [{old}] → [{new}]{fold}{suffix}");
            }
            if opts.values {
                match &g.values {
                    Some(vd) if vd.differing > 0 => {
                        let _ = writeln!(s, "{}", value_line(vd));
                    }
                    Some(_) => {
                        let _ = writeln!(s, "    values: identical");
                    }
                    // --values requested but a shape change made it undefined.
                    None => {
                        let _ = writeln!(s, "    values: not compared (shapes differ)");
                    }
                }
            }
            if opts.histogram {
                let _ = writeln!(s, "{}", histogram_line(&g.hist_tvds, g.hist_bins));
            }
        }

        if opts.metadata {
            let _ = writeln!(
                s,
                "\nmetadata: -{} +{} ~{} ({} unchanged)",
                self.meta_removed.len(),
                self.meta_added.len(),
                self.meta_changed.len(),
                self.meta_unchanged,
            );
            let meta_grouped = |items: &[(String, MetaVal)]| {
                if opts.group {
                    group_entries(items)
                } else {
                    singletons(items)
                }
            };
            for g in meta_grouped(&self.meta_removed) {
                let line = format!(
                    "- {} = {}",
                    display_name(&g.template, &g.indices),
                    quote_trunc(&g.key.value)
                );
                let _ = writeln!(
                    s,
                    "  {}{}",
                    paint(&line, opts.color, RED),
                    count_suffix(g.count)
                );
            }
            for g in meta_grouped(&self.meta_added) {
                let line = format!(
                    "+ {} = {}",
                    display_name(&g.template, &g.indices),
                    quote_trunc(&g.key.value)
                );
                let _ = writeln!(
                    s,
                    "  {}{}",
                    paint(&line, opts.color, GREEN),
                    count_suffix(g.count)
                );
            }
            let mchanged: Vec<(String, (MetaVal, MetaVal))> = self
                .meta_changed
                .iter()
                .map(|c| (c.name.clone(), (c.old.clone(), c.new.clone())))
                .collect();
            let mchanged_groups = if opts.group {
                group_entries(&mchanged)
            } else {
                singletons(&mchanged)
            };
            for g in &mchanged_groups {
                let (old, new) = (&g.key.0, &g.key.1);
                let name = display_name(&g.template, &g.indices);
                let suffix = count_suffix(g.count);
                if old.value == new.value {
                    // Same value, different declared type.
                    let _ = writeln!(
                        s,
                        "  ~ {name} (type {} → {}){suffix}",
                        paint(&old.value_type, opts.color, RED),
                        paint(&new.value_type, opts.color, GREEN),
                    );
                } else {
                    // Prefer a git-style line diff for long values: JSON is
                    // pretty-printed first (so even a minified one-liner diffs
                    // line-by-line), else any already-multi-line value is diffed
                    // as-is. Short single-line values stay inline, windowed around
                    // where they first diverge.
                    let w = meta_line_width();
                    let line_pair = match (pretty_json(&old.value, w), pretty_json(&new.value, w)) {
                        // JSON on both sides: decide purely on the width-aware pretty
                        // form — line diff if it expanded, else inline (small JSON
                        // stays compact even if its raw form had newlines).
                        (Some(o), Some(n)) => {
                            (is_multiline(&o) || is_multiline(&n)).then_some((o, n))
                        }
                        // Non-JSON: line diff a raw multi-line value; else inline.
                        _ if is_multiline(&old.value) || is_multiline(&new.value) => {
                            Some((old.value.clone(), new.value.clone()))
                        }
                        _ => None,
                    };
                    if let Some((o, n)) = line_pair {
                        let _ = writeln!(s, "  ~ {name}:{suffix}");
                        write_meta_line_diff(&mut s, &o, &n, opts.color);
                    } else {
                        let (o, n) = quote_diff(&old.value, &new.value);
                        let _ = writeln!(
                            s,
                            "  ~ {name} = {} → {}{suffix}",
                            paint(&o, opts.color, RED),
                            paint(&n, opts.color, GREEN),
                        );
                    }
                }
            }
        } else {
            // Make it obvious the metadata was deliberately left out, rather than
            // silently showing only the tensors section, and say why.
            let reason = if opts.filtered {
                "filtered subset"
            } else {
                "--only-tensors"
            };
            let _ = writeln!(s, "\nmetadata: not compared ({reason})");
        }

        // S3 object metadata (s3-vs-s3 only). The lines come from `S3Diff::summary_lines`, shared
        // with the JSON API — the browser shows this section too, and what a matching multipart
        // ETag does and does not prove is not a claim to make twice.
        if let Some(s3) = &self.s3 {
            let _ = writeln!(s);
            for line in s3.summary_lines(opts.group, opts.filtered) {
                let (indent, code) = match line.kind {
                    S3LineKind::Heading => ("", None),
                    S3LineKind::Removed => ("  ", Some(RED)),
                    S3LineKind::Added => ("  ", Some(GREEN)),
                    S3LineKind::Changed => ("  ", None),
                    S3LineKind::Note => ("  ", Some(DIM)),
                };
                let text = format!("{indent}{}", line.text);
                let _ = match code {
                    Some(code) => writeln!(s, "{}", paint(&text, opts.color, code)),
                    None => writeln!(s, "{text}"),
                };
            }
        }
        s
    }
}

// ── S3 object metadata diff (s3-vs-s3 only) ─────────────────────────────────

/// An S3 object present in both checkpoints (matched by prefix-relative key) that
/// differs in one or more **material** fields (never a timestamp — those are info).
#[derive(serde::Serialize)]
pub struct S3ObjectChange {
    pub key: String,
    /// Which material fields differ: any of `size`, `etag`, `checksum`, `tags`, `meta`.
    pub fields: Vec<&'static str>,
}

/// An object whose only differences are timestamp-like (last-modified, or a
/// timestamp-valued tag / metadata entry) — reported as info, never a difference.
#[derive(serde::Serialize)]
pub struct S3InfoChange {
    pub key: String,
    pub fields: Vec<&'static str>,
}

/// What was actually checked across the objects present on both sides — so the
/// report can spell out which fields carried a real signal (ETag/size always do;
/// checksums only when stored; tags only when readable), rather than leaving
/// "N unchanged" ambiguous.
#[derive(Default, serde::Serialize)]
pub struct S3Scope {
    /// Objects present on both sides (the ones actually compared field-by-field).
    pub matched: usize,
    /// …of which had an additional checksum on **both** sides (so it was compared).
    pub checksum_both: usize,
    /// …whose `ETag` is a multipart composite (`<md5>-<parts>`) rather than a plain
    /// single-part MD5 — determines how much a matching `ETag` proves (see the report's
    /// confidence note).
    pub etag_multipart: usize,
    /// …of which carried any user metadata on either side.
    pub user_meta_any: usize,
    /// Whether tags were readable on both sides for every matched object (else they
    /// were skipped — see `warnings`).
    pub tags_compared: bool,
}

/// Whether an S3 `ETag` is a **multipart** upload's composite tag (`<hex>-<parts>`)
/// rather than a single-part object's plain MD5. A multipart `ETag` isn't a content
/// hash — it depends on the part size — so a matching one implies identical content
/// only when the part layout also matches.
#[must_use]
pub fn is_multipart_etag(etag: &str) -> bool {
    etag.rsplit_once('-')
        .is_some_and(|(_, parts)| !parts.is_empty() && parts.bytes().all(|b| b.is_ascii_digit()))
}

/// The S3-object-metadata difference between two `s3://` checkpoints.
#[derive(serde::Serialize)]
pub struct S3Diff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<S3ObjectChange>,
    pub unchanged: usize,
    /// Objects differing only in timestamp-like fields — informational.
    pub info_only: Vec<S3InfoChange>,
    /// What was compared across matched objects (for the report's scope line).
    pub scope: S3Scope,
    /// Warnings from either side (e.g. tags denied), each prefixed with its side.
    pub warnings: Vec<String>,
}

/// What one line of the S3 section is, so each surface can style it in its own medium: the terminal
/// paints it, the browser gives it a class. The *words* come from [`S3Diff::summary_lines`] either
/// way — there is one implementation of them.
#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum S3LineKind {
    /// The counts line that opens the section.
    Heading,
    /// An object only in the baseline.
    Removed,
    /// An object only in the newer side.
    Added,
    /// An object whose material fields differ.
    Changed,
    /// What was checked, a confidence caveat, a timestamp-only delta, a warning — never a difference.
    Note,
}

/// One line of the S3 object section: what it says, and what kind of line it is.
#[derive(Clone, serde::Serialize)]
pub struct S3Line {
    pub kind: S3LineKind,
    pub text: String,
}

impl S3Diff {
    /// Material changes only — timestamp-only deltas never count (never affect the
    /// exit code).
    #[must_use]
    pub fn has_material_changes(&self) -> bool {
        !self.added.is_empty() || !self.removed.is_empty() || !self.changed.is_empty()
    }

    /// The S3 object section, line by line and uncoloured.
    ///
    /// One implementation for two surfaces: [`DiffReport::render`] paints these for a terminal and
    /// `/api/diff` sends them to the browser, which had no S3 section at all — the one part of a
    /// checkpoint comparison the web could not show for the `s3://` pairs it exists to compare. The
    /// wording is subtle enough (what a multipart `ETag` proves, what "unchanged" is worth without a
    /// stored checksum) that a second implementation would be a second set of claims about evidence.
    ///
    /// `group` collapses key lists into index templates (`--full` turns it off); `filtered` adds the
    /// note that whole-prefix object changes are outside a `--name` scope.
    #[must_use]
    pub fn summary_lines(&self, group: bool, filtered: bool) -> Vec<S3Line> {
        let mut out = Vec::new();
        let mut push = |kind: S3LineKind, text: String| out.push(S3Line { kind, text });
        push(
            S3LineKind::Heading,
            format!(
                "S3 objects: -{} +{} ~{} ({} unchanged)",
                self.removed.len(),
                self.added.len(),
                self.changed.len(),
                self.unchanged,
            ),
        );
        // Spell out exactly what was compared per object, so "N unchanged" isn't
        // ambiguous: ETag + size always carry a signal; checksums only when the
        // objects stored one; tags only when readable.
        let sc = &self.scope;
        let m = sc.matched;
        // ETag confidence: a single-part ETag is a full MD5 content hash; a
        // multipart one is a part-layout-dependent composite (not a content hash).
        let etag = if m == 0 {
            "ETag".to_string()
        } else if sc.etag_multipart == m {
            "ETag (multipart composite)".to_string()
        } else if sc.etag_multipart == 0 {
            "ETag (single-part MD5)".to_string()
        } else {
            format!(
                "ETag ({} multipart, {} single-part)",
                sc.etag_multipart,
                m - sc.etag_multipart
            )
        };
        let checksum = if sc.checksum_both == m && m > 0 {
            "checksum".to_string()
        } else {
            format!("checksum ({}/{m} stored)", sc.checksum_both)
        };
        let umeta = if sc.user_meta_any > 0 {
            format!("user metadata ({}/{m})", sc.user_meta_any)
        } else {
            "user metadata (none set)".to_string()
        };
        let tags = if sc.tags_compared {
            "tags"
        } else {
            "tags (unavailable — not compared)"
        };
        push(
            S3LineKind::Note,
            format!("checked per object: {etag}, size, {checksum}, {umeta}, {tags}"),
        );
        // How much "unchanged" is worth when no checksum backs it: with a
        // single-part ETag it's a real content hash; with multipart it confirms
        // sameness only when the part layout also matches.
        if sc.checksum_both < m && m > 0 {
            let note = if sc.etag_multipart == 0 {
                "no stored checksums — single-part ETags are full MD5 content hashes, so matching ETag + size means identical content"
            } else {
                "no stored checksums — equality rests on the ETag; a multipart ETag matches only when content AND part layout match (upload with S3 checksums for a definitive compare)"
            };
            push(S3LineKind::Note, format!("note: {note}"));
        }
        // Collapse a set of object keys into index-templated lines with counts
        // (e.g. `model.layers.{0-61}.….weight  (×62)`), like the tensor lists —
        // or one line per key under `--full`. Keeps a 934-object list readable.
        let collapse = |keys: &[&str]| -> Vec<(String, usize)> {
            if group {
                name_schema(keys)
            } else {
                keys.iter().map(|k| ((*k).to_string(), 1)).collect()
            }
        };
        let count = |n: usize| {
            if n > 1 {
                format!("  (×{n})")
            } else {
                String::new()
            }
        };
        let removed: Vec<&str> = self.removed.iter().map(String::as_str).collect();
        for (tmpl, n) in collapse(&removed) {
            push(S3LineKind::Removed, format!("- {tmpl}{}", count(n)));
        }
        let added: Vec<&str> = self.added.iter().map(String::as_str).collect();
        for (tmpl, n) in collapse(&added) {
            push(S3LineKind::Added, format!("+ {tmpl}{}", count(n)));
        }
        // Group material changes by which fields differ, then collapse the keys
        // within each group.
        let mut by_fields: BTreeMap<String, Vec<&str>> = BTreeMap::new();
        for c in &self.changed {
            by_fields
                .entry(c.fields.join(", "))
                .or_default()
                .push(&c.key);
        }
        for (fields, keys) in &by_fields {
            for (tmpl, n) in collapse(keys) {
                push(
                    S3LineKind::Changed,
                    format!("~ {tmpl}{}  ({fields})", count(n)),
                );
            }
        }
        // Under a `--name` filter the diff is scoped to a subset of tensors, but
        // the S3 comparison is always whole-prefix — so these object-level changes
        // are outside the compared scope and don't drive the exit code (the value
        // verdict / tensor diff of the subset does). Say so, or a `~1` above next
        // to exit 0 looks contradictory.
        if filtered && !by_fields.is_empty() {
            push(
                S3LineKind::Note,
                "info: S3 object changes are whole-prefix — not counted for the exit code under a \
                 --name filter (compares the named subset only)"
                    .to_string(),
            );
        }
        // Timestamp-only deltas are informational — never a difference. Summarise
        // them by field-set (a per-object list of "differs only in last-modified"
        // is pure noise); `--full` lists each.
        if group {
            let mut info: BTreeMap<String, usize> = BTreeMap::new();
            for i in &self.info_only {
                *info.entry(i.fields.join(", ")).or_default() += 1;
            }
            for (fields, n) in &info {
                push(
                    S3LineKind::Note,
                    format!("info: {n} object(s) differ only in {fields}"),
                );
            }
        } else {
            for i in &self.info_only {
                push(
                    S3LineKind::Note,
                    format!("info: {} differs only in {}", i.key, i.fields.join(", ")),
                );
            }
        }
        for w in &self.warnings {
            push(S3LineKind::Note, format!("note: {w}"));
        }
        out
    }
}

/// Whether a tag / user-metadata entry is timestamp-like — so a difference in it is
/// treated as information, not a real change. True when the key names a time (common
/// substrings) or the value parses as an ISO-8601 / RFC3339 datetime or a unix epoch.
pub fn is_timestamp_like(key: &str, value: &str) -> bool {
    const PATS: &[&str] = &[
        "time",
        "date",
        "timestamp",
        "modified",
        "created",
        "updated",
        "mtime",
        "ctime",
    ];
    let k = key.to_ascii_lowercase();
    if PATS.iter().any(|p| k.contains(p)) {
        return true;
    }
    let b = value.as_bytes();
    // ISO-8601 / RFC3339: `YYYY-MM-DD` then `T`/space then `HH:MM`.
    // `YYYY-MM-DD` then `T`/space, an hour digit, and the `:` before the minutes. Taken as
    // one 14-byte prefix so the offsets are read from a slice that is known to be there.
    let iso = b.get(..14).is_some_and(|head| {
        let date_ok = head.iter().take(10).enumerate().all(|(i, &c)| {
            if i == 4 || i == 7 {
                c == b'-'
            } else {
                c.is_ascii_digit()
            }
        });
        let sep_ok = matches!(head.get(10), Some(b'T' | b' '));
        let time_ok = head.get(11).is_some_and(u8::is_ascii_digit) && head.get(13) == Some(&b':');
        b.len() >= 16 && date_ok && sep_ok && time_ok
    });
    // Unix epoch seconds (10) or milliseconds (13).
    let epoch = (value.len() == 10 || value.len() == 13) && b.iter().all(u8::is_ascii_digit);
    iso || epoch
}

/// The material vs timestamp-only split of two objects' tag/metadata maps: `true`
/// when the non-timestamp entries differ (material), and `true` when a timestamp-like
/// entry differs (info). A `None` map (couldn't be read) is skipped — not a change.
fn map_differs(
    old: Option<&BTreeMap<String, String>>,
    new: Option<&BTreeMap<String, String>>,
) -> (bool, bool) {
    let (Some(o), Some(n)) = (old, new) else {
        return (false, false); // one side unreadable → can't compare
    };
    let (mut material, mut info) = (false, false);
    for key in o.keys().chain(n.keys()) {
        let (ov, nv) = (o.get(key), n.get(key));
        if ov == nv {
            continue;
        }
        // Timestamp-like on either side's value (or by key name) → info, else material.
        if is_timestamp_like(key, ov.map_or("", |s| s))
            || is_timestamp_like(key, nv.map_or("", |s| s))
        {
            info = true;
        } else {
            material = true;
        }
    }
    (material, info)
}

/// Compare two `s3://` checkpoints' object metadata, matching objects by their
/// prefix-relative key. Timestamps (last-modified, timestamp-like tags/metadata) are
/// bucketed as info and never counted as a difference.
#[must_use]
pub fn compare_s3(old: &S3Meta, new: &S3Meta) -> S3Diff {
    let index = |m: &S3Meta| -> BTreeMap<String, S3Object> {
        m.objects
            .iter()
            .map(|o| (o.key.clone(), o.clone()))
            .collect()
    };
    let (om, nm) = (index(old), index(new));

    let removed: Vec<String> = om
        .keys()
        .filter(|k| !nm.contains_key(*k))
        .cloned()
        .collect();
    let added: Vec<String> = nm
        .keys()
        .filter(|k| !om.contains_key(*k))
        .cloned()
        .collect();

    let mut changed = Vec::new();
    let mut info_only = Vec::new();
    let mut unchanged = 0usize;
    let mut scope = S3Scope {
        tags_compared: true, // until a matched object proves tags weren't readable
        ..Default::default()
    };
    for (key, o) in &om {
        let Some(n) = nm.get(key) else { continue };
        // Tally what was actually comparable across matched objects (for the scope
        // line): ETag/size always; checksum only when both stored one; tags only
        // when readable on both sides.
        scope.matched += 1;
        if o.checksum.is_some() && n.checksum.is_some() {
            scope.checksum_both += 1;
        }
        if is_multipart_etag(&o.etag) {
            scope.etag_multipart += 1;
        }
        if !o.user_meta.is_empty() || !n.user_meta.is_empty() {
            scope.user_meta_any += 1;
        }
        if o.tags.is_none() || n.tags.is_none() {
            scope.tags_compared = false;
        }
        let mut material: Vec<&'static str> = Vec::new();
        let mut info: Vec<&'static str> = Vec::new();
        if o.size != n.size {
            material.push("size");
        }
        if o.etag != n.etag {
            material.push("etag");
        }
        if o.checksum != n.checksum {
            material.push("checksum");
        }
        let (tag_mat, tag_info) = map_differs(o.tags.as_ref(), n.tags.as_ref());
        if tag_mat {
            material.push("tags");
        }
        if tag_info {
            info.push("tags");
        }
        let (meta_mat, meta_info) = map_differs(Some(&o.user_meta), Some(&n.user_meta));
        if meta_mat {
            material.push("meta");
        }
        if meta_info {
            info.push("meta");
        }
        if o.last_modified != n.last_modified {
            info.push("last-modified");
        }
        if !material.is_empty() {
            changed.push(S3ObjectChange {
                key: key.clone(),
                fields: material,
            });
        } else if !info.is_empty() {
            info_only.push(S3InfoChange {
                key: key.clone(),
                fields: info,
            });
        } else {
            unchanged += 1;
        }
    }

    let mut warnings: Vec<String> = old
        .warnings
        .iter()
        .map(|w| format!("old: {w}"))
        .chain(new.warnings.iter().map(|w| format!("new: {w}")))
        .collect();
    warnings.dedup();
    // No matched objects → tags weren't "compared" (nothing to compare).
    if scope.matched == 0 {
        scope.tags_compared = false;
    }
    S3Diff {
        added,
        removed,
        changed,
        unchanged,
        info_only,
        scope,
        warnings,
    }
}

/// Structural comparison of two checkpoint summaries (old → new). Tensor values
/// are not read; see [`compare_with`].
#[must_use]
pub fn compare(old: &CheckpointSummary, new: &CheckpointSummary) -> DiffReport {
    compare_with(old, new, |_| TensorExtras::default())
}

/// Like [`compare`] but also runs `extras_fn(name)` for each tensor present in
/// both checkpoints — its element-value (`--values`) and/or distribution
/// (`--histogram`) comparison. A tensor counts as changed when its dtype or shape
/// differs *or* its extras indicate a difference, so a values-only / distribution
/// change surfaces even when the signature is unchanged.
pub fn compare_with(
    old: &CheckpointSummary,
    new: &CheckpointSummary,
    extras_fn: impl Fn(&str) -> TensorExtras,
) -> DiffReport {
    let mut tensors_removed = Vec::new();
    let mut tensors_changed = Vec::new();
    let mut tensors_unchanged = 0usize;
    for (name, osig) in &old.tensors {
        let Some(nsig) = new.tensors.get(name) else {
            tensors_removed.push((name.clone(), osig.clone()));
            continue;
        };
        let extras = extras_fn(name);
        if nsig != osig || extras.differ() {
            tensors_changed.push(TensorChange {
                name: name.clone(),
                old: osig.clone(),
                new: nsig.clone(),
                values: extras.values,
                histogram: extras.histogram,
            });
        } else {
            tensors_unchanged += 1;
        }
    }
    let mut tensors_added: Vec<_> = new
        .tensors
        .iter()
        .filter(|(name, _)| !old.tensors.contains_key(*name))
        .map(|(name, sig)| (name.clone(), sig.clone()))
        .collect();

    let mut meta_removed = Vec::new();
    let mut meta_changed = Vec::new();
    let mut meta_unchanged = 0usize;
    for (name, oval) in &old.metadata {
        match new.metadata.get(name) {
            None => meta_removed.push((name.clone(), oval.clone())),
            Some(nval) if nval == oval => meta_unchanged += 1,
            Some(nval) => meta_changed.push(MetaChange {
                name: name.clone(),
                old: oval.clone(),
                new: nval.clone(),
            }),
        }
    }
    let mut meta_added: Vec<_> = new
        .metadata
        .iter()
        .filter(|(name, _)| !old.metadata.contains_key(*name))
        .map(|(name, v)| (name.clone(), v.clone()))
        .collect();

    // Natural order, not the `BTreeMap`'s lexicographic one.
    //
    // Iterating the maps gives `experts.0, experts.1, experts.10, experts.100, experts.101, …`, which
    // for a checkpoint whose whole structure is indexed by layer and expert makes a 31k-line report
    // unreadable — and disagrees with the tensor tree and the side-by-side view, which both sort
    // naturally. The same key those use (`tree::natural_sort_key`) is what makes them agree.
    let by_name = |name: &str| crate::tree::natural_sort_key(name);
    tensors_removed.sort_by_cached_key(|(name, _)| by_name(name));
    tensors_added.sort_by_cached_key(|(name, _)| by_name(name));
    tensors_changed.sort_by_cached_key(|c| by_name(&c.name));
    meta_removed.sort_by_cached_key(|(name, _)| by_name(name));
    meta_added.sort_by_cached_key(|(name, _)| by_name(name));
    meta_changed.sort_by_cached_key(|c| by_name(&c.name));

    DiffReport {
        tensors_removed,
        tensors_added,
        tensors_changed,
        tensors_unchanged,
        meta_removed,
        meta_added,
        meta_changed,
        meta_unchanged,
        // Over the tensors **in scope**: a filtered summary carries only the footprints it kept.
        old_bytes: old.total_bytes(),
        new_bytes: new.total_bytes(),
        old_params: old.total_params(),
        new_params: new.total_params(),
        s3: None, // attached by the caller for an s3-vs-s3 diff
        old_modified: None,
        new_modified: None,
        folded: {
            // Every name either side folded, with both sides' counts — `1` where that side has one
            // tensor for it, `0` where it has none at all.
            let (of, nf) = (old.folds(), new.folds());
            let mut names: Vec<&String> = of.keys().chain(nf.keys()).collect();
            names.sort_unstable();
            names.dedup();
            names
                .into_iter()
                .map(|n| {
                    let side = |sum: &CheckpointSummary, folds: &BTreeMap<String, usize>| {
                        folds
                            .get(n)
                            .copied()
                            .unwrap_or_else(|| usize::from(sum.tensors.contains_key(n)))
                    };
                    (n.clone(), (side(old, &of), side(new, &nf)))
                })
                .collect()
        },
    }
}

/// One row of the **grouped** report: a name template and how many tensors it stands for.
///
/// `model.layers.{0-61}.inv_freq_default` with `count: 62`, rather than 62 rows that differ only by a
/// number. This is what the `diff` subcommand prints by default; `--full` is what turns it off.
#[derive(serde::Serialize)]
pub struct GroupedEntry {
    /// The display name, with each index run filled in as its value or range.
    pub name: String,
    /// How many tensors this row stands for. `1` for a name with nothing to group with.
    pub count: usize,
    pub sig: TensorSig,
}

/// One row of the grouped report's *changed* section.
#[derive(serde::Serialize)]
pub struct GroupedChange {
    pub name: String,
    pub count: usize,
    pub old: TensorSig,
    pub new: TensorSig,
    /// `[old parts, new parts]` when an alignment folded these rows and every member agrees about it —
    /// see [`DiffReport::folded`]. `None` otherwise, including for a group whose members disagree,
    /// where one member's count would be a claim about the others.
    pub fold: Option<[usize; 2]>,
}

/// The report with its tensor sections collapsed into index-templated families.
///
/// **Why the server does this.** The templating rule is subtle — which digit runs become placeholders,
/// how `{0-3,5}` summarises a set, when two names may merge (same template *and* same change) — and it
/// already exists here, driving what the terminal prints. A second implementation in the browser would
/// be a second answer to "are these the same family", and the two would drift. So the browser is sent
/// the grouped rows and chooses which list to show.
#[derive(serde::Serialize)]
pub struct GroupedReport {
    pub tensors_added: Vec<GroupedEntry>,
    pub tensors_removed: Vec<GroupedEntry>,
    pub tensors_changed: Vec<GroupedChange>,
}

impl DiffReport {
    /// Collapse the tensor sections into families — the same grouping the terminal renders.
    #[must_use]
    pub fn grouped(&self) -> GroupedReport {
        let entries = |items: &[(String, TensorSig)]| -> Vec<GroupedEntry> {
            group_entries(items)
                .into_iter()
                .map(|g| GroupedEntry {
                    name: display_name(&g.template, &g.indices),
                    count: g.count,
                    sig: g.key,
                })
                .collect()
        };
        GroupedReport {
            tensors_added: entries(&self.tensors_added),
            tensors_removed: entries(&self.tensors_removed),
            tensors_changed: group_changed(&self.tensors_changed, true)
                .into_iter()
                .map(|g| GroupedChange {
                    name: display_name(&g.template, &g.indices),
                    count: g.count,
                    old: g.old,
                    new: g.new,
                    fold: self.fold_of(&g.names),
                })
                .collect(),
        }
    }
}

/// Whether the focused (`--tensor`) diff counts as a difference — drives exit `1`
/// vs `0`. The tensor differs if it's present on only one side, its signature
/// changed, or (same signature) its values changed.
#[must_use]
pub fn tensor_focus_differs(
    old: Option<&TensorSig>,
    new: Option<&TensorSig>,
    values: Option<&ValueCmp>,
) -> bool {
    match (old, new) {
        (Some(o), Some(n)) => o != n || matches!(values, Some(ValueCmp::Differ(_))),
        // Present on only one side (the both-absent case is handled as "not found"
        // by the caller, which exits 2 before reaching here).
        _ => true,
    }
}

/// Render the focused single-tensor diff: the `[old] → [new]` signature line (or
/// added/removed/identical), then an indented `values:` line from the element
/// comparison when both sides exist.
#[must_use]
pub fn render_tensor_focus(
    old_label: &str,
    new_label: &str,
    name: &str,
    old: Option<&TensorSig>,
    new: Option<&TensorSig>,
    values: Option<&ValueCmp>,
    color: bool,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "{}", paint(&format!("--- {old_label}"), color, RED));
    let _ = writeln!(s, "{}", paint(&format!("+++ {new_label}"), color, GREEN));
    let _ = writeln!(s);
    match (old, new) {
        (Some(o), None) => {
            let line = format!("- {name}  [{}]  (only in old)", o.render());
            let _ = writeln!(s, "  {}", paint(&line, color, RED));
        }
        (None, Some(n)) => {
            let line = format!("+ {name}  [{}]  (only in new)", n.render());
            let _ = writeln!(s, "  {}", paint(&line, color, GREEN));
        }
        (Some(o), Some(n)) if o == n => {
            // Same dtype & shape: the only possible difference is in the values.
            match values {
                Some(ValueCmp::Differ(vd)) => {
                    let _ = writeln!(s, "  ~ {name}  [{}]  (values differ)", o.render());
                    let _ = writeln!(s, "{}", value_line(vd));
                }
                Some(ValueCmp::Skipped(why)) => {
                    let _ = writeln!(s, "  = {name}  [{}]", o.render());
                    let _ = writeln!(s, "    values: not compared ({why})");
                }
                _ => {
                    let _ = writeln!(s, "  = {name}  [{}]  (identical)", o.render());
                }
            }
        }
        (Some(o), Some(n)) => {
            // dtype and/or shape changed.
            let (orender, nrender) = render_change(o, n, color);
            let _ = writeln!(s, "  ~ {name}  [{orender}] → [{nrender}]");
            match values {
                Some(ValueCmp::Differ(vd)) => {
                    let _ = writeln!(s, "{}", value_line(vd));
                }
                Some(ValueCmp::Identical) => {
                    let _ = writeln!(s, "    values: identical");
                }
                Some(ValueCmp::Skipped(why)) => {
                    let _ = writeln!(s, "    values: not compared ({why})");
                }
                None => {}
            }
        }
        (None, None) => {}
    }
    s
}

/// The indented `histogram:` summary line for a group's distribution shift(s):
/// the total variation distance (max & mean across the group).
fn histogram_line(tvds: &[f64], bins: usize) -> String {
    if tvds.is_empty() {
        return "    histogram: not compared (shapes differ)".to_string();
    }
    let max = tvds.iter().copied().fold(0.0_f64, f64::max);
    if tvds.len() == 1 {
        format!("    histogram: TVD {} ({bins} bins)", fmt_delta(max))
    } else {
        let mean = tvds.iter().sum::<f64>() / tvds.len() as f64;
        format!(
            "    histogram: TVD max {} mean {} ({bins} bins)",
            fmt_delta(max),
            fmt_delta(mean)
        )
    }
}

/// The full per-tensor histogram comparison table for `diff --tensor --histogram`:
/// one row per shared bin with its label and the old / new counts and delta. Only
/// bins where at least one side is non-empty are shown.
#[must_use]
pub fn render_histogram_table(name: &str, hd: &HistogramDiff, color: bool) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "  histogram of {name}  ({} bins, TVD {})",
        hd.n,
        fmt_delta(hd.tvd())
    );
    let _ = writeln!(
        s,
        "    {:>18}  {:>12}  {:>12}  {:>12}",
        "bin", "old", "new", "Δ"
    );
    for i in 0..hd.n {
        let (o, n) = (
            hd.old.get(i).copied().unwrap_or(0),
            hd.new.get(i).copied().unwrap_or(0),
        );
        if o == 0 && n == 0 {
            continue;
        }
        let delta = n as i64 - o as i64;
        let delta_s = match delta.cmp(&0) {
            std::cmp::Ordering::Greater => paint(&format!("+{delta}"), color, GREEN),
            std::cmp::Ordering::Less => paint(&format!("{delta}"), color, RED),
            std::cmp::Ordering::Equal => "0".to_string(),
        };
        let _ = writeln!(
            s,
            "    {:>18}  {o:>12}  {n:>12}  {delta_s:>12}",
            bin_label(hd.bins, i, hd.n)
        );
    }
    if hd.old_nonfinite > 0 || hd.new_nonfinite > 0 {
        let _ = writeln!(
            s,
            "    {:>18}  {:>12}  {:>12}",
            "non-finite", hd.old_nonfinite, hd.new_nonfinite
        );
    }
    s
}

/// A short label for histogram bin `i` of `n`: the integer (or integer range) for
/// `IntBins`, or the `[lo, hi)` interval for `Range`.
fn bin_label(bins: HistBins, i: usize, n: usize) -> String {
    match bins {
        HistBins::IntBins { start, step } => {
            let lo = start + i as i64 * step;
            if step == 1 {
                format!("{lo}")
            } else {
                format!("{lo}..{}", lo + step - 1)
            }
        }
        HistBins::Range { lo, hi } => {
            let w = if n > 0 { (hi - lo) / n as f64 } else { 0.0 };
            fmt_delta((i as f64).mul_add(w, lo))
        }
    }
}

/// The indented `values:` summary line for a value difference.
fn value_line(vd: &ValueDiff) -> String {
    let mut line = format!(
        "    values: {} of {} differ  (max |Δ| {}, mean |Δ| {})",
        vd.differing,
        vd.elements,
        fmt_delta(vd.max_abs),
        fmt_delta(vd.mean_abs),
    );
    if vd.nonfinite_mismatch > 0 {
        let _ = write!(line, "  [{} non-finite mismatch]", vd.nonfinite_mismatch);
    }
    line
}

/// Format a difference magnitude compactly: fixed-point with trailing zeros
/// trimmed for everyday magnitudes, scientific for very small/large ones.
fn fmt_delta(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let a = x.abs();
    if (1e-3..1e6).contains(&a) {
        let fixed = format!("{x:.6}");
        let trimmed = fixed.trim_end_matches('0').trim_end_matches('.');
        trimmed.to_string()
    } else {
        format!("{x:.3e}")
    }
}

/// Quote a metadata value for one-line display: flatten newlines to spaces and
/// truncate to a readable length (multi-line JSON blobs are common).
fn quote_trunc(v: &str) -> String {
    const MAX: usize = 60;
    let flat = v.replace(['\n', '\r'], " ");
    if flat.chars().count() > MAX {
        let head: String = flat.chars().take(MAX).collect();
        format!("\"{head}…\"")
    } else {
        format!("\"{flat}\"")
    }
}

/// Quote a *changed* value pair for one-line display, each windowed around the
/// first character where they differ (with `…` where truncated) — so the actual
/// change is visible even in a long JSON blob, where head-truncation would print
/// the same shared prefix for both sides. Newlines are flattened to spaces. Short
/// values that fit are shown in full.
fn quote_diff(old: &str, new: &str) -> (String, String) {
    const WINDOW: usize = 60; // chars shown per side
    const CTX: usize = 12; // shared context kept before the first difference
    let o: Vec<char> = old.replace(['\n', '\r'], " ").chars().collect();
    let n: Vec<char> = new.replace(['\n', '\r'], " ").chars().collect();
    let prefix = o.iter().zip(&n).take_while(|(a, b)| a == b).count();
    let start = prefix.saturating_sub(CTX);
    let render = |chars: &[char]| -> String {
        let end = (start + WINDOW).min(chars.len());
        let mut s = String::new();
        if start > 0 {
            s.push('…');
        }
        s.extend(chars.get(start..end).unwrap_or_default());
        if end < chars.len() {
            s.push('…');
        }
        format!("\"{s}\"")
    };
    (render(&o), render(&n))
}

/// Max diff lines shown for one changed metadata value before the rest is
/// summarised as a count — bounds a huge value (e.g. a big `weight_map`).
const MAX_META_DIFF_LINES: usize = 20;

/// Max columns for a single metadata diff line before it's clipped with `…` —
/// bounds a value with one enormous line (e.g. a nested, serialised tensor list
/// that pretty-JSON leaves as a single escaped string). Uses the terminal width
/// when attached, else a sane default; leaves room for the indent.
fn meta_line_width() -> usize {
    crate::utils::term_width(120).saturating_sub(6).max(40)
}

/// Window two long, differing lines each to `max` columns around where they first
/// diverge — `…<shared context><difference>…` — so a changed line whose values
/// share a long prefix still shows the actual change (rather than both clipping to
/// the same prefix). Keeps ~a quarter of the window as leading shared context.
fn window_pair(o: &str, n: &str, max: usize) -> (String, String) {
    let oc: Vec<char> = o.chars().collect();
    let nc: Vec<char> = n.chars().collect();
    let prefix = oc.iter().zip(&nc).take_while(|(a, b)| a == b).count();
    let start = prefix.saturating_sub(max / 4);
    let render = |chars: &[char]| -> String {
        let end = (start + max).min(chars.len());
        let mut s = String::new();
        if start > 0 {
            s.push('…');
        }
        s.extend(chars.get(start..end).unwrap_or_default());
        if end < chars.len() {
            s.push('…');
        }
        s
    };
    (render(&oc), render(&nc))
}

/// Clip `line` to `max` columns, appending `…` when truncated.
fn clip_width(line: String, max: usize) -> String {
    if line.chars().count() <= max {
        line
    } else {
        line.chars()
            .take(max.saturating_sub(1))
            .chain(std::iter::once('…'))
            .collect()
    }
}

/// Whether a metadata value spans multiple lines — the cue to show a line diff
/// rather than a one-line `old → new` (typically a pretty-printed JSON blob).
fn is_multiline(v: &str) -> bool {
    v.contains('\n')
}

/// Pretty-print `v` if it parses as JSON, expanded to one field/element per line
/// so it diffs readably — but **width-aware**: any object/array whose one-line
/// form fits in `width` stays inline (a small `{"bit_widths": [3, 3, 3]}` isn't
/// blown up into eight lines). `None` when `v` isn't JSON.
fn pretty_json(v: &str, width: usize) -> Option<String> {
    let value: Value = serde_json::from_str(v.trim()).ok()?;
    let mut out = String::new();
    write_json(&mut out, &value, 0, 0, width);
    Some(out)
}

/// A JSON value on one line with `: `/`, ` separators (no newlines) — the inline
/// form the width test compares against.
fn compact_json(v: &Value) -> String {
    // `serde_json::Value` is a foreign enum: it can gain a variant in a dependency upgrade without a decision
    // on our side, so a wildcard here is the right shape — the point of
    // `wildcard_enum_match_arm` is to catch a `_` that hides OUR OWN future variants.
    #[allow(clippy::wildcard_enum_match_arm)]
    match v {
        Value::Object(m) => {
            let items: Vec<String> = m
                .iter()
                .map(|(k, val)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(k).unwrap_or_default(),
                        compact_json(val)
                    )
                })
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(compact_json).collect();
            format!("[{}]", items.join(", "))
        }
        other => other.to_string(),
    }
}

/// Write `v` starting at column `col`: inline (via [`compact_json`]) when it fits
/// in `width`, else expanded one child per line at `indent`, recursing so nested
/// values that fit stay inline.
fn write_json(out: &mut String, v: &Value, indent: usize, col: usize, width: usize) {
    let compact = compact_json(v);
    if col + compact.chars().count() <= width || !matches!(v, Value::Object(_) | Value::Array(_)) {
        out.push_str(&compact);
        return;
    }
    let (pad, cpad) = ("  ".repeat(indent), "  ".repeat(indent + 1));
    // `serde_json::Value` is a foreign enum: it can gain a variant in a dependency upgrade without a decision
    // on our side, so a wildcard here is the right shape — the point of
    // `wildcard_enum_match_arm` is to catch a `_` that hides OUR OWN future variants.
    #[allow(clippy::wildcard_enum_match_arm)]
    match v {
        Value::Object(m) => {
            out.push_str("{\n");
            for (i, (k, val)) in m.iter().enumerate() {
                let key = serde_json::to_string(k).unwrap_or_default();
                out.push_str(&cpad);
                out.push_str(&key);
                out.push_str(": ");
                write_json(
                    out,
                    val,
                    indent + 1,
                    cpad.len() + key.chars().count() + 2,
                    width,
                );
                out.push_str(if i + 1 < m.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad);
            out.push('}');
        }
        Value::Array(a) => {
            out.push_str("[\n");
            for (i, val) in a.iter().enumerate() {
                out.push_str(&cpad);
                write_json(out, val, indent + 1, cpad.len(), width);
                out.push_str(if i + 1 < a.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad);
            out.push(']');
        }
        _ => {}
    }
}

/// Write a git-style line diff of two metadata values, indented under the entry
/// name: removed lines red `-`, added lines green `+`, a few lines of context
/// (dim), with `⋮` between hunks. Uses [`similar`] for the line matching. Capped
/// at [`MAX_META_DIFF_LINES`] so one huge value (e.g. a big `weight_map`) can't
/// flood the output — the remainder is summarised as a count.
fn write_meta_line_diff(s: &mut String, old: &str, new: &str, color: bool) {
    use similar::{DiffOp, TextDiff};
    let width = meta_line_width();
    let diff = TextDiff::from_lines(old, new);
    let (ol, nl) = (diff.old_slices(), diff.new_slices());
    let strip = |line: &str| line.strip_suffix('\n').unwrap_or(line).to_string();
    // Render the diff lines (with `⋮` between hunks), tallying total changes.
    let mut lines: Vec<String> = Vec::new();
    let (mut removed, mut added) = (0usize, 0usize);
    // A removed/added/context line, clipped to the width from the left.
    let push = |lines: &mut Vec<String>, sign: char, code: &str, line: &str| {
        lines.push(paint(
            &clip_width(format!("{sign} {line}"), width),
            color,
            code,
        ));
    };
    for (hunk, group) in diff.grouped_ops(3).iter().enumerate() {
        if hunk > 0 {
            lines.push(paint("⋮", color, DIM));
        }
        for op in group {
            match *op {
                DiffOp::Equal { old_index, len, .. } => {
                    for l in ol.get(old_index..old_index + len).unwrap_or_default() {
                        push(&mut lines, ' ', DIM, &strip(l));
                    }
                }
                DiffOp::Delete {
                    old_index, old_len, ..
                } => {
                    removed += old_len;
                    for l in ol.get(old_index..old_index + old_len).unwrap_or_default() {
                        push(&mut lines, '-', RED, &strip(l));
                    }
                }
                DiffOp::Insert {
                    new_index, new_len, ..
                } => {
                    added += new_len;
                    for l in nl.get(new_index..new_index + new_len).unwrap_or_default() {
                        push(&mut lines, '+', GREEN, &strip(l));
                    }
                }
                DiffOp::Replace {
                    old_index,
                    old_len,
                    new_index,
                    new_len,
                } => {
                    removed += old_len;
                    added += new_len;
                    // Pair replaced lines old[k]↔new[k]. When a pair is too wide to
                    // show whole, window each around where they diverge (… diff …)
                    // rather than clipping both to the same shared prefix.
                    let pairs = old_len.min(new_len);
                    // Zip the two runs so each pair comes from the slices themselves.
                    let old_run = ol.get(old_index..old_index + pairs).unwrap_or_default();
                    let new_run = nl.get(new_index..new_index + pairs).unwrap_or_default();
                    for (ol_line, nl_line) in old_run.iter().zip(new_run) {
                        let (o, n) = (strip(ol_line), strip(nl_line));
                        if o.chars().count() > width || n.chars().count() > width {
                            let (ow, nw) = window_pair(&o, &n, width.saturating_sub(2));
                            lines.push(paint(&format!("- {ow}"), color, RED));
                            lines.push(paint(&format!("+ {nw}"), color, GREEN));
                        } else {
                            lines.push(paint(&format!("- {o}"), color, RED));
                            lines.push(paint(&format!("+ {n}"), color, GREEN));
                        }
                    }
                    for l in ol
                        .get(old_index + pairs..old_index + old_len)
                        .unwrap_or_default()
                    {
                        push(&mut lines, '-', RED, &strip(l));
                    }
                    for l in nl
                        .get(new_index + pairs..new_index + new_len)
                        .unwrap_or_default()
                    {
                        push(&mut lines, '+', GREEN, &strip(l));
                    }
                }
            }
        }
    }
    for line in lines.iter().take(MAX_META_DIFF_LINES) {
        let _ = writeln!(s, "      {line}");
    }
    if lines.len() > MAX_META_DIFF_LINES {
        let note = format!(
            "… {} more diff line(s) — {removed} removed, {added} added in total",
            lines.len() - MAX_META_DIFF_LINES
        );
        let _ = writeln!(s, "      {}", paint(&note, color, DIM));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_is_humanized() {
        assert_eq!(
            fmt_timestamp("2026-06-26T14:32:01+00:00"),
            "2026-06-26 14:32:01 UTC"
        );
        assert_eq!(
            fmt_timestamp("2026-06-26T14:32:01Z"),
            "2026-06-26 14:32:01 UTC"
        );
        // Fractional seconds + a non-UTC offset: seconds kept, tz dropped, no " UTC".
        assert_eq!(
            fmt_timestamp("2026-06-26T14:32:01.5-05:00"),
            "2026-06-26 14:32:01"
        );
        // Unrecognised → passthrough.
        assert_eq!(fmt_timestamp("whenever"), "whenever");
    }

    #[test]
    fn match_count_counts_matches_even_when_the_name_is_unchanged() {
        let names = [
            "model.layers.0.mlp.experts.down_proj.codebook",
            "model.layers.1.mlp.experts.down_proj.codebook",
            "model.embed_tokens.weight",
        ];
        // A no-op rule (the replacement re-inserts the same capture) still *matches*
        // its tensors, though `plan_renames` reports no *changes*.
        let noop = NameMap::from_pairs([(
            r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj\.codebook$".to_string(),
            "model.layers.$1.mlp.experts.down_proj.codebook".to_string(),
        )])
        .unwrap();
        assert_eq!(noop.match_count(names), 2, "the pattern matches two layers");
        assert!(
            noop.plan_renames(names).renames.is_empty(),
            "but nothing is renamed (new name == old)"
        );

        // A pattern that matches nothing counts zero.
        let miss = NameMap::from_pairs([("^nope$".to_string(), "x".to_string())]).unwrap();
        assert_eq!(miss.match_count(names), 0);
    }

    #[test]
    fn tensor_families_collapse_layers_and_experts() {
        use crate::tree::{Layout, Storage, TensorInfo};
        let mk = |name: &str, shape: &[usize]| TensorInfo {
            name: name.into(),
            dtype: "F16".into(),
            shape: shape.to_vec(),
            size_bytes: shape.iter().product::<usize>() * 2,
            num_elements: shape.iter().product(),
            storage: Storage::Unknown,
            source_path: "s".into(),
            layout: Layout::None,
        };
        let mut tensors = Vec::new();
        for l in 0..3 {
            for e in 0..2 {
                tensors.push(mk(
                    &format!("model.layers.{l}.mlp.experts.{e}.down_proj.weight"),
                    &[8, 6],
                ));
            }
        }
        tensors.push(mk("model.embed_tokens.weight", &[32, 8]));

        let fams = tensor_families(&tensors);
        assert_eq!(fams.len(), 2); // the 6 expert weights collapse; embed is its own
        let moe = fams.iter().find(|f| f.name.contains("experts")).unwrap();
        assert_eq!(
            moe.name,
            "model.layers.{0-2}.mlp.experts.{0-1}.down_proj.weight"
        );
        assert_eq!(moe.count, 6);
        assert_eq!(moe.dtype.as_deref(), Some("F16"));
        assert_eq!(moe.shape, Some(vec![8, 6]));
        assert_eq!(moe.params, 6 * 48);
        let embed = fams.iter().find(|f| f.name.contains("embed")).unwrap();
        assert_eq!(embed.count, 1);
    }

    fn s3obj(
        key: &str,
        etag: &str,
        size: u64,
        tags: &[(&str, &str)],
        meta: &[(&str, &str)],
        last_modified: &str,
    ) -> S3Object {
        let to_map = |kv: &[(&str, &str)]| {
            kv.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>()
        };
        S3Object {
            key: key.into(),
            size,
            etag: etag.into(),
            checksum: None,
            last_modified: last_modified.into(),
            user_meta: to_map(meta),
            tags: Some(to_map(tags)),
        }
    }

    fn empty_summary() -> CheckpointSummary {
        CheckpointSummary::from_loaded(&[], &[])
    }

    #[test]
    fn is_timestamp_like_flags_time_keys_and_datetime_values() {
        // Key-name signals.
        assert!(is_timestamp_like("created_at", "prod"));
        assert!(is_timestamp_like("LastModified", ""));
        assert!(is_timestamp_like("mtime", "x"));
        // Value signals: ISO-8601 and unix epoch.
        assert!(is_timestamp_like("run", "2026-01-02T03:04:05Z"));
        assert!(is_timestamp_like("built", "1700000000"));
        // Non-timestamps.
        assert!(!is_timestamp_like("env", "prod"));
        assert!(!is_timestamp_like("version", "3"));
    }

    #[test]
    fn compare_s3_splits_material_from_added_removed_unchanged() {
        let old = S3Meta {
            objects: vec![
                s3obj("a", "e1", 10, &[("env", "prod")], &[("run", "1")], "t1"),
                s3obj("gone", "e0", 5, &[], &[], "t"),
                s3obj("same", "eq", 7, &[("k", "v")], &[], "t"),
            ],
            warnings: vec![],
        };
        let new = S3Meta {
            objects: vec![
                // etag changed (material) + last-modified changed (info) → changed(etag).
                s3obj("a", "e2", 10, &[("env", "prod")], &[("run", "1")], "t2"),
                s3obj("added", "e9", 1, &[], &[], "t"),
                s3obj("same", "eq", 7, &[("k", "v")], &[], "t"),
            ],
            warnings: vec!["tags unavailable".into()],
        };
        let d = compare_s3(&old, &new);
        assert_eq!(d.removed, vec!["gone".to_string()]);
        assert_eq!(d.added, vec!["added".to_string()]);
        assert_eq!(d.changed.len(), 1);
        assert_eq!(d.changed[0].key, "a");
        assert!(d.changed[0].fields.contains(&"etag"));
        assert_eq!(d.unchanged, 1);
        assert!(d.has_material_changes());
        // Scope reflects what was comparable across the 2 matched objects.
        assert_eq!(d.scope.matched, 2);
        assert_eq!(d.scope.checksum_both, 0); // none stored a checksum
        assert_eq!(d.scope.etag_multipart, 0); // "e1"/"eq" are single-part
        assert!(d.scope.tags_compared); // tags Some on both sides
        // compare_s3 prefixes each warning with its side.
        assert_eq!(d.warnings, vec!["new: tags unavailable".to_string()]);
    }

    #[test]
    fn compare_s3_timestamp_only_delta_is_info_not_a_difference() {
        let old = S3Meta {
            objects: vec![s3obj(
                "a",
                "eq",
                10,
                &[("created", "2026-01-01T00:00:00Z")],
                &[],
                "2026-01-01T00:00:00Z",
            )],
            warnings: vec![],
        };
        // Same content/etag/size; only last-modified and a timestamp-valued tag differ.
        let new = S3Meta {
            objects: vec![s3obj(
                "a",
                "eq",
                10,
                &[("created", "2026-09-09T00:00:00Z")],
                &[],
                "2026-09-09T00:00:00Z",
            )],
            warnings: vec![],
        };
        let d = compare_s3(&old, &new);
        assert!(
            d.changed.is_empty(),
            "timestamp-only delta must not be material"
        );
        assert_eq!(d.info_only.len(), 1);
        assert!(!d.has_material_changes());
        // A report carrying only this S3 delta reports no differences (exit 0).
        let mut report = compare(&empty_summary(), &empty_summary());
        report.s3 = Some(d);
        assert!(!report.has_differences());
    }

    #[test]
    fn is_multipart_etag_detects_the_part_suffix() {
        assert!(is_multipart_etag("d41d8cd98f00b204e9800998ecf8427e-64"));
        assert!(is_multipart_etag("abc-1"));
        assert!(!is_multipart_etag("d41d8cd98f00b204e9800998ecf8427e")); // plain MD5
        assert!(!is_multipart_etag("abc-")); // no part count
        assert!(!is_multipart_etag("abc-def")); // suffix not digits
    }

    #[test]
    fn render_shows_s3_scope_etag_confidence_and_warning() {
        // Multipart ETags, tags denied on both sides (objects carry `tags: None`).
        let denied = |key: &str, etag: &str| {
            let mut o = s3obj(key, etag, 10, &[], &[], "t");
            o.tags = None;
            o
        };
        let old = S3Meta {
            objects: vec![denied("shard0", "aaa-64")],
            warnings: vec!["tags unavailable (needs s3:GetObjectTagging): AccessDenied".into()],
        };
        let new = S3Meta {
            objects: vec![denied("shard0", "bbb-64")],
            warnings: vec!["tags unavailable (needs s3:GetObjectTagging): AccessDenied".into()],
        };
        let mut report = compare(&empty_summary(), &empty_summary());
        report.s3 = Some(compare_s3(&old, &new));
        let opts = DiffOpts {
            color: false,
            metadata: true,
            group: true,
            values: false,
            histogram: false,
            filtered: false,
        };
        let out = report.render("s3://old", "s3://new", opts);
        assert!(out.contains("S3 objects: -0 +0 ~1"), "{out}");
        // The scope line names the ETag type + coverage of the other fields.
        assert!(
            out.contains("checked per object: ETag (multipart composite), size"),
            "{out}"
        );
        assert!(out.contains("checksum (0/1 stored)"), "{out}");
        assert!(out.contains("tags (unavailable — not compared)"), "{out}");
        // No checksums + multipart ETag → the part-layout caveat note.
        assert!(
            out.contains("note: no stored checksums — equality rests on the ETag"),
            "{out}"
        );
        assert!(out.contains("~ shard0  (etag)"), "{out}");
        assert!(out.contains("note: old: tags unavailable"), "{out}");
    }

    #[test]
    fn render_single_part_etag_note_calls_it_a_content_hash() {
        // Single-part ETags (plain MD5) + no checksums → the "content hash" note.
        let old = S3Meta {
            objects: vec![s3obj("shard0", "md5aaa", 10, &[], &[], "t")],
            warnings: vec![],
        };
        let new = S3Meta {
            objects: vec![s3obj("shard0", "md5aaa", 10, &[], &[], "t")],
            warnings: vec![],
        };
        let mut report = compare(&empty_summary(), &empty_summary());
        report.s3 = Some(compare_s3(&old, &new));
        let opts = DiffOpts {
            color: false,
            metadata: true,
            group: true,
            values: false,
            histogram: false,
            filtered: false,
        };
        let out = report.render("s3://old", "s3://new", opts);
        assert!(out.contains("ETag (single-part MD5)"), "{out}");
        assert!(
            out.contains("single-part ETags are full MD5 content hashes"),
            "{out}"
        );
    }

    #[test]
    fn render_compresses_timestamp_only_info_to_one_line() {
        // Many objects, all identical except last-modified → one summary line, not N.
        let mk = |lm: &str| S3Meta {
            objects: (0..50)
                .map(|i| {
                    s3obj(
                        &format!("model.layers.{i}.self_attn.o_proj.weight"),
                        "eq",
                        10,
                        &[],
                        &[],
                        lm,
                    )
                })
                .collect(),
            warnings: vec![],
        };
        let old = mk("2026-01-01T00:00:00Z");
        let new = mk("2026-02-02T00:00:00Z");
        let mut report = compare(&empty_summary(), &empty_summary());
        report.s3 = Some(compare_s3(&old, &new));
        let opts = DiffOpts {
            color: false,
            metadata: true,
            group: true,
            values: false,
            histogram: false,
            filtered: false,
        };
        let out = report.render("s3://old", "s3://new", opts);
        assert!(
            out.contains("info: 50 object(s) differ only in last-modified"),
            "{out}"
        );
        // Collapsed — not one line per object.
        assert_eq!(out.matches("differ only in").count(), 1, "{out}");

        // …but `--full` lists each object.
        let full = report.render(
            "s3://old",
            "s3://new",
            DiffOpts {
                group: false,
                ..opts
            },
        );
        assert_eq!(full.matches("differs only in").count(), 50, "{full}");
    }

    fn sig(dtype: &str, shape: &[usize]) -> TensorSig {
        TensorSig {
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn totals_line_shows_absolute_and_relative_change() {
        assert_eq!(
            totals_line("size", 100, 150, false, format_size),
            "size: 100 B → 150 B (+50 B, +50.0%)"
        );
        assert_eq!(
            totals_line("params", 56, 40, false, format_parameters),
            "params: 56 → 40 (-16, -28.6%)"
        );
        // equal → unchanged; zero baseline → no percentage
        assert_eq!(
            totals_line("size", 100, 100, false, format_size),
            "size: 100 B (unchanged)"
        );
        assert_eq!(
            totals_line("size", 0, 100, false, format_size),
            "size: 0 B → 100 B (+100 B)"
        );
        // Coloured like the tensor diff: old red, new green, delta dimmed.
        assert_eq!(
            totals_line("size", 100, 150, true, format_size),
            "size: \x1b[31m100 B\x1b[0m → \x1b[32m150 B\x1b[0m (\x1b[2m+50 B, +50.0%\x1b[0m)"
        );
    }
    fn mv(value: &str, ty: &str) -> MetaVal {
        MetaVal {
            value: value.to_string(),
            value_type: ty.to_string(),
        }
    }
    /// A summary with a footprint per tensor — one element per unit of shape, one byte per element —
    /// so a test can narrow it and watch the totals narrow with it.
    fn summary(tensors: &[(&str, TensorSig)], metadata: &[(&str, MetaVal)]) -> CheckpointSummary {
        CheckpointSummary {
            tensors: tensors
                .iter()
                .map(|(n, s)| (n.to_string(), s.clone()))
                .collect(),
            metadata: metadata
                .iter()
                .map(|(n, v)| (n.to_string(), v.clone()))
                .collect(),
            footprints: tensors
                .iter()
                .map(|(n, s)| {
                    let params: usize = s.shape.iter().product();
                    (
                        n.to_string(),
                        Footprint {
                            bytes: params,
                            params,
                            parts: 1,
                        },
                    )
                })
                .collect(),
        }
    }

    /// The report lists names in *natural* order, like every other view of a checkpoint.
    ///
    /// The lists are built by walking `BTreeMap`s, which is lexicographic: `experts.0, experts.1,
    /// experts.10, experts.100, …`. For a checkpoint indexed by layer and expert that makes a long
    /// report unreadable, and it disagreed with the tensor tree and the side-by-side view, which both
    /// sort naturally. A map iteration order is easy to reintroduce, hence a test.
    #[test]
    fn the_report_lists_names_in_natural_order() {
        let names = [
            "layers.0.w",
            "layers.1.w",
            "layers.2.w",
            "layers.10.w",
            "layers.11.w",
            "layers.100.w",
        ];
        // The baseline has none of them, so every one lands in `tensors_added`.
        let old = summary(&[], &[]);
        let new = summary(
            &names
                .iter()
                .map(|n| (*n, sig("F16", &[2])))
                .collect::<Vec<_>>(),
            &[],
        );
        let report = compare(&old, &new);
        assert_eq!(
            report
                .tensors_added
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            names,
            "added tensors should read 0, 1, 2, 10, 11, 100 — not 0, 1, 10, 100, 11, 2"
        );

        // Removals take the same route, in the other direction.
        let report = compare(&new, &old);
        assert_eq!(
            report
                .tensors_removed
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            names,
            "removed tensors should be naturally ordered too"
        );
    }

    #[test]
    fn name_map_parses_rules_skipping_blanks_and_comments() {
        let text = "\
            # gpt-oss rename\n\
            \\.mlp\\.experts\\.  =>  .block_sparse_moe.experts.\n\
            \n\
            experts\\.down_proj$ => experts.down_proj.weight\n";
        let pairs = NameMap::parse_rules(text.lines()).unwrap();
        assert_eq!(
            pairs,
            vec![
                (
                    r"\.mlp\.experts\.".to_string(),
                    ".block_sparse_moe.experts.".to_string()
                ),
                (
                    r"experts\.down_proj$".to_string(),
                    "experts.down_proj.weight".to_string()
                ),
            ]
        );
    }

    #[test]
    fn name_map_rule_without_separator_is_an_error() {
        assert!(NameMap::parse_rules(["a.b.c"]).is_err());
    }

    #[test]
    fn name_map_applies_rules_in_order_with_captures() {
        let map = NameMap::from_pairs(
            NameMap::parse_rules([
                r"\.mlp\.experts\.=>.block_sparse_moe.experts.",
                r"experts\.(down|gate_up)_proj$=>experts.${1}_proj.weight",
            ])
            .unwrap(),
        )
        .unwrap();
        // Both rules fire, in order (rename the segment, then append `.weight`).
        assert_eq!(
            map.map("model.layers.0.mlp.experts.down_proj"),
            "model.layers.0.block_sparse_moe.experts.down_proj.weight"
        );
        // A name no rule matches passes through unchanged (and stays borrowed).
        assert!(matches!(map.map("lm_head.weight"), Cow::Borrowed(_)));
    }

    #[test]
    fn name_map_remaps_old_summary_so_renamed_tensors_line_up() {
        let map = NameMap::from_pairs(
            NameMap::parse_rules([
                r"\.mlp\.experts\.down_proj$=>.block_sparse_moe.experts.down_proj.weight",
            ])
            .unwrap(),
        )
        .unwrap();
        let mut old = summary(
            &[("model.layers.0.mlp.experts.down_proj", sig("BF16", &[8]))],
            &[],
        );
        let new = summary(
            &[(
                "model.layers.0.block_sparse_moe.experts.down_proj.weight",
                sig("BF16", &[8]),
            )],
            &[],
        );
        // Before the map: nothing lines up (one removed, one added).
        assert_eq!(compare(&old, &new).tensors_removed.len(), 1);
        // After: the rename aligns them, so it's a match (unchanged, same sig).
        assert!(map.remap_summary(&mut old).is_empty());
        let report = compare(&old, &new);
        assert_eq!(report.tensors_removed.len(), 0);
        assert_eq!(report.tensors_added.len(), 0);
        assert_eq!(report.tensors_unchanged, 1);
    }

    #[test]
    fn name_map_reports_collisions() {
        // A too-broad rule drops the layer prefix, so two distinct names collide
        // onto one — reported so the user knows a rename is over-eager.
        let map =
            NameMap::from_pairs(vec![(r"^.*\.(q_proj)$".into(), "shared.$1".into())]).unwrap();
        let mut old = summary(
            &[
                ("x.q_proj", sig("BF16", &[8])),
                ("y.q_proj", sig("BF16", &[8])),
            ],
            &[],
        );
        let collisions = map.remap_summary(&mut old);
        assert_eq!(collisions, vec!["shared.q_proj".to_string()]);
        assert_eq!(old.tensors.len(), 1); // the two folded into one (last wins)
    }

    #[test]
    fn change_colours_only_differing_dtype_and_dims() {
        // dtype F16→U16 and only the first dim 256→64 differ; 3072/1540 are shared.
        let (o, n) = render_change(
            &sig("F16", &[256, 3072, 1540]),
            &sig("U16", &[64, 3072, 1540]),
            true,
        );
        assert!(o.contains(&format!("{RED}F16{RESET}"))); // dtype coloured
        assert!(n.contains(&format!("{GREEN}U16{RESET}")));
        assert!(o.contains(&format!("{RED}256{RESET}"))); // changed dim coloured
        assert!(n.contains(&format!("{GREEN}64{RESET}")));
        // Unchanged dims are plain (not wrapped in a colour code).
        assert!(o.contains(", 3072, 1540)") && n.contains(", 3072, 1540)"));
    }

    #[test]
    fn change_leaves_dtype_plain_when_only_a_dim_differs() {
        let (o, _n) = render_change(&sig("F16", &[4, 8]), &sig("F16", &[2, 8]), true);
        assert!(!o.contains(&format!("{RED}F16"))); // dtype unchanged → not coloured
        assert!(o.contains(&format!("({RED}4{RESET}, 8)"))); // only dim0 coloured
    }

    #[test]
    fn change_treats_size_one_dims_as_impedance_not_a_shape_change() {
        // (7168, 8192) → (7168, 1, 36864): the inserted `1` is an artefact, so the
        // real dims align as 7168↔7168 (same) and 8192↔36864 (changed).
        let (o, n) = render_change(
            &sig("BF16", &[7168, 8192]),
            &sig("F16", &[7168, 1, 36864]),
            true,
        );
        // 7168 is unchanged on both sides — plain, not coloured.
        assert!(o.contains("(7168, ") && n.contains("(7168, "));
        assert!(!o.contains(&format!("{RED}7168")) && !n.contains(&format!("{GREEN}7168")));
        // Only the genuinely different dim is coloured.
        assert!(o.contains(&format!("{RED}8192{RESET}")));
        assert!(n.contains(&format!("{GREEN}36864{RESET}")));
        // The singleton is dimmed impedance, not a green "added" dim.
        assert!(n.contains(&format!("{DIM}1{RESET}")));
        assert!(!n.contains(&format!("{GREEN}1{RESET}")));
    }

    #[test]
    fn change_colours_whole_shape_only_on_a_real_rank_change() {
        // A rank change that survives the squeeze (2 real dims → 3) still colours
        // the whole shape, since the dimensions genuinely don't line up.
        let (o, n) = render_change(&sig("F16", &[4, 8]), &sig("F16", &[4, 8, 2]), true);
        assert!(o.contains(&format!("{RED}(4, 8){RESET}")));
        assert!(n.contains(&format!("{GREEN}(4, 8, 2){RESET}")));
    }

    #[test]
    fn change_colours_whole_shape_when_ranks_differ() {
        let (o, _n) = render_change(&sig("F16", &[4, 8]), &sig("F16", &[32]), true);
        assert!(o.contains(&format!("{RED}(4, 8){RESET}")));
    }

    #[test]
    fn identical_checkpoints_have_no_differences() {
        let a = summary(&[("w", sig("F16", &[2, 2]))], &[("k", mv("v", "string"))]);
        let b = summary(&[("w", sig("F16", &[2, 2]))], &[("k", mv("v", "string"))]);
        let r = compare(&a, &b);
        assert!(!r.has_differences());
        assert_eq!(r.tensors_unchanged, 1);
        assert_eq!(r.meta_unchanged, 1);
    }

    #[test]
    fn filtered_diff_ignores_whole_prefix_s3_changes_for_exit_code() {
        // Tensors on both sides match, but a whole-prefix S3 object (e.g. a
        // re-uploaded `__METADATA__`) differs in ETag — a material change.
        let a = summary(&[("w", sig("F16", &[2, 2]))], &[]);
        let b = summary(&[("w", sig("F16", &[2, 2]))], &[]);
        let mut r = compare(&a, &b);
        let old = S3Meta {
            objects: vec![s3obj("__METADATA__", "md5aaa", 10, &[], &[], "t")],
            warnings: vec![],
        };
        let new = S3Meta {
            objects: vec![s3obj("__METADATA__", "md5bbb", 10, &[], &[], "t")],
            warnings: vec![],
        };
        r.s3 = Some(compare_s3(&old, &new));
        // Whole-checkpoint compare: the S3 material change counts (exit 1).
        assert!(r.has_differences());
        assert!(r.has_differences_with(true));
        // Under a `--name` filter the S3 diff is out of the compared subset's scope,
        // so it no longer drives the exit code — the tensors matched → exit 0.
        assert!(!r.has_differences_with(false));
        // …and the report explains why a `~1` above sits next to exit 0.
        let opts = DiffOpts {
            color: false,
            metadata: true,
            group: true,
            values: true,
            histogram: false,
            filtered: true,
        };
        let out = r.render("s3://old", "s3://new", opts);
        assert!(out.contains("S3 objects: -0 +0 ~1"), "{out}");
        assert!(
            out.contains("not counted for the exit code under a --name filter"),
            "{out}"
        );
    }

    #[test]
    fn classifies_added_removed_changed_tensors() {
        let old = summary(
            &[
                ("keep", sig("F16", &[2, 2])),
                ("gone", sig("F32", &[8, 8])),
                ("retyped", sig("F32", &[4, 4])),
                ("reshaped", sig("F16", &[10, 4])),
            ],
            &[],
        );
        let new = summary(
            &[
                ("keep", sig("F16", &[2, 2])),
                ("fresh", sig("BF16", &[1, 1])),
                ("retyped", sig("BF16", &[4, 4])),
                ("reshaped", sig("F16", &[20, 2])),
            ],
            &[],
        );
        let r = compare(&old, &new);
        assert!(r.has_differences());
        assert_eq!(
            r.tensors_removed
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["gone"]
        );
        assert_eq!(
            r.tensors_added
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["fresh"]
        );
        let changed: Vec<_> = r.tensors_changed.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(changed, ["reshaped", "retyped"]); // BTreeMap order
        assert_eq!(r.tensors_unchanged, 1);
    }

    #[test]
    fn classifies_metadata_changes_including_type_only() {
        let old = summary(
            &[],
            &[
                ("same", mv("1", "int")),
                ("v", mv("0.4", "string")),
                ("typed", mv("1", "int")),
            ],
        );
        let new = summary(
            &[],
            &[
                ("same", mv("1", "int")),
                ("v", mv("0.5", "string")),
                ("typed", mv("1", "float")),
                ("extra", mv("x", "string")),
            ],
        );
        let r = compare(&old, &new);
        assert_eq!(
            r.meta_added
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["extra"]
        );
        assert!(r.meta_removed.is_empty());
        let changed: Vec<_> = r.meta_changed.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(changed, ["typed", "v"]);
        assert_eq!(r.meta_unchanged, 1);
        // The type-only change renders as a "(type … → …)" note, not a value diff.
        let out = r.render("old", "new", PLAIN);
        assert!(out.contains("~ typed (type int → float)"), "{out}");
        assert!(out.contains("~ v = \"0.4\" → \"0.5\""), "{out}");
    }

    #[test]
    fn render_notes_when_metadata_excluded() {
        let old = summary(&[("w", sig("F16", &[2, 2]))], &[("k", mv("a", "string"))]);
        let new = summary(&[("w", sig("F16", &[2, 2]))], &[("k", mv("b", "string"))]);
        let r = compare(&old, &new);
        // Default: the metadata change is shown.
        assert!(r.render("o", "n", PLAIN).contains("metadata: -0 +0 ~1"));
        // --only-tensors: a clear note instead, and no per-entry metadata lines.
        let without = r.render(
            "o",
            "n",
            DiffOpts {
                metadata: false,
                ..PLAIN
            },
        );
        assert!(
            without.contains("metadata: not compared (--only-tensors)"),
            "{without}"
        );
        assert!(!without.contains("  ~ k"), "{without}");
    }

    #[test]
    fn full_value_diff_promotes_values_only_change() {
        // Same dtype & shape on both sides; a value comparison says they differ.
        let old = summary(&[("model.layers.0.w", sig("U8", &[4]))], &[]);
        let new = summary(&[("model.layers.0.w", sig("U8", &[4]))], &[]);
        let r = compare_with(&old, &new, |_| TensorExtras {
            values: Some(ValueDiff {
                elements: 4,
                differing: 2,
                max_abs: 7.0,
                mean_abs: 3.5,
                nonfinite_mismatch: 0,
            }),
            histogram: None,
        });
        // The structurally-identical tensor is now a change.
        assert_eq!(r.tensors_changed.len(), 1);
        assert_eq!(r.tensors_unchanged, 0);
        let out = r.render(
            "o",
            "n",
            DiffOpts {
                values: true,
                ..PLAIN
            },
        );
        assert!(
            out.contains("~ model.layers.0.w  [U8 (4)]  (values differ)"),
            "{out}"
        );
        assert!(
            out.contains("values: 2 of 4 differ  (max |Δ| 7, mean |Δ| 3.5)"),
            "{out}"
        );
    }

    #[test]
    fn full_value_diff_aggregates_within_a_group() {
        // Two layers, each a values-only change → collapse, stats aggregated.
        let names = ["model.layers.0.w", "model.layers.1.w"];
        let mk = || CheckpointSummary {
            tensors: names
                .iter()
                .map(|n| (n.to_string(), sig("U8", &[4])))
                .collect(),
            metadata: BTreeMap::default(),
            footprints: names
                .iter()
                .map(|n| {
                    (
                        n.to_string(),
                        Footprint {
                            bytes: 4,
                            params: 4,
                            parts: 1,
                        },
                    )
                })
                .collect(),
        };
        let per = ValueDiff {
            elements: 4,
            differing: 1,
            max_abs: 2.0,
            mean_abs: 0.5,
            nonfinite_mismatch: 0,
        };
        let r = compare_with(&mk(), &mk(), |_| TensorExtras {
            values: Some(per),
            histogram: None,
        });
        let out = r.render(
            "o",
            "n",
            DiffOpts {
                values: true,
                ..PLAIN
            },
        );
        // One collapsed line with the aggregate (8 elements, 2 differing, max 2).
        assert!(
            out.contains("~ model.layers.{0-1}.w  [U8 (4)]  (values differ)  (×2)"),
            "{out}"
        );
        assert!(
            out.contains("values: 2 of 8 differ  (max |Δ| 2, mean |Δ| 0.5)"),
            "{out}"
        );
    }

    #[test]
    fn color_highlights_only_the_changed_token() {
        // dtype changed, shape same → colour the dtype, not the shape.
        let old = summary(&[("w", sig("F16", &[2, 2]))], &[]);
        let new = summary(&[("w", sig("BF16", &[2, 2]))], &[]);
        let out = compare(&old, &new).render(
            "o",
            "n",
            DiffOpts {
                color: true,
                ..PLAIN
            },
        );
        assert!(out.contains(&format!("{RED}F16{RESET}")), "{out:?}");
        assert!(out.contains(&format!("{GREEN}BF16{RESET}")), "{out:?}");
        // The unchanged shape isn't wrapped in a colour code.
        assert!(!out.contains(&format!("{RED}(2, 2){RESET}")), "{out:?}");
    }

    #[test]
    fn groups_repeated_per_index_changes() {
        // The same dtype change across layers 0..=3 collapses to one line.
        let mk = |dt: &str| {
            (0..4)
                .map(|n| (format!("model.layers.{n}.mlp.weight"), sig(dt, &[8])))
                .collect::<Vec<_>>()
        };
        let footprints = || -> BTreeMap<String, Footprint> {
            mk("F16")
                .into_iter()
                .map(|(n, _)| {
                    (
                        n,
                        Footprint {
                            bytes: 8,
                            params: 8,
                            parts: 1,
                        },
                    )
                })
                .collect()
        };
        let old = CheckpointSummary {
            tensors: mk("F16").into_iter().collect(),
            metadata: BTreeMap::default(),
            footprints: footprints(),
        };
        let new = CheckpointSummary {
            tensors: mk("BF16").into_iter().collect(),
            metadata: BTreeMap::default(),
            footprints: footprints(),
        };
        let r = compare(&old, &new);
        // Grouped (default): one collapsed line with the range and ×count.
        let grouped = r.render("o", "n", PLAIN);
        assert!(
            grouped.contains("~ model.layers.{0-3}.mlp.weight  [F16 (8)] → [BF16 (8)]  (×4)"),
            "{grouped}"
        );
        assert_eq!(
            grouped.matches(".mlp.weight").count(),
            1,
            "should be one line:\n{grouped}"
        );
        // The counts line still reports the true total (4 changed).
        assert!(grouped.contains("tensors: -0 +0 ~4"), "{grouped}");

        // `--full` (group off): every layer listed, no count suffix.
        let full = r.render(
            "o",
            "n",
            DiffOpts {
                group: false,
                ..PLAIN
            },
        );
        assert_eq!(
            full.matches(".mlp.weight").count(),
            4,
            "should list all four:\n{full}"
        );
        assert!(full.contains("~ model.layers.0.mlp.weight"), "{full}");
        assert!(!full.contains("(×"), "no count suffix when full:\n{full}");
    }

    #[test]
    fn compact_int_ranges_merges_runs() {
        assert_eq!(compact_int_ranges(&[0, 1, 2, 3]), "0-3");
        assert_eq!(compact_int_ranges(&[0, 1, 2, 5, 7, 8]), "0-2,5,7-8");
        assert_eq!(compact_int_ranges(&[4]), "4");
    }

    #[test]
    fn templatize_replaces_digit_runs() {
        let (t, idx) = templatize("model.layers.12.experts.3.weight");
        assert_eq!(t, "model.layers.{}.experts.{}.weight");
        assert_eq!(idx, ["12", "3"]);
    }

    #[test]
    fn pretty_json_is_width_aware() {
        // Small object fits on one line — not blown up.
        assert!(!pretty_json(r#"{"a":1,"b":2}"#, 80).unwrap().contains('\n'));
        // Too wide → expanded, one field per line.
        let big = format!(
            r#"{{"items":[{}]}}"#,
            (0..40)
                .map(|i| format!(r#""x{i}""#))
                .collect::<Vec<_>>()
                .join(",")
        );
        assert!(pretty_json(&big, 80).unwrap().contains('\n'));
        // A nested small object stays inline even inside an expanded parent.
        let nested = format!(r#"{{"pad":"{}","q":{{"bits":[3,3,3]}}}}"#, "z".repeat(90));
        assert!(
            pretty_json(&nested, 80)
                .unwrap()
                .contains(r#""q": {"bits": [3, 3, 3]}"#),
            "{:?}",
            pretty_json(&nested, 80)
        );
        assert!(pretty_json("not json", 80).is_none());
        assert!(pretty_json("d5f887bb41", 80).is_none());
    }

    #[test]
    fn changed_large_json_metadata_renders_as_a_line_diff() {
        let mv = |v: &str| MetaVal {
            value: v.to_string(),
            value_type: "string".to_string(),
        };
        // A JSON object large enough to expand, with one field changed.
        let obj = |val: &str| {
            let mut fields: Vec<String> = (0..20).map(|i| format!(r#""k{i}":"x""#)).collect();
            fields.push(format!(r#""v":"{val}""#));
            format!("{{{}}}", fields.join(","))
        };
        let old = summary(&[], &[("spec", mv(&obj("old")))]);
        let new = summary(&[], &[("spec", mv(&obj("new")))]);
        let out = compare(&old, &new).render("o", "n", PLAIN);
        assert!(out.contains("~ spec:"), "{out}"); // line-diff header, not `= … → …`
        let line = |sign: &str, needle: &str| {
            out.lines()
                .any(|l| l.trim_start().starts_with(sign) && l.contains(needle))
        };
        assert!(line("- ", r#""v": "old""#), "{out}");
        assert!(line("+ ", r#""v": "new""#), "{out}");
    }

    #[test]
    fn long_changed_line_is_windowed_around_its_difference() {
        // A single changed line whose two versions share a long prefix: each is
        // windowed around the divergence (AAA/BBB visible), not clipped to the
        // shared prefix.
        let old = format!("{{\n  \"x\": \"{}AAA\"\n}}", "z".repeat(300));
        let new = format!("{{\n  \"x\": \"{}BBB\"\n}}", "z".repeat(300));
        let mut s = String::new();
        write_meta_line_diff(&mut s, &old, &new, false);
        assert!(
            s.contains("AAA") && s.contains("BBB") && s.contains('…'),
            "{s}"
        );
    }

    #[test]
    fn large_metadata_line_diff_is_capped() {
        let old = (0..100)
            .map(|i| format!("line {i} old"))
            .collect::<Vec<_>>()
            .join("\n");
        let new = (0..100)
            .map(|i| format!("line {i} new"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut s = String::new();
        write_meta_line_diff(&mut s, &old, &new, false);
        let n = s.lines().count();
        assert!(n <= MAX_META_DIFF_LINES + 1, "capped, got {n} lines");
        assert!(s.contains("more diff line"), "{s}");
    }

    #[test]
    fn multiline_metadata_shows_a_line_diff() {
        let old = "{\n  \"k\": 1,\n  \"v\": \"aaa\"\n}";
        let new = "{\n  \"k\": 1,\n  \"v\": \"bbb\"\n}";
        let mut s = String::new();
        write_meta_line_diff(&mut s, old, new, false);
        // The changed line shows as -/+, the unchanged line as context.
        assert!(s.contains("- ") && s.contains("aaa"), "{s}");
        assert!(s.contains("+ ") && s.contains("bbb"), "{s}");
        assert!(s.contains("\"k\": 1"), "{s}");
    }

    #[test]
    fn quote_diff_windows_around_the_first_difference() {
        // Long values sharing a prefix: both are windowed so the diverging token
        // shows (a plain head-truncation would print the same prefix for both).
        let old = format!("{}ALPHA-tail", "x".repeat(100));
        let new = format!("{}BETA-tail", "x".repeat(100));
        let (o, n) = quote_diff(&old, &new);
        assert!(o.starts_with("\"…") && o.contains("ALPHA"), "{o}");
        assert!(n.starts_with("\"…") && n.contains("BETA"), "{n}");
        // Short values are shown in full, no ellipsis.
        assert_eq!(
            quote_diff("d5f887bb41", "46c41d7cf4"),
            ("\"d5f887bb41\"".to_string(), "\"46c41d7cf4\"".to_string())
        );
    }

    #[test]
    fn quote_trunc_flattens_and_truncates() {
        assert_eq!(quote_trunc("a\nb"), "\"a b\"");
        let long = "x".repeat(100);
        let q = quote_trunc(&long);
        assert!(q.starts_with('"') && q.ends_with("…\""));
        assert_eq!(q.chars().count(), 60 + 3); // 60 chars + ellipsis + 2 quotes
    }

    #[test]
    fn fmt_delta_trims_and_switches_to_scientific() {
        assert_eq!(fmt_delta(0.0), "0");
        assert_eq!(fmt_delta(7.0), "7");
        assert_eq!(fmt_delta(0.5), "0.5");
        assert_eq!(fmt_delta(0.001_953_125), "0.001953");
        assert_eq!(fmt_delta(1e-8), "1.000e-8");
    }

    const PLAIN: DiffOpts = DiffOpts {
        color: false,
        metadata: true,
        group: true,
        values: false,
        histogram: false,
        filtered: false,
    };

    const COLOUR: DiffOpts = DiffOpts {
        color: true,
        metadata: true,
        group: true,
        values: false,
        histogram: false,
        filtered: false,
    };

    #[test]
    fn header_colours_old_red_new_green() {
        let s = summary(&[("a", sig("F16", &[2]))], &[]);
        let out = compare(&s, &s).render("OLD", "NEW", COLOUR);
        assert!(out.contains("\x1b[31m--- OLD\x1b[0m"), "{out}");
        assert!(out.contains("\x1b[32m+++ NEW\x1b[0m"), "{out}");
    }

    fn vd(differing: u64, elements: u64, max_abs: f64, mean_abs: f64) -> ValueDiff {
        ValueDiff {
            elements,
            differing,
            max_abs,
            mean_abs,
            nonfinite_mismatch: 0,
        }
    }

    #[test]
    fn focus_differs_predicate() {
        let a = sig("F16", &[2, 2]);
        let b = sig("BF16", &[2, 2]);
        // same sig, identical values → not a difference
        assert!(!tensor_focus_differs(
            Some(&a),
            Some(&a),
            Some(&ValueCmp::Identical)
        ));
        // same sig, values differ → a difference
        assert!(tensor_focus_differs(
            Some(&a),
            Some(&a),
            Some(&ValueCmp::Differ(vd(1, 4, 0.5, 0.1)))
        ));
        // differing sig → a difference regardless of values
        assert!(tensor_focus_differs(
            Some(&a),
            Some(&b),
            Some(&ValueCmp::Identical)
        ));
        // present on one side only → a difference
        assert!(tensor_focus_differs(Some(&a), None, None));
    }

    #[test]
    fn focus_render_same_sig_values_differ() {
        let a = sig("U8", &[4]);
        let out = render_tensor_focus(
            "old",
            "new",
            "w",
            Some(&a),
            Some(&a),
            Some(&ValueCmp::Differ(vd(4, 4, 7.0, 7.0))),
            false,
        );
        assert!(out.contains("~ w  [U8 (4)]  (values differ)"), "{out}");
        assert!(
            out.contains("values: 4 of 4 differ  (max |Δ| 7, mean |Δ| 7)"),
            "{out}"
        );
    }

    #[test]
    fn focus_render_identical_and_added_and_shape_skip() {
        let a = sig("F32", &[4]);
        let ident = render_tensor_focus(
            "o",
            "n",
            "w",
            Some(&a),
            Some(&a),
            Some(&ValueCmp::Identical),
            false,
        );
        assert!(ident.contains("= w  [F32 (4)]  (identical)"), "{ident}");

        let added = render_tensor_focus("o", "n", "w", None, Some(&a), None, false);
        assert!(added.contains("+ w  [F32 (4)]  (only in new)"), "{added}");

        let b = sig("F32", &[8]);
        let reshape = render_tensor_focus(
            "o",
            "n",
            "w",
            Some(&a),
            Some(&b),
            Some(&ValueCmp::Skipped("shapes differ".to_string())),
            false,
        );
        assert!(reshape.contains("~ w  [F32 (4)] → [F32 (8)]"), "{reshape}");
        assert!(
            reshape.contains("values: not compared (shapes differ)"),
            "{reshape}"
        );
    }

    // ---- TensorFilter ----

    fn glob(p: &str) -> Pattern {
        Pattern::new(p).unwrap()
    }

    #[test]
    fn filter_name_glob_matches_any() {
        let f = TensorFilter {
            names: NameFilter {
                include: vec![glob("*.mlp.*.weight"), glob("*.norm.weight")],
                exclude: vec![],
            },
            ..Default::default()
        };
        assert!(f.is_active());
        let s = sig("F16", &[4, 4]);
        assert!(f.matches("model.layers.0.mlp.down_proj.weight", Some(&s), Some(&s)));
        assert!(f.matches("model.norm.weight", Some(&s), None));
        assert!(!f.matches("model.embed_tokens.weight", Some(&s), Some(&s)));
    }

    #[test]
    fn filter_names_exact() {
        let f = TensorFilter {
            names_exact: Some(["a.w", "b.w"].iter().map(ToString::to_string).collect()),
            ..Default::default()
        };
        let s = sig("F16", &[2]);
        assert!(f.matches("a.w", Some(&s), Some(&s)));
        assert!(!f.matches("c.w", Some(&s), Some(&s)));
    }

    #[test]
    fn filter_dtype_glob_is_case_insensitive_and_either_side() {
        let f = TensorFilter {
            dtype: Some(glob("F*")),
            ..Default::default()
        };
        assert!(f.matches("w", Some(&sig("F16", &[2])), Some(&sig("F16", &[2]))));
        assert!(f.matches("w", Some(&sig("f32", &[2])), None)); // lowercase stored dtype
        assert!(!f.matches("w", Some(&sig("BF16", &[2])), Some(&sig("I8", &[2]))));
        // dtype changed F16 → BF16 still matches: the OLD side is F16.
        assert!(f.matches("w", Some(&sig("F16", &[2])), Some(&sig("BF16", &[2]))));
    }

    #[test]
    fn filter_shape_glob_star_one_dim_starstar_any() {
        // `*` matches exactly one dimension (of any size).
        let one = TensorFilter {
            shape: Some(glob("768/*")),
            ..Default::default()
        };
        assert!(one.matches("w", Some(&sig("F16", &[768, 2048])), None));
        assert!(!one.matches("w", Some(&sig("F16", &[768, 2048, 4])), None)); // rank 3
        assert!(!one.matches("w", Some(&sig("F16", &[768])), None)); // rank 1

        // `**` matches any number of dimensions.
        let any = TensorFilter {
            shape: Some(glob("768/**")),
            ..Default::default()
        };
        assert!(any.matches("w", Some(&sig("F16", &[768, 2048])), None));
        assert!(any.matches("w", Some(&sig("F16", &[768, 2048, 4])), None));

        // Trailing dimension at any rank.
        let tail = TensorFilter {
            shape: Some(glob("**/2048")),
            ..Default::default()
        };
        assert!(tail.matches("w", Some(&sig("F16", &[768, 2048])), None));
        assert!(tail.matches("w", Some(&sig("F16", &[6, 3, 2048])), None));
        assert!(!tail.matches("w", Some(&sig("F16", &[2048, 6])), None));
    }

    #[test]
    fn filter_constraints_compose_with_and() {
        let f = TensorFilter {
            names: NameFilter {
                include: vec![glob("*.down_proj.weight")],
                exclude: vec![],
            },
            dtype: Some(glob("BF16")),
            ..Default::default()
        };
        let bf = sig("BF16", &[2048, 768]);
        let f16 = sig("F16", &[2048, 768]);
        assert!(f.matches("model.layers.0.mlp.down_proj.weight", Some(&bf), Some(&bf)));
        assert!(!f.matches(
            "model.layers.0.mlp.down_proj.weight",
            Some(&f16),
            Some(&f16)
        )); // dtype fails
        assert!(!f.matches("model.layers.0.mlp.gate_proj.weight", Some(&bf), Some(&bf))); // name fails
    }

    #[test]
    fn filter_apply_restricts_both_sides_and_keeps_add_remove() {
        let mut old = summary(
            &[
                ("keep.down_proj.weight", sig("BF16", &[8, 4])),
                ("skip.gate_proj.weight", sig("BF16", &[8, 4])),
                ("only_old.down_proj.weight", sig("BF16", &[8, 4])),
            ],
            &[],
        );
        let mut new = summary(
            &[
                ("keep.down_proj.weight", sig("BF16", &[8, 4])),
                ("skip.gate_proj.weight", sig("BF16", &[8, 4])),
                ("only_new.down_proj.weight", sig("BF16", &[8, 4])),
            ],
            &[],
        );
        let f = TensorFilter {
            names: NameFilter {
                include: vec![glob("*.down_proj.weight")],
                exclude: vec![],
            },
            ..Default::default()
        };
        f.apply(&mut old, &mut new);
        assert_eq!(
            old.tensors.keys().cloned().collect::<Vec<_>>(),
            vec!["keep.down_proj.weight", "only_old.down_proj.weight"]
        );
        assert_eq!(
            new.tensors.keys().cloned().collect::<Vec<_>>(),
            vec!["keep.down_proj.weight", "only_new.down_proj.weight"]
        );
        // The diff over the filtered subset: one unchanged, one removed, one added.
        let r = compare(&old, &new);
        assert_eq!(r.tensors_unchanged, 1);
        assert_eq!(r.tensors_removed.len(), 1);
        assert_eq!(r.tensors_added.len(), 1);
    }

    /// **The grouped report and the terminal's own lines agree.**
    ///
    /// The browser is sent grouped rows rather than grouping them itself, because the templating rule is
    /// subtle and already exists here — driving what the terminal prints. This is that claim as a test:
    /// the same names, the same counts, in the same order as the rendered lines.
    #[test]
    fn the_grouped_report_matches_the_rendered_lines() {
        let names: Vec<String> = (0..62)
            .map(|n| format!("model.layers.{n}.inv_freq_default"))
            .collect();
        let mut old_t: Vec<(&str, TensorSig)> = Vec::new();
        let mut new_t: Vec<(&str, TensorSig)> = Vec::new();
        // 62 layers of one name, retyped — the case that fills a screen with rows differing by a number.
        for n in &names {
            old_t.push((n, sig("F32", &[8, 128])));
            new_t.push((n, sig("F16", &[8, 128])));
        }
        // …and one name that stands alone, added on the new side.
        new_t.push(("lm_head.weight", sig("F16", &[4, 4])));
        let (old, new) = (summary(&old_t, &[]), summary(&new_t, &[]));
        let report = compare(&old, &new);
        let grouped = report.grouped();

        assert_eq!(grouped.tensors_changed.len(), 1, "62 rows collapse to one");
        assert_eq!(
            grouped.tensors_changed[0].name,
            "model.layers.{0-61}.inv_freq_default"
        );
        assert_eq!(grouped.tensors_changed[0].count, 62);
        assert_eq!(grouped.tensors_added.len(), 1);
        assert_eq!(grouped.tensors_added[0].name, "lm_head.weight");
        assert_eq!(
            grouped.tensors_added[0].count, 1,
            "a lone name stands for itself"
        );

        // And the terminal prints those very lines, so the two surfaces cannot collapse differently.
        let rendered = report.render("o", "n", PLAIN);
        assert!(
            rendered.contains("~ model.layers.{0-61}.inv_freq_default"),
            "{rendered}"
        );
        assert!(rendered.contains("(×62)"), "{rendered}");
        assert!(rendered.contains("+ lm_head.weight"), "{rendered}");
    }

    /// A folded family reports the fold once — and would report nothing if its members disagreed, since
    /// one member's count would be a claim about the rest.
    #[test]
    fn a_grouped_row_reports_a_fold_only_when_every_member_agrees() {
        let rules = NameMap::from_pairs(fused_layout_rules()).expect("compile");
        let unfused: Vec<(&str, TensorSig)> = vec![
            (
                "model.layers.0.block_sparse_moe.experts.0.w2.weight",
                sig("U8", &[8, 4]),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.1.w2.weight",
                sig("U8", &[8, 4]),
            ),
            (
                "model.layers.1.block_sparse_moe.experts.0.w2.weight",
                sig("U8", &[8, 4]),
            ),
            (
                "model.layers.1.block_sparse_moe.experts.1.w2.weight",
                sig("U8", &[8, 4]),
            ),
        ];
        let fused: Vec<(&str, TensorSig)> = vec![
            (
                "model.layers.0.block_sparse_moe.experts.down_proj.weight",
                sig("U8", &[2, 8, 4]),
            ),
            (
                "model.layers.1.block_sparse_moe.experts.down_proj.weight",
                sig("U8", &[2, 8, 4]),
            ),
        ];
        let mut old = summary(&unfused, &[]);
        rules.remap_summary_with(&mut old, OnCollision::Fold);
        let new = summary(&fused, &[]);
        let grouped = compare(&old, &new).grouped();
        assert_eq!(
            grouped.tensors_changed.len(),
            1,
            "the two layers are one family"
        );
        assert_eq!(grouped.tensors_changed[0].count, 2);
        assert_eq!(
            grouped.tensors_changed[0].fold,
            Some([2, 1]),
            "both members folded two into one"
        );
    }

    /// **An unfused checkpoint lines up with its fused counterpart.**
    ///
    /// The reported case: two layouts of one model share no tensor name, so a plain comparison reports
    /// every tensor of both sides as one-sided — 80,107 against 933 — which is true and answers nothing.
    /// The rules drop the expert index and rename the layout synonyms; the fold is what makes 256
    /// per-expert tensors *correspond to* the one fused tensor that holds them rather than being 255
    /// removals and a change.
    #[test]
    fn the_unfused_layout_folds_onto_the_fused_one() {
        let rules = NameMap::from_pairs(fused_layout_rules()).expect("the canonical rules compile");
        // Two experts of one layer, in the unfused (Mixtral-style) naming.
        let unfused: Vec<(&str, TensorSig)> = vec![
            (
                "model.layers.0.block_sparse_moe.experts.0.w2.weight",
                sig("U8", &[8, 4]),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.1.w2.weight",
                sig("U8", &[8, 4]),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.0.w1.weight",
                sig("U8", &[4, 8]),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.1.w1.weight",
                sig("U8", &[4, 8]),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.0.w3.weight",
                sig("U8", &[4, 8]),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.1.w3.weight",
                sig("U8", &[4, 8]),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                sig("BF16", &[4, 4]),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                sig("BF16", &[4, 4]),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                sig("BF16", &[4, 4]),
            ),
        ];
        let mut old = summary(&unfused, &[]);
        rules.remap_summary_with(&mut old, OnCollision::Fold);

        // Three names, each standing for what folded onto it.
        assert_eq!(
            old.tensors.keys().cloned().collect::<Vec<_>>(),
            vec![
                "model.layers.0.block_sparse_moe.experts.down_proj.weight",
                "model.layers.0.block_sparse_moe.experts.gate_up_proj.weight",
                "model.layers.0.self_attn.qkv_proj.weight",
            ]
        );
        let folds = old.folds();
        assert_eq!(
            folds.get("model.layers.0.block_sparse_moe.experts.down_proj.weight"),
            Some(&2)
        );
        // gate and up, for two experts: four tensors behind one fused name.
        assert_eq!(
            folds.get("model.layers.0.block_sparse_moe.experts.gate_up_proj.weight"),
            Some(&4)
        );
        assert_eq!(
            folds.get("model.layers.0.self_attn.qkv_proj.weight"),
            Some(&3)
        );
        // Nothing was dropped: the folded totals are the sum of the parts (the helper gives each
        // tensor one byte per element).
        // 2 × (8·4) + 4 × (4·8) + 3 × (4·4) — the parts, all still counted.
        assert_eq!(
            old.total_params(),
            64 + 128 + 48,
            "every part still counted"
        );

        // And against the fused side, the report says how many folded onto each row.
        let fused: Vec<(&str, TensorSig)> = vec![
            (
                "model.layers.0.block_sparse_moe.experts.down_proj.weight",
                sig("U8", &[2, 8, 4]),
            ),
            (
                "model.layers.0.block_sparse_moe.experts.gate_up_proj.weight",
                sig("U8", &[2, 8, 8]),
            ),
            (
                "model.layers.0.self_attn.qkv_proj.weight",
                sig("BF16", &[12, 4]),
            ),
        ];
        let mut new = summary(&fused, &[]);
        // The rules are applied to both sides, and are a no-op on one already fused.
        rules.remap_summary_with(&mut new, OnCollision::Fold);
        assert!(new.folds().is_empty(), "nothing to fold on the fused side");

        let report = compare(&old, &new);
        assert_eq!(
            report.tensors_changed.len(),
            3,
            "three rows, not 9 removals"
        );
        assert!(report.tensors_removed.is_empty() && report.tensors_added.is_empty());
        let rendered = report.render("unfused", "fused", PLAIN);
        assert!(rendered.contains("(×2 → ×1)"), "{rendered}");
        assert!(rendered.contains("(×4 → ×1)"), "{rendered}");
        assert!(rendered.contains("(×3 → ×1)"), "{rendered}");
    }

    /// The rules are a no-op on a checkpoint already in the fused layout, which is what lets them be
    /// applied to both sides without asking which is which.
    #[test]
    fn aligning_an_already_fused_checkpoint_changes_nothing() {
        let rules = NameMap::from_pairs(fused_layout_rules()).expect("compile");
        let fused: Vec<(&str, TensorSig)> = vec![
            (
                "model.layers.3.block_sparse_moe.experts.down_proj.weight",
                sig("U8", &[2, 8, 4]),
            ),
            (
                "model.layers.3.block_sparse_moe.experts.down_proj.qscale",
                sig("F16", &[2, 8]),
            ),
            (
                "model.layers.3.block_sparse_moe.gate.bias",
                sig("F32", &[2]),
            ),
            (
                "model.layers.3.self_attn.qkv_proj.weight",
                sig("BF16", &[12, 4]),
            ),
            ("lm_head.weight", sig("F16", &[4, 4])),
        ];
        let before = summary(&fused, &[]);
        let mut after = summary(&fused, &[]);
        rules.remap_summary_with(&mut after, OnCollision::Fold);
        assert_eq!(
            after.tensors.keys().cloned().collect::<Vec<_>>(),
            before.tensors.keys().cloned().collect::<Vec<_>>()
        );
        assert_eq!(after.total_params(), before.total_params());
    }

    /// **A filter narrows the totals with the tensors.**
    ///
    /// It used to narrow only the signatures: the summed size and parameter count were computed once at
    /// load and stayed behind, so a report about nineteen of 117,664 tensors was headed
    /// `size: 1966.5 GiB → 451.8 GiB` — the checkpoints' sizes, which is a true statement about
    /// something the reader is not looking at. The `summary` helper gives each tensor one byte per
    /// element, so these numbers are the element counts of what survived the filter.
    #[test]
    fn a_filtered_comparison_totals_only_the_tensors_it_kept() {
        let tensors = |dt: &str| {
            [
                ("keep.down_proj.weight", sig(dt, &[8, 4])), // 32
                ("skip.gate_proj.weight", sig(dt, &[100])),  // 100, and excluded
            ]
        };
        let (mut old, mut new) = (
            summary(&tensors("BF16"), &[]),
            summary(&tensors("F16"), &[]),
        );
        assert_eq!(old.total_params(), 132, "before: both tensors");
        assert_eq!(old.total_bytes(), 132);

        let f = TensorFilter {
            names: NameFilter {
                include: vec![glob("*.down_proj.weight")],
                exclude: vec![],
            },
            ..Default::default()
        };
        f.apply(&mut old, &mut new);
        assert_eq!(old.total_params(), 32, "after: the kept tensor only");
        assert_eq!(new.total_bytes(), 32);
        // And the report carries those, not the checkpoints'.
        let r = compare(&old, &new);
        assert_eq!((r.old_bytes, r.new_bytes), (32, 32));
        assert_eq!((r.old_params, r.new_params), (32, 32));
        // Which is why the line is labelled: `size:` above a subset reads as the checkpoint's size.
        let filtered = r.render(
            "o",
            "n",
            DiffOpts {
                filtered: true,
                ..PLAIN
            },
        );
        assert!(
            filtered.contains("size (filtered subset): 32 B (unchanged)"),
            "{filtered}"
        );
        // Unfiltered, the label stays bare.
        assert!(
            r.render("o", "n", PLAIN).contains("\nsize: 32 B"),
            "{r:?}",
            r = r.render("o", "n", PLAIN)
        );
    }

    /// A rename moves each tensor's footprint with its name, so the totals still describe the tensors
    /// the summary now has — and two names collapsing onto one stop being counted twice.
    #[test]
    fn renaming_carries_the_footprints() {
        let mut sum = summary(
            &[
                ("old.a.weight", sig("F16", &[10])),
                ("old.b.weight", sig("F16", &[5])),
            ],
            &[],
        );
        assert_eq!(sum.total_params(), 15);
        let map = NameMap::from_pairs([(r"^old\.".to_string(), "new.".to_string())])
            .expect("a valid rule");
        map.remap_summary(&mut sum);
        assert_eq!(
            sum.tensors.keys().cloned().collect::<Vec<_>>(),
            vec!["new.a.weight", "new.b.weight"]
        );
        assert_eq!(
            sum.footprints.keys().cloned().collect::<Vec<_>>(),
            vec!["new.a.weight", "new.b.weight"],
            "the footprints follow the names, or the totals describe names that are gone"
        );
        assert_eq!(sum.total_params(), 15);
    }

    #[test]
    fn name_schema_collapses_layers_and_experts() {
        let mut names = Vec::new();
        for l in 0..3 {
            for e in 0..2 {
                names.push(format!("model.layers.{l}.experts.{e}.down_proj.weight"));
                names.push(format!("model.layers.{l}.experts.{e}.gate_proj.weight"));
            }
        }
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let schema = name_schema(&refs);
        // Two templates (down / gate), each covering 3 layers × 2 experts = 6.
        assert_eq!(schema.len(), 2);
        assert!(
            schema.contains(&(
                "model.layers.{0-2}.experts.{0-1}.down_proj.weight".to_string(),
                6
            )),
            "{schema:?}"
        );
        assert!(
            schema.contains(&(
                "model.layers.{0-2}.experts.{0-1}.gate_proj.weight".to_string(),
                6
            )),
            "{schema:?}"
        );
    }

    #[test]
    fn filter_inactive_is_noop() {
        let f = TensorFilter::default();
        assert!(!f.is_active());
        assert_eq!(f.describe(), None);
        let mut a = summary(&[("w", sig("F16", &[2]))], &[]);
        let mut b = summary(&[("w", sig("F16", &[2]))], &[]);
        f.apply(&mut a, &mut b);
        assert_eq!(a.tensors.len(), 1); // untouched
    }
}
