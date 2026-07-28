//! Serializable wire shapes for the web API — projections of internal types that
//! either leak the server's absolute paths (the file tree) or carry non-serde /
//! heavy fields (`Duration`, raw bits) we don't want on the JSON contract. The
//! tensor tree, layout, and reports serialize directly from core (no DTO needed).

use std::path::Path;

use serde::Serialize;

use crate::filetree::{FileKind, FileNode, IndexMembership, ShardTensors};
use crate::sample::{HistBins, Histogram, Sample, SampleMode, Stats, ViewDtype};

/// A file-tree node with every `path` relativized to the checkpoint root (never
/// leak the server's absolute paths) — mirrors [`crate::filetree::FileNode`]. The
/// client flattens/folds this the way `filetree::flatten` does.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum WebFileNode {
    Dir {
        name: String,
        path: String,
        size: u64,
        files: usize,
        children: Vec<Self>,
    },
    File {
        name: String,
        path: String,
        size: u64,
        file_kind: FileKind,
        /// What the model reads out of this file, for a shard — `null` otherwise.
        shard: Option<ShardTensors>,
        /// Fraction of the largest file's size, for the proportional bar.
        size_share: f64,
        /// Whether the index declares this file — `null` when it can't apply.
        index: Option<IndexMembership>,
    },
}

impl WebFileNode {
    /// Project a local `FileNode` tree into the web shape, making each `path`
    /// relative to `root`.
    pub(crate) fn from_node(node: &FileNode, root: &Path) -> Self {
        match node {
            FileNode::Dir {
                name,
                path,
                children,
                size,
                files,
                ..
            } => Self::Dir {
                name: name.clone(),
                path: rel(path, root),
                size: *size,
                files: *files,
                children: children.iter().map(|c| Self::from_node(c, root)).collect(),
            },
            FileNode::File {
                name,
                path,
                size,
                kind,
                shard,
                size_share,
                index,
            } => Self::File {
                name: name.clone(),
                path: rel(path, root),
                size: *size,
                file_kind: *kind,
                shard: *shard,
                size_share: *size_share,
                index: *index,
            },
        }
    }
}

fn rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// Whole-tensor statistics (`sample::Stats`) with `Duration` → `elapsed_ms`.
#[derive(Serialize, Clone)]
pub(crate) struct StatsDto {
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub zeros: u64,
    pub nonfinite: u64,
    pub zero_fraction: f64,
    pub elapsed_ms: f64,
}

impl From<&Stats> for StatsDto {
    fn from(s: &Stats) -> Self {
        Self {
            count: s.count,
            min: s.min,
            max: s.max,
            mean: s.mean,
            std: s.std,
            zeros: s.zeros,
            nonfinite: s.nonfinite,
            zero_fraction: s.zero_fraction(),
            elapsed_ms: s.elapsed.as_secs_f64() * 1000.0,
        }
    }
}

/// A sampled value grid (`sample::Sample`) — the heatmap / slice payload. Raw
/// stored bits are included only when asked for (the hex/oct/bin value view): each
/// cell as a zero-padded hex string of `raw_width` bits, so the client can reformat
/// to any base via `BigInt` without u64 precision loss.
#[derive(Serialize)]
pub(crate) struct SampleDto {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub values: Vec<Vec<f64>>,
    pub min: f64,
    pub max: f64,
    pub total_rows: usize,
    pub total_cols: usize,
    pub slices: usize,
    pub slice: usize,
    pub display_shape: Vec<usize>,
    pub view: String,
    pub mode: String,
    pub overridable: bool,
    /// Whether these values are integers, and if so whether they're signed. JSON
    /// numbers are f64, which cannot carry a 64-bit integer exactly — so for an
    /// integer view the client formats the decimal from `raw` via `BigInt` (as it
    /// already does for hex/oct/bin) rather than from the lossy `values`.
    pub integer: bool,
    pub signed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_width: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<Vec<Vec<String>>>,
}

impl SampleDto {
    /// `dtype` is the tensor's stored dtype, needed to tell whether the values under
    /// this view are integers (and signed) — see the `integer`/`signed` fields.
    pub(crate) fn from_sample(s: &Sample, dtype: &str, include_raw: bool) -> Self {
        let integer = s.view.is_integer(dtype);
        let signed = s.view.is_signed_integer(dtype);
        let raw_width = s.raw.iter().flatten().next().map(|b| b.width);
        // Always ship the bits for an integer view: they're the only exact
        // representation of a 64-bit value on a JSON wire.
        let raw = (include_raw || integer).then(|| {
            s.raw
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|b| {
                            let hex_digits = (b.width as usize).div_ceil(4).max(1);
                            format!("{:0hex_digits$x}", b.bits)
                        })
                        .collect()
                })
                .collect()
        });
        Self {
            rows: s.rows.clone(),
            cols: s.cols.clone(),
            values: s.values.clone(),
            min: s.min,
            max: s.max,
            total_rows: s.total_rows,
            total_cols: s.total_cols,
            slices: s.slices,
            slice: s.slice,
            display_shape: s.display_shape.clone(),
            view: view_label(s.view),
            mode: mode_label(&s.mode),
            overridable: s.overridable,
            integer,
            signed,
            raw_width,
            raw,
        }
    }
}

/// A value histogram (`sample::Histogram`) with `Duration` → `elapsed_ms`.
#[derive(Serialize)]
pub(crate) struct HistogramDto {
    pub bins: HistBinsDto,
    pub counts: Vec<u64>,
    pub total: u64,
    pub nonfinite: u64,
    pub elapsed_ms: f64,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum HistBinsDto {
    /// Integer bins: bin `i` covers `[start + i*step, start + (i+1)*step)`.
    Int { start: i64, step: i64 },
    /// Equal-width bins spanning `[lo, hi]`.
    Range { lo: f64, hi: f64 },
}

impl From<&Histogram> for HistogramDto {
    fn from(h: &Histogram) -> Self {
        Self {
            bins: match h.bins {
                HistBins::IntBins { start, step } => HistBinsDto::Int { start, step },
                HistBins::Range { lo, hi } => HistBinsDto::Range { lo, hi },
            },
            counts: h.counts.clone(),
            total: h.total,
            nonfinite: h.nonfinite,
            elapsed_ms: h.elapsed.as_secs_f64() * 1000.0,
        }
    }
}

/// The `?dtype=` value that re-selects a view (`stored` when using the real dtype).
pub(crate) fn view_label(v: ViewDtype) -> String {
    v.label().unwrap_or("stored").to_string()
}

fn mode_label(m: &SampleMode) -> String {
    match m {
        SampleMode::Grid => "grid",
        SampleMode::GridMax => "abs-max",
        SampleMode::Edges { .. } => "edges",
        SampleMode::Window { .. } => "window",
    }
    .to_string()
}

/// The stats screen's S3 section, as the TUI words it.
///
/// The phrases (`checksums`, `tags`, `modified`) come from the *same* core functions
/// the TUI renders, rather than being re-derived in TypeScript: their rules are subtle
/// — "unavailable (permission)" versus "none", a single date versus an "earliest –
/// latest" span — and two implementations of that would drift. The browser only prints
/// what it's given. Per-object rows come from `footprint.S3.objects`, which the stats
/// payload already carries.
#[derive(serde::Serialize)]
pub(crate) struct S3SummaryDto {
    pub count: usize,
    pub total_bytes: u64,
    /// Stored-checksum coverage, e.g. `SHA256 on 126 of 1155` or `none`.
    pub checksums: String,
    /// `1155 of 1155 present`.
    pub etags: String,
    /// `none`, `12 of 1155 tagged`, or `unavailable (permission)`.
    pub tags: String,
    /// A single date, or `earliest – latest`; absent when no object reported one.
    pub modified: Option<String>,
    /// How many objects carry user metadata (`x-amz-meta-*`); 0 when none do.
    pub user_meta_objects: usize,
    /// Per-object detail tail (`   2.2 GiB  etag …`), keyed by object key — the same
    /// string the TUI's folded per-object breakdown shows.
    pub object_detail: std::collections::BTreeMap<String, String>,
    /// Anything that went wrong while fetching the metadata (e.g. tags denied).
    pub warnings: Vec<String>,
}

impl S3SummaryDto {
    /// Build from the stats module's S3 view, or `None` for a checkpoint without one.
    pub(crate) fn from_stats(stats: &crate::stats::CheckpointStats) -> Option<Self> {
        let s3 = stats.s3()?;
        Some(Self {
            count: s3.count(),
            total_bytes: s3.total_bytes(),
            checksums: crate::stats::s3_checksums_phrase(s3),
            etags: format!("{} of {} present", s3.etags(), s3.count()),
            tags: crate::stats::s3_tags_phrase(s3),
            modified: crate::stats::s3_modified_phrase(s3),
            user_meta_objects: s3.with_user_meta(),
            object_detail: s3
                .objects
                .iter()
                .map(|o| (o.key.clone(), crate::stats::s3_object_detail(o)))
                .collect(),
            warnings: s3.warnings.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn file_node_relativizes_paths() {
        let root = PathBuf::from("/abs/root");
        let node = FileNode::Dir {
            name: "root".into(),
            path: root.clone(),
            expanded: true,
            size: 10,
            files: 1,
            children: vec![FileNode::File {
                name: "a.safetensors".into(),
                path: root.join("sub/a.safetensors"),
                size: 10,
                kind: FileKind::Checkpoint,
                shard: Some(ShardTensors {
                    tensors: 3,
                    params: 40,
                    params_share: 0.25,
                }),
                size_share: 0.5,
                index: Some(IndexMembership::Unlisted),
            }],
        };
        let web = WebFileNode::from_node(&node, &root);
        let json = serde_json::to_value(&web).unwrap();
        // Root path is empty (relative to itself); child path is root-relative,
        // never the absolute server path.
        assert_eq!(json["path"], "");
        assert_eq!(json["children"][0]["path"], "sub/a.safetensors");
        assert_eq!(json["children"][0]["kind"], "file");
        assert!(!json.to_string().contains("/abs/root"));
        // The shard annotation rides along, shaped as `types.ts` declares it.
        assert_eq!(json["children"][0]["shard"]["tensors"], 3);
        assert_eq!(json["children"][0]["shard"]["params"], 40);
        assert_eq!(json["children"][0]["shard"]["params_share"], 0.25);
    }

    #[test]
    // 25/100 is exactly representable, so the fraction is compared exactly on purpose.
    #[allow(clippy::float_cmp)]
    fn stats_dto_converts_duration_and_zero_fraction() {
        let stats = Stats {
            count: 100,
            min: -1.0,
            max: 2.0,
            mean: 0.5,
            std: 1.0,
            zeros: 25,
            nonfinite: 0,
            elapsed: Duration::from_millis(12),
        };
        let dto = StatsDto::from(&stats);
        assert_eq!(dto.zero_fraction, 0.25);
        assert!((dto.elapsed_ms - 12.0).abs() < 1e-6);
    }

    #[test]
    fn s3_summary_is_absent_without_an_s3_footprint() {
        let stats = crate::stats::CheckpointStats::compute(&[], None, None);
        assert!(S3SummaryDto::from_stats(&stats).is_none());
    }

    #[test]
    fn s3_summary_words_the_section_the_way_the_tui_does() {
        use crate::stats::{S3ObjectStat, S3Stats};
        let s3 = S3Stats {
            objects: vec![
                S3ObjectStat {
                    key: "a.weight".into(),
                    size: 2048,
                    etag: "abc".into(),
                    checksum: None,
                    last_modified: "2026-06-26T10:00:00+00:00".into(),
                    tags: Some(0),
                    user_meta: 1,
                },
                S3ObjectStat {
                    key: "__METADATA__".into(),
                    size: 1024,
                    etag: String::new(), // no ETag on this one
                    checksum: None,
                    last_modified: "2026-06-27T10:00:00+00:00".into(),
                    tags: Some(0),
                    user_meta: 0,
                },
            ],
            warnings: vec!["tags unavailable".into()],
        };
        let stats =
            crate::stats::CheckpointStats::compute(&[], None, None).with_s3(Some(s3.clone()));
        let dto = S3SummaryDto::from_stats(&stats).expect("an s3 footprint yields a summary");

        assert_eq!(dto.count, 2);
        assert_eq!(dto.total_bytes, 3072);
        // The phrases must be the core ones verbatim — that's the point of sending them
        // rather than re-deriving them in the browser.
        assert_eq!(dto.checksums, crate::stats::s3_checksums_phrase(&s3));
        assert_eq!(dto.tags, crate::stats::s3_tags_phrase(&s3));
        assert_eq!(dto.modified, crate::stats::s3_modified_phrase(&s3));
        assert_eq!(dto.etags, "1 of 2 present"); // the empty one doesn't count
        assert_eq!(dto.user_meta_objects, 1);
        assert_eq!(dto.object_detail.len(), 2);
        assert!(dto.object_detail["a.weight"].contains("etag abc"));
        assert_eq!(dto.warnings, vec!["tags unavailable".to_string()]);
    }

    #[test]
    fn histogram_dto_tags_bin_kind() {
        let hist = Histogram {
            bins: HistBins::Range { lo: 0.0, hi: 1.0 },
            counts: vec![3, 5],
            total: 8,
            nonfinite: 1,
            elapsed: Duration::from_millis(4),
        };
        let json = serde_json::to_value(HistogramDto::from(&hist)).unwrap();
        assert_eq!(json["bins"]["type"], "range");
        assert_eq!(json["counts"][1], 5);
        assert_eq!(json["nonfinite"], 1);
    }
}
