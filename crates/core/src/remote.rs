//! Read a remote checkpoint's **structure** (tensor names, dtypes, shapes) by
//! delegating to a machine that already has access to it, over one authenticated
//! SSH session ([`crate::sftp::RemoteSession`] — pure Rust, no external binary
//! runs locally or on the server):
//!
//! - a **safetensors directory or file** is read over SFTP — only each shard's
//!   header bytes are fetched, parsed with the local safetensors parser.
//! - an **`s3://…` cstorch checkpoint** is read by running a small
//!   `cerebras.pytorch` script (lazy load, metadata only) in the remote venv over
//!   an SSH exec channel — the one path that inherently needs Python/cstorch on
//!   the remote.
//!
//! Both share the one session, so a read — or `diff`'s two reads — costs a single
//! authentication / password prompt. Credentials/data stay on the remote (nothing
//! is copied locally). Metadata-only: only header/metadata bytes cross the wire.

use std::collections::HashSet;

use anyhow::{Context, Result, anyhow, bail};

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::progress::LoadProgress;
use crate::sftp::RemoteSession;
use crate::stats::{DiskUsage, ShardDisk};
use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

/// A remote read's result: tensors, metadata, `config.json`, and the shards'
/// on-disk footprint (all but the tensors optional).
type FetchedCheckpoint = (
    Vec<TensorInfo>,
    Vec<MetadataInfo>,
    Option<crate::config::ModelConfig>,
    Option<DiskUsage>,
    Vec<crate::health::HealthReport>,
);

/// What [`RemoteRead::read`] returns: the tensors, metadata, the shards' on-disk
/// footprint, and the index/file health — all from one pass (shard headers and the
/// index read once), so the health check reuses what the read already parsed
/// rather than fetching headers or the index a second time.
pub struct RemoteCheckpoint {
    pub tensors: Vec<TensorInfo>,
    pub metadata: Vec<MetadataInfo>,
    pub disk: Option<DiskUsage>,
    pub health: Vec<crate::health::HealthReport>,
    /// The underlying S3 objects' metadata — `Some` only for an `s3://` source
    /// (fetched best-effort by the remote script); `None` for a local/SFTP read.
    pub s3: Option<S3Meta>,
}

/// An object's additional stored checksum — a named `{algorithm, value}` pair
/// instead of a positional `(String, String)` tuple that could be stored swapped.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct S3Checksum {
    /// e.g. `"sha256"`, `"crc32c"`.
    pub algorithm: String,
    pub value: String,
}

/// One S3 object under a checkpoint's prefix, with the metadata `diff` compares.
/// Fetched best-effort by the remote dump script via boto3 (the remote's own AWS
/// credentials — nothing S3 happens locally).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct S3Object {
    /// Key relative to the checkpoint prefix, so two prefixes line up by shard.
    pub key: String,
    pub size: u64,
    pub etag: String,
    /// The object's additional stored checksum, when present.
    pub checksum: Option<S3Checksum>,
    pub last_modified: String,
    /// User-defined `x-amz-meta-*` metadata.
    pub user_meta: BTreeMap<String, String>,
    /// Object tags, or `None` when they couldn't be read (permission) — distinct
    /// from `Some(empty)` meaning "read, no tags".
    pub tags: Option<BTreeMap<String, String>>,
}

/// The S3 objects under an `s3://` checkpoint's prefix, plus any warnings raised
/// while fetching them (e.g. tags denied). `Some` only for an `s3://` source.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct S3Meta {
    pub objects: Vec<S3Object>,
    pub warnings: Vec<String>,
}

/// What a remote value comparison ([`RemoteRead::value_diff`]) computes per tensor.
#[derive(Clone, Debug)]
pub struct RemoteValueOpts {
    /// Compare element values: max/mean `|Δ|`, differing + non-finite-mismatch counts.
    pub values: bool,
    /// Compare value distributions: total-variation distance over a shared range.
    pub histogram: bool,
    /// Histogram bucket count; `None` uses the same default the local diff does.
    pub bins: Option<usize>,
    /// Ship the full per-bin histogram (the single-tensor `--tensor` table needs it);
    /// the bulk path leaves this off so only the small TVD summary crosses the wire.
    pub full_hist: bool,
    /// How many tensors to read + compare concurrently on the proxy. Reading each
    /// tensor's S3 object is latency-bound, so overlapping them is the main speedup;
    /// `1` runs sequentially (the safe fallback if cstorch dislikes threads).
    pub jobs: usize,
}

/// One tensor's remote value/distribution comparison. Fields are independently
/// optional: `values`/`hist_*` appear only when requested and computable; `error`
/// marks a per-tensor skip (e.g. materialising its data failed) — not a whole-diff
/// failure, so the other tensors still report.
#[derive(Debug, Default)]
pub struct RemoteTensorDiff {
    pub values: Option<crate::sample::ValueDiff>,
    /// `(tvd, bins)` summary for the bulk `--histogram` path.
    pub hist_shift: Option<(f64, usize)>,
    /// Full per-bin histogram for the single-tensor table (`full_hist`).
    pub hist_full: Option<crate::sample::HistogramDiff>,
    pub error: Option<String>,
}

/// I/O + timing metrics the proxy reports after a [`RemoteRead::value_diff`] run, so
/// the caller can show how much data was read and how fast. `elapsed_s` is measured
/// on the remote (compute + S3 read), so it excludes the SSH handshake and the
/// transfer of the small result JSON.
#[derive(Debug, Default, Clone)]
pub struct RemoteValueStats {
    /// Tensor pairs requested.
    pub tensors: usize,
    /// Pairs actually read + compared (excludes shape-mismatch / errored ones).
    pub compared: usize,
    /// Total tensor bytes read from S3 (both sides, at their stored dtype).
    pub bytes: u64,
    /// Wall-clock seconds on the remote for the whole comparison.
    pub elapsed_s: f64,
}

/// One tensor's repack-equivalence verification ([`RemoteRead::verify_repack`]):
/// does the old ("sparse", one 3-bit index per 16-bit word) and new ("dense",
/// `fold` 3-bit indices per word, folded along dim 0) encode the same indices? Plus
/// format sanity: `sparse_bad` / `dense_bad` count words whose bits above the used
/// range are non-zero (a non-zero count means the format assumption is wrong).
#[derive(Debug, Default, Clone)]
pub struct RepackResult {
    /// Logical 3-bit indices compared (`E × prod(inner_dims)`).
    pub elements: u64,
    /// Decoded indices that differ between old and new (0 ⇒ equivalent).
    pub differing: u64,
    /// Largest `|old − new|` over the decoded indices (1 ⇒ every difference is to an
    /// adjacent index — the signature of an independent re-quantization).
    pub max_delta: u32,
    /// Differing indices that differ by more than 1 (0 ⇒ all differences are ±1).
    pub differing_gt1: u64,
    /// Total `Σ|old − new|` over all indices.
    pub sum_abs: u64,
    /// Mean `|old − new|` per index (per parameter).
    pub mean_abs: f64,
    /// Mean decoded index on each side — near-equal means the *average value* is
    /// preserved even where individual indices moved.
    pub mean_old: f64,
    pub mean_new: f64,
    /// Old words with non-zero bits above `bits` (top-13-zero check; 0 ⇒ ok).
    pub sparse_bad: u64,
    /// New words with a non-zero bit above `fold*bits` (MSB check; 0 ⇒ ok).
    pub dense_bad: u64,
    /// The fold factor used (experts per packed word). `1` for a sparse↔sparse
    /// compare (the auto-detected `--values` path on packed expert weights).
    pub fold: usize,
    /// The index bit-width used to decode + format-check (derived from the codebook's
    /// centroid count for the auto path, else `16/fold`).
    pub bits: usize,
    /// Fraction of decoded indices that are 0 across both sides — the "amount of
    /// zeroes" that (with a sibling codebook) marks a tensor as sparse-packed.
    pub zero_frac: f64,
    /// Set when the top-bits format check failed (the words don't look like packed
    /// indices), so the tensor was compared as plain stored-dtype *values* instead —
    /// carries that float comparison so the report can fall back to it.
    pub fallback: Option<RepackFallback>,
    /// First differing `(expert, inner_offset, old_idx, new_idx)`, for diagnostics.
    pub first_mismatch: Option<(u64, u64, u32, u32)>,
    /// A small decoded window (experts × inner-offset), centred on the first
    /// mismatch, so the caller can show where old and new diverge.
    pub sample: Option<RepackSample>,
    /// Value diff of the sibling `codebook` tensor (the float centroids), when
    /// present — a codebook difference explains index differences.
    pub codebook: Option<RepackAux>,
    /// Value diff of the sibling `qscale` tensor (the per-group scales), when present.
    pub qscale: Option<RepackAux>,
    /// Bytes read from S3 for this tensor (both sides).
    pub bytes: u64,
    /// Set when the tensor couldn't be verified (bad shapes / read error).
    pub error: Option<String>,
}

/// The value diff of a sibling float tensor (codebook / scale) between the two
/// checkpoints — these have the same shape on both sides, so the structural diff
/// shows them "unchanged" even when their values differ.
#[derive(Debug, Clone)]
pub struct RepackAux {
    /// The tensor names actually looked up (old side, new side) — so the report can
    /// show exactly what was compared. Equal when there was no rename.
    pub old_name: String,
    pub new_name: String,
    /// Whether the sibling was found on each side (the derived name may be wrong).
    pub old_present: bool,
    pub new_present: bool,
    /// The new side's shape (empty when not found).
    pub shape: Vec<usize>,
    /// `Some((old_shape, new_shape))` when the shapes differ (not value-compared).
    pub shape_mismatch: Option<(Vec<usize>, Vec<usize>)>,
    pub elements: u64,
    pub differing: u64,
    pub max_abs: f64,
    pub mean_abs: f64,
}

impl RepackAux {
    /// Found on both sides.
    pub fn present(&self) -> bool {
        self.old_present && self.new_present
    }
}

/// A decoded 2-D window of indices for both sides — the top-left is
/// `(expert e0, inner-offset off0)`, `cols` columns wide.
#[derive(Debug, Clone, Default)]
pub struct RepackSample {
    pub e0: u64,
    pub off0: u64,
    pub old: Vec<Vec<u32>>,
    pub new: Vec<Vec<u32>>,
}

/// A plain stored-dtype *value* comparison used when the top-bits format check
/// fails, so a codebooked tensor that turns out **not** to be packed indices is
/// still meaningfully diffed (the auto `--values` fallback).
#[derive(Debug, Clone, Default)]
pub struct RepackFallback {
    /// The stored dtype the words were reinterpreted as (e.g. `F16`).
    pub dtype: String,
    pub elements: u64,
    pub differing: u64,
    pub max_abs: f64,
    pub mean_abs: f64,
}

impl RepackResult {
    /// Whether this tensor verified as equivalent: same indices and both format
    /// checks clean.
    pub fn equivalent(&self) -> bool {
        self.error.is_none() && self.differing == 0 && self.sparse_bad == 0 && self.dense_bad == 0
    }
}

/// Line prefix the remote script tags its JSON with, so we can pick it out of any
/// motd / cstorch chatter on the SSH stdout.
const SENTINEL: &str = "CKPT_EXPLORER_META:";

/// Line prefix for the dump script's `done/total` progress reports, streamed
/// ahead of the final metadata so the load bar fills for an `s3://` read too.
const PROGRESS_TAG: &str = "CKPT_EXPLORER_PROG:";

/// Line prefix for the value-diff script's live status events (load phase +
/// per-tensor start), so the two-line comparison view shows what's happening.
const STATUS_TAG: &str = "CKPT_EXPLORER_STAT:";

/// The outcome of one tensor's value comparison, for the live view's status mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareStatus {
    /// Compared and identical.
    Identical,
    /// Compared and the values (or distribution) differ.
    Changed,
    /// Couldn't be compared (shape mismatch / read error).
    Error,
}

/// A live event streamed from a remote value comparison (`--values` or
/// `--verify-repack`), driving the standard per-tensor progress bars: it reports
/// each side's S3 download size + byte progress so the bars fill for real, not just
/// spin, then a per-tensor outcome as each one lands.
#[derive(Debug)]
pub enum RepackEvent<'a> {
    /// A checkpoint is loading (`"old"` / `"new"`).
    Loading(&'a str),
    /// Tensor `done` of `total` has begun.
    Start {
        done: usize,
        total: usize,
        name: &'a str,
    },
    /// The download sizes for this tensor's two sides are known.
    Size {
        name: &'a str,
        old_bytes: u64,
        new_bytes: u64,
    },
    /// Cumulative bytes downloaded for this tensor's two sides.
    Bytes {
        name: &'a str,
        old_done: u64,
        new_done: u64,
    },
    /// Download done; now decoding + comparing the indices.
    Comparing(&'a str),
    /// The tensor finished, with its verification outcome.
    Done {
        name: &'a str,
        status: CompareStatus,
    },
}

/// Upper bound on SSH sessions used to read one safetensors dir's shards in
/// parallel (work-stealing) — roughly one per shard for a typical sharded model,
/// so no worker is more than ~1 shard deep. If opening this many trips sshd's
/// concurrent-connection limit (e.g. two dirs diffed at once), the refused opens
/// just mean fewer readers and the work-stealing counter still covers every shard.
const MAX_SHARD_SESSIONS: usize = 12;

/// Per-shard header parse output, tagged with the shard's index so results from
/// several parallel readers can be merged back into a deterministic order.
type ShardParse = (usize, Vec<TensorInfo>, Vec<MetadataInfo>);

/// Whether a tensor's `source_path` refers to a remote (`--ssh-read`) source — an
/// `s3://…` URI or an scp-style `[user@]host:path` — for which data views aren't
/// available locally. The scp test (a `:` before any `/`, with a non-empty host to
/// its left) matches how `scp` itself distinguishes a remote target from a local
/// path, so local absolute/relative paths are never misread as remote.
pub fn is_remote_source(source_path: &str) -> bool {
    if source_path.starts_with("s3://") {
        return true;
    }
    match source_path.find(':') {
        Some(colon) if colon > 0 => !source_path[..colon].contains('/'),
        _ => false,
    }
}

/// A remote host + cstorch venv to read checkpoint metadata through (`--ssh-read`
/// / `--ssh-venv`).
#[derive(Clone, Debug)]
pub struct RemoteRead {
    pub host: String,
    pub venv: String,
}

impl RemoteRead {
    pub fn new(host: String, venv: String) -> Self {
        RemoteRead { host, venv }
    }

    /// Read a remote checkpoint's structure over a fresh SSH session (one auth),
    /// with a progress spinner, and also fetch its `config.json` over the *same*
    /// session (no second auth prompt) so the `check`/health config-consistency
    /// check runs against a remote checkpoint too. `None` config for an `s3://`
    /// cstorch checkpoint (no HF `config.json`) or when the sidecar is
    /// absent/unreadable. For several reads sharing one session/prompt (e.g.
    /// `diff`), use [`Self::open_with`] + [`Self::read`] directly.
    pub fn fetch_with_config(&self, src: &str) -> Result<FetchedCheckpoint> {
        let mut password = None;
        let session = self.open_with(&mut password)?;
        eprintln!("checkpoint-explorer: reading tensor metadata over ssh …");
        let bars = crate::progress::Bars::start(vec![src.to_string()]);
        let progress = bars.progress(0);
        // Interactive browse doesn't use S3 object metadata (that's a `diff`-only
        // comparison), so skip the extra per-object HEADs here.
        let out = self.read(&session, src, &password, progress.as_deref(), false);
        bars.finish(0, out.is_ok());
        bars.join();
        let rc = out?;
        let config = self.read_config(&session, src);
        // The index/file health was computed by `read` from the same pass (no
        // second index read or header fetch).
        Ok((rc.tensors, rc.metadata, config, rc.disk, rc.health))
    }

    /// Read a remote checkpoint's structure into the central [`Checkpoint`] model —
    /// the same model [`crate::readers::read_local`] produces — so the web server
    /// and `--print-model` can serve a remote source too. Metadata-only: tensors,
    /// `__metadata__`, and `config.json` come over SSH; there are no local files,
    /// index, or byte data (so on-disk stats, the file browser, and data-value
    /// views are unavailable — the structure, tree, layout, and per-tensor info are).
    /// Tensors are grouped by their stamped shard path: one shard for an `s3://`
    /// cstorch checkpoint, one per file for a remote safetensors directory.
    pub fn read_checkpoint(&self, src: &str) -> Result<crate::model::Checkpoint> {
        let (tensors, metadata, config, _disk, _health) = self.fetch_with_config(src)?;
        Ok(assemble_remote_checkpoint(
            &self.host, src, tensors, metadata, config,
        ))
    }

    /// Fetch + parse the remote `config.json` for `src` over an already-open
    /// session. `None` for `s3://` (no HF config) or on any read/parse failure —
    /// the config check then reports `n/a` rather than erroring the whole load.
    pub fn read_config(
        &self,
        session: &RemoteSession,
        src: &str,
    ) -> Option<crate::config::ModelConfig> {
        let path = crate::config::remote_path(src)?;
        let bytes = session.read_file(&path).ok()?;
        let text = String::from_utf8(bytes).ok()?;
        crate::config::ModelConfig::parse(&text).filter(crate::config::ModelConfig::is_meaningful)
    }

    /// List the objects under an `s3://…` checkpoint's prefix over an already-open
    /// session — `(prefix-relative key, size)` per object — via a tiny boto3
    /// `list_objects_v2` (list only, **no** per-object HEAD, so it's one cheap
    /// paginated call). Read-only; the s3-native file browser turns these keys into
    /// a tree ([`crate::filetree::build_from_keys`]).
    pub fn list_s3(&self, session: &RemoteSession, uri: &str) -> Result<Vec<(String, u64)>> {
        let script = list_script(uri);
        let command = format!("source {}/bin/activate && python3 -", self.venv);
        let out = session.exec_capture(&command, &script, |_| {})?;
        let json = out
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix(SENTINEL))
            .ok_or_else(|| {
                anyhow!(
                    "no object listing returned from {} — remote output was:\n{}",
                    self.host,
                    out.trim()
                )
            })?;
        parse_list(json)
    }

    /// Open an authenticated SSH session to the host, reusing/recording a password
    /// so a subsequent session to the same host authenticates without prompting
    /// again — used to read two checkpoints in parallel, and a dir's shards across
    /// a pool, all behind one prompt.
    pub fn open_with(&self, password: &mut Option<String>) -> Result<RemoteSession> {
        RemoteSession::connect_with(&self.host, password)
    }

    /// Read one checkpoint over an already-open session: an `s3://…` cstorch
    /// checkpoint via the cstorch dump over an SSH exec channel, or a remote
    /// safetensors dir/file over SFTP. Tensor data is never read. `password` (the
    /// one already entered for `session`) lets a multi-shard dir open a few more
    /// sessions to read shards in parallel without prompting again.
    pub fn read(
        &self,
        session: &RemoteSession,
        src: &str,
        password: &Option<String>,
        progress: Option<&LoadProgress>,
        want_s3: bool,
    ) -> Result<RemoteCheckpoint> {
        if src.starts_with("s3://") {
            // An s3:// cstorch checkpoint isn't a local filesystem path, so there's
            // no block allocation to measure, and it has no HF index to check. The
            // S3 object metadata (an extra HEAD per object) is fetched only when the
            // caller wants it (`diff`) — a plain browse skips that cost.
            let (tensors, metadata, s3) = self.read_cstorch(session, src, progress, want_s3)?;
            Ok(RemoteCheckpoint {
                tensors,
                metadata,
                disk: None,
                health: Vec::new(),
                s3: want_s3.then_some(s3),
            })
        } else {
            self.read_dir(session, src, password, progress)
        }
    }

    /// A remote safetensors directory/file over SFTP. Its shards' headers are read
    /// **in parallel** across a pool of sessions — `session` plus up to
    /// [`MAX_SHARD_SESSIONS`]`- 1` more opened here (reusing `password`, so no extra
    /// prompt) — sharing one **work-stealing** shard counter, then merged in shard
    /// order deduped by name.
    ///
    /// Work-stealing (rather than a fixed split) means `session` starts reading
    /// immediately while the extra sessions are still completing their SSH
    /// handshakes — hiding that setup latency — and a session drawing a slow or
    /// large-headered shard doesn't hold up the others. A shard is claimed with one
    /// atomic increment; a failed extra-open just means one fewer reader, not a
    /// failed read.
    fn read_dir(
        &self,
        session: &RemoteSession,
        path: &str,
        password: &Option<String>,
        progress: Option<&LoadProgress>,
    ) -> Result<RemoteCheckpoint> {
        use std::sync::atomic::AtomicUsize;

        // One pass over the directory: the shard read order plus the index +
        // listing the health check needs (read once, shared below).
        let crate::sftp::ShardListing {
            files,
            index_path,
            weight_map,
            actual,
        } = session.list_shards(path)?;
        if files.is_empty() {
            bail!("no safetensors files found at {}", self.source_path(path));
        }
        // Now the shard count is known — the bar switches from spinner to filling.
        if let Some(p) = progress {
            p.set_total(files.len());
            p.set_unit(crate::progress::Unit::Shards);
        }
        // Stamp each tensor with *its own* shard's scp-style path (not the dir), so
        // the status line / `f` shows the exact file and it's usable with scp.
        let displays: Vec<String> = files.iter().map(|f| self.source_path(f)).collect();

        let workers = files.len().min(MAX_SHARD_SESSIONS);
        let next = AtomicUsize::new(0);
        let parts: Vec<Result<Vec<ShardParse>>> = std::thread::scope(|s| {
            let (files, displays, next) = (&files, &displays, &next);
            let mut handles = Vec::with_capacity(workers);
            // The already-open session reads straight away.
            handles.push(s.spawn(move || session.read_shards(files, displays, next, progress)));
            // Extra sessions connect in parallel, then join the same queue.
            for _ in 1..workers {
                handles.push(s.spawn(move || {
                    let mut pw = password.clone();
                    match self.open_with(&mut pw) {
                        Ok(extra) => extra.read_shards(files, displays, next, progress),
                        Err(_) => Ok(Vec::new()), // one fewer reader; others cover it
                    }
                }));
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(anyhow!("shard read thread panicked")))
                })
                .collect()
        });

        let mut all: Vec<ShardParse> = Vec::new();
        for part in parts {
            all.extend(part?);
        }
        let (tensors, metadata) = merge_shards(all);
        if tensors.is_empty() {
            bail!(
                "no tensors in the safetensors headers at {}",
                self.source_path(path)
            );
        }
        // Best-effort filesystem footprint of the shards (one read-only `stat`
        // over SSH). A failure here — no `stat`, non-GNU, restricted shell — just
        // drops the on-disk section from the stats popup; it never fails the load.
        let disk = session
            .allocated_sizes(&files)
            .ok()
            .map(|rows| {
                rows.into_iter()
                    .map(|(p, apparent, allocated)| ShardDisk {
                        name: crate::stats::shard_name(&p),
                        apparent,
                        allocated,
                    })
                    .collect::<Vec<_>>()
            })
            .and_then(DiskUsage::from_shards);

        // Index/file health from the same pass: the index we already read (its
        // `weight_map`) and the directory listing (`actual`), compared against the
        // tensor names the shard read already parsed — grouped by their shard's file
        // name. No second index read, no re-read of any header. A botched/stale
        // index (references shards that aren't there, or lists tensors a shard
        // doesn't hold) surfaces in the tree's health popup and `⚠ health` badge,
        // just as for a local checkpoint.
        let health = match index_path {
            Some(index_path) => {
                let mut present_by_file: HashMap<String, BTreeSet<String>> = HashMap::new();
                for t in &tensors {
                    if let Some(name) = std::path::Path::new(&t.source_path)
                        .file_name()
                        .and_then(|n| n.to_str())
                    {
                        present_by_file
                            .entry(name.to_string())
                            .or_default()
                            .insert(t.name.clone());
                    }
                }
                let report =
                    crate::health::reconcile(&index_path, &weight_map, &actual, &present_by_file);
                if report.has_issues() {
                    vec![report]
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        };

        Ok(RemoteCheckpoint {
            tensors,
            metadata,
            disk,
            health,
            s3: None, // a safetensors dir/file has no S3 object metadata
        })
    }

    /// The `source_path` stamped on each remote tensor: an `s3://…` URI as-is, or a
    /// remote path in **scp form** `[user@]host:path` — so the status line and the
    /// `f` (copy file path) command yield something you can hand straight to
    /// `scp`/`rsync`, and [`is_remote_source`] can still tell it's remote (data
    /// views need the bytes locally).
    fn source_path(&self, src: &str) -> String {
        if src.starts_with("s3://") {
            src.to_string()
        } else {
            format!("{}:{}", self.host, src)
        }
    }

    /// `s3://` cstorch checkpoint: run the (lazy) cstorch dump script in the venv
    /// over an SSH exec channel and parse the sentinel-tagged JSON it prints.
    fn read_cstorch(
        &self,
        session: &RemoteSession,
        src: &str,
        progress: Option<&LoadProgress>,
        want_s3: bool,
    ) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>, S3Meta)> {
        let script = dump_script(src, want_s3);
        let command = format!("source {}/bin/activate && python3 -", self.venv);
        // Feed the streamed `PROG:done/total` lines into the load bar as they land.
        let out = session.exec_capture(&command, &script, |line| {
            // `PROG:done/total[/unit]` — the optional unit switches the bar's label
            // for the second (S3-metadata) phase; absent ⇒ tensors (back-compat).
            if let Some(rest) = line.strip_prefix(PROGRESS_TAG) {
                let mut parts = rest.splitn(3, '/');
                if let (Some(d), Some(t)) = (parts.next(), parts.next())
                    && let (Ok(done), Ok(total)) = (d.trim().parse(), t.trim().parse())
                    && let Some(p) = progress
                {
                    let unit = match parts.next().map(str::trim) {
                        Some("s3") => crate::progress::Unit::S3Objects,
                        _ => crate::progress::Unit::Tensors,
                    };
                    p.set_total(total);
                    p.set_done(done);
                    p.set_unit(unit);
                }
            }
        })?;
        let json = out
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix(SENTINEL))
            .ok_or_else(|| {
                anyhow!(
                    "no metadata returned from {} — remote output was:\n{}",
                    self.host,
                    out.trim()
                )
            })?;
        parse_dump(json, &self.source_path(src))
    }

    /// Compare two `s3://` cstorch checkpoints' tensor **values** on the remote proxy
    /// (which has the S3 access) over an already-open session. A single torch script
    /// loads both checkpoints and, for each `(old_name, new_name)` pair, realises the
    /// two tensors and streams them in row-chunks to compute max/mean `|Δ|` and/or a
    /// shared-range histogram. Only the small per-tensor result JSON crosses the wire
    /// (never tensor data); the map returned is keyed by the *new* (post-rename)
    /// tensor name — the same key the diff report is built from.
    ///
    /// Each side's S3 object is streamed on the proxy in chunks (in parallel across
    /// `opts.jobs` tensors) purely so the byte progress can be reported — the values
    /// are still compared on the proxy; only the counts and the small result cross
    /// ssh. `on_event` receives live [`RepackEvent`]s (checkpoint loading, per-tensor
    /// download size + byte progress, then each tensor's outcome) so the caller can
    /// drive one standard progress bar per compared tensor.
    ///
    /// **Read-only:** loads (lazily) and reads tensor data to compare; it never
    /// writes, saves, or otherwise mutates either checkpoint.
    pub fn value_diff(
        &self,
        session: &RemoteSession,
        old_uri: &str,
        new_uri: &str,
        pairs: &[(String, String)],
        opts: &RemoteValueOpts,
        mut on_event: impl FnMut(RepackEvent),
    ) -> Result<(HashMap<String, RemoteTensorDiff>, Option<RemoteValueStats>)> {
        let script = value_diff_script(old_uri, new_uri, pairs, opts);
        let command = format!("source {}/bin/activate && python3 -", self.venv);
        let out = session.exec_capture(&command, &script, |line| {
            dispatch_stream_line(line, &compare_status, &mut on_event);
        })?;
        parse_value_diff(&out, &self.host)
    }

    /// Verify that shape-folded expert tensors in two `s3://` cstorch checkpoints
    /// encode the *same* 3-bit indices in different packings (old: one index per
    /// 16-bit word; new: `fold` indices per word, folded along dim 0). Runs on the
    /// proxy: per pair it streams both tensors' S3 objects (in parallel, with byte
    /// progress), reinterprets the raw 16-bit words, decodes the indices, checks
    /// equality, and validates the format (old words' top bits and new words' MSB
    /// must be zero). Emits [`RepackEvent`]s for the live per-tensor download bars.
    /// Returns per-tensor [`RepackResult`]s keyed by name. **Read-only.**
    #[allow(clippy::too_many_arguments)]
    pub fn verify_repack(
        &self,
        session: &RemoteSession,
        old_uri: &str,
        new_uri: &str,
        pairs: &[(String, String)],
        bits: usize,
        auto_sparse: bool,
        mut on_event: impl FnMut(RepackEvent),
    ) -> Result<(HashMap<String, RepackResult>, Option<RemoteValueStats>)> {
        let script = repack_verify_script(old_uri, new_uri, pairs, bits, auto_sparse);
        let command = format!("source {}/bin/activate && python3 -", self.venv);
        let out = session.exec_capture(&command, &script, |line| {
            dispatch_stream_line(line, &repack_status, &mut on_event);
        })?;
        parse_repack(&out, &self.host)
    }
}

/// Parse one streamed line into a [`RepackEvent`] — the `STAT`
/// `load`/`start`/`size`/`bytes`/`phase` events, or a sentinel per-tensor result
/// (classified by `status`). Shared by the value diff and the repack verify (both
/// stream the same way).
fn dispatch_stream_line(
    line: &str,
    status: &impl Fn(&serde_json::Value) -> CompareStatus,
    on_event: &mut impl FnMut(RepackEvent),
) {
    if let Some(rest) = line.strip_prefix(STATUS_TAG) {
        let mut f = rest.split('\t');
        match f.next() {
            Some("load") => {
                if let Some(w) = f.next() {
                    on_event(RepackEvent::Loading(w));
                }
            }
            Some("start") => {
                if let (Some(i), Some(t), Some(name)) = (f.next(), f.next(), f.next())
                    && let (Ok(done), Ok(total)) = (i.parse(), t.parse())
                {
                    on_event(RepackEvent::Start { done, total, name });
                }
            }
            Some("size") => {
                if let (Some(name), Some(o), Some(n)) = (f.next(), f.next(), f.next())
                    && let (Ok(old_bytes), Ok(new_bytes)) = (o.parse(), n.parse())
                {
                    on_event(RepackEvent::Size {
                        name,
                        old_bytes,
                        new_bytes,
                    });
                }
            }
            Some("bytes") => {
                if let (Some(name), Some(o), Some(n)) = (f.next(), f.next(), f.next())
                    && let (Ok(old_done), Ok(new_done)) = (o.parse(), n.parse())
                {
                    on_event(RepackEvent::Bytes {
                        name,
                        old_done,
                        new_done,
                    });
                }
            }
            Some("phase") => {
                if let Some(name) = f.next() {
                    on_event(RepackEvent::Comparing(name));
                }
            }
            // A sibling tensor's own bar finished downloading (codebook / qscale) —
            // it has no per-tensor verdict of its own, so mark the bar done (✓).
            Some("done") => {
                if let Some(name) = f.next() {
                    on_event(RepackEvent::Done {
                        name,
                        status: CompareStatus::Identical,
                    });
                }
            }
            _ => {}
        }
    } else if let Some(payload) = line.strip_prefix(SENTINEL)
        && let Ok(v) = serde_json::from_str::<serde_json::Value>(payload)
        && let Some(name) = v.get("name").and_then(|x| x.as_str())
    {
        on_event(RepackEvent::Done {
            name,
            status: status(&v),
        });
    }
}

/// Classify a per-tensor repack result for the live view's status mark: a format
/// violation or a read error is `Error` (✗), differing indices are `Changed` (≠),
/// equivalent is `Identical` (✓).
fn repack_status(v: &serde_json::Value) -> CompareStatus {
    let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
    if v.get("error").is_some() || u("sparse_bad") > 0 || u("dense_bad") > 0 {
        return CompareStatus::Error;
    }
    if u("differing") > 0 {
        CompareStatus::Changed
    } else {
        CompareStatus::Identical
    }
}

/// Classify a per-tensor value result line for the live view's status mark.
fn compare_status(v: &serde_json::Value) -> CompareStatus {
    if v.get("error").is_some() {
        return CompareStatus::Error;
    }
    let values_changed = v
        .get("values")
        .and_then(|x| x.get("differing"))
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|d| d > 0);
    let hist_changed = v
        .get("histogram")
        .and_then(|x| x.get("tvd"))
        .and_then(serde_json::Value::as_f64)
        .is_some_and(|t| t > 0.0);
    if values_changed || hist_changed {
        CompareStatus::Changed
    } else {
        CompareStatus::Identical
    }
}

/// The cstorch dump script for an `s3://…` checkpoint: `cstorch.load` (lazy — no
/// tensor data) and emit each tensor's name/dtype/shape/itemsize as a
/// sentinel-tagged JSON line. The URI is embedded as a JSON string literal (valid
/// Python), so nothing needs quoting at the shell. (Safetensors dirs/files don't
/// use this — they're read over SFTP; see [`crate::sftp`].)
///
/// **Read-only:** the script only *loads* (lazily) and writes its output to
/// stdout — it never opens a file for writing, calls `cstorch.save`/`torch.save`,
/// or otherwise mutates the checkpoint. The remote checkpoint is never modified.
fn dump_script(src: &str, want_s3: bool) -> String {
    let src_lit = serde_json::to_string(src).unwrap_or_else(|_| "\"\"".into());
    const TEMPLATE: &str = r#"
import sys, json
SRC = __URI__
WANT_S3 = __WANT_S3__
S = "__SENTINEL__"
P = "__PROGRESS__"
def emit(obj):
    sys.stdout.write(S + json.dumps(obj) + "\n")
    sys.stdout.flush()
def prog(done, total, unit=None):
    tail = ("/" + unit) if unit else ""
    sys.stdout.write("%s%d/%d%s\n" % (P, done, total, tail))
    sys.stdout.flush()
def probe_s3(src):
    # Best-effort, LIST-ONLY diagnosis of a load failure: enumerate the objects
    # under the prefix and report how many are empty (0 bytes) plus whether the
    # cstorch metadata object itself is empty, so the caller can explain *why*
    # cstorch.load failed (empty checkpoint / missing metadata / wrong prefix)
    # instead of surfacing only the raw dill traceback. Read-only; never mutates.
    if not src.startswith("s3://"):
        return {}
    try:
        import boto3
        from urllib.parse import urlparse
        u = urlparse(src)
        bucket, prefix = u.netloc, u.path.lstrip("/")
        cli = boto3.client("s3")
        total = empty = nbytes = 0
        meta_key = None; meta_empty = False; sample = []; tok = None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if tok: kw["ContinuationToken"] = tok
            resp = cli.list_objects_v2(**kw)
            for it in resp.get("Contents", []):
                k = it["Key"]; sz = int(it.get("Size", 0))
                rel = k[len(prefix):].lstrip("/") if k.startswith(prefix) else k
                if not rel:
                    continue
                total += 1; nbytes += sz
                if sz == 0: empty += 1
                if rel.rsplit("/", 1)[-1] == "__METADATA__":
                    meta_key = rel; meta_empty = (sz == 0)
                if len(sample) < 12: sample.append([rel, sz])
            if resp.get("IsTruncated"): tok = resp.get("NextContinuationToken")
            else: break
        return {"s3_probe": {"prefix": prefix, "total": total, "empty": empty,
                             "bytes": nbytes, "metadata_key": meta_key,
                             "metadata_empty": meta_empty, "sample": sample}}
    except Exception as e:
        return {"s3_probe_error": "%r" % (e,)}
try:
    import cerebras.pytorch as cstorch
except Exception as e:
    emit({"error": "import cerebras.pytorch failed: %r" % (e,)}); sys.exit(0)
try:
    sd = cstorch.load(SRC, map_location=None)   # lazy: metadata only, no data
except Exception as e:
    emit(dict({"error": "cstorch.load failed: %r" % (e,)}, **probe_s3(SRC))); sys.exit(0)
keys = list(sd.keys())
total = len(keys)
prog(0, total)                                  # total known → bar goes determinate
step = max(1, total // 100)                     # ~100 updates, not one per tensor
tensors = []
for i, name in enumerate(keys):
    try:
        t = sd[name]
        it = int(t.element_size()) if hasattr(t, "element_size") else 0
        tensors.append({"name": str(name), "dtype": str(getattr(t, "dtype", "")), "shape": [int(d) for d in t.shape], "itemsize": it})
    except Exception:
        pass
    if (i + 1) % step == 0 or i + 1 == total:
        prog(i + 1, total)
# S3 object metadata (best-effort, read-only): list the objects under the prefix
# and HEAD each for size/etag/checksum/last-modified/user-metadata; tags need a
# separate (often ungranted) permission, so they're tried per object and a single
# warning is emitted if denied. Any failure degrades to fewer objects + a warning.
s3_objects = []
s3_warnings = []
if WANT_S3:
    try:
        import boto3
        from urllib.parse import urlparse
        u = urlparse(SRC)
        bucket, prefix = u.netloc, u.path.lstrip("/")
        cli = boto3.client("s3")
        keys, tok = [], None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if tok: kw["ContinuationToken"] = tok
            resp = cli.list_objects_v2(**kw)
            keys.extend(it["Key"] for it in resp.get("Contents", []))
            if resp.get("IsTruncated"): tok = resp.get("NextContinuationToken")
            else: break
        # Second progress phase: HEADing each object is the slow part, so drive the
        # bar off it (relabelled "S3 objects") instead of sitting at 100% tensors.
        nkeys = len(keys)
        s3_step = max(1, nkeys // 100)
        prog(0, nkeys, "s3")
        tags_denied = False
        for i, k in enumerate(keys):
            if i % s3_step == 0:
                prog(i, nkeys, "s3")
            try:
                h = cli.head_object(Bucket=bucket, Key=k, ChecksumMode="ENABLED")
            except Exception as e:
                s3_warnings.append("head_object failed for %s: %r" % (k, e)); continue
            rel = k[len(prefix):].lstrip("/") if k.startswith(prefix) else k
            lm = h.get("LastModified")
            obj = {"key": rel, "size": int(h.get("ContentLength", 0)),
                   "etag": str(h.get("ETag", "")).strip('"'),
                   "last_modified": lm.isoformat() if lm else "",
                   "metadata": {mk: str(mv) for mk, mv in (h.get("Metadata") or {}).items()}}
            for algo in ("CRC32", "CRC32C", "SHA1", "SHA256"):
                cv = h.get("Checksum" + algo)
                if cv:
                    obj["checksum"] = [algo.lower(), str(cv)]; break
            try:
                tg = cli.get_object_tagging(Bucket=bucket, Key=k)
                obj["tags"] = {t["Key"]: t["Value"] for t in tg.get("TagSet", [])}
            except Exception as e:
                if not tags_denied:
                    s3_warnings.append("tags unavailable (needs s3:GetObjectTagging): %r" % (e,))
                    tags_denied = True
            s3_objects.append(obj)
        prog(nkeys, nkeys, "s3")
    except Exception as e:
        s3_warnings.append("s3 metadata unavailable: %r" % (e,))
emit({"tensors": tensors, "metadata": [], "s3_objects": s3_objects, "s3_warnings": s3_warnings})
"#;
    TEMPLATE
        .replace("__URI__", &src_lit)
        .replace("__WANT_S3__", if want_s3 { "True" } else { "False" })
        .replace("__SENTINEL__", SENTINEL)
        .replace("__PROGRESS__", PROGRESS_TAG)
}

/// The value-comparison script for two `s3://` cstorch checkpoints: load both,
/// then for each `(old_name, new_name)` pair realise the tensors and stream them in
/// row-chunks to compute (a) element `|Δ|` stats — matching [`crate::sample`]'s
/// `DiffAcc` semantics: bit-equal (or both-NaN) slots are unchanged, `max`/`mean`
/// `|Δ|` are over finite differing pairs, others count as non-finite mismatches —
/// and (b) a shared-range histogram (both sides binned over their combined finite
/// range into `n` equal-width bins, like the local `Range` layout). Streams status
/// events for the live view — `STAT:load\t<old|new>` while loading each checkpoint,
/// `STAT:start\t<i>\t<total>\t<name>` as each tensor begins — plus one sentinel
/// JSON result line per tensor (carrying its `bytes`) and a final `summary`. Both
/// URIs and the name pairs are embedded as JSON literals (valid Python), so nothing
/// needs shell quoting.
///
/// **Read-only:** it only `cstorch.load`s (lazily) and reads tensor data to
/// compare — it never opens anything for writing, never `save`s, and touches no S3
/// write API. Neither checkpoint is modified.
fn value_diff_script(
    old_uri: &str,
    new_uri: &str,
    pairs: &[(String, String)],
    opts: &RemoteValueOpts,
) -> String {
    let old_lit = serde_json::to_string(old_uri).unwrap_or_else(|_| "\"\"".into());
    let new_lit = serde_json::to_string(new_uri).unwrap_or_else(|_| "\"\"".into());
    let pairs_lit = serde_json::to_string(pairs).unwrap_or_else(|_| "[]".into());
    let bins_lit = opts
        .bins
        .map(|b| b.to_string())
        .unwrap_or_else(|| "None".into());
    let b = |x: bool| if x { "True" } else { "False" };
    let jobs_lit = opts.jobs.max(1).to_string();
    const TEMPLATE: &str = r#"
import sys, json, time, threading
from concurrent.futures import ThreadPoolExecutor
OLD = __OLD__
NEW = __NEW__
PAIRS = __PAIRS__
WANT_VALUES = __WANT_VALUES__
WANT_HIST = __WANT_HIST__
FULL_HIST = __FULL_HIST__
BINS = __BINS__
JOBS = __JOBS__
CHUNK = 4000000
DL = 16 << 20        # S3 download chunk
EMIT = 64 << 20      # byte-progress emit granularity
S = "__SENTINEL__"
ST = "__STATUS__"
_lock = threading.Lock()
def emit(o):
    with _lock:
        sys.stdout.write(S + json.dumps(o) + "\n"); sys.stdout.flush()
def stat(s):
    with _lock:
        sys.stdout.write(ST + s + "\n"); sys.stdout.flush()
try:
    import cerebras.pytorch as cstorch
except Exception as e:
    emit({"error": "import cerebras.pytorch failed: %r" % (e,)}); sys.exit(0)
try:
    import numpy as np
    import torch
    import zstandard
except Exception as e:
    emit({"error": "import numpy/torch/zstandard failed (needed to compare values): %r" % (e,)}); sys.exit(0)
try:
    stat("load\told")
    old_sd = cstorch.load(OLD, map_location=None)
    stat("load\tnew")
    new_sd = cstorch.load(NEW, map_location=None)
except Exception as e:
    emit({"error": "cstorch.load failed: %r" % (e,)}); sys.exit(0)
def to_f64(rt, i, j):
    sl = rt if i is None else rt[i:j]
    return sl.to(torch.float64).numpy().reshape(-1)
# Map a numpy dtype name (from an object's __NUMPY__ metadata) to a torch dtype, so
# we can stream the S3 object ourselves (chunked, with progress) rather than letting
# cstorch materialise it in one shot with no progress. Comparison stays on the proxy.
NP2T = {}
for _n in ("float16","float32","float64","bfloat16","uint8","int8","int16","uint16","int32","uint32","int64","uint64","bool"):
    _t = getattr(torch, _n, None)
    if _t is not None: NP2T[_n] = _t
def prep(dt):
    do = getattr(dt, "deferred", None)
    if do is None: return None
    r = do._reader
    key = "%s/%s" % (r.key, do._key)
    try:
        head = r.s3_client.head_object(Bucket=r.bucket, Key=key)
        md = head.get("Metadata") or {}
        meta = json.loads(md.get("metadata") or md.get("Metadata") or "{}")
    except Exception:
        return None
    nm = meta.get("__NUMPY__")
    td = NP2T.get(nm.get("dtype")) if nm else None
    if td is None: return None
    return (r.s3_client, r.bucket, key, int(head.get("ContentLength", 0)), bool(meta.get("compressed")), td, [int(x) for x in nm["shape"]])
def stream_raw(p, on):
    # Just the network read (with byte progress). Decompression + reshape is deferred
    # (see decode_tensor) so the bar reaches 100% before the slow decode, not during it.
    client, bucket, key, total, comp, td, shape = p
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    buf = bytearray()
    while True:
        c = body.read(DL)
        if not c: break
        buf += c; on(len(buf))
    return buf
def decode_tensor(buf, p):
    client, bucket, key, total, comp, td, shape = p
    raw = zstandard.decompress(bytes(buf)) if comp else buf
    return torch.frombuffer(bytearray(raw), dtype=td).reshape(shape)
total = len(PAIRS)
t0 = time.time()
read_bytes = 0
compared = 0
def work(idx):
    oname = PAIRS[idx][0]; nname = PAIRS[idx][1]
    stat("start\t%d\t%d\t%s" % (idx + 1, total, nname))
    res = {"name": nname}
    tb = 0
    try:
        a = old_sd[oname]; b = new_sd[nname]
        ashape = [int(d) for d in a.shape]; bshape = [int(d) for d in b.shape]
        if ashape != bshape:
            res["error"] = "shapes differ"; emit(res); return 0
        # Stream each side's S3 object on the proxy (chunked, with byte progress); the
        # per-chunk float64 conversions below work off the in-memory copy. Values are
        # compared here — only progress + the small result cross ssh. Fall back to
        # cstorch materialise (no byte progress) for a non-numpy-stored tensor.
        pa = prep(a); pb = prep(b)
        streamed = None   # (raw_a, raw_b) when read over the network
        if pa is not None and pb is not None:
            osz = pa[3]; nsz = pb[3]
            stat("size\t%s\t%d\t%d" % (nname, osz, nsz))
            od = [0]; nd = [0]; last = [0]
            def bump():
                s = od[0] + nd[0]
                if s - last[0] >= EMIT:
                    last[0] = s; stat("bytes\t%s\t%d\t%d" % (nname, od[0], nd[0]))
            rawa = stream_raw(pa, lambda n: (od.__setitem__(0, n), bump()))
            rawb = stream_raw(pb, lambda n: (nd.__setitem__(0, n), bump()))
            stat("bytes\t%s\t%d\t%d" % (nname, osz, nsz))
            streamed = (rawa, rawb)
            tb = osz + nsz
        else:
            ra = a.to("cpu"); rb = b.to("cpu")
            tb = ra.element_size() * ra.nelement() + rb.element_size() * rb.nelement()
            stat("size\t%s\t%d\t%d" % (nname, int(ra.element_size() * ra.nelement()), int(rb.element_size() * rb.nelement())))
        # Bytes are all read (bar at 100%); decompress + decode below is the slow part,
        # so flag "comparing" now — not after it.
        stat("phase\t%s\tcomparing" % (nname,))
        if streamed is not None:
            ra = decode_tensor(streamed[0], pa); rb = decode_tensor(streamed[1], pb)
        res["bytes"] = int(tb)
        scalar = (len(ashape) == 0)
        d0 = 0 if scalar else ashape[0]
        inner = 1
        for d in ashape[1:]: inner *= d
        rows = max(1, CHUNK // max(1, inner))
        spans = [(None, None)] if scalar else [(i, min(i + rows, d0)) for i in range(0, d0, rows)]
        if not spans: spans = [(None, None)]
        elements = 0; differing = 0; nfm = 0; max_abs = 0.0; sum_abs = 0.0
        omin = omax = nmin = nmax = None
        for (i, j) in spans:
            av = to_f64(ra, i, j); bv = to_f64(rb, i, j)
            if WANT_VALUES:
                both_nan = np.isnan(av) & np.isnan(bv)
                dmask = ~((av == bv) | both_nan)
                elements += int(av.size)
                differing += int(np.count_nonzero(dmask))
                bothfin = np.isfinite(av) & np.isfinite(bv)
                fdm = dmask & bothfin
                if bool(np.any(fdm)):
                    dd = np.abs(av[fdm] - bv[fdm])
                    m = float(dd.max())
                    if m > max_abs: max_abs = m
                    sum_abs += float(dd.sum())
                nfm += int(np.count_nonzero(dmask & ~bothfin))
            if WANT_HIST:
                af = av[np.isfinite(av)]; bf = bv[np.isfinite(bv)]
                if af.size:
                    lo = float(af.min()); hi = float(af.max())
                    omin = lo if omin is None else min(omin, lo)
                    omax = hi if omax is None else max(omax, hi)
                if bf.size:
                    lo = float(bf.min()); hi = float(bf.max())
                    nmin = lo if nmin is None else min(nmin, lo)
                    nmax = hi if nmax is None else max(nmax, hi)
        if WANT_VALUES:
            mean_abs = (sum_abs / elements) if elements > 0 else 0.0
            res["values"] = {"elements": elements, "differing": differing, "max_abs": max_abs, "mean_abs": mean_abs, "nonfinite_mismatch": nfm}
        if WANT_HIST:
            mins = [x for x in (omin, nmin) if x is not None]
            maxs = [x for x in (omax, nmax) if x is not None]
            n = BINS if BINS else 40
            lo_c = min(mins) if mins else 0.0
            hi_c = max(maxs) if maxs else 0.0
            if hi_c <= lo_c: n = 1
            oc = np.zeros(n, dtype=np.int64); nc = np.zeros(n, dtype=np.int64)
            onf = 0; nnf = 0
            for (i, j) in spans:
                av = to_f64(ra, i, j); bv = to_f64(rb, i, j)
                afin = np.isfinite(av); bfin = np.isfinite(bv)
                onf += int(av.size) - int(np.count_nonzero(afin))
                nnf += int(bv.size) - int(np.count_nonzero(bfin))
                af = av[afin]; bf = bv[bfin]
                if hi_c <= lo_c:
                    oc[0] += int(af.size); nc[0] += int(bf.size)
                else:
                    hc, _ = np.histogram(af, bins=n, range=(lo_c, hi_c)); oc += hc.astype(np.int64)
                    hc, _ = np.histogram(bf, bins=n, range=(lo_c, hi_c)); nc += hc.astype(np.int64)
            ot = int(oc.sum()); nt = int(nc.sum())
            if ot == 0 and nt == 0: tvd = 0.0
            elif ot == 0 or nt == 0: tvd = 1.0
            else: tvd = 0.5 * float(np.abs(oc.astype(np.float64) / ot - nc.astype(np.float64) / nt).sum())
            h = {"tvd": tvd, "n": int(n)}
            if FULL_HIST:
                h["lo"] = lo_c; h["hi"] = hi_c
                h["old"] = [int(x) for x in oc.tolist()]; h["new"] = [int(x) for x in nc.tolist()]
                h["old_total"] = ot; h["new_total"] = nt; h["old_nonfinite"] = onf; h["new_nonfinite"] = nnf
            res["histogram"] = h
    except Exception as e:
        res["error"] = "%r" % (e,)
    emit(res)
    return tb
# Reading each tensor's S3 object is latency-bound, so overlap JOBS of them; results
# are order-independent. JOBS<=1 stays sequential (the safe fallback). Byte + count
# tallies happen in this main thread as results land, so no locking is needed there.
if JOBS <= 1:
    tbs = (work(i) for i in range(total))
else:
    ex = ThreadPoolExecutor(max_workers=JOBS)
    tbs = ex.map(work, range(total))
for tb in tbs:
    if tb > 0:
        read_bytes += tb; compared += 1
emit({"summary": {"tensors": total, "compared": compared, "bytes": int(read_bytes), "elapsed_s": time.time() - t0}})
"#;
    TEMPLATE
        .replace("__OLD__", &old_lit)
        .replace("__NEW__", &new_lit)
        .replace("__PAIRS__", &pairs_lit)
        .replace("__WANT_VALUES__", b(opts.values))
        .replace("__WANT_HIST__", b(opts.histogram))
        .replace("__FULL_HIST__", b(opts.full_hist))
        .replace("__BINS__", &bins_lit)
        .replace("__JOBS__", &jobs_lit)
        .replace("__SENTINEL__", SENTINEL)
        .replace("__STATUS__", STATUS_TAG)
}

/// The repack-equivalence script for two `s3://` cstorch checkpoints: for each
/// `(old_name, new_name)` pair whose shapes fold along dim 0, reinterpret both
/// tensors' raw 16-bit words, decode the `bits`-wide indices (old: one per word;
/// new: `fold = ceil(E/W)` per word, expert `e` at word `e//fold` shift
/// `(e%fold)*bits`), and check they match — chunked over the inner dims so memory
/// stays bounded. Also validates the packing: old words must have zero bits above
/// `bits`, new words zero bits above `fold*bits`. Streams `STAT` events + one
/// sentinel result line per tensor.
///
/// **Read-only:** only `cstorch.load` (lazy) + per-tensor materialize; no writes.
fn repack_verify_script(
    old_uri: &str,
    new_uri: &str,
    pairs: &[(String, String)],
    bits: usize,
    auto_sparse: bool,
) -> String {
    let old_lit = serde_json::to_string(old_uri).unwrap_or_else(|_| "\"\"".into());
    let new_lit = serde_json::to_string(new_uri).unwrap_or_else(|_| "\"\"".into());
    let pairs_lit = serde_json::to_string(pairs).unwrap_or_else(|_| "[]".into());
    let bits_lit = bits.max(1).to_string();
    let jobs_lit = pairs.len().clamp(1, 4).to_string();
    let auto_lit = if auto_sparse { "True" } else { "False" };
    const TEMPLATE: &str = r#"
import sys, json, time, threading
from concurrent.futures import ThreadPoolExecutor
OLD = __OLD__
NEW = __NEW__
PAIRS = __PAIRS__
BITS = __BITS__
JOBS = __JOBS__
AUTO = __AUTO__      # sparse<->sparse auto (--values): same shape both sides, fold 1
DL = 16 << 20        # S3 download chunk
CMP = 4000000        # compare column block
EMIT = 64 << 20      # byte-progress emit granularity
S = "__SENTINEL__"
ST = "__STATUS__"
_lock = threading.Lock()
def emit(o):
    with _lock:
        sys.stdout.write(S + json.dumps(o) + "\n"); sys.stdout.flush()
def stat(s):
    with _lock:
        sys.stdout.write(ST + s + "\n"); sys.stdout.flush()
try:
    import cerebras.pytorch as cstorch
except Exception as e:
    emit({"error": "import cerebras.pytorch failed: %r" % (e,)}); sys.exit(0)
try:
    import numpy as np
    import torch
    import zstandard
except Exception as e:
    emit({"error": "import numpy/torch/zstandard failed: %r" % (e,)}); sys.exit(0)
try:
    stat("load\told")
    old_sd = cstorch.load(OLD, map_location=None)
    stat("load\tnew")
    new_sd = cstorch.load(NEW, map_location=None)
except Exception as e:
    emit({"error": "cstorch.load failed: %r" % (e,)}); sys.exit(0)
mask = np.uint16((1 << BITS) - 1)
# Resolve a lazy tensor to its backing S3 object (thread-safe client + key) so we
# can stream it ourselves with progress, rather than cstorch's one-shot read.
def resolve(dt):
    do = getattr(dt, "deferred", None)
    if do is None:
        return None
    r = do._reader
    return (r.s3_client, r.bucket, "%s/%s" % (r.key, do._key))
def head(loc):
    client, bucket, key = loc
    h = client.head_object(Bucket=bucket, Key=key)
    md = h.get("Metadata") or {}
    try:
        meta = json.loads(md.get("metadata") or md.get("Metadata") or "{}")
    except Exception:
        meta = {}
    return int(h.get("ContentLength", 0)), bool(meta.get("compressed")), ("__NUMPY__" in meta)
def download_raw(loc, on):
    # Just the network read (with byte progress). Decompression is deferred so the
    # bar reaches 100% before the slow decode, not during it.
    client, bucket, key = loc
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    buf = bytearray()
    while True:
        c = body.read(DL)
        if not c:
            break
        buf += c
        on(len(buf))
    return buf
def decode_u16(buf, compressed):
    if compressed:
        return np.frombuffer(zstandard.decompress(bytes(buf)), dtype=np.uint16)
    return np.frombuffer(buf, dtype=np.uint16)     # raw 16-bit words, zero-copy
def to_u16(dt):   # fallback via cstorch (no byte progress)
    return dt.to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)
NPF = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
def aux_meta(loc):
    # (compressed, numpy_dtype, shape, size) for a numpy-float-stored sibling, else None.
    client, bucket, key = loc
    h = client.head_object(Bucket=bucket, Key=key)
    md = h.get("Metadata") or {}
    try:
        meta = json.loads(md.get("metadata") or md.get("Metadata") or "{}")
    except Exception:
        return None
    nm = meta.get("__NUMPY__")
    if not nm or nm.get("dtype") not in NPF:
        return None
    return (bool(meta.get("compressed")), NPF[nm["dtype"]], [int(x) for x in nm["shape"]], int(h.get("ContentLength", 0)))
def stream_aux(sd, name, on_side):
    # Read a sibling float tensor (codebook / scale) over S3 with byte progress → f64.
    # Falls back to cstorch materialise (no progress) if it isn't numpy-float-stored.
    dt = sd[name]
    loc = resolve(dt)
    am = aux_meta(loc) if loc is not None else None
    if loc is None or am is None:
        on_side(0)
        return dt.to("cpu").to(torch.float64).numpy(), 0
    comp, npdt, shape, size = am
    buf = download_raw(loc, on_side)
    raw = zstandard.decompress(bytes(buf)) if comp else buf
    return np.frombuffer(raw, dtype=npdt).reshape(shape).astype(np.float64), size
def cmp_aux(oname_a, nname_a):
    # Streams both sides over S3 (its own progress bar, keyed by the new name) and
    # records the NAMES tried, so the caller shows exactly what was compared.
    out = {"old_name": oname_a, "new_name": nname_a,
           "old_present": oname_a in old_sd, "new_present": nname_a in new_sd}
    if not out["old_present"] or not out["new_present"]:
        return out
    osz = 0; nsz = 0
    olo = resolve(old_sd[oname_a]); nlo = resolve(new_sd[nname_a])
    oam = aux_meta(olo) if olo is not None else None
    nam = aux_meta(nlo) if nlo is not None else None
    if oam: osz = oam[3]
    if nam: nsz = nam[3]
    stat("size\t%s\t%d\t%d" % (nname_a, osz, nsz))
    od = [0]; nd = [0]; last = [0]
    def bump():
        s = od[0] + nd[0]
        if s - last[0] >= EMIT:
            last[0] = s; stat("bytes\t%s\t%d\t%d" % (nname_a, od[0], nd[0]))
    a, _ = stream_aux(old_sd, oname_a, lambda x: (od.__setitem__(0, x), bump()))
    b, _ = stream_aux(new_sd, nname_a, lambda x: (nd.__setitem__(0, x), bump()))
    stat("bytes\t%s\t%d\t%d" % (nname_a, osz, nsz))
    stat("done\t%s" % (nname_a,))
    with agg_lock:
        read_bytes[0] += osz + nsz
    out["shape_old"] = [int(x) for x in a.shape]; out["shape_new"] = [int(x) for x in b.shape]
    if list(a.shape) != list(b.shape):
        return out
    d = np.abs(a - b)
    out.update({"elements": int(a.size), "differing": int(np.count_nonzero(a != b)),
                "max_abs": float(d.max()) if d.size else 0.0, "mean_abs": float(d.mean()) if d.size else 0.0})
    return out
total = len(PAIRS)
t0 = time.time()
agg_lock = threading.Lock()
read_bytes = [0]
compared = [0]
def work(idx):
    oname = PAIRS[idx][0]; nname = PAIRS[idx][1]
    stat("start\t%d\t%d\t%s" % (idx + 1, total, nname))
    res = {"name": nname}
    try:
        da = old_sd[oname]; db = new_sd[nname]
        ashape = [int(x) for x in da.shape]; bshape = [int(x) for x in db.shape]
        E = ashape[0] if ashape else 0
        W = bshape[0] if bshape else 0
        if AUTO:
            # Sparse<->sparse: same shape both sides, one index per word (fold 1).
            if not ashape or ashape != bshape:
                res["error"] = "sparse compare needs equal shapes (%r vs %r)" % (ashape, bshape); emit(res); return
            fold = 1
        else:
            if not ashape or not bshape or ashape[1:] != bshape[1:] or W <= 0 or E <= W:
                res["error"] = "not a fold pair (shapes %r vs %r)" % (ashape, bshape); emit(res); return
            fold = (E + W - 1) // W
            if W != (E + fold - 1) // fold:
                res["error"] = "fold mismatch (E=%d W=%d -> fold=%d)" % (E, W, fold); emit(res); return
        if fold * BITS > 16:
            res["error"] = "fold*bits=%d exceeds the 16-bit word" % (fold * BITS); emit(res); return
        lo = resolve(da); ln = resolve(db)
        odone = [0]; ndone = [0]; last = [0]
        def bump():
            s = odone[0] + ndone[0]
            if s - last[0] >= EMIT:
                last[0] = s
                stat("bytes\t%s\t%d\t%d" % (nname, odone[0], ndone[0]))
        streamed = None   # (raw_a, comp_a, raw_b, comp_b) when read over the network
        if lo is not None and ln is not None:
            osz, ocomp, onp = head(lo); nsz, ncomp, nnp = head(ln)
            stat("size\t%s\t%d\t%d" % (nname, osz, nsz))
            if onp and nnp:
                def oon(n): odone[0] = n; bump()
                def non(n): ndone[0] = n; bump()
                ra = download_raw(lo, oon)
                rb = download_raw(ln, non)
                stat("bytes\t%s\t%d\t%d" % (nname, osz, nsz))
                streamed = (ra, ocomp, rb, ncomp)
                tb = osz + nsz
            else:
                ao = to_u16(da); bo = to_u16(db)   # not numpy-stored: no byte progress
                tb = int(ao.nbytes) + int(bo.nbytes)
        else:
            ao = to_u16(da); bo = to_u16(db)
            tb = int(ao.nbytes) + int(bo.nbytes)
            stat("size\t%s\t%d\t%d" % (nname, int(ao.nbytes), int(bo.nbytes)))
        # Bytes are all read (bar at 100%); the decompress + decode below is the slow
        # part, so flag "comparing" now — not after it.
        stat("phase\t%s\tcomparing" % (nname,))
        if streamed is not None:
            ao = decode_u16(streamed[0], streamed[1]); bo = decode_u16(streamed[2], streamed[3])
        ao = ao.reshape(E, -1); bo = bo.reshape(W, -1)
        N = ao.shape[1]
        we = (np.arange(E) // fold)
        se = ((np.arange(E) % fold) * BITS).astype(np.uint16)
        # Format checks: old words' bits above BITS, new words' bits above fold*BITS,
        # must all be zero (else the packing assumption is wrong). When fold*BITS==16
        # (e.g. 4-bit ×4) there are no unused high bits — and shifting a uint16 by 16
        # is undefined — so skip the dense check.
        sparse_bad = int(np.count_nonzero(ao >> np.uint16(BITS)))
        dshift = fold * BITS
        dense_bad = 0 if dshift >= 16 else int(np.count_nonzero(bo >> np.uint16(dshift)))
        differing = 0; first = None; maxdelta = 0; big = 0
        sum_abs = 0; sum_old = 0; sum_new = 0; zeros = 0
        blk = max(1, CMP // max(1, E))
        for n0 in range(0, N, blk):
            n1 = min(n0 + blk, N)
            o = ao[:, n0:n1] & mask
            nd = (bo[we, n0:n1] >> se[:, None]) & mask
            # Aggregate |Δ| and per-side sums (for mean |Δ|/parameter + mean index).
            dd = o.astype(np.int64) - nd.astype(np.int64)
            ad = np.abs(dd)
            sum_abs += int(ad.sum())
            sum_old += int(o.sum(dtype=np.uint64))
            sum_new += int(nd.sum(dtype=np.uint64))
            zeros += int(np.count_nonzero(o == 0)) + int(np.count_nonzero(nd == 0))  # both sides
            ne = o != nd
            cnt = int(np.count_nonzero(ne))
            differing += cnt
            if cnt:
                m = int(ad.max())
                if m > maxdelta: maxdelta = m
                big += int(np.count_nonzero(ad > 1))   # differ by more than ±1
            if first is None and cnt:
                p = np.argwhere(ne)[0]; e = int(p[0]); col = n0 + int(p[1])
                first = [e, col, int(o[p[0], p[1]]), int(nd[p[0], p[1]])]
        elems = E * N
        res.update({"elements": elems, "differing": differing, "sparse_bad": sparse_bad, "dense_bad": dense_bad, "fold": fold, "bits": BITS, "bytes": tb, "maxdelta": maxdelta, "big": big,
                    "sum_abs": int(sum_abs),
                    "mean_abs": (sum_abs / elems) if elems else 0.0,
                    "mean_old": (sum_old / elems) if elems else 0.0,
                    "mean_new": (sum_new / elems) if elems else 0.0,
                    "zero_frac": (zeros / (2.0 * elems)) if elems else 0.0})
        # Auto (--values) fallback: the top-bits check failed, so these aren't packed
        # indices — compare the raw words as stored floats instead (same bytes, viewed
        # as float16), so a mis-detected tensor is still meaningfully diffed.
        if AUTO and (sparse_bad > 0 or dense_bad > 0):
            dts = str(db.dtype).replace("torch.", "")
            # Compare in the real stored dtype: F16 words as floats, otherwise the
            # raw 16-bit integers (dense-packed U16, etc.).
            if dts == "float16":
                fa = ao.view(np.float16).astype(np.float64); fb = bo.view(np.float16).astype(np.float64)
            else:
                fa = ao.astype(np.float64); fb = bo.astype(np.float64)
            fd = np.abs(fa - fb)
            fin = np.isfinite(fd)   # F16 bits can be NaN/Inf → JSON-unsafe; mask them
            label = {"float16": "F16", "bfloat16": "BF16", "uint16": "U16", "int16": "I16"}.get(dts, dts.upper())
            res["fallback"] = {"dtype": label,
                               "elements": int(ao.size),
                               # Exact bit inequality (NaN-safe; identical bits ⇒ 0).
                               "differing": int(np.count_nonzero(ao != bo)),
                               "max_abs": float(fd[fin].max()) if bool(fin.any()) else 0.0,
                               "mean_abs": float(fd[fin].mean()) if bool(fin.any()) else 0.0}
        if first is not None:
            res["first"] = first
        # A decoded window (experts × inner-offset), centred on the first mismatch
        # (or the top-left corner), so the caller can SHOW old vs new and see
        # where/how they diverge.
        se0 = max(0, first[0] - 6) if first else 0
        so0 = max(0, first[1] - 16) if first else 0
        se1 = min(E, se0 + 16); so1 = min(N, so0 + 48)
        ew = np.arange(se0, se1)
        sold = (ao[se0:se1, so0:so1] & mask).astype(np.uint16).tolist()
        snew = ((bo[ew // fold][:, so0:so1] >> ((ew % fold) * BITS).astype(np.uint16)[:, None]) & mask).astype(np.uint16).tolist()
        res["sample"] = {"e0": int(se0), "off0": int(so0), "cols": int(so1 - so0), "old": sold, "new": snew}
        # Also diff the sibling codebook + scale tensors (float centroids/scales),
        # since these have the same shape/dtype on both sides and so don't show up in
        # the structural diff — but a codebook difference explains index differences
        # (the same weights quantised against a different codebook).
        if oname.endswith(".weight") and nname.endswith(".weight"):
            op = oname[:-7]; npx = nname[:-7]
            res["codebook"] = cmp_aux(op + ".codebook", npx + ".codebook")
            res["qscale"] = cmp_aux(op + ".qscale", npx + ".qscale")
        with agg_lock:
            read_bytes[0] += tb; compared[0] += 1
    except Exception as e:
        res["error"] = "%r" % (e,)
    emit(res)
if JOBS <= 1:
    for i in range(total):
        work(i)
else:
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        list(ex.map(work, range(total)))
emit({"summary": {"tensors": total, "compared": compared[0], "bytes": int(read_bytes[0]), "elapsed_s": time.time() - t0}})
"#;
    TEMPLATE
        .replace("__OLD__", &old_lit)
        .replace("__NEW__", &new_lit)
        .replace("__PAIRS__", &pairs_lit)
        .replace("__BITS__", &bits_lit)
        .replace("__JOBS__", &jobs_lit)
        .replace("__AUTO__", auto_lit)
        .replace("__SENTINEL__", SENTINEL)
        .replace("__STATUS__", STATUS_TAG)
}

/// The boto3 object-listing script for an `s3://…` checkpoint: a single paginated
/// `list_objects_v2` emitting one sentinel-tagged JSON line
/// `{objects:[[rel_key,size],…]}` (or `{error:…}`). Keys are made
/// prefix-relative so the browser shows them s3-natively. Distinct from
/// [`dump_script`]'s S3 phase, which additionally HEADs each object for the diff
/// metadata compare — this is **list only**, no per-object request.
///
/// **Read-only:** `list_objects_v2` is a read; the script never writes.
fn list_script(uri: &str) -> String {
    let src_lit = serde_json::to_string(uri).unwrap_or_else(|_| "\"\"".into());
    const TEMPLATE: &str = r#"
import sys, json
SRC = __URI__
S = "__SENTINEL__"
def emit(obj):
    sys.stdout.write(S + json.dumps(obj) + "\n"); sys.stdout.flush()
try:
    import boto3
    from urllib.parse import urlparse
except Exception as e:
    emit({"error": "import boto3 failed: %r" % (e,)}); sys.exit(0)
try:
    u = urlparse(SRC)
    bucket, prefix = u.netloc, u.path.lstrip("/")
    cli = boto3.client("s3")
    objects, tok = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if tok: kw["ContinuationToken"] = tok
        resp = cli.list_objects_v2(**kw)
        for it in resp.get("Contents", []):
            k = it["Key"]
            rel = k[len(prefix):].lstrip("/") if k.startswith(prefix) else k
            if rel:
                objects.append([rel, int(it.get("Size", 0))])
        if resp.get("IsTruncated"): tok = resp.get("NextContinuationToken")
        else: break
except Exception as e:
    emit({"error": "s3 list failed: %r" % (e,)}); sys.exit(0)
emit({"objects": objects})
"#;
    TEMPLATE
        .replace("__URI__", &src_lit)
        .replace("__SENTINEL__", SENTINEL)
}

/// Parse the object-listing JSON (`{objects:[[key,size],…]}` or `{error:…}`) into
/// `(prefix-relative key, size)` pairs. Malformed entries are skipped.
fn parse_list(json: &str) -> Result<Vec<(String, u64)>> {
    let v: serde_json::Value =
        serde_json::from_str(json).with_context(|| format!("parsing object listing: {json}"))?;
    if let Some(e) = v.get("error").and_then(serde_json::Value::as_str) {
        bail!("{e}");
    }
    let arr = v
        .get("objects")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("object listing had no `objects` array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for it in arr {
        if let Some(pair) = it.as_array()
            && pair.len() == 2
            && let (Some(k), Some(sz)) = (pair[0].as_str(), pair[1].as_u64())
        {
            out.push((k.to_string(), sz));
        }
    }
    Ok(out)
}

/// Parse the remote JSON (`{tensors:[…], metadata:[…]}` or `{error:…}`) into
/// [`TensorInfo`]s + [`MetadataInfo`], stamping each tensor with `source_path`
/// (already remote-marked; see [`RemoteRead::source_path`]) so display, the `y`
/// command, and the data-view "local-only" guard all behave.
fn parse_dump(
    json: &str,
    source_path: &str,
) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>, S3Meta)> {
    let v: serde_json::Value =
        serde_json::from_str(json).context("parsing the remote metadata JSON")?;
    if v.get("error").is_some() {
        bail!("remote: {}", diagnose_remote_error(&v));
    }
    let mut tensors = Vec::new();
    if let Some(arr) = v.get("tensors").and_then(|t| t.as_array()) {
        for item in arr {
            let name = item
                .get("name")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string();
            let dtype = map_dtype(
                item.get("dtype")
                    .and_then(|x| x.as_str())
                    .unwrap_or_default(),
            );
            let shape: Vec<usize> = item
                .get("shape")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|d| d.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();
            let itemsize = item.get("itemsize").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let num_elements: usize = shape.iter().product();
            tensors.push(TensorInfo {
                name,
                dtype,
                shape,
                size_bytes: num_elements * itemsize,
                num_elements,
                storage: Storage::Unknown,
                source_path: source_path.to_string(),
                layout: Layout::None,
            });
        }
    }
    if tensors.is_empty() {
        bail!("the remote returned no tensors for {source_path}");
    }
    // safetensors `__metadata__` entries (name/value/value_type), when present.
    let metadata = v
        .get("metadata")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let name = e.get("name").and_then(|x| x.as_str())?.to_string();
                    let value = e.get("value").and_then(|x| x.as_str())?.to_string();
                    let value_type = e
                        .get("value_type")
                        .and_then(|x| x.as_str())
                        .unwrap_or("string")
                        .to_string();
                    Some(MetadataInfo {
                        name,
                        value,
                        value_type,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let s3 = parse_s3_meta(&v);
    Ok((tensors, metadata, s3))
}

/// Parse [`RemoteRead::value_diff`]'s streamed output: one sentinel-tagged JSON line
/// per tensor (`{name, values?, histogram?, error?}`), collected into a map keyed by
/// the tensor's (new/post-rename) name. A sentinel line that carries an `error` but
/// no `name` is a fatal, whole-run failure (import / load) → an `Err`. Non-sentinel
/// lines (motd, cstorch chatter) are ignored.
fn parse_value_diff(
    out: &str,
    host: &str,
) -> Result<(HashMap<String, RemoteTensorDiff>, Option<RemoteValueStats>)> {
    let mut map = HashMap::new();
    let mut stats = None;
    let mut saw = false;
    for line in out.lines() {
        let Some(payload) = line.strip_prefix(SENTINEL) else {
            continue;
        };
        saw = true;
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        match v.get("name").and_then(|x| x.as_str()) {
            Some(name) => {
                map.insert(name.to_string(), parse_tensor_diff(&v));
            }
            // A `name`-less line with an `error` is a fatal failure for the whole run.
            None if v.get("error").is_some() => bail!("remote: {}", diagnose_remote_error(&v)),
            None => {
                if let Some(s) = v.get("summary") {
                    stats = parse_value_stats(s);
                }
            }
        }
    }
    if !saw {
        bail!(
            "no value-comparison output returned from {host} — remote output was:\n{}",
            out.trim()
        );
    }
    Ok((map, stats))
}

/// The final `{"summary": {...}}` line's I/O + timing metrics.
fn parse_value_stats(s: &serde_json::Value) -> Option<RemoteValueStats> {
    let u = |k: &str| s.get(k).and_then(serde_json::Value::as_u64);
    Some(RemoteValueStats {
        tensors: u("tensors")? as usize,
        compared: u("compared").unwrap_or(0) as usize,
        bytes: u("bytes").unwrap_or(0),
        elapsed_s: s
            .get("elapsed_s")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
    })
}

/// Parse [`RemoteRead::verify_repack`]'s streamed output into per-tensor
/// [`RepackResult`]s keyed by name, plus the final I/O summary. Mirrors
/// [`parse_value_diff`]: a `name`-less `error` line is fatal.
fn parse_repack(
    out: &str,
    host: &str,
) -> Result<(HashMap<String, RepackResult>, Option<RemoteValueStats>)> {
    let mut map = HashMap::new();
    let mut stats = None;
    let mut saw = false;
    for line in out.lines() {
        let Some(payload) = line.strip_prefix(SENTINEL) else {
            continue;
        };
        saw = true;
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        match v.get("name").and_then(|x| x.as_str()) {
            Some(name) => {
                map.insert(name.to_string(), parse_repack_result(&v));
            }
            None if v.get("error").is_some() => bail!("remote: {}", diagnose_remote_error(&v)),
            None => {
                if let Some(s) = v.get("summary") {
                    stats = parse_value_stats(s);
                }
            }
        }
    }
    if !saw {
        bail!(
            "no repack-verification output returned from {host} — remote output was:\n{}",
            out.trim()
        );
    }
    Ok((map, stats))
}

/// Build one tensor's [`RepackResult`] from its result JSON.
fn parse_repack_result(v: &serde_json::Value) -> RepackResult {
    let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
    let first = v.get("first").and_then(|f| f.as_array()).and_then(|a| {
        Some((
            a.first()?.as_u64()?,
            a.get(1)?.as_u64()?,
            a.get(2)?.as_u64()? as u32,
            a.get(3)?.as_u64()? as u32,
        ))
    });
    let grid = |val: Option<&serde_json::Value>| -> Vec<Vec<u32>> {
        val.and_then(|g| g.as_array())
            .map(|rows| {
                rows.iter()
                    .map(|r| {
                        r.as_array()
                            .map(|cs| cs.iter().map(|c| c.as_u64().unwrap_or(0) as u32).collect())
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let sample = v.get("sample").map(|s| RepackSample {
        e0: s.get("e0").and_then(serde_json::Value::as_u64).unwrap_or(0),
        off0: s
            .get("off0")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        old: grid(s.get("old")),
        new: grid(s.get("new")),
    });
    let f = |k: &str| v.get(k).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
    let aux = |val: Option<&serde_json::Value>| -> Option<RepackAux> {
        let a = val?;
        let dims = |k: &str| -> Vec<usize> {
            a.get(k)
                .and_then(|x| x.as_array())
                .map(|d| {
                    d.iter()
                        .filter_map(|n| n.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default()
        };
        let au = |k: &str| a.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
        let af = |k: &str| a.get(k).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
        let astr = |k: &str| {
            a.get(k)
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string()
        };
        let ab = |k: &str| {
            a.get(k)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        };
        let (so, sn) = (dims("shape_old"), dims("shape_new"));
        let shape_mismatch = (!so.is_empty() && so != sn).then(|| (so, sn.clone()));
        Some(RepackAux {
            old_name: astr("old_name"),
            new_name: astr("new_name"),
            old_present: ab("old_present"),
            new_present: ab("new_present"),
            shape: sn,
            shape_mismatch,
            elements: au("elements"),
            differing: au("differing"),
            max_abs: af("max_abs"),
            mean_abs: af("mean_abs"),
        })
    };
    RepackResult {
        elements: u("elements"),
        differing: u("differing"),
        max_delta: u("maxdelta") as u32,
        differing_gt1: u("big"),
        sum_abs: u("sum_abs"),
        mean_abs: f("mean_abs"),
        mean_old: f("mean_old"),
        mean_new: f("mean_new"),
        sparse_bad: u("sparse_bad"),
        dense_bad: u("dense_bad"),
        fold: u("fold") as usize,
        bits: u("bits") as usize,
        zero_frac: f("zero_frac"),
        fallback: v.get("fallback").filter(|x| !x.is_null()).map(|fb| {
            let fu = |k: &str| fb.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
            let ff = |k: &str| fb.get(k).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            RepackFallback {
                dtype: fb
                    .get("dtype")
                    .and_then(|x| x.as_str())
                    .unwrap_or_default()
                    .to_string(),
                elements: fu("elements"),
                differing: fu("differing"),
                max_abs: ff("max_abs"),
                mean_abs: ff("mean_abs"),
            }
        }),
        first_mismatch: first,
        sample,
        codebook: aux(v.get("codebook")),
        qscale: aux(v.get("qscale")),
        bytes: u("bytes"),
        error: v.get("error").and_then(|e| e.as_str()).map(str::to_string),
    }
}

/// Build one tensor's [`RemoteTensorDiff`] from its result JSON.
fn parse_tensor_diff(v: &serde_json::Value) -> RemoteTensorDiff {
    let error = v.get("error").and_then(|e| e.as_str()).map(str::to_string);
    let values = v.get("values").and_then(parse_value_fields);
    let (hist_shift, hist_full) = match v.get("histogram") {
        Some(h) => (parse_hist_shift(h), parse_hist_full(h)),
        None => (None, None),
    };
    RemoteTensorDiff {
        values,
        hist_shift,
        hist_full,
        error,
    }
}

/// A `{elements, differing, max_abs, mean_abs, nonfinite_mismatch}` object into a
/// [`ValueDiff`](crate::sample::ValueDiff). Requires the two count fields; the rest
/// default to 0.
fn parse_value_fields(v: &serde_json::Value) -> Option<crate::sample::ValueDiff> {
    let u = |k: &str| v.get(k).and_then(serde_json::Value::as_u64);
    let f = |k: &str| v.get(k).and_then(serde_json::Value::as_f64);
    Some(crate::sample::ValueDiff {
        elements: u("elements")?,
        differing: u("differing")?,
        max_abs: f("max_abs").unwrap_or(0.0),
        mean_abs: f("mean_abs").unwrap_or(0.0),
        nonfinite_mismatch: u("nonfinite_mismatch").unwrap_or(0),
    })
}

/// The `(tvd, bins)` summary always present on a histogram result (bulk path).
fn parse_hist_shift(h: &serde_json::Value) -> Option<(f64, usize)> {
    let tvd = h.get("tvd").and_then(serde_json::Value::as_f64)?;
    let n = h.get("n").and_then(serde_json::Value::as_u64)? as usize;
    Some((tvd, n))
}

/// The full per-bin histogram, present only when `full_hist` was requested (the
/// single-tensor `--tensor` table). `None` when the bin arrays are absent.
fn parse_hist_full(h: &serde_json::Value) -> Option<crate::sample::HistogramDiff> {
    let arr_u64 = |val: Option<&serde_json::Value>| -> Option<Vec<u64>> {
        Some(
            val?.as_array()?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0))
                .collect(),
        )
    };
    let old = arr_u64(h.get("old"))?;
    let new = arr_u64(h.get("new"))?;
    let n = h.get("n").and_then(serde_json::Value::as_u64)? as usize;
    let lo = h.get("lo").and_then(serde_json::Value::as_f64)?;
    let hi = h.get("hi").and_then(serde_json::Value::as_f64)?;
    let u = |k: &str| h.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
    Some(crate::sample::HistogramDiff {
        bins: crate::sample::HistBins::Range { lo, hi },
        n,
        old,
        new,
        old_total: u("old_total"),
        new_total: u("new_total"),
        old_nonfinite: u("old_nonfinite"),
        new_nonfinite: u("new_nonfinite"),
    })
}

/// Turn the remote script's `error` — optionally accompanied by a best-effort,
/// list-only `s3_probe` of the objects under the prefix — into a message that
/// explains *why* a `cstorch.load` failed (empty checkpoint, missing/empty
/// metadata object, wrong prefix) rather than surfacing only the raw
/// cstorch/dill traceback tail (`EOFError('Ran out of input')`), which on its own
/// tells a user nothing actionable.
fn diagnose_remote_error(v: &serde_json::Value) -> String {
    let err = v
        .get("error")
        .and_then(|e| e.as_str())
        .unwrap_or("remote read failed");
    if let Some(probe) = v.get("s3_probe") {
        return format!("{}\n\ncstorch: {err}", diagnose_s3_probe(probe));
    }
    if let Some(pe) = v.get("s3_probe_error").and_then(|e| e.as_str()) {
        return format!("{err}\n(couldn't list the S3 objects to diagnose: {pe})");
    }
    err.to_string()
}

/// Human-readable verdict from the list-only S3 probe (object counts + how many
/// are 0 bytes + whether the cstorch `__METADATA__` object is empty). Ordered
/// most-diagnostic-first: no objects → wrong path; all empty → empty checkpoint;
/// empty metadata → interrupted save; some empty → partial; otherwise the data
/// looks intact so it's a format/version issue rather than missing bytes.
fn diagnose_s3_probe(p: &serde_json::Value) -> String {
    let u64_field = |k: &str| p.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
    let total = u64_field("total");
    let empty = u64_field("empty");
    let bytes = u64_field("bytes");
    let prefix = p
        .get("prefix")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let meta_key = p
        .get("metadata_key")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("__METADATA__");
    let meta_empty = p
        .get("metadata_empty")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    if total == 0 {
        format!(
            "No objects exist under the S3 prefix `{prefix}` — the path is wrong or the \
             checkpoint isn't there. Check the prefix / step directory."
        )
    } else if empty == total {
        format!(
            "The checkpoint is empty: all {total} objects under `{prefix}` are 0 bytes \
             ({} total). Every key was created but no data was ever written — an incomplete \
             or failed save/upload. Re-save or re-upload the checkpoint, or point at a \
             complete one (check for a sibling step under the same prefix).",
            crate::utils::format_size(bytes as usize)
        )
    } else if meta_empty {
        format!(
            "The checkpoint's metadata object `{meta_key}` is empty (0 bytes), so cstorch has \
             nothing to load ({empty} of {total} objects are 0 bytes). The metadata wasn't \
             written — most likely an interrupted save."
        )
    } else if empty > 0 {
        format!(
            "The checkpoint looks partially written: {empty} of {total} objects under \
             `{prefix}` are 0 bytes ({} total). It's incomplete.",
            crate::utils::format_size(bytes as usize)
        )
    } else {
        format!(
            "The {total} objects under `{prefix}` look intact ({} total), so this is likely a \
             format/version cstorch can't load rather than a missing/empty checkpoint.",
            crate::utils::format_size(bytes as usize)
        )
    }
}

/// Parse the remote script's optional `s3_objects` / `s3_warnings` fields into
/// [`S3Meta`]. Missing / malformed fields degrade to empty (never an error) — the
/// tensor dump is what matters; S3 metadata is best-effort.
fn parse_s3_meta(v: &serde_json::Value) -> S3Meta {
    let str_map = |val: Option<&serde_json::Value>| -> BTreeMap<String, String> {
        val.and_then(|m| m.as_object())
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
                    .collect()
            })
            .unwrap_or_default()
    };
    let objects = v
        .get("s3_objects")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|o| {
                    let key = o.get("key").and_then(|x| x.as_str())?.to_string();
                    // The remote dump script emits `["algo", "value"]`; store it as
                    // a named pair.
                    let checksum = o
                        .get("checksum")
                        .and_then(|c| c.as_array())
                        .and_then(|c| Some((c.first()?.as_str()?, c.get(1)?.as_str()?)))
                        .map(|(algo, value)| S3Checksum {
                            algorithm: algo.to_string(),
                            value: value.to_string(),
                        });
                    Some(S3Object {
                        key,
                        size: o.get("size").and_then(|x| x.as_u64()).unwrap_or(0),
                        etag: o
                            .get("etag")
                            .and_then(|x| x.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        checksum,
                        last_modified: o
                            .get("last_modified")
                            .and_then(|x| x.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        user_meta: str_map(o.get("metadata")),
                        // `tags` absent in the JSON ⇒ couldn't be read (None);
                        // present (even empty) ⇒ read successfully.
                        tags: o.get("tags").map(|t| str_map(Some(t))),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let warnings = v
        .get("s3_warnings")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|w| w.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    S3Meta { objects, warnings }
}

/// Assemble a remote read's tensors/metadata/config into the central
/// [`Checkpoint`](crate::model::Checkpoint) model. Pure (no I/O) so it's unit-tested
/// without a live SSH host: groups tensors by their stamped `source_path` into one
/// [`ShardHeader`](crate::model::ShardHeader) each (one shard for an `s3://`
/// checkpoint, one per file for a remote safetensors dir), first-seen order;
/// `__metadata__` (unstamped) rides on the first shard. No local files/index/bytes.
fn assemble_remote_checkpoint(
    host: &str,
    src: &str,
    tensors: Vec<TensorInfo>,
    mut metadata: Vec<MetadataInfo>,
    config: Option<crate::config::ModelConfig>,
) -> crate::model::Checkpoint {
    use crate::model::{Checkpoint, ShardHeader, Source};
    let source = if src.starts_with("s3://") {
        Source::S3 {
            uri: src.to_string(),
        }
    } else {
        Source::Sftp {
            host: host.to_string(),
            root: src.to_string(),
        }
    };
    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<TensorInfo>> = HashMap::new();
    for t in tensors {
        let p = t.source_path.clone();
        if !groups.contains_key(&p) {
            order.push(p.clone());
        }
        groups.entry(p).or_default().push(t);
    }
    let shards: Vec<ShardHeader> = order
        .into_iter()
        .enumerate()
        .map(|(i, p)| ShardHeader {
            tensors: groups.remove(&p).unwrap_or_default(),
            metadata: if i == 0 {
                std::mem::take(&mut metadata)
            } else {
                Vec::new()
            },
            path: p,
            total_len: 0,
            header_len: 0,
        })
        .collect();
    Checkpoint {
        source,
        root: src.to_string(),
        files: Vec::new(),
        shards,
        config,
        index: Vec::new(),
        s3: None,
    }
}

/// Map a torch dtype string (`torch.float16`) to the display name used elsewhere
/// (`F16`); unknown types pass through uppercased.
fn map_dtype(torch: &str) -> String {
    let s = torch.strip_prefix("torch.").unwrap_or(torch);
    match s {
        "float16" => "F16",
        "bfloat16" => "BF16",
        "float32" => "F32",
        "float64" => "F64",
        "float8_e4m3fn" => "F8_E4M3",
        "float8_e5m2" => "F8_E5M2",
        "int8" => "I8",
        "uint8" => "U8",
        "int16" => "I16",
        "uint16" => "U16",
        "int32" => "I32",
        "uint32" => "U32",
        "int64" => "I64",
        "uint64" => "U64",
        "bool" => "BOOL",
        other => return other.to_uppercase(),
    }
    .to_string()
}

/// Merge per-shard parse results into one checkpoint: order by shard index (so the
/// result is deterministic regardless of which parallel reader finished first),
/// then flatten, keeping the first occurrence of each tensor / metadata name.
fn merge_shards(mut shards: Vec<ShardParse>) -> (Vec<TensorInfo>, Vec<MetadataInfo>) {
    shards.sort_by_key(|(idx, _, _)| *idx);
    let (mut tensors, mut metadata) = (Vec::new(), Vec::new());
    let (mut seen_t, mut seen_m) = (HashSet::new(), HashSet::new());
    for (_, ts, ms) in shards {
        for t in ts {
            if seen_t.insert(t.name.clone()) {
                tensors.push(t);
            }
        }
        for m in ms {
            if seen_m.insert(m.name.clone()) {
                metadata.push(m);
            }
        }
    }
    (tensors, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cstorch_script_embeds_source_safely() {
        let s = dump_script("s3://b/k", false);
        assert!(s.contains("SRC = \"s3://b/k\""));
        assert!(s.contains("import cerebras.pytorch"));
        assert!(s.contains(SENTINEL));
    }

    #[test]
    fn cstorch_script_probes_s3_on_load_failure() {
        // A load failure lists the objects (read-only) and merges the probe into
        // the error object so the caller can diagnose an empty/partial checkpoint.
        let s = dump_script("s3://b/k", false);
        assert!(s.contains("def probe_s3"), "{s}");
        assert!(s.contains("**probe_s3(SRC)"), "{s}");
        assert!(s.contains("\"s3_probe\""));
        assert!(s.contains("__METADATA__"));
        // The probe must stay list-only + read-only.
        assert!(s.contains("list_objects_v2"));
        for forbidden in ["put_object", "delete_object", "upload", "copy_object"] {
            assert!(
                !s.contains(forbidden),
                "probe must stay read-only: {forbidden}"
            );
        }
    }

    #[test]
    fn cstorch_script_fetches_s3_metadata_only_when_wanted() {
        // Not wanted (interactive browse / check) → no boto3 work.
        let off = dump_script("s3://b/ckpt", false);
        assert!(off.contains("WANT_S3 = False"), "{off}");

        let s = dump_script("s3://b/ckpt", true);
        assert!(s.contains("WANT_S3 = True"), "{s}");
        // Fetches object metadata via boto3, read-only calls only.
        assert!(s.contains("boto3.client(\"s3\")"));
        assert!(s.contains("list_objects_v2"));
        assert!(s.contains("head_object"));
        assert!(s.contains("ChecksumMode=\"ENABLED\""));
        assert!(s.contains("get_object_tagging"));
        assert!(s.contains("s3_objects"));
        assert!(s.contains("s3_warnings"));
        // Reports S3 progress as a second phase (so the bar doesn't sit at 100%).
        assert!(s.contains("\"s3\""));
        // Read-only: never writes / uploads / deletes / puts tags.
        for forbidden in [
            "put_object",
            "upload",
            "delete_object",
            "put_object_tagging",
            "copy_object",
        ] {
            assert!(
                !s.contains(forbidden),
                "script must stay read-only: {forbidden}"
            );
        }
    }

    #[test]
    fn list_script_is_read_only_list_and_embeds_uri() {
        let s = list_script("s3://bucket/ckpt");
        assert!(s.contains("SRC = \"s3://bucket/ckpt\""));
        assert!(s.contains("boto3.client(\"s3\")"));
        assert!(s.contains("list_objects_v2"));
        assert!(s.contains(SENTINEL));
        // List only — no per-object HEAD/tag calls, and nothing that mutates.
        for forbidden in [
            "head_object",
            "get_object_tagging",
            "put_object",
            "upload",
            "delete_object",
            "copy_object",
        ] {
            assert!(
                !s.contains(forbidden),
                "list script must stay list-only + read-only: {forbidden}"
            );
        }
    }

    #[test]
    fn value_diff_script_embeds_inputs_and_is_read_only() {
        let pairs = vec![
            ("old.w".to_string(), "new.w".to_string()),
            ("x".to_string(), "x".to_string()),
        ];
        let opts = RemoteValueOpts {
            values: true,
            histogram: true,
            bins: Some(32),
            full_hist: false,
            jobs: 8,
        };
        let s = value_diff_script("s3://b/old", "s3://b/new", &pairs, &opts);
        assert!(s.contains(r#"OLD = "s3://b/old""#), "{s}");
        assert!(s.contains(r#"NEW = "s3://b/new""#));
        assert!(s.contains(r#"[["old.w","new.w"],["x","x"]]"#), "{s}");
        assert!(s.contains("WANT_VALUES = True"));
        assert!(s.contains("WANT_HIST = True"));
        assert!(s.contains("FULL_HIST = False"));
        assert!(s.contains("BINS = 32"));
        assert!(s.contains("import cerebras.pytorch"));
        assert!(s.contains("np.histogram"));
        assert!(s.contains(SENTINEL));
        // Streams live status events (load phase + per-tensor start) for the view.
        assert!(s.contains(STATUS_TAG), "{s}");
        assert!(s.contains("stat(\"load\\told\")"), "{s}");
        assert!(s.contains("stat(\"start\\t"), "{s}");
        assert!(s.contains("res[\"bytes\"]"), "{s}");
        // Parallel reads driven by JOBS via a thread pool.
        assert!(s.contains("JOBS = 8"), "{s}");
        assert!(s.contains("ThreadPoolExecutor"), "{s}");
        // Read-only: it loads + reads to compare, never saves/writes/uploads/deletes.
        for forbidden in [
            "cstorch.save",
            "torch.save",
            ".save(",
            "put_object",
            "upload",
            "delete_object",
            "copy_object",
            "open(",
        ] {
            assert!(
                !s.contains(forbidden),
                "value-diff script must stay read-only: {forbidden}"
            );
        }
        // No histogram wanted → the branch and its bins are dropped from the script.
        let vonly = value_diff_script(
            "s3://b/o",
            "s3://b/n",
            &pairs,
            &RemoteValueOpts {
                values: true,
                histogram: false,
                bins: None,
                full_hist: false,
                jobs: 1,
            },
        );
        assert!(vonly.contains("WANT_HIST = False"), "{vonly}");
        assert!(vonly.contains("BINS = None"), "{vonly}");
        assert!(vonly.contains("JOBS = 1"), "{vonly}");
    }

    #[test]
    fn parse_value_diff_collects_results_and_flags_fatal() {
        let out = format!(
            "{s}{{\"name\":\"a.w\",\"values\":{{\"elements\":10,\"differing\":3,\"max_abs\":0.5,\"mean_abs\":0.1,\"nonfinite_mismatch\":1}}}}\n\
             motd noise\n\
             {s}{{\"name\":\"b.w\",\"histogram\":{{\"tvd\":0.25,\"n\":40}}}}\n\
             {s}{{\"name\":\"c.w\",\"error\":\"shapes differ\"}}\n\
             {s}{{\"summary\":{{\"tensors\":3,\"compared\":2,\"bytes\":4096,\"elapsed_s\":1.5}}}}\n",
            s = SENTINEL
        );
        let (m, stats) = parse_value_diff(&out, "host").unwrap();
        assert_eq!(m.len(), 3);
        let st = stats.expect("summary line parsed");
        assert_eq!(st.tensors, 3);
        assert_eq!(st.compared, 2);
        assert_eq!(st.bytes, 4096);
        assert!((st.elapsed_s - 1.5).abs() < 1e-9);
        let vd = m["a.w"].values.as_ref().unwrap();
        assert_eq!(vd.elements, 10);
        assert_eq!(vd.differing, 3);
        assert_eq!(vd.max_abs, 0.5);
        assert_eq!(vd.nonfinite_mismatch, 1);
        assert_eq!(m["b.w"].hist_shift, Some((0.25, 40)));
        assert!(m["b.w"].hist_full.is_none()); // summary only (no bin arrays)
        assert_eq!(m["c.w"].error.as_deref(), Some("shapes differ"));

        // A name-less error line is a fatal whole-run failure.
        let fatal = format!("{SENTINEL}{{\"error\":\"import numpy/torch failed\"}}\n");
        assert!(parse_value_diff(&fatal, "host").is_err());
        // No sentinel output at all is also an error.
        assert!(parse_value_diff("just motd\n", "host").is_err());
        // A run with no summary line still parses (stats just absent).
        let (_m, none) =
            parse_value_diff(&format!("{SENTINEL}{{\"name\":\"z\"}}\n"), "host").unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn compare_status_classifies_results() {
        let v = |s: &str| serde_json::from_str::<serde_json::Value>(s).unwrap();
        assert_eq!(
            compare_status(&v(r#"{"name":"a","error":"shapes differ"}"#)),
            CompareStatus::Error
        );
        assert_eq!(
            compare_status(&v(r#"{"name":"a","values":{"differing":0}}"#)),
            CompareStatus::Identical
        );
        assert_eq!(
            compare_status(&v(r#"{"name":"a","values":{"differing":5}}"#)),
            CompareStatus::Changed
        );
        // A distribution shift alone counts as changed.
        assert_eq!(
            compare_status(&v(r#"{"name":"a","histogram":{"tvd":0.3,"n":40}}"#)),
            CompareStatus::Changed
        );
        assert_eq!(
            compare_status(&v(r#"{"name":"a","histogram":{"tvd":0.0,"n":40}}"#)),
            CompareStatus::Identical
        );
    }

    #[test]
    fn parse_value_diff_reads_full_histogram() {
        let out = format!(
            "{SENTINEL}{{\"name\":\"w\",\"histogram\":{{\"tvd\":0.5,\"n\":2,\"lo\":-1.0,\"hi\":1.0,\
             \"old\":[3,1],\"new\":[1,3],\"old_total\":4,\"new_total\":4,\"old_nonfinite\":0,\"new_nonfinite\":0}}}}\n"
        );
        let (m, _stats) = parse_value_diff(&out, "h").unwrap();
        let hf = m["w"].hist_full.as_ref().unwrap();
        assert_eq!(hf.n, 2);
        assert_eq!(hf.old, vec![3, 1]);
        assert_eq!(hf.new, vec![1, 3]);
        assert_eq!(hf.old_total, 4);
        assert!((hf.tvd() - 0.5).abs() < 1e-9);
        match hf.bins {
            crate::sample::HistBins::Range { lo, hi } => {
                assert_eq!(lo, -1.0);
                assert_eq!(hi, 1.0);
            }
            _ => panic!("expected Range bins"),
        }
        // The summary shift is present alongside the full data.
        assert_eq!(m["w"].hist_shift, Some((0.5, 2)));
    }

    #[test]
    fn repack_verify_script_embeds_inputs_and_checks_format() {
        let pairs = vec![("w.down".to_string(), "w.down".to_string())];
        let s = repack_verify_script("s3://b/old", "s3://b/new", &pairs, 3, false);
        assert!(s.contains(r#"OLD = "s3://b/old""#), "{s}");
        assert!(s.contains(r#"[["w.down","w.down"]]"#), "{s}");
        assert!(s.contains("BITS = 3"), "{s}");
        // Derives the fold, decodes, and runs BOTH format checks.
        assert!(s.contains("fold = (E + W - 1) // W"), "{s}");
        assert!(s.contains("sparse_bad"), "{s}");
        assert!(s.contains("dense_bad"), "{s}");
        // Streams the objects itself (thread-safe client) with byte progress, a
        // decoded sample window, in parallel.
        assert!(s.contains("ThreadPoolExecutor"), "{s}");
        assert!(s.contains("get_object"), "{s}");
        assert!(s.contains("stat(\"size"), "{s}");
        assert!(s.contains("stat(\"bytes"), "{s}");
        assert!(s.contains("res[\"sample\"]"), "{s}");
        // Also diffs the sibling codebook + scale tensors.
        assert!(
            s.contains("cmp_aux") && s.contains(".codebook") && s.contains(".qscale"),
            "{s}"
        );
        assert!(s.contains(STATUS_TAG) && s.contains(SENTINEL));
        // Read-only: reads objects, never writes/uploads/deletes.
        for forbidden in [
            "cstorch.save",
            "torch.save",
            "put_object",
            "delete_object",
            "open(",
        ] {
            assert!(
                !s.contains(forbidden),
                "repack script must stay read-only: {forbidden}"
            );
        }
    }

    #[test]
    fn parse_repack_reads_results_and_flags_format() {
        let out = format!(
            "{s}{{\"name\":\"a\",\"elements\":100,\"differing\":0,\"sparse_bad\":0,\"dense_bad\":0,\"fold\":5,\
             \"codebook\":{{\"old_name\":\"a.codebook\",\"new_name\":\"a.codebook\",\"old_present\":true,\"new_present\":true,\
             \"shape_old\":[8,8],\"shape_new\":[8,8],\"elements\":64,\"differing\":0,\"max_abs\":0.0,\"mean_abs\":0.0}}}}\n\
             {s}{{\"name\":\"b\",\"elements\":100,\"differing\":7,\"sparse_bad\":0,\"dense_bad\":0,\"fold\":5,\"first\":[3,12,5,2],\
             \"codebook\":{{\"old_name\":\"b.codebook\",\"new_name\":\"b.codebook\",\"old_present\":true,\"new_present\":true,\
             \"shape_old\":[8,8],\"shape_new\":[8,8],\"elements\":64,\"differing\":9,\"max_abs\":0.03,\"mean_abs\":0.001}},\
             \"qscale\":{{\"old_name\":\"b.qscale\",\"new_name\":\"b.qscale\",\"old_present\":true,\"new_present\":true,\
             \"shape_old\":[8,2],\"shape_new\":[8,3]}}}}\n\
             {s}{{\"name\":\"c\",\"elements\":100,\"differing\":0,\"sparse_bad\":9,\"dense_bad\":0,\"fold\":5,\
             \"codebook\":{{\"old_name\":\"c.codebook\",\"new_name\":\"c.codebook\",\"old_present\":true,\"new_present\":false}}}}\n\
             {s}{{\"summary\":{{\"tensors\":3,\"compared\":3,\"bytes\":2048,\"elapsed_s\":2.0}}}}\n",
            s = SENTINEL
        );
        let (m, stats) = parse_repack(&out, "host").unwrap();
        assert_eq!(m.len(), 3);
        assert!(m["a"].equivalent(), "a should be equivalent");
        let acb = m["a"].codebook.as_ref().unwrap();
        assert!(acb.present() && acb.differing == 0);
        assert_eq!(acb.new_name, "a.codebook");
        assert_eq!(m["b"].differing, 7);
        assert_eq!(m["b"].first_mismatch, Some((3, 12, 5, 2)));
        assert!(!m["b"].equivalent());
        // The sibling codebook diff is parsed alongside, with its name + shape.
        let cb = m["b"].codebook.as_ref().unwrap();
        assert_eq!(cb.new_name, "b.codebook");
        assert_eq!(cb.shape, vec![8, 8]);
        assert_eq!(cb.differing, 9);
        assert!((cb.max_abs - 0.03).abs() < 1e-9);
        // A shape mismatch on a sibling is carried too.
        assert_eq!(
            m["b"].qscale.as_ref().unwrap().shape_mismatch,
            Some((vec![8, 2], vec![8, 3]))
        );
        // A sibling missing on one side is reported (not silently dropped).
        assert!(!m["c"].codebook.as_ref().unwrap().present());
        // A format violation makes it non-equivalent even with 0 differing.
        assert_eq!(m["c"].sparse_bad, 9);
        assert!(!m["c"].equivalent());
        assert_eq!(stats.unwrap().compared, 3);

        // A name-less error line is fatal.
        assert!(parse_repack(&format!("{SENTINEL}{{\"error\":\"boom\"}}\n"), "h").is_err());
    }

    #[test]
    fn parse_list_reads_pairs_and_surfaces_errors() {
        let ok = parse_list(r#"{"objects":[["a/b.bin",100],["c.json",5]]}"#).unwrap();
        assert_eq!(
            ok,
            vec![("a/b.bin".to_string(), 100), ("c.json".to_string(), 5)]
        );
        // A remote error becomes an Err.
        assert!(parse_list(r#"{"error":"access denied"}"#).is_err());
        // Malformed entries are skipped, not fatal.
        let mixed = parse_list(r#"{"objects":[["good",1],["bad"],[1,2]]}"#).unwrap();
        assert_eq!(mixed, vec![("good".to_string(), 1)]);
    }

    #[test]
    fn parses_safetensors_dump_with_metadata_and_marks_source() {
        let json = r#"{"tensors":[
            {"name":"model.embed_tokens.weight","dtype":"BF16","shape":[151936,2048],"itemsize":2}
        ],"metadata":[{"name":"format","value":"pt","value_type":"string"}]}"#;
        let (t, m, s3) = parse_dump(json, "lab@host:/opt/models/ckpt").unwrap();
        assert_eq!(t[0].dtype, "BF16");
        assert_eq!(t[0].shape, vec![151936, 2048]);
        assert_eq!(t[0].source_path, "lab@host:/opt/models/ckpt");
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "format");
        assert_eq!(m[0].value, "pt");
        // No `s3_objects` in the JSON → empty (this path is for safetensors dumps).
        assert!(s3.objects.is_empty() && s3.warnings.is_empty());
    }

    #[test]
    fn parses_s3_objects_and_warnings() {
        let json = r#"{"tensors":[
            {"name":"w","dtype":"F16","shape":[4],"itemsize":2}
        ],"metadata":[],"s3_objects":[
            {"key":"shard0.dat","size":1024,"etag":"abc","last_modified":"2026-01-02T03:04:05+00:00",
             "checksum":["sha256","deadbeef"],"metadata":{"run":"42"},"tags":{"env":"prod"}},
            {"key":"shard1.dat","size":2048,"etag":"def","last_modified":"2026-01-02T03:04:06+00:00"}
        ],"s3_warnings":["tags unavailable (needs s3:GetObjectTagging): AccessDenied"]}"#;
        let (_t, _m, s3) = parse_dump(json, "s3://b/ckpt").unwrap();
        assert_eq!(s3.objects.len(), 2);
        let o0 = &s3.objects[0];
        assert_eq!(o0.key, "shard0.dat");
        assert_eq!(o0.size, 1024);
        assert_eq!(
            o0.checksum,
            Some(S3Checksum {
                algorithm: "sha256".into(),
                value: "deadbeef".into()
            })
        );
        assert_eq!(o0.user_meta.get("run").map(String::as_str), Some("42"));
        assert_eq!(
            o0.tags
                .as_ref()
                .and_then(|t| t.get("env"))
                .map(String::as_str),
            Some("prod")
        );
        // Second object had no `tags` key → None (couldn't be read), distinct from empty.
        assert!(s3.objects[1].tags.is_none());
        assert_eq!(s3.warnings.len(), 1);
    }

    #[test]
    fn source_path_is_scp_style_but_leaves_s3() {
        let r = RemoteRead::new("lab@host".into(), "~/venv".into());
        assert_eq!(r.source_path("s3://b/k"), "s3://b/k");
        assert_eq!(r.source_path("/opt/models/x"), "lab@host:/opt/models/x");
    }

    #[test]
    fn detects_remote_sources() {
        assert!(is_remote_source("s3://bucket/ckpt"));
        assert!(is_remote_source("lab@host:/opt/models/x"));
        assert!(is_remote_source("host:relative/path"));
        // local paths are never remote, even with a ':' inside a subdir
        assert!(!is_remote_source("/opt/models/x"));
        assert!(!is_remote_source("./model.safetensors"));
        assert!(!is_remote_source("dir/a:b"));
    }

    #[test]
    fn parses_dump_into_tensors() {
        let json = r#"{"tensors":[
            {"name":"a.weight","dtype":"torch.float16","shape":[6,4],"itemsize":2},
            {"name":"b","dtype":"torch.int32","shape":[5],"itemsize":4}
        ],"metadata":[]}"#;
        let (t, _m, _s3) = parse_dump(json, "s3://bucket/ckpt").unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].dtype, "F16");
        assert_eq!(t[0].shape, vec![6, 4]);
        assert_eq!(t[0].num_elements, 24);
        assert_eq!(t[0].size_bytes, 48);
        assert_eq!(t[0].source_path, "s3://bucket/ckpt");
        assert_eq!(t[1].dtype, "I32");
    }

    #[test]
    fn assembles_s3_checkpoint_into_one_shard() {
        let ts = vec![tensor("a"), tensor("b")]; // both carry source_path "h:/p"
        let ck = assemble_remote_checkpoint(
            "lab@host",
            "s3://bucket/ckpt",
            ts,
            vec![meta("format")],
            None,
        );
        assert!(matches!(ck.source, crate::model::Source::S3 { .. }));
        assert_eq!(ck.root, "s3://bucket/ckpt");
        assert!(ck.files.is_empty() && ck.index.is_empty() && ck.s3.is_none());
        // All tensors share one source_path → a single shard, carrying the metadata.
        assert_eq!(ck.shards.len(), 1);
        assert_eq!(ck.shards[0].tensors.len(), 2);
        assert_eq!(ck.shards[0].metadata.len(), 1);
        assert_eq!(ck.tensors().count(), 2);
    }

    #[test]
    fn assembles_sftp_dir_into_one_shard_per_file() {
        let mk = |name: &str, path: &str| {
            let mut t = tensor(name);
            t.source_path = path.to_string();
            t
        };
        // Two shard files, tensors interleaved — grouping is by source_path, and
        // metadata rides on the first shard only.
        let ts = vec![
            mk("a", "host:/ckpt/shard-0.safetensors"),
            mk("b", "host:/ckpt/shard-1.safetensors"),
            mk("c", "host:/ckpt/shard-0.safetensors"),
        ];
        let ck = assemble_remote_checkpoint("host", "/ckpt", ts, vec![meta("format")], None);
        assert!(matches!(ck.source, crate::model::Source::Sftp { .. }));
        assert_eq!(ck.shards.len(), 2);
        assert_eq!(ck.shards[0].path, "host:/ckpt/shard-0.safetensors");
        assert_eq!(ck.shards[0].tensors.len(), 2); // a, c
        assert_eq!(ck.shards[1].tensors.len(), 1); // b
        assert_eq!(ck.shards[0].metadata.len(), 1);
        assert!(ck.shards[1].metadata.is_empty());
    }

    #[test]
    fn surfaces_remote_error() {
        // No probe attached → the raw cstorch error passes through unchanged.
        let err = parse_dump(r#"{"error":"cstorch.load failed: boom"}"#, "s3://x/y");
        assert!(err.unwrap_err().to_string().contains("boom"));
    }

    #[test]
    fn diagnoses_all_empty_checkpoint() {
        // The real-world case: every object under the prefix (incl. __METADATA__)
        // is 0 bytes → cstorch's dill.loads hits EOFError.
        let json = r#"{"error":"cstorch.load failed: EOFError('Ran out of input')",
            "s3_probe":{"prefix":"kimi/3bit/260720","total":1642,"empty":1642,"bytes":0,
                        "metadata_key":"__METADATA__","metadata_empty":true,
                        "sample":[["__METADATA__",0],["lm_head.weight",0]]}}"#;
        let msg = parse_dump(json, "s3://b/k").unwrap_err().to_string();
        assert!(msg.contains("empty"), "{msg}");
        assert!(msg.contains("all 1642 objects"), "{msg}");
        assert!(msg.contains("0 bytes"), "{msg}");
        // The raw cstorch error is still included, but after the diagnosis.
        assert!(msg.contains("Ran out of input"), "{msg}");
    }

    #[test]
    fn diagnoses_empty_metadata_object() {
        // Data objects have bytes, but the metadata object is empty.
        let json = r#"{"error":"cstorch.load failed: EOFError('Ran out of input')",
            "s3_probe":{"prefix":"m/step","total":10,"empty":1,"bytes":4096,
                        "metadata_key":"__METADATA__","metadata_empty":true,"sample":[]}}"#;
        let msg = parse_dump(json, "s3://b/k").unwrap_err().to_string();
        assert!(msg.contains("`__METADATA__` is empty"), "{msg}");
        assert!(msg.contains("interrupted save"), "{msg}");
    }

    #[test]
    fn diagnoses_wrong_prefix() {
        let json = r#"{"error":"cstorch.load failed: boom",
            "s3_probe":{"prefix":"typo/here","total":0,"empty":0,"bytes":0,
                        "metadata_key":null,"metadata_empty":false,"sample":[]}}"#;
        let msg = parse_dump(json, "s3://b/k").unwrap_err().to_string();
        assert!(msg.contains("No objects exist"), "{msg}");
        assert!(msg.contains("typo/here"), "{msg}");
    }

    #[test]
    fn diagnoses_intact_objects_as_format_issue() {
        // Nothing empty → the bytes are there, so it's a format/version problem.
        let json = r#"{"error":"cstorch.load failed: KeyError('spec')",
            "s3_probe":{"prefix":"m/s","total":5,"empty":0,"bytes":1048576,
                        "metadata_key":"__METADATA__","metadata_empty":false,"sample":[]}}"#;
        let msg = parse_dump(json, "s3://b/k").unwrap_err().to_string();
        assert!(msg.contains("look intact"), "{msg}");
        assert!(msg.contains("format/version"), "{msg}");
    }

    #[test]
    fn falls_back_when_probe_itself_failed() {
        let json = r#"{"error":"cstorch.load failed: boom",
            "s3_probe_error":"NoCredentialsError()"}"#;
        let msg = parse_dump(json, "s3://b/k").unwrap_err().to_string();
        assert!(msg.contains("boom"), "{msg}");
        assert!(msg.contains("couldn't list the S3 objects"), "{msg}");
        assert!(msg.contains("NoCredentialsError"), "{msg}");
    }

    #[test]
    fn maps_common_dtypes() {
        assert_eq!(map_dtype("torch.bfloat16"), "BF16");
        assert_eq!(map_dtype("torch.uint8"), "U8");
        assert_eq!(map_dtype("torch.weirdtype"), "WEIRDTYPE");
    }

    fn tensor(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: "F16".into(),
            shape: vec![1],
            size_bytes: 2,
            num_elements: 1,
            storage: Storage::Unknown,
            source_path: "h:/p".into(),
            layout: Layout::None,
        }
    }
    fn meta(name: &str) -> MetadataInfo {
        MetadataInfo {
            name: name.to_string(),
            value: "v".into(),
            value_type: "string".into(),
        }
    }

    #[test]
    fn merge_shards_orders_by_index_and_dedups_first_seen() {
        // Deliberately out of order (as parallel readers may finish): shard 2 then
        // 0 then 1. `b` appears in shards 0 and 2 → the shard-0 copy wins.
        let parts = vec![
            (2, vec![tensor("c")], vec![meta("fmt")]),
            (0, vec![tensor("a"), tensor("b")], vec![meta("fmt")]),
            (1, vec![tensor("b")], vec![]),
        ];
        let (t, m) = merge_shards(parts);
        let names: Vec<&str> = t.iter().map(|x| x.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]); // shard order, `b` deduped
        assert_eq!(m.len(), 1); // duplicate `fmt` metadata collapsed
        assert_eq!(m[0].name, "fmt");
    }
}
