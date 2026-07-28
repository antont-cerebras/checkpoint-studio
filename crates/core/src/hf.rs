//! Reading a **Hugging Face Hub** repository's structure over HTTPS — no download.
//!
//! A safetensors file starts with an 8-byte little-endian header length followed by that
//! many bytes of JSON describing every tensor: name, dtype, shape, and its byte range in
//! the file. The Hub serves files with HTTP `Range` support, so the whole structure of a
//! multi-terabyte repo is a couple of small requests per shard — typically a few kilobytes
//! each. Nothing else is fetched: the weights stay on the Hub.
//!
//! Because the result is an ordinary [`Checkpoint`], every derived view works with no
//! further work — the tensor tree, the compact (family-folded) tree, the statistics, the
//! health check, and the byte-layout map, which needs exactly the `data_offsets` the
//! header already carries.
//!
//! **What this cannot do.** Data views (heatmap, values, histogram, whole-tensor
//! statistics) read tensor *bytes*, which is the one thing this reader deliberately never
//! fetches — the same restriction a `--ssh-proxy` source has, and reported the same way.
//!
//! Two Hub endpoints are used, both public and unauthenticated for a public repo:
//!
//! - `GET /api/models/{id}/tree/{rev}?recursive=1` — the file listing with sizes.
//! - `GET /{id}/resolve/{rev}/{path}` with a `Range` header — the bytes of one file.
//!
//! A `HF_TOKEN` in the environment is sent as a bearer token, so a gated or private repo
//! works with the same token the `huggingface_hub` client uses.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::model::{Checkpoint, FileEntry, FsNode, ShardHeader, Source};
use crate::tree::{MetadataInfo, TensorInfo};

/// Where the Hub lives. Overridable so a test or a mirror can point elsewhere.
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

/// Per-request timeout. Generous: the listing for a 96-shard repo is a single large JSON
/// response, and the Hub redirects file reads to a CDN.
const TIMEOUT: Duration = Duration::from_mins(1);

/// How many shard headers to read at once. Each read is almost entirely waiting (~0.35 s of
/// latency for a few KB), so concurrency is what determines the wall clock — 16 keeps a
/// 96-shard repo to a few seconds. Higher risks the Hub throttling, which costs more than it
/// saves.
const PARALLEL_READS: usize = 16;

/// How far a [`read_checkpoint`] has got, shared with whoever is drawing.
///
/// Header reads happen on a worker pool while the UI draws, so the two need a cell rather
/// than a callback return: `total` is 0 until the listing lands (nothing to count yet — a
/// caller shows a spinner), then it is the shard count and `done` climbs to it. Shard count
/// rather than bytes because every header read costs about the same, so the count is the
/// honest unit — a bytes-based bar over 96 wildly different shard *sizes* would jump.
#[derive(Debug, Default)]
pub struct ReadProgress {
    pub done: std::sync::atomic::AtomicUsize,
    pub total: std::sync::atomic::AtomicUsize,
}

impl ReadProgress {
    /// `(done, total)`; `total == 0` means the listing hasn't landed yet.
    #[must_use]
    pub fn get(&self) -> (usize, usize) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.done.load(Relaxed), self.total.load(Relaxed))
    }
}

/// A repository and the revision to read it at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoRef {
    /// `owner/name`.
    pub id: String,
    /// A branch, tag or commit sha. `main` unless the spec said otherwise.
    pub revision: String,
}

impl RepoRef {
    /// The display form, which is also a spec [`parse`] accepts back.
    #[must_use]
    pub fn spec(&self) -> String {
        if self.revision == "main" {
            format!("hf://{}", self.id)
        } else {
            format!("hf://{}@{}", self.id, self.revision)
        }
    }
}

/// Whether `s` looks like a Hugging Face repo reference — an `hf://` URI or a
/// huggingface.co URL. Deliberately *not* a bare `owner/name`: that is indistinguishable
/// from a relative directory path, and silently reaching the network for something the
/// user meant as a local path would be the wrong surprise.
#[must_use]
pub fn is_uri(s: &str) -> bool {
    s.starts_with("hf://") || host_url_rest(s).is_some()
}

/// The path part of a `https://huggingface.co/…` URL, or `None` when `s` is not one.
fn host_url_rest(s: &str) -> Option<&str> {
    for prefix in [
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://www.huggingface.co/",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return Some(rest);
        }
    }
    None
}

/// Parse a repo reference:
///
/// - `hf://owner/name` / `hf://owner/name@revision`
/// - `https://huggingface.co/owner/name` — with an optional `/tree/REV` or `/blob/REV/…`
///   as the web UI writes it, and any `#fragment` or `?query` ignored, so a URL pasted
///   from a browser works as-is.
pub fn parse(spec: &str) -> Result<RepoRef> {
    let rest = if let Some(r) = spec.strip_prefix("hf://") {
        r
    } else if let Some(r) = host_url_rest(spec) {
        r
    } else {
        bail!("not a Hugging Face reference: {spec} (expected hf://owner/name)");
    };
    // A pasted URL may carry a fragment (`#2-model-summary`) or query.
    let rest = rest
        .split(['#', '?'])
        .next()
        .unwrap_or(rest)
        .trim_end_matches('/');

    // `owner/name@rev` — the `@` form only applies to the last segment we keep.
    let (path, at_revision) = rest
        .split_once('@')
        .map_or((rest, None), |(p, r)| (p, Some(r)));

    let mut segments = path.split('/').filter(|s| !s.is_empty());
    let (Some(owner), Some(name)) = (segments.next(), segments.next()) else {
        bail!("a Hugging Face reference needs owner/name: {spec}");
    };
    // `/tree/REV` or `/blob/REV/...` from the web UI names the revision.
    let url_revision = match (segments.next(), segments.next()) {
        (Some("tree" | "blob" | "resolve"), Some(rev)) => Some(rev.to_string()),
        _ => None,
    };
    let revision = at_revision
        .map(ToString::to_string)
        .or(url_revision)
        .unwrap_or_else(|| "main".to_string());
    if revision.is_empty() {
        bail!("empty revision in {spec}");
    }
    Ok(RepoRef {
        id: format!("{owner}/{name}"),
        revision,
    })
}

/// One file in the repo, as the listing reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubFile {
    /// Path within the repo (POSIX, `/`-separated).
    pub path: String,
    pub size: u64,
}

/// Parse the `…/tree/…?recursive=1` response: an array of `{type, path, size}`, where
/// directories have `type: "directory"` and no meaningful size.
///
/// LFS-backed files (every real shard) report the pointer's size at the top level and the
/// *object's* size under `lfs.size`; the latter is the one that matters, so it wins when
/// present. Getting this backwards would report a 96-shard checkpoint as a few kilobytes.
pub fn parse_listing(json: &serde_json::Value) -> Result<Vec<HubFile>> {
    let entries = json
        .as_array()
        .context("the Hub file listing should be a JSON array")?;
    let mut out = Vec::new();
    for e in entries {
        if e.get("type").and_then(|t| t.as_str()) == Some("directory") {
            continue;
        }
        let Some(path) = e.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        let size = e
            .get("lfs")
            .and_then(|l| l.get("size"))
            .or_else(|| e.get("size"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        out.push(HubFile {
            path: path.to_string(),
            size,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// The shards whose headers are worth reading, in listing order.
#[must_use]
pub fn safetensors_shards(files: &[HubFile]) -> Vec<&HubFile> {
    files
        .iter()
        .filter(|f| f.path.to_ascii_lowercase().ends_with(".safetensors"))
        .collect()
}

// ---- HTTP ------------------------------------------------------------------------

/// The bearer token to send, if one is in the environment. The same variables the official
/// client reads, so a machine already set up for `huggingface-cli` needs no extra config.
fn token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// A `GET` through `agent` with the Hub's auth applied.
///
/// Every request goes through here — the listing *and* every range read. They didn't
/// before: the range reads omitted the token, so a private repo listed fine and then failed
/// on the first header.
fn get(agent: &ureq::Agent, url: &str) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
    let req = agent.get(url);
    match token() {
        Some(t) => req.header("authorization", &format!("Bearer {t}")),
        None => req,
    }
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(TIMEOUT))
        // Keep enough pooled connections for every reader thread, to *both* hosts involved
        // (the Hub redirects file reads to a CDN). Without this the pool thrashes and each
        // shard pays a fresh TLS handshake — which was the bulk of the wall clock, not the
        // bytes: a 16 KB range and a 256 KB range both cost ~0.35 s, nearly all of it setup.
        .max_idle_connections(PARALLEL_READS * 4)
        .max_idle_connections_per_host(PARALLEL_READS * 2)
        .build()
        .into()
}

/// Fetch the repo's file listing.
fn fetch_listing(repo: &RepoRef) -> Result<Vec<HubFile>> {
    let url = format!(
        "{}/api/models/{}/tree/{}?recursive=1",
        endpoint(),
        repo.id,
        repo.revision
    );
    let mut resp = get(&agent(), &url)
        .call()
        .with_context(|| format!("listing {} at {}", repo.id, repo.revision))?;
    let json: serde_json::Value = resp
        .body_mut()
        .read_json()
        .with_context(|| format!("parsing the file listing for {}", repo.id))?;
    parse_listing(&json)
}

/// Fetch `len` bytes of `path` starting at `start`, via a Range request.
fn fetch_range(
    agent: &ureq::Agent,
    repo: &RepoRef,
    path: &str,
    start: u64,
    len: u64,
) -> Result<Vec<u8>> {
    let url = format!(
        "{}/{}/resolve/{}/{}",
        endpoint(),
        repo.id,
        repo.revision,
        path
    );
    let end = start + len - 1;
    let mut resp = get(agent, &url)
        .header("range", &format!("bytes={start}-{end}"))
        .call()
        .with_context(|| format!("reading {path} bytes {start}-{end}"))?;
    let bytes = resp
        .body_mut()
        .with_config()
        .limit(len + 1024)
        .read_to_vec()
        .with_context(|| format!("reading {path} bytes {start}-{end}"))?;
    // A *short* read is fine — a speculative probe deliberately over-asks, and the server
    // returns what exists. An over-long read would mean ranges were ignored and we got the
    // whole (multi-GB) file, which must not be treated as a header.
    if bytes.len() as u64 > len {
        bail!(
            "{path}: asked for {len} bytes at {start}, got {} — the server ignored the range",
            bytes.len()
        );
    }
    Ok(bytes)
}

/// How much of a shard to read speculatively. The length prefix and the header almost always
/// fit — Kimi-K3's are ~3 KB, and a 500-tensor shard stays well inside this — so it is one
/// request per shard instead of two (a length read, then the JSON). Guessing too small costs
/// one extra request for that shard, not a failure.
///
/// Sized down from 256 KB once measurement showed the cost is latency, not bytes: a range
/// read is ~0.35 s almost regardless of size, so over-asking bought nothing and moved 24 MB
/// for a repo whose headers total ~300 KB.
const HEADER_PROBE: u64 = 32 * 1024;

/// Read one shard's safetensors header: the 8-byte little-endian length, then that many
/// bytes of JSON. Fetched as a single speculative prefix, with a second request only when
/// the header turns out to be larger than [`HEADER_PROBE`].
fn fetch_header(agent: &ureq::Agent, repo: &RepoRef, file: &HubFile) -> Result<ShardHeader> {
    // Never ask for more than the file holds: a tiny shard would otherwise get a range the
    // server rejects as unsatisfiable.
    let probe_len = HEADER_PROBE.min(file.size.max(8));
    let probe = fetch_range(agent, repo, &file.path, 0, probe_len)?;
    let raw = u64::from_le_bytes(
        probe
            .get(..8)
            .and_then(|s| <[u8; 8]>::try_from(s).ok())
            .context("short read on the header length")?,
    );
    let len = crate::stheader::header_len(raw, &file.path)?;
    let json = match probe.get(8..8 + len) {
        // The speculative read covered it.
        Some(slice) => slice.to_vec(),
        // A header bigger than the probe: ask for exactly it.
        None => fetch_range(agent, repo, &file.path, 8, len as u64)?,
    };
    // Tag each tensor with the repo, not just the shard path: everything downstream asks
    // `remote::is_remote_source(&t.source_path)` before opening tensor bytes, and a bare
    // `model-00001.safetensors` looks local — so a data view on a Hub tensor tried to open
    // it from the working directory and reported `No such file or directory` instead of
    // saying the weights aren't here. The `hf://…` prefix makes it unmistakably remote.
    let source = format!("{}/{}", repo.spec(), file.path);
    let (tensors, metadata) = crate::stheader::parse_header(&json, &source)?;
    Ok(ShardHeader {
        path: file.path.clone(),
        total_len: file.size,
        header_len: 8 + len as u64,
        tensors,
        metadata,
    })
}

/// Fetch a small text file whole (a config or index), or `None` when it isn't there.
fn fetch_text(agent: &ureq::Agent, repo: &RepoRef, path: &str, cap: u64) -> Option<String> {
    let url = format!(
        "{}/{}/resolve/{}/{}",
        endpoint(),
        repo.id,
        repo.revision,
        path
    );
    let mut resp = get(agent, &url).call().ok()?;
    resp.body_mut()
        .with_config()
        .limit(cap)
        .read_to_string()
        .ok()
}

/// Read a repo's structure into a [`Checkpoint`] — the listing, then every safetensors
/// header, then `config.json` and the shard index when present.
///
/// `progress` is updated as headers land, so a caller can draw a determinate bar; the reads
/// run `PARALLEL_READS` at a time.
pub fn read_checkpoint(repo: &RepoRef, progress: &ReadProgress) -> Result<Checkpoint> {
    let files = fetch_listing(repo)?;
    if files.is_empty() {
        bail!(
            "{} has no files at revision {} (private or gated? set HF_TOKEN)",
            repo.id,
            repo.revision
        );
    }
    let shards = safetensors_shards(&files);
    if shards.is_empty() {
        bail!(
            "{} has no .safetensors files at revision {}",
            repo.id,
            repo.revision
        );
    }

    let agent = agent();
    // The denominator is known now, before any header read — which is what makes this a
    // real bar rather than a spinner.
    progress
        .total
        .store(shards.len(), std::sync::atomic::Ordering::Relaxed);
    // A bounded pool: `PARALLEL_READS` in flight, which is what keeps a 96-shard repo to a
    // few seconds without hammering the Hub.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(PARALLEL_READS)
        .build()
        .context("building the header-read pool")?;
    // Per shard: its header, or why it couldn't be fetched. A torn shard doesn't sink the
    // repo — the rest are worth showing and the bad one is named, exactly as for a local
    // directory (`readers::read_local`) and an `--ssh-proxy` one (`remote::read`).
    let outcomes: Vec<std::result::Result<ShardHeader, crate::model::UnreadableShard>> = pool
        .install(|| {
            shards
                .par_iter()
                .map(|f| {
                    let h = fetch_header(&agent, repo, f).map_err(|e| {
                        crate::model::UnreadableShard {
                            // The `hf://…` form, so what's reported names the shard the way
                            // the rest of the UI does.
                            path: format!("{}/{}", repo.spec(), f.path),
                            error: format!("{e:#}"),
                        }
                    });
                    progress
                        .done
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    h
                })
                .collect()
        });
    let (mut headers, mut unreadable): (Vec<_>, Vec<_>) = (Vec::new(), Vec::new());
    for outcome in outcomes {
        match outcome {
            Ok(h) => headers.push(h),
            Err(bad) => unreadable.push(bad),
        }
    }
    // Listing order, so the tree and the layout map read in shard order rather than in
    // whatever order the parallel reads happened to finish — and so two reads of the same
    // broken repo report the same failures.
    headers.sort_by(|a, b| a.path.cmp(&b.path));
    unreadable.sort_by(|a, b| a.path.cmp(&b.path));
    // Nothing fetched: the reason is the answer, not an empty repo.
    if headers.is_empty()
        && let Some(first) = unreadable.first()
    {
        anyhow::bail!(
            "{}{}",
            first.error,
            if unreadable.len() > 1 {
                format!(" (and {} more shard(s) unreadable)", unreadable.len() - 1)
            } else {
                String::new()
            }
        );
    }

    let config = fetch_text(&agent, repo, "config.json", 4 << 20)
        .and_then(|t| crate::config::ModelConfig::parse(&t))
        .filter(crate::config::ModelConfig::is_meaningful);
    let index = fetch_text(&agent, repo, "model.safetensors.index.json", 64 << 20)
        .and_then(|t| crate::model::IndexEntry::parse("model.safetensors.index.json", &t))
        .map(|e| vec![e])
        .unwrap_or_default();

    Ok(Checkpoint {
        source: Source::Hf {
            repo: repo.id.clone(),
            revision: repo.revision.clone(),
        },
        root: repo.spec(),
        files: hub_file_entries(&files),
        shards: headers,
        config,
        index,
        s3: None,
        unreadable,
    })
}

/// The repo listing as [`FileEntry`]s for the file browser. Flat, with `rel_path`
/// carrying the `/`s — the file-tree builder folds those into directories, exactly as it
/// does for a remote S3 listing.
fn hub_file_entries(files: &[HubFile]) -> Vec<FileEntry> {
    files
        .iter()
        .map(|f| FileEntry {
            rel_path: f.path.clone(),
            name: f.path.rsplit('/').next().unwrap_or(&f.path).to_string(),
            depth: 0,
            mode: None,
            mtime: None,
            inode: None,
            node: FsNode::File {
                apparent: f.size,
                // The Hub reports one size; claiming a different on-disk figure would
                // invent a compression saving that nothing measured.
                allocated: f.size,
                kind: crate::filetree::FileKind::of(&f.path),
                links: 1,
            },
        })
        .collect()
}

/// Tensors and metadata across every shard, in shard order — the flat lists a caller
/// hands to a [`crate::kernel::Session`] when it isn't going through [`Checkpoint`].
#[must_use]
pub fn flatten(shards: &[ShardHeader]) -> (Vec<TensorInfo>, Vec<MetadataInfo>) {
    let mut tensors = Vec::new();
    let mut metadata: Vec<MetadataInfo> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for s in shards {
        tensors.extend(s.tensors.iter().cloned());
        for m in &s.metadata {
            // The same `format`/`total_size` entry appears in every shard; keep one.
            if seen.insert(m.name.clone()) {
                metadata.push(m.clone());
            }
        }
    }
    (tensors, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_only_explicit_references() {
        assert!(is_uri("hf://meta-llama/Llama-3-8B"));
        assert!(is_uri("https://huggingface.co/moonshotai/Kimi-K3"));
        // A bare `owner/name` is a relative path as far as we are concerned — reaching
        // the network for it would be the wrong surprise.
        assert!(!is_uri("moonshotai/Kimi-K3"));
        assert!(!is_uri("/models/local"));
        assert!(!is_uri("s3://bucket/key"));
    }

    #[test]
    fn parses_every_spelling_a_user_might_paste() {
        let main = RepoRef {
            id: "moonshotai/Kimi-K3".to_string(),
            revision: "main".to_string(),
        };
        assert_eq!(parse("hf://moonshotai/Kimi-K3").unwrap(), main);
        assert_eq!(
            parse("https://huggingface.co/moonshotai/Kimi-K3").unwrap(),
            main
        );
        assert_eq!(
            parse("https://huggingface.co/moonshotai/Kimi-K3/").unwrap(),
            main
        );
        // A URL copied from the browser, fragment and all — this is the case that made
        // fragment-stripping necessary.
        assert_eq!(
            parse("https://huggingface.co/moonshotai/Kimi-K3#2-model-summary").unwrap(),
            main
        );
        // Revisions, both spellings.
        assert_eq!(parse("hf://owner/name@abc123").unwrap().revision, "abc123");
        assert_eq!(
            parse("https://huggingface.co/owner/name/tree/refs%2Fpr%2F3")
                .unwrap()
                .revision,
            "refs%2Fpr%2F3"
        );
        assert_eq!(
            parse("https://huggingface.co/owner/name/blob/v2/config.json")
                .unwrap()
                .revision,
            "v2"
        );
    }

    #[test]
    fn rejects_what_it_cannot_read() {
        for bad in [
            "hf://owner",              // no name
            "hf://",                   // nothing
            "https://example.com/a/b", // not the Hub
            "hf://owner/name@",        // empty revision
        ] {
            assert!(parse(bad).is_err(), "{bad} should not parse");
        }
    }

    #[test]
    fn the_spec_round_trips() {
        for spec in ["hf://owner/name", "hf://owner/name@v2"] {
            assert_eq!(parse(spec).unwrap().spec(), spec);
        }
        // A URL normalises to the `hf://` form.
        assert_eq!(
            parse("https://huggingface.co/owner/name").unwrap().spec(),
            "hf://owner/name"
        );
    }

    /// The listing's LFS trap: a shard's real size is under `lfs.size`, while the
    /// top-level `size` is the pointer file's. Taking the wrong one reports a
    /// multi-terabyte checkpoint as a few kilobytes.
    #[test]
    fn the_listing_prefers_the_lfs_object_size() {
        let json = serde_json::json!([
            { "type": "directory", "path": "assets" },
            { "type": "file", "path": "config.json", "size": 1234 },
            { "type": "file", "path": "model-00001.safetensors", "size": 135,
              "lfs": { "size": 2_341_216_112u64 } },
        ]);
        let files = parse_listing(&json).unwrap();
        assert_eq!(files.len(), 2, "directories are not files: {files:?}");
        assert_eq!(files[0].path, "config.json");
        assert_eq!(files[0].size, 1234, "a plain file uses its own size");
        assert_eq!(
            files[1].size, 2_341_216_112,
            "an LFS file uses the object size, not the 135-byte pointer"
        );
    }

    #[test]
    fn shards_are_the_safetensors_files_only() {
        let files = vec![
            HubFile {
                path: "config.json".into(),
                size: 1,
            },
            HubFile {
                path: "model-1.safetensors".into(),
                size: 2,
            },
            HubFile {
                path: "sub/model-2.SAFETENSORS".into(),
                size: 3,
            },
            HubFile {
                path: "tokenizer.model".into(),
                size: 4,
            },
        ];
        let shards = safetensors_shards(&files);
        assert_eq!(shards.len(), 2, "case-insensitive suffix: {shards:?}");
        assert!(
            shards
                .iter()
                .all(|s| s.path.to_lowercase().ends_with(".safetensors"))
        );
    }

    #[test]
    fn file_entries_carry_the_repo_relative_path_and_size() {
        let files = vec![HubFile {
            path: "sub/model.safetensors".into(),
            size: 99,
        }];
        let entries = hub_file_entries(&files);
        assert_eq!(entries[0].rel_path, "sub/model.safetensors");
        assert_eq!(
            entries[0].name, "model.safetensors",
            "the display name is the basename"
        );
        assert_eq!(entries[0].apparent(), 99);
        assert_eq!(
            entries[0].allocated(),
            99,
            "the Hub reports one size; inventing a different on-disk figure would be a lie"
        );
    }

    /// Shard metadata repeats the same `format` / `total_size` entry in every file; the
    /// flattened list keeps one of each rather than 96.
    /// A Hub tensor must look remote to everything that gates on the source path, or a data
    /// view will try to open it as a local file. Pinned because the failure was a raw
    /// `No such file or directory` rather than the capability's explanation.
    #[test]
    fn a_hub_tensor_source_path_reads_as_remote() {
        let repo = RepoRef {
            id: "owner/name".to_string(),
            revision: "main".to_string(),
        };
        let source = format!("{}/{}", repo.spec(), "model-00001-of-00002.safetensors");
        assert!(
            crate::remote::is_remote_source(&source),
            "{source} must be recognised as remote"
        );
        // The bare shard path — what this used to be — does not.
        assert!(!crate::remote::is_remote_source(
            "model-00001-of-00002.safetensors"
        ));
    }

    #[test]
    fn flattening_deduplicates_repeated_shard_metadata() {
        let meta = |k: &str, v: &str| MetadataInfo {
            name: k.to_string(),
            value: v.to_string(),
            value_type: "string".to_string(),
        };
        let shard = |path: &str| ShardHeader {
            path: path.to_string(),
            total_len: 0,
            header_len: 0,
            tensors: Vec::new(),
            metadata: vec![meta("format", "pt")],
        };
        let (tensors, metadata) = flatten(&[shard("a"), shard("b"), shard("c")]);
        assert!(tensors.is_empty());
        assert_eq!(metadata.len(), 1, "one `format`, not three: {metadata:?}");
    }
}
