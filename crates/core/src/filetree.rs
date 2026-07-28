//! The file-browser tree: a checkpoint's directory shown as a hierarchy of
//! directories and files (the `Tab` file view). Kept separate from the tensor
//! [`crate::tree::TreeNode`] so the mature tensor paths stay untouched — this
//! models only what a file explorer needs (name, path, size, kind, expansion).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::tree::{TensorInfo, natural_sort_key};

/// How much of a sidecar either frontend previews.
///
/// Shared because it decides whether a file *parses*: the web served 1 MiB while the TUI
/// read 4 MiB, so a real `model.safetensors.index.json` (1.7 MB for a 30B checkpoint)
/// arrived at the browser cut mid-string — unhighlightable, and reported as "truncated"
/// on a file the TUI showed whole. One number, one behaviour.
pub const PREVIEW_CAP: u64 = 4 << 20;

/// What a file is, for its glyph and what `Enter` does with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FileKind {
    /// A checkpoint we can open in the tensor view.
    Checkpoint,
    /// JSON — previewed with syntax highlighting.
    Json,
    /// Other UTF-8 text (README, LICENSE, .txt, .py, …) — previewed plain.
    Text,
    /// Anything else (binary, unknown) — info only.
    Other,
}

impl FileKind {
    /// Classify by extension / name. `Text` is a best-effort guess refined when
    /// the file is actually read (a non-UTF-8 "text" file falls back to info).
    #[must_use]
    pub fn of(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        let ext = lower.rsplit('.').next().unwrap_or("");
        match ext {
            "safetensors" | "gguf" | "npy" | "npz" | "h5" | "hdf5" => Self::Checkpoint,
            "json" => Self::Json,
            "txt" | "md" | "py" | "yaml" | "yml" | "toml" | "cfg" | "ini" | "csv" | "tsv"
            | "jsonl" | "text" | "log" | "sh" | "rs" => Self::Text,
            // Extensionless docs that are conventionally text.
            _ if matches!(
                lower.as_str(),
                "readme" | "license" | "licence" | "notice" | "authors" | "copying" | "changelog"
            ) =>
            {
                Self::Text
            }
            _ => Self::Other,
        }
    }
}

/// What a checkpoint file contributes to the model: the tensors read out of it and
/// their share of the parameters.
///
/// Without this a sharded checkpoint browses as sixteen identical-looking rows —
/// `model-000NN-of-00016.safetensors  3.7 GiB` — while the shards are not alike at
/// all: one carries the embedding, one the odd tail of layers, one nothing but
/// codebooks. Every shard's header is already parsed when the checkpoint is opened,
/// so this is a projection of what we know, not extra I/O.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ShardTensors {
    /// Tensors the model read from this file.
    pub tensors: usize,
    /// Parameters (elements) those tensors hold.
    pub params: usize,
    /// Fraction of the whole checkpoint's parameters, `0.0..=1.0`. Carried rather
    /// than computed per frontend because a row knows its own numbers but not the
    /// checkpoint's total — and because two independent divisions are two chances
    /// for the terminal and the browser to disagree.
    pub params_share: f64,
}

impl ShardTensors {
    /// The file-browser row's suffix: `1062 tensors · 6.4% of params`.
    ///
    /// The share is what finds the odd file out — in a uniformly sharded checkpoint it
    /// tracks the size, but a codebook or embedding-only shard is large on disk and
    /// small in parameters (or the reverse), and the two disagreeing is the point.
    /// Rendered by the shared [`crate::utils::format_percent`], so a share too small
    /// for one decimal reads as scientific notation rather than a misleading `0.0%`.
    ///
    /// In core rather than in either frontend because both write this row, so the
    /// wording is contracted in `shared/parity/format.json` — see that harness.
    #[must_use]
    pub fn note(&self) -> String {
        format!(
            "{} {} · {} of params",
            self.tensors,
            if self.tensors == 1 {
                "tensor"
            } else {
                "tensors"
            },
            crate::utils::format_percent(self.params_share, self.params == 0),
        )
    }
}

/// Whether a checkpoint file is one the index declares.
///
/// `Option::None` at the use sites means the question doesn't apply — the file isn't a
/// checkpoint, or the checkpoint has no `model.safetensors.index.json` to be in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexMembership {
    /// Named in an index's weight map — the ordinary case, and unmarked.
    Listed,
    /// A checkpoint file on disk that no index names.
    ///
    /// Not an error, and often expected: a LUT-quantised checkpoint ships codebooks
    /// and scales the index never mentions. But a loader that follows only the index
    /// will not read it, which is exactly what you want to know while looking at the
    /// file — the health check already says so, one screen away.
    Unlisted,
}

/// A node in the file-browser tree.
#[derive(Debug, Clone)]
pub enum FileNode {
    Dir {
        name: String,
        path: PathBuf,
        children: Vec<Self>,
        expanded: bool,
        /// Aggregate size (bytes) and file count of everything under here.
        size: u64,
        files: usize,
        /// How many of those files are hardlinked (their bytes have another name).
        ///
        /// The per-row marks say which files share their bytes; this says how much of
        /// the directory does, which is the question you actually have when a listing
        /// adds up to 57 GiB. Deliberately a **count, not a byte total**: hardlinked
        /// bytes are shared, not free — someone pays for them once — and a "55 GiB
        /// shared" figure would read as "this checkpoint is nearly free", which is only
        /// true if you already have the other copy.
        hardlinked: usize,
    },
    File {
        name: String,
        path: PathBuf,
        size: u64,
        kind: FileKind,
        /// The tensors this file contributes, once [`FileNode::attribute_tensors`]
        /// has run. `None` for a file the model reads nothing from (a README, a
        /// tokenizer, a shard belonging to some *other* checkpoint in the same
        /// directory) and for a tree nobody attributed.
        shard: Option<ShardTensors>,
        /// This file's size as a fraction of the **largest file in the tree**, for the
        /// proportional bar beside the size — so a listing shows at a glance which
        /// files carry the weight, without reading twenty numbers.
        ///
        /// Relative to the biggest file rather than to the directory total: shares of
        /// a 57 GiB checkpoint are all a couple of percent, which draws twenty
        /// identical slivers. Relative to the biggest, the shards fill the bar and the
        /// sidecars are visibly nothing — which is the true shape of a checkpoint.
        /// Computed by [`build_from`], so it is the same number in both frontends and
        /// doesn't drift as the tree is folded.
        size_share: f64,
        /// Whether the index declares this file, once [`FileNode::attribute_index`] has
        /// run. `None` when the question doesn't apply — see [`IndexMembership`].
        index: Option<IndexMembership>,
        /// How many names this file's bytes have — see [`DirEntry::File::links`]. `>1`
        /// means the file is hardlinked, so its size is shared rather than its own:
        /// deleting this name frees nothing, and the checkpoint's real footprint is
        /// smaller than the sum of its rows.
        links: u64,
    },
}

impl FileNode {
    #[must_use]
    pub fn size(&self) -> u64 {
        match self {
            Self::Dir { size, .. } | Self::File { size, .. } => *size,
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Dir { name, .. } | Self::File { name, .. } => name,
        }
    }

    /// Annotate every file the model read tensors from with its [`ShardTensors`].
    ///
    /// A pass over an already-built tree rather than an argument to [`build`]: the
    /// tree has four producers (a local walk, an SFTP listing, S3 keys, the cached
    /// model walk) and threading a tensor list through all of them — including the
    /// ones that browse a directory holding no checkpoint at all — would put the
    /// same `Option` in four signatures instead of one field.
    ///
    /// Tensors are attributed by their `source_path`. When that doesn't match a
    /// node's path, an unambiguous file *name* is the fallback: the tree and the
    /// tensor list come from different producers, so they can disagree about the
    /// path while agreeing about the file (a Hub snapshot's symlink into the blob
    /// store, or a remote listing rooted differently from the proxy's own paths).
    pub fn attribute_tensors(&mut self, tensors: &[TensorInfo]) {
        let mut by_path: HashMap<&str, (usize, usize)> = HashMap::new();
        let mut total = 0usize;
        for t in tensors {
            let e = by_path.entry(t.source_path.as_str()).or_default();
            e.0 += 1;
            e.1 += t.num_elements;
            total += t.num_elements;
        }
        // `None` marks a name more than one source file shares, so the fallback
        // stays off for it rather than picking one of them.
        let mut by_name: HashMap<&str, Option<&str>> = HashMap::new();
        for key in by_path.keys() {
            let name = key.rsplit(['/', '\\']).next().unwrap_or(key);
            by_name
                .entry(name)
                .and_modify(|slot| *slot = None)
                .or_insert(Some(key));
        }
        self.annotate(&by_path, &by_name, total);
    }

    fn annotate(
        &mut self,
        by_path: &HashMap<&str, (usize, usize)>,
        by_name: &HashMap<&str, Option<&str>>,
        total: usize,
    ) {
        match self {
            Self::Dir { children, .. } => {
                for child in children {
                    child.annotate(by_path, by_name, total);
                }
            }
            Self::File {
                name, path, shard, ..
            } => {
                let hit = by_path.get(path.to_string_lossy().as_ref()).or_else(|| {
                    by_name
                        .get(name.as_str())
                        .and_then(|slot| slot.as_ref())
                        .and_then(|key| by_path.get(*key))
                });
                *shard = hit.map(|&(tensors, params)| ShardTensors {
                    tensors,
                    params,
                    params_share: share(params, total),
                });
            }
        }
    }
}

impl FileNode {
    /// Mark each checkpoint file [`IndexMembership::Listed`] or
    /// [`IndexMembership::Unlisted`] against `indexes`; everything else keeps `None`.
    /// With no index there is nothing to be in, so every file keeps `None` — a
    /// single-file checkpoint should not report itself as an extra.
    ///
    /// Matched on the **file name**, deliberately. A weight map's values are bare
    /// basenames (`"model-00001-of-00016.safetensors"`) because an index governs its
    /// own directory; that is the index format's own notion of identity, so matching
    /// anything more precise would be inventing one. It also means this works for a
    /// remote or `s3://` listing, whose node paths aren't in the same form as the
    /// index's own path.
    pub fn attribute_index(&mut self, indexes: &[crate::model::IndexEntry]) {
        if indexes.is_empty() {
            return;
        }
        let listed: std::collections::HashSet<&str> = indexes
            .iter()
            .flat_map(|i| i.weight_map.values())
            .map(String::as_str)
            .collect();
        self.mark_index(&listed);
    }

    fn mark_index(&mut self, listed: &std::collections::HashSet<&str>) {
        match self {
            Self::Dir { children, .. } => {
                for child in children {
                    child.mark_index(listed);
                }
            }
            Self::File {
                name, kind, index, ..
            } => {
                *index = (*kind == FileKind::Checkpoint).then(|| {
                    if listed.contains(name.as_str()) {
                        IndexMembership::Listed
                    } else {
                        IndexMembership::Unlisted
                    }
                });
            }
        }
    }
}

/// `params / total` as a fraction, `0.0` for an empty checkpoint.
#[allow(clippy::cast_precision_loss)] // a display ratio; f64 covers any real param count
fn share(params: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        params as f64 / total as f64
    }
}

/// One entry in a directory listing from any backend (local `std::fs`, remote
/// SFTP, or synthetic S3 keys) — the **shared** interchange type every backend
/// produces and [`build_from`] consumes, so there's one filesystem-entry sum type
/// rather than a `(name, size, is_dir)` bag per backend. A `File` carries its
/// **readable content size** (symlinks are *followed*, so a linked shard reports
/// its target's size — what opening it yields); a symlink *to a directory* is
/// represented as a `File` leaf (size 0) so the recursive walk can't cycle. Only a
/// real, descendable directory is `Directory`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirEntry {
    File {
        name: String,
        size: u64,
        /// `st_nlink` of the file's inode: `1` for an ordinary file, `>1` when the
        /// same bytes are reachable under more than one name. `1` when unknown, which
        /// is what every remote backend reports — `st_nlink` needs a local `stat`, and
        /// an S3 object has no inode to count names of. Same convention (and the same
        /// meaning) as [`crate::model::FsNode::File::links`].
        links: u64,
    },
    Directory {
        name: String,
    },
}

impl DirEntry {
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::File { name, .. } | Self::Directory { name } => name,
        }
    }
}

/// One entry in a **flat** listing for [`build_from_keys`] — an S3 object key or a
/// remote file's checkpoint-relative path, with what is known about it.
///
/// Named rather than a `(String, u64, u64)` triple for the same reason [`DirEntry`]
/// exists: two adjacent byte counts that could be stored swapped, and a caller with no
/// way to see it had.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectEntry {
    /// Prefix-relative key, `/`-separated (`layer_0/weight`). Becomes the node's path
    /// verbatim, so a browser can rebuild the full URI from it.
    pub key: String,
    pub size: u64,
    /// Names the bytes have — see [`DirEntry::File::links`]. Use [`Self::object`] for a
    /// listing that cannot know (a real S3 object has no inode to share).
    pub links: u64,
}

impl ObjectEntry {
    /// An entry from a listing with no notion of hard links — an S3 object.
    #[must_use]
    pub fn object(key: String, size: u64) -> Self {
        Self {
            key,
            size,
            links: 1,
        }
    }
}

/// Build the file tree rooted at `root` from the **local filesystem**, recursively
/// (bounded by `max_depth`). Directories sort first, then files, each in natural
/// order; hidden entries (dotfiles) are skipped. The returned node is the root
/// directory itself, expanded. Symlinks are followed for a file's size (so a linked
/// shard shows its target's size) but a symlinked directory is a leaf, not
/// descended, to avoid cycles — see [`DirEntry`].
#[must_use]
pub fn build(root: &Path, max_depth: usize) -> FileNode {
    build_from(&local_list, root, max_depth)
}

/// Build the file tree from a pluggable directory-listing backend `list` — so the
/// same tree (and everything downstream: [`flatten`], [`toggle_by_index`],
/// `FileRow`, the renderer) works for a local walk or a remote SFTP `readdir`.
/// `list(dir)` returns that directory's entries (already dotfile-filtered); the
/// child path is `dir.join(name)`, so remote paths compose the same way.
pub fn build_from(
    list: &dyn Fn(&Path) -> Vec<DirEntry>,
    root: &Path,
    max_depth: usize,
) -> FileNode {
    let mut node = build_dir(list, root, root_name(root), max_depth);
    // The root is expanded so its contents show immediately; nested dirs start
    // collapsed (a checkpoint is usually flat, so this rarely matters).
    if let FileNode::Dir { expanded, .. } = &mut node {
        *expanded = true;
    }
    // Size shares are derivable from the tree alone, so they're part of building it —
    // not a pass every one of the four producers has to remember to run.
    set_size_shares(&mut node);
    node
}

/// Fill in every file's [`FileNode::File::size_share`], relative to the largest file.
fn set_size_shares(node: &mut FileNode) {
    let max = largest_file(node);
    apply_size_shares(node, max);
}

fn largest_file(node: &FileNode) -> u64 {
    match node {
        FileNode::Dir { children, .. } => children.iter().map(largest_file).max().unwrap_or(0),
        FileNode::File { size, .. } => *size,
    }
}

#[allow(clippy::cast_precision_loss)] // a display ratio over file sizes
fn apply_size_shares(node: &mut FileNode, max: u64) {
    match node {
        FileNode::Dir { children, .. } => {
            for child in children {
                apply_size_shares(child, max);
            }
        }
        FileNode::File {
            size, size_share, ..
        } => {
            *size_share = if max == 0 {
                0.0
            } else {
                *size as f64 / max as f64
            }
        }
    }
}

/// Build an **s3-native** browse tree from a flat object listing (prefix-relative
/// `(key, size)` pairs): each key is split on `/`, shared prefixes become
/// expandable directories and the leaves are the objects; sizes and file counts
/// aggregate bottom-up, exactly like the local/SFTP walk (it reuses
/// [`build_from`] over a synthetic in-memory listing). `root_label` is the root
/// node's display name; every other node's `path` is its **exact prefix-relative
/// key** (`a/b/c` for the object `a/b/c`), so the browser rebuilds the full
/// `s3://…` URI as `{uri}/{path}`. Browse-only — no per-object layout or preview.
#[must_use]
pub fn build_from_keys(root_label: &str, objects: &[ObjectEntry]) -> FileNode {
    use std::collections::{HashMap, HashSet};
    // Directory (a relative path; "" is the root) → its immediate entries.
    // Intermediate dirs are materialized once (`dir_seen`); files carry their
    // size, dirs 0 (aggregated by `build_from`). Rooting at "" makes each node's
    // composed path (`dir.join(name)`) equal its exact prefix-relative key.
    let mut listing: HashMap<PathBuf, Vec<DirEntry>> = HashMap::new();
    let mut dir_seen: HashSet<PathBuf> = HashSet::new();
    for ObjectEntry { key, size, links } in objects {
        let comps: Vec<&str> = key.split('/').filter(|s| !s.is_empty()).collect();
        let Some((leaf, dirs)) = comps.split_last() else {
            continue;
        };
        let mut cur = PathBuf::new();
        for comp in dirs {
            let child = cur.join(comp);
            if dir_seen.insert(child.clone()) {
                listing
                    .entry(cur.clone())
                    .or_default()
                    .push(DirEntry::Directory {
                        name: (*comp).to_string(),
                    });
            }
            cur = child;
        }
        listing.entry(cur).or_default().push(DirEntry::File {
            name: (*leaf).to_string(),
            size: *size,
            links: *links,
        });
    }
    let list = move |p: &Path| listing.get(p).cloned().unwrap_or_default();
    // Depth generous enough for any realistic object-key nesting.
    let mut node = build_from(&list, Path::new(""), 64);
    if let FileNode::Dir { name, .. } = &mut node {
        *name = root_label.to_string();
    }
    node
}

/// The local-filesystem listing backend for [`build`].
fn local_list(dir: &Path) -> Vec<DirEntry> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = match entry.file_name().to_str() {
                Some(n) if !n.starts_with('.') => n.to_string(),
                _ => continue, // unreadable name or a dotfile
            };
            // Follow symlinks (`fs::metadata`, not the link's own `entry.metadata`)
            // so a symlinked shard shows the *target's* real size — what opening it
            // reads — not the ~link-path length. A *real* directory descends; a
            // symlinked directory stays a leaf so the walk can't cycle. A broken
            // link falls back to its own (lstat) metadata.
            let is_symlink = entry.file_type().ok().is_some_and(|t| t.is_symlink());
            let meta = std::fs::metadata(entry.path()).or_else(|_| entry.metadata());
            out.push(match meta {
                // A real directory descends; a symlinked directory is a File leaf.
                Ok(m) if m.is_dir() && !is_symlink => DirEntry::Directory { name },
                Ok(m) if m.is_dir() => DirEntry::File {
                    name,
                    size: 0,
                    links: 1,
                },
                Ok(m) => DirEntry::File {
                    name,
                    size: m.len(),
                    links: link_count(&m),
                },
                Err(_) => DirEntry::File {
                    name,
                    size: 0,
                    links: 1,
                },
            });
        }
    }
    out
}

/// The inode's hard-link count, or `1` where the platform has no such notion.
///
/// Free here: [`local_list`] already `stat`s every entry to follow symlinks, so this
/// reads a field it has rather than making a syscall of its own.
#[cfg(unix)]
fn link_count(meta: &std::fs::Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::nlink(meta)
}

#[cfg(not(unix))]
fn link_count(_meta: &std::fs::Metadata) -> u64 {
    1
}

/// The label for the root directory node — its final component, or the whole
/// path when it has none (e.g. `/`).
fn root_name(root: &Path) -> String {
    root.file_name().map_or_else(
        || root.to_string_lossy().into_owned(),
        |s| s.to_string_lossy().into_owned(),
    )
}

fn build_dir(
    list: &dyn Fn(&Path) -> Vec<DirEntry>,
    dir: &Path,
    name: String,
    depth_left: usize,
) -> FileNode {
    let mut dirs: Vec<FileNode> = Vec::new();
    let mut files: Vec<FileNode> = Vec::new();
    for entry in list(dir) {
        let path = dir.join(entry.name());
        match entry {
            DirEntry::Directory { name } if depth_left > 0 => {
                dirs.push(build_dir(list, &path, name, depth_left - 1));
            }
            DirEntry::Directory { name } => {
                // Depth limit reached: represent as an empty (unexpanded) dir.
                dirs.push(FileNode::Dir {
                    name,
                    path,
                    children: Vec::new(),
                    expanded: false,
                    size: 0,
                    files: 0,
                    hardlinked: 0,
                });
            }
            DirEntry::File { name, size, links } => {
                let kind = FileKind::of(&name);
                files.push(FileNode::File {
                    name,
                    path,
                    size,
                    kind,
                    // Filled in by `attribute_tensors`, which needs the tensor list.
                    shard: None,
                    // Filled in by `build_from`, which can see the whole tree.
                    size_share: 0.0,
                    // Filled in by `attribute_index`, which needs the index.
                    index: None,
                    links,
                });
            }
        }
    }
    let by_name =
        |a: &FileNode, b: &FileNode| natural_sort_key(a.name()).cmp(&natural_sort_key(b.name()));
    dirs.sort_by(by_name);
    files.sort_by(by_name);
    let mut children = dirs;
    children.extend(files);

    let size = children.iter().map(FileNode::size).sum();
    let file_count = children
        .iter()
        .map(|c| match c {
            FileNode::Dir { files, .. } => *files,
            FileNode::File { .. } => 1,
        })
        .sum();
    let hardlinked = children
        .iter()
        .map(|c| match c {
            FileNode::Dir { hardlinked, .. } => *hardlinked,
            FileNode::File { links, .. } => usize::from(*links > 1),
        })
        .sum();

    FileNode::Dir {
        name,
        path: dir.to_path_buf(),
        children,
        expanded: false, // the root is expanded by `build`; nested dirs collapsed
        size,
        files: file_count,
        hardlinked,
    }
}

/// One visible row of the flattened file tree — the data a row needs to render
/// and to act on (`Enter`), so the browser never re-walks the tree per frame.
/// The dir-vs-file split is a tagged [`FileRowKind`] (no `is_dir` bool + dummy
/// `expanded`/`files`/`kind` fields that only some rows use).
#[derive(Debug, Clone)]
pub struct FileRow {
    pub depth: usize,
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    pub kind: FileRowKind,
}

/// A flattened row is either a directory (with its fold state + child count) or a
/// file (with its content kind and, for a shard, what it contributes) — the two
/// carry disjoint data.
// No `Eq`: `ShardTensors` holds a display ratio, and a float has no total equality.
#[derive(Debug, Clone, PartialEq)]
pub enum FileRowKind {
    Dir {
        expanded: bool,
        files: usize,
        /// Hardlinked files under here — see [`FileNode::Dir::hardlinked`].
        hardlinked: usize,
    },
    File {
        kind: FileKind,
        shard: Option<ShardTensors>,
        /// Fraction of the largest file's size — see [`FileNode::File::size_share`].
        size_share: f64,
        /// Whether the index declares this file — see [`IndexMembership`].
        index: Option<IndexMembership>,
        /// How many names this file's bytes have — see [`DirEntry::File::links`].
        links: u64,
    },
}

impl FileRow {
    /// Whether this row is a directory.
    #[must_use]
    pub fn is_dir(&self) -> bool {
        matches!(self.kind, FileRowKind::Dir { .. })
    }
    /// Whether this (directory) row is expanded — `false` for a file row.
    #[must_use]
    pub fn expanded(&self) -> bool {
        matches!(self.kind, FileRowKind::Dir { expanded: true, .. })
    }
    /// Child-file count for a directory row (0 for a file row).
    #[must_use]
    pub fn files(&self) -> usize {
        match self.kind {
            FileRowKind::Dir { files, .. } => files,
            FileRowKind::File { .. } => 0,
        }
    }
    /// The content classification for a file row, else `None` for a directory.
    #[must_use]
    pub fn file_kind(&self) -> Option<FileKind> {
        match self.kind {
            FileRowKind::File { kind, .. } => Some(kind),
            FileRowKind::Dir { .. } => None,
        }
    }

    /// This file's share of the largest file's size, or `None` for a directory —
    /// whose aggregate is its children's total, not a size to compare against them.
    #[must_use]
    pub fn size_share(&self) -> Option<f64> {
        match self.kind {
            FileRowKind::File { size_share, .. } => Some(size_share),
            FileRowKind::Dir { .. } => None,
        }
    }

    /// Whether the index declares this file — `None` when the question doesn't apply
    /// (a directory, a sidecar, or a checkpoint with no index). See
    /// [`IndexMembership`].
    #[must_use]
    pub fn index_membership(&self) -> Option<IndexMembership> {
        match self.kind {
            FileRowKind::File { index, .. } => index,
            FileRowKind::Dir { .. } => None,
        }
    }

    /// How many names this file's bytes have (`1` for an ordinary file or a directory)
    /// — see [`DirEntry::File::links`].
    #[must_use]
    pub fn links(&self) -> u64 {
        match self.kind {
            FileRowKind::File { links, .. } => links,
            FileRowKind::Dir { .. } => 1,
        }
    }

    /// Hardlinked files under this directory row (0 for a file row) — see
    /// [`FileNode::Dir::hardlinked`].
    #[must_use]
    pub fn hardlinked(&self) -> usize {
        match self.kind {
            FileRowKind::Dir { hardlinked, .. } => hardlinked,
            FileRowKind::File { .. } => 0,
        }
    }
}

/// Flatten the tree into the visible rows (a collapsed directory hides its
/// subtree), root first, mirroring the tensor tree's flattening.
#[must_use]
pub fn flatten(root: &FileNode) -> Vec<FileRow> {
    let mut out = Vec::new();
    flatten_node(root, 0, &mut out);
    out
}

fn flatten_node(node: &FileNode, depth: usize, out: &mut Vec<FileRow>) {
    match node {
        FileNode::Dir {
            name,
            path,
            children,
            expanded,
            size,
            files,
            hardlinked,
        } => {
            out.push(FileRow {
                depth,
                name: name.clone(),
                path: path.clone(),
                size: *size,
                kind: FileRowKind::Dir {
                    expanded: *expanded,
                    files: *files,
                    hardlinked: *hardlinked,
                },
            });
            if *expanded {
                for child in children {
                    flatten_node(child, depth + 1, out);
                }
            }
        }
        FileNode::File {
            name,
            path,
            size,
            kind,
            shard,
            size_share,
            index,
            links,
        } => out.push(FileRow {
            depth,
            name: name.clone(),
            path: path.clone(),
            size: *size,
            kind: FileRowKind::File {
                kind: *kind,
                shard: *shard,
                size_share: *size_share,
                index: *index,
                links: *links,
            },
        }),
    }
}

/// Toggle the expanded state of the directory at flattened index `idx` (in the
/// same visit order as [`flatten`]). Returns whether a directory was toggled.
pub fn toggle_by_index(root: &mut FileNode, idx: usize) -> bool {
    let mut cur = 0usize;
    toggle_walk(root, idx, &mut cur)
}

fn toggle_walk(node: &mut FileNode, target: usize, cur: &mut usize) -> bool {
    let here = *cur;
    *cur += 1;
    if let FileNode::Dir {
        children, expanded, ..
    } = node
    {
        if here == target {
            *expanded = !*expanded;
            return true;
        }
        if *expanded {
            for child in children.iter_mut() {
                if toggle_walk(child, target, cur) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_classifies_by_extension_and_known_names() {
        assert_eq!(
            FileKind::of("model-00001-of-2.safetensors"),
            FileKind::Checkpoint
        );
        assert_eq!(FileKind::of("weights.gguf"), FileKind::Checkpoint);
        assert_eq!(FileKind::of("config.json"), FileKind::Json);
        assert_eq!(FileKind::of("tokenizer_config.json"), FileKind::Json);
        assert_eq!(FileKind::of("README"), FileKind::Text);
        assert_eq!(FileKind::of("LICENSE"), FileKind::Text);
        assert_eq!(FileKind::of("notes.md"), FileKind::Text);
        assert_eq!(FileKind::of("tool_parser.py"), FileKind::Text);
        assert_eq!(FileKind::of("mystery.bin"), FileKind::Other);
    }

    #[test]
    fn flatten_hides_collapsed_subtrees_and_toggle_reveals_them() {
        let dir = std::env::temp_dir().join("ce_filetree_flatten_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub/a.json"), b"{}").unwrap();
        std::fs::write(dir.join("top.json"), b"{}").unwrap();

        let mut root = build(&dir, 8);
        // `sub` starts collapsed, so its child is hidden; root + sub + top.json.
        let rows = flatten(&root);
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"sub") && names.contains(&"top.json"));
        assert!(
            !names.contains(&"a.json"),
            "collapsed subtree hidden: {names:?}"
        );

        // Toggle `sub` (index 1: root is 0) → its child appears.
        assert!(toggle_by_index(&mut root, 1));
        assert!(
            flatten(&root).iter().any(|r| r.name == "a.json"),
            "expanded subtree shows its child"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_sorts_dirs_first_and_sums_sizes() {
        let dir = std::env::temp_dir().join("ce_filetree_build_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap(); // 2 bytes
        std::fs::write(dir.join("model.safetensors"), vec![0u8; 100]).unwrap();
        std::fs::write(dir.join("sub/extra.json"), vec![0u8; 8]).unwrap();
        std::fs::write(dir.join(".hidden"), b"x").unwrap(); // skipped

        let root = build(&dir, 8);
        let FileNode::Dir {
            children,
            size,
            files,
            ..
        } = &root
        else {
            panic!("root is a dir");
        };
        // Directory ("sub") sorts before the files.
        assert!(matches!(&children[0], FileNode::Dir { name, .. } if name == "sub"));
        // Files after, natural-sorted, dotfile skipped.
        let names: Vec<&str> = children.iter().map(FileNode::name).collect();
        assert_eq!(names, ["sub", "config.json", "model.safetensors"]);
        // Aggregate size = 2 + 100 + 8 (sub) = 110; 3 files counted.
        assert_eq!(*size, 110);
        assert_eq!(*files, 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_from_walks_a_synthetic_listing() {
        // A remote-like listing backend keyed by directory path — no filesystem.
        let list = |dir: &Path| -> Vec<DirEntry> {
            match dir.to_str().unwrap() {
                "/ckpt" => vec![
                    DirEntry::File {
                        name: "model.safetensors".into(),
                        size: 100,
                        links: 1,
                    },
                    DirEntry::Directory { name: "sub".into() },
                    DirEntry::File {
                        name: "config.json".into(),
                        size: 2,
                        links: 1,
                    },
                ],
                "/ckpt/sub" => vec![
                    DirEntry::File {
                        name: "extra.json".into(),
                        size: 8,
                        links: 1,
                    },
                    DirEntry::Directory {
                        name: "deep".into(),
                    },
                ],
                "/ckpt/sub/deep" => vec![DirEntry::File {
                    name: "leaf.bin".into(),
                    size: 4,
                    links: 1,
                }],
                _ => Vec::new(),
            }
        };

        let root = build_from(&list, Path::new("/ckpt"), 8);
        let FileNode::Dir {
            children,
            size,
            files,
            expanded,
            ..
        } = &root
        else {
            panic!("root is a dir");
        };
        assert!(*expanded, "root is expanded");
        // Dirs first, then files natural-sorted; child paths compose from the parent.
        let names: Vec<&str> = children.iter().map(FileNode::name).collect();
        assert_eq!(names, ["sub", "config.json", "model.safetensors"]);
        assert!(matches!(&children[0], FileNode::Dir { path, expanded, .. }
            if path == Path::new("/ckpt/sub") && !*expanded));
        // Bottom-up aggregation: 100 + 2 + (8 + 4) = 114; 4 files across all depths.
        assert_eq!(*size, 114);
        assert_eq!(*files, 4);

        // Depth cap: at depth 1, `sub` is entered but `deep` is a stubbed empty dir.
        let shallow = build_from(&list, Path::new("/ckpt"), 1);
        let FileNode::Dir { children, .. } = &shallow else {
            panic!("root is a dir");
        };
        let FileNode::Dir {
            children: sub_children,
            ..
        } = &children[0]
        else {
            panic!("sub is a dir");
        };
        let deep = sub_children
            .iter()
            .find(|c| c.name() == "deep")
            .expect("deep dir present");
        assert!(matches!(deep, FileNode::Dir { children, .. } if children.is_empty()));
    }

    #[cfg(unix)]
    #[test]
    fn build_follows_symlinked_files_to_their_real_size() {
        // A blob-dedup checkpoint: the shard in the checkpoint dir is a *symlink* to
        // the real file living elsewhere. The browser must show the target's size —
        // so it agrees with the layout map, which opens (follows) the file. A
        // directory listing's own (lstat) size would be the ~link-path length.
        use std::os::unix::fs::symlink;
        let base = std::env::temp_dir().join("ce_filetree_symlink_test");
        let _ = std::fs::remove_dir_all(&base);
        let ckpt = base.join("ckpt");
        let store = base.join("store");
        std::fs::create_dir_all(&ckpt).unwrap();
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(store.join("blob"), vec![0u8; 4096]).unwrap();
        symlink(store.join("blob"), ckpt.join("model-00000.safetensors")).unwrap();
        // A symlinked *directory* must stay a leaf (no descent → no cycle).
        symlink(&store, ckpt.join("linkdir")).unwrap();

        let root = build(&ckpt, 8);
        let FileNode::Dir { children, .. } = &root else {
            panic!("root is a dir");
        };
        let shard = children
            .iter()
            .find(|c| c.name() == "model-00000.safetensors")
            .expect("shard present");
        assert!(
            matches!(shard, FileNode::File { size, .. } if *size == 4096),
            "symlinked shard shows the target's real size, not the link length"
        );
        let linkdir = children.iter().find(|c| c.name() == "linkdir").unwrap();
        assert!(
            matches!(linkdir, FileNode::File { .. }),
            "a symlinked directory is a leaf (not descended)"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// A tensor of `elements` parameters read from `source_path`.
    fn tensor(name: &str, source_path: &str, elements: usize) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: "F16".to_string(),
            shape: vec![elements],
            size_bytes: elements * 2,
            num_elements: elements,
            storage: crate::tree::Storage::Raw,
            source_path: source_path.to_string(),
            layout: crate::tree::Layout::None,
        }
    }

    /// A two-shard checkpoint plus a sidecar, as `build_from` would produce it.
    fn two_shard_tree() -> FileNode {
        let list = |dir: &Path| -> Vec<DirEntry> {
            if dir == Path::new("/ckpt") {
                vec![
                    DirEntry::File {
                        name: "model-00001.safetensors".into(),
                        size: 600,
                        links: 1,
                    },
                    DirEntry::File {
                        name: "model-00002.safetensors".into(),
                        size: 400,
                        links: 1,
                    },
                    DirEntry::File {
                        name: "config.json".into(),
                        size: 2,
                        links: 1,
                    },
                ]
            } else {
                Vec::new()
            }
        };
        build_from(&list, Path::new("/ckpt"), 8)
    }

    /// The annotated files of a tree, by name.
    fn shards_of(root: &FileNode) -> HashMap<String, Option<ShardTensors>> {
        flatten(root)
            .into_iter()
            .filter_map(|r| match r.kind {
                FileRowKind::File { shard, .. } => Some((r.name, shard)),
                FileRowKind::Dir { .. } => None,
            })
            .collect()
    }

    #[test]
    fn attribute_tensors_counts_per_shard_and_shares_the_params() {
        let mut root = two_shard_tree();
        root.attribute_tensors(&[
            tensor("a", "/ckpt/model-00001.safetensors", 200),
            tensor("b", "/ckpt/model-00001.safetensors", 100),
            tensor("c", "/ckpt/model-00002.safetensors", 100),
        ]);

        let shards = shards_of(&root);
        let first = shards["model-00001.safetensors"].expect("first shard attributed");
        assert_eq!((first.tensors, first.params), (2, 300));
        // 300 of 400 params — and 0.75 is exactly representable, so compare it exactly.
        assert!(
            (first.params_share - 0.75).abs() < f64::EPSILON,
            "{first:?}"
        );
        let second = shards["model-00002.safetensors"].expect("second shard attributed");
        assert_eq!((second.tensors, second.params), (1, 100));
        // A file the model reads nothing from stays unannotated — not "0 tensors".
        assert!(shards["config.json"].is_none(), "{shards:?}");
    }

    #[test]
    fn attribute_tensors_falls_back_to_an_unambiguous_file_name() {
        // The tensor list is rooted somewhere else than the browsed tree — a Hub
        // snapshot's symlink into the blob store, or a remote listing vs the proxy's
        // own paths. A name only one source file carries still attributes.
        let mut root = two_shard_tree();
        root.attribute_tensors(&[
            tensor("a", "/blobs/deadbeef/model-00001.safetensors", 10),
            tensor("b", "/blobs/cafe/model-00002.safetensors", 10),
        ]);
        let shards = shards_of(&root);
        assert_eq!(
            shards["model-00001.safetensors"].map(|s| s.tensors),
            Some(1)
        );
        assert_eq!(
            shards["model-00002.safetensors"].map(|s| s.tensors),
            Some(1)
        );

        // Ambiguous names don't guess: two source files called the same thing leave
        // the row unannotated rather than crediting it with one of them.
        let mut root = two_shard_tree();
        root.attribute_tensors(&[
            tensor("a", "/x/model-00001.safetensors", 10),
            tensor("b", "/y/model-00001.safetensors", 10),
        ]);
        assert!(
            shards_of(&root)["model-00001.safetensors"].is_none(),
            "an ambiguous name is not attributed"
        );
    }

    #[test]
    fn attribute_tensors_works_for_the_remote_source_path_forms() {
        // A remote read stamps `source_path` in a form the browse tree's own paths can
        // never equal: scp form for `--ssh-proxy` (`host:/dir/shard`, tree nodes have no
        // `host:`) and the full URI for `s3://` (tree nodes are prefix-relative keys).
        // The name fallback is what makes the counts appear at all there, so it is
        // tested on the shapes those two readers actually produce.
        for source in [
            "lab@host:/remote/ckpt/model-00001.safetensors",
            "s3://bucket/ckpt/model-00001.safetensors",
        ] {
            let mut root = two_shard_tree();
            root.attribute_tensors(&[tensor("a", source, 10), tensor("b", source, 30)]);
            let shard = shards_of(&root)["model-00001.safetensors"]
                .unwrap_or_else(|| panic!("attributed from {source}"));
            assert_eq!((shard.tensors, shard.params), (2, 40));
            assert!((shard.params_share - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn attribute_index_works_for_a_remote_index_path() {
        // A remote index's `path` is scp form or a URI, and its weight-map values are
        // still bare basenames — which is exactly why membership is matched on the name.
        // Without this the `--ssh-proxy` file browser marked nothing.
        for path in [
            "lab@host:/remote/ckpt/model.safetensors.index.json",
            "s3://bucket/ckpt/model.safetensors.index.json",
        ] {
            let index = crate::model::IndexEntry {
                path: path.to_string(),
                weight_map: std::iter::once((
                    "a".to_string(),
                    "model-00001.safetensors".to_string(),
                ))
                .collect(),
            };
            let mut root = two_shard_tree();
            root.attribute_index(std::slice::from_ref(&index));
            let rows = flatten(&root);
            let membership = |name: &str| {
                rows.iter()
                    .find(|r| r.name == name)
                    .and_then(FileRow::index_membership)
            };
            assert_eq!(
                membership("model-00001.safetensors"),
                Some(IndexMembership::Listed),
                "listed, from {path}"
            );
            assert_eq!(
                membership("model-00002.safetensors"),
                Some(IndexMembership::Unlisted),
                "an extra, from {path}"
            );
        }
    }

    #[test]
    fn attribute_index_marks_only_the_files_no_index_names() {
        // An index that lists one of the two shards — the other is an extra on disk,
        // which is what a LUT checkpoint's codebooks/qscales files are.
        let index = crate::model::IndexEntry {
            path: "/ckpt/model.safetensors.index.json".into(),
            weight_map: [
                ("a".to_string(), "model-00001.safetensors".to_string()),
                ("b".to_string(), "model-00001.safetensors".to_string()),
            ]
            .into_iter()
            .collect(),
        };
        let mut root = two_shard_tree();
        root.attribute_index(std::slice::from_ref(&index));

        let by_name: HashMap<String, Option<IndexMembership>> = flatten(&root)
            .into_iter()
            .filter_map(|r| {
                r.index_membership()
                    .map(|m| (r.name.clone(), Some(m)))
                    .or(match r.kind {
                        FileRowKind::File { .. } => Some((r.name, None)),
                        FileRowKind::Dir { .. } => None,
                    })
            })
            .collect();
        assert_eq!(
            by_name["model-00001.safetensors"],
            Some(IndexMembership::Listed)
        );
        assert_eq!(
            by_name["model-00002.safetensors"],
            Some(IndexMembership::Unlisted)
        );
        // A sidecar isn't a checkpoint file, so the question doesn't apply to it —
        // `config.json` must not read as an extra shard.
        assert_eq!(by_name["config.json"], None, "{by_name:?}");
    }

    #[test]
    fn with_no_index_nothing_is_an_extra() {
        // A single-file checkpoint has no index to be in; calling itself "not in the
        // index" would be an accusation about a file that is the whole model.
        let mut root = two_shard_tree();
        root.attribute_index(&[]);
        assert!(
            flatten(&root)
                .iter()
                .all(|r| r.index_membership().is_none()),
            "no index, no verdict"
        );
    }

    #[test]
    fn attribute_tensors_survives_an_empty_checkpoint() {
        let mut root = two_shard_tree();
        root.attribute_tensors(&[]);
        assert!(shards_of(&root).values().all(Option::is_none));
        // A zero-parameter tensor list divides by no total.
        root.attribute_tensors(&[tensor("a", "/ckpt/model-00001.safetensors", 0)]);
        let first = shards_of(&root)["model-00001.safetensors"].expect("attributed");
        assert_eq!((first.tensors, first.params), (1, 0));
        assert!(first.params_share.abs() < f64::EPSILON);
    }

    #[cfg(unix)]
    #[test]
    fn build_counts_the_names_a_hardlinked_file_has() {
        // A blob-dedup layout: the shard in the checkpoint is a *hardlink*, so its bytes
        // are shared and the checkpoint occupies less than its rows add up to. Sixteen
        // of the eighteen shards in a real LUT checkpoint look like this, and nothing in
        // the app said so until the browser started showing it.
        let base = std::env::temp_dir().join("ce_filetree_hardlink_test");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("shared.safetensors"), vec![0u8; 512]).unwrap();
        std::fs::hard_link(
            base.join("shared.safetensors"),
            base.join("second-name.safetensors"),
        )
        .unwrap();
        std::fs::write(base.join("alone.safetensors"), vec![0u8; 512]).unwrap();

        let links: HashMap<String, u64> = flatten(&build(&base, 8))
            .into_iter()
            .map(|r| (r.name.clone(), r.links()))
            .collect();
        assert_eq!(links["shared.safetensors"], 2, "{links:?}");
        assert_eq!(links["second-name.safetensors"], 2, "{links:?}");
        assert_eq!(
            links["alone.safetensors"], 1,
            "an ordinary file has one name"
        );

        // …and the directory row totals them, so a listing that adds up to more than
        // the checkpoint occupies says how much of itself is shared.
        let root = build(&base, 8);
        assert_eq!(flatten(&root)[0].hardlinked(), 2, "two of the three files");
        assert!(matches!(root, FileNode::Dir { files: 3, .. }));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn build_from_keys_makes_an_s3_native_tree() {
        // A flat object listing with a shared "layer_0/" prefix + a top-level file.
        let objects = vec![
            ObjectEntry::object("layer_0/weight".to_string(), 100),
            ObjectEntry::object("layer_0/bias".to_string(), 10),
            ObjectEntry::object("metadata.json".to_string(), 5),
        ];
        let root = build_from_keys("my-ckpt", &objects);
        let FileNode::Dir {
            name,
            children,
            size,
            files,
            expanded,
            ..
        } = &root
        else {
            panic!("root is a dir");
        };
        assert_eq!(name, "my-ckpt");
        assert!(*expanded);
        // Dir ("layer_0") sorts before the top-level file.
        let names: Vec<&str> = children.iter().map(FileNode::name).collect();
        assert_eq!(names, ["layer_0", "metadata.json"]);
        // Bottom-up aggregation: (100 + 10) + 5 = 115; 3 objects.
        assert_eq!(*size, 115);
        assert_eq!(*files, 3);

        // A nested object's `path` is its exact prefix-relative key (for URI rebuild).
        let FileNode::Dir {
            children: layer_children,
            ..
        } = &children[0]
        else {
            panic!("layer_0 is a dir");
        };
        let weight = layer_children
            .iter()
            .find(|c| c.name() == "weight")
            .expect("weight object present");
        assert!(
            matches!(weight, FileNode::File { path, .. } if path == Path::new("layer_0/weight"))
        );
        // An object listing knows nothing of hard links, so every row has one name.
        assert!(flatten(&root).iter().all(|r| r.links() == 1));
    }

    #[test]
    fn build_from_keys_carries_a_remote_listing_s_link_counts() {
        // The web builds a remote browse tree from the model's own file listing, and an
        // `--ssh-proxy` read does know `st_nlink` (its batched `stat` asks for `%h`). So
        // this path has to carry it, or the browser would show hardlinks for a local
        // checkpoint and nothing for the same checkpoint read over ssh.
        let objects = vec![
            ObjectEntry {
                key: "model-00001.safetensors".to_string(),
                size: 100,
                links: 2,
            },
            ObjectEntry::object("config.json".to_string(), 2),
        ];
        let root = build_from_keys("ckpt", &objects);
        let rows = flatten(&root);
        let links = |name: &str| rows.iter().find(|r| r.name == name).map(FileRow::links);
        assert_eq!(links("model-00001.safetensors"), Some(2));
        assert_eq!(links("config.json"), Some(1));
        assert_eq!(rows[0].hardlinked(), 1, "the root row totals them");
    }
}
