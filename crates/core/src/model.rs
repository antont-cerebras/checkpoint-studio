//! The central, serializable checkpoint model — **the one datatype** all
//! primary metadata is read into, and everything else is derived from.
//!
//! A [`Checkpoint`] holds the filesystem structure of a checkpoint (every file's
//! size, on-disk allocation, kind, symlink target, permissions/mtime) *and* each
//! safetensors file's parsed header (tensors + `__metadata__`), plus the parsed
//! `config.json`, index health inputs, and — for an `s3://` source — the S3 object
//! metadata. Readers ([`crate::readers`], Stage 3) fill it in **one pass**; the
//! tensor tree, file tree, byte-layout map, and every report are then pure
//! functions of it with **no further disk access**. It round-trips through JSON
//! (and any other serde format), which is the on-the-wire contract for the future
//! web-server / MCP frontends.

use crate::config::ModelConfig;
use crate::filetree::FileKind;
use crate::remote::S3Meta;
use crate::stats::DiskUsage;
use crate::tree::{MetadataInfo, TensorInfo};

/// Where a checkpoint was read from — determines how paths are interpreted and
/// which reader produced the model.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Source {
    /// A local directory / file on this machine.
    Local,
    /// A remote safetensors directory read over SFTP (`--ssh-proxy host /path`).
    Sftp { host: String, root: String },
    /// An `s3://…` cstorch checkpoint read via the remote host (`--ssh-proxy`).
    S3 { uri: String },
    /// A Hugging Face Hub repository, read over HTTPS — headers only, no weights
    /// (`crate::hf`).
    Hf { repo: String, revision: String },
}

/// The filesystem kind of a [`FileEntry`], as a **tagged sum type**: a regular
/// **file**, a **directory**, or a **symlink** each carry exactly the fields that
/// make sense for them — instead of the old flat struct with an `is_dir` flag plus
/// an optional `symlink_target`, where illegal states (a directory with a symlink
/// target, a symlink with `is_dir`) were representable. Serialized
/// internally-tagged: `{"type":"file","apparent":…,"allocated":…,"kind":…}`,
/// `{"type":"directory"}`, `{"type":"symlink","target":…,"apparent":…,…}`.
///
/// Sizes are **symlink-followed** (the single-source-of-truth invariant):
/// `apparent` is `st_size` of the target, `allocated` its on-disk block allocation
/// (0 when unknown — e.g. over SFTP without a `stat -L`, or for an s3 object).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FsNode {
    /// A regular file — its own apparent/allocated size, content [`FileKind`], and
    /// hard-link count (`links > 1` ⇒ the inode has other names; a hardlink is a
    /// regular file, not a distinct node kind, so it's this variant with `links`
    /// set, never a variant of its own).
    File {
        apparent: u64,
        allocated: u64,
        kind: FileKind,
        /// `st_nlink`; 1 for an ordinary file, `>1` when hardlinked. `1` when
        /// unknown (remote / non-Unix).
        links: u64,
    },
    /// A real, descendable directory. (A symlink *to* a directory is a
    /// [`FsNode::Symlink`], kept as a leaf so the walk can't cycle.)
    Directory,
    /// A symbolic link: `target` is the raw link text; the size/kind/links are the
    /// **followed** target's (0 / a broken-link fallback when it can't be statted).
    Symlink {
        target: String,
        apparent: u64,
        allocated: u64,
        kind: FileKind,
        links: u64,
    },
}

impl FsNode {
    /// Whether this is a real, descendable directory.
    #[must_use]
    pub fn is_dir(&self) -> bool {
        matches!(self, Self::Directory)
    }
    /// Apparent size in bytes (0 for a directory).
    #[must_use]
    pub fn apparent(&self) -> u64 {
        match self {
            Self::File { apparent, .. } | Self::Symlink { apparent, .. } => *apparent,
            Self::Directory => 0,
        }
    }
    /// On-disk allocation in bytes (0 for a directory / when unknown).
    #[must_use]
    pub fn allocated(&self) -> u64 {
        match self {
            Self::File { allocated, .. } | Self::Symlink { allocated, .. } => *allocated,
            Self::Directory => 0,
        }
    }
    /// The content classification, for a file or a symlink's target.
    #[must_use]
    pub fn file_kind(&self) -> Option<FileKind> {
        match self {
            Self::File { kind, .. } | Self::Symlink { kind, .. } => Some(*kind),
            Self::Directory => None,
        }
    }
    /// The raw link text when this is a symlink, else `None`.
    #[must_use]
    pub fn symlink_target(&self) -> Option<&str> {
        match self {
            Self::Symlink { target, .. } => Some(target),
            Self::File { .. } | Self::Directory => None,
        }
    }
    /// Hard-link count (`st_nlink`) of the underlying inode; `>1` means the file
    /// is hardlinked. 0 for a directory.
    #[must_use]
    pub fn links(&self) -> u64 {
        match self {
            Self::File { links, .. } | Self::Symlink { links, .. } => *links,
            Self::Directory => 0,
        }
    }
    /// Whether this entry is a hardlinked file (its inode has more than one name).
    #[must_use]
    pub fn is_hardlinked(&self) -> bool {
        self.links() > 1
    }
}

/// One entry in the checkpoint's directory tree — the unified filesystem metadata
/// that used to be scattered across `filetree::FileNode`, `stats::ShardDisk`,
/// `sftp::RemoteStat`, and `remote::S3Object`. The path/name/depth/permissions/
/// mtime are common to every entry; the filesystem kind and its kind-specific data
/// (size, content kind, link target) live in the tagged [`FsNode`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileEntry {
    /// The filesystem kind + kind-specific data (size / content kind / link target).
    /// Declared **first** so its internally-tagged `"type"` key leads the JSON
    /// object for every entry, ahead of the common path/name/… fields.
    #[serde(flatten)]
    pub node: FsNode,
    /// Path relative to the checkpoint root (POSIX `/`-separated).
    pub rel_path: String,
    /// The final path component (the display name).
    pub name: String,
    /// Depth below the root (0 = a top-level entry).
    pub depth: usize,
    /// Unix mode bits, when known (local reads).
    pub mode: Option<u32>,
    /// Modification time (seconds since the epoch), when known.
    pub mtime: Option<i64>,
    /// The **followed** inode number (`st_ino`), for de-duplicating shared inodes
    /// in the on-disk rollup: two hardlinks — or two symlinks to one blob — share
    /// it, so their bytes are counted once. `None` for a directory, a remote read,
    /// or a non-Unix platform. (Scoped to the checkpoint's own filesystem, so the
    /// bare inode number suffices — no `st_dev`.)
    pub inode: Option<u64>,
}

impl FileEntry {
    /// Whether this entry is a real, descendable directory.
    #[must_use]
    pub fn is_dir(&self) -> bool {
        self.node.is_dir()
    }
    /// Apparent (symlink-followed) size in bytes.
    #[must_use]
    pub fn apparent(&self) -> u64 {
        self.node.apparent()
    }
    /// On-disk allocation in bytes.
    #[must_use]
    pub fn allocated(&self) -> u64 {
        self.node.allocated()
    }
    /// The content classification (file / symlink target), else `None` for a dir.
    #[must_use]
    pub fn file_kind(&self) -> Option<FileKind> {
        self.node.file_kind()
    }
    /// The raw symlink target text, when this entry is a symlink.
    #[must_use]
    pub fn symlink_target(&self) -> Option<&str> {
        self.node.symlink_target()
    }
}

/// One safetensors file's parsed header — the tensors it stores and its
/// `__metadata__`, plus the byte sizes needed for the layout map. Non-safetensors
/// checkpoint files (gguf/npy/hdf5) also land here, one `ShardHeader` per file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShardHeader {
    /// The `source_path` these tensors carry (a local path, or a remote marker).
    pub path: String,
    /// Whole-file size in bytes (for the layout map's trailing gap); 0 if unknown.
    pub total_len: u64,
    /// Size of the header region (`8 + N` for safetensors), or 0 for other formats.
    pub header_len: u64,
    pub tensors: Vec<TensorInfo>,
    pub metadata: Vec<MetadataInfo>,
    /// Top-level keys this file's header declares **more than once**, from the parse that
    /// read it — see [`crate::stheader::ParsedHeader::duplicate_keys`].
    ///
    /// Carried on the model rather than re-derived, because it can only be seen *while*
    /// parsing: the JSON map keeps the last of two identical keys, so by the time anything
    /// holds a tensor list the first declaration is gone. Recording it here is what makes
    /// the check work for a remote or Hub shard, whose text nothing can read again.
    #[serde(default)]
    pub duplicate_keys: Vec<String>,
}

/// A checkpoint's `model.safetensors.index.json` (the pieces the health check
/// needs), in a serde-friendly form (no `PathBuf`).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IndexEntry {
    /// The index file's path (display form).
    pub path: String,
    /// tensor name → shard file basename.
    pub weight_map: std::collections::BTreeMap<String, String>,
    /// The index's own `metadata.total_size` — the byte total it *claims* the
    /// checkpoint's tensors come to. `None` when the index omits it (it's optional).
    ///
    /// Kept because it is a claim that can be wrong, and often is: it's written once by
    /// whatever produced the checkpoint and not recomputed when a shard is re-quantised
    /// or a tensor dropped. Loaders that pre-allocate from it then get it wrong.
    #[serde(default)]
    pub total_size: Option<u64>,
}

impl IndexEntry {
    /// Parse a `model.safetensors.index.json` body into its weight map. `None` when the
    /// JSON has no `weight_map` object — which is what "not an index" looks like.
    ///
    /// Shared by the local reader and the Hub reader: the file is the same JSON wherever
    /// it is read from, and the health check compares it against the loaded tensors
    /// identically in both cases.
    #[must_use]
    pub fn parse(path: &str, json: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        let wm = v.get("weight_map")?.as_object()?;
        Some(Self {
            path: path.to_string(),
            weight_map: wm
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect(),
            total_size: v
                .get("metadata")
                .and_then(|m| m.get("total_size"))
                .and_then(serde_json::Value::as_u64),
        })
    }
}

/// A checkpoint file whose header could not be read, and why.
///
/// A directory read keeps going past one of these: fifteen good shards are worth showing,
/// and refusing all of them because the sixteenth is a truncated download tells you less
/// than showing the fifteen and naming the sixteenth. The tensors it would have
/// contributed are simply absent — which is exactly what `check` reports as an error and
/// the file browser marks on its row.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnreadableShard {
    /// The file's path, as the reader saw it.
    pub path: String,
    /// The reader's message chain, flattened (`{:#}`) — e.g. `Failed to parse
    /// SafeTensors header: …: expected value at line 1 column 1`.
    pub error: String,
}

/// The one serializable checkpoint model. Read once; everything derives from it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub source: Source,
    /// The checkpoint's root directory / prefix (display form) — what `f` on the
    /// tree root copies, and the base `rel_path`s are relative to.
    pub root: String,
    /// Every file in the checkpoint directory (recursively), for the file browser
    /// and the on-disk stats — no further `readdir`/`stat` needed after this.
    pub files: Vec<FileEntry>,
    /// Per-file parsed headers, for the tensor tree, layout map, and reports.
    pub shards: Vec<ShardHeader>,
    /// The parsed `config.json`, when present.
    pub config: Option<ModelConfig>,
    /// Parsed index(es), for the health check.
    pub index: Vec<IndexEntry>,
    /// S3 object metadata — `Some` only for an `s3://` source.
    pub s3: Option<S3Meta>,
    /// Checkpoint files whose headers wouldn't parse. Empty for a healthy checkpoint;
    /// non-empty means what you're looking at is *part* of one — see [`UnreadableShard`].
    #[serde(default)]
    pub unreadable: Vec<UnreadableShard>,
}

/// The single path that denotes a checkpoint made of `files`: the file itself when there
/// is one, the directory its shards share when they all sit in one, and `None` when they
/// span directories — then no single path names them and a caller must list them all.
///
/// This is the decision behind both "what to call the checkpoint" ([`root_label`]) and
/// "what to write on a command line", which is why it is one function. Getting it wrong
/// is not cosmetic: naming *a shard* where the checkpoint was meant produces a command
/// that compares something else. (It did — the web compare screen offered
/// `diff OLD <dir>/codebooks.safetensors` for a sharded checkpoint, because it took the
/// first resolved file instead of asking this question.)
#[must_use]
pub fn checkpoint_path(files: &[std::path::PathBuf]) -> Option<&std::path::Path> {
    match files.split_first() {
        None => None,
        Some((only, [])) => Some(only.as_path()),
        Some((first, _)) => {
            let dir = first.parent()?;
            files.iter().all(|f| f.parent() == Some(dir)).then_some(dir)
        }
    }
}

/// A concise display name for a checkpoint made of `files`: the file's own name for a
/// single file; the shared parent directory's name when every file sits in one
/// directory; `"checkpoint"` when neither applies (no files, or a glob spanning
/// directories).
///
/// Shared because every frontend puts this string at the top of its tree, and each had
/// derived it separately — the TUI from the file list, the web from
/// [`Checkpoint::root`]'s basename. Those disagree on the commonest case there is: open
/// one `model.safetensors` and the terminal called the root `model.safetensors` while
/// the browser called it by the *containing directory*. Deriving a display name twice
/// is how that happens, so it is derived once, here.
#[must_use]
pub fn root_label(files: &[std::path::PathBuf]) -> String {
    checkpoint_path(files)
        .and_then(std::path::Path::file_name)
        .map_or_else(
            || "checkpoint".to_string(),
            |s| s.to_string_lossy().into_owned(),
        )
}

impl Checkpoint {
    /// Every tensor across all shards, in shard order (the flattened primary
    /// tensor list the tree / stats / diff consume).
    pub fn tensors(&self) -> impl Iterator<Item = &TensorInfo> {
        self.shards.iter().flat_map(|s| s.tensors.iter())
    }

    /// Every `__metadata__` entry across all shards, in shard order.
    pub fn metadata(&self) -> impl Iterator<Item = &MetadataInfo> {
        self.shards.iter().flat_map(|s| s.metadata.iter())
    }

    /// Owned copies of the flattened tensors — a bridge for the (still
    /// `Vec<TensorInfo>`-based) views/reports until they take `&Checkpoint`.
    #[must_use]
    pub fn tensors_vec(&self) -> Vec<TensorInfo> {
        self.tensors().cloned().collect()
    }

    /// Owned copies of the flattened metadata (same bridging role).
    #[must_use]
    pub fn metadata_vec(&self) -> Vec<MetadataInfo> {
        self.metadata().cloned().collect()
    }

    /// The on-disk footprint rolled up from every **checkpoint file** the walk
    /// found (all `.safetensors`/`.gguf`/… in the directory, not just the loaded
    /// shards) — the `DiskUsage` the stats "on disk" section shows, now derived
    /// from the cached model (symlink-followed sizes) instead of a live `stat`.
    #[must_use]
    pub fn disk_usage(&self) -> Option<DiskUsage> {
        use crate::stats::ShardDisk;
        // Count each physical inode once: a hardlink, or a symlink to a blob that
        // another shard also links, shares an inode and so adds no real bytes.
        // Entries without a known inode (remote / non-Unix) are never deduped.
        let mut seen = std::collections::HashSet::new();
        let shards: Vec<ShardDisk> = self
            .files
            .iter()
            .filter(|f| f.file_kind() == Some(FileKind::Checkpoint))
            .filter(|f| f.inode.is_none_or(|ino| seen.insert(ino)))
            .map(|f| ShardDisk {
                name: f.name.clone(),
                apparent: f.apparent(),
                allocated: f.allocated(),
                links: f.node.links(),
            })
            .collect();
        DiskUsage::from_shards(shards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Layout, Storage};

    /// Every branch of the root label, because both frontends now put its output at
    /// the top of their tree and a wrong answer is visible on the first screen.
    #[test]
    fn the_root_label_names_a_file_a_directory_or_neither() {
        use std::path::PathBuf;

        assert_eq!(root_label(&[]), "checkpoint", "no files");
        assert_eq!(
            root_label(&[PathBuf::from("/m/model.safetensors")]),
            "model.safetensors",
            "a single file is named by the file — not by its directory"
        );
        assert_eq!(
            root_label(&[
                PathBuf::from("/m/model-00001-of-00002.safetensors"),
                PathBuf::from("/m/model-00002-of-00002.safetensors"),
            ]),
            "m",
            "shards sharing a directory are named by that directory"
        );
        assert_eq!(
            root_label(&[
                PathBuf::from("/a/one.safetensors"),
                PathBuf::from("/b/two.safetensors"),
            ]),
            "checkpoint",
            "files spanning directories have no shared name"
        );
        // A relative single file still resolves to its own name, since that is how
        // the CLI is normally invoked (`checkpoint-studio model.safetensors`).
        assert_eq!(
            root_label(&[PathBuf::from("model.safetensors")]),
            "model.safetensors"
        );
    }

    fn sample() -> Checkpoint {
        Checkpoint {
            source: Source::Sftp {
                host: "net004".into(),
                root: "/opt/ckpt".into(),
            },
            root: "/opt/ckpt".into(),
            files: vec![FileEntry {
                rel_path: "model-00001-of-00002.safetensors".into(),
                name: "model-00001-of-00002.safetensors".into(),
                depth: 0,
                mode: Some(0o644),
                mtime: Some(1_700_000_000),
                inode: Some(42),
                // An HF-cache-style symlink into a blob store: the followed target
                // sizes drive disk usage.
                node: FsNode::Symlink {
                    target: "/blobs/abc".into(),
                    apparent: 4_000_000_000,
                    allocated: 4_000_000_000,
                    kind: FileKind::Checkpoint,
                    links: 1,
                },
            }],
            shards: vec![ShardHeader {
                path: "net004:/opt/ckpt/model-00001-of-00002.safetensors".into(),
                total_len: 4_000_000_000,
                header_len: 8 + 512,
                tensors: vec![TensorInfo {
                    name: "model.embed_tokens.weight".into(),
                    dtype: "BF16".into(),
                    shape: vec![152_064, 4096],
                    size_bytes: 152_064 * 4096 * 2,
                    num_elements: 152_064 * 4096,
                    storage: Storage::Unknown,
                    source_path: "net004:/opt/ckpt/model-00001-of-00002.safetensors".into(),
                    layout: Layout::ByteRange {
                        start: 0,
                        end: 1_245_708_288,
                    },
                }],
                metadata: vec![MetadataInfo {
                    name: "format".into(),
                    value: "pt".into(),
                    value_type: "string".into(),
                }],
                duplicate_keys: Vec::new(),
            }],
            config: Some(ModelConfig {
                model_type: Some("qwen3_moe".into()),
                num_hidden_layers: Some(48),
                ..Default::default()
            }),
            index: vec![IndexEntry {
                path: "model.safetensors.index.json".into(),
                weight_map: std::iter::once((
                    "model.embed_tokens.weight".to_string(),
                    "model-00001-of-00002.safetensors".to_string(),
                ))
                .collect(),
                total_size: None,
            }],
            s3: None,
            unreadable: Vec::new(),
        }
    }

    #[test]
    fn checkpoint_round_trips_through_json() {
        let cp = sample();
        let json = serde_json::to_string(&cp).unwrap();
        let back: Checkpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(back.root, "/opt/ckpt");
        assert_eq!(back.tensors().count(), 1);
        assert_eq!(back.metadata().count(), 1);
        assert_eq!(back.source, cp.source);
        // Disk usage is rolled up from the file entries (symlink-followed sizes).
        let disk = back.disk_usage().unwrap();
        assert_eq!(disk.total_apparent, 4_000_000_000);
        assert_eq!(disk.shards.len(), 1);
        // config + index + symlink target survive the round-trip.
        assert_eq!(back.config.unwrap().num_hidden_layers, Some(48));
        assert_eq!(back.index[0].weight_map.len(), 1);
        // The tagged fs-node round-trips: a symlink with its followed target.
        assert_eq!(back.files[0].symlink_target(), Some("/blobs/abc"));
        assert!(matches!(back.files[0].node, FsNode::Symlink { .. }));
        assert_eq!(back.files[0].file_kind(), Some(FileKind::Checkpoint));
    }
}
