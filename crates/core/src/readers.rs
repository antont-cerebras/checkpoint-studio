//! Readers: the **only** code that touches disk / SSH. Each fills the central
//! [`crate::model::Checkpoint`] in one pass — the filesystem walk, every
//! safetensors (and gguf/npy/hdf5) header, `config.json`, and the index — so that
//! afterwards the tensor tree, file browser, byte-layout map, and reports are all
//! pure functions of the cached model with no further disk access.
//!
//! This module owns the local reader; the remote (SFTP / s3-cstorch) readers stay
//! in [`crate::remote`] / [`crate::sftp`] and are adapted to produce a
//! `Checkpoint` in a later step.

use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::filetree::FileKind;
use crate::model::{Checkpoint, FileEntry, FsNode, IndexEntry, ShardHeader, Source};
use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

/// Read a local checkpoint (a directory, a single file, or several files) fully
/// into a [`Checkpoint`]: the recursive filesystem walk (sizes symlink-followed,
/// with on-disk allocation / mode / mtime), every checkpoint file's header, the
/// sidecar `config.json`, and any `model.safetensors.index.json`.
pub fn read_local(files: &[PathBuf]) -> Result<Checkpoint> {
    let root = common_root(files);
    let root_str = root.to_string_lossy().into_owned();

    // The whole directory tree — one walk, reused by the file browser and the
    // on-disk stats (no later `readdir`/`stat`).
    let mut entries = Vec::new();
    walk(&root, &root, 0, &mut entries);
    entries.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));

    // Per checkpoint file: its parsed header (header-only; never the tensor data).
    let mut shards = Vec::new();
    for file_path in files {
        if let Some(shard) = read_shard_header(file_path)? {
            shards.push(shard);
        }
    }

    let config = crate::config::load_local(files);
    let index = read_indexes(&root);

    Ok(Checkpoint {
        source: Source::Local,
        root: root_str,
        files: entries,
        shards,
        config,
        index,
        s3: None,
    })
}

/// The directory a set of paths shares — a single file's parent, or the common
/// parent of several; `.` when there's nothing to anchor to.
fn common_root(files: &[PathBuf]) -> PathBuf {
    match files {
        [] => PathBuf::from("."),
        [one] => one
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf),
        [first, others @ ..] => {
            // Longest shared directory prefix by component.
            let mut common = first
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            for f in others {
                let p = f.parent().unwrap_or_else(|| Path::new("."));
                while !p.starts_with(&common) {
                    if !common.pop() {
                        return PathBuf::from(".");
                    }
                }
            }
            if common.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                common
            }
        }
    }
}

/// Recursively collect [`FileEntry`]s under `dir`. Symlinks are followed for size
/// (matching the file browser / layout invariant) but a symlinked directory is a
/// leaf (not descended) so the walk can't cycle. Dotfiles are skipped.
fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<FileEntry>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let name = match entry.file_name().to_str() {
            Some(n) if !n.starts_with('.') => n.to_string(),
            _ => continue,
        };
        let path = entry.path();
        let is_symlink = entry.file_type().ok().is_some_and(|t| t.is_symlink());
        // Followed metadata (target), with the link's own as a broken-link fallback.
        let meta = std::fs::metadata(&path).or_else(|_| entry.metadata());
        let (is_dir, apparent, allocated, mode, mtime, links, inode) =
            meta.as_ref()
                .map_or((false, 0, 0, None, None, 1, None), |m| {
                    (
                        m.is_dir(),
                        if m.is_dir() { 0 } else { m.len() },
                        block_bytes(m),
                        unix_mode(m),
                        mtime_secs(m),
                        nlink(m),
                        inode_of(m),
                    )
                });
        // A real directory descends; a symlinked directory stays a leaf.
        let descendable = is_dir && !is_symlink;
        let rel_path = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        // Classify into the tagged fs-node: a symlink (with its followed sizes),
        // a real directory, or a regular file.
        let node = if is_symlink {
            let target = std::fs::read_link(&path)
                .ok()
                .map(|t| t.to_string_lossy().into_owned())
                .unwrap_or_default();
            FsNode::Symlink {
                target,
                apparent,
                allocated,
                kind: FileKind::of(&name),
                links,
            }
        } else if descendable {
            FsNode::Directory
        } else {
            FsNode::File {
                apparent,
                allocated,
                kind: FileKind::of(&name),
                links,
            }
        };
        out.push(FileEntry {
            rel_path,
            name: name.clone(),
            depth,
            mode,
            mtime,
            inode,
            node,
        });
        if descendable {
            walk(root, &path, depth + 1, out);
        }
    }
}

#[cfg(unix)]
fn block_bytes(m: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    if m.is_dir() { 0 } else { m.blocks() * 512 }
}
#[cfg(not(unix))]
fn block_bytes(_m: &std::fs::Metadata) -> u64 {
    0
}

#[cfg(unix)]
// The `Option` is the shared signature of a cfg pair: off Unix there is no mode to
// report and the sibling below returns `None`. Unwrapping it here would mean two
// different return types for one call site.
#[allow(clippy::unnecessary_wraps)]
fn unix_mode(m: &std::fs::Metadata) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    Some(m.mode())
}
#[cfg(not(unix))]
fn unix_mode(_m: &std::fs::Metadata) -> Option<u32> {
    None
}

/// Hard-link count (`st_nlink`) of the (followed) target; `1` when unknown.
#[cfg(unix)]
fn nlink(m: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    m.nlink()
}
#[cfg(not(unix))]
fn nlink(_m: &std::fs::Metadata) -> u64 {
    1
}

/// The (followed) inode number (`st_ino`), for the on-disk dedup; `None` off-Unix.
#[cfg(unix)]
// The `Option` is the shared signature of a cfg pair: off Unix there is no inode to
// report and the sibling below returns `None`. Unwrapping it here would mean two
// different return types for one call site.
#[allow(clippy::unnecessary_wraps)]
fn inode_of(m: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(m.ino())
}
#[cfg(not(unix))]
fn inode_of(_m: &std::fs::Metadata) -> Option<u64> {
    None
}

fn mtime_secs(m: &std::fs::Metadata) -> Option<i64> {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
}

/// Read one checkpoint file's header into a [`ShardHeader`], dispatching by
/// extension. Non-checkpoint files (and unsupported formats) yield `None`.
fn read_shard_header(file_path: &Path) -> Result<Option<ShardHeader>> {
    let source_path = absolute_path(file_path);
    match file_path.extension().and_then(|s| s.to_str()) {
        Some("safetensors") => {
            let mut file = std::fs::File::open(file_path)
                .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
            let total_len = file.metadata().map_or(0, |m| m.len());
            let mut len_buf = [0u8; 8];
            file.read_exact(&mut len_buf).with_context(|| {
                format!("Failed to read header length: {}", file_path.display())
            })?;
            let n = crate::stheader::header_len(u64::from_le_bytes(len_buf), &source_path)?;
            let mut header_buf = vec![0u8; n];
            file.read_exact(&mut header_buf)
                .with_context(|| format!("Failed to read header: {}", file_path.display()))?;
            let (tensors, metadata) = crate::stheader::parse_header(&header_buf, &source_path)?;
            Ok(Some(ShardHeader {
                path: source_path,
                total_len,
                header_len: 8 + n as u64,
                tensors,
                metadata,
            }))
        }
        Some("gguf") => {
            let (tensors, metadata) = read_gguf(file_path, &source_path)?;
            Ok(Some(shard(source_path, file_path, tensors, metadata)))
        }
        Some("npy") => {
            let (tensors, metadata) = read_numpy(file_path, &source_path)?;
            Ok(Some(shard(source_path, file_path, tensors, metadata)))
        }
        Some("npz") => {
            let (tensors, metadata) = read_npz(file_path, &source_path)?;
            Ok(Some(shard(source_path, file_path, tensors, metadata)))
        }
        // Recognized HDF5 by extension, OR an extensionless file whose magic says
        // HDF5 (Cerebras checkpoints are often written without an extension).
        Some("h5" | "hdf5") => read_hdf5_shard(file_path, source_path),
        other if other != Some("safetensors") && looks_like_hdf5(file_path) => {
            read_hdf5_shard(file_path, source_path)
        }
        _ => Ok(None),
    }
}

/// Read an HDF5 shard header (a no-op returning `None` when the `hdf5` feature is off).
///
/// The `Result` is the signature of a cfg pair, as with `unix_mode` above: with the feature
/// on, the HDF5 read can fail; with it off the body is a stub that cannot. One call site,
/// so one return type.
#[allow(clippy::unnecessary_wraps)]
fn read_hdf5_shard(file_path: &Path, source_path: String) -> Result<Option<ShardHeader>> {
    #[cfg(feature = "hdf5")]
    {
        let (tensors, metadata) = crate::hdf5::read(file_path)?;
        Ok(Some(shard(source_path, file_path, tensors, metadata)))
    }
    #[cfg(not(feature = "hdf5"))]
    {
        let _ = (file_path, source_path);
        Ok(None)
    }
}

/// Whether `path`'s first bytes are the HDF5 signature (`\x89HDF\r\n\x1a\n`) — so an
/// extensionless HDF5 checkpoint is recognized. Cheap (reads 8 bytes); false on any
/// read error.
#[must_use]
pub fn looks_like_hdf5(path: &Path) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 8];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut buf))
        .is_ok()
        && buf == [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n']
}

/// A [`ShardHeader`] for a non-safetensors format: `total_len` = the file size,
/// `header_len` = 0 (no safetensors-style header region).
fn shard(
    source_path: String,
    file_path: &Path,
    tensors: Vec<TensorInfo>,
    metadata: Vec<MetadataInfo>,
) -> ShardHeader {
    let total_len = std::fs::metadata(file_path).map_or(0, |m| m.len());
    ShardHeader {
        path: source_path,
        total_len,
        header_len: 0,
        tensors,
        metadata,
    }
}

fn read_gguf(file_path: &Path, source_path: &str) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
    use crate::gguf::{GGUFFile, GGUFValue};
    let mut file = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    let mut buffer = Vec::new();
    // `read_to_end` on an open handle: the caller already opened it to read the
    // header, so re-opening by path (what `fs::read` would do) would race a rename.
    #[allow(clippy::verbose_file_reads)]
    file.read_to_end(&mut buffer)
        .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
    let gguf = GGUFFile::read(&buffer)
        .with_context(|| format!("Failed to parse GGUF file: {}", file_path.display()))?;
    let mut metadata = Vec::new();
    for (key, value) in &gguf.metadata {
        let value_type = match value {
            GGUFValue::U8(_) => "u8",
            GGUFValue::I8(_) => "i8",
            GGUFValue::U16(_) => "u16",
            GGUFValue::I16(_) => "i16",
            GGUFValue::U32(_) => "u32",
            GGUFValue::I32(_) => "i32",
            GGUFValue::F32(_) => "f32",
            GGUFValue::U64(_) => "u64",
            GGUFValue::I64(_) => "i64",
            GGUFValue::F64(_) => "f64",
            GGUFValue::Bool(_) => "bool",
            GGUFValue::String(_) => "string",
            GGUFValue::Array(_) => "array",
        };
        metadata.push(MetadataInfo {
            name: key.clone(),
            value: value.to_string(),
            value_type: value_type.to_string(),
        });
    }
    let mut tensors = Vec::new();
    for tensor in &gguf.tensors {
        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        let num_elements = shape.iter().product::<usize>();
        // Exact integer block arithmetic — an f32 bytes-per-element product lost
        // precision on large tensors and dropped the final partial block.
        let size_bytes = tensor.tensor_type.stored_size(num_elements);
        tensors.push(TensorInfo {
            name: tensor.name.clone(),
            dtype: tensor.tensor_type.to_string(),
            shape,
            size_bytes,
            num_elements,
            storage: Storage::Unknown,
            source_path: source_path.to_string(),
            layout: Layout::Offset(tensor.offset),
        });
    }
    Ok((tensors, metadata))
}

fn read_numpy(file_path: &Path, source_path: &str) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
    let mut file = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    let total_len = file.metadata().map_or(0, |m| m.len());
    let name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("array")
        .to_string();
    let header =
        crate::npy::parse_header(&mut file).map_err(|e| anyhow::anyhow!("{source_path}: {e}"))?;
    let num_elements = header.shape.iter().product::<usize>();
    let tensor = TensorInfo {
        name,
        dtype: header.dtype,
        shape: header.shape,
        size_bytes: (total_len as usize).saturating_sub(header.data_offset),
        num_elements,
        storage: Storage::Unknown,
        source_path: source_path.to_string(),
        layout: Layout::ByteRange {
            start: header.data_offset as u64,
            end: total_len,
        },
    };
    Ok((vec![tensor], Vec::new()))
}

fn read_npz(file_path: &Path, source_path: &str) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
    let file = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open file: {}", file_path.display()))?;
    read_npz_reader(file, source_path)
}

fn read_npz_reader<R: Read + Seek>(
    reader: R,
    source_path: &str,
) -> Result<(Vec<TensorInfo>, Vec<MetadataInfo>)> {
    let mut tensors = Vec::new();
    let mut zip = zip::ZipArchive::new(reader)
        .with_context(|| format!("Failed to read .npz archive: {source_path}"))?;
    let entries: Vec<String> = zip.file_names().map(String::from).collect();
    for entry_name in entries {
        let Some(name) = entry_name.strip_suffix(".npy") else {
            continue;
        };
        let mut entry = zip
            .by_name(&entry_name)
            .with_context(|| format!("Failed to read {entry_name} in {source_path}"))?;
        let stored_bytes = entry.compressed_size() as usize;
        let uncompressed = entry.size() as usize;
        let compressed = entry.compression() != zip::CompressionMethod::Stored;
        let header = crate::npy::parse_header(&mut entry)
            .map_err(|e| anyhow::anyhow!("{source_path}: {entry_name}: {e}"))?;
        let num_elements = header.shape.iter().product::<usize>();
        let storage = if compressed {
            Storage::Compressed {
                codec: "deflate".to_string(),
                stored_bytes,
            }
        } else {
            Storage::Raw
        };
        tensors.push(TensorInfo {
            name: name.to_string(),
            dtype: header.dtype,
            shape: header.shape,
            size_bytes: uncompressed.saturating_sub(header.data_offset),
            num_elements,
            storage,
            source_path: source_path.to_string(),
            layout: Layout::None,
        });
    }
    Ok((tensors, Vec::new()))
}

/// Read every `model.safetensors.index.json` under `root` into serde-friendly
/// [`IndexEntry`]s (for the health check).
fn read_indexes(root: &Path) -> Vec<IndexEntry> {
    let mut out = Vec::new();
    let index_path = root.join("model.safetensors.index.json");
    if let Ok(text) = std::fs::read_to_string(&index_path)
        && let Some(entry) = IndexEntry::parse(&index_path.to_string_lossy(), &text)
    {
        out.push(entry);
    }
    out
}

/// Absolute path of `p` (best-effort: canonicalization-free, just prefixes the
/// current dir when `p` is relative) — the `source_path` tensors carry.
fn absolute_path(p: &Path) -> String {
    if p.is_absolute() {
        p.to_string_lossy().into_owned()
    } else {
        std::env::current_dir()
            .map_or_else(|_| p.to_path_buf(), |cwd| cwd.join(p))
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal safetensors file: one f32 tensor `w` of shape [2,2].
    fn write_st(path: &Path) {
        let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn read_local_fills_the_model_in_one_pass() {
        let dir = std::env::temp_dir().join("ce_readers_local_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_st(&dir.join("model.safetensors"));
        std::fs::write(
            dir.join("config.json"),
            br#"{"model_type":"llama","num_hidden_layers":2}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map":{"w":"model.safetensors"}}"#,
        )
        .unwrap();

        let cp = read_local(&[dir.join("model.safetensors")]).unwrap();
        // Header parsed into the shard.
        assert_eq!(cp.shards.len(), 1);
        assert_eq!(cp.tensors().count(), 1);
        assert_eq!(cp.tensors().next().unwrap().name, "w");
        assert_eq!(cp.metadata().count(), 1);
        assert!(cp.shards[0].total_len > cp.shards[0].header_len);
        // Filesystem walk captured the files (with sizes), so the browser + on-disk
        // stats need no further disk access.
        assert!(
            cp.files
                .iter()
                .any(|f| f.name == "model.safetensors" && f.apparent() > 0)
        );
        assert!(cp.files.iter().any(|f| f.name == "config.json"));
        // Sidecar config + index parsed in the same pass.
        assert_eq!(
            cp.config.as_ref().unwrap().model_type.as_deref(),
            Some("llama")
        );
        assert_eq!(cp.index.len(), 1);
        assert_eq!(
            cp.index[0].weight_map.get("w").map(String::as_str),
            Some("model.safetensors")
        );
        // The whole model serializes.
        let json = serde_json::to_string(&cp).unwrap();
        assert!(json.contains("\"model_type\":\"llama\""));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The walk classifies each entry into the tagged [`FsNode`]: a regular file,
    /// a real directory, and a symlink (carrying its raw target + followed size).
    #[cfg(unix)]
    #[test]
    fn walk_tags_files_dirs_and_symlinks() {
        use crate::model::FsNode;
        let dir = std::env::temp_dir().join("ce_readers_fsnode_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        write_st(&dir.join("model.safetensors"));
        // An HF-cache-style symlink to the real shard.
        std::os::unix::fs::symlink(dir.join("model.safetensors"), dir.join("link.safetensors"))
            .unwrap();

        let cp = read_local(&[dir.join("model.safetensors")]).unwrap();
        let node = |name: &str| {
            cp.files
                .iter()
                .find(|f| f.name == name)
                .map(|f| f.node.clone())
        };

        // Regular shard → File, with a nonzero size and the Checkpoint content kind.
        assert!(matches!(
            node("model.safetensors"),
            Some(FsNode::File { kind: FileKind::Checkpoint, apparent, .. }) if apparent > 0
        ));
        // Subdirectory → Directory (no size fields at all).
        assert!(matches!(node("sub"), Some(FsNode::Directory)));
        // Symlink → Symlink, carrying its raw target and the *followed* size/kind.
        match node("link.safetensors") {
            Some(FsNode::Symlink {
                target,
                apparent,
                kind,
                ..
            }) => {
                assert!(target.ends_with("model.safetensors"), "target: {target}");
                assert!(apparent > 0, "followed size");
                assert_eq!(kind, FileKind::Checkpoint);
            }
            other => panic!("expected a symlink fs-node, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two hard links to one shard share an inode, so the on-disk rollup counts
    /// its bytes once (not twice) — and the walk records `links > 1`.
    #[cfg(unix)]
    #[test]
    fn disk_usage_dedups_hardlinked_shards() {
        use crate::model::FsNode;
        let dir = std::env::temp_dir().join("ce_readers_hardlink_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_st(&dir.join("model.safetensors"));
        // A second name for the same inode (a hard link, not a symlink).
        std::fs::hard_link(
            dir.join("model.safetensors"),
            dir.join("model-copy.safetensors"),
        )
        .unwrap();

        let cp = read_local(&[dir.join("model.safetensors")]).unwrap();
        // Both names are regular files reporting links == 2 (the inode has 2 names).
        for name in ["model.safetensors", "model-copy.safetensors"] {
            let f = cp.files.iter().find(|f| f.name == name).unwrap();
            assert!(
                matches!(f.node, FsNode::File { links: 2, .. }),
                "{name} should report 2 hard links, got {:?}",
                f.node
            );
        }
        // Two shard files on disk, but one physical inode → counted once.
        let disk = cp.disk_usage().unwrap();
        assert_eq!(disk.shards.len(), 1, "shared inode counted once");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Reading real files through the one entry point every frontend uses.
///
/// `read_local` is what turns paths into the `Checkpoint` model — the dispatch by
/// format, the directory walk, the header parse, the shard grouping. It was exercised
/// only end-to-end through the CLI; these pin the model it produces and the errors it
/// gives for input a user can actually hand it.
#[cfg(test)]
mod local_reads {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("the workspace root")
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn reads_a_safetensors_file_into_the_model() {
        let ck = read_local(&[fixture("tiny.safetensors")]).expect("the fixture reads");
        assert!(!ck.shards.is_empty(), "a file yields at least one shard");
        let tensors: Vec<_> = ck.shards.iter().flat_map(|s| &s.tensors).collect();
        assert!(!tensors.is_empty(), "the fixture has tensors");
        // Every tensor must carry the file it came from, or the detail screen can't
        // open it and the layout link has nothing to point at.
        for t in &tensors {
            assert!(
                t.source_path.ends_with("tiny.safetensors"),
                "{} has source {}",
                t.name,
                t.source_path
            );
            assert!(
                t.size_bytes > 0 && t.num_elements > 0,
                "{} is empty",
                t.name
            );
        }
        // The root is the containing directory, so the file browser has somewhere to go.
        assert!(ck.root.ends_with("fixtures"), "root was {}", ck.root);
    }

    #[test]
    fn reads_several_files_as_several_shards() {
        let ck = read_local(&[
            fixture("diff_old.safetensors"),
            fixture("diff_new.safetensors"),
        ])
        .expect("both fixtures read");
        assert_eq!(ck.shards.len(), 2, "one shard per file");
        let names: Vec<&str> = ck.shards.iter().map(|s| s.path.as_str()).collect();
        assert!(names.iter().any(|p| p.ends_with("diff_old.safetensors")));
        assert!(names.iter().any(|p| p.ends_with("diff_new.safetensors")));
    }

    #[test]
    fn metadata_travels_with_its_shard() {
        let ck = read_local(&[fixture("diff_meta.safetensors")]).expect("reads");
        let meta: Vec<_> = ck.shards.iter().flat_map(|s| &s.metadata).collect();
        assert!(!meta.is_empty(), "this fixture carries __metadata__");
        for m in meta {
            assert!(!m.name.is_empty() && !m.value_type.is_empty(), "{m:?}");
        }
    }

    #[test]
    fn a_missing_file_is_an_error_naming_the_path() {
        let missing = fixture("no-such-checkpoint.safetensors");
        let err =
            read_local(std::slice::from_ref(&missing)).expect_err("a missing file must error");
        let text = format!("{err:#}");
        assert!(
            text.contains("no-such-checkpoint"),
            "the error should name the file: {text}"
        );
    }

    #[test]
    fn a_file_that_is_not_a_checkpoint_is_rejected() {
        // The fixtures directory has non-checkpoint files (this source tree does too);
        // point the reader at something with the wrong contents.
        let dir = std::env::temp_dir().join("ckpt_studio_reader_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let bogus = dir.join("not-a-checkpoint.safetensors");
        std::fs::write(&bogus, b"this is not a safetensors header").expect("write");
        assert!(
            read_local(std::slice::from_ref(&bogus)).is_err(),
            "garbage must not parse as a checkpoint"
        );
        let _ = std::fs::remove_file(&bogus);
    }

    #[test]
    fn an_empty_file_list_yields_an_empty_model_rather_than_an_error() {
        // `--filter` can exclude everything; the app then shows an empty tree, so the
        // reader must not fail on it.
        let ck = read_local(&[]).expect("no files is not an error");
        assert!(ck.shards.is_empty());
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn reads_an_hdf5_checkpoint() {
        let ck = read_local(&[fixture("tiny.hdf5")]).expect("the hdf5 fixture reads");
        let tensors: Vec<_> = ck.shards.iter().flat_map(|s| &s.tensors).collect();
        assert!(!tensors.is_empty(), "the hdf5 fixture has datasets");
        // HDF5 tracks chunked storage and compression; the model must carry both.
        assert!(
            tensors.iter().any(|t| !matches!(t.layout, Layout::None)),
            "an hdf5 tensor should report its layout"
        );
    }

    // --- the non-safetensors formats -----------------------------------------------
    //
    // GGUF, `.npy` and `.npz` are advertised as supported but had no fixture and no test:
    // the dispatch, the per-format header read and the `TensorInfo` each produces were
    // only ever exercised by opening a real file by hand. These build one of each on disk
    // and go in through `read_local`, the same way the CLI does.

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cs_readers_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("a scratch dir");
        dir
    }

    /// A `.npy` file: the v1.0 header for `shape`/`descr`, then `data` bytes.
    fn write_npy(path: &Path, descr: &str, shape: &str, data: &[u8]) {
        let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend((dict.len() as u16).to_le_bytes());
        bytes.extend(dict.as_bytes());
        bytes.extend(data);
        std::fs::write(path, bytes).expect("write the .npy");
    }

    #[test]
    fn reads_a_gguf_file_into_the_model() {
        use crate::gguf::testing::{Gguf, gguf_str};
        let dir = scratch("gguf");
        let path = dir.join("model.gguf");
        let mut g = Gguf::default();
        // Types 8 (string), 4 (u32) and 7 (bool) — three of the value kinds the reader
        // labels, so the `value_type` mapping is exercised, not just the happy path.
        g.kv("general.architecture", 8, &gguf_str("llama"))
            .kv("block_count", 4, &32u32.to_le_bytes())
            .kv("general.quantized", 7, &[1u8])
            // Q4_0 is block-quantized, so `stored_size` is block arithmetic rather than
            // elements × itemsize — the case an f32 product used to get wrong.
            .tensor("blk.0.attn_q.weight", &[64, 32], 2, 0)
            .tensor("output_norm.weight", &[64], 0, 4096)
            .u64(0); // padding byte the reader skips past
        std::fs::write(&path, g.finish()).expect("write the gguf");

        let ck = read_local(&[path]).expect("the gguf reads");
        let tensors: Vec<_> = ck.shards.iter().flat_map(|s| &s.tensors).collect();
        assert_eq!(tensors.len(), 2);
        let q = tensors
            .iter()
            .find(|t| t.name == "blk.0.attn_q.weight")
            .expect("the quantized tensor");
        assert_eq!(q.shape, vec![64, 32]);
        assert_eq!(q.num_elements, 2048);
        assert_eq!(q.dtype, "Q4_0");
        // 2048 elements at 32 per block, 18 bytes a block — exact, not a float product.
        assert_eq!(q.size_bytes, 2048 / 32 * 18);
        assert!(matches!(q.layout, Layout::Offset(0)));
        let norm = tensors
            .iter()
            .find(|t| t.name == "output_norm.weight")
            .expect("the f32 tensor");
        assert_eq!((norm.dtype.as_str(), norm.size_bytes), ("F32", 256));
        assert!(matches!(norm.layout, Layout::Offset(4096)));

        // The metadata comes through with each value's type named.
        let meta: Vec<_> = ck.shards.iter().flat_map(|s| &s.metadata).collect();
        let by = |k: &str| meta.iter().find(|m| m.name == k).expect(k);
        assert_eq!(by("general.architecture").value_type, "string");
        // Strings render quoted, so a value that merely *looks* numeric stays readable.
        assert_eq!(by("general.architecture").value, "\"llama\"");
        assert_eq!(by("block_count").value_type, "u32");
        assert_eq!(by("general.quantized").value_type, "bool");
    }

    #[test]
    fn reads_a_npy_file_into_a_single_tensor() {
        let dir = scratch("npy");
        let path = dir.join("weights.npy");
        write_npy(&path, "<f4", "(4, 5)", &[0u8; 80]);
        let ck = read_local(&[path]).expect("the .npy reads");
        let tensors: Vec<_> = ck.shards.iter().flat_map(|s| &s.tensors).collect();
        assert_eq!(tensors.len(), 1, "a .npy holds exactly one array");
        let t = tensors[0];
        // The tensor is named after the file, since a .npy carries no name of its own.
        assert_eq!(t.name, "weights");
        assert_eq!((t.dtype.as_str(), t.shape.clone()), ("F32", vec![4, 5]));
        assert_eq!((t.num_elements, t.size_bytes), (20, 80));
        // The byte range must start after the header, or the first values read as header.
        let Layout::ByteRange { start, end } = t.layout else {
            panic!("a .npy tensor is a byte range, got {:?}", t.layout);
        };
        assert_eq!(end - start, 80);
        assert!(start > 10, "data must start past the magic + header");
    }

    #[test]
    fn reads_a_npz_archive_reporting_compression_per_entry() {
        let dir = scratch("npz");
        let path = dir.join("bundle.npz");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).expect("create the npz"));
        let mut member = |file: &str, body: &[u8], deflate: bool| {
            let opts: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                .compression_method(if deflate {
                    zip::CompressionMethod::Deflated
                } else {
                    zip::CompressionMethod::Stored
                });
            zip.start_file(file.to_string(), opts).expect("entry");
            std::io::Write::write_all(&mut zip, body).expect("write the entry");
        };
        let npy = |descr: &str, shape: &str, data: &[u8]| {
            let dict =
                format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
            let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
            bytes.extend((dict.len() as u16).to_le_bytes());
            bytes.extend(dict.as_bytes());
            bytes.extend(data);
            bytes
        };
        member("stored.npy", &npy("<f4", "(2, 2)", &[0u8; 16]), false);
        member("squashed.npy", &npy("<f8", "(3,)", &[0u8; 24]), true);
        // A member that is not an array at all — numpy writes none, but archives get
        // edited, and a stray file must be skipped rather than parsed as a header.
        member("README.txt", b"not an array", false);
        zip.finish().expect("finish the npz");

        let ck = read_local(&[path]).expect("the .npz reads");
        let tensors: Vec<_> = ck.shards.iter().flat_map(|s| &s.tensors).collect();
        assert_eq!(
            tensors.len(),
            2,
            "one tensor per .npy member, and only those"
        );
        let stored = tensors.iter().find(|t| t.name == "stored").expect("stored");
        assert_eq!(
            (stored.dtype.as_str(), stored.shape.clone()),
            ("F32", vec![2, 2])
        );
        assert!(
            matches!(stored.storage, Storage::Raw),
            "an uncompressed member is Raw, got {:?}",
            stored.storage
        );
        let squashed = tensors
            .iter()
            .find(|t| t.name == "squashed")
            .expect("squashed");
        assert_eq!(squashed.dtype, "F64");
        // The compressed size is what the file actually costs, and it drives the
        // `A → B` display; it must be recorded, not inferred from the shape.
        let Storage::Compressed {
            codec,
            stored_bytes,
        } = &squashed.storage
        else {
            panic!(
                "a deflated member must report its codec, got {:?}",
                squashed.storage
            );
        };
        assert_eq!(codec, "deflate");
        assert!(*stored_bytes > 0 && *stored_bytes <= 24 + 64);
        // Data inside a zip isn't addressable by byte range, so reads go through the
        // archive instead — the layout says so rather than pointing somewhere wrong.
        assert!(matches!(squashed.layout, Layout::None));
    }

    /// The HDF5 sniff is what lets an extensionless checkpoint be recognized, and it must
    /// not claim anything else — including a file it can't read at all.
    #[test]
    fn the_hdf5_signature_sniff_only_matches_hdf5() {
        let dir = scratch("sniff");
        let h5 = dir.join("nameless");
        std::fs::write(&h5, b"\x89HDF\r\n\x1a\n and then whatever").expect("write");
        assert!(looks_like_hdf5(&h5));
        let st = dir.join("plain.safetensors");
        std::fs::write(&st, b"\x08\x00\x00\x00\x00\x00\x00\x00{}").expect("write");
        assert!(!looks_like_hdf5(&st));
        // Shorter than the signature, and absent entirely — both false, not a panic.
        let tiny = dir.join("tiny");
        std::fs::write(&tiny, b"\x89HDF").expect("write");
        assert!(!looks_like_hdf5(&tiny));
        assert!(!looks_like_hdf5(&dir.join("nope")));
    }
}
