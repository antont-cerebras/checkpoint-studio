//! Checkpoint health check: compare a `model.safetensors.index.json` against
//! the `.safetensors` files actually present, at both the file and tensor
//! level, and report any mismatch.

use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use crate::tree::TensorInfo;

/// What a [`HealthReport`] compared — the two are read differently and deserve
/// different wording, and only one of them can carry the s3 cross-check's findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthKind {
    /// A `model.safetensors.index.json` against the `.safetensors` files present.
    #[default]
    IndexVsFiles,
    /// An `s3://` checkpoint's index against what each object says about itself
    /// (see [`check_s3_correspondence`]).
    S3Correspondence,
}

/// The result of comparing an index against the files on disk.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthReport {
    /// Which comparison produced this report.
    #[serde(default)]
    pub kind: HealthKind,
    /// The index file that was checked.
    pub index_path: String,
    /// Files referenced by the index but absent on disk.
    pub missing_files: Vec<String>,
    /// `.safetensors` files present on disk but not referenced by the index.
    pub extra_files: Vec<String>,
    /// Tensors the index assigns to a present file that the file does not
    /// contain (formatted with the expected file).
    pub missing_tensors: Vec<String>,
    /// Tensors found in a referenced, present file that the index does not
    /// assign there (formatted with the containing file).
    pub extra_tensors: Vec<String>,
    /// Tensors whose two descriptions of themselves disagree — an `s3://` cstorch
    /// checkpoint records every tensor's dtype/shape in the checkpoint index *and*
    /// again in its own object's metadata (see [`check_s3_correspondence`]). Each
    /// entry names the tensor and both readings. Errors: one of the two is stale, so
    /// any read of that tensor is suspect.
    #[serde(default)]
    pub mismatched_tensors: Vec<String>,
    /// Tensors that could not be cross-checked (the object carries no usable metadata,
    /// or holds several tensors so a reading can't be attributed to one). Warnings —
    /// nothing is known to be wrong, but nothing was verified either, and a check that
    /// silently verifies nothing is worse than one that says so.
    #[serde(default)]
    pub unverified_tensors: Vec<String>,
}

impl HealthReport {
    #[must_use]
    pub fn has_issues(&self) -> bool {
        !self.missing_files.is_empty()
            || !self.extra_files.is_empty()
            || !self.missing_tensors.is_empty()
            || !self.extra_tensors.is_empty()
            || !self.mismatched_tensors.is_empty()
            || !self.unverified_tensors.is_empty()
    }

    /// Whether any issue is a real *error* (something the index references is
    /// absent) rather than a benign *warning* (something on disk the index doesn't
    /// mention). Matches the `check` report's severities: missing files/tensors are
    /// errors; extra files/tensors are warnings. Drives the health badge's colour.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.missing_files.is_empty()
            || !self.missing_tensors.is_empty()
            || !self.mismatched_tensors.is_empty()
    }
}

/// A checkpoint directory's index parsed once, ready to health-check against the
/// tensors the loader parses — so the shard headers are read a single time (by the
/// loader) rather than again here. Built by [`parse_index_spec`]; consumed by
/// [`check_loaded`].
pub struct IndexSpec {
    /// The directory the index and its shards live in.
    pub dir: PathBuf,
    /// The `model.safetensors.index.json` path (for the report label).
    pub index_path: PathBuf,
    /// tensor name -> file the index claims it lives in.
    pub weight_map: HashMap<String, String>,
}

/// Read and parse a `model.safetensors.index.json` once into an [`IndexSpec`].
pub fn parse_index_spec(dir: &Path, index_path: &Path) -> Result<IndexSpec> {
    Ok(IndexSpec {
        dir: dir.to_path_buf(),
        index_path: index_path.to_path_buf(),
        weight_map: parse_weight_map(index_path)?,
    })
}

/// The files these reports found on disk but absent from the index, as **absolute
/// paths** — so they compare directly against each [`TensorInfo::source_path`], which
/// is how both frontends mark an unindexed tensor.
///
/// In core because every frontend needs the same set from the same reports: the
/// terminal marks its tree rows from it and the web server sends it to the browser for
/// the same purpose. Two derivations of "which files are extras" would be two chances
/// to disagree about it.
#[must_use]
pub fn unindexed_files(reports: &[HealthReport]) -> std::collections::HashSet<String> {
    let mut unindexed = std::collections::HashSet::new();
    for report in reports {
        if let Some(dir) = Path::new(&report.index_path).parent() {
            for file in &report.extra_files {
                let path = dir.join(file);
                unindexed.insert(
                    std::path::absolute(&path)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    unindexed
}

/// Compare an index against the checkpoint as actually loaded: the `.safetensors`
/// files on disk (a directory listing — no header reads) and the tensor names the
/// loader already parsed from each shard's header. `tensors` is the whole loaded
/// set; only those whose `source_path` is a file directly in `spec.dir` count, so
/// this is safe when several checkpoints are loaded together.
#[must_use]
pub fn check_loaded(spec: &IndexSpec, tensors: &[TensorInfo]) -> HealthReport {
    let actual = list_safetensors(&spec.dir);

    // Tensor names present per file, taken from the already-parsed tensors (grouped
    // by the file name of their `source_path`) — keyed to `spec.dir` so a same-named
    // shard in another loaded directory can't leak in.
    let abs_dir = std::path::absolute(&spec.dir).unwrap_or_else(|_| spec.dir.clone());
    let mut present_by_file: HashMap<String, BTreeSet<String>> = HashMap::new();
    for t in tensors {
        let path = Path::new(&t.source_path);
        let in_dir = path.parent().is_some_and(|p| p == abs_dir || p == spec.dir);
        if !in_dir {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            present_by_file
                .entry(name.to_string())
                .or_default()
                .insert(t.name.clone());
        }
    }

    reconcile(
        &spec.index_path.display().to_string(),
        &spec.weight_map,
        &actual,
        &present_by_file,
    )
}

/// The pure index-vs-checkpoint comparison shared by the local
/// ([`check_loaded`]) and remote (`--ssh-proxy`) health checks: given the index's
/// `weight_map` (tensor -> file), the `.safetensors` files actually present, and
/// the tensor names present in each file (from the already-parsed headers), report
/// the file- and tensor-level mismatches. No I/O — both callers supply the pieces
/// from data they've already read, so a header is never read twice.
pub fn reconcile<S: std::hash::BuildHasher>(
    index_path: &str,
    weight_map: &HashMap<String, String, S>,
    actual: &BTreeSet<String>,
    present_by_file: &HashMap<String, BTreeSet<String>, S>,
) -> HealthReport {
    let referenced: BTreeSet<String> = weight_map.values().cloned().collect();

    // File-level diff.
    let missing_files: Vec<String> = referenced.difference(actual).cloned().collect();
    let extra_files: Vec<String> = actual.difference(&referenced).cloned().collect();

    // Tensor-level diff, limited to files that are both referenced and present
    // (wholesale-missing / wholesale-extra files are already covered above).
    let mut claimed_by_file: HashMap<String, BTreeSet<String>> = HashMap::new();
    for (tensor, file) in weight_map {
        claimed_by_file
            .entry(file.clone())
            .or_default()
            .insert(tensor.clone());
    }

    let mut missing_tensors = Vec::new();
    let mut extra_tensors = Vec::new();
    for file in referenced.intersection(actual) {
        let present = present_by_file.get(file).cloned().unwrap_or_default();
        let claimed = claimed_by_file.get(file).cloned().unwrap_or_default();
        for tensor in claimed.difference(&present) {
            missing_tensors.push(format!("{tensor}  (expected in {file})"));
        }
        for tensor in present.difference(&claimed) {
            extra_tensors.push(format!("{tensor}  (in {file})"));
        }
    }
    missing_tensors.sort();
    extra_tensors.sort();

    HealthReport {
        kind: HealthKind::IndexVsFiles,
        index_path: index_path.to_string(),
        missing_files,
        extra_files,
        missing_tensors,
        extra_tensors,
        // The correspondence check is s3-only (see `check_s3_correspondence`); a
        // local index carries no second description of a tensor to compare against.
        mismatched_tensors: Vec::new(),
        unverified_tensors: Vec::new(),
    }
}

/// Cross-check an `s3://` cstorch checkpoint's index against what each object says
/// about *itself*.
///
/// cstorch writes every tensor's dtype and shape twice: once into the checkpoint's
/// single `__METADATA__` index (which is what a load reads) and again into the
/// tensor's own object as `x-amz-meta-metadata` (which is what the storage layer
/// sees). We already fetch both — the object metadata comes back with the sizes and
/// `ETags` the stats screen shows — so agreement is free to verify, and disagreement
/// means one of the two is stale: the index would then describe a tensor that isn't
/// what the bytes actually are.
///
/// Three things are compared, all from data already in memory (no extra requests):
///
/// 1. dtype and shape, index versus the object's own header.
/// 2. the object's byte size on S3 versus the size its own header declares — a
///    truncated or partially-overwritten upload, which matching dtypes can't reveal.
/// 3. presence in both directions: a tensor with no object, an object with no tensor.
///
/// Inapplicable unless the checkpoint stores one object per tensor keyed by the
/// tensor's name (what cstorch does today). If no tensor name matches an object key
/// the layout is something else, and an empty report is returned rather than 1155
/// bogus "missing" findings.
pub fn check_s3_correspondence(
    uri: &str,
    tensors: &[TensorInfo],
    s3: &crate::remote::S3Meta,
) -> HealthReport {
    /// What one object's `x-amz-meta-metadata` says about the tensor it holds.
    struct ObjectClaim {
        dtype: String,
        shape: Vec<usize>,
        /// Sum of the declared per-tensor data sizes, when stated uncompressed.
        data_bytes: Option<u64>,
    }

    /// Parse an object's `x-amz-meta-metadata` JSON. `None` when it's absent,
    /// unparseable, or describes several tensors (then a reading can't be attributed
    /// to the one tensor keyed by this object).
    fn claim_of(o: &crate::remote::S3Object) -> Option<ObjectClaim> {
        let raw = o
            .user_meta
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("metadata"))
            .map(|(_, v)| v.as_str())?;
        let v: serde_json::Value = serde_json::from_str(raw).ok()?;
        let shapes = v.get("shapes")?.as_array()?;
        let dtypes = v.get("dtypes")?.as_array()?;
        if shapes.len() != 1 || dtypes.len() != 1 {
            return None; // a packed object: several tensors behind one key
        }
        // Exactly one entry each, checked just above.
        let shape: Vec<usize> = shapes
            .first()?
            .as_array()?
            .iter()
            .map(|d| d.as_u64().map(|n| n as usize))
            .collect::<Option<_>>()?;
        // Only trust the declared size when the object says it is stored
        // uncompressed; otherwise the stored bytes legitimately differ. `None` here
        // skips just the size comparison — the dtype and shape are still checked.
        let uncompressed = v.get("compressed").and_then(serde_json::Value::as_bool) == Some(false);
        let data_bytes = if uncompressed {
            v.get("data_sizes")
                .and_then(serde_json::Value::as_array)
                .map(|a| a.iter().filter_map(serde_json::Value::as_u64).sum())
        } else {
            None
        };
        Some(ObjectClaim {
            // The object states the raw torch name (`torch.float16`); a tensor's dtype
            // is the display form (`F16`). Compare like with like.
            dtype: crate::remote::map_dtype(dtypes.first()?.as_str()?),
            shape,
            data_bytes,
        })
    }

    let by_key: HashMap<&str, &crate::remote::S3Object> =
        s3.objects.iter().map(|o| (o.key.as_str(), o)).collect();
    // One object per tensor, keyed by name? If not, this check doesn't apply.
    if !tensors.iter().any(|t| by_key.contains_key(t.name.as_str())) {
        return HealthReport {
            kind: HealthKind::S3Correspondence,
            index_path: format!("{uri}/__METADATA__"),
            missing_files: Vec::new(),
            extra_files: Vec::new(),
            missing_tensors: Vec::new(),
            extra_tensors: Vec::new(),
            mismatched_tensors: Vec::new(),
            unverified_tensors: Vec::new(),
        };
    }

    let mut missing_files = Vec::new();
    let mut mismatched_tensors = Vec::new();
    let mut unverified_tensors = Vec::new();
    for t in tensors {
        let Some(o) = by_key.get(t.name.as_str()) else {
            missing_files.push(format!("{}  (no S3 object)", t.name));
            continue;
        };
        let Some(claim) = claim_of(o) else {
            unverified_tensors.push(format!(
                "{}  (the object states no single dtype/shape)",
                t.name
            ));
            continue;
        };
        if claim.dtype != t.dtype || claim.shape != t.shape {
            mismatched_tensors.push(format!(
                "{}  (index: {} {}, object: {} {})",
                t.name,
                t.dtype,
                crate::utils::format_shape(&t.shape),
                claim.dtype,
                crate::utils::format_shape(&claim.shape),
            ));
            continue; // the size comparison below would just restate the same problem
        }
        if let Some(declared) = claim.data_bytes
            && declared != o.size
        {
            mismatched_tensors.push(format!(
                "{}  (object is {}, its own metadata declares {})",
                t.name,
                crate::utils::format_size(o.size as usize),
                crate::utils::format_size(declared as usize),
            ));
        }
    }

    // Objects with no tensor behind them. `__METADATA__` is the index itself, not a
    // tensor, so it is expected; anything else is a leftover from another write.
    let named: BTreeSet<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    let mut extra_files: Vec<String> = s3
        .objects
        .iter()
        .map(|o| o.key.as_str())
        .filter(|k| *k != "__METADATA__" && !named.contains(k))
        .map(str::to_string)
        .collect();
    extra_files.sort();
    missing_files.sort();
    mismatched_tensors.sort();
    unverified_tensors.sort();

    HealthReport {
        kind: HealthKind::S3Correspondence,
        index_path: format!("{uri}/__METADATA__"),
        missing_files,
        extra_files,
        // Tensor-in-file placement is a safetensors-index notion; an s3 checkpoint's
        // objects are keyed by tensor name, so those two lists have no meaning here.
        missing_tensors: Vec::new(),
        extra_tensors: Vec::new(),
        mismatched_tensors,
        unverified_tensors,
    }
}

/// Parse the `weight_map` of an index into a tensor-name -> file-name map.
fn parse_weight_map(index_path: &Path) -> Result<HashMap<String, String>> {
    let content = std::fs::read_to_string(index_path)
        .with_context(|| format!("Failed to read index file: {}", index_path.display()))?;
    let index: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse index file: {}", index_path.display()))?;

    let mut map = HashMap::new();
    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
        for (tensor, file) in weight_map {
            if let Some(file) = file.as_str() {
                map.insert(tensor.clone(), file.to_string());
            }
        }
    }
    Ok(map)
}

/// The set of `.safetensors` file names directly inside `dir`.
fn list_safetensors(dir: &Path) -> BTreeSet<String> {
    let mut files = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
            {
                files.insert(name.to_string());
            }
        }
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Layout, Storage};

    // ---- the s3 index-vs-object cross-check -------------------------------
    //
    // Built from the real shape of a cstorch checkpoint: one object per tensor keyed
    // by the tensor's name, whose `x-amz-meta-metadata` header repeats the dtype and
    // shape the index already gave, plus the byte size it stores.

    fn s3_object(key: &str, size: u64, meta: Option<&str>) -> crate::remote::S3Object {
        let mut user_meta = std::collections::BTreeMap::new();
        if let Some(m) = meta {
            user_meta.insert("metadata".to_string(), m.to_string());
        }
        crate::remote::S3Object {
            key: key.into(),
            size,
            etag: "e".into(),
            checksum: None,
            last_modified: String::new(),
            user_meta,
            tags: None,
        }
    }

    /// The header cstorch actually writes (trimmed to the fields we read).
    fn claim(dtype: &str, shape: &[usize], data_size: u64, compressed: bool) -> String {
        format!(
            r#"{{"__TORCH__": true, "shapes": [{shape:?}], "dtypes": ["{dtype}"], "compressed": {compressed}, "data_sizes": [{data_size}], "boundaries": [{data_size}]}}"#
        )
    }

    fn tensor(name: &str, dtype: &str, shape: &[usize], bytes: usize) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype: dtype.into(),
            shape: shape.to_vec(),
            size_bytes: bytes,
            num_elements: shape.iter().product(),
            storage: Storage::Unknown,
            source_path: "s3://b/k".into(),
            layout: Layout::None,
        }
    }

    fn s3(objects: Vec<crate::remote::S3Object>) -> crate::remote::S3Meta {
        crate::remote::S3Meta {
            objects,
            warnings: Vec::new(),
        }
    }

    /// The two sources spell dtypes differently — a tensor carries the display form
    /// (`F16`), the object states the raw torch name (`torch.float16`) — so the check
    /// has to normalise before comparing. Every fixture here uses that real pairing:
    /// without the mapping this reports a mismatch for every tensor, which is exactly
    /// what it did against the live checkpoint before the fix.
    #[test]
    fn s3_correspondence_is_silent_when_both_descriptions_agree() {
        let tensors = vec![tensor("a.weight", "F16", &[4, 8], 64)];
        let meta = s3(vec![
            s3_object("__METADATA__", 1234, None),
            s3_object(
                "a.weight",
                64,
                Some(&claim("torch.float16", &[4, 8], 64, false)),
            ),
        ]);
        let r = check_s3_correspondence("s3://b/k", &tensors, &meta);
        assert!(!r.has_issues(), "{r:?}");
        assert_eq!(r.index_path, "s3://b/k/__METADATA__");
    }

    #[test]
    fn s3_correspondence_catches_a_dtype_or_shape_that_drifted() {
        let tensors = vec![
            tensor("a.weight", "torch.float16", &[4, 8], 64),
            tensor("b.weight", "torch.float16", &[4, 8], 64),
        ];
        let meta = s3(vec![
            // dtype differs …
            s3_object(
                "a.weight",
                64,
                Some(&claim("torch.bfloat16", &[4, 8], 64, false)),
            ),
            // … and shape differs.
            s3_object(
                "b.weight",
                64,
                Some(&claim("torch.float16", &[8, 4], 64, false)),
            ),
        ]);
        let r = check_s3_correspondence("s3://b/k", &tensors, &meta);
        assert_eq!(r.mismatched_tensors.len(), 2, "{r:?}");
        assert!(r.mismatched_tensors[0].contains("BF16"));
        assert!(r.mismatched_tensors[1].contains("(8, 4)"));
        // A disagreement about what a tensor *is* is an error, not a warning.
        assert!(r.has_errors());
    }

    #[test]
    fn s3_correspondence_catches_a_truncated_object() {
        // The header says it stores 64 bytes; the object holds 32 — a partial upload
        // that agreeing dtypes and shapes would never reveal.
        let tensors = vec![tensor("a.weight", "F16", &[4, 8], 64)];
        let meta = s3(vec![s3_object(
            "a.weight",
            32,
            Some(&claim("torch.float16", &[4, 8], 64, false)),
        )]);
        let r = check_s3_correspondence("s3://b/k", &tensors, &meta);
        assert_eq!(r.mismatched_tensors.len(), 1, "{r:?}");
        assert!(r.mismatched_tensors[0].contains("declares"));
        assert!(r.has_errors());
    }

    #[test]
    fn s3_correspondence_leaves_a_compressed_object_size_alone() {
        // Stored compressed, so fewer bytes than the declared data size is correct.
        let tensors = vec![tensor("a.weight", "F16", &[4, 8], 64)];
        let meta = s3(vec![s3_object(
            "a.weight",
            20,
            Some(&claim("torch.float16", &[4, 8], 64, true)),
        )]);
        assert!(!check_s3_correspondence("s3://b/k", &tensors, &meta).has_issues());
    }

    #[test]
    fn s3_correspondence_reports_presence_in_both_directions() {
        let tensors = vec![
            tensor("a.weight", "F16", &[4, 8], 64),
            tensor("gone.weight", "F16", &[4, 8], 64),
        ];
        let meta = s3(vec![
            s3_object(
                "a.weight",
                64,
                Some(&claim("torch.float16", &[4, 8], 64, false)),
            ),
            s3_object("__METADATA__", 99, None),
            s3_object("leftover.weight", 64, None),
        ]);
        let r = check_s3_correspondence("s3://b/k", &tensors, &meta);
        assert_eq!(r.missing_files, vec!["gone.weight  (no S3 object)"]);
        assert_eq!(r.extra_files, vec!["leftover.weight"]); // __METADATA__ is expected
        assert!(r.has_errors()); // a tensor with no object is an error
    }

    #[test]
    fn s3_correspondence_says_so_when_it_could_not_verify() {
        let tensors = vec![
            tensor("a.weight", "F16", &[4, 8], 64),
            tensor("packed.weight", "F16", &[4, 8], 64),
        ];
        let meta = s3(vec![
            // No metadata header at all …
            s3_object("a.weight", 64, None),
            // … and one object standing for several tensors, so no reading can be
            // attributed to this one.
            s3_object(
                "packed.weight",
                128,
                Some(
                    r#"{"shapes": [[4, 8], [4, 8]], "dtypes": ["torch.float16", "torch.float16"], "compressed": false, "data_sizes": [64, 64]}"#,
                ),
            ),
        ]);
        let r = check_s3_correspondence("s3://b/k", &tensors, &meta);
        assert_eq!(r.unverified_tensors.len(), 2, "{r:?}");
        assert!(
            r.has_issues(),
            "an unverified check must not read as verified"
        );
        assert!(!r.has_errors(), "but nothing is known to be wrong either");
    }

    #[test]
    fn s3_correspondence_does_not_apply_to_another_layout() {
        // Keys that aren't tensor names (e.g. sharded blobs): comparing would produce
        // one bogus "missing" per tensor, so the check stands down instead.
        let tensors = vec![tensor("a.weight", "F16", &[4, 8], 64)];
        let meta = s3(vec![s3_object("shard-00000-of-00002.bin", 64, None)]);
        let r = check_s3_correspondence("s3://b/k", &tensors, &meta);
        assert!(!r.has_issues(), "{r:?}");
    }

    #[test]
    fn has_errors_only_for_missing_not_extra() {
        let report = |missing_f: &[&str], extra_f: &[&str], missing_t: &[&str]| HealthReport {
            kind: HealthKind::IndexVsFiles,
            index_path: "idx".into(),
            missing_files: missing_f.iter().map(ToString::to_string).collect(),
            extra_files: extra_f.iter().map(ToString::to_string).collect(),
            missing_tensors: missing_t.iter().map(ToString::to_string).collect(),
            extra_tensors: Vec::new(),
            mismatched_tensors: Vec::new(),
            unverified_tensors: Vec::new(),
        };
        // Extra files on disk are only a warning (benign — e.g. codebooks/qscales).
        assert!(!report(&[], &["codebooks.safetensors"], &[]).has_errors());
        // A referenced file or tensor that's missing on disk is a real error.
        assert!(report(&["model-00007.safetensors"], &[], &[]).has_errors());
        assert!(report(&[], &[], &["a.weight  (expected in x)"]).has_errors());
    }

    #[test]
    fn unindexed_files_resolves_against_the_index_directory() {
        // The paths have to come out as `source_path` records them — absolute, in the
        // index's own directory — because both UIs mark a row by comparing the two
        // strings. A bare `extra_files` basename would match nothing and the mark
        // would silently never appear.
        let report = HealthReport {
            kind: HealthKind::IndexVsFiles,
            index_path: "/ckpt/model.safetensors.index.json".into(),
            missing_files: Vec::new(),
            extra_files: vec!["codebooks.safetensors".into(), "qscales.safetensors".into()],
            missing_tensors: Vec::new(),
            extra_tensors: Vec::new(),
            mismatched_tensors: Vec::new(),
            unverified_tensors: Vec::new(),
        };
        let set = unindexed_files(std::slice::from_ref(&report));
        assert!(set.contains("/ckpt/codebooks.safetensors"), "{set:?}");
        assert!(set.contains("/ckpt/qscales.safetensors"), "{set:?}");
        assert_eq!(set.len(), 2);

        // Nothing extra, nothing marked — and a report with no directory in its index
        // path doesn't panic on the way to that answer.
        assert!(unindexed_files(&[]).is_empty());
        assert!(
            unindexed_files(&[HealthReport {
                index_path: "idx.json".into(),
                extra_files: vec!["x.safetensors".into()],
                ..report
            }])
            .iter()
            .all(|p| p.ends_with("x.safetensors")),
            "a relative index path still resolves"
        );
    }

    /// A `TensorInfo` named `name` whose `source_path` is `dir/file` (absolute,
    /// matching what the loader records) — enough for the health check's grouping.
    fn ti(name: &str, dir: &Path, file: &str) -> TensorInfo {
        let src = std::path::absolute(dir.join(file))
            .unwrap()
            .to_string_lossy()
            .into_owned();
        TensorInfo {
            name: name.to_string(),
            dtype: "F32".into(),
            shape: vec![1],
            size_bytes: 4,
            num_elements: 1,
            storage: Storage::Unknown,
            source_path: src,
            layout: Layout::None,
        }
    }

    #[test]
    fn detects_file_and_tensor_mismatches() {
        let dir = std::env::temp_dir().join("checkpoint_studio_health_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Index references model-00001 (present) and model-00002 (missing).
        let index = serde_json::json!({
            "weight_map": {
                "a.weight": "model-00001.safetensors",
                "b.weight": "model-00002.safetensors"
            }
        });
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();

        // On disk: model-00001 (present) and model-00003 (never referenced); the
        // file-level check only lists directory entries, so empty files suffice.
        std::fs::write(dir.join("model-00001.safetensors"), b"").unwrap();
        std::fs::write(dir.join("model-00003.safetensors"), b"").unwrap();

        // The tensors the loader parsed: model-00001 holds the claimed a.weight
        // plus an unlisted extra; model-00003 holds c.weight.
        let tensors = vec![
            ti("a.weight", &dir, "model-00001.safetensors"),
            ti("extra.weight", &dir, "model-00001.safetensors"),
            ti("c.weight", &dir, "model-00003.safetensors"),
        ];

        let spec = parse_index_spec(&dir, &index_path).unwrap();
        let report = check_loaded(&spec, &tensors);
        assert!(report.has_issues());
        assert_eq!(report.missing_files, vec!["model-00002.safetensors"]);
        assert_eq!(report.extra_files, vec!["model-00003.safetensors"]);
        // `extra.weight` lives in a referenced+present file but the index does
        // not list it there.
        assert!(
            report
                .extra_tensors
                .iter()
                .any(|t| t.starts_with("extra.weight"))
        );
        // `a.weight` matches; `b.weight`'s file is missing (covered by the file
        // diff), so there are no tensor-level "missing" entries.
        assert!(report.missing_tensors.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
