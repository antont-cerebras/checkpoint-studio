//! Loading the *other* side of a structural comparison — shared by the TUI's compare
//! screen and the web's `/api/diff`, so the two interactive surfaces and the `diff`
//! subcommand all answer with the same [`DiffReport`].
//!
//! **Structure only.** This reads shard headers (names, dtypes, shapes, metadata) and
//! nothing else: no tensor bytes, so comparing two multi-GB checkpoints is as fast as
//! opening one. Value comparison (`diff --values`) and repack verification
//! (`diff --verify-repack`) stay on the CLI, where a long scan has a progress bar and a
//! place to report per-tensor findings — see the README.
//!
//! **Which side is which.** The checkpoint you have open is the **new** side and
//! `against` is the **baseline (old)**, so the report reads as "what changed relative to
//! the thing I'm comparing with". That matches the reopen command each screen emits
//! (`checkpoint-studio diff AGAINST OPEN`), which is what lets a differential test
//! assert the two interactive surfaces and the CLI produce the same report.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::diff::{CheckpointSummary, DiffReport};

/// Reduce the checkpoint at `path` to its comparable structure — the baseline side of a
/// diff. `path` is a file, a directory of shards, or a glob, resolved exactly the way
/// the CLI resolves its own arguments (so `diff` and the interactive screens accept the
/// same spellings).
pub(crate) fn summarize(path: &Path) -> Result<CheckpointSummary> {
    let paths = [path.to_path_buf()];
    // `no_health_check = true`: a baseline's index/shard cross-check is not part of a
    // structural diff, and parsing it would only slow the read down.
    let (files, _) = crate::collect_safetensors_files(&paths, false, true)
        .with_context(|| format!("resolving {}", path.display()))?;
    if files.is_empty() {
        anyhow::bail!("no checkpoint files found at {}", path.display());
    }
    let model = crate::readers::read_local(&files)
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(CheckpointSummary::from_loaded(
        &model.tensors_vec(),
        &model.metadata_vec(),
    ))
}

/// The structural diff of the open checkpoint (`tensors` / `metadata`, the **new** side)
/// against the checkpoint at `against` (the **old** side).
pub(crate) fn structural_diff(
    tensors: &[crate::tree::TensorInfo],
    metadata: &[crate::tree::MetadataInfo],
    against: &Path,
) -> Result<DiffReport> {
    let old = summarize(against)?;
    let new = CheckpointSummary::from_loaded(tensors, metadata);
    Ok(crate::diff::compare(&old, &new))
}

/// A one-line verdict for a screen's header: what the report found, or that the two are
/// structurally identical. Shared so the terminal and the browser say the same thing.
pub(crate) fn verdict(report: &DiffReport) -> String {
    if !report.has_differences_with(true) {
        return "structurally identical".to_string();
    }
    let mut parts = Vec::new();
    for (n, what) in [
        (report.tensors_added.len(), "added"),
        (report.tensors_removed.len(), "removed"),
        (report.tensors_changed.len(), "changed"),
    ] {
        if n > 0 {
            parts.push(format!("{n} {what}"));
        }
    }
    let meta = report.meta_added.len() + report.meta_removed.len() + report.meta_changed.len();
    if meta > 0 {
        parts.push(format!(
            "{meta} metadata {}",
            if meta == 1 { "change" } else { "changes" }
        ));
    }
    if parts.is_empty() {
        // `has_differences_with` was true, so something differs that isn't in the counts above
        // (currently only the S3 object metadata of an s3-vs-s3 diff).
        return "differs".to_string();
    }
    format!("{} tensors: {}", report.tensors_unchanged, parts.join(", "))
        .replace("tensors: ", "unchanged; ")
}

/// The path a comparison names, as the reopen (`y`) command spells it.
pub(crate) fn reopen_command(against: &Path, open: &[PathBuf]) -> String {
    let open = open
        .first()
        .map_or_else(|| ".".to_string(), |p| p.display().to_string());
    format!(
        "checkpoint-studio diff {} {}",
        shell_quote(&against.display().to_string()),
        shell_quote(&open)
    )
}

/// Single-quote a path for a copyable shell command when it needs it.
fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-/=:".contains(&b))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', r"'\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn a_checkpoint_compared_with_itself_is_identical() {
        let path = fixture("diff_old.safetensors");
        let sum = summarize(&path).expect("the fixture summarizes");
        let report = crate::diff::compare(&sum, &sum);
        assert!(
            !report.has_differences_with(true),
            "a checkpoint differs from itself"
        );
        assert_eq!(verdict(&report), "structurally identical");
    }

    #[test]
    fn the_two_diff_fixtures_differ_and_the_verdict_says_how() {
        let old = summarize(&fixture("diff_old.safetensors")).unwrap();
        let new = summarize(&fixture("diff_new.safetensors")).unwrap();
        let report = crate::diff::compare(&old, &new);
        assert!(
            report.has_differences_with(true),
            "the diff fixtures should differ"
        );
        let v = verdict(&report);
        assert!(
            v.contains("unchanged"),
            "the verdict should count the unchanged tensors: {v}"
        );
    }

    #[test]
    fn a_missing_baseline_is_an_error_naming_the_path() {
        let Err(err) = summarize(Path::new("/nonexistent/checkpoint")) else {
            panic!("reading a nonexistent path should fail");
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("/nonexistent/checkpoint"),
            "the error should name the path it could not read: {msg}"
        );
    }

    #[test]
    fn the_reopen_command_puts_the_baseline_first() {
        // `diff OLD NEW`: the baseline is the first argument, the open checkpoint the
        // second — so pasting the command reproduces the screen's own report.
        let cmd = reopen_command(Path::new("/a/old"), &[PathBuf::from("/b/new")]);
        assert_eq!(cmd, "checkpoint-studio diff /a/old /b/new");
        let quoted = reopen_command(Path::new("/a/needs quoting"), &[PathBuf::from("/b/new")]);
        assert!(quoted.contains("'/a/needs quoting'"), "{quoted}");
    }
}
