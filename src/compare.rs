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
    // A leading `~` is expanded by `collect_safetensors_files` (one rule for every path
    // the program is handed); resolve it here too so an error can name both spellings.
    let paths = [crate::utils::expand_tilde(&path.to_string_lossy())];
    // `no_health_check = true`: a baseline's index/shard cross-check is not part of a
    // structural diff, and parsing it would only slow the read down.
    let (files, _) = crate::collect_safetensors_files(&paths, false, true)
        .with_context(|| format!("resolving {}", path.display()))?;
    if files.is_empty() {
        // Name the resolved path as well when it differs from what was typed (a `~` was
        // expanded): otherwise "no checkpoint files found at ~/ckpt" leaves the reader
        // unable to tell whether the tilde was understood.
        let resolved = paths[0].display().to_string();
        let typed = path.display().to_string();
        if resolved == typed {
            anyhow::bail!("no checkpoint files found at {typed}");
        }
        anyhow::bail!("no checkpoint files found at {typed} ({resolved})");
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

/// The `diff OLD NEW` command that reproduces a comparison — the batch equivalent a UI
/// offers so a finding can be re-run in a terminal (and extended with `--values`).
///
/// `open` is the *resolved* file list, which for a sharded checkpoint is every shard. The
/// command must name the checkpoint, not one of its shards: naming a shard silently
/// compares something else. [`crate::model::checkpoint_path`] answers that; `None` means
/// the files span directories, in which case no single path names them and there is no
/// honest one-line command to offer.
pub(crate) fn cli_diff_command(against: &Path, open: &[PathBuf]) -> Option<String> {
    let new = crate::model::checkpoint_path(open)?;
    Some(format!(
        "checkpoint-studio diff {} {}",
        shell_quote(&against.display().to_string()),
        shell_quote(&new.display().to_string())
    ))
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
    fn the_cli_command_puts_the_baseline_first() {
        // `diff OLD NEW`: the baseline is the first argument, the open checkpoint the
        // second — so pasting the command reproduces the screen's own report.
        let cmd = cli_diff_command(Path::new("/a/old"), &[PathBuf::from("/b/new")]).unwrap();
        assert_eq!(cmd, "checkpoint-studio diff /a/old /b/new");
        let quoted =
            cli_diff_command(Path::new("/a/needs quoting"), &[PathBuf::from("/b/new")]).unwrap();
        assert!(quoted.contains("'/a/needs quoting'"), "{quoted}");
    }

    /// The regression this function exists for. `open` is the *resolved* file list, so a
    /// sharded checkpoint arrives here as every shard — and naming the first one produced
    /// a command that compared a single shard against the baseline instead of the
    /// checkpoint. Reported from the web compare screen, which offered
    /// `diff OLD <ckpt>/codebooks.safetensors` for a directory of shards.
    #[test]
    fn the_cli_command_names_a_sharded_checkpoint_not_one_of_its_shards() {
        let shards = [
            PathBuf::from("/ckpt/codebooks.safetensors"),
            PathBuf::from("/ckpt/model-00001-of-00002.safetensors"),
            PathBuf::from("/ckpt/model-00002-of-00002.safetensors"),
        ];
        let cmd = cli_diff_command(Path::new("/base"), &shards).unwrap();
        assert_eq!(
            cmd, "checkpoint-studio diff /base /ckpt",
            "a sharded checkpoint is named by its directory"
        );
        assert!(
            !cmd.contains("codebooks"),
            "naming a shard would compare something else: {cmd}"
        );
    }

    /// Files spanning directories have no single name, so there is no honest one-line
    /// command — better to offer nothing than a command that means something else.
    #[test]
    fn there_is_no_command_when_the_files_span_directories() {
        let scattered = [
            PathBuf::from("/a/one.safetensors"),
            PathBuf::from("/b/two.safetensors"),
        ];
        assert!(cli_diff_command(Path::new("/base"), &scattered).is_none());
        assert!(cli_diff_command(Path::new("/base"), &[]).is_none());
    }
}
