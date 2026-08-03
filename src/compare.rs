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

/// Reduce the checkpoint named by `spec` to its comparable structure — the baseline side of a diff.
///
/// **One resolver, deliberately.** `spec` goes through [`crate::opening::resolve`], which is the same
/// path `/api/open` and `/api/compare` take, so every surface that compares two checkpoints accepts
/// the same addresses: a file, a directory of shards, a glob, `hf://`, `s3://`, `[user@]host:/path`
/// and the `:PATH` proxy shorthand. The diff report used to resolve its own side with
/// [`crate::collect_safetensors_files`] over a bare `Path`, which meant it rejected — as *"no
/// checkpoint files found"* — every remote address the comparison beside it resolved happily.
///
/// [`Want::Parts`](crate::opening::Want::Parts): a structural summary needs names, dtypes, shapes and
/// metadata, not an assembled model or per-object sizes, and skipping those is what keeps comparing
/// two multi-GB checkpoints as fast as opening one.
pub(crate) fn summarize_spec(
    spec: &str,
    opts: &crate::opening::Options,
) -> Result<CheckpointSummary> {
    let opened = crate::opening::resolve(spec, opts)
        .with_context(|| format!("resolving {spec}"))?
        .read(
            crate::opening::Want::Parts,
            &crate::hf::ReadProgress::default(),
        )
        .with_context(|| format!("reading {spec}"))?;
    let (tensors, metadata) = (&opened.parts.tensors, &opened.parts.metadata);
    Ok(CheckpointSummary::from_loaded(tensors, metadata))
}

/// A comparison's baseline: its comparable structure, plus the S3 object metadata when the source
/// carries any.
pub(crate) struct Baseline {
    pub summary: CheckpointSummary,
    /// Per-object `ETag` / size / checksums / tags, for an `s3://` baseline — the input to
    /// [`crate::diff::compare_s3`]. `None` for every other source, which has no such thing.
    pub s3: Option<crate::remote::S3Meta>,
}

// `summarize_baseline` used to live here: the report route's own read of its baseline, with
// `Want::Model` so an `s3://` side arrived carrying its per-object metadata. Both diff views read the
// pair from the comparison slot now (`Current::read_side`, which asks for `Want::Model` for the same
// reason), so a report costs no read at all — and the two views cannot describe different pairs.

/// Compare the two sides' S3 object metadata, attach it to the report, and say what happened.
///
/// Only an s3-vs-s3 pair has it on both sides, so that is the only pair compared.
///
/// The returned sentence is a note **about the comparison**, not a result of it: it says what was
/// compared, or why something was not. The CLI prints it on stderr and the browser sets it apart from
/// the counts (`DiffView`'s `.method`), because among them it read as a finding. Shared so neither
/// surface can quietly skip the comparison the other performs.
///
/// Timestamp-like deltas are informational and never a difference — see [`crate::diff::compare_s3`].
pub(crate) fn attach_s3(
    report: &mut DiffReport,
    old: Option<&crate::remote::S3Meta>,
    new: Option<&crate::remote::S3Meta>,
) -> Option<String> {
    match (old, new) {
        (Some(o), Some(n)) => {
            let count = o.objects.len().max(n.objects.len());
            // Each checkpoint's last-modified = the newest object under its prefix
            // (ISO-8601 UTC strings sort chronologically), shown in the summary.
            let latest = |m: &crate::remote::S3Meta| {
                m.objects
                    .iter()
                    .map(|x| x.last_modified.clone())
                    .filter(|s| !s.is_empty())
                    .max()
            };
            report.old_modified = latest(o);
            report.new_modified = latest(n);
            report.s3 = Some(crate::diff::compare_s3(o, n));
            Some(format!("compared {count} S3 object(s)' metadata"))
        }
        (Some(_), None) | (None, Some(_)) => Some(
            "S3 object metadata is compared only when both sides are s3:// — one of these is not, \
             so this comparison is of structure alone."
                .to_string(),
        ),
        (None, None) => None,
    }
}

/// Reduce the checkpoint at `path` to its comparable structure — the local-path entry point, kept
/// for the `diff` subcommand, which resolves its own arguments before it gets here.
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
/// against the checkpoint named by `against` (the **old** side).
///
/// Takes a spec rather than a path — see [`summarize_spec`] for why that matters.
pub(crate) fn structural_diff(
    tensors: &[crate::tree::TensorInfo],
    metadata: &[crate::tree::MetadataInfo],
    against: &str,
    opts: &crate::opening::Options,
) -> Result<DiffReport> {
    let old = summarize_spec(against, opts)?;
    let new = CheckpointSummary::from_loaded(tensors, metadata);
    Ok(crate::diff::compare(&old, &new))
}

/// Everything a `diff` can be scoped by, built from **text** rather than from files.
///
/// The CLI reads `--names-from` and `--map-from` off disk; the browser posts the same content pasted
/// into a box. Both then call the functions below, so "which tensors does this comparison cover" has
/// one implementation — the point of the exercise. Two surfaces each parsing globs their own way is
/// how they end up scoping differently and reporting different diffs of the same pair.
pub(crate) struct ScopeText<'a> {
    /// `--name`: globs, `!`-prefixed to exclude, a tensor passing if it matches ANY.
    pub name: &'a [String],
    /// `--names`: exact names, comma-separated.
    pub names_csv: Option<&'a str>,
    /// The *content* of `--names-from`: one exact name per line; blank lines and `#` comments ignored.
    pub names_lines: Option<&'a str>,
    /// `--dtype-is`: a glob against the uppercased dtype.
    pub dtype_is: Option<&'a str>,
    /// `--shape-is`: a glob over comma/`x`-separated dims.
    pub shape_is: Option<&'a str>,
}

/// Compile a [`crate::diff::TensorFilter`] from those inputs.
///
/// `--shape-is` dims are joined with `/` so the glob's `*`/`**` act per dimension. A bad glob is an
/// error rather than a filter that silently matches nothing.
pub(crate) fn tensor_filter(text: &ScopeText<'_>) -> Result<crate::diff::TensorFilter> {
    use glob::Pattern;
    use std::collections::HashSet;

    let names = crate::filter::NameFilter::parse(text.name)?;

    let mut exact: HashSet<String> = HashSet::new();
    if let Some(list) = text.names_csv {
        // Commas **or** newlines. `--names` is documented as a comma list and stays one, but the same
        // field in the browser is a box you paste into, and a pasted column of names is the commonest
        // thing to put in it. A tensor name contains neither character, so accepting both costs nothing
        // — unlike `--name`, where the comma belongs to the `{a,b}` alternation and cannot be a
        // separator (see `NameFilter`).
        exact.extend(
            list.split([',', '\n'])
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        );
    }
    if let Some(lines) = text.names_lines {
        exact.extend(
            lines
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_string),
        );
    }
    // `Some(empty)` and `None` are different: an exact list that matched nothing scopes the diff to
    // nothing, where no list at all leaves it unconstrained.
    let names_exact = (text.names_csv.is_some() || text.names_lines.is_some()).then_some(exact);

    let dtype = text
        .dtype_is
        .map(|d| {
            Pattern::new(&d.to_uppercase()).with_context(|| format!("invalid dtype glob {d:?}"))
        })
        .transpose()?;

    let shape = text
        .shape_is
        .map(|s| {
            let path: String = s
                .chars()
                .map(|c| if matches!(c, ',' | 'x' | 'X') { '/' } else { c })
                .collect();
            Pattern::new(&path).with_context(|| format!("invalid shape pattern {s:?}"))
        })
        .transpose()?;

    Ok(crate::diff::TensorFilter {
        names,
        names_exact,
        dtype,
        shape,
    })
}

/// Compile a [`crate::diff::NameMap`] from `--map` rules plus the content of `--map-from`.
///
/// `lines` is the plain form (`PATTERN=>REPLACEMENT` per line); `json` is a `[[pattern, replacement]]`
/// array. The CLI picks between them by file extension and the web by which box was filled; either way
/// the rules land in order — `--map` first, then the file — because later rules apply to the result of
/// earlier ones.
pub(crate) fn name_map(
    rules: &[String],
    lines: Option<&str>,
    json: Option<&str>,
) -> Result<crate::diff::NameMap> {
    let mut pairs = crate::diff::NameMap::parse_rules(rules.iter().map(String::as_str))?;
    if let Some(text) = json {
        let parsed: Vec<(String, String)> = serde_json::from_str(text)
            .context("parsing rename rules as JSON [[pattern, replacement], …]")?;
        pairs.extend(parsed);
    }
    if let Some(text) = lines {
        pairs.extend(crate::diff::NameMap::parse_rules(text.lines())?);
    }
    crate::diff::NameMap::from_pairs(pairs)
}

/// What a value comparison needs beyond the two tensors: how to decode them, and what to compute.
///
/// Shared by the `diff` subcommand and the web's job, so `--values` means the same thing in a terminal
/// and in a browser — down to the decode view and the bin count.
pub(crate) struct ValueOpts<'a> {
    /// `--dtype`: decode both sides under this view before comparing.
    pub view: crate::sample::ViewDtype,
    /// `--bins`: histogram bucket count; `None` picks a sensible count per dtype.
    pub bins: Option<usize>,
    /// `--values`: compare element values.
    pub values: bool,
    /// `--histogram`: compare value distributions.
    pub histogram: bool,
    /// Packing schemas per side, for decoding fused-codebook weights.
    pub old_schemas: &'a std::collections::HashMap<String, crate::sample::PackingSchema>,
    pub new_schemas: &'a std::collections::HashMap<String, crate::sample::PackingSchema>,
}

/// The value/distribution findings for one tensor present on both sides.
///
/// Empty when the shapes differ: comparing element values across a reshape has no meaning, and the
/// structural diff already reports the shape change.
pub(crate) fn tensor_extras(
    a: &crate::tree::TensorInfo,
    b: &crate::tree::TensorInfo,
    opts: &ValueOpts<'_>,
) -> crate::diff::TensorExtras {
    if a.shape != b.shape {
        return crate::diff::TensorExtras::default();
    }
    let values = opts.values.then(|| {
        crate::sample::compare_values(
            a,
            opts.old_schemas.get(&a.name),
            b,
            opts.new_schemas.get(&b.name),
            opts.view,
        )
        .ok()
    });
    let histogram = opts.histogram.then(|| {
        let hd = crate::sample::histogram_diff(
            a,
            opts.old_schemas.get(&a.name),
            b,
            opts.new_schemas.get(&b.name),
            opts.view,
            opts.bins,
        )
        .ok()?;
        Some(crate::diff::HistShift {
            tvd: hd.tvd(),
            bins: hd.n,
        })
    });
    crate::diff::TensorExtras {
        values: values.flatten(),
        histogram: histogram.flatten(),
    }
}

/// Whether two shapes are a repack **fold pair**: `(E, inner…)` against `(ceil(E/fold), inner…)`.
///
/// The sparse side stores one index per 16-bit word; the dense side folds `fold` experts along dim 0
/// into one word. So a candidate pair has the same rank and the same inner dims, with dim 0 divided.
pub(crate) fn detect_fold(old: &[usize], new: &[usize]) -> Option<usize> {
    let (Some((&e, old_inner)), Some((&w, new_inner))) = (old.split_first(), new.split_first())
    else {
        return None;
    };
    if old_inner != new_inner {
        return None;
    }
    if w == 0 || e <= w {
        return None;
    }
    let fold = e.div_ceil(w);
    if !(2..=16).contains(&fold) || w != e.div_ceil(fold) {
        return None;
    }
    Some(fold)
}

/// What a `--verify-repack` run will do: which tensors to verify, at what bit width, and whether
/// anything *else* differs.
pub(crate) struct RepackPlan {
    /// `(old name, new name)` per candidate — the same name on both sides today, but kept as a pair
    /// because a rename map could make them differ.
    pub pairs: Vec<(String, String)>,
    /// Index bit width: as asked, else the max-density packing for the detected fold (`16 / fold`, so
    /// fold 5 ⇒ 3-bit, fold 4 ⇒ 4-bit).
    pub bits: usize,
    /// Whether anything other than the verified fold-pairs differs — an add, a removal, a non-fold
    /// signature change, or a metadata change.
    ///
    /// The fold-pairs themselves always read as "changed" (their shapes differ by construction), so
    /// they are excluded: if they verify equivalent and nothing else differs, the two checkpoints are
    /// the same weights in different packings.
    pub other_differs: bool,
}

/// Can a `--verify-repack` run happen at all, given where the two checkpoints live?
///
/// The verification decodes packed indices **where the data is**: on the ssh proxy, which addresses
/// each side by its `s3://` URI, or here for two local files. A remote *safetensors directory* has no
/// URI the proxy can load, so there is nothing to decode and the answer is a refusal.
///
/// Two properties this signature exists to hold:
///
/// * **One wording.** The web's job and the `diff` subcommand call this, so they cannot refuse the same
///   pair in different words — or, worse, one refuse and the other accept.
/// * **Before the reads.** It depends on the two *specs* and the proxy alone, never on their contents,
///   so it is answerable before a byte is read. The CLI used to check it after two structure reads and
///   a printed diff, which read as "accepted, nothing to verify" — indistinguishable from a run that
///   found no fold-pairs.
pub(crate) fn repack_supported(proxied: bool, s3_pair: bool) -> Result<()> {
    if proxied && !s3_pair {
        anyhow::bail!(
            "--verify-repack over an ssh proxy needs both sides to be s3:// cstorch checkpoints \
             (a remote safetensors dir isn't supported)"
        );
    }
    Ok(())
}

/// Whether the two sides' **values** can be compared at all, from their addresses alone.
///
/// Asked before either side is read. The job used to read both checkpoints first — minutes over an ssh
/// proxy — and only then discover it had no way to the bytes, so the answer arrived after the wait it
/// should have prevented. Nothing about it depends on the read: the specs say where each side lives, and
/// `proxied` says whether there is a proxy to compare on.
///
/// Three answers, not two: both local (read here), two `s3://` with a proxy (compared *there*), and
/// everything else — a remote safetensors directory has no reader on either side of the wire.
///
/// The message names what to do about each case, which is why this is one sentence in one place rather
/// than a guess at each call site.
pub(crate) fn values_supported(left: &str, right: &str, proxied: bool) -> Result<()> {
    let remote: Vec<&str> = [left, right]
        .into_iter()
        .filter(|spec| !crate::capability::Location::of_source_path(spec).is_local())
        .collect();
    if remote.is_empty() {
        return Ok(());
    }
    let s3 = |spec: &str| {
        matches!(
            crate::capability::Location::of_source_path(spec),
            crate::capability::Location::S3
        )
    };
    let both_s3 = s3(left) && s3(right);
    // **Two `s3://` cstorch checkpoints are supported** — on the proxy, which holds the credentials and
    // the data, so the comparison runs where the bytes already are and only the results come back
    // (`remote::RemoteRead::value_diff`). It needs a proxy to run on, which is the one thing about it
    // this cannot infer from the addresses.
    if both_s3 && proxied {
        return Ok(());
    }
    anyhow::bail!(
        "this build compares values by reading the tensors here, and {} {} read remotely — only the \
         structure comes from there.{}",
        remote.join(" and "),
        if remote.len() == 1 { "is" } else { "are" },
        if both_s3 {
            // Supported, but not from here: the comparison needs the proxy that holds the data.
            " Both sides are s3:// cstorch checkpoints, which can be compared on the proxy — but no \
             ssh proxy is configured: start with --ssh-proxy, or set ssh_proxy in the config."
        } else {
            " Copy a checkpoint down to compare its values here. Two s3:// cstorch checkpoints are \
             compared on the proxy instead; a remote safetensors directory cannot be yet."
        }
    );
}

/// Plan a repack verification over two **already-scoped** summaries.
///
/// Shared by the `diff` subcommand and the web's job, so the browser verifies exactly the pairs a
/// terminal would. `Err` when nothing folds — which is a user-facing message, not a bug: it usually
/// means the `--name` scope selected the wrong tensors.
pub(crate) fn plan_repack(
    old: &CheckpointSummary,
    new: &CheckpointSummary,
    repack_bits: Option<usize>,
) -> Result<RepackPlan> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut fold0 = None;
    for (name, osig) in &old.tensors {
        if let Some(nsig) = new.tensors.get(name)
            && let Some(fold) = detect_fold(&osig.shape, &nsig.shape)
        {
            fold0.get_or_insert(fold);
            pairs.push((name.clone(), name.clone()));
        }
    }
    if pairs.is_empty() {
        anyhow::bail!(
            "no fold-pair tensors matched — check the name scope, and that the shapes fold along \
             dim 0 (old E, new ceil(E/fold))"
        );
    }
    let bits = repack_bits.unwrap_or_else(|| (16 / fold0.unwrap_or(1)).max(1));

    let verified: std::collections::HashSet<&String> = pairs.iter().map(|(_, n)| n).collect();
    let added = new.tensors.keys().any(|k| !old.tensors.contains_key(k));
    let removed = old.tensors.keys().any(|k| !new.tensors.contains_key(k));
    let other_changed = old.tensors.iter().any(|(k, osig)| {
        new.tensors.get(k).is_some_and(|nsig| nsig != osig) && !verified.contains(k)
    });
    Ok(RepackPlan {
        pairs,
        bits,
        other_differs: added || removed || other_changed || old.metadata != new.metadata,
    })
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
            parts.push(format!("{} {what}", crate::utils::format_count(n)));
        }
    }
    let meta = report.meta_added.len() + report.meta_removed.len() + report.meta_changed.len();
    if meta > 0 {
        parts.push(format!(
            "{} metadata {}",
            crate::utils::format_count(meta),
            if meta == 1 { "change" } else { "changes" }
        ));
    }
    if parts.is_empty() {
        // `has_differences_with` was true, so something differs that isn't in the counts above
        // (currently only the S3 object metadata of an s3-vs-s3 diff).
        return "differs".to_string();
    }
    format!(
        "{} tensors: {}",
        crate::utils::format_count(report.tensors_unchanged),
        parts.join(", ")
    )
    .replace("tensors: ", "unchanged; ")
}

/// Which of a comparison's two checkpoints is the **old** side.
///
/// A diff is directional — added and removed swap over when you turn it round — and the side-by-side
/// view has always been able to turn it round. The report could not, so seeing the same pair the other
/// way meant editing the URL. Naming the two arrangements keeps "which way round is this" a stated
/// fact rather than an argument order to get right at three call sites.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sides {
    /// The baseline is old, the open checkpoint new — how a report reads by default.
    BaselineFirst,
    /// Swapped: the open checkpoint is the baseline. The same pair, the other direction.
    OpenFirst,
}

/// How to name one side of a comparison on a command line.
///
/// A local checkpoint is named by the **checkpoint**, not by one of its shards: the resolved file
/// list of a sharded directory is every shard, and naming the first compares something else
/// ([`crate::model::checkpoint_path`] answers which single path names them all; `None` when they span
/// directories, where no one path does). A remote side has no local files at all — an `s3://` prefix,
/// an `hf://` repo, a path on an ssh proxy — and is named by the address it was opened as, which is
/// exactly what `diff` accepts. That second case is the one that used to fall through to no command,
/// which the browser then printed as the word `null`.
pub(crate) fn side_operand(spec: &str, files: &[PathBuf]) -> Option<String> {
    if files.is_empty() {
        let spec = spec.trim();
        return (!spec.is_empty()).then(|| spec.to_string());
    }
    crate::model::checkpoint_path(files).map(|p| p.display().to_string())
}

/// The `diff OLD NEW` command that reproduces a comparison — the batch equivalent a UI
/// offers so a finding can be re-run in a terminal (and extended with `--values`).
///
/// Both operands come from [`side_operand`], which names a local checkpoint by its path and a remote
/// one by its address. `diff` accepts either.
///
/// `extra` are the scope flags, so a copied command reproduces the comparison *on screen* rather than
/// an unscoped one over every tensor — which is what it used to hand over.
///
/// `None` when a side cannot be named in one word — see [`side_operand`].
pub(crate) fn cli_diff_command(
    baseline: &str,
    candidate: &str,
    extra: &[String],
    sides: Sides,
    // `#subtree` per side, when a comparison is scoped to one — `(baseline, newer)`. It rides on the
    // operand rather than on a flag, which is how `diff` spells it, so it has to be appended here or
    // the offered command would compare two whole checkpoints.
    subtrees: (Option<&str>, Option<&str>),
) -> Option<String> {
    if baseline.trim().is_empty() || candidate.trim().is_empty() {
        return None;
    }
    let suffix = |spec: &str, sub: Option<&str>| {
        sub.map_or_else(|| spec.to_string(), |p| format!("{spec}#{p}"))
    };
    let against = suffix(baseline, subtrees.0);
    let open = suffix(candidate, subtrees.1);
    let (old, new) = match sides {
        Sides::BaselineFirst => (&against, &open),
        Sides::OpenFirst => (&open, &against),
    };
    let mut cmd = format!(
        "checkpoint-studio diff {} {}",
        shell_quote(old),
        shell_quote(new)
    );
    for arg in extra {
        cmd.push(' ');
        cmd.push_str(&shell_quote(arg));
    }
    Some(cmd)
}

/// Single-quote a path for a copyable shell command when it needs it.
fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.bytes()
            // `@` among the safe ones: every scp-style address has one (`lab@host:/opt/…`), and
            // quoting a whole remote path to protect a character the shell does not treat specially
            // just made the offered command harder to read.
            .all(|b| b.is_ascii_alphanumeric() || b"._-/=:@".contains(&b))
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

    /// The baseline of a diff resolves through `opening`, so it takes every address the rest of the
    /// app takes.
    ///
    /// This is the fix for two diff features that disagreed about what a checkpoint address is: the
    /// report handed its `?against=` to `Path::new` and looked for files, so `s3://…`, `hf://…`,
    /// `[user@]host:/path` and the `:PATH` shorthand all came back as *"no checkpoint files found"*
    /// from the same server whose side-by-side comparison resolved them without complaint.
    ///
    /// Classification only — no read, so no network. That is the part that was wrong; whether a
    /// remote host answers is not this test's business.
    #[test]
    fn a_diff_baseline_accepts_the_addresses_the_rest_of_the_app_accepts() {
        let opts = crate::opening::Options::default();
        // scp-style carries its own host, so it needs no configured proxy and cannot depend on
        // whose machine the suite runs on.
        let target = crate::opening::resolve("host.invalid:/models/ckpt", &opts)
            .expect("an scp-style spec resolves");
        assert!(
            target.remote.is_some(),
            "a diff baseline spelled `host:/path` should resolve to a remote read, \
             not be looked for on this disk"
        );
        assert!(
            target.resolved.is_empty(),
            "a remote target has no local shard files"
        );
    }

    /// A local spec that names nothing still fails the way it always did, with the typed spelling in
    /// the message — going through `opening` must not cost the error its usefulness.
    #[test]
    fn a_diff_baseline_that_names_nothing_says_so() {
        let Err(e) = summarize_spec(
            "/nope/not/a/checkpoint",
            &crate::opening::Options::default(),
        ) else {
            panic!("a missing path should not summarize");
        };
        let msg = format!("{e:#}");
        assert!(
            msg.contains("/nope/not/a/checkpoint"),
            "the rejection should name what was typed: {msg}"
        );
    }

    /// The spec-based entry point and the path-based one agree on a local checkpoint — so routing
    /// `/api/diff` through `opening` changed which addresses are accepted and nothing else.
    #[test]
    fn resolving_a_local_baseline_by_spec_matches_resolving_it_by_path() {
        let path = fixture("diff_old.safetensors");
        let by_path = summarize(&path).expect("the fixture summarizes");
        let by_spec = summarize_spec(
            &path.display().to_string(),
            &crate::opening::Options::default(),
        )
        .expect("the same fixture summarizes as a spec");
        // Through serde, which these types derive: it gives a readable diff on failure without
        // asking the comparable-structure types to carry a `Debug` they have no other use for.
        assert_eq!(
            serde_json::to_value(&by_path).unwrap(),
            serde_json::to_value(&by_spec).unwrap(),
            "the same local checkpoint should summarize identically by path and by spec"
        );
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
        let cmd = cli_diff_command("/a/old", "/b/new", &[], Sides::BaselineFirst, (None, None))
            .expect("both sides have an address");
        assert_eq!(cmd, "checkpoint-studio diff /a/old /b/new");
        let quoted = cli_diff_command(
            "/a/needs quoting",
            "/b/new",
            &[],
            Sides::BaselineFirst,
            (None, None),
        )
        .expect("both sides have an address");
        assert!(quoted.contains("'/a/needs quoting'"), "{quoted}");
    }

    /// Turned round, the command has to turn round with it: a swapped report whose command still
    /// said `diff OLD NEW` would hand over the opposite of what is on screen.
    #[test]
    fn a_swapped_report_offers_the_swapped_command() {
        let cmd = cli_diff_command("/a/old", "/b/new", &[], Sides::OpenFirst, (None, None))
            .expect("both sides have an address");
        assert_eq!(cmd, "checkpoint-studio diff /b/new /a/old");
    }

    /// A checkpoint is named by the address it was opened as, whatever it is made of.
    ///
    /// The regression behind this: the candidate used to be re-derived from its *resolved file list*,
    /// so a sharded directory arrived as every shard and naming the first produced
    /// `diff OLD <ckpt>/codebooks.safetensors` — a command comparing one shard. A spec cannot make
    /// that mistake, and it also covers the sides that have no local files at all.
    #[test]
    fn the_cli_command_names_each_side_as_it_was_addressed() {
        let cmd = cli_diff_command("/base", "/ckpt", &[], Sides::BaselineFirst, (None, None))
            .expect("both sides have an address");
        assert_eq!(cmd, "checkpoint-studio diff /base /ckpt");

        // A remote pair: neither side is a path on this machine, and both are still nameable.
        let remote = cli_diff_command(
            "s3://bucket/ckpt-2000",
            "lab@host:/opt/models/ckpt-1000",
            &[],
            Sides::BaselineFirst,
            (None, None),
        )
        .expect("a remote pair has addresses too");
        assert_eq!(
            remote, "checkpoint-studio diff s3://bucket/ckpt-2000 lab@host:/opt/models/ckpt-1000",
            "the report used to offer no command at all for these, which the browser printed as `null`"
        );
    }

    /// Whether a value comparison can happen at all — asked of the two addresses, before any read.
    ///
    /// The reported bug: a remote pair was accepted, both checkpoints were read (minutes over an ssh
    /// proxy), and *then* the job said a remote source serves no tensor data. The question never
    /// needed the read.
    #[test]
    fn comparing_values_says_where_it_can_happen_before_reading_anything() {
        // Both local: read here, proxy or not.
        assert!(values_supported("/models/a", "/models/b", false).is_ok());
        assert!(values_supported("/models/a", "/models/b", true).is_ok());

        // **Two s3:// cstorch checkpoints are supported** — on the proxy that holds the data.
        assert!(
            values_supported("s3://bucket/old", "s3://bucket/new", true).is_ok(),
            "the proxy compares them where the bytes already are"
        );

        // One remote side: named, so the reader knows which one to copy down.
        let one = values_supported("/models/a", "lab@host:/opt/models/b", true)
            .expect_err("a remote side has no bytes to compare");
        let msg = format!("{one:#}");
        assert!(msg.contains("lab@host:/opt/models/b"), "{msg}");
        assert!(
            !msg.contains("/models/a"),
            "the local side is not the problem: {msg}"
        );
        assert!(msg.contains("is read remotely"), "{msg}");

        // Two `s3://` sides: the one remote pair that *has* an answer, so the refusal names it.
        // The same pair with no proxy to run on: refused, and the refusal says what is missing rather
        // than implying the pair itself cannot be compared.
        let pair = values_supported("s3://bucket/old", "s3://bucket/new", false)
            .expect_err("there is nowhere to compare them");
        let msg = format!("{pair:#}");
        assert!(
            msg.contains("are read remotely"),
            "both sides, plural: {msg}"
        );
        assert!(
            msg.contains("no ssh proxy is configured"),
            "the missing piece is the proxy, not the capability: {msg}"
        );

        // A mixed remote pair cannot use verify-repack either, and must not be told it can.
        let mixed =
            values_supported("lab@host:/opt/a", "s3://bucket/new", true).expect_err("no bytes");
        let msg = format!("{mixed:#}");
        assert!(msg.contains("Copy a checkpoint down"), "{msg}");
        assert!(
            msg.contains("a remote safetensors directory cannot be yet"),
            "and says which case is genuinely unsupported anywhere: {msg}"
        );
    }

    /// Which pairs `--verify-repack` can run over, in the one place both surfaces ask.
    ///
    /// The middle case is the reported bug: a `:PATH` (remote safetensors directory) against an
    /// `s3://` checkpoint. The web refused it; the CLI checked the same condition only after two
    /// structure reads and a printed diff, so it looked accepted. Both now call this, before reading.
    #[test]
    fn a_repack_needs_either_two_local_files_or_two_s3_uris() {
        // Two local checkpoints: decoded here, no proxy involved.
        assert!(repack_supported(false, false).is_ok());
        // Two s3:// checkpoints over the proxy: what the mode exists for.
        assert!(repack_supported(true, true).is_ok());
        // Mixed: the proxy addresses each side by URI, so there is nothing to decode.
        let refused = repack_supported(true, false).expect_err("a mixed remote pair is refused");
        let msg = format!("{refused:#}");
        assert!(
            msg.contains("both sides to be s3:// cstorch checkpoints"),
            "the refusal should say what is needed: {msg}"
        );
    }

    /// How each side is named: a local checkpoint by the path that covers its files, a remote one by
    /// its address. The second case had no answer at all before, and the browser drew the absence.
    #[test]
    fn a_side_is_named_by_its_checkpoint_or_by_its_address() {
        let shards = [
            PathBuf::from("/ckpt/codebooks.safetensors"),
            PathBuf::from("/ckpt/model-00001-of-00002.safetensors"),
        ];
        assert_eq!(side_operand("/ckpt", &shards).as_deref(), Some("/ckpt"));
        // No local files: the address is what names it.
        assert_eq!(
            side_operand("s3://bucket/ckpt", &[]).as_deref(),
            Some("s3://bucket/ckpt")
        );
        // Files spanning directories have no single name, and neither does nothing at all.
        let scattered = [
            PathBuf::from("/a/one.safetensors"),
            PathBuf::from("/b/two.safetensors"),
        ];
        assert!(side_operand("/a /b", &scattered).is_none());
        assert!(side_operand("  ", &[]).is_none());
    }

    /// A side with no address gets no command — better nothing than `diff /base ''`.
    #[test]
    fn there_is_no_command_without_two_addresses() {
        assert!(cli_diff_command("/base", "", &[], Sides::BaselineFirst, (None, None)).is_none());
        assert!(cli_diff_command("", "/new", &[], Sides::BaselineFirst, (None, None)).is_none());
        assert!(
            cli_diff_command("/base", "   ", &[], Sides::BaselineFirst, (None, None)).is_none()
        );
    }
}
