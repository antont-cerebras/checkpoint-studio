//! The `--values` / `--histogram` / `--tensor` worker: does the *data* differ, not just the structure?
//!
//! A structural diff answers "same names, dtypes and shapes". These answer the question that follows —
//! are the numbers the same? — which means reading every selected tensor on both sides. Minutes for a
//! large checkpoint, so a [`super::jobs`] job rather than a response.
//!
//! **The comparison itself is not reimplemented.** Each tensor's findings come from
//! [`crate::compare::tensor_extras`], shared with the `diff` subcommand, and the report is
//! `diff::compare_with` fed those extras — the same two calls the CLI makes, in the same order, so
//! `--values` means the same thing in a terminal and in a browser down to the decode view and bin count.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::json;

use super::diffscope::DiffScope;
use super::jobs::Job;

/// What kind of value comparison to run — the flags, as a request carries them.
pub(crate) struct What {
    /// `--values`: compare element values.
    pub values: bool,
    /// `--histogram`: compare value distributions.
    pub histogram: bool,
    /// `--bins`.
    pub bins: Option<usize>,
    /// `--dtype`: the decode view.
    pub view: crate::sample::ViewDtype,
    /// `--jobs`: how many tensors to read at once. Reading is I/O-bound, so overlapping helps; 1 is
    /// sequential.
    pub jobs: usize,
    /// `--tensor NAME`: compare just this one, values included. Takes precedence over the scope, as it
    /// does on the command line.
    pub tensor: Option<String>,
}

/// One side, read far enough to compare values: tensors with byte access, plus packing schemas.
struct Side {
    tensors: Vec<crate::tree::TensorInfo>,
    metadata: Vec<crate::tree::MetadataInfo>,
    schemas: HashMap<String, crate::sample::PackingSchema>,
    local: bool,
}

fn read_side(spec: &str, opts: &crate::opening::Options, job: &Job) -> Result<Side> {
    let target =
        crate::opening::resolve(spec, opts).with_context(|| format!("resolving {spec}"))?;
    // Value comparison needs the *bytes*, which only a local source can give: a remote read carries
    // structure alone. Reported here rather than as a failure per tensor.
    let local = target.remote.is_none();
    let opened = target
        .read(crate::opening::Want::Model, job.read_progress())
        .with_context(|| format!("reading {spec}"))?;
    let (tensors, metadata) = (opened.parts.tensors, opened.parts.metadata);
    // Packing schemas come from the metadata, so a fused-codebook weight decodes to real values.
    let schemas = crate::sample::parse_packing_schemas(&tensors, &metadata);
    Ok(Side {
        tensors,
        metadata,
        schemas,
        local,
    })
}

/// Compare two checkpoints' values, recording each tensor's findings as they land.
pub(crate) fn run(
    current: &Arc<super::Current>,
    job: &Job,
    left: &str,
    right: &str,
    scope: &DiffScope,
    what: &What,
) -> Result<()> {
    let opts = current.read_options();
    job.progress_to(0, left);
    let old = read_side(left, opts, job)?;
    if job.cancelled() {
        return Ok(());
    }
    job.progress_to(0, right);
    let new = read_side(right, opts, job)?;
    if job.cancelled() {
        return Ok(());
    }
    if !old.local || !new.local {
        anyhow::bail!(
            "comparing values needs both checkpoints' tensor data, which a remote source does not \
             provide — for two s3:// checkpoints use verify-repack, which decodes on the proxy"
        );
    }

    // Rename rules first, as the CLI does — otherwise two lined-up schemes read as every tensor added
    // and removed, and nothing gets its values compared at all.
    //
    // The renamed names are used for *pairing*; the reads use the original `TensorInfo`, whose name also
    // keys the packing schemas. So `renamed_to_original` maps one to the other rather than rewriting the
    // tensor and losing its schema.
    let renamed_old = scope.rename_tensors(&old.tensors).tensors;
    let renamed_to_original: HashMap<String, crate::tree::TensorInfo> = renamed_old
        .iter()
        .zip(old.tensors.iter())
        .map(|(r, o)| (r.name.clone(), o.clone()))
        .collect();
    let mut old_sum = crate::diff::CheckpointSummary::from_loaded(&renamed_old, &old.metadata);
    let mut new_sum = crate::diff::CheckpointSummary::from_loaded(&new.tensors, &new.metadata);

    // `--tensor` is its own selection and takes precedence over the scope — the CLI says so outright
    // ("--tensor takes precedence; filters ignored"), and silently intersecting them would compare a
    // different set than the same flags do in a terminal.
    let selected: Option<std::collections::HashSet<String>> = if let Some(one) = &what.tensor {
        Some(std::iter::once(one.clone()).collect())
    } else {
        let scoped = scope.compare(
            crate::diff::CheckpointSummary::from_loaded(&old.tensors, &old.metadata),
            crate::diff::CheckpointSummary::from_loaded(&new.tensors, &new.metadata),
        );
        scoped
            .matched
            .as_ref()
            .map(|m| m.names.iter().cloned().collect())
    };
    if let Some(keep) = &selected {
        // `retain_tensors`, not `tensors.retain`: it drops each tensor's footprint with it, so the
        // report's `size:` / `params:` describe the selected tensors rather than the whole checkpoints.
        old_sum.retain_tensors(|n| keep.contains(n));
        new_sum.retain_tensors(|n| keep.contains(n));
        if what.tensor.is_some() && old_sum.tensors.is_empty() && new_sum.tensors.is_empty() {
            anyhow::bail!(
                "{}: no such tensor in either checkpoint",
                what.tensor.as_deref().unwrap_or_default()
            );
        }
    }
    // Metadata is not compared under a scope — the CLI's rule; see `diffscope`.
    if selected.is_some() {
        old_sum.metadata.clear();
        new_sum.metadata.clear();
    }

    // Only names on both sides have values to compare; one-sided tensors are already a difference.
    let common: Vec<String> = old_sum
        .tensors
        .keys()
        .filter(|n| new_sum.tensors.contains_key(*n))
        .cloned()
        .collect();
    job.set_total(common.len());

    let by_name = |ts: &[crate::tree::TensorInfo]| -> HashMap<String, crate::tree::TensorInfo> {
        ts.iter().map(|t| (t.name.clone(), t.clone())).collect()
    };
    // The old side is looked up by its *renamed* name and yields the original tensor — see above.
    let (old_map, new_map) = (renamed_to_original, by_name(&new.tensors));
    let vopts = crate::compare::ValueOpts {
        view: what.view,
        bins: what.bins,
        values: what.values,
        histogram: what.histogram,
        old_schemas: &old.schemas,
        new_schemas: &new.schemas,
    };

    let done = std::sync::atomic::AtomicUsize::new(0);
    let compute = |name: &String| -> (String, crate::diff::TensorExtras) {
        if job.cancelled() {
            return (name.clone(), crate::diff::TensorExtras::default());
        }
        let at = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        job.progress_to(at, name);
        let (Some(a), Some(b)) = (old_map.get(name), new_map.get(name)) else {
            return (name.clone(), crate::diff::TensorExtras::default());
        };
        let extras = crate::compare::tensor_extras(a, b, &vopts);
        // Recorded as it lands — the first differing tensor is often the answer.
        if extras.values.is_some() || extras.histogram.is_some() {
            job.add_finding(json!({
                "kind": "tensor",
                "name": name,
                "values": extras.values.map(|v| json!({
                    "differing": v.differing,
                    "elements": v.elements,
                    "nonfinite_mismatch": v.nonfinite_mismatch,
                    "max_abs": v.max_abs,
                    "mean_abs": v.mean_abs,
                })),
                "histogram": extras.histogram.map(|h| json!({ "tvd": h.tvd, "bins": h.bins })),
            }));
        }
        (name.clone(), extras)
    };

    // Reading tensor data is I/O-bound, so overlap up to `jobs` tensors — the same reasoning, and the
    // same default, as the CLI's `--jobs`. Results are order-independent.
    let computed: Vec<(String, crate::diff::TensorExtras)> = if what.jobs <= 1 {
        common.iter().map(compute).collect()
    } else {
        use rayon::prelude::*;
        rayon::ThreadPoolBuilder::new()
            .num_threads(what.jobs)
            .build()
            .map_or_else(
                |_| common.iter().map(compute).collect(),
                |pool| pool.install(|| common.par_iter().map(compute).collect()),
            )
    };
    job.progress_to(common.len(), "");

    let compared = computed.len();
    let differ = computed
        .iter()
        .filter(|(_, e)| {
            e.values.is_some_and(|v| v.differing > 0) || e.histogram.is_some_and(|h| h.tvd > 0.0)
        })
        .count();
    // `RefCell` because `compare_with` takes an `Fn`: each name is asked for exactly once, so the entry
    // is *moved* out rather than cloned — the same trick the CLI uses for the same reason.
    let extras: std::cell::RefCell<HashMap<String, crate::diff::TensorExtras>> =
        std::cell::RefCell::new(computed.into_iter().collect());
    // The report, with the value findings folded in — so a tensor whose dtype and shape match but whose
    // numbers differ reads as *changed* rather than unchanged, which is the whole point of `--values`.
    let report = crate::diff::compare_with(&old_sum, &new_sum, |name| {
        extras.borrow_mut().remove(name).unwrap_or_default()
    });
    job.add_finding(json!({
        "kind": "verdict",
        "compared": compared,
        "differ": differ,
        "verdict": crate::compare::verdict(&report),
        "report": report,
    }));
    Ok(())
}
