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

use anyhow::{Context, Result, bail};
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

/// Where the values are compared.
///
/// Decided from the two sides once they are read, and named rather than inferred at each use: the two
/// arms read *different data in different places*, and the remote one never sees a tensor byte here.
enum Mode {
    /// Both sides local: read and compare in this process.
    Local,
    /// Two `s3://` cstorch checkpoints: compared by a script on the proxy, which has the credentials and
    /// the data. Only counts and per-tensor results come back. The same call the `diff` subcommand makes
    /// (`remote::RemoteRead::value_diff`), so a browser run and a terminal run answer alike.
    Proxy {
        proxy: crate::remote::RemoteRead,
        old_side: crate::remote::RemoteSide,
        new_side: crate::remote::RemoteSide,
    },
}

/// One side, read far enough to compare values: tensors with byte access, plus packing schemas.
struct Side {
    tensors: Vec<crate::tree::TensorInfo>,
    metadata: Vec<crate::tree::MetadataInfo>,
    /// The `s3://` URI, when this side is one — the proxy's comparison addresses objects by URI.
    s3_uri: Option<String>,
    /// The path on the proxy, when this side is a remote safetensors file or directory. Without any
    /// `host:` prefix: the session is already on that host.
    remote_path: Option<String>,
    /// The proxy this side is read through, when it is remote.
    remote: Option<crate::remote::RemoteRead>,
    schemas: HashMap<String, crate::sample::PackingSchema>,
    local: bool,
}

fn read_side(spec: &str, opts: &crate::opening::Options, job: &Job) -> Result<Side> {
    let target =
        crate::opening::resolve(spec, opts).with_context(|| format!("resolving {spec}"))?;
    // Where this side's *bytes* can be read: here, if it is local. A remote read carries structure
    // alone — but an `s3://` cstorch checkpoint can be compared **on the proxy**, which is what
    // `s3_uri` and `remote` are kept for.
    let local = target.remote.is_none();
    let s3_uri = target
        .requested
        .first()
        .map(|p| p.to_string_lossy().into_owned())
        .filter(|s| s.starts_with("s3://"));
    let remote = target.remote.clone();
    // The path as the proxy sees it — what `safetensors` opens over there. `requested` is already the
    // remote path: the host was split off when the spec was resolved.
    let remote_path = (remote.is_some() && s3_uri.is_none())
        .then(|| {
            target
                .requested
                .first()
                .map(|p| p.to_string_lossy().into_owned())
        })
        .flatten();
    let opened = target
        .read(crate::opening::Want::Model, job.read_progress())
        .with_context(|| format!("reading {spec}"))?;
    let (tensors, metadata) = (opened.parts.tensors, opened.parts.metadata);
    // Packing schemas come from the metadata, so a fused-codebook weight decodes to real values.
    let schemas = crate::sample::parse_packing_schemas(&tensors, &metadata);
    Ok(Side {
        tensors,
        metadata,
        s3_uri,
        remote_path,
        remote,
        schemas,
        local,
    })
}

impl Side {
    /// How the proxy reads this side, or `None` when it cannot.
    fn as_remote(&self) -> Option<crate::remote::RemoteSide> {
        if let Some(uri) = &self.s3_uri {
            return Some(crate::remote::RemoteSide::S3(uri.clone()));
        }
        self.remote_path
            .as_ref()
            .map(|p| crate::remote::RemoteSide::Safetensors(p.clone()))
    }
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
    // Before either read: a remote side serves no tensor data, and finding that out after two
    // multi-minute reads is the failure this check exists to prevent. The start handler asks the same
    // question, so this is the second line of defence rather than the only one.
    let at = crate::compare::data_where(
        left,
        right,
        current.proxy_host(),
        crate::compare::Work::Values,
    )?;
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
    // **Where the comparison happens**, from the answer the addresses already gave — confirmed against
    // what the two sides turned out to be, so a spec that resolved to something its address did not
    // promise is caught before a byte is compared.
    let mode = match at {
        crate::compare::ValuesAt::Here if old.local && new.local => Mode::Local,
        crate::compare::ValuesAt::OnProxy => {
            let (Some(proxy), Some(old_side), Some(new_side)) = (
                old.remote.as_ref().or(new.remote.as_ref()),
                old.as_remote(),
                new.as_remote(),
            ) else {
                bail!(
                    "comparing these on the proxy needs both sides addressed as the proxy reads them — \
                     an s3:// URI or a path on that host"
                );
            };
            Mode::Proxy {
                proxy: proxy.clone(),
                old_side,
                new_side,
            }
        }
        crate::compare::ValuesAt::Here => bail!(
            "one side turned out not to be readable here after all — its address said it was local"
        ),
    };

    // Rename rules first, as the CLI does — otherwise two lined-up schemes read as every tensor added
    // and removed, and nothing gets its values compared at all.
    //
    // The renamed names are used for *pairing*; the reads use the original `TensorInfo`, whose name also
    // keys the packing schemas. So `renamed_to_original` maps one to the other rather than rewriting the
    // tensor and losing its schema.
    let renamed = scope.rename_tensors(&old.tensors);
    // Which names the alignment *folded*: 256 per-expert tensors onto the one fused tensor that holds
    // them. Kept because a folded name cannot have its values compared element by element — the two
    // sides are not the same array — and saying which names those are is the difference between a
    // result a reader can act on and a silent `0 of 0 compared`.
    let folded = renamed.folds.clone();
    let renamed_old = renamed.tensors;
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

    // The remote arm compares the whole selection in one call — the proxy loads both checkpoints once and
    // walks the pairs itself, so asking per tensor would pay for that load per tensor.
    if let Mode::Proxy {
        proxy,
        old_side,
        new_side,
    } = &mode
    {
        let extras = compare_on_proxy(job, proxy, old_side, new_side, &common, what, &folded)?;
        return finish(job, old_sum, new_sum, extras);
    }

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
    finish(job, old_sum, new_sum, computed.into_iter().collect())
}

/// The run's verdict and report, from whatever the comparison produced — the same tail for both modes,
/// so a remote run and a local one are reported in the same shape.
fn finish(
    job: &Job,
    old_sum: crate::diff::CheckpointSummary,
    new_sum: crate::diff::CheckpointSummary,
    computed: HashMap<String, crate::diff::TensorExtras>,
) -> Result<()> {
    let compared = computed.len();
    let differ = computed
        .values()
        .filter(|e| {
            e.values.is_some_and(|v| v.differing > 0) || e.histogram.is_some_and(|h| h.tvd > 0.0)
        })
        .count();
    // `RefCell` because `compare_with` takes an `Fn`: each name is asked for exactly once, so the entry
    // is *moved* out rather than cloned — the same trick the CLI uses for the same reason.
    let extras: std::cell::RefCell<HashMap<String, crate::diff::TensorExtras>> =
        std::cell::RefCell::new(computed);
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

/// Compare the selection **on the ssh proxy**, streaming its progress into the job.
///
/// The same call the `diff` subcommand makes for an s3-vs-s3 pair: a script on the proxy loads both
/// checkpoints, realises each pair of tensors and streams row-blocks through the comparison there. No
/// tensor data crosses the wire — only the byte counts the bars are drawn from and the small per-tensor
/// result. The browser had no path to this at all and refused the pair outright, while a terminal on the
/// same machine could do it.
fn compare_on_proxy(
    job: &Job,
    proxy: &crate::remote::RemoteRead,
    old_side: &crate::remote::RemoteSide,
    new_side: &crate::remote::RemoteSide,
    names: &[String],
    what: &What,
    // Names the alignment folded, and into how many parts — see `run`.
    folded: &std::collections::BTreeMap<String, usize>,
) -> Result<HashMap<String, crate::diff::TensorExtras>> {
    // The proxy pairs by name: the selection is already keyed by the *renamed* old name, which is the
    // new side's name — the same key the report is built from.
    let pairs: Vec<(String, String)> = names.iter().map(|n| (n.clone(), n.clone())).collect();
    let vopts = crate::remote::RemoteValueOpts {
        values: what.values,
        histogram: what.histogram,
        bins: what.bins,
        // The per-bin table is for a single tensor's detail view; a run over many wants the summary.
        full_hist: what.tensor.is_some(),
        // A safetensors side is read whole into the proxy's memory, and an embedding weight is over a
        // gigabyte — so fewer at once when one is involved. The s3 path streams its objects in chunks
        // and can afford the wider fan-out that hides their latency.
        jobs: if matches!(
            (old_side, new_side),
            (
                crate::remote::RemoteSide::S3(_),
                crate::remote::RemoteSide::S3(_)
            )
        ) {
            what.jobs.clamp(1, 32)
        } else {
            what.jobs.clamp(1, 4)
        },
    };
    let mut password = None;
    let session = proxy
        .open_with(&mut password)
        .with_context(|| format!("opening an ssh session to {}", proxy.host))?;
    let mut counted = ProxyBytes::default();
    let (map, _stats) = proxy.value_diff(
        &session,
        old_side,
        new_side,
        &pairs,
        &vopts,
        // The job's own flag: *Stop* tears the ssh channel down, which ends the remote python with it.
        // Without it a stop stopped only the *waiting* — the proxy went on comparing every tensor of a
        // checkpoint nobody was watching, and the only way to end it was to log in and kill it.
        Some(job.read_progress().abort_flag()),
        |ev| proxy_event(job, &ev, &mut counted),
    )?;
    let mut out: HashMap<String, crate::diff::TensorExtras> = HashMap::new();
    for (name, diff) in map {
        if let Some(e) = &diff.error {
            // A folded name's sides are not the same array: `×256 → ×1` means 256 per-expert tensors
            // against the one fused tensor holding them, in a packed layout whose inner dimension is
            // padded. There is no element-by-element comparison to run, and the check that *does*
            // answer "are these the same weights" for such a pair is the repack verification.
            let why = folded.get(&name).map_or_else(
                || e.clone(),
                |parts| {
                    format!(
                        "{e} — the alignment folded {parts} tensors onto this one (×{parts} → ×1), so \
                         the two sides are not the same array. Use verify-repack, which decodes the \
                         packing."
                    )
                },
            );
            job.add_finding(json!({ "kind": "tensor", "name": name, "error": why }));
            continue;
        }
        let extras = crate::diff::TensorExtras {
            values: diff.values,
            histogram: diff
                .hist_shift
                .map(|(tvd, bins)| crate::diff::HistShift { tvd, bins }),
        };
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
        out.insert(name, extras);
    }
    Ok(out)
}

/// Bytes read across a whole remote run, from per-tensor counters that restart at zero.
///
/// Without the running total the counter jumps backwards mid-run — `27.14 GB`, then `0.01 GB` — which
/// reads as a bug in the tool rather than as the next tensor starting. (The repack job keeps its own for
/// the same reason; the two events are different types.)
#[derive(Default)]
struct ProxyBytes {
    base: u64,
    current: (String, u64),
}

impl ProxyBytes {
    fn observe(&mut self, name: &str, bytes: u64) -> u64 {
        if self.current.0 != name {
            self.base += self.current.1;
            self.current = (name.to_string(), 0);
        }
        // `max`, not assignment: the two sides report independently and can arrive out of order.
        self.current.1 = self.current.1.max(bytes);
        self.base + self.current.1
    }
}

/// One remote progress event, as job progress. Bytes matter: a single expert weight can be gigabytes, so
/// "tensor 3 of 19" alone would sit still for minutes.
fn proxy_event(job: &Job, ev: &crate::remote::RepackEvent<'_>, counted: &mut ProxyBytes) {
    match *ev {
        crate::remote::RepackEvent::Loading(which) => job.progress_to(0, which),
        crate::remote::RepackEvent::Start { done, total, name } => {
            job.set_total(total);
            job.progress_to(done, name);
        }
        crate::remote::RepackEvent::Bytes {
            name,
            old_done,
            new_done,
        } => job.set_bytes(counted.observe(name, old_done + new_done)),
        // The comparison of one tensor, span by span: for a safetensors side there is no download to
        // animate, and even for s3 the read finishes long before the compare does — so this is what the
        // reader watches while a gigabyte-scale weight is worked through.
        crate::remote::RepackEvent::Comparing { name, spans } => {
            let label = match spans {
                Some((done, total)) => format!("{name} · comparing {done}/{total}"),
                None => format!("{name} · comparing"),
            };
            job.set_current(&label);
        }
        crate::remote::RepackEvent::Size { .. } | crate::remote::RepackEvent::Done { .. } => {}
    }
}
