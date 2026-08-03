//! The `--verify-repack` worker: do two checkpoints hold the **same weights in different packings**?
//!
//! The question a re-quantization raises. The sparse side stores one 3- or 4-bit index per 16-bit word;
//! the dense side folds several experts into one word. Their shapes therefore differ by construction, so
//! a structural diff can only ever say "changed" — the interesting answer needs the *indices decoded on
//! both sides and compared*, which means reading both tensors in full.
//!
//! Minutes of reading, and per-tensor findings, so this is a [`super::jobs`] job rather than a response.
//!
//! **Nothing here decides anything the CLI decides differently.** The candidate pairs, the bit width and
//! the "does anything else differ" question all come from [`crate::compare::plan_repack`], shared with
//! the `diff` subcommand; the verification itself is `remote::RemoteRead::verify_repack` on the ssh proxy
//! for `s3://` pairs, or `crate::repack` locally — the same two implementations the CLI calls, producing
//! the same [`crate::remote::RepackResult`].

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::json;

use super::diffscope::DiffScope;
use super::jobs::Job;

/// Where the decoding happens. Decided before planning, because the precondition is about the *sources*
/// and reporting it as a planning failure names the wrong problem.
enum Mode {
    /// Both sides readable by the ssh proxy — decoded over there, only results cross the wire. Each
    /// side is an `s3://` URI or a path on that host, which is what `RemoteSide` says.
    Proxy {
        proxy: crate::remote::RemoteRead,
        old_side: crate::remote::RemoteSide,
        new_side: crate::remote::RemoteSide,
    },
    /// Local files on both sides, decoded here.
    Local,
}

/// Read one side far enough to verify it: its tensors, and the proxy it lives behind.
struct Side {
    tensors: Vec<crate::tree::TensorInfo>,
    metadata: Vec<crate::tree::MetadataInfo>,
    /// The `s3://` URI, when this side is one — the proxy addresses cstorch objects by URI.
    s3_uri: Option<String>,
    /// The path on the proxy, when this side is a remote safetensors file or directory. Without any
    /// `host:` prefix: the session is already on that host.
    remote_path: Option<String>,
    /// The proxy this side is read through, when it is remote.
    remote: Option<crate::remote::RemoteRead>,
    /// Whether this side's bytes are readable *here*.
    local: bool,
}

impl Side {
    /// How the proxy reads this side, or `None` when it cannot. The same two kinds
    /// `valuesjob::Side::as_remote` answers with, because the proxy opens them the same way.
    fn as_remote(&self) -> Option<crate::remote::RemoteSide> {
        if let Some(uri) = &self.s3_uri {
            return Some(crate::remote::RemoteSide::S3(uri.clone()));
        }
        self.remote_path
            .as_ref()
            .map(|p| crate::remote::RemoteSide::Safetensors(p.clone()))
    }
}

/// Read a side, reporting the read through the job's own handle so cancelling reaches it.
fn read_side(spec: &str, opts: &crate::opening::Options, job: &Job) -> Result<Side> {
    let target =
        crate::opening::resolve(spec, opts).with_context(|| format!("resolving {spec}"))?;
    let s3_uri = target
        .requested
        .first()
        .map(|p| p.to_string_lossy().into_owned())
        .filter(|s| s.starts_with("s3://"));
    let remote = target.remote.clone();
    let local = remote.is_none();
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
        .read(crate::opening::Want::Parts, job.read_progress())
        .with_context(|| format!("reading {spec}"))?;
    let (tensors, metadata) = (opened.parts.tensors, opened.parts.metadata);
    Ok(Side {
        tensors,
        metadata,
        s3_uri,
        remote_path,
        remote,
        local,
    })
}

/// Verify a repack and record what it finds.
///
/// `Ok(())` means the run finished — *not* that the two sides matched; that is in the findings, which is
/// the same distinction the CLI draws between its exit code and its report.
pub(crate) fn run(
    current: &Arc<super::Current>,
    job: &Job,
    left: &str,
    right: &str,
    scope: &DiffScope,
    bits: Option<usize>,
    // How each side packs its indices, as written — see `compare::RepackSchemas`. These checkpoints
    // declare their packing nowhere, so for a sparse-vs-merged pair it has to be said.
    schemas: crate::compare::RepackSchemas<'_>,
) -> Result<()> {
    let opts = current.read_options();
    // Before either read: the same question the value job asks, of the same function — can both sides
    // be read in one place? Finding that out after two multi-minute reads is the failure this prevents.
    let at = crate::compare::data_where(
        left,
        right,
        current.proxy_host(),
        crate::compare::Work::Repack,
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

    // Scope first, exactly as the CLI does: `--verify-repack` is almost always used with `--name`, and
    // verifying 117k tensors when nineteen were asked for is not a slow answer but a wrong one.
    let scoped = scope.compare(
        crate::diff::CheckpointSummary::from_loaded(&old.tensors, &old.metadata),
        crate::diff::CheckpointSummary::from_loaded(&new.tensors, &new.metadata),
    );
    let old_sum = crate::diff::CheckpointSummary::from_loaded(&old.tensors, &old.metadata);
    let new_sum = crate::diff::CheckpointSummary::from_loaded(&new.tensors, &new.metadata);
    let selected: Option<std::collections::HashSet<String>> = scoped
        .matched
        .as_ref()
        .map(|m| m.names.iter().cloned().collect());
    let (mut old_sum, mut new_sum) = (old_sum, new_sum);
    if let Some(keep) = &selected {
        // `retain_tensors`, not `tensors.retain`: it drops each tensor's footprint with it, so the
        // report's `size:` / `params:` describe the selected tensors rather than the whole checkpoints.
        old_sum.retain_tensors(|n| keep.contains(n));
        new_sum.retain_tensors(|n| keep.contains(n));
    }

    // **Where the decoding happens**, from the answer the two addresses already gave (asked again at
    // the start handler, before either read) — confirmed against what the sides turned out to be.
    // Answered before planning, because planning first would report "no fold-pair tensors matched", a
    // true statement about the wrong problem.
    let mode = match at {
        crate::compare::ValuesAt::Here if old.local && new.local => Mode::Local,
        crate::compare::ValuesAt::OnProxy => {
            let (Some(proxy), Some(old_side), Some(new_side)) = (
                old.remote.as_ref().or(new.remote.as_ref()),
                old.as_remote(),
                new.as_remote(),
            ) else {
                bail!(
                    "verifying these on the proxy needs both sides addressed as the proxy reads them — \
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

    let plan = crate::compare::plan_repack(&old_sum, &new_sum, bits, schemas)?;
    job.set_total(plan.pairs.len());

    let results = match &mode {
        Mode::Proxy {
            proxy,
            old_side,
            new_side,
        } => verify_on_proxy(proxy, job, old_side, new_side, &plan)?,
        Mode::Local => verify_locally(job, &old.tensors, &new.tensors, &plan),
    };

    // The verdict, as the CLI words it, plus every per-tensor result.
    let equivalent = plan
        .pairs
        .iter()
        .all(|(_, n)| results.get(n).is_some_and(|r| r.differing == 0));
    job.add_finding(json!({
        "kind": "verdict",
        "bits": plan.bits,
        // How each side was decoded, in the CLI's words (`compare::RepackPlan::packing_note`). A
        // verdict is only as good as this assumption, and `at 4-bit` is false of a `[3,4,4,4]`
        // candidate — the same wording the terminal's own header carries.
        "packing": plan.packing_note(),
        "pairs": plan.pairs.len(),
        "equivalent": equivalent,
        // Whether anything *outside* the verified pairs differs. The pairs always read as "changed"
        // structurally (their shapes differ by design), so this is what decides whether the two
        // checkpoints are the same weights modulo packing.
        "other_differs": plan.other_differs,
    }));
    Ok(())
}

/// Verify on the ssh proxy, streaming the remote's progress into the job.
fn verify_on_proxy(
    r: &crate::remote::RemoteRead,
    job: &Job,
    old_side: &crate::remote::RemoteSide,
    new_side: &crate::remote::RemoteSide,
    plan: &crate::compare::RepackPlan,
) -> Result<std::collections::HashMap<String, crate::remote::RepackResult>> {
    let mut password = None;
    let session = r
        .open_with(&mut password)
        .with_context(|| format!("opening an ssh session to {}", r.host))?;
    // The remote reports bytes *per tensor*, restarting at zero for each one, so they are accumulated
    // here into a run total. Storing them directly made the counter jump backwards mid-run —
    // 27.14 GB, then 0.01 GB — which reads as a bug in the tool rather than as the next tensor starting.
    let mut counted = ByteTally::default();
    let (map, _stats) = r.verify_repack(
        &session,
        old_side,
        new_side,
        &plan.pairs,
        plan.bits,
        plan.widths(),
        false,
        // The job's own flag, so *Stop* tears the channel down and the remote verification ends with
        // it — rather than stopping the waiting while the proxy carries on alone.
        Some(job.read_progress().abort_flag()),
        |ev| on_event(job, &ev, &mut counted),
    )?;
    record(job, plan, &map);
    Ok(map)
}

/// Bytes read across a whole run, from per-tensor counters that restart.
#[derive(Default)]
struct ByteTally {
    /// Finished tensors' bytes.
    base: u64,
    /// The tensor being read, and how much of it has been counted.
    current: (String, u64),
}

impl ByteTally {
    /// Fold one per-tensor reading into the run total, and return it.
    fn observe(&mut self, name: &str, bytes: u64) -> u64 {
        if self.current.0 != name {
            // A new tensor: bank what the previous one reached.
            self.base += self.current.1;
            self.current = (name.to_string(), 0);
        }
        // `max`, not assignment: the two sides are reported independently and their sum can arrive out
        // of order, and a total that goes backwards is worse than one that is briefly behind.
        self.current.1 = self.current.1.max(bytes);
        self.base + self.current.1
    }
}

/// Turn one remote progress event into job progress. Bytes matter here: a single expert weight can be
/// gigabytes, so "tensor 3 of 19" alone would sit still for minutes.
fn on_event(job: &Job, ev: &crate::remote::RepackEvent<'_>, counted: &mut ByteTally) {
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
        // The announced sizes, the decode step and per-tensor completion: all reported through
        // `Start`/`Bytes` already, or in the findings. Listed rather than wildcarded so a new event
        // variant is a compile error here and gets a decision.
        crate::remote::RepackEvent::Size { .. }
        | crate::remote::RepackEvent::Comparing { .. }
        | crate::remote::RepackEvent::Done { .. } => {}
    }
}

/// Verify from local files.
///
/// Delegates to the `diff` subcommand's own `local_repack`, which assembles each pair's sibling codebook
/// and qscale tensors before decoding — that assembly is the fiddly part, and a second copy of it is a
/// second thing to get subtly wrong. Sequential there too: each expert weight is read whole, so memory
/// stays at one pair at a time.
///
/// The trade-off is that progress arrives per *run* rather than per tensor, since that function reports
/// none. Local checkpoints are the fast case (no S3 round trips), so a job that says "working" and then
/// hands over every finding at once is honest here; the remote path, which is the slow one, streams.
fn verify_locally(
    job: &Job,
    old_t: &[crate::tree::TensorInfo],
    new_t: &[crate::tree::TensorInfo],
    plan: &crate::compare::RepackPlan,
) -> std::collections::HashMap<String, crate::remote::RepackResult> {
    job.progress_to(0, "decoding both packings");
    let out = crate::local_repack(old_t, new_t, &plan.pairs, plan.bits, plan.schemas(), None);
    job.progress_to(plan.pairs.len(), "");
    record(job, plan, &out);
    out
}

/// Append one finding per verified pair, in the plan's order.
fn record(
    job: &Job,
    plan: &crate::compare::RepackPlan,
    results: &std::collections::HashMap<String, crate::remote::RepackResult>,
) {
    for (_, name) in &plan.pairs {
        let Some(r) = results.get(name) else { continue };
        job.add_finding(json!({
            "kind": "tensor",
            "name": name,
            // The numbers the CLI prints per tensor: how many decoded indices differ, and how far.
            // `differing == 0` is the whole point — the two packings hold the same weights.
            "elements": r.elements,
            "differing": r.differing,
            "max_delta": r.max_delta,
            "differing_gt1": r.differing_gt1,
            "mean_abs": r.mean_abs,
            "mean_old": r.mean_old,
            "mean_new": r.mean_new,
            // Non-zero means the words do not look like packed indices at this width, so the
            // comparison's premise is wrong — worth showing rather than burying.
            "sparse_bad": r.sparse_bad,
            "dense_bad": r.dense_bad,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::ByteTally;

    /// The run total must never go backwards.
    ///
    /// The remote counts each tensor from zero, so storing its reading directly made the job report
    /// 27.14 GB and then 0.01 GB when the next tensor started — which reads as a broken tool. Seen in a
    /// real 318-second run before it was fixed.
    #[test]
    fn bytes_accumulate_across_tensors_and_never_regress() {
        let mut tally = ByteTally::default();
        assert_eq!(tally.observe("a", 100), 100);
        assert_eq!(tally.observe("a", 500), 500);
        // The next tensor restarts at zero; the total carries `a`'s 500 with it.
        assert_eq!(tally.observe("b", 10), 510);
        assert_eq!(tally.observe("b", 300), 800);
        // A third, and the total keeps climbing.
        assert_eq!(tally.observe("c", 1), 801);
    }

    /// The two sides are reported independently, so a reading can arrive lower than one already seen.
    /// Taking the max keeps the total monotonic rather than letting it dip.
    #[test]
    fn an_out_of_order_reading_does_not_lower_the_total() {
        let mut tally = ByteTally::default();
        assert_eq!(tally.observe("a", 900), 900);
        assert_eq!(tally.observe("a", 400), 900, "a lower reading is ignored");
        assert_eq!(tally.observe("a", 1_000), 1_000);
    }
}
