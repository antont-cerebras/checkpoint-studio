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
    /// Two `s3://` checkpoints, decoded on the ssh proxy — only results cross the wire.
    Proxy,
    /// Local files on both sides, decoded here.
    Local,
}

/// Read one side far enough to verify it: its tensors, and the proxy it lives behind.
struct Side {
    tensors: Vec<crate::tree::TensorInfo>,
    metadata: Vec<crate::tree::MetadataInfo>,
    /// The `s3://` URI, when this side is one — `verify_repack` addresses objects by URI.
    s3_uri: Option<String>,
    remote: Option<crate::remote::RemoteRead>,
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
    let opened = target
        .read(crate::opening::Want::Parts, job.read_progress())
        .with_context(|| format!("reading {spec}"))?;
    let (tensors, metadata) = (opened.parts.tensors, opened.parts.metadata);
    Ok(Side {
        tensors,
        metadata,
        s3_uri,
        remote,
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

    // **Which mode, before planning.** Refusing an unsupported pair outright is the first thing that
    // happens, because planning first would report "no fold-pair tensors matched" — a true statement
    // about the wrong problem. The precondition itself is `compare::repack_supported`, shared with the
    // `diff` subcommand so a pair either surface refuses is refused by both, in the same words.
    let both_s3 = old.s3_uri.is_some() && new.s3_uri.is_some();
    let anywhere_remote = old.remote.is_some() || new.remote.is_some();
    crate::compare::repack_supported(anywhere_remote, both_s3)?;
    let mode = if anywhere_remote {
        Mode::Proxy
    } else {
        // Local files on both sides: decode here.
        Mode::Local
    };

    let plan = crate::compare::plan_repack(&old_sum, &new_sum, bits)?;
    job.set_total(plan.pairs.len());

    let results = match mode {
        Mode::Proxy => {
            let (Some(r), Some(old_uri), Some(new_uri)) = (
                old.remote.as_ref(),
                old.s3_uri.as_deref(),
                new.s3_uri.as_deref(),
            ) else {
                bail!(
                    "verifying two s3:// checkpoints needs an ssh proxy to decode on — start the \
                     server with --ssh-proxy, or set ssh_proxy in the config"
                );
            };
            verify_on_proxy(r, job, old_uri, new_uri, &plan)?
        }
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
    old_uri: &str,
    new_uri: &str,
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
        old_uri,
        new_uri,
        &plan.pairs,
        plan.bits,
        false,
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
        | crate::remote::RepackEvent::Comparing(_)
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
    let out = crate::local_repack(old_t, new_t, &plan.pairs, plan.bits, None);
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
