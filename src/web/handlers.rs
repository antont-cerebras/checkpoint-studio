//! One function per API route. Each takes `&WebState` (+ the parsed query) and
//! returns `(status, json)` — no socket, so they're unit-testable directly. The
//! metadata/view routes read precomputed state (instant); the `/api/tensor/*`
//! data routes read tensor bytes on demand (local-only) via `crate::sample`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde::Serialize;
use serde_json::{Value, json};

use super::WebState;
use super::current::{ComparisonLookup, WhenBusy};
use crate::sample::{self, SampleMode, ViewDtype};
use crate::tree::TensorInfo;
use crate::web::dto::{self, HistogramDto, SampleDto, StatsDto};

pub(crate) type Query = HashMap<String, String>;
/// An HTTP status plus the response body ALREADY serialised to JSON bytes (not a
/// `serde_json::Value`) — see `ok`.
pub(crate) type Reply = (u16, Vec<u8>);

fn ok<T: Serialize>(v: T) -> Reply {
    // Serialise STRAIGHT to bytes. Going via `serde_json::to_value` first materialised
    // the whole response as a `Value` tree — for `/api/tree` on a 31k-tensor checkpoint
    // that was ~250 MB of transient allocation and a second full pass over the data,
    // every request.
    (
        200,
        serde_json::to_vec(&v).unwrap_or_else(|_| b"null".to_vec()),
    )
}

pub(crate) fn err(status: u16, msg: impl Into<String>) -> Reply {
    let body = json!({ "error": msg.into() });
    (
        status,
        serde_json::to_vec(&body)
            .unwrap_or_else(|_| br#"{"error":"serialisation failed"}"#.to_vec()),
    )
}

/// How to treat a read that is already running, from `?stop_other=1`.
///
/// Opt-in per request rather than a server setting: stopping someone else's read is a decision the
/// person making it should have taken, and the refusal that precedes it names what would be stopped.
fn when_busy(q: &Query) -> Result<WhenBusy, Reply> {
    Ok(if switch(q, "stop_other")? {
        WhenBusy::StopTheOther
    } else {
        WhenBusy::Refuse
    })
}

/// A refusal that offers the way out, as the `{error}` envelope plus the fields a UI needs to render a
/// button rather than a sentence telling the reader to wait.
fn busy_reply(current: &super::Current, e: &anyhow::Error) -> Reply {
    let (spec, secs) = current.busy_with().unwrap_or_else(|| (String::new(), 0.0));
    let body = json!({
        "error": format!("{e:#}"),
        // What is running, so the offer can name it, and a flag so the client does not have to
        // pattern-match on prose to know a retry-with-takeover is available.
        "busy_with": spec,
        "busy_for_seconds": secs,
        "can_stop_other": true,
    });
    (
        409,
        serde_json::to_vec(&body)
            .unwrap_or_else(|_| br#"{"error":"serialisation failed"}"#.to_vec()),
    )
}

/// Data-value views need the tensor bytes locally; a remote (`--ssh-proxy`) source
/// only carries its structure. Returns a friendly 400 for a remote tensor (so the
/// UI shows a clear note instead of a cryptic open-file failure), else `None`.
/// Reject a data request when the source can't give us tensor bytes, with the reason that
/// source has. Asks the **capability**, not "is the path remote": a Hugging Face repo and an
/// ssh-proxied directory both lack byte access but for different reasons, and the note that
/// explains it lives with the capability so the terminal says the same thing.
fn require_bytes(s: &WebState) -> Option<Reply> {
    let caps = s.checkpoint.capabilities();
    if caps.read_bytes {
        return None;
    }
    Some(err(
        400,
        crate::capability::Capabilities::data_view_note(s.checkpoint.location())
            .unwrap_or("This checkpoint's tensor data is not reachable from here."),
    ))
}

// ---- changing which checkpoint is served ----

/// `POST /api/open?path=SPEC` — read another checkpoint and serve it instead.
///
/// Synchronous: the response comes back when the new checkpoint is *ready*, which is what
/// lets the client treat "this resolved" and "the data is there" as one thing rather than
/// polling for a state it then has to reconcile. The read can take seconds (longer over ssh),
/// and the browser shows its elapsed timer meanwhile; every other request keeps being served
/// from the previous checkpoint until the swap lands (see `crate::web::current`).
///
/// The reply is deliberately small — the client refetches from the ordinary endpoints once
/// this returns, rather than this one trying to bundle a new tree, files and stats into a
/// response that would duplicate four routes.
pub(crate) fn open(current: &super::Current, q: &Query) -> Reply {
    let Some(spec) = q.get("path").map(String::as_str).filter(|p| !p.is_empty()) else {
        return err(
            400,
            "open needs ?path=SPEC (a checkpoint file, directory, glob, hf:// repo, or a \
             path on the ssh proxy)",
        );
    };
    let busy = match when_busy(q) {
        Ok(v) => v,
        Err(e) => return e,
    };
    match current.open(spec, busy) {
        Ok(state) => ok(json!({
            "root": state.root,
            "tensor_count": state.tensors.len(),
            // What opened, as typed — so the client can show the recents list without a
            // second request, and see its own entry in it.
            "opened": spec,
            "recents": current.recents(),
        })),
        // Something else is mid-read: a 409 with the offer to stop it, not a 400 — the request was
        // fine, the server was busy.
        Err(e) if current.busy_with().is_some() => busy_reply(current, &e),
        // A spec that doesn't resolve is a client mistake, not a server fault: this message
        // is what the open prompt shows inline, so it carries the whole `anyhow` chain
        // (`opening /nope: no checkpoint files found at …`).
        Err(e) => err(400, format!("{e:#}")),
    }
}

/// `DELETE /api/recents?path=SPEC` — drop one checkpoint from the list.
///
/// A `DELETE`, not another `POST`: it removes one identified thing, which is what the verb is
/// for, and it keeps the "anything that changes server state is not a GET" rule intact.
///
/// Removing something that is not in the list is a 404 rather than a silent success: the client
/// shows the list it is deleting from, so a miss means the two disagree and saying so is more
/// useful than pretending.
pub(crate) fn forget_recent(current: &super::Current, q: &Query) -> Reply {
    let Some(spec) = q.get("path").map(String::as_str).filter(|p| !p.is_empty()) else {
        return err(400, "forgetting a checkpoint needs ?path=SPEC");
    };
    if current.forget_recent(spec) {
        ok(json!({ "forgot": spec, "recents": current.recents() }))
    } else {
        err(404, format!("not in the recents list: {spec}"))
    }
}

/// `GET /api/version` — which build of the app this is, so a browser tab can tell it has gone stale.
///
/// A tab outlives the server it was loaded from. This project restarts the server under open tabs as a
/// matter of routine, and the failure that produces is silent and unbounded: an old client reading a
/// newer response shape declared two checkpoints that share no tensor name "structurally identical",
/// because every counter it looked for was missing and `NaN > 0` is false. One defensive check fixed
/// that symptom; this answers the question behind it.
///
/// `assets` is the entry script's hashed name — the identity of the UI being served, which the tab
/// compares against the script it is running (`web/src/lib/build.ts`). `app` is the binary's version,
/// for a human reading the reply.
pub(crate) fn version(s: &WebState) -> Reply {
    ok(json!({
        "app": env!("CARGO_PKG_VERSION"),
        "assets": super::assets::build_id(),
        // Which checkpoint this server holds, so the one cheap poll answers "is anything about this
        // server different from when I loaded" rather than only "is the UI different".
        "spec": if s.spec.is_empty() { &s.root } else { &s.spec },
    }))
}

/// `GET /api/reading` — how far the read in flight has got, or `{"reading":null}`.
///
/// Polled while a wait is on screen. `/api/open` and `/api/compare` are synchronous — the answer means
/// *ready*, which is what spares the client a state machine — so this is the only way the browser can
/// see inside a read it is waiting on. Without it the wait had an elapsed timer and no numbers, while a
/// terminal reading the same checkpoint counted `1155/1155 S3 objects`.
///
/// Deliberately not part of the busy refusal: that says what is *blocking you*, this says how the thing
/// you asked for is going, and they are different questions asked by different code.
pub(crate) fn reading(current: &super::Current) -> Reply {
    ok(json!({ "reading": current.reading() }))
}

/// `GET /api/recents` — the checkpoints opened this run, most recent first.
///
/// Not folded into `/api/tree`: that body is encoded once per checkpoint and cached, so a
/// list that grows with each open would be served stale from it.
pub(crate) fn recents(current: &super::Current) -> Reply {
    ok(json!({
        "recents": current.recents(),
        // Whether this server reads over an ssh proxy, so the prompt can say which kind of
        // path it takes: a `--ssh-proxy` server opens remote paths, a local one local ones.
        "proxied": current.is_proxied(),
        // And *which* host, so the client can show `:/path` as the address it resolves to.
        "proxy_host": current.proxy_host(),
    }))
}

// ---- comparing two checkpoints ----

/// A checkpoint named for a client: what to address it by, what to label it, and whether it is the
/// one being *served*.
///
/// `served` is stated rather than left to be derived. The client used to decide it by string-comparing
/// this `spec` against `/api/tree`'s, which only worked because both are built by the same expression
/// — and which produced a false "not loaded" for any two spellings that resolve to the same
/// checkpoint (a glob versus the directory it expands to, `:path` versus `host:/path`). The server
/// holds the fact outright, so it says it. Same rule as `crate::capability`: record it at the source,
/// never re-derive it from a path's shape.
fn side_json(s: &WebState, served: bool, totals: crate::diff::Footprint) -> Value {
    json!({
        "spec": if s.spec.is_empty() { &s.root } else { &s.spec },
        "root": s.root,
        "tensor_count": s.tensors.len(),
        "served": served,
        // The two overall totals, so the side-by-side can head itself with the same `size:` and
        // `params:` lines the report and the terminal show. It had neither, which made it the one
        // view of a re-quantization that never said the checkpoint got four times smaller.
        //
        // Passed in rather than read off `s`, because under a scope they cover the *selected* tensors:
        // the rows on this screen are the selection, and totals describing the whole checkpoints would
        // be a true statement about something the reader is not looking at (see `totals_of`).
        "params": totals.params,
        "bytes": totals.bytes,
    })
}

/// The summed footprint of `tensors`, or of the subset `keep` names.
///
/// Over the **deduped** canonical list, like `diff::CheckpointSummary::from_loaded`: a sharded
/// checkpoint lists a shared name once per shard, and counting it per shard would put a different total
/// on the side-by-side than on the report — which a test pins together.
fn totals_of(
    tensors: &[TensorInfo],
    keep: Option<&std::collections::HashSet<String>>,
) -> crate::diff::Footprint {
    tensors
        .iter()
        .filter(|t| keep.is_none_or(|k| k.contains(&t.name)))
        .fold(crate::diff::Footprint::default(), |acc, t| {
            crate::diff::Footprint {
                bytes: acc.bytes + t.size_bytes,
                params: acc.params + t.num_elements,
                // A total, not a fold: `parts` counts tensors behind *one name*, and this is a sum
                // over many names.
                parts: 1,
            }
        })
}

/// `POST /api/compare?left=SPEC&right=SPEC` — set up a comparison between two checkpoints.
///
/// `right` may be omitted to mean "the checkpoint that is open", which is the common case and costs
/// no second read. Naming a different one compares two checkpoints that are both other than the
/// served one — either way the served checkpoint stays loaded and untouched, which is what makes the
/// right-hand box overridable rather than decorative.
///
/// Synchronous like `/api/open`, and for the same reason: the answer means "ready", so the client has
/// no state to poll and reconcile.
pub(crate) fn set_comparison(current: &super::Current, q: &Query) -> Reply {
    let Some(left) = q.get("left").map(String::as_str).filter(|p| !p.is_empty()) else {
        return err(
            400,
            "a comparison needs ?left=SPEC (the baseline), and optionally &right=SPEC",
        );
    };
    let right = q.get("right").map_or("", String::as_str);
    let busy = match when_busy(q) {
        Ok(v) => v,
        Err(e) => return e,
    };
    match current.set_comparison(left, right, busy) {
        // The id is what the follow-up `/api/difftree` must quote, and the two specs are what they
        // *resolved* to — so the client can check that the comparison it gets back is the one it asked
        // for, without re-deriving a resolution only this side performed.
        Ok(set) => ok(json!({
            "id": set.id,
            "left": set.left_spec,
            "right": set.right_spec,
            "recents": current.recents(),
        })),
        Err(e) if current.busy_with().is_some() => busy_reply(current, &e),
        Err(e) => err(400, format!("{e:#}")),
    }
}

/// `DELETE /api/compare` — drop the comparison, freeing whatever it held.
pub(crate) fn clear_comparison(current: &super::Current) -> Reply {
    current.clear_comparison();
    ok(json!({ "comparison": Value::Null }))
}

/// `GET /api/difftree?id=N` — the two checkpoints aligned into one tree.
///
/// The whole side-by-side comes from this one response: one row per name, each side's content on its
/// own side, a per-row status, per-group differing counts, and the ordered list of differences that
/// `n`/`N` step through. Aligning here rather than in each frontend is what stops the terminal and
/// the browser from drawing different comparisons of the same two checkpoints
/// (see `checkpoint_studio_core::difftree`).
///
/// **`id` is required.** There is one comparison slot per server, and this route used to answer from
/// whatever was in it. Two overlapping clients therefore swapped results — A set up its pair, B
/// replaced it, and A's request returned B's comparison with a `200`. Quoting the id from
/// `POST /api/compare` makes that a `409` instead of a wrong answer.
pub(crate) fn difftree(current: &super::Current, _s: &WebState, q: &Query) -> Reply {
    // The request's own parameters first: a typo is a typo whether or not the comparison it names
    // still exists, and reporting the id problem for a malformed switch sends the reader looking in
    // the wrong place.
    let full = match switch(q, "full") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let (base, right) = match comparison_sides(current, q, "difftree") {
        Ok(pair) => pair,
        Err(e) => return e,
    };
    let served = current.snapshot();
    // Identity, not string equality: whether a side *is* the served state is a fact about the
    // pointers, and no spelling of a path can make it wrong.
    let base_served = Arc::ptr_eq(&base, &served);
    let right_served = Arc::ptr_eq(&right, &served);
    let s = right.as_ref();

    let scope = match super::diffscope::DiffScope::from_query(q) {
        Ok(scope) => scope,
        Err(e) => return err(400, format!("{e:#}")),
    };
    // Rename rules first, then scoping — the CLI's order, and the reason both are here rather than only
    // on the report: a side-by-side that ignored `--map` would compare two schemes that were meant to be
    // lined up, and read as every tensor added and removed.
    //
    // The baseline's tree is *rebuilt* from the renamed names, because groups are named from name
    // segments and rewriting a leaf alone leaves the path above it describing the old name.
    let base_meta = base.checkpoint.metadata_vec();
    let right_meta = s.checkpoint.metadata_vec();
    // A `#subtree` on either side first: it re-keys that side's tensors to their sub-path, so the two
    // checkpoints line up under different namespaces (`hf#language_model` against a converted
    // `model.…`), and everything below — renames, filters, totals — is written against those names.
    // Rebuilt trees, because a group is named from name segments.
    let base_sub = scope.reroot_tensors(super::diffscope::Sub::Baseline, &base.tensors);
    let right_sub = scope.reroot_tensors(super::diffscope::Sub::Newer, &s.tensors);
    let renamed = scope.rename_tensors(&base_sub);
    let (base_tensors, rename_collisions) = (renamed.tensors, renamed.collisions);
    let renamed_tree = (scope.has_rename_rules() || scope.reroots())
        .then(|| crate::difftree::tree_from_tensors(&base.root, &base_tensors, &base_meta));
    // The newer side is rebuilt only when *it* is re-rooted: no rename rule touches it.
    let right_tree = scope
        .subtrees()
        .1
        .map(|_| crate::difftree::tree_from_tensors(&s.root, &right_sub, &right_meta));

    // Scope the trees, if a *filter* narrows them.
    //
    // `scope.compare` is given the **original** names and renames them itself — passing the renamed ones
    // would apply the rules twice, which is a no-op for most rules and silently wrong for any that match
    // their own output. Its `matched` names are therefore post-rename, which is what the rebuilt tree's
    // leaves are called.
    //
    // Keyed on `matched`, not on `is_active()`: a scope can be active because of a rename rule alone, and
    // an empty keep-set then pruned *every* row — a comparison of two lined-up checkpoints came back
    // completely empty rather than as two unchanged tensors.
    // The summaries are built from the **re-rooted** names, and `scope.compare` re-roots nothing it has
    // already been given — it renames and filters from here.
    let scoped = scope.is_active().then(|| {
        scope.compare(
            crate::diff::CheckpointSummary::from_loaded(&base_sub, &base_meta),
            crate::diff::CheckpointSummary::from_loaded(&right_sub, &right_meta),
        )
    });
    let matched = scoped.and_then(|out| out.matched);
    // The selected names, once — the pruning below and the totals both work from it.
    let keep: Option<std::collections::HashSet<String>> =
        matched.as_ref().map(|m| m.names.iter().cloned().collect());
    // Which tree each side starts from: the rebuilt one when a rename rule or a `#subtree` changed the
    // names, else the one the read already built.
    let base_full = renamed_tree.as_deref().unwrap_or(&base.tree);
    let right_full = right_tree.as_deref().unwrap_or(&s.tree);
    let pruned = keep.as_ref().map(|keep| {
        (
            scope.prune_tree(base_full, keep),
            scope.prune_tree(right_full, keep),
        )
    });
    let base_rows = pruned.as_ref().map_or(base_full, |(b, _)| b.as_slice());
    let right_rows = pruned.as_ref().map_or(right_full, |(_, r)| r.as_slice());

    // `align_rooted`: each tree hangs under a root named after its own checkpoint, and those
    // names never match across two files — aligning the roots would pair nothing.
    let mut rows = crate::difftree::align_rooted(base_rows, right_rows);
    // A folded baseline leaf says how many tensors it stands for (`×256`), because the fused side has
    // one of them and the question is whether the conversion kept them all. On the aligned rows, so the
    // note sits on the side it describes.
    crate::difftree::note_folds(&mut rows, &renamed.folds);
    // The headline is counted over **every** row, before families are folded: turning a view control on
    // must not change what the comparison says. (`tally` below reads `rows`, which is why this is taken
    // here and the folded tree is a separate value.)
    let tally = crate::difftree::tally(&rows);
    // Uniform layers fold into one row unless the reader asked for all of them (`full`, read at the
    // top) — the report's default, and the terminal's, for the same reason: 62 rows differing only by
    // a layer number say less than one row that says so.
    let rows = if full {
        rows
    } else {
        crate::difftree::fold_families(&rows)
    };
    // What `n`/`N` walk — over the rows actually returned, so a jump lands on a row that is on screen.
    let differences = crate::difftree::differences(&rows);
    // The totals follow the scope: over the selected tensors when there is a selection, over the
    // checkpoint when there is not. The baseline's are summed from its **renamed** tensors, since that
    // is what the selection names.
    let base_totals = totals_of(&base_tensors, keep.as_ref());
    let right_totals = totals_of(&right_sub, keep.as_ref());
    let (size_label, params_label) = crate::diff::totals_labels(scope.is_filtered());
    ok(json!({
        "base": side_json(&base, base_served, base_totals),
        "current": side_json(s, right_served, right_totals),
        // What to call those totals — `size (filtered subset)` under a filter, since then they describe
        // the rows on screen rather than the checkpoints. The server words it, so the two views and the
        // terminal cannot label the same numbers differently.
        "totals_labels": { "size": size_label, "params": params_label },
        // The headline, counted here so the side-by-side and the one-page report cannot print
        // different totals for the same pair — which they did.
        "tally": tally,
        // Whether the rows below are every layer or families folded onto one row each. The server says
        // so, because the client's checkbox and the tree it is looking at can be one request apart.
        "full": full,
        // What the scope selected, in the report's own words. `null` when nothing narrowed it.
        "matched": matched.as_ref().map(|m| json!({
            "selected": m.selected,
            "total": m.total,
        })),
        // Two old names that map onto one lose a tensor from the comparison; the report says so too.
        "rename_collisions": rename_collisions,
        // What `n`/`N` walk, in draw order. Precomputed server-side because the walk is over the
        // whole tree and the client would otherwise redo it on every keypress.
        "differences": differences,
        "rows": rows,
    }))
}

// ---- long-running work ----

/// `POST /api/jobs/verify-repack?left=SPEC&right=SPEC&<scope>[&repack_bits=N]`
///
/// The browser's `diff --verify-repack`: do two checkpoints hold the **same weights in different
/// packings**? Reads both tensors of every candidate pair, so it takes minutes and reports per-tensor
/// findings as they land — hence a job rather than a response (see [`super::jobs`]).
///
/// Answers immediately with the id to poll. The work runs on its own thread, so the request does not
/// hold a `tiny_http` worker for the run.
pub(crate) fn start_verify_repack(current: &Arc<super::Current>, q: &Query) -> Reply {
    let Some(left) = q.get("left").map(String::as_str).filter(|s| !s.is_empty()) else {
        return err(
            400,
            "verify-repack needs ?left=SPEC&right=SPEC (the two checkpoints to compare)",
        );
    };
    let right = q.get("right").map_or("", String::as_str);
    if right.is_empty() {
        return err(400, "verify-repack needs ?right=SPEC as well as ?left=SPEC");
    }
    let scope = match super::diffscope::DiffScope::from_query(q) {
        Ok(scope) => scope,
        Err(e) => return err(400, format!("{e:#}")),
    };
    if scope.reroots() {
        return err(
            400,
            "a #subtree comparison is structure-only here — clear the subtree fields to verify a repack",
        );
    }
    // Parsed here so a bad value is a 400 rather than a job that fails a minute later.
    let bits = match q.get("repack_bits").map(|v| v.parse::<usize>()) {
        None => None,
        Some(Ok(n)) if (1..=16).contains(&n) => Some(n),
        Some(_) => return err(400, "repack_bits must be a whole number from 1 to 16"),
    };

    let job = current.jobs().start("verify-repack");
    let id = job.id;
    let (owner, left, right) = (Arc::clone(current), left.to_string(), right.to_string());
    // A named thread, so a stuck run is identifiable in a debugger or a `ps` listing.
    let spawned = std::thread::Builder::new()
        .name(format!("verify-repack-{id}"))
        .spawn(move || {
            let outcome = super::repackjob::run(&owner, &job, &left, &right, &scope, bits);
            super::jobs::Jobs::finish(&job, outcome);
        });
    match spawned {
        Ok(_) => ok(json!({ "id": id })),
        Err(e) => err(500, format!("could not start the job: {e}")),
    }
}

/// `POST /api/jobs/values?left=SPEC&right=SPEC&<scope>[&values=1][&histogram=1][&bins=N][&dtype=V][&jobs=N][&tensor=NAME]`
///
/// The browser's `diff --values` / `--histogram` / `--tensor`: do the *numbers* differ, not just the
/// structure? Reads every selected tensor on both sides, so a job (see [`super::jobs`]).
///
/// At least one of `values` / `histogram` must be asked for — a job that computed neither would read
/// every tensor and report nothing.
pub(crate) fn start_values(current: &Arc<super::Current>, q: &Query) -> Reply {
    let Some(left) = q.get("left").map(String::as_str).filter(|s| !s.is_empty()) else {
        return err(400, "values needs ?left=SPEC&right=SPEC");
    };
    let right = q.get("right").map_or("", String::as_str);
    if right.is_empty() {
        return err(400, "values needs ?right=SPEC as well as ?left=SPEC");
    }
    // Refused **now**, from the two addresses, rather than after reading both checkpoints: a remote
    // source hands over no tensor data, and the job used to discover that having already spent the
    // minutes the answer would have saved. `--verify-repack` has always checked its own support up
    // front, for the same reason.
    if let Err(e) = crate::compare::values_where(left, right, current.proxy_host()) {
        return err(400, format!("{e:#}"));
    }
    let scope = match super::diffscope::DiffScope::from_query(q) {
        Ok(scope) => scope,
        Err(e) => return err(400, format!("{e:#}")),
    };
    // A `#subtree` re-root is structure-only here: the value comparison reads tensors by their real
    // names, and re-rooting changes only the *match key*. Refused rather than ignored — a job that
    // quietly compared the whole checkpoint would answer a different question than the screen asked.
    if scope.reroots() {
        return err(
            400,
            "a #subtree comparison is structure-only here — clear the subtree fields to compare values",
        );
    }
    let (want_values, want_hist) = match (switch(q, "values"), switch(q, "histogram")) {
        (Ok(v), Ok(h)) => (v, h),
        (Err(e), _) | (_, Err(e)) => return e,
    };
    // `--tensor` compares values by definition, so it implies them rather than needing both flags.
    let one = q
        .get("tensor")
        .map(String::as_str)
        .filter(|s| !s.is_empty());
    if !want_values && !want_hist && one.is_none() {
        return err(
            400,
            "ask for values=1 and/or histogram=1 (or tensor=NAME) — otherwise this would read every \
             tensor and report nothing",
        );
    }
    let bins = match q.get("bins").map(|v| v.parse::<usize>()) {
        None => None,
        Some(Ok(n)) if (1..=512).contains(&n) => Some(n),
        Some(_) => return err(400, "bins must be a whole number from 1 to 512"),
    };
    // `view_of` is what every `/api/tensor/*` route uses, so `dtype` spells the same thing everywhere.
    let view = match view_of(q) {
        Ok(view) => view,
        Err(reply) => return reply,
    };
    // Default parallelism = logical CPUs, as the CLI does; `jobs=0` is treated as sequential.
    let jobs = match q.get("jobs").map(|v| v.parse::<usize>()) {
        None => std::thread::available_parallelism().map_or(4, std::num::NonZero::get),
        Some(Ok(n)) => n.max(1),
        Some(Err(_)) => return err(400, "jobs must be a whole number"),
    };

    let what = super::valuesjob::What {
        values: want_values || one.is_some(),
        histogram: want_hist,
        bins,
        view,
        jobs,
        tensor: one.map(str::to_string),
    };
    let job = current.jobs().start("values");
    let id = job.id;
    let (owner, left, right) = (Arc::clone(current), left.to_string(), right.to_string());
    let spawned = std::thread::Builder::new()
        .name(format!("values-{id}"))
        .spawn(move || {
            let outcome = super::valuesjob::run(&owner, &job, &left, &right, &scope, &what);
            super::jobs::Jobs::finish(&job, outcome);
        });
    match spawned {
        Ok(_) => ok(json!({ "id": id })),
        Err(e) => err(500, format!("could not start the job: {e}")),
    }
}

/// `GET /api/jobs/<id>` — where a job has got to, and what it has found so far.
pub(crate) fn job_status(current: &super::Current, id: u64) -> Reply {
    // 404, not 409: an id that was never handed out, or one evicted long after finishing.
    current.jobs().get(id).map_or_else(
        || err(404, format!("no job {id}")),
        |job| ok(job.snapshot()),
    )
}

/// `DELETE /api/jobs/<id>` — ask a job to stop.
///
/// Cooperative, like every other cancellation here: it sets the flag the remote reader checks between
/// chunks. The reply says the request was accepted, not that the work has already stopped — a poll
/// reports `cancelled` once it has.
pub(crate) fn cancel_job(current: &super::Current, id: u64) -> Reply {
    current.jobs().get(id).map_or_else(
        || err(404, format!("no job {id}")),
        |job| {
            job.read_progress().cancel();
            ok(job.snapshot())
        },
    )
}

// ---- metadata / derived-view routes (served from precomputed state) ----

pub(crate) fn tree(s: &WebState) -> Reply {
    ok(json!({
        "root": s.root,
        // What to *address* this checkpoint by, which is not always its display root — see
        // `WebState::spec`. Falls back to the root for a state built without one.
        "spec": if s.spec.is_empty() { &s.root } else { &s.spec },
        "tensor_count": s.tensors.len(),
        // What this source can do, so the client asks a capability instead of guessing from
        // the source's shape (see `crate::capability`).
        "capabilities": s.checkpoint.capabilities(),
        "format": s.checkpoint.format(),
        "location": s.checkpoint.location(),
        // Why a data view is unavailable, or `null` when it isn't — from the one function
        // that words it, so the pane the client disables says what the 400 would have.
        "data_view_note": crate::capability::Capabilities::data_view_note(
            s.checkpoint.location(),
        ),
        // `null` unless the server is reachable off this machine — the client shows it as
        // a banner. Carried on the tree because that is the first thing the UI fetches.
        "access_warning": s.access_warning,
        // Already rooted by `Session::build_rooted_tree` — the same tree, with the same
        // summarising root and label, that the TUI renders.
        "tree": s.tree,
        // Which tensors are extras — on disk but absent from the index. Sent with the
        // tree because that is where they're marked, and keyed on `source_path` so the
        // client's test is the same one the terminal makes.
        "unindexed": s.unindexed,
    }))
}

pub(crate) fn files(s: &WebState) -> Reply {
    ok(&s.file_tree)
}

/// Rich tensor filtering: parse the `?q=` text query (see [`crate::tensorfilter`])
/// with the shared matcher and return the names of the tensors that pass, so the
/// client masks its tree to them. `active:false` for an empty query (show all); a
/// malformed query is a `400` whose message the filter bar shows inline.
pub(crate) fn filter(s: &WebState, q: &Query) -> Reply {
    let query = q.get("q").map_or("", String::as_str);
    match crate::tensorfilter::TensorFilter::parse(query) {
        Ok(f) if !f.is_active() => ok(json!({ "active": false })),
        Ok(f) => {
            let names: Vec<&str> = s
                .tensors
                .iter()
                .filter(|t| f.matches(t))
                .map(|t| t.name.as_str())
                .collect();
            ok(json!({ "active": true, "names": names }))
        }
        Err(e) => err(400, e.to_string()),
    }
}

/// The **compact tree**: the tensor tree with uniform layer / expert stacks folded into
/// single templated subtrees, so only irregularities stand out
/// ([`crate::compact::compact_tree`]). Optionally scoped by `?q=` — the same filter
/// grammar the tree and `/api/filter` use — so you can fold a subset.
///
/// This replaced a flat per-family *list* (`/api/schema`, still served): a list destroys
/// the nesting, and the nesting is what makes an outlier visible next to its siblings.
pub(crate) fn compact(s: &WebState, q: &Query) -> Reply {
    let query = q.get("q").map_or("", String::as_str);
    let filter = match crate::tensorfilter::TensorFilter::parse(query) {
        Ok(f) => f,
        Err(e) => return err(400, e.to_string()),
    };
    if filter.is_active() {
        let scoped: Vec<TensorInfo> = s
            .tensors
            .iter()
            .filter(|t| filter.matches(t))
            .cloned()
            .collect();
        return ok(crate::compact::compact_rooted(&scoped, &s.files));
    }
    ok(crate::compact::compact_rooted(&s.tensors, &s.files))
}

/// Compact per-family listing: collapse the (optionally `?q=`-filtered) tensors into
/// index-templated families (`model.layers.{0-47}.…experts.{0-3}.down_proj.weight`)
/// with per-family count + uniform dtype/shape + total params/bytes — a "what's in
/// here, per layer / per expert" summary (same collapsing as `diff`).
pub(crate) fn schema(s: &WebState, q: &Query) -> Reply {
    let query = q.get("q").map_or("", String::as_str);
    let filter = match crate::tensorfilter::TensorFilter::parse(query) {
        Ok(f) => f,
        Err(e) => return err(400, e.to_string()),
    };
    let families = if filter.is_active() {
        let matched: Vec<TensorInfo> = s
            .tensors
            .iter()
            .filter(|t| filter.matches(t))
            .cloned()
            .collect();
        crate::diff::tensor_families(&matched)
    } else {
        crate::diff::tensor_families(&s.tensors)
    };
    ok(json!({ "families": families }))
}

/// The checkpoint statistics, plus the S3 section's ready-made phrases for an
/// `s3://` source (see [`dto::S3SummaryDto`]). Flattened, so every existing key keeps
/// its place and `s3_summary` is simply absent for a local checkpoint.
pub(crate) fn stats(s: &WebState) -> Reply {
    #[derive(serde::Serialize)]
    struct StatsResponse<'a> {
        #[serde(flatten)]
        stats: &'a crate::stats::CheckpointStats,
        #[serde(skip_serializing_if = "Option::is_none")]
        s3_summary: Option<dto::S3SummaryDto>,
    }
    ok(&StatsResponse {
        stats: &s.stats,
        s3_summary: dto::S3SummaryDto::from_stats(&s.stats),
    })
}

pub(crate) fn health(s: &WebState) -> Reply {
    ok(&s.health)
}

pub(crate) fn check(s: &WebState) -> Reply {
    s.check
        .as_ref()
        .map_or_else(|| ok(Value::Null), |report| ok(report.to_json(false)))
}

/// The two sides of the comparison a request quotes, or the refusal to answer without one.
///
/// Shared by the three routes that read a comparison by id — the aligned tree, the report and the name
/// picker. All take `?id=N` from `POST /api/compare`, and all have to distinguish *no comparison* from
/// *someone else's*: answering from "whatever is in the slot" once handed one client another's pair.
fn comparison_sides(
    current: &super::Current,
    q: &Query,
    route: &str,
) -> Result<(Arc<WebState>, Arc<WebState>), Reply> {
    let Some(id) = q.get("id").and_then(|v| v.parse::<u64>().ok()) else {
        return Err(err(
            400,
            format!(
                "{route} needs ?id=N, the id returned by POST /api/compare — \
                 without it this could answer about a comparison you did not ask for"
            ),
        ));
    };
    match current.comparison_for(id) {
        ComparisonLookup::None => Err(err(
            409,
            "no comparison set up — POST /api/compare?left=SPEC first",
        )),
        ComparisonLookup::Replaced { current: now } => Err(err(
            409,
            format!(
                "comparison {id} was replaced by another request (now {now}) — \
                 POST /api/compare again to set up yours"
            ),
        )),
        ComparisonLookup::Found { base, right } => Ok((base, right)),
    }
}

/// The names a comparison's two sides **share**, for the exact-name picker: `?id=N`, the alignment
/// parameters, and an optional `?q=` to search with.
///
/// Why a route rather than reading the aligned tree the browser can already ask for: that body is the
/// largest this API serves (91 MB on a real pair), and this is a list of strings behind a search box.
/// It answers with at most `limit` of them (100 by default, 500 at most), so a keystroke costs a few
/// kilobytes.
///
/// The alignment matters and the filter does not — see [`super::diffscope::DiffScope::aligned_names`].
/// `q` is the same fuzzy match the tensor tree's search uses, ranked best-first; without it the list is
/// alphabetical, which is what a reader scrolling for a prefix wants.
pub(crate) fn diff_names(current: &super::Current, q: &Query) -> Reply {
    let scope = match super::diffscope::DiffScope::from_query(q) {
        Ok(scope) => scope,
        Err(e) => return err(400, format!("{e:#}")),
    };
    let limit = match whole(q, "limit", 100, 1) {
        Ok(n) => n.min(500),
        Err(e) => return e,
    };
    let (base, candidate) = match comparison_sides(current, q, "diffnames") {
        Ok(pair) => pair,
        Err(e) => return e,
    };
    let mut old =
        crate::diff::CheckpointSummary::from_loaded(&base.tensors, &base.checkpoint.metadata_vec());
    let mut new = crate::diff::CheckpointSummary::from_loaded(
        &candidate.tensors,
        &candidate.checkpoint.metadata_vec(),
    );
    if let Err(e) = scope.reroot_sides(&mut old, &mut new) {
        return err(400, format!("{e:#}"));
    }
    let names = scope.aligned_names(old, new);
    let query = q.get("q").map_or("", String::as_str).trim();
    let matched: Vec<&String> = if query.is_empty() {
        names.iter().collect()
    } else {
        // The tree screen's matcher, so a query that finds a tensor there finds it here.
        use fuzzy_matcher::FuzzyMatcher as _;
        let matcher = fuzzy_matcher::skim::SkimMatcherV2::default();
        let mut scored: Vec<(i64, &String)> = names
            .iter()
            .filter_map(|n| matcher.fuzzy_match(n, query).map(|score| (score, n)))
            .collect();
        // Best first, then alphabetical, so equal scores do not shuffle between keystrokes.
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        scored.into_iter().map(|(_, n)| n).collect()
    };
    ok(json!({
        // How many line up in all, so the picker can say "100 of 1,254" rather than implying that the
        // hundred it drew is everything.
        "total": names.len(),
        "matched": matched.len(),
        "names": matched.iter().take(limit).collect::<Vec<_>>(),
    }))
}

/// **The terminal invocation for a set of parameters** — the one place a command string is produced.
///
/// `GET /api/command?left=SPEC&right=SPEC&<scope>&<check>`, answering `{"command": "…"}`.
///
/// Every surface that offers "run this in a terminal" asks here with the parameters it holds, so there
/// is one renderer of `diff` arguments (the table in [`super::params`]) and one assembler of the line
/// (`compare::cli_diff_command`, which quotes and carries `#subtree` on the operands). The browser used
/// to build the Data view's command itself out of the two addresses — and so offered
/// `diff --values OLD NEW` for a comparison scoped to a single tensor with a fused alignment: a command
/// that compares every tensor of both checkpoints, unaligned. A string assembled beside the state it is
/// supposed to describe will always be one control behind it.
///
/// `null` when a side cannot be named in one word (see `compare::side_operand`) — better nothing than a
/// command that means something else.
pub(crate) fn command(q: &Query) -> Reply {
    let Some(left) = q.get("left").map(String::as_str).filter(|s| !s.is_empty()) else {
        return err(400, "command needs ?left=SPEC&right=SPEC");
    };
    let right = q.get("right").map_or("", String::as_str);
    if right.is_empty() {
        return err(400, "command needs ?right=SPEC as well as ?left=SPEC");
    }
    // Parsed, though only the render uses the query: a scope that cannot compile is a client mistake
    // worth naming here rather than a command built from an invalid glob.
    if let Err(e) = super::diffscope::DiffScope::from_query(q) {
        return err(400, format!("{e:#}"));
    }
    let args = super::params::render(q, &[super::params::SCOPE, super::params::CHECK]);
    ok(json!({
        "command": crate::compare::cli_diff_command(
            left,
            right,
            &args,
            crate::compare::Sides::BaselineFirst,
            super::params::subtrees(q),
        ),
    }))
}

/// One side's **namespaces**, for the subtree pickers: `?id=N&side=old|new`, with an optional `?q=`.
///
/// A subtree is a wrapper to drop — `language_model`, `model`, `vision_tower` — and typing one is a
/// guess at a name the reader has not seen yet: a prefix that selects nothing is a 400, and one that
/// selects the wrong thing is an empty comparison. So the panel offers what is there, with the number of
/// tensors under each, which is what tells `language_model` (312) from a typo (0).
///
/// **Namespaces, not paths.** Prefixes whose last segment is an index are left out: 62 entries reading
/// `model.layers.0`, `model.layers.1`, … are not alignment targets, they are the thing an alignment
/// looks *through*. Depth is capped for the same reason — a wrapper is one or two segments in practice,
/// four is already generous — and each is offered with its count, biggest first.
///
/// Before any alignment, deliberately: re-rooting is what the answer feeds, so applying it here would
/// offer prefixes of prefixes.
pub(crate) fn subtrees(current: &super::Current, q: &Query) -> Reply {
    /// A wrapper deeper than this is not what this control is for.
    const MAX_DEPTH: usize = 4;
    let side = match q.get("side").map(String::as_str) {
        Some("old") => Side::Old,
        Some("new") => Side::New,
        _ => {
            return err(
                400,
                "subtrees needs ?side=old or ?side=new — the two checkpoints have different namespaces",
            );
        }
    };
    let limit = match whole(q, "limit", 100, 1) {
        Ok(n) => n.min(500),
        Err(e) => return e,
    };
    let (base, candidate) = match comparison_sides(current, q, "subtrees") {
        Ok(pair) => pair,
        Err(e) => return e,
    };
    let state = match side {
        Side::Old => &base,
        Side::New => &candidate,
    };
    // Count the tensors under every namespace prefix, in one pass over the names.
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for tensor in &state.tensors {
        let name = tensor.name.as_str();
        let mut at = 0;
        for depth in 0..MAX_DEPTH {
            match name[at..].find('.') {
                None => break,
                Some(dot) => {
                    at += dot;
                    let prefix = &name[..at];
                    at += 1;
                    let _ = depth;
                    // A namespace has no *index* in it. `model.layers.0` is a layer and
                    // `model.layers.0.mlp` is one layer's block — the things an alignment looks
                    // *through*, not targets for it, and there are as many of them as there are layers.
                    if prefix
                        .split('.')
                        .any(|seg| !seg.is_empty() && seg.chars().all(|c| c.is_ascii_digit()))
                    {
                        continue;
                    }
                    *counts.entry(prefix).or_default() += 1;
                }
            }
        }
    }
    let query = q.get("q").map_or("", String::as_str).trim().to_lowercase();
    let mut found: Vec<(&str, usize)> = counts
        .into_iter()
        .filter(|(prefix, _)| query.is_empty() || prefix.to_lowercase().contains(&query))
        .collect();
    // The biggest namespace first — that is the one a wrapper usually is — then alphabetically, so the
    // list does not shuffle between keystrokes.
    found.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    ok(json!({
        "total": found.len(),
        "subtrees": found
            .iter()
            .take(limit)
            .map(|(prefix, count)| json!({ "prefix": prefix, "tensors": count }))
            .collect::<Vec<_>>(),
    }))
}

/// Which side of a comparison a request is about.
#[derive(Clone, Copy)]
enum Side {
    Old,
    New,
}

/// The structural report for a comparison the server holds: `?id=N`, from `POST /api/compare`.
///
/// Both checkpoints were read when the pair was set up, so this route reads nothing — it categorises
/// what is already in the slot, under whatever selection the query carries. Value comparison
/// (`diff --values`) is deliberately not here: a scan that takes minutes needs progress and
/// cancellation, which the jobs API and the CLI provide.
///
/// The result is **not** cached: it is keyed by the scope, the direction and the fold, which come
/// from the request, so a cache would grow without bound under a client that varies them. Deriving
/// it from two checkpoints already in memory is cheap.
pub(crate) fn diff(current: &super::Current, q: &Query) -> Reply {
    // **The same pair the other views read.**
    //
    // This used to resolve and read its own baseline (`?against=SPEC`) and compare it against
    // whatever checkpoint the server had open. Two things were wrong with that once the report became
    // one view of a comparison rather than a screen of its own: naming a candidate that is not the
    // open checkpoint gave a report about a *different pair* than the tree beside it, and switching
    // between the two views re-read both checkpoints — seconds each, over an ssh proxy.
    //
    // The comparison slot already holds both sides, read once. Everything below works from it, and
    // this route reads nothing.
    let scope = match super::diffscope::DiffScope::from_query(q) {
        Ok(scope) => scope,
        // A bad glob or rename rule is a client mistake worth naming: the alternative is an empty diff
        // that looks like "nothing matched".
        Err(e) => return err(400, format!("{e:#}")),
    };
    let swapped = match switch(q, "swap") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let (base, candidate) = match comparison_sides(current, q, "diff") {
        Ok(pair) => pair,
        Err(e) => return e,
    };
    let against = base.spec.as_str();
    // What each side is called on a command line, for the invocation offered at the bottom of the
    // report. Empty when a side cannot be named in one word, which `cli_diff_command` turns into no
    // command rather than a wrong one.
    let baseline_operand = crate::compare::side_operand(against, &base.files).unwrap_or_default();
    let candidate_operand =
        crate::compare::side_operand(&candidate.spec, &candidate.files).unwrap_or_default();
    let mut baseline = crate::compare::Baseline {
        summary: crate::diff::CheckpointSummary::from_loaded(
            &base.tensors,
            &base.checkpoint.metadata_vec(),
        ),
        s3: base.checkpoint.s3.clone(),
    };
    let mut served = crate::diff::CheckpointSummary::from_loaded(
        &candidate.tensors,
        &candidate.checkpoint.metadata_vec(),
    );
    // `#subtree`, per side and before anything else: it re-keys a side's tensors to their sub-path, so
    // two checkpoints under different namespaces line up. A prefix that selects nothing is a typo, and
    // the message says which side it was on rather than showing an empty report.
    if let Err(e) = scope.reroot_sides(&mut baseline.summary, &mut served) {
        return err(400, format!("{e:#}"));
    }
    // `?swap=1` turns the comparison round: the open checkpoint becomes the baseline. A diff is
    // directional — what was added one way is removed the other — and the side-by-side has always had
    // a swap while the report had none, so seeing the same pair the other way meant editing a URL.
    //
    // Everything downstream follows from this one pair, including the rename rules (which always
    // rewrite the *old* side's names, whichever checkpoint that now is) and the CLI command offered
    // at the bottom.
    let sides = if swapped {
        crate::compare::Sides::OpenFirst
    } else {
        crate::compare::Sides::BaselineFirst
    };
    let (old, new, old_s3, new_s3) = match sides {
        crate::compare::Sides::BaselineFirst => (
            baseline.summary,
            served,
            baseline.s3.as_ref(),
            candidate.checkpoint.s3.as_ref(),
        ),
        crate::compare::Sides::OpenFirst => (
            served,
            baseline.summary,
            candidate.checkpoint.s3.as_ref(),
            baseline.s3.as_ref(),
        ),
    };
    // Rename, then filter, then compare — `crate::web::diffscope`, shared with the CLI's own order.
    let mut scoped = scope.compare(old, new);
    // For an s3-vs-s3 pair, the objects themselves: ETag, size, checksums, tags. The same call the
    // `diff` subcommand makes, so the browser stops being the one surface that compares two `s3://`
    // checkpoints without mentioning their object-level differences.
    let s3_note = crate::compare::attach_s3(&mut scoped.report, old_s3, new_s3);
    // The section as the terminal words it — one implementation of what a multipart `ETag` proves.
    //
    // Grouped, which is the CLI's own default here: a re-quantization changes an object per expert per
    // layer, and 361 lines reading `model.layers.N.…codebook (etag)` say less than six lines reading
    // `model.layers.{1-60}.…codebook (×60)`. (The tensor sections are ungrouped because the client
    // filters and caps them; these are a fixed informational block.)
    let s3_lines = scoped
        .report
        .s3
        .as_ref()
        .map(|s3| s3.summary_lines(true, scope.is_filtered()));
    ok(json!({
        "against": against,
        // The other side, as the server resolved it — *not* whatever this server has open. Naming the
        // served checkpoint there was how a report of `hf ↔ candidate` came to head itself
        // `new /tmp/mapfix/new.safetensors`: the label was reading a different variable than the
        // comparison was.
        "candidate": candidate.spec,
        // Which way round this report reads, so the view labels its two sides from the server's answer
        // rather than from its own copy of the parameter.
        "swapped": sides == crate::compare::Sides::OpenFirst,
        "verdict": crate::compare::verdict(&scoped.report),
        // Why the metadata section is empty, when it is — the CLI's `not compared (filtered subset)`.
        // `null` means it really was compared and really has nothing in it.
        "metadata_note": scope.metadata_note(),
        // The same sections with index-templated families collapsed onto one row each — what the
        // terminal prints by default (`--full` turns it off). Both lists are sent: grouping is the
        // cheap part, and which one to show is the reader's choice, made without a round trip.
        "grouped": scoped.report.grouped(),
        // What an unfused/fused alignment folded: `name → [old parts, new parts]`, so a row can read
        // `×256 → ×1` the way the terminal's does. Empty unless `align_fused=1` changed something.
        "folded": scoped.report.folded.iter()
            .map(|(n, (o, w))| (n.clone(), json!([o, w])))
            .collect::<serde_json::Map<_, _>>(),
        "aligns_fused": scope.aligns_fused(),
        // What to call the two totals lines. Under a filter the report's `old_bytes` / `new_bytes` cover
        // the **matched tensors** (`TensorFilter::apply` narrows the footprints with the signatures), so
        // a bare `size:` would read as the checkpoint's size. The server words it — `diff::totals_labels`,
        // shared with the terminal — rather than the browser re-deriving the rule from the parameters.
        "totals_labels": {
            "size": crate::diff::totals_labels(scope.is_filtered()).0,
            "params": crate::diff::totals_labels(scope.is_filtered()).1,
        },
        // **Whether the numbers can be compared at all**, and why not when they cannot.
        //
        // Asked of the two addresses, so the Data view can say it *before* a reader spends minutes on
        // a job that ends in a refusal — which is exactly how this was reported. One answer from the
        // one function both surfaces use (`compare::values_supported`).
        "values_note": crate::compare::values_where(
            &baseline_operand,
            &candidate_operand,
            current.proxy_host(),
        )
            .err()
            .map(|e| format!("{e:#}")),
        // What the S3 object comparison did, or why it did not happen. `null` when neither side is
        // `s3://`, which is the ordinary local case.
        "s3_note": s3_note,
        "s3_lines": s3_lines,
        // `modified: OLD → NEW`, humanised by the same rule the terminal uses. `null` unless both
        // sides carry timestamps.
        "modified_line": scoped.report.modified_line(false),
        // The equivalent CLI invocation, so a browser finding can be reproduced (and
        // extended with --values) in a terminal. `null` when the served files span
        // directories, since then no single path names this side of the comparison.
        // Carries the scope, so the command compares what is on screen rather than everything.
        // Each side named the way a command line can name it — see `compare::side_operand`.
        "command": crate::compare::cli_diff_command(
            &baseline_operand,
            &candidate_operand,
            // `--full` when the reader has expanded the families, so the command reproduces the screen
            // rather than the default the screen is not showing.
            // The scope *and* the check, from the one table — `--full` included, which is why this no
            // longer appends it by hand.
            &super::params::render(q, &[super::params::SCOPE, super::params::CHECK]),
            sides,
            scope.subtrees(),
        ),
        // What the scope selected, for the CLI's `filter [...] matched 19 of 117664` line. `null` when
        // nothing narrowed the comparison.
        "matched": scoped.matched.as_ref().map(|m| json!({
            "selected": m.selected,
            "total": m.total,
            "names": m.names,
        })),
        // Two old names that map onto one lose a tensor from the comparison. The CLI warns; so does this.
        "rename_collisions": scoped.rename_collisions,
        "report": scoped.report,
    }))
}

pub(crate) fn model(s: &WebState) -> Reply {
    ok(&s.checkpoint)
}

pub(crate) fn tensor(s: &WebState, q: &Query) -> Reply {
    match lookup(s, q) {
        Ok(t) => ok(t),
        Err(e) => e,
    }
}

/// Read a text/JSON file's content (capped) for the file browser's preview. Only
/// serves paths that are in the checkpoint's own file list — no path traversal.
pub(crate) fn file(s: &WebState, q: &Query) -> Reply {
    const CAP: usize = crate::filetree::PREVIEW_CAP as usize;
    let Some(rel) = q.get("path") else {
        return err(400, "missing ?path=");
    };
    let Some(entry) = s
        .checkpoint
        .files
        .iter()
        .find(|f| f.rel_path == *rel && !f.is_dir())
    else {
        return err(404, format!("no such file: {rel}"));
    };
    let abs = std::path::Path::new(&s.root).join(&entry.rel_path);
    // Resolve before reading. The entry came from this checkpoint's own directory walk, so
    // the *name* is in bounds — but a symlink inside the checkpoint can point anywhere, and
    // reading through it would serve a file outside the tree the server was pointed at. The
    // walk records symlinks as leaves, so they are reachable from the browser.
    let root = std::path::Path::new(&s.root).canonicalize();
    let real = abs.canonicalize();
    match (&root, &real) {
        (Ok(root), Ok(real)) if !real.starts_with(root) => {
            return err(
                403,
                format!("{rel} resolves outside the checkpoint ({})", real.display()),
            );
        }
        _ => {}
    }
    match std::fs::read(&abs) {
        Ok(bytes) => {
            let truncated = bytes.len() > CAP;
            let head = bytes.get(..bytes.len().min(CAP)).unwrap_or(&bytes);
            // Preview as text, but say when it ISN'T: a lossy conversion of binary is a wall
            // of U+FFFD, and the client can't tell that from a file that really contains
            // replacement characters. `text` stays lossy so a mostly-text file with a stray
            // byte still previews.
            let binary = std::str::from_utf8(head).is_err();
            let text = String::from_utf8_lossy(head).into_owned();
            ok(json!({
                "path": rel,
                "name": entry.name,
                "size": entry.apparent(),
                "truncated": truncated,
                "cap": CAP,
                "binary": binary,
                "text": text,
            }))
        }
        Err(e) => err(500, format!("read failed: {e}")),
    }
}

pub(crate) fn layout(s: &WebState, q: &Query) -> Reply {
    let Some(file) = q.get("file") else {
        return err(400, "missing ?file=");
    };
    s.layouts
        .iter()
        .find(|l| l.name == *file || basename(&l.name) == file.as_str())
        .map_or_else(|| err(404, format!("no layout for file: {file}")), ok)
}

// ---- on-demand tensor-data routes (read bytes; local only) ----

pub(crate) fn tensor_stats(s: &WebState, q: &Query) -> Reply {
    let (t, view) = match data_request(s, q) {
        Ok(v) => v,
        Err(e) => return e,
    };
    match scan_stats(s, t, view) {
        Ok(dto) => ok(dto),
        Err(e) => e,
    }
}

pub(crate) fn tensor_sample(s: &WebState, q: &Query) -> Reply {
    let (t, view) = match data_request(s, q) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // A window of no rows is not a window, so both are at least one.
    let (rows, cols, slice) = match (
        whole(q, "rows", 32, 1),
        whole(q, "cols", 32, 1),
        whole(q, "slice", 0, 0),
    ) {
        (Ok(rows), Ok(cols), Ok(slice)) => (rows, cols, slice),
        (Err(e), ..) | (_, Err(e), _) | (.., Err(e)) => return e,
    };
    let mode = match one_of(q, "mode", &["grid", "window", "edges", "max"], "grid") {
        Ok("window") => match (whole(q, "row_off", 0, 0), whole(q, "col_off", 0, 0)) {
            (Ok(row_off), Ok(col_off)) => SampleMode::Window { row_off, col_off },
            (Err(e), _) | (_, Err(e)) => return e,
        },
        Ok("edges") => match (fraction(q, "row_tail", 0.5), fraction(q, "col_tail", 0.5)) {
            (Ok(row_tail), Ok(col_tail)) => SampleMode::Edges { row_tail, col_tail },
            (Err(e), _) | (_, Err(e)) => return e,
        },
        Ok("max") => SampleMode::GridMax,
        Ok(_) => SampleMode::Grid,
        Err(e) => return e,
    };
    let schema = s.schemas.get(name_of(q));
    let include_raw = match switch(q, "raw") {
        Ok(v) => v,
        Err(e) => return e,
    };
    match sample::sample_tensor(t, rows, cols, slice, view, mode, schema) {
        Ok(sample) => ok(SampleDto::from_sample(&sample, &t.dtype, include_raw)),
        Err(e) => err(500, e),
    }
}

pub(crate) fn tensor_histogram(s: &WebState, q: &Query) -> Reply {
    let (t, view) = match data_request(s, q) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // Absent means "choose for me"; present and unreadable is a mistake, not a request to choose.
    let bins = match q.get("bins") {
        None => None,
        Some(_) => match whole(q, "bins", 0, 1) {
            Ok(n) => Some(n),
            Err(e) => return e,
        },
    };

    // Float / wide-int bins need the value range; reuse the cached stats or scan.
    let range = match scan_stats(s, t, view) {
        Ok(dto) => Some((dto.min, dto.max)),
        Err(e) => return e,
    };
    let Some((hist_bins, n)) = sample::histogram_bins(view, &t.dtype, range, bins) else {
        return err(400, format!("no histogram for dtype {}", t.dtype));
    };
    let shared = sample::HistShared::new(n);
    let (cancel, pause) = (AtomicBool::new(false), AtomicBool::new(false));
    let schema = s.schemas.get(name_of(q));
    if let Err(e) = sample::tensor_histogram_into(
        t, view, schema, hist_bins, n, &shared, &cancel, &pause, None,
    ) {
        return err(500, e);
    }
    ok(HistogramDto::from(&shared.snapshot(hist_bins)))
}

// ---- helpers ----

/// Compute (or fetch the cached) whole-tensor stats for `(name, view)`.
fn scan_stats(s: &WebState, t: &TensorInfo, view: ViewDtype) -> Result<StatsDto, Reply> {
    let key = (t.name.clone(), dto::view_label(view));
    // `unwrap_or_else(into_inner)`: this is a pure memo, so a mutex poisoned by an
    // unrelated panic carries no broken invariant — but `.unwrap()` would turn that into
    // a permanent 500 for this endpoint for the rest of the process's life.
    let hit = s
        .stats_cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
        .cloned();
    if let Some(hit) = hit {
        return Ok(hit);
    }
    let (cancel, pause) = (AtomicBool::new(false), AtomicBool::new(false));
    let schema = s.schemas.get(&t.name);
    let stats =
        sample::tensor_stats(t, view, schema, &cancel, &pause, None).map_err(|e| err(500, e))?;
    let dto = StatsDto::from(&stats);
    s.stats_cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(key, dto.clone());
    Ok(dto)
}

/// What every on-demand data route needs before it can read bytes: the tensor, a guarantee
/// it is local, and the dtype view to read it through. `Err` is the error envelope to
/// return as-is.
///
/// `tensor_stats`, `tensor_sample` and `tensor_histogram` each opened with these same three
/// lookups, and their early returns are exactly what the client sees on a bad request — so
/// changing any of them meant changing it in three places.
fn data_request<'a>(s: &'a WebState, q: &Query) -> Result<(&'a TensorInfo, ViewDtype), Reply> {
    let t = lookup(s, q)?;
    if let Some(e) = require_bytes(s) {
        return Err(e);
    }
    Ok((t, view_of(q)?))
}

fn lookup<'a>(s: &'a WebState, q: &Query) -> Result<&'a TensorInfo, Reply> {
    let name = q.get("name").ok_or_else(|| err(400, "missing ?name="))?;
    let idx = s
        .tensor_index
        .get(name)
        .ok_or_else(|| err(404, format!("unknown tensor: {name}")))?;
    // The index came from `tensor_index`, which is built from `tensors` itself.
    s.tensors
        .get(*idx)
        .ok_or_else(|| err(500, format!("tensor index out of range for {name}")))
}

fn view_of(q: &Query) -> Result<ViewDtype, Reply> {
    q.get("dtype").map_or(Ok(ViewDtype::Stored), |d| {
        sample::parse_view_dtype(d).map_err(|e| err(400, e))
    })
}

fn name_of(q: &Query) -> &str {
    q.get("name").map_or("", String::as_str)
}

// ---- query values ----
//
// **A malformed value is refused, not defaulted.** The router already refuses an unknown parameter
// *name* — `?nmae=…` is a typo, not a filter — for the reason `clap` refuses an unknown flag. The
// values had the opposite rule: `?rows=lots` sampled 32 rows, `?mode=windwo` returned a grid,
// `?bins=many` chose the bin count itself, and each answered `200`. A confident wrong answer to a
// question nobody asked is the failure that is hardest to notice, and it was reachable by one typo.
//
// Each of these returns the default when the parameter is absent, and a `400` naming the parameter,
// what arrived and what is allowed when it is present and wrong.

/// A whole-number parameter, at least `least`.
fn whole(q: &Query, key: &str, default: usize, least: usize) -> Result<usize, Reply> {
    let Some(raw) = q.get(key) else {
        return Ok(default);
    };
    match raw.parse::<usize>() {
        Ok(n) if n >= least => Ok(n),
        Ok(n) => Err(err(
            400,
            format!("{key}={n} is too small — it must be at least {least}"),
        )),
        Err(_) => Err(err(
            400,
            format!("{key}={raw:?} is not a whole number (expected a count like {default})"),
        )),
    }
}

/// A fraction parameter, within `0.0..=1.0`.
fn fraction(q: &Query, key: &str, default: f32) -> Result<f32, Reply> {
    let Some(raw) = q.get(key) else {
        return Ok(default);
    };
    match raw.parse::<f32>() {
        Ok(f) if (0.0..=1.0).contains(&f) => Ok(f),
        Ok(f) => Err(err(
            400,
            format!("{key}={f} is out of range — it is a fraction of the axis, from 0 to 1"),
        )),
        Err(_) => Err(err(
            400,
            format!("{key}={raw:?} is not a number (expected a fraction like {default})"),
        )),
    }
}

/// One of a fixed set of words.
fn one_of<'q>(
    q: &'q Query,
    key: &str,
    allowed: &[&str],
    default: &'q str,
) -> Result<&'q str, Reply> {
    let Some(raw) = q.get(key).map(String::as_str) else {
        return Ok(default);
    };
    if allowed.contains(&raw) {
        return Ok(raw);
    }
    Err(err(
        400,
        format!("{key}={raw:?} is not one of {}", allowed.join(", ")),
    ))
}

/// A switch: `1`/`true` on, `0`/`false` off. Anything else is a mistake, not "off" — `?full=yes`
/// silently meaning `full=0` is the same silent wrong answer as a mistyped number.
fn switch(q: &Query, key: &str) -> Result<bool, Reply> {
    match q.get(key).map(String::as_str) {
        None => Ok(false),
        Some("1" | "true") => Ok(true),
        Some("0" | "false") => Ok(false),
        Some(other) => Err(err(
            400,
            format!("{key}={other:?} is not a switch — use {key}=1 or {key}=0"),
        )),
    }
}

fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Fixture + JSON helpers shared by the handler and contract test modules.
#[cfg(test)]
mod tests_support {
    use super::*;
    use std::path::PathBuf;

    pub(super) const TENSOR: &str = "model.layers.0.mlp.down_proj.weight";

    /// Build the shared state from a checked-in fixture, exactly as `run_web` does.
    pub(super) fn state() -> WebState {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.safetensors");
        let files = vec![fixture];
        let model = crate::readers::read_local(&files).expect("fixture reads");
        WebState::build(model, &files, &[])
    }

    /// Parse a reply body back into JSON so tests assert on the values a client sees.
    pub(super) fn json(reply: &Reply) -> Value {
        serde_json::from_slice(&reply.1).unwrap_or_else(|e| {
            panic!(
                "reply body is not JSON ({e}): {}",
                String::from_utf8_lossy(&reply.1)
            )
        })
    }

    pub(super) fn query(pairs: &[(&str, &str)]) -> Query {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;

    /// The file preview reads `root.join(rel_path)`. The name comes from this checkpoint's own
    /// walk, so it can't contain `..` — but a SYMLINK inside the checkpoint can point anywhere,
    /// and the walk records symlinks as browsable leaves. Following one served a file from
    /// outside the tree the server was pointed at.
    #[cfg(unix)]
    #[test]
    fn the_file_preview_refuses_a_symlink_that_escapes_the_checkpoint() {
        use std::path::PathBuf;
        let dir = std::env::temp_dir().join("cs_web_symlink_escape");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("ckpt")).unwrap();
        // A secret next to the checkpoint, and a link to it from inside.
        std::fs::write(dir.join("outside.txt"), b"not yours").unwrap();
        std::fs::write(dir.join("ckpt/config.json"), b"{}").unwrap();
        std::os::unix::fs::symlink(dir.join("outside.txt"), dir.join("ckpt/leak.txt")).unwrap();
        // A real safetensors file so the read produces a checkpoint at all.
        let shard = dir.join("ckpt/model.safetensors");
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 4]);
        std::fs::write(&shard, bytes).unwrap();

        let files = vec![shard];
        let model = crate::readers::read_local(&files).expect("fixture reads");
        let s = WebState::build(model, &files, &[]);

        // A file genuinely inside the checkpoint previews.
        let q: Query = std::iter::once(("path".to_string(), "config.json".to_string())).collect();
        assert_eq!(file(&s, &q).0, 200, "an in-tree file still previews");

        // The symlink is listed (the walk records it) but reading it is refused.
        let q: Query = std::iter::once(("path".to_string(), "leak.txt".to_string())).collect();
        let reply = file(&s, &q);
        assert_eq!(reply.0, 403, "a symlink out of the tree must not be served");
        let body = String::from_utf8_lossy(&reply.1);
        assert!(body.contains("outside the checkpoint"), "{body}");
        assert!(!body.contains("not yours"), "the content leaked: {body}");
        let _ = std::fs::remove_dir_all(&dir);
        let _: PathBuf = dir; // keep the binding used on all cfgs
    }

    #[test]
    fn every_endpoint_answers_200_with_the_documented_shape() {
        let s = state();

        let tree = json(&tree(&s));
        assert!(tree["root"].is_string(), "tree exposes the checkpoint root");
        assert!(tree["tree"].is_array(), "tree exposes the node array");

        assert!(json(&files(&s))["kind"].is_string(), "files is an FsNode");
        let st = json(&stats(&s));
        assert_eq!(st["files"]["count"], 1, "one fixture shard");
        assert!(
            st["tensors"].is_object() || st["dtypes"].is_array(),
            "stats reports tensor facets: {st}"
        );
        assert!(json(&health(&s)).is_array(), "health is a list of reports");
        assert!(json(&check(&s))["summary"].is_object());
        assert!(json(&model(&s))["root"].is_string());

        // The whole point of `/api/tensor`: one tensor's metadata by exact name.
        let t = json(&tensor(&s, &query(&[("name", TENSOR)])));
        assert_eq!(t["name"], TENSOR);
        assert_eq!(t["dtype"], "U16");
        assert_eq!(t["shape"], serde_json::json!([3, 4, 5]));

        // Filtering is server-side so the web and TUI agree; check it actually filters.
        let f = json(&filter(&s, &query(&[("q", "dtype:F32")])));
        assert_eq!(f["active"], true);
        let names: Vec<&str> = f["names"]
            .as_array()
            .expect("names")
            .iter()
            .map(|n| n.as_str().unwrap_or(""))
            .collect();
        assert_eq!(
            names,
            ["model.layers.0.input_layernorm.weight", "model.norm.weight"]
        );

        // An empty query is "inactive" (show everything), not "match nothing".
        assert_eq!(json(&filter(&s, &query(&[("q", "")])))["active"], false);

        assert!(json(&schema(&s, &query(&[("q", "")])))["families"].is_array());
    }

    #[test]
    fn data_view_endpoints_return_the_requested_window() {
        let s = state();
        let sample = json(&tensor_sample(
            &s,
            &query(&[
                ("name", TENSOR),
                ("mode", "window"),
                ("rows", "2"),
                ("cols", "3"),
            ]),
        ));
        assert_eq!(sample["values"].as_array().map(Vec::len), Some(2));
        assert_eq!(sample["values"][0].as_array().map(Vec::len), Some(3));
        // U16 is an integer view, so the client is told to format from the raw bits.
        assert_eq!(sample["integer"], true);
        assert_eq!(sample["signed"], false);
        assert!(
            sample["raw"].is_array(),
            "integer views always ship raw bits"
        );

        let st = json(&tensor_stats(&s, &query(&[("name", TENSOR)])));
        assert_eq!(st["count"], 60); // 3*4*5
        assert!(st["min"].is_number() && st["max"].is_number());

        let h = json(&tensor_histogram(
            &s,
            &query(&[("name", TENSOR), ("bins", "8")]),
        ));
        assert!(h["counts"].as_array().is_some_and(|c| !c.is_empty()));

        let l = json(&layout(&s, &query(&[("file", "tiny.safetensors")])));
        assert!(l["segments"].is_array(), "byte-layout segments");
    }

    #[test]
    fn bad_input_is_a_4xx_with_a_message_never_a_panic() {
        let s = state();
        for (label, reply) in [
            ("unknown tensor", tensor(&s, &query(&[("name", "nope")]))),
            ("missing name", tensor(&s, &query(&[]))),
            (
                "unknown layout file",
                layout(&s, &query(&[("file", "nope.safetensors")])),
            ),
            ("missing file param", layout(&s, &query(&[]))),
            ("unknown file", file(&s, &query(&[("path", "nope.txt")]))),
            (
                "sample of unknown tensor",
                tensor_sample(&s, &query(&[("name", "nope")])),
            ),
            (
                "stats of unknown tensor",
                tensor_stats(&s, &query(&[("name", "nope")])),
            ),
            ("bad filter facet", filter(&s, &query(&[("q", "bogus:1")]))),
            (
                "bad filter number",
                filter(&s, &query(&[("q", "size:abc")])),
            ),
        ] {
            assert!(
                (400..500).contains(&reply.0),
                "{label}: expected a 4xx, got {}",
                reply.0
            );
            let msg = json(&reply)["error"].as_str().unwrap_or("").to_string();
            assert!(!msg.is_empty(), "{label}: a 4xx must explain itself");
        }
    }

    /// A dtype override must reinterpret the SAME bytes, not re-read the tensor: the
    /// packed 4-bit view yields more values than the stored U16 one.
    #[test]
    fn dtype_override_reinterprets_the_same_bytes() {
        let s = state();
        let stored = json(&tensor_sample(
            &s,
            &query(&[
                ("name", TENSOR),
                ("mode", "window"),
                ("rows", "1"),
                ("cols", "40"),
            ]),
        ));
        let as_u4 = json(&tensor_sample(
            &s,
            &query(&[
                ("name", TENSOR),
                ("dtype", "u4"),
                ("mode", "window"),
                ("rows", "1"),
                ("cols", "40"),
            ]),
        ));
        assert_eq!(stored["view"], "stored");
        assert_eq!(as_u4["view"], "u4");
        assert!(
            as_u4["total_cols"].as_u64() > stored["total_cols"].as_u64(),
            "unpacking 4-bit nibbles must widen the logical row"
        );
    }
}

/// Contract tests: the JSON keys `web/src/lib/types.ts` declares must actually exist in
/// what the server sends.
///
/// This is the gap that nothing else covers. `svelte-check` validates the UI against
/// `types.ts`, and Rust validates the DTO structs — but the two are hand-mirrored, with
/// no schema or codegen in between. Rename a Rust field and every gate stays green while
/// the UI silently renders `undefined`. Listing the keys the client actually reads makes
/// that a build failure instead.
#[cfg(test)]
mod contract {
    use super::tests_support::*;
    use serde_json::Value;

    /// Assert `value` is an object carrying every one of `keys`.
    fn has_keys(what: &str, value: &Value, keys: &[&str]) {
        let obj = value
            .as_object()
            .unwrap_or_else(|| panic!("{what}: expected a JSON object, got {value}"));
        let missing: Vec<&str> = keys
            .iter()
            .copied()
            .filter(|k| !obj.contains_key(*k))
            .collect();
        assert!(
            missing.is_empty(),
            "{what}: web/src/lib/types.ts expects {missing:?}, which the server no longer sends. \
             Present: {:?}",
            obj.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn tree_response_and_nodes_match_types_ts() {
        let s = state();
        let tree = json(&super::tree(&s));
        has_keys(
            "TreeResponse",
            &tree,
            &[
                "root",
                "tensor_count",
                "tree",
                "unindexed",
                "capabilities",
                "format",
                "location",
                "data_view_note",
            ],
        );
        // The capability set the client gates its data views on — every row, since a
        // missing one reads as `undefined` and so as "not allowed".
        has_keys(
            "Capabilities",
            &tree["capabilities"],
            &[
                "read_bytes",
                "modify_in_place",
                "repack",
                "layout_map",
                "browse_files",
                "object_metadata",
                "codec_info",
                "reach",
            ],
        );
        // A list, even when empty — the client turns it straight into a Set.
        assert!(tree["unindexed"].is_array(), "unindexed is a list");

        // Walk to one group and one tensor node — both variants are consumed by the UI.
        let nodes = tree["tree"].as_array().expect("tree array");
        let mut group = None;
        let mut tensor = None;
        let mut stack: Vec<&Value> = nodes.iter().collect();
        while let Some(n) = stack.pop() {
            match n["kind"].as_str() {
                Some("group") => {
                    if group.is_none() {
                        group = Some(n);
                    }
                    if let Some(kids) = n["children"].as_array() {
                        stack.extend(kids.iter());
                    }
                }
                Some("tensor") if tensor.is_none() => tensor = Some(n),
                _ => {}
            }
        }
        has_keys(
            "TreeNode::group",
            group.expect("a group node"),
            &[
                "kind",
                "name",
                "children",
                "expanded",
                "tensor_count",
                "params",
                "total_size",
                "stored_size",
            ],
        );
        let tensor = tensor.expect("a tensor node");
        has_keys("TreeNode::tensor", tensor, &["kind", "info"]);
        has_keys(
            "TensorInfo",
            &tensor["info"],
            &[
                "name",
                "dtype",
                "shape",
                "size_bytes",
                "num_elements",
                "storage",
                "source_path",
                "layout",
            ],
        );
    }

    #[test]
    fn file_node_matches_types_ts() {
        let s = state();
        let root = json(&super::files(&s));
        has_keys(
            "FileNode::dir",
            &root,
            &[
                "kind",
                "name",
                "path",
                "size",
                "files",
                "hardlinked",
                "children",
            ],
        );
        let file = root["children"]
            .as_array()
            .and_then(|c| c.iter().find(|n| n["kind"] == "file"))
            .expect("a file child");
        has_keys(
            "FileNode::file",
            file,
            &[
                "kind",
                "name",
                "path",
                "size",
                "file_kind",
                "shard",
                "size_share",
                "index",
                "links",
                "read_error",
            ],
        );
        // The fixture's own shard is attributed, so `ShardTensors` is on the wire too
        // (a sidecar's `shard` is null, which is why the key alone isn't enough).
        let shard = root["children"]
            .as_array()
            .and_then(|c| c.iter().find(|n| !n["shard"].is_null()))
            .map(|n| n["shard"].clone())
            .expect("a shard child, attributed");
        has_keys(
            "ShardTensors",
            &shard,
            &["tensors", "params", "params_share"],
        );
    }

    #[test]
    fn sample_and_stats_dtos_match_types_ts() {
        let s = state();
        let sample = json(&super::tensor_sample(
            &s,
            &query(&[
                ("name", TENSOR),
                ("mode", "window"),
                ("rows", "2"),
                ("cols", "2"),
            ]),
        ));
        has_keys(
            "SampleDto",
            &sample,
            &[
                "rows",
                "cols",
                "values",
                "min",
                "max",
                "total_rows",
                "total_cols",
                "slices",
                "slice",
                "display_shape",
                "view",
                "mode",
                "overridable",
                "integer",
                "signed",
            ],
        );
        has_keys(
            "StatsDto",
            &json(&super::tensor_stats(&s, &query(&[("name", TENSOR)]))),
            &[
                "count",
                "min",
                "max",
                "mean",
                "std",
                "zeros",
                "nonfinite",
                "zero_fraction",
                "elapsed_ms",
            ],
        );
    }

    #[test]
    fn stats_view_s3_section_keys_match_the_component() {
        // StatsView renders the S3 section from `s3_summary` (server-worded phrases) and
        // the per-object rows from `footprint.S3.objects`. The fixture is local, so the
        // summary is absent there — pin the serialised shape directly.
        let s = state();
        assert!(
            json(&super::stats(&s)).get("s3_summary").is_none(),
            "a local checkpoint has no S3 section"
        );

        let s3 = crate::stats::S3Stats {
            objects: vec![crate::stats::S3ObjectStat {
                key: "a.weight".into(),
                size: 2048,
                etag: "abc".into(),
                checksum: None,
                last_modified: "2026-06-26T10:00:00+00:00".into(),
                tags: Some(0),
                user_meta: 1,
            }],
            warnings: Vec::new(),
        };
        let stats = crate::stats::CheckpointStats::compute(&[], None, None).with_s3(Some(s3));
        let summary = serde_json::to_value(
            crate::web::dto::S3SummaryDto::from_stats(&stats)
                .expect("an s3 footprint yields a summary"),
        )
        .expect("the summary serialises");
        has_keys(
            "stats.s3_summary",
            &summary,
            &[
                "count",
                "total_bytes",
                "checksums",
                "etags",
                "tags",
                "modified",
                "user_meta_objects",
                "object_detail",
                "warnings",
            ],
        );
        // The per-object rows come from the footprint, which must keep its `S3` tag and
        // its objects' `key`/`size`.
        let footprint = serde_json::to_value(&stats).expect("stats serialise");
        let first = &footprint["footprint"]["S3"]["objects"][0];
        has_keys("stats.footprint.S3.objects[]", first, &["key", "size"]);
    }

    #[test]
    fn health_view_keys_match_the_component() {
        let s = state();
        // HealthView.svelte reads `format` + per-check `note`, both added late; a rename
        // would silently blank the explanations and the format-specific sections.
        let check = json(&super::check(&s));
        has_keys(
            "CheckReport",
            &check,
            &["format", "summary", "checks", "healthy"],
        );
        has_keys(
            "CheckReport.summary",
            &check["summary"],
            &["files", "tensors", "params", "errors", "warnings"],
        );
        let first = &check["checks"].as_array().expect("checks")[0];
        has_keys(
            "CheckReport.checks[]",
            first,
            &["id", "title", "note", "status", "findings"],
        );

        // The per-shard reconciliation lists HealthView renders. The fixture has no
        // index.json, so pin the serialised shape directly — a renamed field would
        // otherwise blank a whole section in the browser with nothing failing here.
        let report = serde_json::to_value(crate::health::HealthReport {
            kind: crate::health::HealthKind::IndexVsFiles,
            index_path: "idx".into(),
            missing_files: Vec::new(),
            extra_files: Vec::new(),
            missing_tensors: Vec::new(),
            extra_tensors: Vec::new(),
            mismatched_tensors: Vec::new(),
            unverified_tensors: Vec::new(),
        })
        .expect("a health report serialises");
        has_keys(
            "HealthReport",
            &report,
            &[
                "kind",
                "index_path",
                "missing_files",
                "extra_files",
                "missing_tensors",
                "extra_tensors",
                "mismatched_tensors",
                "unverified_tensors",
            ],
        );
    }
}
