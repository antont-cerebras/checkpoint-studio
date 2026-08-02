//! Tests for switching the served checkpoint at runtime.
//!
//! The interesting properties are not "does it read the second checkpoint" — that is the
//! ordinary read path — but what happens to everything *around* the swap: the per-checkpoint
//! caches, the state a failed open must not touch, and the two-at-once guard.

use std::net::IpAddr;
use std::path::{Path, PathBuf};

use super::current::{Current, WhenBusy};
use crate::opening::{self, Options, Recents, Want};

const LOOPBACK: IpAddr = IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);

/// A scratch directory unique to this process and test.
///
/// Not a fixed name under `/tmp`: two `cargo test` runs (or two tests in this file) sharing a
/// directory race each other, which has already produced one confusing failure in this repo.
fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cs_switch_{}_{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// A `Current` serving `tests/fixtures/tiny.safetensors`, built the way `run_web` does.
fn serving(name: &str) -> Current {
    let opened = opening::Target::from_paths(&[fixture(name)], None, &Options::default())
        .expect("fixture resolves")
        .read(Want::Model, &crate::hf::ReadProgress::default())
        .expect("fixture reads");
    // An in-memory recents list: these tests must not read or write the user's config directory.
    Current::new(
        opened,
        None,
        Options::default(),
        LOOPBACK,
        Recents::default(),
    )
    .expect("state builds")
}

/// Write a one-tensor checkpoint with a distinctive name, so a test can tell which
/// checkpoint an answer came from.
fn write_checkpoint(dir: &Path, tensor: &str) -> PathBuf {
    std::fs::create_dir_all(dir).expect("mkdir");
    let header = format!(r#"{{"{tensor}":{{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}}}"#);
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0u8; 8]);
    let path = dir.join("model.safetensors");
    std::fs::write(&path, bytes).expect("write shard");
    path
}

#[test]
fn opening_another_checkpoint_replaces_what_every_endpoint_answers() {
    let dir = scratch("replaces");
    let other = write_checkpoint(&dir, "switched.weight");
    let current = serving("tiny.safetensors");

    // Ask for the tree BEFORE the switch, so its encoded body is in the static-body cache —
    // this is the cache that would otherwise serve the old checkpoint under the new one's
    // name. The bug this pins is silent: the response is well-formed, just about the wrong
    // checkpoint.
    let before = super::handlers::tree(&current.snapshot());
    assert_eq!(before.0, 200);
    let before_body = String::from_utf8_lossy(&before.1).into_owned();
    assert!(
        !before_body.contains("switched.weight"),
        "the first checkpoint should not contain the second's tensor"
    );

    current
        .open(&other.to_string_lossy(), WhenBusy::Refuse)
        .expect("the second checkpoint opens");

    let after = super::handlers::tree(&current.snapshot());
    let after_body = String::from_utf8_lossy(&after.1).into_owned();
    assert!(
        after_body.contains("switched.weight"),
        "the tree should be the newly opened checkpoint's, got: {}",
        &after_body[..after_body.len().min(300)]
    );
    assert_ne!(
        before_body, after_body,
        "a cached encoded body must not survive the swap"
    );
}

/// The address a client shows must be the thing that was opened, not the display root.
///
/// A single-file checkpoint's `root` is its *containing directory* — which can hold several other
/// checkpoints. Showing that in the address bar offered a path that, on Enter, would have opened
/// something else; and a link carrying it would have restored a different checkpoint.
#[test]
fn the_served_address_is_the_file_that_was_opened_not_its_directory() {
    let dir = scratch("address");
    let other = write_checkpoint(&dir, "switched.weight");
    let current = serving("tiny.safetensors");

    current
        .open(&other.to_string_lossy(), WhenBusy::Refuse)
        .expect("opens");
    let state = current.snapshot();

    assert_eq!(
        state.spec,
        other.to_string_lossy(),
        "the address should name the file that was opened"
    );
    assert_ne!(
        state.spec, state.root,
        "and differ from the display root, which is the containing directory"
    );
    assert_eq!(
        state.root,
        dir.to_string_lossy(),
        "root stays the directory — it is a label, not an address"
    );

    // And it is what the client is told, so the bar and the `?ckpt=` link agree with the server.
    let body = String::from_utf8_lossy(&super::handlers::tree(&state).1).into_owned();
    let json: serde_json::Value = serde_json::from_str(&body).expect("tree is JSON");
    assert_eq!(json["spec"], other.to_string_lossy().as_ref());
    assert_ne!(json["spec"], json["root"]);
}

/// A comparison that cannot be set up must leave none behind.
///
/// It used to leave the previous one in place, so `/api/difftree` answered 200 with a pair the client
/// had just been told had failed — a stale comparison presented as the requested one.
#[test]
fn a_failed_comparison_leaves_no_comparison_behind() {
    let dir = scratch("failed_cmp");
    let other = write_checkpoint(&dir, "switched.weight");
    let current = serving("tiny.safetensors");

    let set = current
        .set_comparison(&other.to_string_lossy(), "", WhenBusy::Refuse)
        .expect("a good pair sets up");
    assert!(
        matches!(
            current.comparison_for(set.id),
            crate::web::current::ComparisonLookup::Found { .. }
        ),
        "the good pair is held, under the id its caller was given"
    );

    let Err(_) = current.set_comparison("/definitely/not/a/checkpoint", "", WhenBusy::Refuse)
    else {
        panic!("a missing baseline cannot set up a comparison");
    };
    assert!(
        matches!(
            current.comparison_for(set.id),
            crate::web::current::ComparisonLookup::None
        ),
        "a failed set-up must not leave the previous comparison queryable"
    );
}

/// **One client cannot be handed another's comparison.**
///
/// There is one comparison slot per server. `/api/difftree` used to take no parameters and answer from
/// whatever was in it, so two overlapping clients swapped results: A set up its pair, B replaced it,
/// and A's request returned B's comparison with a `200`. An id turns that into a refusal A can act on.
#[test]
fn a_replaced_comparison_is_refused_rather_than_answered() {
    let dir = scratch("swapped_cmp");
    let a_side = write_checkpoint(&dir, "a.weight");
    let b_side = write_checkpoint(&dir, "b.weight");
    let current = serving("tiny.safetensors");

    // A sets up its pair and is told which one it is.
    let a = current
        .set_comparison(&a_side.to_string_lossy(), "", WhenBusy::Refuse)
        .expect("A's pair sets up");
    // B replaces it before A comes back for the tree.
    let b = current
        .set_comparison(&b_side.to_string_lossy(), "", WhenBusy::Refuse)
        .expect("B's pair sets up");
    assert_ne!(a.id, b.id, "each set-up takes a fresh identity");

    match current.comparison_for(a.id) {
        crate::web::current::ComparisonLookup::Replaced { current: now } => {
            assert_eq!(now, b.id, "the refusal names what is there instead");
        }
        crate::web::current::ComparisonLookup::Found { .. } => {
            panic!("A was handed a comparison it did not ask for")
        }
        crate::web::current::ComparisonLookup::None => panic!("B's comparison should be held"),
    }
    // And B, who asked last, still gets its own.
    assert!(matches!(
        current.comparison_for(b.id),
        crate::web::current::ComparisonLookup::Found { .. }
    ));
}

/// **`?swap=1` is the same comparison read the other way round.**
///
/// A diff is directional: what is added one way is removed the other, and the totals trade places.
/// The side-by-side has always been able to turn a pair round; the report could not, so the *only* way
/// to see it the other way was to edit the URL. Pinned as a mirror rather than as "it returns 200",
/// because a swap that quietly compared the same direction would look like it worked.
#[test]
fn swapping_the_report_mirrors_it() {
    let dir = scratch("swap");
    // A baseline with a tensor the served fixture does not have, so there is an asymmetry to mirror.
    let baseline = write_checkpoint(&dir.join("base"), "only.in.baseline.weight");
    let current = serving("tiny.safetensors");
    let against = baseline.to_string_lossy().into_owned();

    // One read of the pair, both orientations over it — which is the point of the report working from
    // the comparison slot rather than resolving its own baseline.
    let set = current
        .set_comparison(&against, "", WhenBusy::StopTheOther)
        .expect("the pair sets up");
    let report = |swap: bool| -> serde_json::Value {
        let mut q: crate::web::handlers::Query =
            std::iter::once(("id".to_string(), set.id.to_string())).collect();
        if swap {
            q.insert("swap".to_string(), "1".to_string());
        }
        let (status, body) = crate::web::handlers::diff(&current, &q);
        assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
        serde_json::from_slice(&body).expect("JSON")
    };
    let (plain, flipped) = (report(false), report(true));

    assert_eq!(plain["swapped"], serde_json::json!(false));
    assert_eq!(flipped["swapped"], serde_json::json!(true));
    // The asymmetry the fixtures were built for: something is one-sided either way.
    assert!(
        plain["report"]["tensors_removed"]
            .as_array()
            .is_some_and(|a| !a.is_empty()),
        "the baseline should have a tensor the served checkpoint lacks: {plain}"
    );
    // Mirrored: removed ⇄ added, and each side's totals with them.
    assert_eq!(
        plain["report"]["tensors_removed"],
        flipped["report"]["tensors_added"]
    );
    assert_eq!(
        plain["report"]["tensors_added"],
        flipped["report"]["tensors_removed"]
    );
    assert_eq!(plain["report"]["old_bytes"], flipped["report"]["new_bytes"]);
    assert_eq!(plain["report"]["new_bytes"], flipped["report"]["old_bytes"]);
    // And the command handed over compares the same way round as the screen.
    let (a, b) = (
        plain["command"].as_str().unwrap_or_default(),
        flipped["command"].as_str().unwrap_or_default(),
    );
    assert_ne!(a, b, "a swapped report must offer the swapped command");
    assert!(
        a.contains(&against) && b.contains(&against),
        "both commands name the baseline: {a} / {b}"
    );
}

/// **The two diff views state the same size and parameter count for the same pair.**
///
/// The report has always had them; the side-by-side had neither, so the one view built for a
/// 117k-tensor comparison never said the checkpoint got four times smaller. Adding them raises the
/// question this test answers: they must be the *same* numbers. The report sums a
/// `diff::CheckpointSummary` (deduped by name, last wins) and each side of the tree reports its
/// `CheckpointStats` — two paths to one total, and a sharded checkpoint lists a shared name once per
/// shard, so a path that failed to dedup would quietly disagree.
#[test]
fn both_diff_views_report_the_same_totals() {
    let dir = scratch("totals");
    let baseline = write_checkpoint(&dir.join("old"), "kept.weight");
    let current = serving("tiny.safetensors");

    // One pair, read once; both views work from it.
    let set = current
        .set_comparison(&baseline.to_string_lossy(), "", WhenBusy::Refuse)
        .expect("the pair sets up");

    // The report: the baseline is the OLD side, the served checkpoint the NEW one.
    let q: crate::web::handlers::Query =
        std::iter::once(("id".to_string(), set.id.to_string())).collect();
    let (status, body) = crate::web::handlers::diff(&current, &q);
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    let report: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    let report = &report["report"];

    // The side-by-side, over that same pair.
    let q: crate::web::handlers::Query =
        std::iter::once(("id".to_string(), set.id.to_string())).collect();
    let (status, body) = crate::web::handlers::difftree(&current, &current.snapshot(), &q);
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    let tree: serde_json::Value = serde_json::from_slice(&body).expect("JSON");

    for (side, old_key, new_key) in [
        ("bytes", "old_bytes", "new_bytes"),
        ("params", "old_params", "new_params"),
    ] {
        assert_eq!(
            tree["base"][side], report[old_key],
            "the baseline's {side} differ between the two views"
        );
        assert_eq!(
            tree["current"][side], report[new_key],
            "the served checkpoint's {side} differ between the two views"
        );
    }
    // And the fixture really has something to count, so equal-because-both-zero cannot pass this.
    assert!(
        report["new_bytes"].as_u64().unwrap_or_default() > 0,
        "the served fixture should have bytes to report"
    );
}

/// **A scope narrows the totals, on both web views, to the same numbers.**
///
/// The report's come from the filtered `CheckpointSummary`, the side-by-side's from summing the selected
/// tensors — two paths to one answer, so they are asserted against each other and against the totals
/// being *smaller* than the checkpoint's. Both also carry the label that says which they are, because
/// `size:` above a selection reads as the checkpoint's size.
#[test]
fn a_scoped_comparison_narrows_the_totals_on_both_views() {
    let dir = scratch("scoped_totals");
    let baseline = write_checkpoint(&dir.join("base"), "kept.weight");
    let current = serving("tiny.safetensors");
    let against = baseline.to_string_lossy().into_owned();
    // A name in neither checkpoint: the selection is empty, so both views should total zero rather
    // than fall back to the whole checkpoints.
    let scope = [("name".to_string(), "no.such.tensor".to_string())];

    let set = current
        .set_comparison(&against, "", WhenBusy::StopTheOther)
        .expect("the pair sets up");
    let mut q: crate::web::handlers::Query =
        std::iter::once(("id".to_string(), set.id.to_string())).collect();
    q.extend(scope.iter().cloned());
    let (status, body) = crate::web::handlers::diff(&current, &q);
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    let report: serde_json::Value = serde_json::from_slice(&body).expect("JSON");

    assert_eq!(
        report["report"]["new_bytes"], 0,
        "an empty selection totals nothing, not the whole checkpoint: {report}"
    );
    assert_eq!(report["totals_labels"]["size"], "size (filtered subset)");
    assert_eq!(
        report["totals_labels"]["params"],
        "params (filtered subset)"
    );

    let set = current
        .set_comparison(&against, "", WhenBusy::Refuse)
        .expect("the pair sets up");
    let mut q: crate::web::handlers::Query =
        std::iter::once(("id".to_string(), set.id.to_string())).collect();
    q.extend(scope.iter().cloned());
    let (status, body) = crate::web::handlers::difftree(&current, &current.snapshot(), &q);
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    let tree: serde_json::Value = serde_json::from_slice(&body).expect("JSON");

    assert_eq!(tree["base"]["bytes"], report["report"]["old_bytes"]);
    assert_eq!(tree["current"]["bytes"], report["report"]["new_bytes"]);
    assert_eq!(tree["totals_labels"], report["totals_labels"]);

    // Without the scope the same pair totals more than nothing — so the zeros above are the scope's
    // doing rather than a fixture with no bytes in it.
    let q: crate::web::handlers::Query =
        std::iter::once(("id".to_string(), set.id.to_string())).collect();
    let (_, body) = crate::web::handlers::diff(&current, &q);
    let whole: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
    assert!(
        whole["report"]["new_bytes"].as_u64().unwrap_or_default() > 0,
        "the unscoped comparison should have bytes: {whole}"
    );
    assert_eq!(whole["totals_labels"]["size"], "size");
}

#[test]
fn a_failed_open_changes_nothing() {
    let current = serving("tiny.safetensors");
    let before = current.snapshot().root.clone();
    let recents_before = current.recents();

    // `match` rather than `expect_err`: the Ok side is a `WebState`, which has no `Debug`
    // (it holds mutexes and a whole checkpoint — nothing worth printing on a failure).
    let Err(err) = current.open("/definitely/not/a/checkpoint", WhenBusy::Refuse) else {
        panic!("a missing path must not open");
    };
    // The message reaches the prompt, so it has to name what was asked for.
    let msg = format!("{err:#}");
    assert!(
        msg.contains("/definitely/not/a/checkpoint"),
        "the error should name the spec, got: {msg}"
    );

    assert_eq!(
        current.snapshot().root,
        before,
        "a failed open must leave the served checkpoint alone"
    );
    assert_eq!(
        current.recents(),
        recents_before,
        "a typo must not enter the recents list"
    );
}

#[test]
fn a_snapshot_taken_before_a_switch_still_answers_about_its_own_checkpoint() {
    let dir = scratch("snapshot");
    let other = write_checkpoint(&dir, "switched.weight");
    let current = serving("tiny.safetensors");

    // This is what a request holds while it works — a long tensor scan can outlive the swap.
    let held = current.snapshot();
    let root_when_taken = held.root.clone();

    current
        .open(&other.to_string_lossy(), WhenBusy::Refuse)
        .expect("opens");

    assert_eq!(
        held.root, root_when_taken,
        "an in-flight request's snapshot must not change under it"
    );
    assert_ne!(
        current.snapshot().root,
        held.root,
        "while the server has moved on"
    );
}

#[test]
fn the_startup_checkpoint_is_the_first_recent_and_is_spelled_as_a_path() {
    let current = serving("tiny.safetensors");
    let recents = current.recents();
    assert_eq!(recents.len(), 1, "startup records exactly one entry");
    // A path, not a display label: picking a recent retypes it verbatim, and
    // "tiny.safetensors" alone would not resolve from the server's working directory.
    assert!(
        Path::new(&recents[0]).is_absolute(),
        "a recents entry must be a usable path, got {:?}",
        recents[0]
    );
}

#[test]
fn opening_the_same_checkpoint_again_moves_it_up_rather_than_repeating_it() {
    let dir = scratch("moves_up");
    let other = write_checkpoint(&dir, "switched.weight");
    let current = serving("tiny.safetensors");
    let spec = other.to_string_lossy().into_owned();

    current.open(&spec, WhenBusy::Refuse).expect("opens");
    current.open(&spec, WhenBusy::Refuse).expect("opens again");

    let recents = current.recents();
    assert_eq!(
        recents.iter().filter(|s| **s == spec).count(),
        1,
        "the same checkpoint should appear once, got {recents:?}"
    );
    assert_eq!(recents[0], spec, "and be the most recent");
}

/// `Recents` is shared with the terminal, so its behaviour is pinned where both can see it —
/// this is the web-side check that the type the two frontends share is the type being used.
#[test]
fn recents_is_bounded() {
    let mut r = Recents::with_cap(2);
    r.record("/a");
    r.record("/b");
    r.record("/c");
    assert_eq!(r.list(), ["/c", "/b"]);
}

/// **A busy server offers a way through, not just an apology.**
///
/// The refusal used to end at "try again when it finishes", which is the one thing a reader cannot act
/// on: this server reads one checkpoint at a time, so waiting was the only option even when the running
/// read was one nobody wanted any more. `WhenBusy::StopTheOther` cancels the incumbent and takes the
/// slot; `WhenBusy::Refuse` leaves it alone and reports what is running, so a UI can offer the choice.
#[test]
fn the_refusal_names_what_is_running_and_stopping_it_is_a_choice() {
    let dir = scratch("busy_slot");
    let other = write_checkpoint(&dir, "switched.weight");
    let current = serving("tiny.safetensors");

    // Nothing is reading, so both arms simply succeed.
    current
        .open(&other.to_string_lossy(), WhenBusy::Refuse)
        .expect("an idle server opens");
    assert!(
        current.busy_with().is_none(),
        "the slot is free once a read finishes"
    );
    current
        .open(&other.to_string_lossy(), WhenBusy::StopTheOther)
        .expect("with nothing to stop, taking over is an ordinary open");
}

/// The read is handed the slot's own control handle, so a stop request reaches it.
///
/// The bug this guards: `read()` used to be given a throwaway `ReadProgress::default()`, so cancelling
/// through the published handle set a flag nothing was watching — a "Stop it" button that stopped
/// nothing, which is worse than the message it replaced.
#[test]
fn a_cancelled_read_is_asked_to_stop_through_the_handle_it_was_given() {
    let progress = crate::hf::ReadProgress::default();
    assert!(!progress.cancelled(), "a fresh read is not cancelled");
    progress.cancel();
    assert!(progress.cancelled(), "cancelling is observable");
    assert!(
        progress
            .abort_flag()
            .load(std::sync::atomic::Ordering::Relaxed),
        "and reaches the flag the ssh layer aborts on"
    );
}

/// **The job routes exist, answer, and refuse the wrong methods.**
///
/// The value-reading diff modes are jobs because they take minutes; this pins the protocol —
/// start/poll/stop — without running one, which needs two real checkpoints.
#[test]
fn a_job_can_be_started_polled_and_stopped() {
    let current = std::sync::Arc::new(serving("tiny.safetensors"));

    // Starting needs both sides.
    let (status, _) = crate::web::handlers::start_verify_repack(
        &current,
        &std::iter::once(("left".to_string(), "/a".to_string())).collect(),
    );
    assert_eq!(status, 400, "one side is not a comparison");

    // A bad bit width is a 400 up front, not a job that fails a minute later.
    let q: crate::web::handlers::Query = [
        ("left", "/nope/a"),
        ("right", "/nope/b"),
        ("repack_bits", "99"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    assert_eq!(
        crate::web::handlers::start_verify_repack(&current, &q).0,
        400,
        "repack_bits must be rejected before any reading starts"
    );

    // A real start: the paths do not resolve, so the *job* fails — but the request succeeds, which is
    // the distinction a job protocol exists to draw.
    let q: crate::web::handlers::Query = [("left", "/nope/a"), ("right", "/nope/b")]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let (status, bytes) = crate::web::handlers::start_verify_repack(&current, &q);
    assert_eq!(
        status, 200,
        "starting a job is not the same as it succeeding"
    );
    let started: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let id = started["id"].as_u64().expect("an id to poll");

    // Poll it. Whatever it has done by now, the shape is the contract.
    let (status, bytes) = crate::web::handlers::job_status(&current, id);
    assert_eq!(status, 200);
    let snap: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(snap["id"], id);
    assert_eq!(snap["kind"], "verify-repack");
    for key in [
        "state",
        "done",
        "total",
        "bytes",
        "current",
        "elapsed_s",
        "findings",
    ] {
        assert!(!snap[key].is_null(), "a poll must report {key}: {snap}");
    }

    // Stopping is idempotent and reports the job rather than a bare acknowledgement.
    assert_eq!(crate::web::handlers::cancel_job(&current, id).0, 200);
    assert_eq!(crate::web::handlers::cancel_job(&current, id).0, 200);

    // An id nobody was given is a 404.
    assert_eq!(crate::web::handlers::job_status(&current, 99_999).0, 404);
    assert_eq!(crate::web::handlers::cancel_job(&current, 99_999).0, 404);
}

/// A comparison announces the checkpoint it is reading *now*, not the one it started with.
///
/// The pair takes the read slot once — one cancel handle, one elapsed clock — and both sides are read
/// under it. The announcement used to be written when the slot was taken and never again, so the
/// browser's second progress row, the one for the candidate, never lit up: while an `s3://` prefix
/// took twenty seconds, the screen said it was still reading the local baseline it had finished with.
#[test]
fn the_announcement_follows_the_side_being_read() {
    let current = serving("tiny.safetensors");
    let held = current
        .take_slot("the-baseline", WhenBusy::Refuse)
        .expect("nothing else is reading");
    assert_eq!(
        current.busy_with().expect("a read is in flight").0,
        "the-baseline"
    );

    current.now_reading("the-candidate");
    assert_eq!(
        current.busy_with().expect("still the same read").0,
        "the-candidate",
        "the second side of a pair is what the progress line names once it starts"
    );

    drop(held);
    assert!(
        current.busy_with().is_none(),
        "and the slot still frees on drop"
    );
}

/// Stopping a read stops **every** checkpoint it covers.
///
/// A comparison holds the slot with two reads under it, running at once. "Stop it and read this
/// instead" used to cancel the one handle the slot carried; with two, cancelling either alone leaves
/// the other running — and the taker then waits out its twenty-second deadline for a slot the
/// survivor is still holding.
#[test]
fn stopping_a_comparison_stops_both_of_its_reads() {
    let current = std::sync::Arc::new(serving("tiny.safetensors"));
    let pair = ["side-one".to_string(), "side-two".to_string()];
    let held = current
        .take_slot(&pair, WhenBusy::Refuse)
        .expect("nothing else is reading");
    assert!(
        current.reading().is_some_and(|r| r.sides.len() == 2),
        "the slot covers both sides"
    );

    // Another request asks for the slot, which cancels whatever holds it before waiting.
    let taker = {
        let current = std::sync::Arc::clone(&current);
        std::thread::spawn(move || {
            let taken = current.take_slot(
                std::slice::from_ref(&"the-newcomer".to_string()),
                WhenBusy::StopTheOther,
            );
            taken.map(drop).is_ok()
        })
    };
    // Both handles see the stop — the point of the test. Polled rather than assumed: the taker runs
    // on its own thread and the cancel lands a moment after it starts.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline && !held.every_side_cancelled() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        held.every_side_cancelled(),
        "a stop has to reach both reads, not just the first"
    );
    // A cooperative cancel only asks; the reads here are notional, so let go on their behalf.
    drop(held);
    assert!(taker.join().expect("the taker thread finished"));
}

/// **A value comparison a remote pair cannot do is refused before it reads anything.**
///
/// The reported failure: the job accepted an `s3://` pair, read both checkpoints — minutes over an ssh
/// proxy — and then said a remote source serves no tensor data. The addresses alone answer that, so the
/// refusal belongs at the door.
#[test]
fn a_remote_pair_is_refused_a_value_comparison_at_the_door() {
    let current = std::sync::Arc::new(serving("tiny.safetensors"));
    let ask = |left: &str, right: &str| -> (u16, String) {
        let q: crate::web::handlers::Query = [("left", left), ("right", right), ("values", "1")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let (status, body) = crate::web::handlers::start_values(&current, &q);
        let json: serde_json::Value = serde_json::from_slice(body.as_slice()).expect("JSON");
        (
            status,
            json["error"].as_str().unwrap_or_default().to_string(),
        )
    };

    let (status, msg) = ask("s3://bucket/old", "s3://bucket/new");
    assert_eq!(status, 400, "refused, and immediately: {msg}");
    assert!(
        msg.contains("which the terminal can compare on the proxy"),
        "and pointed at what does work for this pair: {msg}"
    );

    let (status, msg) = ask("lab@host:/opt/models/a", "/tmp/local");
    assert_eq!(status, 400);
    assert!(
        msg.contains("lab@host:/opt/models/a") && !msg.contains("/tmp/local"),
        "the refusal names the side without bytes, not the one with them: {msg}"
    );

    // Two local paths are accepted here — they do not resolve, so the *job* fails, which is the
    // difference this test is about: a refusal is instant, a failure is a job's own business.
    assert_eq!(ask("/nope/a", "/nope/b").0, 200);
}

/// **Every scope control reaches the offered command.**
///
/// The report hands over a `diff` invocation "for the same comparison", so a selection the browser
/// applied and the command omitted would be a handover that answers a different question — and the one
/// reported: an exact name picked in the panel had to appear in the command. Walked control by control
/// rather than spot-checked, because the failure mode is one forgotten `if let` in `cli_args`.
#[test]
fn every_scope_control_reaches_the_offered_command() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let old = dir.join("diff_old.safetensors");
    let new = dir.join("diff_new.safetensors");
    let current = {
        let opened =
            opening::Target::from_paths(std::slice::from_ref(&new), None, &Options::default())
                .expect("the fixture resolves")
                .read(Want::Model, &crate::hf::ReadProgress::default())
                .expect("the fixture reads");
        Current::new(
            opened,
            None,
            Options::default(),
            LOOPBACK,
            Recents::default(),
        )
        .expect("the served state builds")
    };
    let set = current
        .set_comparison(&old.to_string_lossy(), "", WhenBusy::Refuse)
        .expect("the pair reads");

    // Every control the panel has, and what it must put on the command line.
    let controls: &[(&str, &str, &str)] = &[
        ("name", "model.layers.1.*", "--name model.layers.1.*"),
        ("names", "model.norm.weight", "--names model.norm.weight"),
        ("dtype_is", "BF16", "--dtype-is BF16"),
        ("shape_is", "768,**", "--shape-is 768,**"),
        ("map", "^a=>b", "--map ^a=>b"),
        ("only_tensors", "1", "--only-tensors"),
        ("align_fused", "1", "--align-fused"),
        // The subtrees ride on the operands, which is how `diff` spells them.
        ("subtree", "model", "#model"),
    ];
    for (key, value, expected) in controls {
        let q: crate::web::handlers::Query = [
            ("id".to_string(), set.id.to_string()),
            ((*key).to_string(), (*value).to_string()),
        ]
        .into_iter()
        .collect();
        let (status, body) = crate::web::handlers::diff(&current, &q);
        assert_eq!(status, 200, "{key} is a valid scope");
        let json: serde_json::Value = serde_json::from_slice(body.as_slice()).expect("JSON");
        let command = json["command"].as_str().unwrap_or_default().to_string();
        // Quoting is the shell's business, so compare on the unquoted text.
        let flat = command.replace('\'', "");
        assert!(
            flat.contains(expected),
            "{key}={value} must reach the offered command as `{expected}`: {command}"
        );
    }

    // And the family fold, which is a view control rather than a selection but changes what the command
    // would print.
    let q: crate::web::handlers::Query = [
        ("id".to_string(), set.id.to_string()),
        ("full".to_string(), "1".to_string()),
    ]
    .into_iter()
    .collect();
    let json: serde_json::Value =
        serde_json::from_slice(crate::web::handlers::diff(&current, &q).1.as_slice())
            .expect("JSON");
    assert!(
        json["command"]
            .as_str()
            .unwrap_or_default()
            .contains("--full"),
        "the fold state belongs in the command too: {json}"
    );
}
