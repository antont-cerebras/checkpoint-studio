//! Tests for switching the served checkpoint at runtime.
//!
//! The interesting properties are not "does it read the second checkpoint" — that is the
//! ordinary read path — but what happens to everything *around* the swap: the per-checkpoint
//! caches, the state a failed open must not touch, and the two-at-once guard.

use std::net::IpAddr;
use std::path::{Path, PathBuf};

use super::current::Current;
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
    Current::new(opened, None, Options::default(), LOOPBACK).expect("state builds")
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
        .open(&other.to_string_lossy())
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

#[test]
fn a_failed_open_changes_nothing() {
    let current = serving("tiny.safetensors");
    let before = current.snapshot().root.clone();
    let recents_before = current.recents();

    // `match` rather than `expect_err`: the Ok side is a `WebState`, which has no `Debug`
    // (it holds mutexes and a whole checkpoint — nothing worth printing on a failure).
    let Err(err) = current.open("/definitely/not/a/checkpoint") else {
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

    current.open(&other.to_string_lossy()).expect("opens");

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

    current.open(&spec).expect("opens");
    current.open(&spec).expect("opens again");

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
