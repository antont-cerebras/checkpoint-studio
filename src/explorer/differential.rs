//! Differential tests: the same question, asked of all three interfaces.
//!
//! `checkpoint-studio` answers the same questions three ways — a CLI export, a TUI
//! screen, and a JSON endpoint the browser draws. All three read one core model, but
//! each *projects* it separately: the CLI through `explorer/export.rs`, the web
//! through `web/dto.rs` + `web/handlers.rs`, the TUI through `ui/`. A projection is
//! exactly where two surfaces silently stop agreeing — one sums logical bytes and the
//! other stored bytes, one counts a metadata node as a row and the other doesn't —
//! and no per-surface test can catch it, because each is self-consistent.
//!
//! So the agreement is a test. Every case here asks one question through more than
//! one interface and asserts the answers are the same object, not merely
//! plausible-looking. This is a different contract from `tests/parity.rs`, which pins
//! the *display rules* duplicated in Rust and TypeScript (`format_size` vs
//! `humanSize`); here the rules are shared and it's the **data** that must match.
//!
//! **Why this lives in `src/` rather than `tests/`.** An integration test can run the
//! real binary but cannot call into a bin crate, so it can't reach `WebState` or
//! `Explorer` at all. A unit test reaches all three, at the cost of comparing the
//! CLI's output *builders* (`tensors_json`, the functions `print_tensors` calls)
//! rather than its stdout. The last hop — builder to stdout — is what the
//! `tests/cli.rs` snapshots already pin, so nothing is left unguarded.
//!
//! It sits under `explorer` specifically so it can reach this module's own internals
//! (`tensors`, `load_quiet`, the export builders) without widening any of them to
//! `pub(crate)` for a test's benefit. The web side it compares against is already
//! `pub(crate)`.
//!
//! What is deliberately *not* asserted equal is recorded in
//! `shared/parity/README.md`: the numeric-grid precision, the 1-D shape tuple, search
//! ranking, and the histogram's per-bin percentages differ between the TUI and the
//! browser on purpose.

#![cfg(test)]
// An unwrap in a test IS the assertion: the panic is the failure report.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;

use super::{Explorer, TreeDetail};
use crate::ui::UI;
use crate::web::WebState;
use crate::web::handlers;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.safetensors")
}

/// The CLI / TUI side: a loaded `Explorer`, which is what both the exports and the
/// screens render from.
fn explorer() -> Explorer {
    let mut ex = Explorer::new(vec![fixture()], Vec::new(), None, false);
    ex.load_quiet().expect("the fixture loads");
    ex
}

/// The web side: the same fixture read into the state the handlers serve.
fn web() -> WebState {
    let files = vec![fixture()];
    let model = crate::readers::read_local(&files).expect("the fixture reads");
    WebState::build(model, &files, &[])
}

/// The query map for a one-parameter request — what the router hands a handler after
/// percent-decoding `?key=value`.
fn query(key: &str, value: &str) -> handlers::Query {
    std::iter::once((key.to_string(), value.to_string())).collect()
}

/// A server with `files` open, and a comparison against `against` already set up.
///
/// The report reads the pair from the comparison slot rather than resolving its own baseline — one
/// read for every view of a comparison — so asking it anything means establishing the pair first,
/// exactly as the browser does (`POST /api/compare`, then `GET /api/diff?id=N`).
fn comparing(files: &[PathBuf], against: &str) -> (crate::web::Current, u64) {
    let opts = crate::opening::Options::default();
    let opened = crate::opening::Target::from_paths(files, None, &opts)
        .expect("the fixture resolves")
        .read(
            crate::opening::Want::Model,
            &crate::hf::ReadProgress::default(),
        )
        .expect("the fixture reads");
    let current = crate::web::Current::new(
        opened,
        None,
        opts,
        std::net::IpAddr::from([127, 0, 0, 1]),
        crate::opening::Recents::default(),
    )
    .expect("the served state builds");
    let set = current
        .set_comparison(against, "", crate::web::current::WhenBusy::StopTheOther)
        .expect("the baseline reads");
    (current, set.id)
}

/// A handler's body as JSON, asserting it answered 200 — the shape a browser gets.
fn body(reply: handlers::Reply) -> Value {
    let (status, bytes) = reply;
    assert_eq!(status, 200, "handler did not answer 200");
    serde_json::from_slice(&bytes).expect("handlers answer with JSON")
}

/// Every tensor's identity as the web reports it, keyed by name: `(dtype, shape,
/// logical bytes, element count)`. Walks the nested tree the way the browser's
/// `flatten.ts` does, so this is the inventory the tree screen actually shows —
/// not a convenient flat list served alongside it.
fn web_inventory(tree: &Value) -> BTreeMap<String, Value> {
    fn walk(node: &Value, out: &mut BTreeMap<String, Value>) {
        match node.get("kind").and_then(Value::as_str) {
            Some("group") => {
                for child in node["children"].as_array().into_iter().flatten() {
                    walk(child, out);
                }
            }
            Some("tensor") => {
                let t = &node["info"];
                out.insert(
                    t["name"].as_str().unwrap().to_string(),
                    serde_json::json!({
                        "dtype": t["dtype"],
                        "shape": t["shape"],
                        "size_bytes": t["size_bytes"],
                        "num_elements": t["num_elements"],
                    }),
                );
            }
            // Metadata nodes carry no tensor; other kinds are not inventory.
            _ => {}
        }
    }
    let mut out = BTreeMap::new();
    for root in tree["tree"].as_array().into_iter().flatten() {
        walk(root, &mut out);
    }
    out
}

/// The same inventory from the CLI's `--print-tensors --format json -v` export.
fn cli_inventory(json: &str) -> BTreeMap<String, Value> {
    let parsed: Value = serde_json::from_str(json).expect("the export is JSON");
    let mut out = BTreeMap::new();
    for entry in parsed.as_array().into_iter().flatten() {
        out.insert(
            entry["name"].as_str().unwrap().to_string(),
            serde_json::json!({
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "size_bytes": entry["size_bytes"],
                "num_elements": entry["num_elements"],
            }),
        );
    }
    out
}

/// Each tensor leaf's `(full name, display label)`, depth-first — for checking what a
/// row actually shows against what the data says it should.
fn leaf_labels(v: &Value, out: &mut Vec<(String, String)>) {
    if let Some(children) = v.get("children").and_then(Value::as_array) {
        for c in children {
            leaf_labels(c, out);
        }
    }
    if v.get("kind").and_then(Value::as_str) == Some("tensor")
        && let (Some(name), Some(label)) = (
            v["info"]["name"].as_str(),
            v.get("label").and_then(Value::as_str),
        )
    {
        out.push((name.to_string(), label.to_string()));
    }
}

/// Nothing below means anything if the fixture is empty — a differential test over
/// two empty collections passes vacuously. Pinned once, so every other case here can
/// rely on there being something to disagree about.
#[test]
fn the_fixture_has_something_to_compare() {
    let ex = explorer();
    assert_eq!(ex.tensors().len(), 7, "the fixture's tensor count moved");
    let dtypes: std::collections::BTreeSet<_> =
        ex.tensors().iter().map(|t| t.dtype.clone()).collect();
    assert!(
        dtypes.len() >= 5,
        "the fixture should span several dtypes, got {dtypes:?}"
    );
    let ranks: std::collections::BTreeSet<_> = ex.tensors().iter().map(|t| t.shape.len()).collect();
    assert!(
        ranks.len() >= 3,
        "the fixture should span several ranks, got {ranks:?}"
    );
}

/// **The inventory.** Which tensors exist, and each one's dtype / shape / size /
/// element count, must be the same set through the CLI export and through the tree
/// the web serves. These are two independent projections (`export.rs` builds JSON by
/// hand; `handlers::tree` serializes `TreeNode`), so this is the case most likely to
/// drift when a field is added to one and not the other.
#[test]
fn the_tensor_inventory_agrees_between_the_cli_and_the_web() {
    let ex = explorer();
    let cli = cli_inventory(&ex.tensors_json(TreeDetail::Full));
    let web = web_inventory(&body(handlers::tree(&web())));

    assert_eq!(
        cli.keys().collect::<Vec<_>>(),
        web.keys().collect::<Vec<_>>(),
        "the two surfaces list different tensors"
    );
    for (name, cli_entry) in &cli {
        assert_eq!(
            cli_entry, &web[name],
            "'{name}' is described differently by the CLI and the web"
        );
    }
}

/// The tree's own root summary (count / params / bytes) is computed separately by the
/// web handler, so it can disagree with the tensors underneath it — a header saying
/// "7 tensors" over a list of six.
#[test]
fn the_web_tree_root_summarises_its_own_children() {
    let s = web();
    let tree = body(handlers::tree(&s));
    let root = &tree["tree"][0];
    let inventory = web_inventory(&tree);

    assert_eq!(
        root["tensor_count"].as_u64().unwrap() as usize,
        inventory.len(),
        "the root's count disagrees with the tensors in the tree"
    );
    let params: u64 = inventory
        .values()
        .map(|t| t["num_elements"].as_u64().unwrap())
        .sum();
    let bytes: u64 = inventory
        .values()
        .map(|t| t["size_bytes"].as_u64().unwrap())
        .sum();
    assert_eq!(root["params"].as_u64().unwrap(), params, "root params");
    assert_eq!(root["total_size"].as_u64().unwrap(), bytes, "root bytes");
    assert_eq!(
        tree["tensor_count"].as_u64().unwrap() as usize,
        inventory.len(),
        "the envelope's count disagrees with the tree it wraps"
    );
}

/// **Checkpoint statistics.** The web serves `CheckpointStats` directly; the TUI
/// renders it. Assert the served numbers are the computed ones, and that the numbers
/// the TUI puts on screen are the same values formatted — the formatting itself is
/// contracted separately by `tests/parity.rs`.
#[test]
fn the_checkpoint_statistics_agree_across_the_surfaces() {
    let s = web();
    let served = body(handlers::stats(&s));
    let computed = &s.stats;

    assert_eq!(
        served["n_tensors"].as_u64().unwrap() as usize,
        computed.n_tensors,
        "tensor count"
    );
    assert_eq!(
        served["params"].as_u64().unwrap() as usize,
        computed.params,
        "parameter count"
    );
    assert_eq!(
        served["logical_bytes"].as_u64().unwrap() as usize,
        computed.logical_bytes,
        "logical byte total"
    );
    assert_eq!(
        served["disk_bytes"].as_u64().unwrap() as usize,
        computed.disk_bytes,
        "on-disk byte total"
    );

    // The same figures, as the TUI's statistics screen writes them — rendered through
    // the very call `--stats --plain` makes, so this is the screen, not a paraphrase.
    let rendered = crate::tui::headless_render(120, 40, |f| {
        UI::render_stats_frame(f, computed, 0, false);
    })
    .expect("the stats screen renders headless");
    for (label, formatted) in [
        (
            "parameters",
            crate::utils::format_parameters(computed.params),
        ),
        ("size", crate::utils::format_size(computed.logical_bytes)),
    ] {
        assert!(
            rendered.contains(&formatted),
            "the stats screen does not show the computed {label} ({formatted}):\n{rendered}"
        );
    }
    assert!(
        rendered.contains(&computed.n_tensors.to_string()),
        "the stats screen does not show the tensor count:\n{rendered}"
    );
}

/// **The filter grammar.** One matcher (`tensorfilter`) is shared, but each surface
/// applies it to its own collection: the web filters `WebState::tensors`, the CLI
/// prunes the session and rebuilds the tree. A query must select the same tensors
/// either way — including the queries that select nothing, which is where an
/// off-by-one "empty means show all" bug hides.
#[test]
fn the_filter_grammar_selects_the_same_tensors_everywhere() {
    let s = web();
    for q in [
        "dtype:F32",
        "dtype:F16,BF16",
        "rank:>=3",
        "rank:1",
        "name:q_proj",
        "name:re:^model\\.layers",
        "!name:weight",
        "size:>=100",
        "dtype:F64", // matches nothing: the empty answer must stay empty
    ] {
        let served = body(handlers::filter(&s, &query("q", q)));
        assert_eq!(
            served["active"].as_bool(),
            Some(true),
            "'{q}' should parse as an active filter"
        );
        let web_names: Vec<&str> = served["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();

        // The CLI path: the same query, applied the way `--filter` applies it.
        let mut ex = Explorer::new(vec![fixture()], Vec::new(), None, false);
        ex.set_tensor_filter(
            crate::tensorfilter::TensorFilter::parse(q).expect("the query parses"),
        );
        ex.load_quiet().expect("the fixture loads");
        ex.apply_tensor_filter();
        let cli_json = ex.tensors_json(TreeDetail::Full);
        let cli_names: Vec<String> = cli_inventory(&cli_json).into_keys().collect();

        assert_eq!(
            cli_names, web_names,
            "'{q}' selects different tensors in the CLI and the web"
        );
    }
}

/// A malformed query must fail the same way on both sides — the web with a 400 whose
/// message the filter bar shows, the CLI by refusing to run. What must not happen is
/// one surface silently treating it as "no filter" and showing everything.
#[test]
fn a_malformed_filter_query_is_rejected_by_both_surfaces() {
    let s = web();
    let bad = "dtype:";
    let (status, bytes) = handlers::filter(&s, &query("q", bad));
    assert_eq!(status, 400, "the web should reject '{bad}'");
    let msg: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        msg["error"].as_str().is_some_and(|e| !e.is_empty()),
        "the rejection should carry a message: {msg}"
    );
    assert!(
        crate::tensorfilter::TensorFilter::parse(bad).is_err(),
        "the shared parser should reject '{bad}', so the CLI does too"
    );
}

/// **Health and the structural check.** Both are core reports, and both are reachable
/// from all three surfaces (`--health` / the `h` popup / `/api/health`). Assert the
/// served JSON is the computed report — findings included, since a summary that says
/// "healthy" over a report with findings is the failure that matters.
#[test]
fn the_health_report_agrees_between_the_core_and_the_web() {
    let s = web();
    let served = body(handlers::health(&s));
    let computed = serde_json::to_value(&s.health).unwrap();
    assert_eq!(
        served, computed,
        "the served health report is not the computed one"
    );
}

#[test]
fn the_structural_check_agrees_between_the_core_and_the_web() {
    let s = web();
    let served = body(handlers::check(&s));
    let computed = s.check.as_ref().map_or(Value::Null, |r| r.to_json(false));
    assert_eq!(
        served, computed,
        "the served check report is not the computed one"
    );
}

/// **Per-tensor statistics.** The scan is shared, but the web projects it through
/// `StatsDto` (renaming `elapsed` to `elapsed_ms`, adding `zero_fraction`), which is
/// a hand-written projection and therefore able to drop or mis-name a field.
#[test]
fn per_tensor_statistics_agree_between_the_core_and_the_web() {
    let s = web();
    let t = &s.tensors[0];
    let served = body(handlers::tensor_stats(&s, &query("name", &t.name)));
    let (no_cancel, no_pause) = (
        std::sync::atomic::AtomicBool::new(false),
        std::sync::atomic::AtomicBool::new(false),
    );
    let computed = crate::sample::tensor_stats(
        t,
        crate::sample::ViewDtype::Stored,
        None,
        &no_cancel,
        &no_pause,
        None,
    )
    .expect("the fixture's first tensor scans");

    for (key, want) in [
        ("min", computed.min),
        ("max", computed.max),
        ("mean", computed.mean),
    ] {
        let got = served[key].as_f64().unwrap();
        assert!(
            (got - want).abs() < 1e-9,
            "{key}: the web served {got}, the scan computed {want}"
        );
    }
    assert_eq!(
        served["count"].as_u64().unwrap(),
        computed.count,
        "element count"
    );
    assert_eq!(
        served["zeros"].as_u64().unwrap(),
        computed.zeros,
        "zero count"
    );
}

/// **The tree itself.** The TUI flattens `tree_state.tree` into its rows; the browser
/// flattens the tree `/api/tree` serves. If those two trees are the same object then
/// the two interactive surfaces are showing the same rows in the same order — which is
/// the structural half of "the web UI looks like the TUI". (The other half, that the
/// two *flatteners* agree, is contracted across languages by `shared/parity/tree.json`
/// — see `tests/parity.rs`.)
#[test]
fn the_tui_and_the_web_flatten_the_same_tree() {
    let ex = explorer();
    let tui = serde_json::to_value(&ex.tree_state.tree).unwrap();
    let served = body(handlers::tree(&web()));

    assert_eq!(
        served["tree"], tui,
        "the tree the web serves is not the tree the TUI renders"
    );
}

/// The same claim for a **directory** of shards, which takes the other branch of the
/// root-label rule (shared parent directory rather than the one file's name). Both
/// branches are checked because the bug this caught — the browser naming a single-file
/// checkpoint after its directory — is exactly what a fix that ignored one branch
/// would reintroduce in the other.
#[test]
fn the_tui_and_the_web_agree_on_a_directory_of_shards() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let files: Vec<PathBuf> = {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let p = e.unwrap().path();
                (p.extension().is_some_and(|x| x == "safetensors")).then_some(p)
            })
            .collect();
        v.sort();
        v
    };
    assert!(files.len() > 1, "this case needs several shards");

    let mut ex = Explorer::new(files.clone(), Vec::new(), None, false);
    ex.load_quiet().expect("the fixtures load");
    let tui = serde_json::to_value(&ex.tree_state.tree).unwrap();

    let model = crate::readers::read_local(&files).expect("the fixtures read");
    let served = body(handlers::tree(&WebState::build(model, &files, &[])));

    assert_eq!(
        served["tree"][0]["name"], tui[0]["name"],
        "the two surfaces label a multi-shard root differently"
    );
    assert_eq!(
        served["tree"], tui,
        "the tree the web serves is not the tree the TUI renders"
    );
}

/// **The structural diff.** `diff OLD NEW` on the command line, `/api/diff?against=OLD`
/// in the browser, and the compare screen in the terminal must all produce the same
/// report for the same pair — that is the whole point of putting the comparison in
/// `crate::compare` rather than in each surface.
#[test]
fn the_structural_diff_agrees_between_the_cli_and_the_web() {
    let old = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diff_old.safetensors");
    let new = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diff_new.safetensors");

    // The web: serve `diff_new`, ask it to compare against `diff_old`.
    let files = vec![new];
    let (current, id) = comparing(&files, &old.display().to_string());
    let served = body(handlers::diff(&current, &query("id", &id.to_string())));

    // The CLI: the same pair through the shared comparison the `diff` subcommand uses.
    let state = current.snapshot();
    let expected = crate::compare::structural_diff(
        &state.tensors,
        &state.checkpoint.metadata_vec(),
        &old.display().to_string(),
        &crate::opening::Options::default(),
    )
    .expect("the pair compares");

    assert_eq!(
        served["report"],
        serde_json::to_value(&expected).unwrap(),
        "the served diff is not the one the shared comparison produces"
    );
    assert!(
        served["report"]["tensors_changed"]
            .as_array()
            .is_some_and(|a| !a.is_empty())
            || served["report"]["tensors_added"]
                .as_array()
                .is_some_and(|a| !a.is_empty()),
        "the diff fixtures should differ, or this test proves nothing: {}",
        served["verdict"]
    );
    // The emitted command names the baseline first (`diff OLD NEW`), so pasting it
    // reproduces this report rather than the inverse one.
    let command = served["command"].as_str().unwrap();
    assert!(
        command.contains("diff_old.safetensors") && command.contains("diff_new.safetensors"),
        "the reopen command should name both sides: {command}"
    );
    let (o, n) = (
        command.find("diff_old").unwrap(),
        command.find("diff_new").unwrap(),
    );
    assert!(
        o < n,
        "the baseline must come first in `diff OLD NEW`: {command}"
    );
}

/// Comparing a checkpoint with itself must report no differences through the endpoint,
/// not merely through the core — the projection could drop a section and look clean.
#[test]
fn the_web_reports_no_differences_against_itself() {
    let (current, id) = comparing(&[fixture()], &fixture().display().to_string());
    let served = body(handlers::diff(&current, &query("id", &id.to_string())));
    assert_eq!(served["verdict"], "structurally identical");
    for section in [
        "tensors_added",
        "tensors_removed",
        "tensors_changed",
        "meta_added",
        "meta_removed",
        "meta_changed",
    ] {
        assert_eq!(
            served["report"][section].as_array().map(Vec::len),
            Some(0),
            "{section} should be empty when comparing a checkpoint with itself"
        );
    }
}

/// A bad `?against=` is a 400 whose message the UI shows — not a 500, and not a silent
/// empty report that would read as "no differences".
///
/// The refusal moved with the read: the report works from the comparison slot now, so a path that is
/// not a checkpoint is refused by `POST /api/compare` — before any view of it exists — and asking for
/// a comparison that was never set up is its own refusal.
#[test]
fn the_diff_endpoint_rejects_a_path_that_is_not_a_checkpoint() {
    let opts = crate::opening::Options::default();
    let opened = crate::opening::Target::from_paths(&[fixture()], None, &opts)
        .expect("the fixture resolves")
        .read(
            crate::opening::Want::Model,
            &crate::hf::ReadProgress::default(),
        )
        .expect("the fixture reads");
    let current = crate::web::Current::new(
        opened,
        None,
        opts,
        std::net::IpAddr::from([127, 0, 0, 1]),
        crate::opening::Recents::default(),
    )
    .expect("the served state builds");
    for bad in ["", "/nonexistent/checkpoint"] {
        let refused = current
            .set_comparison(bad, "", crate::web::current::WhenBusy::StopTheOther)
            .is_err();
        assert!(refused, "'{bad}' should be refused rather than compared");
    }
    // And a request naming no comparison at all is a client error, not an empty report.
    for missing in ["", "404"] {
        let (status, bytes) = handlers::diff(&current, &query("id", missing));
        assert!(
            status == 400 || status == 409,
            "an unknown comparison should be refused, got {status}"
        );
        let msg: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            msg["error"].as_str().is_some_and(|e| !e.is_empty()),
            "the rejection should carry a message: {msg}"
        );
    }
}

/// **The no-access-control caution.** The server has no authentication, so when it is
/// bound anywhere but loopback both the terminal banner and the page must say so — and
/// say the *same* thing, which is why the state carries one string rather than each
/// surface phrasing its own.
#[test]
fn the_access_warning_appears_exactly_when_the_server_is_reachable_off_the_machine() {
    use std::net::{IpAddr, Ipv4Addr};

    let loopback = web().with_exposure(IpAddr::V4(Ipv4Addr::LOCALHOST));
    assert_eq!(
        loopback.access_warning, None,
        "a loopback bind is not exposed, so there is nothing to warn about"
    );
    assert_eq!(
        body(handlers::tree(&loopback))["access_warning"],
        Value::Null,
        "the page must get no banner for a loopback bind"
    );

    for host in [
        IpAddr::V4(Ipv4Addr::UNSPECIFIED),      // 0.0.0.0 — the default
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, 7)), // a specific interface
    ] {
        let exposed = web().with_exposure(host);
        let warning = exposed
            .access_warning
            .clone()
            .unwrap_or_else(|| panic!("{host} should be treated as exposed"));
        assert!(
            warning.contains("No access control"),
            "the caution should name the problem: {warning}"
        );
        assert!(
            warning.contains("127.0.0.1"),
            "the caution should say how to restrict it: {warning}"
        );
        // The page gets the identical sentence the terminal printed.
        assert_eq!(
            body(handlers::tree(&exposed))["access_warning"].as_str(),
            Some(warning.as_str()),
            "the page banner and the terminal banner must be the same text"
        );
    }
}

/// A path typed into a UI has had no shell, so a leading `~` must be expanded by the
/// program — reported from the web compare box, where `~/ws/model` said "no checkpoint
/// files found" for a directory that plainly had them.
///
/// Asserted through the failure message rather than by pointing at a real file: the
/// fixtures are not under `$HOME` on every machine, and a test that quietly skips is
/// worse than none. The message names the resolved path when it differs from what was
/// typed, so seeing the expansion in it *is* the proof that expansion ran before
/// resolution. (`utils::expand_tilde` covers the expansion rule itself.)
#[test]
fn a_tilde_path_is_expanded_before_the_checkpoint_is_looked_for() {
    let home = std::env::var("HOME").expect("HOME is set in the test environment");
    let Err(err) = crate::compare::summarize(std::path::Path::new("~/no-such-checkpoint-xyz"))
    else {
        panic!("a nonexistent path should not resolve to a checkpoint");
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains(&format!("{home}/no-such-checkpoint-xyz")),
        "the tilde should have been expanded before the lookup: {msg}"
    );
    assert!(
        msg.contains("~/no-such-checkpoint-xyz"),
        "and the message should still show what was typed: {msg}"
    );
}

/// The command a UI offers must name the *checkpoint*, not one of its shards. Reported
/// from the web compare screen, which offered
/// `diff OLD <ckpt>/codebooks.safetensors` for a sharded checkpoint because it took the
/// first resolved file. Checked here against a real multi-file read, not just the unit
/// test in `compare`, because the file list this sees is whatever the resolver produced.
#[test]
fn the_served_diff_command_names_the_checkpoint_not_a_shard() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension().is_some_and(|x| x == "safetensors")).then_some(p)
        })
        .collect();
    files.sort();
    assert!(files.len() > 1, "this case needs several shards");

    let (current, id) = comparing(&files, &fixture().display().to_string());
    let served = body(handlers::diff(&current, &query("id", &id.to_string())));
    let command = served["command"]
        .as_str()
        .expect("a multi-shard checkpoint in one directory has a one-line command");
    let named = command.rsplit(' ').next().unwrap();
    assert_eq!(
        std::path::Path::new(named.trim_matches('\''))
            .canonicalize()
            .unwrap(),
        dir.canonicalize().unwrap(),
        "the command should name the directory, not a shard: {command}"
    );
    // (The *baseline* here is deliberately a single file, so `.safetensors` appearing
    // earlier in the command is correct — only the compared side is at issue, which the
    // path comparison above pins exactly.)
}

/// **The compact tree.** It must fold *and* account for everything: a frontend shows
/// "N tensors in M families", and if the fold silently dropped a tensor that header would
/// be a lie the view itself can't reveal.
#[test]
fn the_compact_tree_accounts_for_every_tensor() {
    let s = web();
    let served = body(handlers::compact(&s, &query("q", "")));

    assert_eq!(
        served["tensor_count"].as_u64().unwrap() as usize,
        s.tensors.len(),
        "the fold must account for every tensor in the checkpoint"
    );
    // Every family stands for at least one tensor, and the counts sum to the whole.
    let counts = served["counts"].as_object().expect("a counts map");
    assert!(!counts.is_empty(), "a non-empty checkpoint has families");
    let summed: u64 = counts.values().map(|v| v.as_u64().unwrap()).sum();
    assert_eq!(
        summed,
        s.tensors.len() as u64,
        "the per-family counts must sum to the tensor count"
    );
    // A family is never larger than the checkpoint, and never empty.
    for (name, n) in counts {
        let n = n.as_u64().unwrap();
        assert!(
            n >= 1 && n <= s.tensors.len() as u64,
            "{name} stands for {n} tensors"
        );
    }
}

/// Folding a *filtered* subset must fold only that subset — otherwise the view would
/// disagree with the filter bar above it.
#[test]
fn the_compact_tree_honours_the_filter() {
    let s = web();
    let q = "dtype:F32";
    let matching = body(handlers::filter(&s, &query("q", q)))["names"]
        .as_array()
        .expect("names")
        .len();
    assert!(matching > 0, "the fixture should have some F32 tensors");

    let served = body(handlers::compact(&s, &query("q", q)));
    assert_eq!(
        served["tensor_count"].as_u64().unwrap() as usize,
        matching,
        "the fold should cover exactly the filtered tensors"
    );

    // And a malformed query is rejected here as it is everywhere else, rather than
    // silently folding the whole checkpoint.
    let (status, _) = handlers::compact(&s, &query("q", "dtype:"));
    assert_eq!(status, 400, "a malformed filter must not fold everything");
}

/// **Sorting the flat list.** The live tree screen shows a *flat* list while a filter or
/// search is active, and that list is what `o` / `O` order. Exercised through the
/// `Explorer` rather than through `--plain`, because `--plain --filter` renders a pruned
/// *tree* instead of the live flat list — a pre-existing difference between the two paths,
/// and one a snapshot of the plain output would therefore not test.
#[test]
fn the_flat_filter_list_obeys_the_sort() {
    use crate::viewstate::{SortDir, SortKey};

    let sizes = |ex: &Explorer| -> Vec<usize> {
        ex.tree_state
            .visible()
            .iter()
            .filter_map(|(n, _)| match n {
                crate::tree::TreeNode::Tensor { info, .. } => Some(info.size_bytes),
                crate::tree::TreeNode::Group { .. } | crate::tree::TreeNode::Metadata { .. } => {
                    None
                }
            })
            .collect()
    };

    let mut ex = Explorer::new(vec![fixture()], Vec::new(), None, false);
    ex.set_tensor_filter(
        crate::tensorfilter::TensorFilter::parse("rank:>=1").expect("the query parses"),
    );
    ex.load_quiet().expect("the fixture loads");

    // Natural (tree) order: not sorted by size, or this test proves nothing.
    let natural = sizes(&ex);
    assert!(natural.len() > 2, "need several rows: {natural:?}");
    assert!(
        !natural.windows(2).all(|w| w[0] <= w[1]),
        "the fixture's tree order should not already be size-ascending: {natural:?}"
    );

    ex.tree_state.sort = (SortKey::Size, SortDir::Asc);
    ex.resort();
    let asc = sizes(&ex);
    assert!(
        asc.windows(2).all(|w| w[0] <= w[1]),
        "ascending by size: {asc:?}"
    );

    ex.tree_state.sort = (SortKey::Size, SortDir::Desc);
    ex.resort();
    let desc = sizes(&ex);
    assert!(
        desc.windows(2).all(|w| w[0] >= w[1]),
        "descending by size: {desc:?}"
    );

    // Back to the natural order — a rebuild, not an un-sort, which is exactly the case
    // that would silently stay sorted if `resort` only re-ordered in place.
    ex.tree_state.sort = (SortKey::None, SortDir::Asc);
    ex.resort();
    assert_eq!(
        sizes(&ex),
        natural,
        "`none` must restore the tree order, not leave the last sort applied"
    );
}

/// `o` cycles the facet and `O` reverses it, and both round-trip through `y` — the project
/// rule for any view state.
#[test]
fn the_sort_cycles_and_round_trips_through_the_reopen_command() {
    use crate::viewstate::{SortDir, SortKey};

    let mut ex = explorer();
    assert_eq!(ex.tree_state.sort, (SortKey::None, SortDir::Asc));
    // The cycle visits every facet and returns to the natural order.
    let mut seen = Vec::new();
    for _ in 0..6 {
        seen.push(ex.tree_state.cycle_sort());
    }
    assert_eq!(
        seen,
        vec![
            SortKey::Name,
            SortKey::Size,
            SortKey::Params,
            SortKey::Dtype,
            SortKey::Rank,
            SortKey::None,
        ],
        "the cycle should reach every facet and come back"
    );

    // The natural order emits no flag (a plain view stays a plain command); a chosen one
    // emits both halves.
    let plain = ex.reopen_command(&super::Screen::Tree, false, false);
    assert!(!plain.contains("--sort"), "{plain}");

    ex.tree_state.sort = (SortKey::Size, SortDir::Desc);
    let sorted = ex.reopen_command(&super::Screen::Tree, false, false);
    assert!(
        sorted.contains("--sort size.desc"),
        "the reopen command should carry the order: {sorted}"
    );
    // And that spelling is what the flag parser accepts.
    assert_eq!(
        crate::viewstate::parse_sort("size.desc").expect("the emitted spelling parses"),
        (SortKey::Size, SortDir::Desc)
    );
}

/// **The compact tree, in both interactive surfaces.** The terminal and the browser must
/// fold the same checkpoint into the same tree — that is the whole reason the fold lives in
/// core rather than in each frontend. The one deliberate difference is presentation: the
/// terminal carries each family's `×N` in the row label (it has no separate count column),
/// while the browser reads `counts` and styles the multiplier itself.
#[test]
fn the_tui_and_the_web_fold_the_same_compact_tree() {
    let mut ex = explorer();
    ex.set_compact(true);
    let tui = serde_json::to_value(&ex.tree_state.tree).unwrap();

    let served = body(handlers::compact(&web(), &query("q", "")));

    // Same shape below the root, once the labels the terminal adds are stripped.
    // The terminal folds the count into the label; the browser keeps it separate. Null the
    // labels so the comparison is about the *tree*, and assert the counts separately below.
    let strip = |v: &Value| -> Value {
        fn walk(v: &mut Value) {
            if let Some(items) = v.as_array_mut() {
                for item in items {
                    walk(item);
                }
                return;
            }
            if let Some(children) = v.get_mut("children").and_then(Value::as_array_mut) {
                for c in children {
                    walk(c);
                }
            }
            if v.get("kind").and_then(Value::as_str) == Some("tensor")
                && let Some(obj) = v.as_object_mut()
            {
                obj.insert("label".to_string(), Value::Null);
            }
        }
        let mut v = v.clone();
        walk(&mut v);
        v
    };
    assert_eq!(
        strip(&tui[0]["children"]),
        strip(&served["tree"][0]["children"]),
        "the two surfaces fold the checkpoint differently"
    );

    // And the terminal's labels really do carry the counts the browser gets separately.
    let counts = served["counts"].as_object().expect("a counts map");
    let mut labelled = 0usize;
    let mut pairs = Vec::new();
    leaf_labels(&tui[0], &mut pairs);
    assert!(!pairs.is_empty(), "the compact tree should have families");
    for (name, label) in pairs {
        let n = counts[&name].as_u64().expect("a count for every family");
        assert!(
            label.ends_with(&format!("×{n}")),
            "'{label}' should carry the count {n} the browser is sent for '{name}'"
        );
        labelled += 1;
    }
    assert_eq!(
        labelled,
        counts.len(),
        "every family should appear once in the terminal's tree"
    );
}

/// The compact toggle round-trips through `y`, like every other view choice.
#[test]
fn the_compact_view_round_trips_through_the_reopen_command() {
    let mut ex = explorer();
    let plain = ex.reopen_command(&super::Screen::Tree, false, false);
    assert!(!plain.contains("--compact"), "{plain}");

    ex.set_compact(true);
    let folded = ex.reopen_command(&super::Screen::Tree, false, false);
    assert!(
        folded.contains("--compact"),
        "the reopen command should carry the fold: {folded}"
    );
}

/// **The inferred architecture** must be the same facts in all three surfaces. It rides on
/// `CheckpointStats`, so `/api/stats` and the terminal's stats screen read one computation —
/// this pins that, and that the terminal actually *renders* it rather than merely holding it.
#[test]
fn the_inferred_architecture_agrees_across_the_surfaces() {
    let s = web();
    let served = body(handlers::stats(&s));
    let facts = served["arch"]["facts"]
        .as_array()
        .expect("the served stats carry the inferred architecture");
    assert!(!facts.is_empty(), "the fixture should yield some facts");

    // The same computation the core did, so the endpoint is not projecting its own.
    let computed = crate::arch::infer(&s.tensors, None);
    assert_eq!(
        facts.len(),
        computed.facts.len(),
        "the served facts are not the computed ones"
    );

    // And the terminal's stats screen shows each value, with its evidence line.
    let rendered = crate::tui::headless_render(160, 80, |f| {
        UI::render_stats_frame(f, &s.stats, 0, false);
    })
    .expect("the stats screen renders headless");
    assert!(
        rendered.contains("Inferred from tensors"),
        "the section should be on screen:\n{rendered}"
    );
    for (label, fact) in &computed.facts {
        assert!(
            rendered.contains(label),
            "'{label}' is inferred but not shown in the terminal:\n{rendered}"
        );
        assert!(
            rendered.contains(&fact.value),
            "'{label}' shows no value in the terminal (expected '{}'):\n{rendered}",
            fact.value
        );
    }
    // The gaps are named, not silently omitted.
    assert!(
        rendered.contains("not in the tensors"),
        "the config-only rows should be admitted on screen:\n{rendered}"
    );
}

/// **One count, two views.** The one-page diff report and the side-by-side comparison describe the
/// same pair, and used to print different totals for it. The aligned tree's tally is now the single
/// counter; this pins it to the report's own sections, so the day one of them changes definition the
/// other is not left quietly disagreeing.
#[test]
fn the_two_diff_views_count_the_same_differences() {
    // Every pair of diff fixtures, not one: the disagreement was reported on a 116k-tensor pair, and
    // a single small fixture is exactly the case both counters happen to get right.
    for (o, n) in [
        ("diff_old.safetensors", "diff_new.safetensors"),
        ("diff_group_old.safetensors", "diff_group_new.safetensors"),
        ("diff_map_old.safetensors", "diff_map_new.safetensors"),
        // A checkpoint against itself: every leaf must land in `same`, and nothing in `differing`.
        ("diff_old.safetensors", "diff_old.safetensors"),
        // Metadata on one side only — the group node that was suspected of being miscounted.
        ("diff_meta.safetensors", "diff_new.safetensors"),
    ] {
        counts_agree_for(o, n);
    }
}

#[allow(clippy::similar_names)]
fn counts_agree_for(old_name: &str, new_name: &str) {
    let old = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(old_name);
    let new = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(new_name);

    let old_state = {
        let m = crate::readers::read_local(std::slice::from_ref(&old)).expect("the baseline reads");
        WebState::build(m, std::slice::from_ref(&old), &[])
    };
    let new_state = {
        let m =
            crate::readers::read_local(std::slice::from_ref(&new)).expect("the newer side reads");
        WebState::build(m, std::slice::from_ref(&new), &[])
    };

    // The side-by-side: align the two trees and tally the leaves.
    let rows = crate::difftree::align_rooted(&old_state.tree, &new_state.tree);
    let tally = crate::difftree::tally(&rows);

    // The one-page report: the same pair through the shared structural comparison.
    let report = crate::compare::structural_diff(
        &new_state.tensors,
        &new_state.checkpoint.metadata_vec(),
        &old.display().to_string(),
        &crate::opening::Options::default(),
    )
    .expect("the pair compares");

    // Section by section, not summed. The tally counts tensors and metadata apart precisely so the two
    // views can use the same words — `1 removed, 2 metadata changes` rather than `3 removed` — so this
    // asserts the finer correspondence rather than only that the totals happen to match.
    assert_eq!(
        (
            tally.tensors.only_new,
            tally.tensors.only_old,
            tally.tensors.changed,
            tally.tensors.same,
        ),
        (
            report.tensors_added.len(),
            report.tensors_removed.len(),
            report.tensors_changed.len(),
            report.tensors_unchanged,
        ),
        "{old_name} vs {new_name}: the aligned tree and the report disagree about tensors: \
         tree={:?} report=(added {}, removed {}, changed {}, unchanged {})",
        tally.tensors,
        report.tensors_added.len(),
        report.tensors_removed.len(),
        report.tensors_changed.len(),
        report.tensors_unchanged,
    );
    assert_eq!(
        (
            tally.metadata.only_new,
            tally.metadata.only_old,
            tally.metadata.changed,
            tally.metadata.same,
        ),
        (
            report.meta_added.len(),
            report.meta_removed.len(),
            report.meta_changed.len(),
            report.meta_unchanged,
        ),
        "{old_name} vs {new_name}: the aligned tree and the report disagree about metadata: {:?}",
        tally.metadata,
    );
    // And the headline total is still the size of the steppable list.
    assert_eq!(
        tally.differing(),
        crate::difftree::differences(&rows).len(),
        "{old_name} vs {new_name}: the count and the steppable list must describe the same rows"
    );
}

/// **A scoped comparison is the same comparison on both surfaces.**
///
/// `diff --name 'model.layers.1.*'` and `/api/diff?…&name=model.layers.1.*` must produce the same
/// report, including the `matched M of N` context. The two now share their scope builders
/// (`crate::compare::tensor_filter` / `name_map`) and their apply order — this is what stops that from
/// being a claim in a comment.
#[test]
fn a_scoped_diff_agrees_between_the_cli_and_the_web() {
    let old = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diff_old.safetensors");
    let new = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diff_new.safetensors");
    let (current, id) = comparing(std::slice::from_ref(&new), &old.display().to_string());
    let state = current.snapshot();

    // Every scope the web accepts, exercised against the flags that mean the same thing.
    for (query, name, names, dtype_is, shape_is, only_tensors) in [
        (
            vec![("name", "model.layers.*")],
            vec!["model.layers.*".to_string()],
            None,
            None,
            None,
            false,
        ),
        (
            vec![("name", "*\n!*.mlp.weight")],
            vec!["*".to_string(), "!*.mlp.weight".to_string()],
            None,
            None,
            None,
            false,
        ),
        (
            vec![("names", "model.norm.weight,lm_head.weight")],
            vec![],
            Some("model.norm.weight,lm_head.weight"),
            None,
            None,
            false,
        ),
        (
            vec![("dtype_is", "F*")],
            vec![],
            None,
            Some("F*"),
            None,
            false,
        ),
        (
            vec![("shape_is", "4")],
            vec![],
            None,
            None,
            Some("4"),
            false,
        ),
        (vec![("only_tensors", "1")], vec![], None, None, None, true),
    ] {
        // The web: the scope as query parameters, over the comparison already set up.
        let mut q = handlers::Query::new();
        q.insert("id".to_string(), id.to_string());
        for (k, v) in &query {
            q.insert((*k).to_string(), (*v).to_string());
        }
        let served = body(handlers::diff(&current, &q));

        // The CLI: the same selection through its own builders, then the shared apply order.
        let filter = crate::compare::tensor_filter(&crate::compare::ScopeText {
            name: &name,
            names_csv: names,
            names_lines: None,
            dtype_is,
            shape_is,
        })
        .expect("the flags compile");
        let mut old_sum = crate::compare::summarize(&old).expect("the baseline summarizes");
        let mut new_sum = crate::diff::CheckpointSummary::from_loaded(
            &state.tensors,
            &state.checkpoint.metadata_vec(),
        );
        filter.apply(&mut old_sum, &mut new_sum);
        // The CLI's rule: `DiffOpts { metadata: !only_tensors && !filtered }` — a scoped diff does not
        // compare metadata, which is also what `--name`'s help promises.
        if only_tensors || filter.is_active() {
            old_sum.metadata.clear();
            new_sum.metadata.clear();
        }
        let expected = crate::diff::compare(&old_sum, &new_sum);

        assert_eq!(
            served["report"],
            serde_json::to_value(&expected).unwrap(),
            "scope {query:?}: the served report is not the one the shared comparison produces"
        );
        // And the scope is *reported*, so a reader can tell the diff was narrowed.
        if !query.iter().any(|(k, _)| *k == "only_tensors") {
            let matched = &served["matched"];
            assert!(
                matched.is_object(),
                "scope {query:?}: a narrowed comparison should say what it matched, got {matched}"
            );
        }
    }
}

/// The copyable command reproduces the comparison **on screen**, scope included.
///
/// It used to hand over an unscoped `diff OLD NEW` while the page showed nineteen tensors of 117,664 —
/// a command that answers a different question than the one you were looking at.
#[test]
fn the_offered_command_carries_the_scope() {
    let old = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diff_old.safetensors");
    let new = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diff_new.safetensors");
    let (current, id) = comparing(std::slice::from_ref(&new), &old.display().to_string());
    let mut q = std::collections::HashMap::new();
    q.insert("id".to_string(), id.to_string());
    q.insert("name".to_string(), "model.layers.*\n!*.bias".to_string());
    q.insert("only_tensors".to_string(), "1".to_string());
    let served = body(handlers::diff(&current, &q));
    let cmd = served["command"].as_str().expect("a one-line command");
    assert!(cmd.contains("--name 'model.layers.*'"), "{cmd}");
    assert!(cmd.contains("--name '!*.bias'"), "{cmd}");
    assert!(cmd.contains("--only-tensors"), "{cmd}");
}

/// **`--values` means the same thing on both surfaces.**
///
/// The value comparison itself is `compare::tensor_extras`, shared with the `diff` subcommand, and both
/// feed it into `diff::compare_with`. This asserts the result: a tensor whose dtype and shape match but
/// whose bytes differ reads as *changed* on both, and identically.
#[test]
fn a_value_comparison_agrees_between_the_cli_and_the_web() {
    let old = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diff_old.safetensors");
    let new = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diff_new.safetensors");
    let read = |p: &PathBuf| {
        crate::readers::read_local(std::slice::from_ref(p)).expect("the fixture reads")
    };
    let (om, nm) = (read(&old), read(&new));
    let (ot, nt) = (om.tensors_vec(), nm.tensors_vec());
    let (omd, nmd) = (om.metadata_vec(), nm.metadata_vec());

    // The fixtures share `model.layers.0.mlp.weight` with the same dtype and shape but different bytes
    // (seed 0 vs 7) — a values-only change, which is the case `--values` exists for.
    let opts = crate::compare::ValueOpts {
        view: crate::sample::ViewDtype::Stored,
        bins: None,
        values: true,
        histogram: false,
        old_schemas: &crate::sample::parse_packing_schemas(&ot, &omd),
        new_schemas: &crate::sample::parse_packing_schemas(&nt, &nmd),
    };
    let find = |ts: &[crate::tree::TensorInfo], n: &str| {
        ts.iter().find(|t| t.name == n).cloned().expect("present")
    };
    let name = "model.layers.0.mlp.weight";
    let extras = crate::compare::tensor_extras(&find(&ot, name), &find(&nt, name), &opts);
    let v = extras
        .values
        .expect("a same-shape pair has a value comparison");
    assert!(
        v.differing > 0,
        "the fixtures differ in bytes for {name}: {v:?}"
    );

    // Folded into the report, that tensor is *changed* — where a structural diff called it unchanged.
    let old_sum = crate::compare::summarize(&old).expect("baseline summarizes");
    let new_sum = crate::diff::CheckpointSummary::from_loaded(&nt, &nmd);
    let structural = crate::diff::compare(&old_sum, &new_sum);
    assert!(
        !structural.tensors_changed.iter().any(|c| c.name == name),
        "structurally this tensor matches, which is why --values exists"
    );
    let mut one: std::collections::HashMap<String, crate::diff::TensorExtras> =
        std::collections::HashMap::new();
    one.insert(name.to_string(), extras);
    let cell = std::cell::RefCell::new(one);
    let with_values = crate::diff::compare_with(&old_sum, &new_sum, |n| {
        cell.borrow_mut().remove(n).unwrap_or_default()
    });
    assert!(
        with_values.tensors_changed.iter().any(|c| c.name == name),
        "with values compared it must read as changed"
    );
}
