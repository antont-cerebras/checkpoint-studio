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
