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
    let model = crate::readers::read_local(&files).expect("the fixture reads");
    let state = WebState::build(model, &files, &[]);
    let served = body(handlers::diff(
        &state,
        &query("against", &old.display().to_string()),
    ));

    // The CLI: the same pair through the shared comparison the `diff` subcommand uses.
    let expected =
        crate::compare::structural_diff(&state.tensors, &state.checkpoint.metadata_vec(), &old)
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
    let s = web();
    let served = body(handlers::diff(
        &s,
        &query("against", &fixture().display().to_string()),
    ));
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
#[test]
fn the_diff_endpoint_rejects_a_path_that_is_not_a_checkpoint() {
    let s = web();
    for bad in ["", "/nonexistent/checkpoint"] {
        let (status, bytes) = handlers::diff(&s, &query("against", bad));
        assert_eq!(status, 400, "'{bad}' should be a client error");
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

    let model = crate::readers::read_local(&files).expect("the fixtures read");
    let state = WebState::build(model, &files, &[]);
    let served = body(handlers::diff(
        &state,
        &query("against", &fixture().display().to_string()),
    ));
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
