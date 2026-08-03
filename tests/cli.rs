//! End-to-end "cram"-style tests: run the real binary in `--plain` mode against
//! a generated fixture checkpoint and snapshot its rendered screen. Because
//! almost every screen is reproducible from CLI flags, one case == one command
//! line, and `--plain` makes the output stable plain text.
//!
//! Golden snapshots live under `tests/snapshots/`. After an intentional change,
//! review and accept them with:
//!
//! ```text
//! cargo insta review          # or: INSTA_UPDATE=always cargo test --test cli
//! ```
//!
//! Fixtures: the safetensors one is generated fresh each run (pure Rust,
//! deterministic, git-ignored); the HDF5 one is committed (`tests/fixtures/tiny.hdf5`,
//! regenerated with `cargo run --example gen_hdf5_fixture --features hdf5`),
//! because hdf5-metno isn't a dev-dependency. The HDF5 cases are gated on the
//! `hdf5` feature.

// An unwrap in a test IS the assertion: the panic is the failure report. (Product code
// denies these — see `[workspace.lints.clippy]` in Cargo.toml.)
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::sync::Once;

use safetensors::tensor::{Dtype, TensorView};

const FIXTURE: &str = "tests/fixtures/tiny.safetensors";

/// Build a tiny safetensors checkpoint mirroring a Qwen3-coder-ish layout: a few
/// dtypes and shapes (1D/2D/3D) under dotted names so the tree has structure.
/// Values don't matter for the tree / detail screens (no statistics are scanned
/// in `--plain`), so each payload is just the right number of bytes.
fn write_fixture(path: &str) {
    // (name, dtype, shape) — payloads are a byte ramp of the right size.
    let specs: &[(&str, Dtype, Vec<usize>)] = &[
        ("lm_head.weight", Dtype::I32, vec![2, 4]),
        ("model.embed_tokens.weight", Dtype::F16, vec![6, 4]),
        (
            "model.layers.0.self_attn.q_proj.weight",
            Dtype::BF16,
            vec![4, 4],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            Dtype::U16,
            vec![3, 4, 5],
        ),
        ("model.layers.0.input_layernorm.weight", Dtype::F32, vec![4]),
        ("model.norm.weight", Dtype::F32, vec![4]),
        ("model.scale.u8", Dtype::U8, vec![8]),
    ];

    // Own the byte buffers so the borrowing `TensorView`s stay valid until write.
    let buffers: Vec<Vec<u8>> = specs
        .iter()
        .map(|(_, dt, shape)| {
            let bytes = shape.iter().product::<usize>() * dtype_size(*dt);
            (0..bytes).map(|i| (i % 251) as u8).collect()
        })
        .collect();

    let data: HashMap<String, TensorView> = specs
        .iter()
        .zip(&buffers)
        .map(|((name, dt, shape), buf)| {
            (
                name.to_string(),
                TensorView::new(*dt, shape.clone(), buf).expect("valid tensor view"),
            )
        })
        .collect();

    let metadata = Some(HashMap::from([("format".to_string(), "pt".to_string())]));

    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }
    safetensors::serialize_to_file(&data, &metadata, Path::new(path)).expect("write fixture");
}

// In its own directory (not `tests/fixtures/`) so it doesn't show up in the file
// browser's listing of the main fixture directory (`cli__files_view`).
const FIXTURE_MOE: &str = "tests/fixtures_moe/tiny_moe.safetensors";

/// A small `MoE` checkpoint with 8 transformer layers — each with attention
/// (q/k/v/o), two experts (down/gate/up), and two norms — sized so the per-layer
/// graphs actually show shape (attention grows with depth, so the size / params
/// sparklines ramp up; experts dominate the composition chart).
fn write_moe_fixture(path: &str) {
    let mut specs: Vec<(String, Dtype, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), Dtype::F16, vec![32, 8]),
        ("model.norm.weight".into(), Dtype::F32, vec![8]),
        ("lm_head.weight".into(), Dtype::F16, vec![32, 8]),
    ];
    for l in 0..8usize {
        // Attention grows slightly with depth so the size sparkline isn't flat.
        let a = 4 + l;
        for proj in ["q_proj", "k_proj", "v_proj"] {
            specs.push((
                format!("model.layers.{l}.self_attn.{proj}.weight"),
                Dtype::BF16,
                vec![a, 8],
            ));
        }
        specs.push((
            format!("model.layers.{l}.self_attn.o_proj.weight"),
            Dtype::BF16,
            vec![8, a],
        ));
        for e in 0..2 {
            for proj in ["down_proj", "gate_proj", "up_proj"] {
                specs.push((
                    format!("model.layers.{l}.mlp.experts.{e}.{proj}.weight"),
                    Dtype::F16,
                    vec![8, 6],
                ));
            }
        }
        specs.push((
            format!("model.layers.{l}.input_layernorm.weight"),
            Dtype::F32,
            vec![8],
        ));
        specs.push((
            format!("model.layers.{l}.post_attention_layernorm.weight"),
            Dtype::F32,
            vec![8],
        ));
    }

    let buffers: Vec<Vec<u8>> = specs
        .iter()
        .map(|(_, dt, shape)| {
            let bytes = shape.iter().product::<usize>() * dtype_size(*dt);
            (0..bytes).map(|i| (i % 251) as u8).collect()
        })
        .collect();
    let data: HashMap<String, TensorView> = specs
        .iter()
        .zip(&buffers)
        .map(|((name, dt, shape), buf)| {
            (
                name.clone(),
                TensorView::new(*dt, shape.clone(), buf).expect("valid tensor view"),
            )
        })
        .collect();
    let metadata = Some(HashMap::from([("format".to_string(), "pt".to_string())]));
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }
    safetensors::serialize_to_file(&data, &metadata, Path::new(path)).expect("write MoE fixture");
}

fn dtype_size(dt: Dtype) -> usize {
    match dt {
        Dtype::U8 | Dtype::I8 | Dtype::BOOL => 1,
        Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        Dtype::F8_E5M2 | Dtype::F8_E4M3 | _ => 4,
    }
}

/// Generate the fixture once, even with tests running in parallel.
fn ensure_fixture() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| write_fixture(FIXTURE));
}

/// Generate the multi-layer `MoE` fixture once.
fn ensure_moe_fixture() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| write_moe_fixture(FIXTURE_MOE));
}

/// Run the binary with exactly `args` and return its stdout.
/// A throwaway config directory for every spawned binary, so the test suite never reads or
/// writes the real `~/.config/checkpoint-studio/` — the recents list is persisted there now, and
/// a test run was appending its fixtures to the user's own list.
///
/// `XDG_CONFIG_HOME` is what `CliConfig::path` prefers, so this needs no production hook.
fn scratch_config() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("cs_test_config_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn run_bin(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args(args)
        .output()
        .expect("run checkpoint-studio");
    assert!(
        out.status.success(),
        "non-zero exit; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Run the binary and return `(stdout, exit code)` without asserting success —
/// `check` / `diff` use a nonzero exit to signal findings, not failure.
fn run_bin_status(args: &[&str]) -> (String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args(args)
        .output()
        .expect("run checkpoint-studio");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

/// Run the binary in `--plain` mode against `fixture` and return its screen text.
fn run_plain(fixture: &str, extra_args: &[&str]) -> String {
    let mut args = vec![fixture];
    args.extend_from_slice(extra_args);
    args.push("--plain");
    run_bin(&args)
}

/// Verify the `y` round-trip for a screen: render it directly, take the CLI
/// command `y` would copy to reopen it (`--emit-command`), re-render from that,
/// and require the two screens to be identical. Catches any state a screen shows
/// but its reopen command fails to express.
fn assert_y_roundtrip(fixture: &str, extra_args: &[&str]) {
    let direct = run_plain(fixture, extra_args);

    let mut emit = vec![fixture];
    emit.extend_from_slice(extra_args);
    emit.push("--emit-command");
    let command = run_bin(&emit);

    // The command is `checkpoint-studio <path> <flags…>`; drop the program name
    // and render what's left (the fixture's names/paths are shell-safe, so the
    // tokens never need de-quoting).
    let mut reopen: Vec<&str> = command.split_whitespace().skip(1).collect();
    reopen.push("--plain");
    let reopened = run_bin(&reopen);

    // The two renders are independent scans, so a statistics / histogram duration
    // (`(2ms)`) differs run to run — normalize it before comparing.
    assert_eq!(
        strip_scan_time(&direct),
        strip_scan_time(&reopened),
        "y round-trip diverged\n  opened with: {extra_args:?}\n  reopened by: {}",
        command.trim(),
    );
}

/// Replace the scan-duration suffix (`(2ms)`, `(1.0s)`) with a stable token, so
/// the round-trip compares everything except the inherently-varying timing.
fn strip_scan_time(s: &str) -> String {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"\(\d+(?:\.\d+)?m?s\)").unwrap())
        .replace_all(s, "(<time>)")
        .into_owned()
}

/// The generated safetensors fixture, in `--plain`.
fn plain(extra_args: &[&str]) -> String {
    ensure_fixture();
    run_plain(FIXTURE, extra_args)
}

/// Normalize the fixture's path (shown verbatim, absolute, or left-elided with
/// `…` depending on the screen) to a stable token, so snapshots don't depend on
/// the checkout location.
fn settings() -> insta::Settings {
    let mut s = insta::Settings::clone_current();
    // Match the fixture path/basename but not a surrounding quote, so JSON
    // exports (`"…tiny.safetensors"`) keep their delimiters in the snapshot.
    // The MoE fixture is matched first (its name isn't caught by the `tiny.` rule).
    s.add_filter(r#"[^\s"]*tiny_moe\.safetensors"#, "[FIXTURE]");
    s.add_filter(r#"[^\s"]*tiny\.(?:safetensors|hdf5)"#, "[FIXTURE]");
    // The statistics / histogram scan duration (e.g. `(2ms)`, `(1.0s)`) is timing.
    s.add_filter(r"\(\d+(?:\.\d+)?m?s\)", "(<time>)");
    // The access badge (`read-only` / `editable`) is right-aligned, so when the
    // status line to its left carries the (now `[FIXTURE]`-normalized) checkpoint
    // path, the run of spaces between them reflects that path's *real* length —
    // which varies with the checkout location (local vs CI). Collapse it so the
    // snapshot is stable.
    s.add_filter(
        r"(?m)\[FIXTURE\] {2,}(read-only|editable)$",
        "[FIXTURE]  $1",
    );
    s
}

#[test]
fn plain_tree() {
    settings().bind(|| insta::assert_snapshot!(plain(&[])));
}

/// The rich `--filter` applied to the interactive tree: the title shows the query
/// + match count and only matching tensors remain (flat list).
#[test]
fn plain_tree_filtered() {
    settings().bind(|| insta::assert_snapshot!(plain(&["--filter", "name:mlp"])));
}

/// `--print-tensors --filter` — the print export narrowed by the rich filter.
#[test]
fn print_tensors_filter() {
    settings()
        .bind(|| insta::assert_snapshot!(export(&["--print-tensors", "--filter", "name:mlp"])));
}

/// The `s` view: the full-screen checkpoint-stats report.
#[test]
fn stats_popup() {
    settings().bind(|| insta::assert_snapshot!(plain(&["--stats"])));
}

/// The per-layer graphs on a multi-layer `MoE` checkpoint — so the sparkline shape
/// (attention ramps with depth) and the stacked composition bands are asserted,
/// not just the degenerate single-layer case of the main fixture.
#[test]
fn stats_graphs() {
    ensure_moe_fixture();
    settings().bind(|| insta::assert_snapshot!(run_plain(FIXTURE_MOE, &["--stats"])));
}

/// Run a one-shot `--print-*` export (no `--plain`) and capture stdout.
fn export(extra_args: &[&str]) -> String {
    ensure_fixture();
    let mut args = vec![FIXTURE];
    args.extend_from_slice(extra_args);
    run_bin(&args)
}

#[test]
fn print_tree_text() {
    settings().bind(|| insta::assert_snapshot!(export(&["--print-tree"])));
}

#[test]
fn print_tree_name_filter() {
    // Include glob: only the matching tensors (and their groups) survive.
    settings().bind(|| insta::assert_snapshot!(export(&["--print-tree", "--name", "*.mlp.*"])));
}

#[test]
fn print_tensors_name_exclude() {
    // Negated glob: everything except the pattern.
    settings().bind(|| insta::assert_snapshot!(export(&["--print-tensors", "--name", "!*.mlp.*"])));
}

/// `--print-view` dumps the tensor-tree screen's `ViewModel` as JSON — the
/// kernel's frontend-agnostic output contract, projected from the same live tree
/// state the TUI renders. Deterministic (row labels/depths only), so snapshotted.
#[test]
fn print_view_emits_viewmodel_json() {
    settings().bind(|| insta::assert_snapshot!(export(&["--print-view"])));
}

/// The `--name` filter scopes the `ViewModel` rows too (same path as the other
/// exports).
#[test]
fn print_view_name_filter() {
    settings().bind(|| insta::assert_snapshot!(export(&["--print-view", "--name", "*.mlp.*"])));
}

#[test]
fn check_healthy_fixture() {
    ensure_fixture();
    let (out, code) = run_bin_status(&["check", FIXTURE]);
    assert_eq!(code, 0, "healthy fixture should pass; got:\n{out}");
    settings().bind(|| insta::assert_snapshot!(out));
}

#[test]
fn check_json_healthy() {
    ensure_fixture();
    let (out, code) = run_bin_status(&["check", FIXTURE, "--format", "json"]);
    assert_eq!(code, 0);
    settings().bind(|| insta::assert_snapshot!(out));
}

#[test]
fn check_sarif_healthy() {
    ensure_fixture();
    let (out, code) = run_bin_status(&["check", FIXTURE, "--format", "sarif"]);
    assert_eq!(code, 0);
    let mut s = settings();
    // The crate version (0.x.y) is in the SARIF driver; normalize it so the
    // snapshot survives version bumps (the SARIF "2.1.0" is left as-is).
    s.add_filter(r#""version": "0\.\d+\.\d+""#, r#""version": "[VERSION]""#);
    s.bind(|| insta::assert_snapshot!(out));
}

#[test]
fn check_values_scans_data() {
    ensure_fixture();
    // --values runs the value scan (with the progress bar, a no-op when piped).
    let (out, code) = run_bin_status(&["check", FIXTURE, "--values"]);
    assert_eq!(code, 0, "got:\n{out}");
    assert!(out.contains("Value scan"));
    assert!(
        out.contains("no NaN"),
        "value scan should have run; got:\n{out}"
    );
    assert!(
        !out.contains("skipped"),
        "value scan should not be skipped; got:\n{out}"
    );
}

#[test]
fn check_detects_truncation() {
    ensure_fixture();
    // A copy with the last 8 data bytes lopped off — a classic interrupted
    // download. The byte-range check should fail the run (exit 1).
    let bytes = std::fs::read(FIXTURE).expect("read fixture");
    let dir = std::env::temp_dir().join("checkpoint_studio_check_trunc");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("model.safetensors");
    std::fs::write(&path, &bytes[..bytes.len() - 8]).expect("write truncated copy");

    let (out, code) = run_bin_status(&["check", path.to_str().unwrap()]);
    assert_eq!(code, 1, "truncated file should fail; got:\n{out}");
    assert!(
        out.contains("file truncated"),
        "expected a truncation finding; got:\n{out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn print_tree_json() {
    settings().bind(|| insta::assert_snapshot!(export(&["--print-tree", "--format", "json"])));
}

#[test]
fn print_tree_json_verbose() {
    settings()
        .bind(|| insta::assert_snapshot!(export(&["--print-tree", "--format", "json", "-v"])));
}

#[test]
fn print_tensors_text() {
    settings().bind(|| insta::assert_snapshot!(export(&["--print-tensors"])));
}

#[test]
fn print_tensors_text_verbose() {
    settings().bind(|| insta::assert_snapshot!(export(&["--print-tensors", "-v"])));
}

#[test]
fn print_tensors_json() {
    settings().bind(|| insta::assert_snapshot!(export(&["--print-tensors", "--format", "json"])));
}

/// `--print-model` dumps the whole central serializable model as JSON. Not
/// snapshotted (it carries machine-specific mtimes / absolute paths / block
/// sizes); instead assert the structure + key contents are present.
#[test]
fn print_model_emits_json() {
    ensure_fixture();
    let out = run_bin(&[FIXTURE, "--print-model"]);
    // The top-level model shape.
    assert!(out.contains("\"source\""), "has a source:\n{out}");
    assert!(out.contains("\"Local\""), "local source:\n{out}");
    assert!(out.contains("\"files\""), "has the fs walk:\n{out}");
    assert!(out.contains("\"shards\""), "has parsed headers:\n{out}");
    // The fixture's tensors made it into a shard header.
    assert!(out.contains("lm_head.weight"), "tensor present:\n{out}");
    assert!(out.contains("\"dtype\""), "tensor dtype present:\n{out}");
    // It's valid JSON (balanced enough to parse as a value via a trivial check:
    // starts with `{` and the fixture file name appears in `files`).
    assert!(out.trim_start().starts_with('{'), "json object:\n{out}");
    assert!(out.contains("tiny.safetensors"), "file entry:\n{out}");
}

#[test]
fn print_tensors_json_verbose() {
    settings()
        .bind(|| insta::assert_snapshot!(export(&["--print-tensors", "--format", "json", "-v"])));
}

#[test]
fn plain_detail_u16() {
    settings().bind(|| {
        insta::assert_snapshot!(plain(&["--tensor", "model.layers.0.mlp.down_proj.weight"]));
    });
}

#[test]
fn plain_detail_f16() {
    settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", "model.embed_tokens.weight"])));
}

#[test]
fn plain_values_u16() {
    settings().bind(|| {
        insta::assert_snapshot!(plain(&[
            "--tensor",
            "model.layers.0.mlp.down_proj.weight",
            "--values"
        ]));
    });
}

#[test]
fn plain_histogram_u16() {
    settings().bind(|| {
        insta::assert_snapshot!(plain(&[
            "--tensor",
            "model.layers.0.mlp.down_proj.weight",
            "--histogram"
        ]));
    });
}

#[test]
fn plain_tree_expanded() {
    settings().bind(|| insta::assert_snapshot!(plain(&["--tree-state", "expanded"])));
}

/// The file browser screen (`Tab` / `--files`).
#[test]
fn files_view() {
    settings().bind(|| insta::assert_snapshot!(plain(&["--files"])));
}

/// The safetensors byte-layout map (`--layout <file>`).
#[test]
fn layout_view() {
    settings().bind(|| insta::assert_snapshot!(plain(&["--layout", FIXTURE])));
}

/// The in-place rename editor (`R` / `--rename`).
#[test]
fn rename_view() {
    settings().bind(|| insta::assert_snapshot!(plain(&["--rename"])));
}

#[test]
fn y_roundtrips() {
    ensure_fixture();
    let t = "model.layers.0.mlp.down_proj.weight";
    for extra in [
        vec![],                             // tree (default expansion)
        vec!["--tree-state", "expanded"],   // E
        vec!["--tree-state", "collapsed"],  // C
        vec!["--tensor", t],                // detail
        vec!["--tensor", t, "--histogram"], // detail + histogram
        vec!["--tensor", t, "--values", "--slice", "1"],
        vec!["--tensor", t, "--values", "--overview", "--base", "hex"],
        vec!["--tensor", t, "--heatmap"],
        vec!["--health"],          // the health-check popup over the tree
        vec!["--health-findings"], // …with the per-finding detail expanded
        vec!["--stats"],           // the full-screen checkpoint-stats view
        vec!["--stats-shards"],    // …with the per-shard breakdown expanded
    ] {
        assert_y_roundtrip(FIXTURE, &extra);
    }
}

/// Run a failing `--plain` request: assert it exits non-zero (a snapshot can't
/// see the exit code) and return the command line + its stderr for snapshotting.
fn run_plain_err(extra_args: &[&str]) -> String {
    ensure_fixture();
    let mut args = vec![FIXTURE];
    args.extend_from_slice(extra_args);
    args.push("--plain");
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args(&args)
        .output()
        .expect("run checkpoint-studio");
    assert!(
        !out.status.success(),
        "expected non-zero exit for {extra_args:?}, got success"
    );
    format!(
        "$ checkpoint-studio {}\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// A request that can't be honored must exit non-zero with an explanation, not
/// silently fall back to an unrelated screen. `--plain` exercises the same
/// resolution path as the interactive `--exit` one-shot (both headless), so it
/// stands in for the `--exit` exit code (which needs a tty to reach). The
/// snapshot pins the exact wording — which names the specific problem rather
/// than a vague "invalid" — so any reword surfaces in `cargo insta review`.
#[test]
fn plain_request_errors() {
    let t = "model.layers.0.mlp.down_proj.weight";
    let report = [
        run_plain_err(&["--tensor", "no.such.tensor"]),
        run_plain_err(&["--metadata", "no.such.meta"]),
        run_plain_err(&["--tensor", t, "--shape", "abc"]),
        run_plain_err(&["--tensor", t, "--slice", "9999"]),
    ]
    .join("\n");
    settings().bind(|| insta::assert_snapshot!(report));
}

/// Opening an HDF5 file with a binary built *without* the `hdf5` feature must
/// fail loudly (non-zero exit + an explanation that names the rebuild flag),
/// rather than silently loading an empty checkpoint that reads "0 tensors". The
/// non-zero exit must hold in headless `--exit`/`--plain` modes too, so scripts
/// detect it. Only meaningful when the feature is off, so it's gated out of the
/// `hdf5` build.
#[cfg(not(feature = "hdf5"))]
#[test]
fn hdf5_without_feature_errors() {
    const H5: &str = "tests/fixtures/tiny.hdf5";
    for extra in [&[][..], &["--exit"][..], &["--plain"][..]] {
        let mut args = vec![H5];
        args.extend_from_slice(extra);
        let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
            .env("XDG_CONFIG_HOME", scratch_config())
            .args(&args)
            .output()
            .expect("run checkpoint-studio");
        assert!(
            !out.status.success(),
            "expected non-zero exit for {args:?}, got success"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("without HDF5 support") && stderr.contains("--features hdf5"),
            "expected an HDF5-support error naming the rebuild flag for {args:?}; stderr:\n{stderr}"
        );
    }
}

/// HDF5 fixture (`tests/fixtures/tiny.hdf5`, committed; regenerate with
/// `cargo run --example gen_hdf5_fixture --features hdf5`). Gated on the `hdf5`
/// feature so it only runs when the binary can read HDF5. Pins the fused-MoE
/// quantization-schema display (top-level + per-tensor + non-uniform), the
/// compression codec / `(uncompressed)` tags, and chunk reporting.
#[cfg(feature = "hdf5")]
mod hdf5 {
    use super::{run_plain, settings};
    use std::fmt::Write as _;

    const H5: &str = "tests/fixtures/tiny.hdf5";
    const MOE: &str = "model.layers.0.block_sparse_moe.experts";

    fn plain(extra_args: &[&str]) -> String {
        run_plain(H5, extra_args)
    }

    #[test]
    fn tree() {
        settings().bind(|| insta::assert_snapshot!(plain(&[])));
    }

    #[test]
    fn detail_down_proj_uniform_schema() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t])));
    }

    #[test]
    fn detail_gate_up_nonuniform_schema() {
        let t = format!("{MOE}.gate_up_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t])));
    }

    #[test]
    fn detail_per_tensor_schema() {
        settings().bind(|| {
            insta::assert_snapshot!(plain(&["--tensor", "model.layers.0.custom_proj.weight"]));
        });
    }

    #[test]
    fn detail_compressed_f16() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", "lm_head.weight"])));
    }

    // Synchronously-scanned screens: the histogram (intrinsic 0..7 span for the
    // unpacked codebook view), statistics, and the numeric / heatmap data views
    // in each layout. The scan time is filtered out by `settings`.

    #[test]
    fn detail_histogram() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--histogram"])));
    }

    #[test]
    fn detail_compute_stats() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--compute-stats"])));
    }

    #[test]
    fn values_edges() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--values"])));
    }

    #[test]
    fn values_overview() {
        let t = format!("{MOE}.down_proj.weight");
        settings()
            .bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--values", "--overview"])));
    }

    #[test]
    fn heatmap() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--heatmap"])));
    }

    // Main-screen keyboard shortcuts, reached via their flags: bulk expand /
    // collapse (E / C), search (/), and the context-sensitive legend (l) over
    // the tree, a detail, and a data view.

    #[test]
    fn tree_expanded() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--tree-state", "expanded"])));
    }

    #[test]
    fn tree_collapsed() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--tree-state", "collapsed"])));
    }

    #[test]
    fn tree_search() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--search", "down_proj"])));
    }

    #[test]
    fn legend_tree() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--legend"])));
    }

    #[test]
    fn legend_detail() {
        let t = format!("{MOE}.down_proj.weight");
        settings().bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--legend"])));
    }

    #[test]
    fn legend_values() {
        let t = format!("{MOE}.down_proj.weight");
        settings()
            .bind(|| insta::assert_snapshot!(plain(&["--tensor", &t, "--values", "--legend"])));
    }

    // The `y` round-trip meta-test: every state-bearing screen must reopen to
    // itself from the command `y` copies. Covers the bulk tree expansion, the
    // schema views, and the full matrix of data-view state (layout + position,
    // slice, zebra, base). (Search / legend are transient overlays you can't `y`
    // from, so they're cram-only above.)
    #[test]
    fn y_roundtrips() {
        let dp = format!("{MOE}.down_proj.weight");
        let cases: &[Vec<&str>] = &[
            vec![],                                     // tree (default expansion)
            vec!["--tree-state", "expanded"],           // E
            vec!["--tree-state", "collapsed"],          // C
            vec!["--tensor", &dp, "--tree"],            // tree with a tensor revealed
            vec!["--tensor", &dp],                      // unpacked detail
            vec!["--tensor", &dp, "--dtype", "stored"], // raw U16 over a schema
            vec!["--tensor", &dp, "--histogram"],
            vec!["--tensor", &dp, "--histogram", "--bins", "4"],
            vec!["--tensor", &dp, "--compute-stats"],
            vec!["--tensor", "model.layers.0.custom_proj.weight"], // per-tensor schema
            vec!["--tensor", &dp, "--values"],
            vec!["--tensor", &dp, "--values", "--overview"],
            vec!["--tensor", &dp, "--values", "--window=1,1"],
            vec!["--tensor", &dp, "--values", "--edge=0.25,0.75"],
            vec!["--tensor", &dp, "--values", "--zebra", "cols"],
            vec!["--tensor", &dp, "--values", "--base", "hex"],
            vec!["--tensor", &dp, "--values", "--slice", "2"],
            vec!["--tensor", &dp, "--heatmap"],
            vec!["--health"],          // the health-check popup over the tree
            vec!["--health-findings"], // …with the per-finding detail expanded
            vec!["--stats"],           // the full-screen checkpoint-stats view
            vec!["--stats-shards"],    // …with the per-shard breakdown expanded
        ];
        for extra in cases {
            super::assert_y_roundtrip(H5, extra);
        }
    }

    /// The `s` popup on a compressed `MoE` checkpoint: exercises the compression
    /// ratio (on-disk vs. logical) and the fused-experts section.
    #[test]
    fn stats_popup() {
        settings().bind(|| insta::assert_snapshot!(plain(&["--stats"])));
    }

    // Pin the actual command `y` copies for each screen (documents the round-trip
    // verified above). The fixture path is filtered to `[FIXTURE]`.
    #[test]
    fn emit_commands() {
        let dp = format!("{MOE}.down_proj.weight");
        let cases: &[(&str, Vec<&str>)] = &[
            ("detail", vec!["--tensor", &dp]),
            ("dtype stored", vec!["--tensor", &dp, "--dtype", "stored"]),
            ("histogram", vec!["--tensor", &dp, "--histogram"]),
            (
                "histogram bins",
                vec!["--tensor", &dp, "--histogram", "--bins", "4"],
            ),
            (
                "values window",
                vec!["--tensor", &dp, "--values", "--window=1,1"],
            ),
            (
                "values hex",
                vec!["--tensor", &dp, "--values", "--base", "hex"],
            ),
            ("heatmap", vec!["--tensor", &dp, "--heatmap"]),
        ];
        let mut out = String::new();
        for (label, args) in cases {
            let mut a = vec![H5];
            a.extend_from_slice(args);
            a.push("--emit-command");
            let _ = writeln!(out, "{label}: {}", super::run_bin(&a).trim());
        }
        settings().bind(|| insta::assert_snapshot!(out));
    }
}

// ---- `diff` subcommand ----

/// Write a safetensors file from (name, dtype, shape, seed) specs + string
/// metadata — a parametric sibling of `write_fixture` for the diff fixtures. The
/// payload is a byte ramp offset by `seed`, so two files can give a tensor the
/// same bytes (equal seed) or differing values (different seed).
fn write_st(path: &str, specs: &[(&str, Dtype, Vec<usize>, u8)], metadata: &[(&str, &str)]) {
    let buffers: Vec<Vec<u8>> = specs
        .iter()
        .map(|(_, dt, shape, seed)| {
            let bytes = shape.iter().product::<usize>() * dtype_size(*dt);
            (0..bytes)
                .map(|i| ((i + *seed as usize) % 251) as u8)
                .collect()
        })
        .collect();
    let data: HashMap<String, TensorView> = specs
        .iter()
        .zip(&buffers)
        .map(|((name, dt, shape, _), buf)| {
            (
                name.to_string(),
                TensorView::new(*dt, shape.clone(), buf).expect("valid tensor view"),
            )
        })
        .collect();
    let meta = Some(
        metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<HashMap<_, _>>(),
    );
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).expect("create fixtures dir");
    }
    safetensors::serialize_to_file(&data, &meta, Path::new(path)).expect("write fixture");
}

const DIFF_OLD: &str = "tests/fixtures/diff_old.safetensors";
const DIFF_NEW: &str = "tests/fixtures/diff_new.safetensors";
const DIFF_META: &str = "tests/fixtures/diff_meta.safetensors";

/// Three checkpoints. OLD vs NEW differ by one removed, one added, and two changed
/// tensors (a dtype change and a shape change), plus one added and one changed
/// metadata entry; `input_layernorm.weight` is identical and `mlp.weight` has the
/// same dtype+shape but different bytes (`seed` 0 vs 7, a values-only change for
/// `--tensor`). META has OLD's exact tensors but different metadata — so OLD vs
/// META differ *only* in metadata, for `--only-tensors`.
fn ensure_diff_fixtures() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let old_tensors: &[(&str, Dtype, Vec<usize>, u8)] = &[
            ("lm_head.weight", Dtype::F16, vec![2, 2], 0),
            ("model.embed_tokens.weight", Dtype::F16, vec![6, 4], 0),
            ("model.norm.weight", Dtype::F32, vec![4], 0),
            (
                "model.layers.0.input_layernorm.weight",
                Dtype::F32,
                vec![4],
                0,
            ),
            ("model.layers.0.mlp.weight", Dtype::U8, vec![4], 0),
        ];
        write_st(
            DIFF_OLD,
            old_tensors,
            &[("format", "pt"), ("note", "original")],
        );
        write_st(
            DIFF_NEW,
            &[
                ("model.embed_tokens.weight", Dtype::BF16, vec![6, 4], 0),
                ("model.norm.weight", Dtype::F32, vec![8], 0),
                (
                    "model.layers.0.input_layernorm.weight",
                    Dtype::F32,
                    vec![4],
                    0,
                ),
                ("model.layers.0.mlp.weight", Dtype::U8, vec![4], 7),
                ("model.rotary_emb.inv_freq", Dtype::F32, vec![16], 0),
            ],
            &[("format", "pt"), ("note", "edited"), ("extra", "x")],
        );
        // Same tensors as OLD, only the metadata differs.
        write_st(
            DIFF_META,
            old_tensors,
            &[("format", "pt"), ("note", "changed")],
        );
    });
}

/// Run `diff` with `args` (relative paths, so the header is checkout-independent)
/// and return its stdout plus exit code.
fn run_diff(args: &[&str]) -> (String, i32) {
    let mut full = vec!["diff"];
    full.extend_from_slice(args);
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args(&full)
        .output()
        .expect("run diff");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn diff_lists_changes_and_exits_1() {
    ensure_diff_fixtures();
    // Full diff is structural: mlp.weight (same dtype+shape, different bytes) is
    // "unchanged" here — value differences only surface under `--tensor`.
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_NEW]);
    assert_eq!(code, 1, "differences should exit 1; stdout:\n{out}");
    insta::assert_snapshot!(out);
}

#[test]
fn diff_identical_exits_0() {
    ensure_diff_fixtures();
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_OLD]);
    assert_eq!(code, 0, "identical should exit 0; stdout:\n{out}");
    assert!(out.contains("tensors: -0 +0 ~0"), "{out}");
    assert!(out.contains("metadata: -0 +0 ~0"), "{out}");
}

#[test]
fn diff_unreadable_path_exits_2() {
    ensure_diff_fixtures();
    let (_out, code) = run_diff(&[DIFF_OLD, "tests/fixtures/does_not_exist.safetensors"]);
    assert_eq!(code, 2, "an unreadable path should exit 2");
}

#[test]
fn diff_tensor_values_differ_and_exits_1() {
    ensure_diff_fixtures();
    // U8 [4]: bytes 0..3 vs 7..10 → all four differ, each by 7.
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--tensor", "model.layers.0.mlp.weight"]);
    assert_eq!(code, 1, "a value change should exit 1; stdout:\n{out}");
    insta::assert_snapshot!(out);
}

#[test]
fn diff_tensor_identical_values_exits_0() {
    ensure_diff_fixtures();
    let (out, code) = run_diff(&[
        DIFF_OLD,
        DIFF_NEW,
        "--tensor",
        "model.layers.0.input_layernorm.weight",
    ]);
    assert_eq!(code, 0, "identical values should exit 0; stdout:\n{out}");
    assert!(out.contains("(identical)"), "{out}");
}

#[test]
fn diff_tensor_shape_change_skips_values() {
    ensure_diff_fixtures();
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--tensor", "model.norm.weight"]);
    assert_eq!(code, 1, "a shape change should exit 1; stdout:\n{out}");
    assert!(
        out.contains("values: not compared (shapes differ)"),
        "{out}"
    );
}

#[test]
fn diff_tensor_missing_exits_2() {
    ensure_diff_fixtures();
    let (_out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--tensor", "no.such.tensor"]);
    assert_eq!(code, 2, "an absent tensor should exit 2");
}

const DIFF_GROUP_OLD: &str = "tests/fixtures/diff_group_old.safetensors";
const DIFF_GROUP_NEW: &str = "tests/fixtures/diff_group_new.safetensors";

/// A 4-layer checkpoint whose per-layer expert weight changes dtype identically
/// across every layer — the case `diff` collapses into one line.
fn ensure_group_fixtures() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let specs = |dt: Dtype| -> Vec<(&'static str, Dtype, Vec<usize>, u8)> {
            let names = [
                "model.layers.0.block_sparse_moe.experts.down_proj.weight",
                "model.layers.1.block_sparse_moe.experts.down_proj.weight",
                "model.layers.2.block_sparse_moe.experts.down_proj.weight",
                "model.layers.3.block_sparse_moe.experts.down_proj.weight",
            ];
            names
                .into_iter()
                .map(|n| (n, dt, vec![2, 5, 3], 0u8))
                .collect()
        };
        write_st(DIFF_GROUP_OLD, &specs(Dtype::U16), &[]);
        write_st(DIFF_GROUP_NEW, &specs(Dtype::F16), &[]);
    });
}

#[test]
fn diff_groups_repeated_layer_changes() {
    ensure_group_fixtures();
    // Default: the four per-layer changes collapse to one line with a range + count.
    let (out, code) = run_diff(&[DIFF_GROUP_OLD, DIFF_GROUP_NEW]);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains(
            "~ model.layers.{0-3}.block_sparse_moe.experts.down_proj.weight  [U16 (2, 5, 3)] → [F16 (2, 5, 3)]  (×4)"
        ),
        "{out}"
    );
    assert!(out.contains("tensors: -0 +0 ~4"), "counts stay raw; {out}");

    // `--full` lists every layer and drops the count suffix.
    let (full, _) = run_diff(&[DIFF_GROUP_OLD, DIFF_GROUP_NEW, "--full"]);
    assert_eq!(full.matches(".down_proj.weight").count(), 4, "{full}");
    assert!(!full.contains("(×"), "{full}");
}

#[test]
fn diff_only_tensors_drops_metadata_section_and_exit() {
    ensure_diff_fixtures();
    // OLD vs META differ only in metadata: by default that's a difference (exit 1)
    // and the section is shown...
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_META]);
    assert_eq!(code, 1, "a metadata-only difference should exit 1; {out}");
    assert!(out.contains("metadata:"), "{out}");
    // ...but `--only-tensors` drops it from the diff *and* the exit code, so the
    // otherwise-identical checkpoints compare equal (exit 0), with a clear note.
    let (out2, code2) = run_diff(&[DIFF_OLD, DIFF_META, "--only-tensors"]);
    assert_eq!(
        code2, 0,
        "ignoring the only difference should exit 0; {out2}"
    );
    assert!(
        out2.contains("metadata: not compared (--only-tensors)"),
        "{out2}"
    );
    assert!(
        !out2.contains("  ~ note"),
        "no per-entry metadata lines; {out2}"
    );
}

#[test]
fn diff_values_detects_value_only_change() {
    ensure_diff_fixtures();
    // mlp.weight has the same dtype+shape but different bytes (seed 0 vs 7).
    // Structural diff calls it unchanged...
    let (plain, _) = run_diff(&[DIFF_OLD, DIFF_NEW]);
    assert!(!plain.contains("mlp.weight"), "{plain}");
    // ...but `--values` reads the data and flags it (4 of 4 bytes differ by 7).
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--values"]);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains("~ model.layers.0.mlp.weight  [U8 (4)]  (values differ)"),
        "{out}"
    );
    assert!(
        out.contains("values: 4 of 4 differ  (max |Δ| 7, mean |Δ| 7)"),
        "{out}"
    );
    // A shape change can't be compared element-wise.
    assert!(
        out.contains("values: not compared (shapes differ)"),
        "{out}"
    );
    // Composes with --only-tensors (value diff kept; metadata noted as skipped).
    let (both, _) = run_diff(&[DIFF_OLD, DIFF_NEW, "--values", "--only-tensors"]);
    assert!(
        both.contains("mlp.weight  [U8 (4)]  (values differ)"),
        "{both}"
    );
    assert!(
        both.contains("metadata: not compared (--only-tensors)"),
        "{both}"
    );
}

#[test]
fn diff_filter_by_name_scopes_to_subset() {
    ensure_diff_fixtures();
    // A name glob scopes the whole diff to matching tensors; metadata is dropped.
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--name", "*.norm.weight"]);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains("~ model.norm.weight  [F32 (4)] → [F32 (8)]"),
        "{out}"
    );
    // Nothing else from the full diff leaks in, and metadata is noted as skipped.
    assert!(!out.contains("lm_head"), "{out}");
    assert!(!out.contains("embed_tokens"), "{out}");
    assert!(
        out.contains("metadata: not compared (filtered subset)"),
        "{out}"
    );
}

#[test]
fn diff_filter_dtype_matches_either_side() {
    ensure_diff_fixtures();
    // --dtype-is globs the stored dtype and matches EITHER side, so it catches
    // embed_tokens (F16 → BF16) as well as the removed F16 lm_head, but not the
    // F32-only norm.
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--dtype-is", "F16"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("- lm_head.weight"), "{out}");
    assert!(
        out.contains("~ model.embed_tokens.weight  [F16 (6, 4)] → [BF16 (6, 4)]"),
        "{out}"
    );
    assert!(
        !out.contains("norm.weight"),
        "F32-only tensor excluded; {out}"
    );
}

#[test]
fn diff_filter_shape_wildcards() {
    ensure_diff_fixtures();
    // `*` matches exactly one dimension: `6,*` selects only the 2-D (6, 4) tensor.
    let (two_d, _) = run_diff(&[DIFF_OLD, DIFF_NEW, "--shape-is", "6,*"]);
    assert!(two_d.contains("model.embed_tokens.weight"), "{two_d}");
    assert!(
        !two_d.contains("norm.weight"),
        "1-D excluded by `6,*`; {two_d}"
    );
    // `*` alone = exactly-1-D tensors (norm changed shape 4 → 8); 2-D is excluded.
    let (one_d, _) = run_diff(&[DIFF_OLD, DIFF_NEW, "--shape-is", "*"]);
    assert!(one_d.contains("model.norm.weight"), "{one_d}");
    assert!(
        !one_d.contains("embed_tokens"),
        "2-D excluded by `*`; {one_d}"
    );
}

#[test]
fn diff_filter_values_on_subset() {
    ensure_diff_fixtures();
    // A values-only change is normally structural-"unchanged"; `--values` scoped
    // to the mlp.weight subset promotes it to a difference — and nothing else.
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--values", "--name", "*.mlp.weight"]);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains("~ model.layers.0.mlp.weight  [U8 (4)]  (values differ)"),
        "{out}"
    );
    assert!(
        !out.contains("norm.weight"),
        "only the subset compared; {out}"
    );
}

#[test]
fn diff_filter_no_match_exits_0() {
    ensure_diff_fixtures();
    // An empty subset has no differences (and the stderr note — not captured here
    // — reports "0 tensors matched").
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--name", "*.does_not_exist"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("tensors: -0 +0 ~0 (0 unchanged)"), "{out}");
}

#[test]
fn diff_filter_bad_glob_exits_2() {
    ensure_diff_fixtures();
    let (_out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--name", "[bad"]);
    assert_eq!(code, 2, "an invalid glob is a usage error");
}

#[test]
fn diff_parallel_matches_sequential_and_reports_time() {
    ensure_diff_fixtures();
    // The result is identical regardless of --jobs (parallelism is order-free).
    let (seq, _) = run_diff(&[DIFF_OLD, DIFF_NEW, "--values", "--jobs", "1"]);
    let (par, _) = run_diff(&[DIFF_OLD, DIFF_NEW, "--values", "--jobs", "4"]);
    assert_eq!(seq, par, "parallel diff must match sequential");
    // Elapsed time is reported by default (on stderr, so stdout stays clean).
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args(["diff", DIFF_OLD, DIFF_NEW, "--values"])
        .output()
        .expect("run diff");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("done in "),
        "stderr should report elapsed time:\n{err}"
    );
}

/// **A filtered diff's `size:` / `params:` cover the tensors it compared.**
///
/// They used to be the whole checkpoints': the summary summed its sizes once at load, and a filter
/// dropped only the signatures — so a report about two of six tensors was headed by the sizes of both
/// files. Nothing marked them as such, so they simply read as the subset's while being something else.
/// Now they *are* the subset's, and the label says so.
#[test]
fn diff_filtered_totals_cover_the_matched_subset() {
    ensure_diff_fixtures();
    let (whole, _) = run_diff(&[DIFF_OLD, DIFF_NEW]);
    let (scoped, _) = run_diff(&[DIFF_OLD, DIFF_NEW, "--name", "model.norm.weight"]);

    // Unfiltered: the bare label, and the checkpoints' own totals.
    assert!(whole.contains("\nsize: 92 B → 164 B"), "{whole}");
    assert!(whole.contains("\nparams: 40 → 56"), "{whole}");

    // Filtered to one tensor — F32 [4] → F32 [8], so 16 B → 32 B and 4 → 8 params.
    assert!(
        scoped.contains("size (filtered subset): 16 B → 32 B (+16 B, +100.0%)"),
        "{scoped}"
    );
    assert!(
        scoped.contains("params (filtered subset): 4 → 8 (+4, +100.0%)"),
        "{scoped}"
    );
    assert!(
        !scoped.contains("92 B"),
        "a scoped report must not carry the whole checkpoint's size:\n{scoped}"
    );
}

#[test]
fn diff_filter_reports_matched_schema_on_stderr() {
    ensure_group_fixtures();
    // The filter context goes to stderr: "matched M of N" plus the matched names
    // collapsed into their index-templated schema (which layers/experts matched).
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args([
            "diff",
            DIFF_GROUP_OLD,
            DIFF_GROUP_NEW,
            "--name",
            "*.down_proj.weight",
        ])
        .output()
        .expect("run diff");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("matched 4 of 4 tensor(s):"), "stderr:\n{err}");
    assert!(
        err.contains("model.layers.{0-3}.block_sparse_moe.experts.down_proj.weight  (×4)"),
        "stderr:\n{err}"
    );
}

const MAP_OLD: &str = "tests/fixtures/diff_map_old.safetensors";
const MAP_NEW: &str = "tests/fixtures/diff_map_new.safetensors";

/// Two checkpoints holding the same three per-layer tensors under *different*
/// naming schemes: OLD uses `…mlp.experts.down_proj`, NEW uses
/// `…block_sparse_moe.experts.down_proj.weight`. A `--map` rename rule that keeps
/// the layer index should line them up.
fn ensure_map_fixtures() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let old_names: Vec<String> = (0..3)
            .map(|i| format!("model.layers.{i}.mlp.experts.down_proj"))
            .collect();
        let new_names: Vec<String> = (0..3)
            .map(|i| format!("model.layers.{i}.block_sparse_moe.experts.down_proj.weight"))
            .collect();
        let old: Vec<(&str, Dtype, Vec<usize>, u8)> = old_names
            .iter()
            .map(|n| (n.as_str(), Dtype::BF16, vec![2, 3], 0u8))
            .collect();
        let new: Vec<(&str, Dtype, Vec<usize>, u8)> = new_names
            .iter()
            .map(|n| (n.as_str(), Dtype::BF16, vec![2, 3], 0u8))
            .collect();
        write_st(MAP_OLD, &old, &[]);
        write_st(MAP_NEW, &new, &[]);
    });
}

#[test]
fn diff_map_aligns_renamed_tensors() {
    ensure_map_fixtures();
    // Without a map, every tensor differs by name → all removed + all added.
    let (out, code) = run_diff(&[MAP_OLD, MAP_NEW]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("tensors: -3 +3 ~0"), "{out}");

    // A rename rule that preserves the layer index lines them up (unchanged).
    let (out, code) = run_diff(&[
        MAP_OLD,
        MAP_NEW,
        "--map",
        r"\.mlp\.experts\.down_proj$=>.block_sparse_moe.experts.down_proj.weight",
    ]);
    assert_eq!(code, 0, "map should align the renamed tensors; {out}");
    assert!(out.contains("tensors: -0 +0 ~0"), "{out}");
}

#[test]
fn diff_map_from_plain_and_json_files() {
    ensure_map_fixtures();
    let dir = std::env::temp_dir();

    // Plain-text rules file: 'PATTERN=>REPL' per line, '#' comments ignored.
    let plain = dir.join("ce_diff_map_rules.txt");
    std::fs::write(
        &plain,
        "# gpt-oss rename\n\
         \\.mlp\\.experts\\.down_proj$ => .block_sparse_moe.experts.down_proj.weight\n",
    )
    .unwrap();
    let (out, code) = run_diff(&[MAP_OLD, MAP_NEW, "--map-from", plain.to_str().unwrap()]);
    assert_eq!(code, 0, "plain rules file should align; {out}");
    assert!(out.contains("tensors: -0 +0 ~0"), "{out}");

    // JSON array of [pattern, replacement] pairs (backslashes escaped for JSON).
    let json = dir.join("ce_diff_map_rules.json");
    std::fs::write(
        &json,
        r#"[["\\.mlp\\.experts\\.down_proj$", ".block_sparse_moe.experts.down_proj.weight"]]"#,
    )
    .unwrap();
    let (out, code) = run_diff(&[MAP_OLD, MAP_NEW, "--map-from", json.to_str().unwrap()]);
    assert_eq!(code, 0, "json rules file should align; {out}");
    assert!(out.contains("tensors: -0 +0 ~0"), "{out}");
}

#[test]
fn diff_map_bad_regex_exits_2() {
    ensure_map_fixtures();
    let (_out, code) = run_diff(&[MAP_OLD, MAP_NEW, "--map", "([unclosed=>x"]);
    assert_eq!(code, 2, "an invalid --map regex should exit 2");
}

/// **`--align-fused` lines an unfused checkpoint up with its fused counterpart.**
///
/// The reported case, in miniature: two layouts of one model share no tensor name, so a plain diff
/// reports every tensor of both sides as one-sided — 80,107 against 933 for the real pair — which is
/// true and answers nothing. Aligned, the per-expert tensors fold onto the fused tensor that holds them
/// and the row says how many did.
#[test]
fn diff_align_fused_folds_the_experts_onto_the_fused_tensor() {
    let dir = std::env::temp_dir().join(format!("cs_fused_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let unfused = dir.join("unfused.safetensors");
    let fused = dir.join("fused.safetensors");
    // Two experts, Mixtral naming; and the one fused tensor with the expert dimension in front.
    write_st(
        unfused.to_str().unwrap(),
        &[
            (
                "model.layers.0.block_sparse_moe.experts.0.w2.weight",
                Dtype::U8,
                vec![4, 2],
                0,
            ),
            (
                "model.layers.0.block_sparse_moe.experts.1.w2.weight",
                Dtype::U8,
                vec![4, 2],
                0,
            ),
        ],
        &[],
    );
    write_st(
        fused.to_str().unwrap(),
        &[(
            "model.layers.0.block_sparse_moe.experts.down_proj.weight",
            Dtype::U8,
            vec![2, 4, 2],
            0,
        )],
        &[],
    );

    // Unaligned: nothing lines up — one added, two removed.
    let (plain, _) = run_diff(&[unfused.to_str().unwrap(), fused.to_str().unwrap()]);
    assert!(plain.contains("tensors: -2 +1 ~0"), "{plain}");

    // Aligned: one row, and it says two tensors fold onto one.
    let (aligned, _) = run_diff(&[
        unfused.to_str().unwrap(),
        fused.to_str().unwrap(),
        "--align-fused",
    ]);
    assert!(aligned.contains("tensors: -0 +0 ~1"), "{aligned}");
    assert!(
        aligned.contains("~ model.layers.0.block_sparse_moe.experts.down_proj.weight"),
        "{aligned}"
    );
    assert!(
        aligned.contains("(×2 → ×1)"),
        "the row should say what folded:\n{aligned}"
    );
}

/// **A path that carries its own host is not given a second one.**
///
/// `--ssh-proxy H` plus `H:/path` used to keep the host on the path and prefix `H:` again, so the read
/// looked for `H:H:/path` — "no safetensors files found" — and that spelling was written to the recents
/// list, where it sat as a row that could never be opened. Nothing here connects: the failure is a
/// resolution failure, and the message names the path it tried.
#[test]
fn diff_accepts_a_scp_path_together_with_the_matching_proxy_flag() {
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args([
            "diff",
            "--ssh-proxy",
            "user@example.invalid",
            "user@example.invalid:/opt/a",
            "user@example.invalid:/opt/b",
        ])
        .output()
        .expect("run diff");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !err.contains("example.invalid:user@example.invalid")
            && !err.contains("invalid:user@example"),
        "the host was prefixed twice; stderr:\n{err}"
    );

    // And two hosts that disagree is a named conflict, not a guess.
    let clash = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args([
            "diff",
            "--ssh-proxy",
            "user@one.invalid",
            "user@two.invalid:/opt/a",
            "user@two.invalid:/opt/b",
        ])
        .output()
        .expect("run diff");
    let err = String::from_utf8_lossy(&clash.stderr);
    assert!(
        err.contains("one checkpoint has one host"),
        "expected the two hosts to be named; stderr:\n{err}"
    );
}

/// `--verify-repack` decodes packed indices **where the data is** — and under `--ssh-proxy` that is the
/// proxy, which can open both an `s3://` cstorch checkpoint and a path on its own filesystem.
///
/// This pair used to be refused outright ("both sides must be s3://"), which was a limitation of the
/// remote call rather than of the verification: the *value* comparison was already reading exactly
/// these two kinds on exactly this host. So the pair is now attempted, and the only thing that stops
/// it here is that `example.invalid` does not resolve — a connection error, naming the host, rather
/// than a refusal about what the addresses are.
#[test]
fn diff_verify_repack_attempts_a_safetensors_side_on_the_proxy() {
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args([
            "diff",
            "--ssh-proxy",
            "user@example.invalid",
            "/tmp/no-such-checkpoint",
            "s3://bucket/key",
            "--verify-repack",
        ])
        .output()
        .expect("run diff --verify-repack");
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "a failed run exits 2; {err}");
    assert!(
        !err.contains("s3:// cstorch checkpoints"),
        "the pair is no longer refused for its address shape; stderr:\n{err}"
    );
    assert!(
        err.contains("example.invalid"),
        "what stopped it is the unreachable proxy, and the message should name it; stderr:\n{err}"
    );
}

#[test]
fn diff_map_collision_warns_on_stderr() {
    ensure_map_fixtures();
    // A rule that drops the layer index collapses all three layers onto one name.
    let out = Command::new(env!("CARGO_BIN_EXE_checkpoint-studio"))
        .env("XDG_CONFIG_HOME", scratch_config())
        .args([
            "diff",
            MAP_OLD,
            MAP_NEW,
            "--map",
            r"model\.layers\.\d+\.=>model.layers.X.",
        ])
        .output()
        .expect("run diff");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("maps multiple tensors onto"),
        "a colliding rename should warn; stderr:\n{err}"
    );
}

#[test]
fn diff_tensor_dtype_view_changes_decode() {
    ensure_diff_fixtures();
    // mlp.weight is U8 [4]; under the u4 view each byte is two nibbles, so the
    // value comparison sees 8 logical values, not 4 — proving --dtype is applied.
    let (out, code) = run_diff(&[
        DIFF_OLD,
        DIFF_NEW,
        "--tensor",
        "model.layers.0.mlp.weight",
        "--dtype",
        "u4",
    ]);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains("of 8 differ"),
        "u4 view should double the count; {out}"
    );
}

#[test]
fn diff_tensor_histogram_table() {
    ensure_diff_fixtures();
    // mlp.weight U8 [4]: old bytes 0..3, new 7..10 — disjoint distributions (TVD 1).
    let (out, code) = run_diff(&[
        DIFF_OLD,
        DIFF_NEW,
        "--tensor",
        "model.layers.0.mlp.weight",
        "--histogram",
    ]);
    assert_eq!(code, 1, "{out}");
    assert!(
        out.contains("histogram of model.layers.0.mlp.weight"),
        "{out}"
    );
    assert!(
        out.contains("TVD 1"),
        "disjoint distributions → TVD 1; {out}"
    );
    assert!(
        out.contains("-1"),
        "a bin only the old side fills → -1; {out}"
    );
}

#[test]
fn diff_histogram_whole_checkpoint_reports_tvd() {
    ensure_diff_fixtures();
    let (out, code) = run_diff(&[DIFF_OLD, DIFF_NEW, "--histogram"]);
    assert_eq!(code, 1, "{out}");
    // mlp.weight: same dtype+shape, disjoint distribution → a distribution change.
    assert!(
        out.contains("~ model.layers.0.mlp.weight  [U8 (4)]  (distribution differs)"),
        "{out}"
    );
    assert!(out.contains("histogram: TVD 1"), "{out}");
    // A shape change can't be binned into a shared layout.
    assert!(
        out.contains("histogram: not compared (shapes differ)"),
        "{out}"
    );
}

// ---------------------------------------------------------------- CLI surface ----
//
// The paths above are snapshot tests of *rendered screens*. These cover the rest of
// the command line — the dispatch, the exit codes and the writes — where the value is
// in the behaviour rather than the pixels, so they assert directly instead of
// snapshotting. Each one also drags a whole slice of `main.rs` / `explorer` /
// `readers` through the binary, which unit tests can't reach.

/// A tensor name from the generated fixture, for the flags that take one.
const A_TENSOR: &str = "model.layers.0.mlp.down_proj.weight";

#[test]
fn metadata_flag_opens_that_entry() {
    ensure_fixture();
    let out = run_plain(FIXTURE, &["--metadata", "format"]);
    assert!(out.contains("format"), "{out}");
}

#[test]
fn compute_stats_scans_the_tensor_and_reports_a_summary() {
    ensure_fixture();
    let out = run_plain(FIXTURE, &["--tensor", A_TENSOR, "--compute-stats"]);
    // A finished scan reports mean/std/zeros — the numbers, not just the offer to scan.
    assert!(out.contains("mean"), "{out}");
    assert!(out.contains("zeros"), "{out}");
}

#[test]
fn a_slice_and_a_reinterpreted_dtype_reach_the_values_grid() {
    ensure_fixture();
    // The 3-D tensor has slices; `--slice 1` must show that one, and `--dtype` must
    // decode the same bytes differently.
    let sliced = run_plain(FIXTURE, &["--tensor", A_TENSOR, "--values", "--slice", "1"]);
    assert!(sliced.contains("Values"), "{sliced}");
    let viewed = run_plain(
        FIXTURE,
        &["--tensor", A_TENSOR, "--values", "--dtype", "F16"],
    );
    assert!(viewed.contains("F16"), "{viewed}");
    assert_ne!(
        sliced, viewed,
        "a dtype override must change what the grid shows"
    );
}

#[test]
fn the_abs_max_heatmap_is_a_different_picture_from_the_sampled_one() {
    ensure_fixture();
    let sampled = run_plain(FIXTURE, &["--tensor", A_TENSOR, "--heatmap"]);
    let abs_max = run_plain(FIXTURE, &["--tensor", A_TENSOR, "--heatmap", "--abs-max"]);
    assert!(abs_max.contains("abs-max"), "the mode is named: {abs_max}");
    assert_ne!(sampled, abs_max, "abs-max scans instead of sampling");
}

#[test]
fn a_histogram_takes_its_bin_count_from_the_flag() {
    ensure_fixture();
    let out = run_plain(
        FIXTURE,
        &["--tensor", A_TENSOR, "--histogram", "--bins", "8"],
    );
    assert!(
        out.contains("Histogram") || out.contains("histogram"),
        "{out}"
    );
}

/// The explore path exits **1** on a bad request. (`diff` and `check` use 0/1/2 as a
/// semantic protocol — 1 there means "differences found", not "error" — so the codes
/// aren't the same across subcommands, and each is asserted where it belongs.)
#[test]
fn an_unknown_tensor_exits_nonzero_rather_than_showing_the_tree() {
    ensure_fixture();
    let (out, code) = run_bin_status(&[FIXTURE, "--plain", "--tensor", "no.such.tensor"]);
    assert_eq!(code, 1, "{out}");
    assert!(
        !out.contains("Checkpoint Studio"),
        "it must not fall back to the tree"
    );
}

#[test]
fn a_malformed_filter_exits_nonzero_rather_than_showing_everything() {
    ensure_fixture();
    // Silently ignoring a bad filter would show the whole checkpoint and look like the
    // filter matched everything.
    let (out, code) = run_bin_status(&[FIXTURE, "--plain", "--filter", "dtpye:F16"]);
    assert_eq!(code, 1);
    assert!(!out.contains("Checkpoint Studio"), "{out}");
}

#[test]
fn a_path_that_does_not_exist_exits_nonzero() {
    let (_out, code) = run_bin_status(&["/no/such/checkpoint.safetensors", "--plain"]);
    assert_eq!(code, 1);
}

#[test]
fn recursive_finds_the_fixture_under_a_directory() {
    ensure_fixture();
    let dir = Path::new(FIXTURE).parent().expect("a fixtures directory");
    let out = run_plain(&dir.to_string_lossy(), &["--recursive"]);
    assert!(out.contains("Checkpoint Studio"), "{out}");
}

#[test]
fn the_health_check_can_be_skipped() {
    ensure_fixture();
    let with = run_plain(FIXTURE, &[]);
    let without = run_plain(FIXTURE, &["--no-health-check"]);
    // Both render; skipping health must not change the tree itself.
    assert!(with.contains("Checkpoint Studio") && without.contains("Checkpoint Studio"));
}

#[test]
fn check_reports_json_and_sarif_for_the_same_findings() {
    ensure_fixture();
    for (format, marker) in [("json", "\"checks\""), ("sarif", "\"runs\"")] {
        let (out, code) = run_bin_status(&["check", FIXTURE, "--format", format]);
        assert!(out.contains(marker), "{format}: {out}");
        assert!(code == 0 || code == 1, "{format} exited {code}");
        // Machine formats must parse.
        serde_json::from_str::<serde_json::Value>(&out)
            .unwrap_or_else(|e| panic!("{format} output is not JSON: {e}\n{out}"));
    }
}

#[test]
fn rename_writes_the_new_names_into_a_copy() {
    ensure_fixture();
    // `convert --rename` edits in place, so work on a copy and read it back through the
    // binary — this is the only write path the CLI has for safetensors.
    let dir = std::env::temp_dir().join("ckpt_studio_cli_rename");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let target = dir.join("renamed.safetensors");
    std::fs::copy(FIXTURE, &target).expect("copy the fixture");
    let path = target.to_string_lossy().into_owned();

    // A rename that would GROW the header is refused: honouring it would mean moving
    // tensor data, which this mode promises not to do. That refusal is the safety
    // property, so pin it first.
    let (grow, grow_code) = run_bin_status(&[
        "convert",
        &path,
        "--map",
        r"model\.norm\.=>model.a_much_longer_prefix_norm.",
        "--force",
    ]);
    assert_ne!(grow_code, 0, "growing the header must be refused: {grow}");

    // A rename that fits (same length or shorter) is applied in place.
    let (out, code) = run_bin_status(&[
        "convert",
        &path,
        "--map",
        r"model\.norm\.=>model.nrm.",
        "--force", // skip the confirmation prompt (there's no tty here)
    ]);
    assert_eq!(code, 0, "rename failed: {out}");

    let after = run_plain(&path, &[]);
    assert!(after.contains("nrm"), "the new name is there:\n{after}");
    assert!(
        !after.contains("norm.weight"),
        "the old name is gone:\n{after}"
    );
    let _ = std::fs::remove_file(&target);
}

#[cfg(feature = "hdf5")]
#[test]
fn repack_writes_a_new_hdf5_and_leaves_the_original_alone() {
    let src = "tests/fixtures/tiny.hdf5";
    let dir = std::env::temp_dir().join("ckpt_studio_cli_repack");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let out_path = dir.join("repacked.hdf5");
    let _ = std::fs::remove_file(&out_path);
    let before = std::fs::metadata(src).expect("the fixture exists").len();

    // Repack mode: the destination is a positional argument, not a flag.
    let (out, code) = run_bin_status(&["convert", src, &out_path.to_string_lossy(), "--force"]);
    assert_eq!(code, 0, "repack failed: {out}");
    assert!(out_path.exists(), "the output file was not written");
    assert_eq!(
        std::fs::metadata(src).expect("still there").len(),
        before,
        "repack must not touch the source"
    );
    // The repacked file reads back as a checkpoint.
    let text = run_plain(&out_path.to_string_lossy(), &[]);
    assert!(text.contains("Checkpoint Studio"), "{text}");
    let _ = std::fs::remove_file(&out_path);
}

/// A sharded checkpoint directory with an index — the layout every real `HuggingFace`
/// model uses, and the one load path the single-file fixture never takes (index
/// parsing, multi-shard grouping, and the index-vs-files health reconcile).
fn write_sharded(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create the shard directory");
    let shards = [
        (
            "model-00001-of-00002.safetensors",
            vec![("model.embed_tokens.weight", Dtype::F16, vec![4, 4])],
        ),
        (
            "model-00002-of-00002.safetensors",
            vec![
                (
                    "model.layers.0.mlp.down_proj.weight",
                    Dtype::F32,
                    vec![2, 4],
                ),
                ("model.norm.weight", Dtype::F32, vec![4]),
            ],
        ),
    ];
    let mut weight_map = serde_json::Map::new();
    for (file, specs) in &shards {
        let buffers: Vec<Vec<u8>> = specs
            .iter()
            .map(|(_, dt, shape)| {
                let bytes = shape.iter().product::<usize>() * dtype_size(*dt);
                (0..bytes).map(|i| (i % 251) as u8).collect()
            })
            .collect();
        let views: Vec<(String, TensorView<'_>)> = specs
            .iter()
            .zip(&buffers)
            .map(|((name, dt, shape), buf)| {
                (
                    (*name).to_string(),
                    TensorView::new(*dt, shape.clone(), buf).expect("view"),
                )
            })
            .collect();
        safetensors::serialize_to_file(views, &None, &dir.join(file)).expect("write shard");
        for (name, ..) in specs {
            weight_map.insert(
                (*name).to_string(),
                serde_json::Value::String((*file).to_string()),
            );
        }
    }
    let index = serde_json::json!({ "metadata": { "total_size": 0 }, "weight_map": weight_map });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).expect("index json"),
    )
    .expect("write the index");
}

#[test]
fn a_sharded_directory_loads_every_shard_and_checks_its_index() {
    let dir = std::env::temp_dir().join("ckpt_studio_sharded");
    let _ = std::fs::remove_dir_all(&dir);
    write_sharded(&dir);
    let path = dir.to_string_lossy().into_owned();

    // The tree shows the tensors from both shards.
    let tree = run_plain(&path, &["--recursive"]);
    for name in ["embed_tokens", "down_proj", "norm"] {
        assert!(tree.contains(name), "{name} missing:\n{tree}");
    }

    // The file browser lists both shards and the index.
    let files = run_plain(&path, &["--files"]);
    assert!(files.contains("00001-of-00002"), "{files}");
    assert!(files.contains("index.json"), "{files}");

    // With the index matching the files, `check` reports the file/sharding check as a
    // pass rather than n/a — the reconcile actually ran.
    let (report, code) = run_bin_status(&["check", &path]);
    assert!(code == 0 || code == 1, "check exited {code}: {report}");
    assert!(report.contains("Files & sharding"), "{report}");
    assert!(
        !report.contains("Files & sharding      — n/a"),
        "the index was ignored:\n{report}"
    );

    // Now break the index: a tensor assigned to a shard that doesn't have it must be
    // reported, not silently ignored.
    let index_path = dir.join("model.safetensors.index.json");
    let mut index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index_path).expect("read")).expect("parse");
    index["weight_map"]["model.ghost.weight"] =
        serde_json::Value::String("model-00001-of-00002.safetensors".into());
    std::fs::write(
        &index_path,
        serde_json::to_vec_pretty(&index).expect("json"),
    )
    .expect("write");
    let (broken, _code) = run_bin_status(&["check", &path]);
    assert!(
        broken.contains("ghost"),
        "a tensor the index invents must be reported:\n{broken}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
