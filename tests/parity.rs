//! The cross-language parity contract between the TUI and the web client.
//!
//! A handful of display rules exist twice — once in Rust for the TUI, once in
//! TypeScript for the browser, which can't call into this crate. Comments saying
//! "mirrors the TUI" don't stop the two from drifting, and drift here is the worst
//! kind: silent. The same tensor simply reports a different size in the two UIs and
//! nobody notices until someone compares screenshots.
//!
//! So the agreement is a test. This file generates `shared/parity/format.json` from
//! the *Rust* implementations — Rust is the reference — and asserts the committed
//! fixture still matches. `web/src/lib/parity.test.ts` reads the same file and
//! asserts the TypeScript produces the same strings. Change a rule on either side
//! and one of the two tests fails, telling you exactly which case moved.
//!
//! Regenerate after an intentional change:
//!
//! ```text
//! UPDATE_PARITY=1 cargo test --test parity
//! ```
//!
//! `shared/parity/README.md` records the rules that are deliberately NOT shared.

// An unwrap in a test IS the assertion: the panic is the failure report. (Product code
// denies these — see `[workspace.lints.clippy]` in Cargo.toml.)
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::path::PathBuf;

use checkpoint_studio_core::kernel::{Session, sort_rows};
use checkpoint_studio_core::tree::{
    Layout, MetadataInfo, Storage, TensorInfo, TreeBuilder, TreeNode,
};
use checkpoint_studio_core::utils::{format_parameters, format_percent, format_size};
use checkpoint_studio_core::viewstate::{SortDir, SortKey};
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use serde_json::{Value, json};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shared/parity/format.json")
}

/// Byte sizes worth pinning: the unit boundaries, the exact-tie cases where two
/// languages' rounding rules disagree (an odd multiple of 0.25 in the chosen unit —
/// `1280 B` is `1.25 KiB`), and one value per unit up to PiB.
const SIZES: &[usize] = &[
    0,
    1,
    999,
    1023,
    1024,
    1025,
    1500,
    1536,
    1280, // 1.25 KiB — exact tie at one decimal
    1792, // 1.75 KiB — exact tie, rounds the other way
    10_240,
    1_048_576,
    1_310_720, // 1.25 MiB — exact tie one unit up
    1_572_864,
    10_485_760,
    622_854_144, // 593.5 MiB — a real embedding shard
    1_073_741_824,
    2_952_790_016, // 2.75 GiB — exact tie
    61_847_529_062,
    1_099_511_627_776,     // 1 TiB
    1_125_899_906_842_624, // 1 PiB
    2_251_799_813_685_248, // 2 PiB — past the old GiB ceiling
];

/// Parameter counts: the K/M/B/T boundaries, plus the same tie cases.
const COUNTS: &[usize] = &[
    0,
    1,
    999,
    1_000,
    1_001,
    1_250, // 1.25K — exact tie
    1_500,
    1_750, // 1.75K — exact tie
    9_999,
    10_000,
    999_999,
    1_000_000,
    1_250_000,
    30_900_000_000,
    999_999_999,
    1_000_000_000,
    1_000_000_000_000, // 1T — past the old B ceiling
    1_750_000_000_000,
];

/// Zero fractions: the exact zero, the scientific-notation threshold either side, and
/// ordinary percentages. `(zeros, count)` so `is_zero` comes from the count, as it
/// does at the call site.
const ZEROS: &[(u64, u64)] = &[
    (0, 1_000),
    (1, 1_000_000_000), // 1e-7 % — must not read as 0.0%
    (1, 1_000),         // 0.1 % — exactly at the threshold
    (9, 10_000),        // 0.09 % — just under it
    (1, 2),
    (1, 3),
    (999, 1_000),
    (1_000, 1_000),
];

/// A realistic slice of tensor names, plus a couple of shapes that exercise
/// case-sensitivity and dotted queries.
const NAMES: &[&str] = &[
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.gate_proj.qscale",
    "model.layers.1.mlp.down_proj.weight",
    "model.layers.10.mlp.up_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
    "vision_tower.blocks.0.attn.proj.weight",
    "MODEL.UPPER.weight",
];

/// Queries: plain, dotted, mixed-case (smart case makes these case-sensitive),
/// non-matching, and empty.
const QUERIES: &[&str] = &[
    "gate",
    "gateproj",
    "qsc",
    "l0mlp",
    "layers.10",
    "Gate",  // uppercase ⇒ smart case turns case-sensitive
    "UPPER", // matches only the uppercase name
    "upper", // ...whereas lowercase matches it too
    "zzz",
    "",
    "wieght", // a transposition: subsequence matching must NOT match it
];

/// A synthetic checkpoint for the tree-row contract: a top-level tensor, a `model`
/// group with a nested `layers.0` group, and one metadata entry — enough to exercise
/// grouping, depth, the single-child chain the builder collapses, and a non-tensor row.
///
/// Hand-built rather than read from `tests/fixtures/`: a `TensorInfo` carries the
/// **absolute** path it was loaded from, which would make the committed fixture differ
/// per machine.
///
/// Returned **rooted** — through [`Session::build_rooted_tree`], the very call both
/// frontends make — so the fixture is the tree a browser is actually served, summarising
/// root and all. That matters for more than realism: with the bare forest, `model` sits
/// at depth 0, and a client that seeded its fold state from only the top level would
/// still produce the right rows. Rooted, `model` is nested, so the fixture can tell the
/// difference.
fn sample_tree() -> Vec<TreeNode> {
    let tensor = |name: &str, dtype: &str, shape: &[usize], size: usize| TensorInfo {
        name: name.to_string(),
        dtype: dtype.to_string(),
        shape: shape.to_vec(),
        size_bytes: size,
        num_elements: shape.iter().product(),
        storage: Storage::Unknown,
        source_path: "model.safetensors".to_string(),
        layout: Layout::None,
    };
    let tensors = vec![
        tensor("lm_head.weight", "I32", &[2, 4], 32),
        tensor("model.embed_tokens.weight", "F16", &[6, 4], 48),
        tensor(
            "model.layers.0.mlp.down_proj.weight",
            "U16",
            &[3, 4, 5],
            120,
        ),
        tensor(
            "model.layers.0.self_attn.q_proj.weight",
            "BF16",
            &[4, 4],
            32,
        ),
        tensor("model.norm.weight", "F32", &[4], 16),
    ];
    let metadata = vec![MetadataInfo {
        name: "format".to_string(),
        value: "pt".to_string(),
        value_type: "string".to_string(),
    }];
    Session::from_parts(tensors, metadata, None)
        .build_rooted_tree(&[PathBuf::from("/models/tiny/model.safetensors")])
}

/// One tree row, projected to what both flatteners must agree on: how deep it sits,
/// what kind of row it is, its name, and whether it can be expanded. Deliberately not
/// the rendered text — that differs by medium (see the README) — but the row list
/// itself is what "the web UI looks like the TUI" means.
fn row_projection(node: &TreeNode, depth: usize) -> Value {
    let kind = match node {
        TreeNode::Group { .. } => "group",
        TreeNode::Tensor { .. } => "tensor",
        TreeNode::Metadata { .. } => "metadata",
    };
    let has_children = matches!(node, TreeNode::Group { children, .. } if !children.is_empty());
    json!([depth, kind, node.name(), has_children])
}

/// Tensors for the sort contract, deliberately in an order no facet already agrees with,
/// and chosen so each key produces a *different* winner — otherwise the fixture would pass
/// for a sort that silently ignored its key. `layers.2` / `layers.10` are here for the
/// numeric-collation rule the two languages implement differently by default.
fn sort_sample() -> Vec<TreeNode> {
    let t = |name: &str, dtype: &str, shape: &[usize]| {
        let n: usize = shape.iter().product();
        TreeNode::Tensor {
            info: TensorInfo {
                name: name.to_string(),
                dtype: dtype.to_string(),
                shape: shape.to_vec(),
                size_bytes: n * 4,
                num_elements: n,
                storage: Storage::Unknown,
                source_path: "model.safetensors".to_string(),
                layout: Layout::None,
            },
            label: None,
        }
    };
    vec![
        t("model.layers.10.mlp.w", "F32", &[4, 4]),
        t("model.layers.2.mlp.w", "BF16", &[64]),
        t("a.big.tensor", "I32", &[8, 8, 8]),
        t("z.small.tensor", "U8", &[2]),
    ]
}

/// The sort contract: the sample tensors, and the order each `(key, direction)` puts them
/// in. Built outside `json!` because a method chain can't live inside the macro.
fn sort_section() -> Value {
    let tensors: Vec<Value> = sort_sample()
        .iter()
        .map(|n| match n {
            TreeNode::Tensor { info, .. } => json!({
                "name": info.name,
                "dtype": info.dtype,
                "shape": info.shape,
                "size_bytes": info.size_bytes,
                "num_elements": info.num_elements,
            }),
            TreeNode::Group { .. } | TreeNode::Metadata { .. } => Value::Null,
        })
        .collect();

    let keys = [
        ("name", SortKey::Name),
        ("size", SortKey::Size),
        ("params", SortKey::Params),
        ("dtype", SortKey::Dtype),
        ("rank", SortKey::Rank),
    ];
    let mut orders: Vec<Value> = Vec::new();
    for (label, key) in keys {
        for (dir, dlabel) in [(SortDir::Asc, "asc"), (SortDir::Desc, "desc")] {
            let mut rows: Vec<(TreeNode, usize)> =
                sort_sample().into_iter().map(|n| (n, 0)).collect();
            sort_rows(&mut rows, key, dir);
            let names: Vec<&str> = rows.iter().map(|(n, _)| n.name()).collect();
            orders.push(json!([format!("{label}.{dlabel}"), names]));
        }
    }
    json!({ "tensors": tensors, "orders": orders })
}

fn build() -> Value {
    let matcher = SkimMatcherV2::default();
    json!({
        "_note": "Generated by `cargo test --test parity` (UPDATE_PARITY=1 to rewrite). \
                  Verified against Rust there and against TypeScript in web/src/lib/parity.test.ts. \
                  See shared/parity/README.md.",
        "size": SIZES.iter().map(|&b| json!([b, format_size(b)])).collect::<Vec<_>>(),
        "count": COUNTS.iter().map(|&n| json!([n, format_parameters(n)])).collect::<Vec<_>>(),
        "percent": ZEROS
            .iter()
            .map(|&(zeros, count)| {
                #[allow(clippy::cast_precision_loss)] // display only, and the inputs are small
                let fraction = zeros as f64 / count as f64;
                json!([zeros, count, format_percent(fraction, zeros == 0)])
            })
            .collect::<Vec<_>>(),
        // The tree the two UIs flatten into rows. `tree` is the served hierarchy;
        // `rows` is what the Rust flattener makes of it, honouring each group's own
        // `expanded` flag — the initial fold state. `web/src/lib/parity.test.ts`
        // flattens the same `tree` and must produce the same `rows`.
        "tree": {
            "nodes": sample_tree(),
            "rows": TreeBuilder::flatten_tree(&sample_tree())
                .iter()
                .map(|(node, depth)| row_projection(node, *depth))
                .collect::<Vec<_>>(),
        },
        "sort": sort_section(),
        "search": {
            "names": NAMES,
            // Which names each query matches. Only the SET is a contract: the two
            // matchers rank differently (see the README), so the names are sorted.
            "matches": QUERIES
                .iter()
                .map(|&q| {
                    let mut hits: Vec<&str> = NAMES
                        .iter()
                        .copied()
                        .filter(|n| matcher.fuzzy_match(n, q).is_some())
                        .collect();
                    hits.sort_unstable();
                    json!([q, hits])
                })
                .collect::<Vec<_>>(),
        },
    })
}

#[test]
fn the_parity_fixture_still_matches_the_rust_side() {
    let built = build();
    let path = fixture_path();
    let pretty = format!("{}\n", serde_json::to_string_pretty(&built).unwrap());

    if std::env::var_os("UPDATE_PARITY").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &pretty).unwrap();
        eprintln!("wrote {}", path.display());
        return;
    }

    let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}\nregenerate with `UPDATE_PARITY=1 cargo test --test parity`",
            path.display()
        )
    });
    let on_disk: Value = serde_json::from_str(&on_disk).expect("fixture is not valid JSON");
    assert_eq!(
        on_disk, built,
        "the Rust formatters no longer agree with shared/parity/format.json.\n\
         If that change was intentional, regenerate with \
         `UPDATE_PARITY=1 cargo test --test parity` — and expect \
         web/src/lib/parity.test.ts to fail until the TypeScript matches."
    );
}

/// The properties the table above is only a sample of. Cheap, and they catch a
/// formatter that happens to agree on the sampled inputs while being wrong.
#[test]
fn sizes_and_counts_stay_within_their_units() {
    for bytes in [0usize, 1, 1023] {
        assert!(
            format_size(bytes).ends_with(" B"),
            "{bytes} should stay in bytes"
        );
    }
    // Under 1024 of a unit, the number part never reaches 1024.
    for shift in 10..=50 {
        let v = 1usize << shift;
        let s = format_size(v);
        let num: f64 = s.split(' ').next().unwrap().parse().unwrap();
        assert!((1.0..1024.0).contains(&num), "{v} → {s}");
    }
    assert_eq!(format_parameters(999), "999", "no unit below 1000");
    for pow in 3..=15u32 {
        let n = 10usize.pow(pow);
        let s = format_parameters(n);
        assert!(
            s.ends_with('K') || s.ends_with('M') || s.ends_with('B') || s.ends_with('T'),
            "{n} → {s}"
        );
    }
}
