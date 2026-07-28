//! The **compact tree**: the tensor tree with uniform layers and experts folded away,
//! so what remains visible is the irregularities.
//!
//! A 48-layer model has 48 structurally identical subtrees. Reading them one after
//! another tells you nothing you didn't learn from the first, and it buries the thing you
//! actually want to notice — the layer that has an extra tensor, or a different dtype, or
//! a shape that doesn't match its siblings. Folding the uniform stacks into one templated
//! subtree (`layers.{0-47}`) leaves the outliers standing on their own, where the eye
//! finds them. That makes this a structure-comprehension view, and a visual diff.
//!
//! **It is a tree, not a list.** This is the point: the hierarchy is what makes an outlier
//! stand out, so the grouping here is the *same* grouping the tensor tree uses — literally
//! [`TreeBuilder::build_tree`], fed the templated names. A flat list of families destroys
//! the nesting and with it the ability to see that one layer differs from its neighbours.
//!
//! The folding itself is [`crate::diff::tensor_families`] — the same collapsing `diff`
//! uses for its own summaries, so a family means one thing across the tool.

use std::collections::BTreeMap;

use crate::diff::{TensorFamily, templatize, tensor_families};
use crate::tree::{Layout, Storage, TensorInfo, TreeBuilder, TreeNode};

/// The tensor tree with uniform index families folded into single subtrees.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CompactTree {
    /// The tree, grouped exactly as the tensor tree groups real names — but built from
    /// *templated* names, so a uniform stack of layers is one subtree. Each
    /// [`TreeNode::Tensor`] is a **family**: its `info.name` is the template, and its
    /// dtype / shape / size / element count are the family's rollup (the dtype and shape
    /// only when uniform across the family — see [`CompactTree::counts`] and
    /// [`TensorFamily`]).
    pub tree: Vec<TreeNode>,
    /// How many real tensors each family stands for, keyed by the templated name that
    /// appears as a leaf's `info.name`. A frontend shows this as the `×48`; it is a
    /// separate map because a family is not a tensor and `TensorInfo` has nowhere honest
    /// to put a member count.
    pub counts: BTreeMap<String, usize>,
    /// Families whose dtype or shape is **not** uniform across their members, keyed the
    /// same way. These are irregularities the fold would otherwise hide: 48 layers that
    /// look like one family but disagree about dtype. A frontend should mark them.
    pub varying: BTreeMap<String, Varying>,
    /// Total real tensors folded into this tree — what the header counts, so the view can
    /// say "192 tensors in 4 families" and be checked against the tree itself.
    pub tensor_count: usize,
}

/// Which of a family's attributes disagree across its members.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Varying {
    pub dtype: bool,
    pub shape: bool,
}

/// Partition tensors into families by **contiguous runs of like layers**.
///
/// The rule this enforces: reading down the tree, layer numbers only ever increase. That
/// is not a sorting property — it constrains how families are formed. Grouping per tensor
/// *template* (the obvious approach, and the one this replaces) puts `self_attn.q_proj` in
/// the 69 layers that have it and `input_layernorm` in all 93, producing `{0-2}` beside
/// `{0-92}`: overlapping sets, so layer 0 appears twice and no ordering of siblings can
/// make the numbers monotonic.
///
/// So the grouping happens at the **layer** level. Each layer index gets a *signature* —
/// the set of tensor templates it has, which is exactly what distinguishes a KDA layer from
/// a gated-MLA one — and consecutive layers with an identical signature form a run. Runs
/// therefore *partition* the index space: `{0-2}`, `{3}`, `{4-6}`, `{7}`, … with every
/// layer in exactly one group, and Kimi-K3's three-then-one alternation becomes the shape
/// of the screen instead of a 300-character label.
///
/// A consequence worth knowing: a tensor present in every layer (`input_layernorm`) now
/// appears once per run at `×run-length` rather than once at `×93`. That is the honest
/// reading — those really are separate layers — and it is what keeps the runs disjoint.
///
/// "Layer" means the **outermost** index. An inner one (the expert id in
/// `layers.{L}.…experts.{E}.…`) is left merged: 896 experts present in every layer is
/// uniformity, and splitting there would produce 896 rows saying the same thing.
///
/// Each bucket is then handed to the shared [`tensor_families`], so the aggregation —
/// counts, uniform dtype/shape, rolled-up params and bytes — keeps one definition, and
/// `diff`'s summaries keep the behaviour they had.
fn contiguous_families(tensors: &[TensorInfo]) -> Vec<TensorFamily> {
    use std::collections::{BTreeSet, HashMap};

    // Per index AXIS, what each index value contains. The axis is the template up to and
    // including its first placeholder (`model.layers.{}`), so a model's layers and a vision
    // tower's layers are partitioned separately rather than conflated by sharing a number.
    let mut axes: HashMap<String, BTreeMap<u64, BTreeSet<String>>> = HashMap::new();
    for t in tensors {
        let (template, idx) = templatize(&t.name);
        let Some(axis) = axis_of(&template) else {
            continue;
        };
        let Some(index) = idx.first().and_then(|v| v.parse::<u64>().ok()) else {
            continue;
        };
        axes.entry(axis)
            .or_default()
            .entry(index)
            .or_default()
            .insert(template);
    }

    // Contiguous runs of equal signature, as inclusive (start, end).
    let runs: HashMap<String, Vec<(u64, u64)>> = axes
        .into_iter()
        .map(|(axis, signatures)| {
            let mut out: Vec<(u64, u64)> = Vec::new();
            let mut current: Option<BTreeSet<String>> = None;
            for (index, signature) in signatures {
                let extends = matches!(out.last(), Some(&(_, end)) if index == end + 1)
                    && current.as_ref() == Some(&signature);
                match out.last_mut() {
                    Some(last) if extends => last.1 = index,
                    _ => {
                        out.push((index, index));
                        current = Some(signature);
                    }
                }
            }
            (axis, out)
        })
        .collect();

    // Bucket every tensor by (axis, which run its index falls in), in first-appearance
    // order. A tensor with no index is its own bucket, keyed by its template.
    let mut order: Vec<(String, usize)> = Vec::new();
    let mut buckets: HashMap<(String, usize), Vec<TensorInfo>> = HashMap::new();
    for t in tensors {
        let (template, idx) = templatize(&t.name);
        let key = match (
            axis_of(&template),
            idx.first().and_then(|v| v.parse::<u64>().ok()),
        ) {
            (Some(axis), Some(index)) => {
                let run = runs
                    .get(&axis)
                    .and_then(|rs| rs.iter().position(|&(lo, hi)| index >= lo && index <= hi))
                    .unwrap_or(0);
                (axis, run)
            }
            // No index: nothing to partition, so the template is the whole bucket.
            _ => (template, usize::MAX),
        };
        if !buckets.contains_key(&key) {
            order.push(key.clone());
        }
        buckets.entry(key).or_default().push(t.clone());
    }

    order
        .iter()
        .filter_map(|k| buckets.remove(k).map(|b| (k.clone(), b)))
        .flat_map(|((axis, run), bucket)| {
            // The run this bucket is, so a single-layer run can be braced like the ranges.
            let span = runs.get(&axis).and_then(|rs| rs.get(run)).copied();
            tensor_families(&bucket)
                .into_iter()
                .map(move |f| brace_single(f, &axis, span))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Render a **single**-layer run as `{3}` rather than bare `3`, so it has the same shape as
/// its `{4-6}` siblings.
///
/// Why this matters beyond looks: the tree sorts siblings by name, and a bare digit and a
/// `{` compare differently, so the two forms landed in separate blocks — every range first,
/// then every single layer, putting layer 0 after layer 92 and breaking the
/// only-ever-increasing rule the runs themselves satisfy.
///
/// Done here, on the family name, because that name is also the `counts` / `varying` key and
/// what the tree builder groups on — they must all be the one string. `diff`'s own summaries
/// keep the bare form, where `layers.3.foo` reads better than `layers.{3}.foo`.
fn brace_single(mut family: TensorFamily, axis: &str, span: Option<(u64, u64)>) -> TensorFamily {
    let Some((lo, hi)) = span else { return family };
    if lo != hi {
        return family; // already `{lo-hi}`
    }
    // `axis` ends in the placeholder, so everything before it is the literal prefix the
    // index follows.
    let Some(prefix) = axis.strip_suffix("{}") else {
        return family;
    };
    let index = lo.to_string();
    let head = format!("{prefix}{index}");
    if let Some(rest) = family.name.strip_prefix(&head) {
        family.name = format!("{prefix}{{{index}}}{rest}");
    }
    family
}

/// The index axis a template belongs to: everything up to and including its first `{}`
/// placeholder (`model.layers.{}`), or `None` for a template with no index.
///
/// Two templates share an axis exactly when they are indexed by the same thing — which is
/// what makes "layer 3 of the language model" and "layer 3 of the vision tower" different
/// layers rather than one.
fn axis_of(template: &str) -> Option<String> {
    let at = template.find("{}")?;
    Some(template[..at + 2].to_string())
}

/// Fold `tensors` into the compact tree. Pure; no disk.
///
/// `tensors` should be the canonical (deduped, natural-sorted) list, as every other view
/// takes — families come out in first-appearance order, which is alphabetical for sorted
/// input.
#[must_use]
pub fn compact_tree(tensors: &[TensorInfo]) -> CompactTree {
    let families = contiguous_families(tensors);

    let mut counts = BTreeMap::new();
    let mut varying = BTreeMap::new();
    // One synthetic tensor per family, carrying the family's rollup, so the shared tree
    // builder can group them: it only reads `name`, `dtype`, `shape`, `size_bytes` and
    // `num_elements`, all of which a family has.
    let synthetic: Vec<TensorInfo> = families
        .iter()
        .map(|f| {
            counts.insert(f.name.clone(), f.count);
            if f.dtype.is_none() || f.shape.is_none() {
                varying.insert(
                    f.name.clone(),
                    Varying {
                        dtype: f.dtype.is_none(),
                        shape: f.shape.is_none(),
                    },
                );
            }
            family_as_tensor(f)
        })
        .collect();

    let mut tree = TreeBuilder::build_tree(&synthetic);
    // The builder counted the synthetic tensors — i.e. families. Restate every group's
    // count in REAL tensors, because `▦` has to mean the same thing on a group as it does
    // on the root: before this, a folded `language_model` read `▦ 44` under a root reading
    // `▦ 497220`, which invites the reader to think 497k tensors went missing.
    recount_groups(&mut tree, &counts);
    CompactTree {
        tree,
        counts,
        varying,
        tensor_count: families.iter().map(|f| f.count).sum(),
    }
}

/// Restate each group's `tensor_count` as the number of real tensors its families stand
/// for, returning that number so parents can sum it. Sizes and parameter counts are
/// already real (they come from the family rollups); only the count was of families.
fn recount_groups(nodes: &mut [TreeNode], counts: &BTreeMap<String, usize>) -> usize {
    let mut total = 0;
    for node in nodes {
        total += match node {
            TreeNode::Group {
                children,
                tensor_count,
                ..
            } => {
                let n = recount_groups(children, counts);
                *tensor_count = n;
                n
            }
            TreeNode::Tensor { info, .. } => counts.get(&info.name).copied().unwrap_or(1),
            TreeNode::Metadata { .. } => 0,
        };
    }
    total
}

/// A family as the one "tensor" that stands for it. `dtype` is `"varies"` and `shape`
/// empty when the members disagree — the fields are non-optional on [`TensorInfo`], and a
/// visible `varies` is better than silently showing one member's value as if it were the
/// family's (`CompactTree::varying` carries the same fact in a form code can branch on).
fn family_as_tensor(f: &TensorFamily) -> TensorInfo {
    TensorInfo {
        name: f.name.clone(),
        dtype: f.dtype.clone().unwrap_or_else(|| "varies".to_string()),
        shape: f.shape.clone().unwrap_or_default(),
        size_bytes: f.size_bytes,
        num_elements: f.params,
        storage: Storage::Unknown,
        // A family spans many tensors in (possibly) many files, so it has no one source
        // or byte range. `Layout::None` is exactly "not tracked".
        source_path: String::new(),
        layout: Layout::None,
    }
}

/// [`compact_tree`], rooted the way every frontend's tree is rooted — so the compact view
/// has the same header row as the tree view it replaces, counting the *real* tensors above
/// a body of families. `files` names the root ([`crate::model::root_label`]).
#[must_use]
pub fn compact_rooted(tensors: &[TensorInfo], files: &[std::path::PathBuf]) -> CompactTree {
    let folded = compact_tree(tensors);
    CompactTree {
        tree: vec![crate::tree::root_group(
            crate::model::root_label(files),
            folded.tree,
            tensors,
        )],
        ..folded
    }
}

/// Fold each family's member count into its display label — `down_proj.weight ×48`.
///
/// For a frontend that renders one label per row and has nowhere to put a separate count
/// column: the terminal. The browser reads [`CompactTree::counts`] instead and styles the
/// multiplier itself. Both show the same fact; only the presentation differs, which is the
/// kind of difference `shared/parity/README.md` records as deliberate.
pub fn label_counts(
    tree: &mut [TreeNode],
    counts: &BTreeMap<String, usize>,
    varying: &BTreeMap<String, Varying>,
) {
    for node in tree {
        match node {
            TreeNode::Group { children, .. } => label_counts(children, counts, varying),
            TreeNode::Tensor { info, label } => {
                let Some(n) = counts.get(&info.name) else {
                    continue;
                };
                // The label the tree builder already chose (a compacted chain), else the
                // last dotted segment — the same text the row would otherwise show.
                let base = label.clone().unwrap_or_else(|| {
                    info.name
                        .rsplit('.')
                        .next()
                        .unwrap_or(&info.name)
                        .to_string()
                });
                // Idempotent: drop a suffix we already added, so calling this twice (a
                // rebuild that re-labels an already-labelled tree) can't produce `w ×3 ×3`.
                let base = base
                    .split_once(" ×")
                    .map_or(base.as_str(), |(head, _)| head);
                // A family whose members disagree gets told so on the row. Without this the
                // renderer prints the *default* shape — `()` — which reads as a
                // zero-dimensional tensor rather than "these differ".
                let note = match varying.get(&info.name) {
                    Some(v) if v.dtype && v.shape => " (dtype and shape vary)",
                    Some(v) if v.shape => " (shape varies)",
                    Some(_) => "",
                    None => "",
                };
                *label = Some(format!("{base} ×{n}{note}"));
            }
            TreeNode::Metadata { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, dtype: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
            size_bytes: shape.iter().product::<usize>() * 2,
            num_elements: shape.iter().product(),
            storage: Storage::Unknown,
            source_path: "model.safetensors".to_string(),
            layout: Layout::None,
        }
    }

    /// A uniform stack folds to one subtree, and the tensor count is preserved — the
    /// whole point of the view: three layers of identical structure read as one.
    #[test]
    fn a_uniform_layer_stack_folds_into_one_subtree() {
        let mut tensors = Vec::new();
        for layer in 0..3 {
            tensors.push(tensor(
                &format!("model.layers.{layer}.mlp.down_proj.weight"),
                "BF16",
                &[8, 4],
            ));
            tensors.push(tensor(
                &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                "BF16",
                &[4, 4],
            ));
        }
        let c = compact_tree(&tensors);

        assert_eq!(c.tensor_count, 6, "every real tensor is accounted for");
        // Two families (down_proj, q_proj), each standing for three tensors.
        assert_eq!(c.counts.len(), 2, "families: {:?}", c.counts.keys());
        for (name, count) in &c.counts {
            assert_eq!(*count, 3, "{name} should stand for three layers");
            assert!(
                name.contains("{0-2}"),
                "the layer index is templated: {name}"
            );
        }
        assert!(c.varying.is_empty(), "a uniform stack varies in nothing");
        // Rolled-up bytes equal the sum of the members'.
        let leaf_bytes: usize = leaves(&c.tree).iter().map(|t| t.size_bytes).sum();
        assert_eq!(
            leaf_bytes,
            tensors.iter().map(|t| t.size_bytes).sum::<usize>()
        );
    }

    /// The reason the view exists: the layer that differs is isolated. Layer 2 has an extra
    /// tensor, so it becomes its own run — `{0-1}`, `{2}`, `{3}` — and the exception is a
    /// row of its own rather than being averaged into a family spanning all four layers.
    #[test]
    fn an_irregularity_stays_visible_beside_the_folded_stack() {
        let mut tensors = Vec::new();
        for layer in 0..4 {
            tensors.push(tensor(
                &format!("model.layers.{layer}.mlp.down_proj.weight"),
                "BF16",
                &[8, 4],
            ));
        }
        // Only layer 2 has this one.
        tensors.push(tensor("model.layers.2.mlp.extra_bias", "F32", &[8]));
        let c = compact_tree(&tensors);

        assert_eq!(c.tensor_count, 5, "every tensor is still accounted for");
        // The uniform pair folds; the odd layer and the layer after it stand alone.
        let names: Vec<&String> = c.counts.keys().collect();
        assert!(
            names.iter().any(|n| n.contains("{0-1}")),
            "layers 0 and 1 are alike and fold: {names:?}"
        );
        assert!(
            names.iter().any(|n| n.contains("extra_bias")),
            "the odd tensor gets its own family: {names:?}"
        );
        assert!(
            names.iter().any(|n| n.contains("layers.{2}.")),
            "layer 2 is named on its own — braced like every other run, so the siblings              sort together: {names:?}"
        );
    }

    /// **The rule**: reading down the tree, layer numbers only increase. That is a property
    /// of how families are formed, not of how siblings are sorted — grouping per tensor
    /// template produced `{0-2}` beside `{0-92}`, overlapping sets in which layer 0 appears
    /// twice, and no ordering could have fixed it. This asserts the runs *partition* the
    /// layer space, which is what makes monotonicity achievable at all.
    #[test]
    fn layer_runs_partition_the_layers_so_indices_only_increase() {
        // Kimi-K3's shape in miniature: three "KDA" layers then one "MLA" layer, twice
        // over, with two tensors present in every layer.
        let mut tensors = Vec::new();
        for layer in 0..8 {
            tensors.push(tensor(
                &format!("model.layers.{layer}.input_layernorm.weight"),
                "BF16",
                &[8],
            ));
            if layer % 4 == 3 {
                tensors.push(tensor(
                    &format!("model.layers.{layer}.self_attn.g_proj.weight"),
                    "BF16",
                    &[8, 8],
                ));
            } else {
                tensors.push(tensor(
                    &format!("model.layers.{layer}.self_attn.A_log"),
                    "F32",
                    &[4],
                ));
            }
        }
        let c = compact_tree(&tensors);
        assert_eq!(c.tensor_count, 16);

        // Every family's layer span, in the order the names sort (which is the order the
        // tree renders them).
        let mut spans: Vec<(u64, u64)> = c
            .counts
            .keys()
            .filter_map(|n| layer_span(n))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        spans.sort_unstable();
        assert!(!spans.is_empty(), "families: {:?}", c.counts.keys());

        // No two runs overlap, and together they cover 0..=7 exactly once — the property
        // that makes "layer numbers only increase" possible.
        for pair in spans.windows(2) {
            assert!(
                pair[0].1 < pair[1].0,
                "runs must not overlap: {:?} then {:?} (all: {spans:?})",
                pair[0],
                pair[1]
            );
        }
        assert_eq!(spans.first().map(|s| s.0), Some(0));
        assert_eq!(spans.last().map(|s| s.1), Some(7));
        // The 3:1 alternation shows up as runs of three then one.
        assert!(
            spans.iter().any(|&(a, b)| b - a == 2),
            "a three-layer run should exist: {spans:?}"
        );
        assert!(
            spans.iter().any(|&(a, b)| a == b),
            "a single-layer run should exist: {spans:?}"
        );
    }

    /// The layer range a templated family name spans, from its `layers.N` or `layers.{A-B}`.
    fn layer_span(name: &str) -> Option<(u64, u64)> {
        let rest = name.split("layers.").nth(1)?;
        let token: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || matches!(c, '{' | '}' | '-' | ','))
            .collect();
        let inner = token.trim_matches(|c| c == '{' || c == '}');
        let (lo, hi) = inner.split_once('-').unwrap_or((inner, inner));
        Some((lo.parse().ok()?, hi.parse().ok()?))
    }

    /// A family whose members disagree about dtype is flagged    /// A family whose members disagree about dtype is flagged rather than shown with one
    /// member's dtype standing in for all of them.
    #[test]
    fn a_family_that_disagrees_about_dtype_is_marked_as_varying() {
        let tensors = vec![
            tensor("model.layers.0.mlp.w", "BF16", &[4, 4]),
            tensor("model.layers.1.mlp.w", "F32", &[4, 4]),
        ];
        let c = compact_tree(&tensors);
        assert_eq!(c.counts.len(), 1, "one family");
        let (name, _) = c.counts.iter().next().unwrap();
        let v = c
            .varying
            .get(name)
            .unwrap_or_else(|| panic!("{name} should be marked varying: {:?}", c.varying));
        assert!(v.dtype, "the dtypes differ");
        assert!(!v.shape, "the shapes do not");
        // The leaf says so in a way a renderer can just print.
        let leaf = leaves(&c.tree)
            .into_iter()
            .next()
            .expect("one leaf for one family");
        assert_eq!(leaf.dtype, "varies");
    }

    /// An empty checkpoint gives an empty tree rather than a tree with an empty family.
    #[test]
    fn no_tensors_folds_to_nothing() {
        let c = compact_tree(&[]);
        assert!(c.tree.is_empty());
        assert!(c.counts.is_empty());
        assert_eq!(c.tensor_count, 0);
    }

    /// The count ends up on the row's label, which is what a terminal renders.
    #[test]
    fn labels_carry_the_member_count() {
        let tensors: Vec<TensorInfo> = (0..3)
            .map(|i| tensor(&format!("model.layers.{i}.mlp.w"), "BF16", &[4, 4]))
            .collect();
        let mut c = compact_tree(&tensors);
        label_counts(&mut c.tree, &c.counts, &c.varying);

        let first = labels(&c.tree);
        assert_eq!(first.len(), 1, "one family: {first:?}");
        assert!(
            first[0].ends_with("×3"),
            "the label should carry the count: {:?}",
            first[0]
        );
        // Idempotent: running it twice must not produce `w ×3 ×3`.
        label_counts(&mut c.tree, &c.counts, &c.varying);
        assert_eq!(labels(&c.tree), first, "labelling twice must not stack");
    }

    /// Every leaf label, depth-first.
    fn labels(nodes: &[TreeNode]) -> Vec<String> {
        let mut out = Vec::new();
        for n in nodes {
            match n {
                TreeNode::Group { children, .. } => out.extend(labels(children)),
                TreeNode::Tensor { info, label } => {
                    out.push(label.clone().unwrap_or_else(|| info.name.clone()));
                }
                TreeNode::Metadata { .. } => {}
            }
        }
        out
    }

    /// Every leaf of the tree, depth-first.
    fn leaves(nodes: &[TreeNode]) -> Vec<TensorInfo> {
        let mut out = Vec::new();
        for n in nodes {
            match n {
                TreeNode::Group { children, .. } => out.extend(leaves(children)),
                TreeNode::Tensor { info, .. } => out.push(info.clone()),
                TreeNode::Metadata { .. } => {}
            }
        }
        out
    }
}

#[cfg(test)]
mod count_tests {
    use super::*;

    fn t(name: &str, dtype: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
            size_bytes: shape.iter().product::<usize>() * 2,
            num_elements: shape.iter().product(),
            storage: Storage::Unknown,
            source_path: "m.safetensors".to_string(),
            layout: Layout::None,
        }
    }

    /// `▦` must mean **real tensors** on every node. It used to count *families* on groups
    /// while the root counted tensors, so a folded `language_model` read `▦ 44` directly
    /// under a root reading `▦ 497220` — inviting the reader to conclude that folding had
    /// lost half a million tensors.
    #[test]
    fn group_counts_stay_real_tensor_counts() {
        let mut tensors = Vec::new();
        for layer in 0..10 {
            tensors.push(t(&format!("model.layers.{layer}.mlp.w"), "BF16", &[4, 4]));
            tensors.push(t(&format!("model.layers.{layer}.attn.w"), "BF16", &[4, 4]));
        }
        let c = compact_tree(&tensors);
        assert_eq!(c.tensor_count, 20);
        // Two families, but the group above them stands for all twenty tensors.
        assert_eq!(c.counts.len(), 2, "families: {:?}", c.counts.keys());
        let counts = group_counts(&c.tree);
        assert!(
            counts.iter().any(|&(_, n)| n == 20),
            "some group should account for all 20 real tensors: {counts:?}"
        );
        assert!(
            !counts.iter().any(|&(_, n)| n == 2),
            "no group should report the family count as its tensor count: {counts:?}"
        );
        // And every group's count is the sum of its children's.
        assert_eq!(
            counts.iter().map(|&(_, n)| n).max(),
            Some(20),
            "the outermost group covers everything: {counts:?}"
        );
    }

    /// A family whose members disagree about shape must say so. It rendered as `()` —
    /// the *default* shape — which reads as a zero-dimensional tensor rather than
    /// "these differ". Reported from the Kimi-K3 tree: `proj.{0,2}.weight ×2 [BF16, ()]`.
    /// A NON-contiguous pair splits, so each row shows its own real shape — which is how
    /// the reported `proj.{0,2}.weight ×2 [BF16, ()]` stops existing: those two are indices
    /// 0 and 2, not a run, so they were never one family's worth of information.
    #[test]
    fn a_non_contiguous_pair_splits_and_keeps_real_shapes() {
        let tensors = vec![
            t("mm_projector.proj.0.weight", "BF16", &[7168, 1024]),
            t("mm_projector.proj.2.weight", "BF16", &[1024, 7168, 2]),
        ];
        let c = compact_tree(&tensors);
        assert_eq!(
            c.counts.len(),
            2,
            "indices 0 and 2 are not a run: {:?}",
            c.counts.keys()
        );
        assert!(
            c.varying.is_empty(),
            "a one-member family agrees with itself: {:?}",
            c.varying
        );
        for n in c.counts.values() {
            assert_eq!(*n, 1, "each split family stands for one tensor");
        }
    }

    /// A *contiguous* family whose members disagree about shape still has to say so — that
    /// is the case `()` was hiding (247,296 packed weights across contiguous experts).
    #[test]
    fn a_varying_shape_says_so_on_the_row() {
        let tensors = vec![
            t("mm_projector.proj.0.weight", "BF16", &[7168, 1024]),
            t("mm_projector.proj.1.weight", "BF16", &[1024, 7168, 2]),
        ];
        let mut c = compact_tree(&tensors);
        label_counts(&mut c.tree, &c.counts, &c.varying);

        let (name, _) = c.counts.iter().next().expect("one family");
        assert!(
            c.varying.get(name).is_some_and(|v| v.shape && !v.dtype),
            "the shapes differ but the dtypes don't: {:?}",
            c.varying
        );
        let label = leaf_label(&c.tree).expect("a labelled leaf");
        assert!(
            label.contains("shape varies"),
            "the row must say the shapes differ rather than printing `()`: {label}"
        );
        assert!(label.contains("×2"), "and still carry the count: {label}");

        // Idempotent even with the note appended.
        let once = label;
        label_counts(&mut c.tree, &c.counts, &c.varying);
        assert_eq!(leaf_label(&c.tree).as_deref(), Some(once.as_str()));
    }

    fn group_counts(nodes: &[TreeNode]) -> Vec<(String, usize)> {
        let mut out = Vec::new();
        for n in nodes {
            if let TreeNode::Group {
                name,
                children,
                tensor_count,
                ..
            } = n
            {
                out.push((name.clone(), *tensor_count));
                out.extend(group_counts(children));
            }
        }
        out
    }

    fn leaf_label(nodes: &[TreeNode]) -> Option<String> {
        for n in nodes {
            match n {
                TreeNode::Group { children, .. } => {
                    if let Some(l) = leaf_label(children) {
                        return Some(l);
                    }
                }
                TreeNode::Tensor { label, .. } => return label.clone(),
                TreeNode::Metadata { .. } => {}
            }
        }
        None
    }
}

#[cfg(test)]
mod order_tests {
    use super::*;

    fn t(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dtype: "BF16".to_string(),
            shape: vec![4],
            size_bytes: 8,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "m.safetensors".to_string(),
            layout: Layout::None,
        }
    }

    /// The reported break: every `{a-b}` range sorted before every bare single layer, so
    /// layer 0 came after layer 92. Single runs are braced now, so a mixed set of single and
    /// multi-layer runs renders in increasing order.
    ///
    /// This asserts what the earlier partition test did not — that the *rendered labels*
    /// sort monotonically, not merely that the runs are disjoint. That gap is what let the
    /// break reach a screen.
    #[test]
    fn single_and_multi_layer_runs_render_in_increasing_order() {
        // Layer 0 alone, then 1-3 alike, then 4 alone: a signature change at each boundary.
        let mut tensors = vec![t("model.layers.0.solo.weight")];
        for layer in 1..4 {
            tensors.push(t(&format!("model.layers.{layer}.shared.weight")));
        }
        tensors.push(t("model.layers.4.other.weight"));

        let c = compact_tree(&tensors);
        // Every family names its layer in braces, so the forms are uniform.
        for name in c.counts.keys() {
            let after = name.split("layers.").nth(1).unwrap_or_default();
            assert!(
                after.starts_with('{'),
                "every run should be braced for a uniform sort: {name}"
            );
        }

        // Sorted by name — which is how the tree orders siblings — the layer numbers only
        // increase.
        let mut names: Vec<&String> = c.counts.keys().collect();
        names.sort_by_key(|n| crate::tree::natural_sort_key(n));
        let firsts: Vec<u64> = names
            .iter()
            .filter_map(|n| {
                let after = n.split("layers.{").nth(1)?;
                after.split(['-', '}']).next()?.parse().ok()
            })
            .collect();
        assert_eq!(
            firsts.len(),
            names.len(),
            "each name yields a layer: {names:?}"
        );
        assert!(
            firsts.windows(2).all(|w| w[0] <= w[1]),
            "layer numbers must only increase: {firsts:?} from {names:?}"
        );
    }
}
