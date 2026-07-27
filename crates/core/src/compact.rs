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

use std::collections::{BTreeMap, HashMap};

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

/// Group tensors into families of **contiguous** index runs, rather than one family per
/// index template regardless of gaps.
///
/// Why: Kimi-K3 alternates three KDA layers then one gated-MLA layer, ninety-three times.
/// Templating alone puts all 69 KDA layers in one family, whose label then enumerates the
/// set — `{0-2,4-6,8-10,…,88-90}`, three hundred characters the terminal truncates anyway,
/// and the *pattern* is invisible. Split on the outermost index's contiguous runs and the
/// same information reads as `{0-2}`, `{4-6}`, `{8-10}` … where the alternation is the
/// obvious thing on screen.
///
/// Only the outermost index splits. An inner one — the expert id in
/// `layers.{L}.…experts.{E}.…` — stays merged, because 896 experts present in every layer
/// is uniformity, not a pattern worth spelling out.
///
/// Implemented by partitioning the tensors so that each bucket's outermost index *is*
/// contiguous and then running the shared [`tensor_families`] over each bucket. The
/// aggregation (counts, uniform dtype/shape, rolled-up params and bytes) therefore has one
/// definition, and `diff`'s own summaries — where merging a scattered set is the right
/// answer — keep the behaviour they had.
fn contiguous_families(tensors: &[TensorInfo]) -> Vec<TensorFamily> {
    // Which outermost-index values each template actually has.
    let mut values: HashMap<String, Vec<u64>> = HashMap::new();
    for t in tensors {
        let (template, idx) = templatize(&t.name);
        if let Some(first) = idx.first().and_then(|v| v.parse::<u64>().ok()) {
            values.entry(template).or_default().push(first);
        }
    }
    // The contiguous runs of those values, as (start, end) inclusive.
    let runs: HashMap<String, Vec<(u64, u64)>> = values
        .into_iter()
        .map(|(template, mut vs)| {
            vs.sort_unstable();
            vs.dedup();
            let mut out: Vec<(u64, u64)> = Vec::new();
            for v in vs {
                match out.last_mut() {
                    // Extend the current run, including a repeat of its end.
                    Some(last) if v == last.1 + 1 || v == last.1 => last.1 = v,
                    _ => out.push((v, v)),
                }
            }
            (template, out)
        })
        .collect();

    // Bucket the tensors by (template, which run their outermost index falls in), keeping
    // first-appearance order so the tree reads in checkpoint order.
    let mut order: Vec<(String, usize)> = Vec::new();
    let mut buckets: HashMap<(String, usize), Vec<TensorInfo>> = HashMap::new();
    for t in tensors {
        let (template, idx) = templatize(&t.name);
        let run = idx
            .first()
            .and_then(|v| v.parse::<u64>().ok())
            .and_then(|first| {
                runs.get(&template)?
                    .iter()
                    .position(|&(lo, hi)| first >= lo && first <= hi)
            })
            // No index (or an unparseable one): one bucket for the template.
            .unwrap_or(usize::MAX);
        let key = (template, run);
        if !buckets.contains_key(&key) {
            order.push(key.clone());
        }
        buckets.entry(key).or_default().push(t.clone());
    }

    // One bucket has one contiguous run, so `tensor_families` yields one family from it —
    // with a label naming just that run.
    order
        .iter()
        .filter_map(|k| buckets.remove(k))
        .flat_map(|bucket| tensor_families(&bucket))
        .collect()
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

    /// The reason the view exists: a tensor present in only one layer does NOT fold in
    /// with its 48 uniform siblings, so it stands out as its own leaf.
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

        assert_eq!(c.tensor_count, 5);
        let names: Vec<&String> = c.counts.keys().collect();
        assert_eq!(
            names.len(),
            2,
            "the odd one out is its own family: {names:?}"
        );
        let odd = c
            .counts
            .iter()
            .find(|&(_, n)| *n == 1)
            .expect("the single-member family");
        assert!(
            odd.0.contains("extra_bias"),
            "the irregularity should be the lone family: {:?}",
            c.counts
        );
        // And it names the actual layer rather than a range, which is what makes it read
        // as an exception.
        assert!(
            odd.0.contains(".2.") || odd.0.contains("{2}"),
            "the lone family should name its layer: {}",
            odd.0
        );
    }

    /// A family whose members disagree about dtype is flagged rather than shown with one
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
