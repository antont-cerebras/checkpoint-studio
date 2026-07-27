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

use crate::diff::{TensorFamily, tensor_families};
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

/// Fold `tensors` into the compact tree. Pure; no disk.
///
/// `tensors` should be the canonical (deduped, natural-sorted) list, as every other view
/// takes — families come out in first-appearance order, which is alphabetical for sorted
/// input.
#[must_use]
pub fn compact_tree(tensors: &[TensorInfo]) -> CompactTree {
    let families = tensor_families(tensors);

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

    CompactTree {
        tree: TreeBuilder::build_tree(&synthetic),
        counts,
        varying,
        tensor_count: families.iter().map(|f| f.count).sum(),
    }
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
pub fn label_counts(tree: &mut [TreeNode], counts: &BTreeMap<String, usize>) {
    for node in tree {
        match node {
            TreeNode::Group { children, .. } => label_counts(children, counts),
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
                    .rsplit_once(" ×")
                    .filter(|(_, n)| !n.is_empty() && n.chars().all(char::is_numeric))
                    .map_or(base.as_str(), |(head, _)| head);
                *label = Some(format!("{base} ×{n}"));
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
        label_counts(&mut c.tree, &c.counts);

        let first = labels(&c.tree);
        assert_eq!(first.len(), 1, "one family: {first:?}");
        assert!(
            first[0].ends_with("×3"),
            "the label should carry the count: {:?}",
            first[0]
        );
        // Idempotent: running it twice must not produce `w ×3 ×3`.
        label_counts(&mut c.tree, &c.counts);
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
