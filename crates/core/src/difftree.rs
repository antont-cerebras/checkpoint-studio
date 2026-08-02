//! **Two checkpoints, one tree** — the aligned model behind side-by-side comparison.
//!
//! [`crate::diff`] answers "what changed" as flat lists: names added, names removed, signatures
//! changed. That is the right shape for a report you read top to bottom, and the wrong shape for
//! *browsing* two checkpoints: a list cannot tell you where a change sits in the hierarchy, cannot
//! be folded, and cannot be scrolled in step with anything.
//!
//! So this walks both trees together and produces one tree of pairs. Every node knows which side
//! it exists on and whether it differs; every group knows how many differing tensors are somewhere
//! beneath it. Three things fall out of that and cost nothing extra:
//!
//! - **Side by side.** One row renders as two columns, and a row missing from one side renders as a
//!   gap there — so the two columns stay aligned without either frontend computing an alignment.
//! - **Lockstep.** There is one tree, so there is one fold state and one selection. Two independent
//!   trees would need their scroll positions reconciled after every expand, and would drift the
//!   moment one side had a group the other lacked.
//! - **Jump to the next difference.** A flat walk of this tree, skipping `Same`, in the order the
//!   rows are drawn.
//!
//! Both frontends render this rather than each aligning the trees itself: an alignment that
//! disagreed between the terminal and the browser would have them draw different comparisons of the
//! same two checkpoints.

use crate::tree::{TensorInfo, TreeNode};

/// How a node's two sides relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Status {
    /// Present on both sides and identical (for a group: identical throughout).
    Same,
    /// Present on both sides, but not the same — a differing dtype or shape, or a group with
    /// something differing inside it.
    Changed,
    /// Only in the baseline: removed by the newer checkpoint.
    OnlyOld,
    /// Only in the newer checkpoint: added.
    OnlyNew,
}

impl Status {
    /// Whether this row is one that "jump to the next difference" should stop on.
    #[must_use]
    pub fn differs(self) -> bool {
        !matches!(self, Self::Same)
    }
}

/// What one side of a row holds. `None` on the side a node is missing from.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Side {
    /// A group, with the totals its own tree carried — so a folded row can still say how big the
    /// subtree is on each side.
    Group {
        tensor_count: usize,
        params: usize,
        total_size: usize,
    },
    /// A tensor, with everything a detail view needs — and how many tensors it stands for, when it
    /// stands for more than itself.
    ///
    /// `fold` is `Some(256)` when an unfused/fused alignment folded 256 per-expert tensors onto the one
    /// fused tensor beside it ([`note_folds`]) — `None` on the ordinary one-to-one row. It belongs to
    /// the *side*, not the row: the fused side has one tensor where the unfused side has 256, and
    /// without it the view showed a shape that gained a leading dimension with nothing to explain why.
    ///
    /// A count rather than a label. It used to ride on [`TreeNode::Tensor`]'s `label`, which already
    /// means something else — the compacted display name of a collapsed single-child chain
    /// (`embed_tokens.weight`) — so every such row printed its own name a second time after its
    /// signature, and a row that really was folded lost its name to `×256`.
    Tensor {
        info: Box<TensorInfo>,
        fold: Option<usize>,
    },
    /// A metadata entry.
    Metadata { name: String, value: String },
}

/// One row of the aligned tree: the same name on both sides, or on only one.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AlignedNode {
    /// The name segment this row is keyed by — what makes the two sides "the same node".
    pub name: String,
    /// The full dotted name, for jumping to a tensor's detail view on either side.
    pub path: String,
    pub old: Option<Side>,
    pub new: Option<Side>,
    pub status: Status,
    /// Differing **tensors** anywhere in this subtree (itself included when it is a tensor).
    ///
    /// What a collapsed group shows — `layers.1 (3 differ)` — so you can tell whether folding hides
    /// anything before you open it. Counted in tensors rather than rows because a group differing
    /// only because its children do is not itself a difference worth counting.
    pub differing: usize,
    /// How many identical index-named sibling subtrees this row stands for: `1` ordinarily, `62` for a
    /// family folded by [`fold_families`] (`{0-61}`). The subtree below is one member — the template —
    /// so the numbers on it describe one layer, and this says how many there are.
    pub members: usize,
    pub children: Vec<Self>,
}

impl AlignedNode {
    /// Whether this row has children to fold. Cheaper for a renderer than testing the vector's
    /// length at every draw, and it says what it means.
    #[must_use]
    pub fn is_group(&self) -> bool {
        !self.children.is_empty()
    }
}

/// Align two trees into one.
///
/// `old` is the baseline, `new` the checkpoint being compared to it — the same orientation as
/// `diff OLD NEW` and [`crate::diff::compare`], so a side-by-side and a report of the same two
/// checkpoints cannot disagree about which one is "added".
#[must_use]
pub fn align(old: &[TreeNode], new: &[TreeNode]) -> Vec<AlignedNode> {
    // Both sides are re-nested to one segment per level *before* they are merged, and the merged tree
    // is re-collapsed afterwards. See [`expand_segments`]: without this, two checkpoints that grouped
    // the same tensor differently could not pair it.
    let rows = align_level(&expand_segments(old), &expand_segments(new), "");
    collapse_chains(rows)
}

/// Re-nest a group whose name carries several dotted segments: `layers.0` → `layers` → `0`.
///
/// **Why alignment cannot use the trees as given.** A group's name is *data-dependent*: the kernel
/// merges a chain of single-child groups onto one row, so `model.layers.0.input_layernorm.weight`
/// hangs under `layers.0` in a one-layer checkpoint and under `layers` → `0` in a 48-layer one. Those
/// two shapes cannot be merged level by level, so that tensor came out as **both** added and removed —
/// double-counted here, and missing from `changed`, which is exactly how this view and the one-page
/// report came to disagree by one tensor on a real pair.
///
/// Expanding is sound because a multi-segment name *means* a single-child chain: every intermediate
/// level has one child and the same subtree, so it carries the same counts.
fn expand_segments(nodes: &[TreeNode]) -> Vec<TreeNode> {
    let mut out = Vec::with_capacity(nodes.len());
    for n in nodes {
        let TreeNode::Group {
            name,
            children,
            expanded,
            tensor_count,
            params,
            total_size,
            stored_size,
        } = n
        else {
            out.push(n.clone());
            continue;
        };
        let mut segments: Vec<&str> = name.split('.').collect();
        // The innermost segment keeps the children; each outer one wraps what follows it.
        let last = segments.pop().unwrap_or(name.as_str());
        let mut node = TreeNode::Group {
            name: last.to_string(),
            children: expand_segments(children),
            expanded: *expanded,
            tensor_count: *tensor_count,
            params: *params,
            total_size: *total_size,
            stored_size: *stored_size,
        };
        for seg in segments.into_iter().rev() {
            node = TreeNode::Group {
                name: seg.to_string(),
                children: vec![node],
                expanded: *expanded,
                tensor_count: *tensor_count,
                params: *params,
                total_size: *total_size,
                stored_size: *stored_size,
            };
        }
        out.push(node);
    }
    out
}

/// Merge a group with exactly one child group back onto one row — the readability rule the kernel
/// applies, re-applied to the *merged* tree so both sides get it identically.
///
/// Safe on statuses: a group with one child spans the same sides as that child (a group present on a
/// side has at least one child there), so the two always agree and the child's — the deeper, more
/// specific one — is kept along with its path.
fn collapse_chains(rows: Vec<AlignedNode>) -> Vec<AlignedNode> {
    rows.into_iter()
        .map(|mut row| {
            // Children first, so a chain of three collapses whole: `a`→`b`→`c` becomes `b.c` here and
            // then `a.b.c` one level up.
            row.children = collapse_chains(row.children);
            let [only] = row.children.as_mut_slice() else {
                return row;
            };
            // A lone *leaf* stays under its group — a tensor is not a level of hierarchy to merge away.
            if only.children.is_empty() {
                return row;
            }
            let child = std::mem::replace(
                only,
                AlignedNode {
                    name: String::new(),
                    path: String::new(),
                    old: None,
                    new: None,
                    status: Status::Same,
                    differing: 0,
                    members: 1,
                    children: Vec::new(),
                },
            );
            AlignedNode {
                name: format!("{}.{}", row.name, child.name),
                path: child.path,
                old: row.old,
                new: row.new,
                status: child.status,
                differing: child.differing,
                members: child.members,
                children: child.children,
            }
        })
        .collect()
}

/// Build a tree from a flat tensor list, grouping on dotted name segments.
///
/// For comparing a side whose names have been **renamed**: rename rules exist so two naming schemes line
/// up, and rewriting a leaf's name inside an existing tree leaves the groups above it describing the old
/// one. Rebuilding from the names is the only way the structure can agree with them.
///
/// A plain segment trie, not the kernel's tree: no family folding, no compacted labels — [`align`]
/// expands to one segment per level and re-collapses afterwards anyway, so anything fancier here would
/// be undone. Metadata hangs under one group, as the kernel puts it.
/// **Rooted**, like the kernel's own tree: everything hangs under one group named after the checkpoint.
/// [`align_rooted`] unwraps a lone top-level group, so an unrooted tree here would have its *first real
/// level* eaten while the other side kept its — and nothing would pair. Not hypothetical: it is what
/// happened, and it is why a unit test that called [`align`] directly did not catch it.
#[must_use]
pub fn tree_from_tensors(
    root: &str,
    tensors: &[TensorInfo],
    metadata: &[crate::tree::MetadataInfo],
) -> Vec<TreeNode> {
    // An intermediate trie, because `TreeNode`'s children are a `Vec` and building in place would be
    // quadratic in the number of segments.
    #[derive(Default)]
    struct Node {
        children: std::collections::BTreeMap<String, Self>,
        tensor: Option<TensorInfo>,
    }
    fn build(name: &str, node: Node) -> TreeNode {
        if let Some(info) = node.tensor {
            return TreeNode::Tensor { info, label: None };
        }
        let mut children: Vec<TreeNode> = node
            .children
            .into_iter()
            .map(|(k, v)| build(&k, v))
            .collect();
        // A chain that ends in a lone tensor collapses onto one row, carrying the merged path as its
        // label — what the kernel's tree does, so `lm_head.weight` is a tensor row here too rather than
        // a `lm_head` group wrapping a `weight` leaf. Without this the two sides differ in *kind* at
        // that level and cannot pair, which is how a renamed tensor still read as added+removed.
        // Tested before taking, so a single *group* child is left where it is.
        if matches!(children.as_slice(), [TreeNode::Tensor { .. }])
            && let Some(TreeNode::Tensor { info, label }) = children.pop()
        {
            let leaf = label.unwrap_or_else(|| last_segment(&info.name).to_string());
            return TreeNode::Tensor {
                info,
                label: Some(format!("{name}.{leaf}")),
            };
        }
        let tensor_count = children
            .iter()
            .map(|c| match c {
                TreeNode::Tensor { .. } => 1,
                TreeNode::Group { tensor_count, .. } => *tensor_count,
                TreeNode::Metadata { .. } => 0,
            })
            .sum();
        TreeNode::Group {
            name: name.to_string(),
            children,
            expanded: false,
            tensor_count,
            // Sizes are not recomputed: a renamed side is compared, not browsed, and `align` reads them
            // only for a folded row's "N tensors" label.
            params: 0,
            total_size: 0,
            stored_size: 0,
        }
    }

    let mut trie = Node::default();
    for t in tensors {
        let mut at = &mut trie;
        let segments: Vec<&str> = t.name.split('.').collect();
        let Some((last, parents)) = segments.split_last() else {
            continue;
        };
        for seg in parents {
            at = at.children.entry((*seg).to_string()).or_default();
        }
        at.children.entry((*last).to_string()).or_default().tensor = Some(t.clone());
    }

    let mut out: Vec<TreeNode> = trie
        .children
        .into_iter()
        .map(|(k, v)| build(&k, v))
        .collect();
    if !metadata.is_empty() {
        out.push(TreeNode::Group {
            name: crate::tree::METADATA_GROUP.to_string(),
            children: metadata
                .iter()
                .map(|info| TreeNode::Metadata { info: info.clone() })
                .collect(),
            expanded: false,
            tensor_count: 0,
            params: 0,
            total_size: 0,
            stored_size: 0,
        });
    }
    let tensor_count = tensors.len();
    // The root the kernel wraps its tree in, so `align_rooted` unwraps one level on both sides.
    vec![TreeNode::Group {
        name: root.to_string(),
        children: out,
        expanded: true,
        tensor_count,
        params: 0,
        total_size: 0,
        stored_size: 0,
    }]
}

/// Keep only the tensors a predicate selects, dropping groups left empty.
///
/// How a *scoped* comparison is built: the CLI narrows a diff to a subset of tensors, and the aligned
/// tree has to be narrowed the same way or the two views describe different comparisons of one pair.
/// Pruning before alignment rather than hiding rows after it means every count, every `(N differ)`
/// badge and the difference list all describe the selected subset, with nothing left to remember.
///
/// `keep_metadata` false is `--only-tensors`.
#[must_use]
pub fn prune<F: Fn(&str) -> bool>(
    nodes: &[TreeNode],
    keep: &F,
    keep_metadata: bool,
) -> Vec<TreeNode> {
    let mut out = Vec::new();
    for n in nodes {
        match n {
            TreeNode::Tensor { info, .. } => {
                if keep(&info.name) {
                    out.push(n.clone());
                }
            }
            TreeNode::Metadata { .. } => {
                if keep_metadata {
                    out.push(n.clone());
                }
            }
            TreeNode::Group {
                name,
                children,
                expanded,
                tensor_count,
                params,
                total_size,
                stored_size,
            } => {
                let kept = prune(children, keep, keep_metadata);
                // A group whose every tensor was filtered out is not a row worth drawing.
                if kept.is_empty() {
                    continue;
                }
                out.push(TreeNode::Group {
                    name: name.clone(),
                    children: kept,
                    expanded: *expanded,
                    tensor_count: *tensor_count,
                    params: *params,
                    total_size: *total_size,
                    stored_size: *stored_size,
                });
            }
        }
    }
    out
}

/// Align two **rooted** trees — the shape `Session::build_rooted_tree` produces, where the whole
/// checkpoint hangs under one group named after itself.
///
/// Those root names differ whenever the two checkpoints are two different files, so aligning the
/// roots pairs nothing and reports every tensor as both added and removed. What the two sides have
/// in common is what is *inside* the root, so that is what gets aligned; the roots themselves are
/// each side's label, which a caller already knows.
#[must_use]
pub fn align_rooted(old: &[TreeNode], new: &[TreeNode]) -> Vec<AlignedNode> {
    align(unwrap_root(old), unwrap_root(new))
}

/// Every tensor name in a tree, for the invariant that alignment loses none of them.
#[cfg(test)]
fn tensor_names(nodes: &[TreeNode]) -> std::collections::BTreeSet<String> {
    fn walk(nodes: &[TreeNode], out: &mut std::collections::BTreeSet<String>) {
        for n in nodes {
            match n {
                TreeNode::Tensor { info, .. } => {
                    out.insert(info.name.clone());
                }
                TreeNode::Group { children, .. } => walk(children, out),
                TreeNode::Metadata { .. } => {}
            }
        }
    }
    let mut out = std::collections::BTreeSet::new();
    walk(nodes, &mut out);
    out
}

/// The children of a lone summarising root, or the slice unchanged when it is not that shape.
fn unwrap_root(nodes: &[TreeNode]) -> &[TreeNode] {
    match nodes {
        [TreeNode::Group { children, .. }] => children,
        other => other,
    }
}

/// One level of the merge, then recursively their children.
fn align_level(old: &[TreeNode], new: &[TreeNode], prefix: &str) -> Vec<AlignedNode> {
    // Merge by name. Both inputs are already natural-sorted by the kernel, but this does not lean
    // on that: it collects each side into a keyed list and walks the union in the order the *new*
    // side presents, then appends what only the old side had. Leaning on a shared sort would make
    // the alignment silently wrong the day one side's ordering changed.
    let mut rows: Vec<AlignedNode> = Vec::new();
    // `Option` per slot rather than a parallel `used` vector: taking a match out is what "paired"
    // means, and it leaves nothing to index (or to index wrongly).
    let mut unpaired: Vec<Option<&TreeNode>> = old.iter().map(Some).collect();

    for n in new {
        let key = node_key(n);
        let taken = unpaired
            .iter_mut()
            .find(|slot| slot.is_some_and(|o| node_key(o) == key))
            .and_then(Option::take);
        rows.push(pair(taken, Some(n), prefix));
    }
    // Whatever the old side had and the new side does not — removals. Appended after the new
    // side's rows at this level, which keeps the new checkpoint's own order readable; a removal has
    // no position in an order it is absent from.
    for o in unpaired.into_iter().flatten() {
        rows.push(pair(Some(o), None, prefix));
    }
    rows
}

/// What makes two nodes "the same node" across checkpoints: kind plus identity. A group and a tensor
/// of the same name are not the same node — that is a structural change, not a modified tensor.
///
/// A tensor's identity is its **full** name, not its last segment. The compacting tree can put two
/// tensors whose names end the same way under one group (`model.weight` beside
/// `model.layers.0.weight` collapsed onto one row), and keying by the segment paired unrelated
/// tensors — and listed the same name twice in the jump list.
fn node_key(n: &TreeNode) -> (u8, String) {
    match n {
        TreeNode::Group { name, .. } => (0, name.clone()),
        TreeNode::Tensor { info, .. } => (1, info.name.clone()),
        TreeNode::Metadata { info } => (2, info.name.clone()),
    }
}

/// What a row is *called* on screen: a tensor's compacted label or last dotted segment, a group's or
/// metadata key's own name. Distinct from its key — two rows can read the same and still be
/// different tensors.
fn display_name(n: &TreeNode) -> String {
    match n {
        TreeNode::Group { name, .. } => name.clone(),
        TreeNode::Tensor { info, label } => label
            .clone()
            .unwrap_or_else(|| last_segment(&info.name).to_string()),
        TreeNode::Metadata { info } => info.name.clone(),
    }
}

/// The display name of a node: a tensor's last dotted segment, a group's or metadata key's name.
fn last_segment(name: &str) -> &str {
    name.rsplit_once('.').map_or(name, |(_, last)| last)
}

/// Build one row from an old/new pair, recursing into children.
fn pair(old: Option<&TreeNode>, new: Option<&TreeNode>, prefix: &str) -> AlignedNode {
    let present = new
        .or(old)
        .unwrap_or_else(|| unreachable!("a pair has at least one side"));
    let name = display_name(present);
    // A tensor's path is its own full name — that is what opens its detail view, and it must not be
    // a synthetic prefix chain (which, on a rooted tree, would carry the checkpoint's filename).
    // Groups and metadata get a prefixed path, used only as a stable fold/selection key.
    let path = match present {
        TreeNode::Tensor { info, .. } => info.name.clone(),
        TreeNode::Group { .. } | TreeNode::Metadata { .. } if prefix.is_empty() => name.clone(),
        TreeNode::Group { .. } | TreeNode::Metadata { .. } => format!("{prefix}.{name}"),
    };

    let children = align_level(children_of(old), children_of(new), &path);
    let differing_below: usize = children.iter().map(|c| c.differing).sum();
    // Whether *anything* below differs — which is not the same question as how many differing
    // tensors there are.
    //
    // `differing` counts tensors on purpose (it drives the "(N differ)" badge, and a count of rows
    // would be a different number). But a metadata entry is a difference that is not a tensor, so
    // deciding the status from that count made `🔧 Metadata` — a group whose only changes are
    // metadata — report itself as `Same`. Everything downstream believed it: the reveal walk left it
    // folded, so a changed metadata entry was never shown on load, and a "differences only" filter
    // dropped the whole subtree.
    let anything_below_differs = children.iter().any(|c| c.status.differs());

    let status = match (old, new) {
        (None, Some(_)) => Status::OnlyNew,
        (Some(_), None) => Status::OnlyOld,
        (Some(o), Some(n)) => {
            if children.is_empty() {
                // A leaf: compare what it actually is.
                if leaf_same(o, n) {
                    Status::Same
                } else {
                    Status::Changed
                }
            } else if anything_below_differs {
                Status::Changed
            } else {
                Status::Same
            }
        }
        (None, None) => unreachable!("a pair has at least one side"),
    };

    // A differing *tensor* is what gets counted; a group inherits its subtree's count. A wholly
    // added or removed subtree counts every tensor in it, since all of them differ.
    let differing = if children.is_empty() {
        usize::from(status.differs() && is_tensor(present))
    } else if matches!(status, Status::OnlyNew | Status::OnlyOld) {
        count_tensors(present)
    } else {
        differing_below
    };

    AlignedNode {
        name,
        path,
        old: old.map(side_of),
        new: new.map(side_of),
        status,
        differing,
        // One subtree stands for itself until [`fold_families`] folds its siblings onto it.
        members: 1,
        children,
    }
}

fn children_of(n: Option<&TreeNode>) -> &[TreeNode] {
    // A wildcard over `TreeNode`, which this crate owns — normally worth listing exhaustively so a
    // new variant is a compile error. Here the question is only "does it have children", and a new
    // leaf kind answers "no" correctly; a new *container* kind would need its own arm, which is
    // why the arms are named rather than collapsed to `_`.
    match n {
        Some(TreeNode::Group { children, .. }) => children,
        Some(TreeNode::Tensor { .. } | TreeNode::Metadata { .. }) | None => &[],
    }
}

fn is_tensor(n: &TreeNode) -> bool {
    matches!(n, TreeNode::Tensor { .. })
}

fn count_tensors(n: &TreeNode) -> usize {
    match n {
        TreeNode::Group { children, .. } => children.iter().map(count_tensors).sum(),
        TreeNode::Tensor { .. } => 1,
        TreeNode::Metadata { .. } => 0,
    }
}

/// Whether two leaves are the same. **Structure only** — dtype and shape for a tensor, the value
/// for a metadata entry. Element values are not compared here for the same reason the report does
/// not: reading every byte of two multi-GB checkpoints belongs on the CLI, where a long scan has a
/// progress bar (see `crate::diff`).
fn leaf_same(o: &TreeNode, n: &TreeNode) -> bool {
    match (o, n) {
        (TreeNode::Tensor { info: a, .. }, TreeNode::Tensor { info: b, .. }) => {
            a.dtype == b.dtype && a.shape == b.shape
        }
        (TreeNode::Metadata { info: a }, TreeNode::Metadata { info: b }) => a.value == b.value,
        // Different kinds under one name: a structural change, not a match.
        _ => false,
    }
}

fn side_of(n: &TreeNode) -> Side {
    match n {
        TreeNode::Group {
            tensor_count,
            params,
            total_size,
            ..
        } => Side::Group {
            tensor_count: *tensor_count,
            params: *params,
            total_size: *total_size,
        },
        // The fold is noted afterwards, by [`note_folds`]: the tree's own `label` is the compacted
        // display name, and folding is a property of the *comparison*, not of either tree.
        TreeNode::Tensor { info, .. } => Side::Tensor {
            info: Box::new(info.clone()),
            fold: None,
        },
        TreeNode::Metadata { info } => Side::Metadata {
            name: info.name.clone(),
            value: info.value.clone(),
        },
    }
}

/// The same comparison the other way round: what was the baseline becomes the newer side.
///
/// A pure transform, not a re-read: both checkpoints are already aligned, and which one is "old" is
/// only a question of which column a side is drawn in and which way `+`/`-` point. So flipping is
/// free, and neither frontend has to fetch anything to do it.
///
/// Row *order* is deliberately preserved. Re-aligning would reorder (each orientation lists the
/// newer side's rows first), and rows jumping around under the cursor is a worse cost than a
/// column order that no longer matches the newer side's.
#[must_use]
pub fn swap(rows: &[AlignedNode]) -> Vec<AlignedNode> {
    rows.iter()
        .map(|r| AlignedNode {
            name: r.name.clone(),
            path: r.path.clone(),
            old: r.new.clone(),
            new: r.old.clone(),
            status: match r.status {
                // "Only in the newer one" and "only in the baseline" trade places; a change is a
                // change either way round, and so is a match.
                Status::OnlyNew => Status::OnlyOld,
                Status::OnlyOld => Status::OnlyNew,
                Status::Changed => Status::Changed,
                Status::Same => Status::Same,
            },
            differing: r.differing,
            members: r.members,
            children: swap(&r.children),
        })
        .collect()
}

/// One visible row: which node, how deep it sits, and whether its children are showing.
pub struct FlatRow<'a> {
    pub node: &'a AlignedNode,
    pub depth: usize,
    pub expanded: bool,
}

/// The rows to draw, given which group paths are unfolded.
///
/// The Rust half of what the browser's `flattenDiff` does, so the terminal folds and scrolls the
/// same tree the same way.
#[must_use]
pub fn flatten<'a, S: std::hash::BuildHasher>(
    rows: &'a [AlignedNode],
    expanded: &std::collections::HashSet<String, S>,
) -> Vec<FlatRow<'a>> {
    let mut out = Vec::new();
    push_rows(rows, expanded, 0, &mut out);
    out
}

fn push_rows<'a, S: std::hash::BuildHasher>(
    rows: &'a [AlignedNode],
    expanded: &std::collections::HashSet<String, S>,
    depth: usize,
    out: &mut Vec<FlatRow<'a>>,
) {
    for node in rows {
        let open = node.is_group() && expanded.contains(&node.path);
        out.push(FlatRow {
            node,
            depth,
            expanded: open,
        });
        if open {
            push_rows(&node.children, expanded, depth + 1, out);
        }
    }
}

/// The group paths to unfold so every difference is visible, and nothing else.
///
/// What "show me what changed" needs: a change three groups deep is no use folded, and unfolding
/// everything would bury it in unchanged rows.
#[must_use]
pub fn expand_to_differences(rows: &[AlignedNode]) -> std::collections::HashSet<String> {
    let mut open = std::collections::HashSet::new();
    reveal(rows, &mut Vec::new(), &mut open);
    open
}

fn reveal(
    rows: &[AlignedNode],
    trail: &mut Vec<String>,
    open: &mut std::collections::HashSet<String>,
) {
    for n in rows {
        if n.is_group() {
            if n.differing > 0 || n.status.differs() {
                open.extend(trail.iter().cloned());
                open.insert(n.path.clone());
            }
            trail.push(n.path.clone());
            reveal(&n.children, trail, open);
            trail.pop();
        } else if n.status.differs() {
            open.extend(trail.iter().cloned());
        }
    }
}

/// The differing tensors in draw order, by path — what `n`/`N` step through.
///
/// Precomputed rather than searched on each keypress: the walk is over the whole tree (31k tensors),
/// and doing it per press would make holding `n` down quadratic. Paths rather than indices because
/// the visible row set changes as groups fold, and an index into it would go stale.
#[must_use]
pub fn differences(rows: &[AlignedNode]) -> Vec<String> {
    let mut out = Vec::new();
    walk_differences(rows, &mut out);
    out
}

fn walk_differences(rows: &[AlignedNode], out: &mut Vec<String>) {
    for r in rows {
        // A leaf difference is a stop; a group is not a stop of its own, but its descendants are —
        // otherwise every ancestor of one changed tensor would be a step of its own, and walking
        // to the next real difference would take as many presses as the tree is deep.
        if r.children.is_empty() {
            if r.status.differs() {
                out.push(r.path.clone());
            }
        } else {
            walk_differences(&r.children, out);
        }
    }
}

/// Note the leaves an alignment folded: the **baseline** side of a row that stands for 256 tensors says
/// so, and the fused side beside it says nothing, because it is one tensor.
///
/// Applied to the aligned rows rather than to the tree they came from. Folding is a property of *this
/// comparison* — a rename rule mapped 256 names onto one — not of either checkpoint, and the tree's own
/// `label` field already means the compacted display name of a collapsed chain. Writing the fold there
/// made `embed_tokens.weight` print its name twice and cost a folded row its name.
///
/// Keyed by the baseline's **renamed** names, which is what its rebuilt tree's leaves are called. Names
/// not in `folds` are left alone; call it before any [`swap`], and the notes follow their sides.
pub fn note_folds(rows: &mut [AlignedNode], folds: &std::collections::BTreeMap<String, usize>) {
    if folds.is_empty() {
        return;
    }
    for row in rows {
        if let Some(Side::Tensor { info, fold }) = row.old.as_mut() {
            *fold = folds.get(&info.name).copied();
        }
        note_folds(&mut row.children, folds);
    }
}

/// Fold **uniform index families**: 62 layers that differ from their counterparts in the same way
/// become one row, `{0-61}`, standing for all of them.
///
/// The side-by-side view of a re-quantization is 117,000 rows, and 116,000 of them say the same thing
/// as the row above. That is the problem the compact tree solves for one checkpoint
/// ([`crate::compact`]) and `diff`'s family collapsing solves for the report; this is the same idea for
/// an aligned pair, and it is what makes the *irregular* layer visible — the one with an extra tensor,
/// or a dtype its siblings don't share, stands alone beside a folded run of its neighbours.
///
/// **What folds.** Sibling groups whose name is an index (`0`, `1`, … — a layer or an expert number)
/// and whose aligned subtrees are *identical*: same shape, same names, same statuses, same signatures
/// on both sides. Anything else is left where it is. Bottom-up, so a layer's 256 experts fold before
/// the layers themselves are compared — otherwise no two layers would ever look alike.
///
/// **What a folded row means.** Its subtree is one member (the template), so the sizes and counts on it
/// describe one layer; [`AlignedNode::members`] says how many there are, and `differing` is summed over
/// all of them, because "how much is hidden in here" is the question a folded row has to answer.
///
/// Index sets cannot overlap: each index falls in exactly one signature bucket, and a bucket is drawn
/// where its first member was — so `{0-2}`, `{3}`, `{4-6}` reads in order. The label is
/// [`crate::diff::summarize_indices`], the same `{0-2,5}` wording the report uses.
#[must_use]
pub fn fold_families(rows: &[AlignedNode]) -> Vec<AlignedNode> {
    rows.iter().map(fold_node).collect()
}

/// One row with its children folded (and theirs, and so on).
fn fold_node(node: &AlignedNode) -> AlignedNode {
    AlignedNode {
        children: fold_children(&node.children),
        ..node.clone()
    }
}

/// Fold what can be folded among one node's children, preserving their order.
fn fold_children(children: &[AlignedNode]) -> Vec<AlignedNode> {
    use std::collections::hash_map::Entry;
    // Depth first: the experts inside each layer collapse before the layers are compared to each other.
    let folded = children.iter().map(fold_node);

    // One bucket per family, in the order the families first appear — so the output reads in the input's
    // order, and index sets cannot overlap because each child lands in exactly one bucket.
    let mut buckets: Vec<Vec<AlignedNode>> = Vec::new();
    let mut at: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for child in folded {
        // A row whose name *starts* with an index. Usually a group (`0` under `layers`) — but a layer or
        // an expert that holds a single tensor has already been merged onto one row by `collapse_chains`
        // (`0.mlp.weight`), and those are families too. Not folding them would leave exactly the
        // one-tensor-per-expert checkpoints unhelped.
        let Some((_, rest)) = index_head(&child.name) else {
            buckets.push(vec![child]); // never a family: a bucket of its own
            continue;
        };
        // What follows the index is part of the identity: `0.mlp.weight` and `1.mlp.bias` are two
        // families of one, not one family of two.
        let key = format!("{rest}\u{1}{}", family_key(&child));
        match at.entry(key) {
            Entry::Occupied(slot) => {
                if let Some(bucket) = buckets.get_mut(*slot.get()) {
                    bucket.push(child);
                }
            }
            Entry::Vacant(slot) => {
                slot.insert(buckets.len());
                buckets.push(vec![child]);
            }
        }
    }
    buckets.into_iter().map(fold_together).collect()
}

/// One row from one bucket: the member itself when it stands alone, else the family they become.
fn fold_together(mut members: Vec<AlignedNode>) -> AlignedNode {
    if members.len() < 2 {
        return members
            .pop()
            .unwrap_or_else(|| unreachable!("a bucket holds at least one member"));
    }
    let count = members.len();
    let indices: Vec<String> = members
        .iter()
        .filter_map(|m| index_head(&m.name).map(|(idx, _)| idx.to_string()))
        .collect();
    // Summed: a folded row's job is to say how much it is hiding.
    let differing: usize = members.iter().map(|m| m.differing).sum();
    let first = members
        .into_iter()
        .next()
        .unwrap_or_else(|| unreachable!("a bucket holds at least one member"));
    let rest = index_head(&first.name).map_or("", |(_, rest)| rest);
    let range = crate::diff::summarize_indices(&indices);
    let name = if rest.is_empty() {
        range
    } else {
        format!("{range}.{rest}")
    };
    // The path with the label in place of the index, so it is a stable fold key and cannot collide with
    // a real one (no tensor name contains `{`).
    let path = first
        .path
        .strip_suffix(&first.name)
        .map_or_else(|| name.clone(), |prefix| format!("{prefix}{name}"));
    AlignedNode {
        name,
        path,
        members: count,
        differing,
        ..first
    }
}

/// A row's leading index and what follows it: `0` → `("0", "")`, `12.mlp.weight` → `("12", "mlp.weight")`,
/// and `None` when the name does not start with one (so it is not part of any family).
fn index_head(name: &str) -> Option<(&str, &str)> {
    let (head, rest) = name.split_once('.').unwrap_or((name, ""));
    let indexed = !head.is_empty() && head.bytes().all(|b| b.is_ascii_digit());
    indexed.then_some((head, rest))
}

/// What makes two sibling subtrees "the same": everything except the index that names them.
///
/// Deliberately *not* including the node's own name (that is the index) nor a tensor's full name or
/// source file (both carry the index, and which shard a tensor landed in is not a difference between
/// layers). Everything else is in: statuses, dtypes, shapes, group totals, metadata values, the child
/// names below, and how many members each child already stands for.
fn family_key(node: &AlignedNode) -> String {
    let mut key = String::new();
    push_key(node, &mut key);
    key
}

fn push_key(node: &AlignedNode, out: &mut String) {
    use std::fmt::Write as _;
    let _ = write!(
        out,
        "({:?}|{}|{}|{}",
        node.status,
        side_key(node.old.as_ref()),
        side_key(node.new.as_ref()),
        node.members
    );
    for child in &node.children {
        // The child's *name* matters — a layer with an extra tensor must not look like one without it.
        let _ = write!(out, "|{}", child.name);
        push_key(child, out);
    }
    out.push(')');
}

/// One side of a row, as a comparable string. Shapes and dtypes, never names.
fn side_key(side: Option<&Side>) -> String {
    match side {
        None => "-".to_string(),
        Some(Side::Group {
            tensor_count,
            params,
            total_size,
        }) => format!("g{tensor_count},{params},{total_size}"),
        Some(Side::Tensor { info, fold }) => {
            format!("t{},{:?},{fold:?}", info.dtype, info.shape)
        }
        Some(Side::Metadata { name, value }) => format!("m{name}={value}"),
    }
}

/// How the comparison's leaves fall out by status — the headline, counted once.
///
/// **Why this exists.** The one-page diff report and the side-by-side view tallied the same pair
/// separately and printed different totals for it. Two counters over two representations are always
/// one change away from disagreeing, so there is one, here, over the aligned tree — and
/// [`Tally::differing`] is the same number [`differences`] returns a list of, which a test asserts
/// rather than assumes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Counts {
    /// Present on both sides and identical.
    pub same: usize,
    /// Present on both sides, but not the same.
    pub changed: usize,
    /// In the baseline only — *removed*, reading the comparison as old → new.
    pub only_old: usize,
    /// In the newer side only — *added*.
    pub only_new: usize,
}

impl Counts {
    /// Everything here that is not a match.
    #[must_use]
    pub const fn differing(&self) -> usize {
        self.changed + self.only_old + self.only_new
    }
}

/// The comparison's leaves by status, **with tensors and metadata counted apart**.
///
/// Apart, because the two views name them apart: the one-page report says
/// `1 removed, 2 metadata changes` where a single set of counters could only say `3 removed`. The
/// totals were always the same — a test pins them — but a metadata entry folded into "removed tensors"
/// is a labelling error, and two views of one comparison should read alike.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Tally {
    pub tensors: Counts,
    pub metadata: Counts,
}

impl Tally {
    /// Everything that is not a match, of either kind — what "N differences" means, and the same
    /// number [`differences`] returns a list of.
    #[must_use]
    pub const fn differing(&self) -> usize {
        self.tensors.differing() + self.metadata.differing()
    }

    /// How many metadata entries differ, however they differ — one figure, because that is how the
    /// report reports them (`2 metadata changes`).
    #[must_use]
    pub const fn metadata_changes(&self) -> usize {
        self.metadata.differing()
    }

    /// Whether the two checkpoints have no **tensor** in common at all.
    ///
    /// Worth stating outright, because it is a different situation from "many differences": two
    /// checkpoints with unrelated naming schemes align nothing, so every tensor of *both* becomes a
    /// one-sided row. Reporting that as a six-figure difference count buries the single fact that
    /// explains the whole comparison. Judged on tensors alone, since that is what the sentence says —
    /// two checkpoints can share a `format` key and still have nothing to do with each other.
    #[must_use]
    pub const fn disjoint(&self) -> bool {
        self.tensors.same == 0
            && self.tensors.changed == 0
            && self.tensors.only_old > 0
            && self.tensors.only_new > 0
    }
}

/// Count the leaves by status, over exactly the rows [`differences`] walks.
#[must_use]
pub fn tally(rows: &[AlignedNode]) -> Tally {
    let mut t = Tally::default();
    walk_tally(rows, &mut t);
    t
}

fn walk_tally(rows: &[AlignedNode], t: &mut Tally) {
    for r in rows {
        // Leaves only, exactly as `walk_differences` does: a group is a container, and counting it
        // too would count everything under it twice.
        if !r.children.is_empty() {
            walk_tally(&r.children, t);
            continue;
        }
        // Which set of counters this leaf belongs to, from whichever side it is present on — a
        // one-sided row has only one to ask.
        let counts = match r.new.as_ref().or(r.old.as_ref()) {
            Some(Side::Metadata { .. }) => &mut t.metadata,
            // A tensor, or a childless group (which holds no tensors and no entries, so it counts
            // with the tensors it stands in for).
            _ => &mut t.tensors,
        };
        match r.status {
            Status::Same => counts.same += 1,
            Status::Changed => counts.changed += 1,
            Status::OnlyOld => counts.only_old += 1,
            Status::OnlyNew => counts.only_new += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::MetadataInfo;

    fn tensor(name: &str, dtype: &str, shape: &[usize]) -> TreeNode {
        let num_elements: usize = shape.iter().product();
        TreeNode::Tensor {
            info: TensorInfo {
                name: name.to_string(),
                dtype: dtype.to_string(),
                shape: shape.to_vec(),
                size_bytes: num_elements * 2,
                num_elements,
                storage: crate::tree::Storage::Raw,
                source_path: "/ckpt/model.safetensors".to_string(),
                layout: crate::tree::Layout::None,
            },
            label: None,
        }
    }

    fn group(name: &str, children: Vec<TreeNode>) -> TreeNode {
        TreeNode::Group {
            name: name.to_string(),
            children,
            expanded: true,
            tensor_count: 0,
            params: 0,
            total_size: 0,
            stored_size: 0,
        }
    }

    /// Children of a notional root, for the tests that do not care about the root itself.
    /// A bare `TensorInfo`, for the tree builder's flat input.
    fn tensor_info(name: &str, dtype: &str, shape: &[usize]) -> TensorInfo {
        let TreeNode::Tensor { info, .. } = tensor(name, dtype, shape) else {
            unreachable!("`tensor` builds a tensor node")
        };
        info
    }

    fn root_children(children: Vec<TreeNode>) -> Vec<TreeNode> {
        children
    }

    fn meta(key: &str, value: &str) -> TreeNode {
        TreeNode::Metadata {
            info: MetadataInfo {
                name: key.to_string(),
                value: value.to_string(),
                value_type: "String".to_string(),
            },
        }
    }

    /// The property the whole side-by-side rests on: one row per name, with each side's content on
    /// its own side and a gap where a side has nothing.
    #[test]
    fn a_shared_name_becomes_one_row_with_two_sides() {
        let old = vec![tensor("w", "F16", &[4, 4])];
        let new = vec![tensor("w", "U16", &[4, 4])];
        let rows = align(&old, &new);
        assert_eq!(rows.len(), 1, "one row, not two");
        assert_eq!(rows[0].status, Status::Changed, "the dtype differs");
        assert!(rows[0].old.is_some() && rows[0].new.is_some());
    }

    #[test]
    fn an_identical_tensor_is_same_and_is_not_a_stop() {
        let t = vec![tensor("w", "F16", &[4, 4])];
        let rows = align(&t, &t);
        assert_eq!(rows[0].status, Status::Same);
        assert_eq!(rows[0].differing, 0);
        assert!(differences(&rows).is_empty(), "nothing to jump to");
    }

    #[test]
    fn a_row_missing_from_one_side_keeps_its_place_with_a_gap() {
        let old = vec![tensor("a", "F16", &[1]), tensor("gone", "F16", &[1])];
        let new = vec![tensor("a", "F16", &[1]), tensor("added", "F32", &[1])];
        let rows = align(&old, &new);
        let by_name: Vec<_> = rows.iter().map(|r| (r.name.as_str(), r.status)).collect();
        assert_eq!(
            by_name,
            [
                ("a", Status::Same),
                ("added", Status::OnlyNew),
                ("gone", Status::OnlyOld)
            ],
            "the new side's order, then what only the old side had"
        );
        // The gap is what lets a renderer leave one column blank.
        let added = rows.iter().find(|r| r.name == "added").expect("added");
        assert!(added.old.is_none() && added.new.is_some());
        let gone = rows.iter().find(|r| r.name == "gone").expect("gone");
        assert!(gone.old.is_some() && gone.new.is_none());
    }

    /// A collapsed group has to say whether folding hides anything, or folding becomes a way to
    /// miss differences.
    #[test]
    fn a_group_counts_the_differing_tensors_beneath_it() {
        let old = vec![group(
            "layers",
            vec![
                tensor("layers.q", "F16", &[1]),
                tensor("layers.k", "F16", &[1]),
                tensor("layers.v", "F16", &[1]),
            ],
        )];
        let new = vec![group(
            "layers",
            vec![
                tensor("layers.q", "F16", &[1]),
                tensor("layers.k", "U16", &[1]),
                tensor("layers.v", "F16", &[2]),
            ],
        )];
        let rows = align(&old, &new);
        assert_eq!(rows[0].status, Status::Changed, "the group differs inside");
        assert_eq!(rows[0].differing, 2, "k (dtype) and v (shape)");
        assert_eq!(
            differences(&rows),
            ["layers.k", "layers.v"],
            "the jump list stops on the tensors, not on their group"
        );
    }

    #[test]
    fn a_group_identical_throughout_is_same() {
        let g = vec![group("layers", vec![tensor("layers.q", "F16", &[1])])];
        let rows = align(&g, &g);
        assert_eq!(rows[0].status, Status::Same);
        assert_eq!(rows[0].differing, 0);
    }

    /// A whole added subtree counts every tensor in it: all of them are absent from the other side,
    /// so a collapsed `(1 differ)` would understate what opening it reveals.
    #[test]
    fn a_wholly_added_group_counts_all_of_its_tensors() {
        let old: Vec<TreeNode> = Vec::new();
        let new = vec![group(
            "experts",
            vec![
                tensor("experts.0", "F16", &[1]),
                group("experts.sub", vec![tensor("experts.sub.1", "F16", &[1])]),
            ],
        )];
        let rows = align(&old, &new);
        assert_eq!(rows[0].status, Status::OnlyNew);
        assert_eq!(rows[0].differing, 2, "both tensors in the new subtree");
    }

    #[test]
    fn a_name_that_changed_kind_is_a_difference_not_a_match() {
        // A group replaced by a tensor of the same name is structural: pairing them would report a
        // modified tensor and silently drop whatever the group contained.
        let old = vec![group("x", vec![tensor("x.inner", "F16", &[1])])];
        let new = vec![tensor("x", "F16", &[1])];
        let rows = align(&old, &new);
        let kinds: Vec<_> = rows.iter().map(|r| (r.name.as_str(), r.status)).collect();
        assert_eq!(kinds, [("x", Status::OnlyNew), ("x", Status::OnlyOld)]);
    }

    #[test]
    fn metadata_is_compared_by_value() {
        let old = vec![meta("format", "pt"), meta("dropped", "1")];
        let new = vec![meta("format", "safetensors")];
        let rows = align(&old, &new);
        let m: Vec<_> = rows.iter().map(|r| (r.name.as_str(), r.status)).collect();
        assert_eq!(
            m,
            [("format", Status::Changed), ("dropped", Status::OnlyOld)]
        );
        // Metadata is not a tensor, so it does not inflate the tensor counts a group reports…
        assert_eq!(rows[0].differing, 0);
        // …but it is still somewhere to jump to.
        assert_eq!(differences(&rows), ["format", "dropped"]);
    }

    /// A group whose only changes are *metadata* still counts as changed.
    ///
    /// `differing` counts tensors, deliberately — it drives the "(N differ)" badge. Deciding a
    /// group's status from that count made `🔧 Metadata` report itself as `Same`, and everything
    /// downstream believed it: [`reveal`] left it folded so a changed entry was never shown on load,
    /// and a "differences only" filter dropped the subtree entirely.
    #[test]
    fn a_group_whose_only_changes_are_metadata_is_not_reported_as_unchanged() {
        let old = root_children(vec![group(
            "🔧 Metadata",
            vec![meta("format", "pt"), meta("note", "a")],
        )]);
        let new = root_children(vec![group(
            "🔧 Metadata",
            vec![meta("format", "pt"), meta("note", "b")],
        )]);
        let rows = align(&old, &new);
        let meta_group = &rows[0];
        assert_eq!(
            meta_group.status,
            Status::Changed,
            "one changed entry makes the group changed"
        );
        // Still zero *tensors*, which is what the badge counts.
        assert_eq!(meta_group.differing, 0);
        // And the tally files the change under metadata, not under removed/changed tensors — which is
        // what lets this view use the report's own words.
        let t = tally(&rows);
        assert_eq!(t.metadata.changed, 1);
        assert_eq!(t.metadata_changes(), 1);
        assert_eq!(t.tensors.differing(), 0, "no tensor differs");
        // And the change is revealed rather than left folded behind a group that claimed to match.
        assert!(
            expand_to_differences(&rows).contains("🔧 Metadata"),
            "the group hiding a metadata change must be opened"
        );
        assert_eq!(differences(&rows), ["🔧 Metadata.note"]);
    }

    #[test]
    fn paths_are_dotted_so_a_row_can_open_its_tensor() {
        let old = vec![group(
            "model",
            vec![group("layers", vec![tensor("model.layers.w", "F16", &[1])])],
        )];
        let rows = align(&old, &old);
        // A single-child chain reads as one row, the way the kernel's own tree writes it — the
        // difference being that this collapsing now happens on the *merged* tree, so both sides get
        // the same shape whatever each side's own grouping was (see `expand_segments`).
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "model.layers");
        // What matters either way: a tensor's path is its full name, so the row can open its detail.
        let leaf = &rows[0].children[0];
        assert!(leaf.children.is_empty());
        assert_eq!(leaf.path, "model.layers.w");
    }

    /// The bug the first real run exposed: each tree hangs under a root named after its own
    /// checkpoint, so aligning the roots paired nothing and every tensor read as both added *and*
    /// removed — a comparison of a file with itself would have shown zero rows in common.
    #[test]
    fn rooted_trees_align_on_their_contents_not_their_root_names() {
        let old = vec![group("old.safetensors", vec![tensor("w", "F16", &[4])])];
        let new = vec![group("new.safetensors", vec![tensor("w", "U16", &[4])])];

        let naive = align(&old, &new);
        assert_eq!(naive.len(), 2, "aligning the roots pairs nothing");

        let rows = align_rooted(&old, &new);
        assert_eq!(rows.len(), 1, "one row for the tensor both sides have");
        assert_eq!(rows[0].status, Status::Changed);
    }

    /// Also from that run: two tensors whose names *end* the same way can sit under one group once
    /// the tree compacts, and keying by the last segment paired unrelated tensors — and listed the
    /// same name twice in the jump list.
    #[test]
    fn tensors_are_paired_by_full_name_not_by_last_segment() {
        let both = |dt: &str| {
            vec![group(
                "model",
                vec![
                    tensor("model.weight", "F16", &[1]),
                    tensor("model.layers.0.weight", dt, &[1]),
                ],
            )]
        };
        let rows = align(&both("F16"), &both("U16"));
        let leaves = &rows[0].children;
        assert_eq!(leaves.len(), 2, "two distinct tensors, not one paired pair");
        assert_eq!(
            differences(&rows),
            ["model.layers.0.weight"],
            "only the one that actually changed, named in full"
        );
        // And a tensor's path is its own name, so a row can open its detail view.
        assert!(leaves.iter().any(|l| l.path == "model.weight"));
    }

    /// Flipping the comparison must be exactly the comparison the other way round — and flipping
    /// twice must be where you started, or the button would drift the view each time it is pressed.
    #[test]
    fn swapping_sides_inverts_added_and_removed_and_is_its_own_inverse() {
        let old = root_children(vec![
            tensor("kept", "F16", &[1]),
            tensor("gone", "F16", &[1]),
        ]);
        let new = root_children(vec![
            tensor("kept", "U16", &[1]),
            tensor("added", "F32", &[1]),
        ]);
        let rows = align(&old, &new);

        let flipped = swap(&rows);
        let by_name =
            |rs: &[AlignedNode], n: &str| rs.iter().find(|r| r.name == n).expect("row").status;
        assert_eq!(by_name(&rows, "added"), Status::OnlyNew);
        assert_eq!(
            by_name(&flipped, "added"),
            Status::OnlyOld,
            "what was added is missing when read the other way"
        );
        assert_eq!(by_name(&flipped, "gone"), Status::OnlyNew);
        assert_eq!(
            by_name(&flipped, "kept"),
            Status::Changed,
            "a change is a change either way"
        );

        // The sides really did trade places, not just the labels.
        let kept = flipped.iter().find(|r| r.name == "kept").expect("kept");
        assert!(matches!(&kept.old, Some(Side::Tensor { info, .. }) if info.dtype == "U16"));
        assert!(matches!(&kept.new, Some(Side::Tensor { info, .. }) if info.dtype == "F16"));

        // Twice is the identity, so the button is safe to lean on.
        let back = swap(&flipped);
        let sig = |rs: &[AlignedNode]| {
            rs.iter()
                .map(|r| (r.name.clone(), r.status, r.differing))
                .collect::<Vec<_>>()
        };
        assert_eq!(sig(&back), sig(&rows));
    }

    #[test]
    fn swapping_reaches_all_the_way_down() {
        let old = root_children(vec![group("g", vec![tensor("g.only_old", "F16", &[1])])]);
        let new = root_children(vec![group("g", vec![])]);
        let flipped = swap(&align(&old, &new));
        let inner = &flipped[0].children[0];
        assert_eq!(inner.status, Status::OnlyNew, "nested rows flip too");
    }

    /// **The same tensor, grouped differently on each side, is one row.**
    ///
    /// Reproduces a real disagreement: `diff_old.safetensors` has one layer, so the kernel collapses
    /// its chain to `layers.0`; a 48-layer checkpoint leaves `layers` → `0`. Merging those level by
    /// level never paired `model.layers.0.input_layernorm.weight`, so it was reported as **both**
    /// added and removed — inflating the count by one and losing it from `changed`, which is precisely
    /// how the side-by-side and the one-page report came to differ by one tensor.
    #[test]
    fn a_tensor_pairs_even_when_the_two_sides_group_it_differently() {
        let name = "model.layers.0.input_layernorm.weight";
        // The small side: a single-child chain, collapsed onto one group row.
        let old = root_children(vec![group(
            "model",
            vec![group("layers.0", vec![tensor(name, "F32", &[4])])],
        )]);
        // The big side: many layers, so the chain stays split.
        let new = root_children(vec![group(
            "model",
            vec![group(
                "layers",
                vec![
                    group("0", vec![tensor(name, "BF16", &[4])]),
                    group(
                        "1",
                        vec![tensor(
                            "model.layers.1.input_layernorm.weight",
                            "BF16",
                            &[4],
                        )],
                    ),
                ],
            )],
        )]);

        let rows = align(&old, &new);
        let t = tally(&rows);
        assert_eq!(
            (t.tensors.only_old, t.tensors.only_new, t.tensors.changed),
            (0, 1, 1),
            "the shared tensor is one *changed* row, and only `layers.1` is added: {t:?}"
        );
        // The decisive assertion: it appears once, not once per side.
        let hits = differences(&rows).into_iter().filter(|p| p == name).count();
        assert_eq!(hits, 1, "the tensor must be one row, not one per side");
    }

    /// No tensor is dropped and none is duplicated, whatever the two shapes were.
    ///
    /// The guard for the whole class of bug above: the aligned tree's leaves are exactly the union of
    /// the two sides' tensor names.
    #[test]
    fn alignment_covers_each_tensor_exactly_once() {
        fn walk(rows: &[AlignedNode], seen: &mut Vec<String>) {
            for r in rows {
                if r.children.is_empty() {
                    if matches!(r.old, Some(Side::Tensor { .. }))
                        || matches!(r.new, Some(Side::Tensor { .. }))
                    {
                        seen.push(r.path.clone());
                    }
                } else {
                    walk(&r.children, seen);
                }
            }
        }

        let old = root_children(vec![group(
            "model",
            vec![
                group("layers.0", vec![tensor("model.layers.0.a", "F32", &[4])]),
                tensor("model.b", "F32", &[2]),
            ],
        )]);
        let new = root_children(vec![group(
            "model",
            vec![group(
                "layers",
                vec![
                    group("0", vec![tensor("model.layers.0.a", "F32", &[4])]),
                    group("1", vec![tensor("model.layers.1.a", "F32", &[4])]),
                ],
            )],
        )]);
        let rows = align(&old, &new);

        let mut seen: Vec<String> = Vec::new();
        walk(&rows, &mut seen);

        let mut union: Vec<String> = tensor_names(&old)
            .union(&tensor_names(&new))
            .cloned()
            .collect();
        union.sort();
        let mut got = seen.clone();
        got.sort();
        assert_eq!(got, union, "aligned leaves must be the union of both sides");
        assert_eq!(
            seen.len(),
            union.len(),
            "a tensor must not appear twice: {seen:?}"
        );
    }

    /// What each node at this level is called, for the pruning assertions.
    fn top_names(nodes: &[TreeNode]) -> Vec<String> {
        nodes
            .iter()
            .map(|c| match c {
                TreeNode::Group { name, .. } => name.clone(),
                TreeNode::Tensor { info, .. } => info.name.clone(),
                TreeNode::Metadata { info } => info.name.clone(),
            })
            .collect()
    }

    /// A tree rebuilt from renamed names must group by those names, not the old ones.
    ///
    /// What makes `--map` work on the side-by-side: rewriting a leaf's name inside an existing tree
    /// leaves the groups above it describing the name it used to have, so the two sides stop lining up
    /// for the very reason the rules were given.
    #[test]
    fn a_tree_built_from_names_groups_by_those_names() {
        let renamed = [
            tensor_info("model.layers.0.w", "F16", &[4]),
            tensor_info("model.layers.1.w", "F16", &[4]),
            tensor_info("lm_head.weight", "F16", &[2, 2]),
        ];
        let built = tree_from_tensors("renamed", &renamed, &[]);
        // Rooted, like the kernel's — so unwrap it to look at the level below.
        let [
            TreeNode::Group {
                children: under, ..
            },
        ] = built.as_slice()
        else {
            panic!("the built tree is rooted in one group")
        };
        assert_eq!(
            top_names(under),
            vec!["lm_head.weight".to_string(), "model".to_string()],
            "grouped on the *new* names — and a lone chain collapses onto one row, as the kernel's \
             tree does, so `lm_head.weight` is a tensor rather than a group wrapping `weight`"
        );
        assert_eq!(
            tensor_names(&built),
            renamed
                .iter()
                .map(|t| t.name.clone())
                .collect::<std::collections::BTreeSet<_>>(),
            "every tensor is present exactly once"
        );

        // And it aligns against an equivalently-named side as identical — the point of renaming.
        //
        // Shaped as the kernel shapes it: a lone chain ending in a tensor is one row, so `layers` holds
        // tensor rows rather than `0`/`1` groups. Tensors pair on their **full** name, so the display
        // labels need not match; the *kinds* at each level must.
        let other = vec![group(
            "other-checkpoint",
            vec![
                group(
                    "model",
                    vec![group(
                        "layers",
                        vec![
                            tensor("model.layers.0.w", "F16", &[4]),
                            tensor("model.layers.1.w", "F16", &[4]),
                        ],
                    )],
                ),
                tensor("lm_head.weight", "F16", &[2, 2]),
            ],
        )];
        // `align_rooted`, as the diff routes use — the roots differ and are unwrapped on both sides.
        let t = tally(&align_rooted(&built, &other));
        assert_eq!(t.differing(), 0, "the two sides now line up: {t:?}");
        assert_eq!(t.tensors.same, 3);
    }

    /// Metadata lands under the same row the kernel's tree puts it in.
    #[test]
    fn a_built_tree_puts_metadata_where_the_kernel_does() {
        let built = tree_from_tensors(
            "ckpt",
            &[tensor_info("a.w", "F16", &[1])],
            &[MetadataInfo {
                name: "format".to_string(),
                value: "pt".to_string(),
                value_type: "String".to_string(),
            }],
        );
        let [
            TreeNode::Group {
                children: under, ..
            },
        ] = built.as_slice()
        else {
            panic!("rooted")
        };
        assert!(
            top_names(under).contains(&crate::tree::METADATA_GROUP.to_string()),
            "metadata should hang under {:?}: {:?}",
            crate::tree::METADATA_GROUP,
            top_names(under)
        );
    }

    /// A scoped comparison: pruning before alignment, so every count describes the subset.
    #[test]
    fn pruning_keeps_the_selected_tensors_and_drops_emptied_groups() {
        let tree = root_children(vec![
            group(
                "layers.0",
                vec![
                    tensor("model.layers.0.w", "F16", &[4]),
                    tensor("model.layers.0.b", "F16", &[2]),
                ],
            ),
            group("layers.1", vec![tensor("model.layers.1.w", "F16", &[4])]),
            meta("format", "pt"),
        ]);

        // Only layer 1: `layers.0` loses both tensors and so disappears entirely.
        let keep = |n: &str| n.starts_with("model.layers.1.");
        let pruned = prune(&tree, &keep, true);
        assert_eq!(
            tensor_names(&pruned),
            std::iter::once("model.layers.1.w".to_string()).collect()
        );
        assert!(
            !top_names(&pruned).contains(&"layers.0".to_string()),
            "an emptied group is not a row: {:?}",
            top_names(&pruned)
        );
        assert!(
            top_names(&pruned).contains(&"format".to_string()),
            "metadata is kept by default"
        );

        // `--only-tensors` drops it.
        let bare = prune(&tree, &keep, false);
        assert!(
            !top_names(&bare).contains(&"format".to_string()),
            "--only-tensors drops metadata"
        );
    }

    /// Scoping the two sides before aligning gives counts that describe the subset, not the whole pair.
    #[test]
    fn a_scoped_comparison_counts_only_what_it_selected() {
        let old = root_children(vec![
            tensor("keep.w", "F16", &[4]),
            tensor("drop.w", "F16", &[4]),
        ]);
        let new = root_children(vec![
            tensor("keep.w", "BF16", &[4]),
            tensor("drop.w", "U16", &[4]),
        ]);
        let keep = |n: &str| n.starts_with("keep.");
        let rows = align(&prune(&old, &keep, true), &prune(&new, &keep, true));
        let t = tally(&rows);
        assert_eq!(t.tensors.changed, 1, "only the selected tensor is compared");
        assert_eq!(t.differing(), differences(&rows).len());
    }

    /// The headline count and the list you step through must be the same size.
    ///
    /// They were computed by two walks over two representations, and the two views printed different
    /// totals for one pair. Sharing the rule is not enough — this asserts it.
    #[test]
    fn the_tally_agrees_with_the_list_of_differences() {
        let old = root_children(vec![
            group(
                "layers.0",
                vec![
                    tensor("layers.0.same", "F16", &[4]),
                    tensor("layers.0.retyped", "F16", &[4]),
                    tensor("layers.0.gone", "F16", &[4]),
                ],
            ),
            tensor("kept", "F16", &[2]),
        ]);
        let new = root_children(vec![
            group(
                "layers.0",
                vec![
                    tensor("layers.0.same", "F16", &[4]),
                    tensor("layers.0.retyped", "BF16", &[4]),
                    tensor("layers.0.fresh", "F16", &[4]),
                ],
            ),
            tensor("kept", "F16", &[2]),
        ]);
        let rows = align(&old, &new);
        let t = tally(&rows);
        assert_eq!(t.tensors.same, 2, "`same` and `kept` match");
        assert_eq!(t.tensors.changed, 1, "`retyped` changed dtype");
        assert_eq!(t.tensors.only_old, 1, "`gone` is only in the baseline");
        assert_eq!(t.tensors.only_new, 1, "`fresh` is only in the newer side");
        assert_eq!(
            t.differing(),
            differences(&rows).len(),
            "the count and the steppable list must describe the same rows"
        );
    }

    /// Two checkpoints with unrelated naming schemes: nothing aligns, and that is the fact worth
    /// leading with rather than a six-figure difference count.
    #[test]
    fn a_pair_with_no_names_in_common_reports_itself_as_disjoint() {
        let old = root_children(vec![tensor("language_model.a", "F16", &[1])]);
        let new = root_children(vec![tensor("model.a", "F16", &[1])]);
        let t = tally(&align(&old, &new));
        assert!(
            t.disjoint(),
            "no tensor is shared, so this pair is disjoint"
        );

        // One shared tensor is enough to make it an ordinary comparison again.
        let overlapping = root_children(vec![
            tensor("model.a", "F16", &[1]),
            tensor("extra", "F16", &[1]),
        ]);
        assert!(
            !tally(&align(&overlapping, &new)).disjoint(),
            "a pair that shares a tensor is not disjoint"
        );
    }

    /// A fold note is a *count on a side*, not a name.
    ///
    /// Both halves of one bug: the fold used to be written into the tree's `label`, which already
    /// means "the compacted display name of a collapsed chain". So `embed_tokens.weight` — a lone
    /// `weight` under `embed_tokens` — had its name repeated after its signature on every row, and a
    /// row that really was folded had its name *replaced* by `×2`.
    #[test]
    fn a_folded_row_keeps_its_name_and_says_what_it_stands_for() {
        let old = tree_from_tensors(
            "old.safetensors",
            &[
                tensor_info("model.embed_tokens.weight", "F16", &[6, 4]),
                tensor_info("model.experts.down_proj.weight", "F16", &[2, 3]),
            ],
            &[],
        );
        let new = tree_from_tensors(
            "new.safetensors",
            &[
                tensor_info("model.embed_tokens.weight", "BF16", &[6, 4]),
                tensor_info("model.experts.down_proj.weight", "BF16", &[2, 2, 3]),
            ],
            &[],
        );
        let mut rows = align_rooted(&old, &new);
        // As the handler does it: two per-expert tensors were folded onto the one fused name.
        note_folds(
            &mut rows,
            &std::collections::BTreeMap::from([(
                "model.experts.down_proj.weight".to_string(),
                2_usize,
            )]),
        );

        let leaves = flat_leaves(&rows);
        let embed = leaves
            .iter()
            .find(|r| r.path == "model.embed_tokens.weight")
            .expect("the collapsed chain is a row");
        assert_eq!(
            embed.name, "embed_tokens.weight",
            "the chain still collapses onto one row, named for the whole of it"
        );
        assert!(
            matches!(embed.old, Some(Side::Tensor { fold: None, .. })),
            "nothing folded onto this row, so it stands for itself: {:?}",
            embed.old
        );

        let experts = leaves
            .iter()
            .find(|r| r.path == "model.experts.down_proj.weight")
            .expect("the folded tensor is a row");
        assert_eq!(
            experts.name, "experts.down_proj.weight",
            "a fold changes what the row stands for, not what it is called"
        );
        assert!(
            matches!(experts.old, Some(Side::Tensor { fold: Some(2), .. })),
            "the baseline side stands for two tensors: {:?}",
            experts.old
        );
        assert!(
            matches!(experts.new, Some(Side::Tensor { fold: None, .. })),
            "the fused side is one tensor, and says nothing: {:?}",
            experts.new
        );

        // Swapping is a pure flip, so the note travels with the side it describes.
        let flipped = swap(&rows);
        let experts = flat_leaves(&flipped)
            .into_iter()
            .find(|r| r.path == "model.experts.down_proj.weight")
            .expect("the folded tensor is still a row");
        assert!(
            matches!(experts.new, Some(Side::Tensor { fold: Some(2), .. })),
            "after a swap the fold is on the other column: {:?}",
            experts.new
        );
    }

    /// Family folding: the uniform layers become one row, and the odd one out stays visible.
    ///
    /// That second half is the whole point. Folding that hid the layer with an extra tensor would be
    /// worse than no folding at all, because the reader would never know to look.
    #[test]
    fn uniform_layers_fold_and_an_irregular_one_does_not() {
        // Four layers; three alike, and layer 2 with an extra tensor. Every layer's `weight` changed
        // dtype, so the fold has to survive a difference — it is a comparison, not a tree.
        let side = |dtype: &str| {
            let mut ts = Vec::new();
            for layer in 0..4 {
                ts.push(tensor_info(
                    &format!("model.layers.{layer}.mlp.weight"),
                    dtype,
                    &[4, 2],
                ));
                if layer == 2 {
                    ts.push(tensor_info("model.layers.2.mlp.bias", dtype, &[4]));
                }
            }
            ts
        };
        let old = tree_from_tensors("old", &side("F16"), &[]);
        let new = tree_from_tensors("new", &side("BF16"), &[]);
        let rows = fold_families(&align_rooted(&old, &new));

        let layers = find_row(&rows, "model.layers").expect("the layers group is a row");
        let names: Vec<&str> = layers.children.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            // `.mlp.weight` rides along because a layer holding one tensor was already merged onto one
            // row — the fold puts the range where the index was.
            vec!["{0-1,3}.mlp.weight", "2.mlp"],
            "the three alike layers fold; the one with an extra tensor stands alone"
        );

        let family = &layers.children[0];
        assert_eq!(family.members, 3, "the folded row stands for three layers");
        assert_eq!(
            family.differing, 3,
            "and says how many differing tensors it hides"
        );
        let odd = &layers.children[1];
        assert_eq!(odd.members, 1, "an unfolded row stands for itself");
        assert_eq!(odd.differing, 2, "both of that layer's tensors changed");
    }

    /// Folding is a *view*: the handler tallies before it, so the headline is unaffected.
    #[test]
    fn folding_is_a_view_and_the_tally_is_taken_before_it() {
        let side = |dtype: &str| {
            (0..6)
                .map(|l| tensor_info(&format!("model.layers.{l}.w"), dtype, &[2, 2]))
                .collect::<Vec<_>>()
        };
        let old = tree_from_tensors("old", &side("F16"), &[]);
        let new = tree_from_tensors("new", &side("BF16"), &[]);
        let rows = align_rooted(&old, &new);
        let folded = fold_families(&rows);
        assert_eq!(
            tally(&rows).differing(),
            6,
            "six tensors changed, one per layer"
        );
        // The folded tree has one row for the six, which is exactly why the tally is taken first.
        let family = &find_row(&folded, "model.layers")
            .expect("the layers group is a row")
            .children[0];
        assert_eq!((family.members, family.differing), (6, 6));
    }

    /// Two layers that differ *from each other* must not fold, even though each differs from its
    /// counterpart in the same way.
    #[test]
    fn layers_whose_signatures_disagree_stay_apart() {
        let old = tree_from_tensors(
            "old",
            &[
                tensor_info("model.layers.0.w", "F16", &[2, 2]),
                tensor_info("model.layers.1.w", "F16", &[4, 4]),
            ],
            &[],
        );
        let new = tree_from_tensors(
            "new",
            &[
                tensor_info("model.layers.0.w", "BF16", &[2, 2]),
                tensor_info("model.layers.1.w", "BF16", &[4, 4]),
            ],
            &[],
        );
        let rows = fold_families(&align_rooted(&old, &new));
        let layers = find_row(&rows, "model.layers").expect("the layers group is a row");
        assert_eq!(
            layers.children.len(),
            2,
            "different shapes are different families: {:?}",
            layers.children.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
    }

    /// The first row whose path is `path`, at any depth.
    fn find_row<'a>(rows: &'a [AlignedNode], path: &str) -> Option<&'a AlignedNode> {
        for r in rows {
            if r.path == path {
                return Some(r);
            }
            if let Some(hit) = find_row(&r.children, path) {
                return Some(hit);
            }
        }
        None
    }

    /// Every tensor row of an aligned tree, at any depth.
    fn flat_leaves(rows: &[AlignedNode]) -> Vec<AlignedNode> {
        fn walk(rows: &[AlignedNode], out: &mut Vec<AlignedNode>) {
            for r in rows {
                if r.children.is_empty() {
                    out.push(r.clone());
                } else {
                    walk(&r.children, out);
                }
            }
        }
        let mut out = Vec::new();
        walk(rows, &mut out);
        out
    }

    /// Alignment must not depend on the two sides arriving in the same order — the day one side's
    /// sort changes, a positional merge would start pairing unrelated tensors.
    #[test]
    fn alignment_is_by_name_not_by_position() {
        let old = vec![tensor("a", "F16", &[1]), tensor("b", "F16", &[1])];
        let new = vec![tensor("b", "F16", &[1]), tensor("a", "F16", &[1])];
        let rows = align(&old, &new);
        assert!(
            rows.iter().all(|r| r.status == Status::Same),
            "reordering is not a change: {:?}",
            rows.iter().map(|r| (&r.name, r.status)).collect::<Vec<_>>()
        );
    }
}
