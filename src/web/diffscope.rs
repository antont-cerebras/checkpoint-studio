//! The `diff` scope, as a request carries it — the browser's equivalent of the CLI's selection flags.
//!
//! `checkpoint-studio diff` narrows a comparison eleven ways: name globs, exact-name lists, dtype and
//! shape globs, rename rules, tensors-only. `/api/diff` accepted `against` and nothing else, so the one
//! thing you always want on a 117k-tensor pair — *look at these nineteen tensors* — was CLI-only.
//!
//! Everything here is parsing and plumbing. The filter itself is [`crate::compare::tensor_filter`] and
//! the rename map [`crate::compare::name_map`], both shared with `main.rs`: the CLI reads
//! `--names-from` off disk and this reads a pasted box, and from there the two are the same code. That
//! is what makes the scoped report the browser shows and the one a terminal prints the same report,
//! rather than two implementations that agree until one is edited.
//!
//! ## Repeated parameters
//!
//! `--name` is repeatable. The router parses a query into a `HashMap`, where a repeated key would
//! silently keep only the last — so lists arrive newline-separated (`name=a%0A!b`) and are split here.
//! One encoding, in one place, rather than every caller remembering.

use anyhow::Result;

use super::handlers::Query;
use crate::diff::{CheckpointSummary, DiffReport, NameMap, TensorFilter};

/// Which side of the comparison a per-side rule applies to. Named, because "the baseline" and "the
/// newer side" are the two things a `#subtree` can be attached to and a `bool` would not say which.
#[derive(Clone, Copy)]
pub(crate) enum Sub {
    Baseline,
    Newer,
}

/// A comparison's selection, compiled and ready to apply.
pub(crate) struct DiffScope {
    filter: TensorFilter,
    map: NameMap,
    /// `--only-tensors`: skip the checkpoints' metadata entirely.
    only_tensors: bool,
    /// `--align-fused`: line an unfused checkpoint up with its fused counterpart before comparing —
    /// the canonical rules from [`crate::diff::fused_layout_rules`], applied to **both** sides with
    /// folding, so 256 per-expert tensors read as `×256 → ×1` against the one fused tensor that holds
    /// them rather than as 255 removals.
    align_fused: Option<NameMap>,
    // The inputs, kept so the scope can render itself back as CLI flags. Compiled globs cannot: a
    // `glob::Pattern` does not give its source string back.
    /// `SOURCE#subtree`, per side: compare *from inside* a subtree. The tensors outside it are out of
    /// scope, and the ones inside are keyed by their sub-path — so a Hugging Face model's
    /// `language_model.model.layers.0.w` lines up with a converted checkpoint's `model.layers.0.w`.
    /// A scope change, not a rename (see [`crate::diff::CheckpointSummary::reroot`]).
    subtree: Option<String>,
    subtree_new: Option<String>,
    name_globs: Vec<String>,
    names_csv: Option<String>,
    names_lines: Option<String>,
    dtype_is: Option<String>,
    shape_is: Option<String>,
    map_rules: Vec<String>,
}

/// What applying a scope did, for the line the CLI prints as
/// `filter [name~model.layers.1.*] matched 19 of 117664 tensor(s)`.
pub(crate) struct Scoped {
    pub report: DiffReport,
    /// `None` when nothing narrowed the comparison — there is no "matched" line to show.
    pub matched: Option<Matched>,
    /// Names the rename map collided on. Kept rather than counted: a collision means two old names
    /// mapped onto one, so a tensor quietly vanished from the comparison, and which ones is the part
    /// worth showing.
    pub rename_collisions: Vec<String>,
}

/// The baseline's tensors after the scope's rules: renamed, folded, and what that cost or counted.
pub(crate) struct Renamed {
    pub tensors: Vec<crate::tree::TensorInfo>,
    /// Names two `--map` rules collapsed onto one, which silently loses a tensor. Empty for a fold,
    /// which loses nothing.
    pub collisions: Vec<String>,
    /// `name → parts` for the leaves an alignment folded, so the tree can label them `×256`.
    pub folds: std::collections::BTreeMap<String, usize>,
}

/// How many of how many tensors a scope selected.
pub(crate) struct Matched {
    pub selected: usize,
    pub total: usize,
    /// The tensor names the scope selected, in order — what the CLI lists under its match line.
    pub names: Vec<String>,
}

/// Split a repeated-list parameter: newline-separated, trimmed, blanks dropped.
fn list(q: &Query, key: &str) -> Vec<String> {
    q.get(key).map_or_else(Vec::new, |v| {
        v.lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    })
}

/// A parameter that is present and non-empty, else `None` — so `?dtype_is=` reads as "unset" rather
/// than as a glob matching nothing.
fn text<'a>(q: &'a Query, key: &str) -> Option<&'a str> {
    q.get(key).map(String::as_str).filter(|s| !s.is_empty())
}

/// A switch: `1`/`true` on, `0`/`false` off, absent off.
///
/// Anything else is a mistake rather than "off". `?only_tensors=yes` quietly comparing the metadata
/// after all is the same silent wrong answer as a mistyped filter — and here it is one keystroke from
/// a scope that says the opposite of what was asked for.
fn flag(q: &Query, key: &str) -> Result<bool> {
    match q.get(key).map(String::as_str) {
        None | Some("0" | "false") => Ok(false),
        Some("1" | "true") => Ok(true),
        Some(other) => Err(anyhow::anyhow!(
            "{key}={other:?} is not a switch — use {key}=1 or {key}=0"
        )),
    }
}

impl DiffScope {
    /// The scope rendered back as `diff` flags, for the copyable command the report offers.
    ///
    /// Without this the handover was a lie: the browser showed nineteen tensors and the command it
    /// gave you compared all 117,664. Rendering from the compiled scope's *inputs* rather than
    /// re-reading the query keeps the two in step.
    pub(crate) fn cli_args(&self) -> Vec<String> {
        let mut out = Vec::new();
        for g in &self.name_globs {
            out.push("--name".to_string());
            out.push(g.clone());
        }
        if let Some(csv) = &self.names_csv {
            out.push("--names".to_string());
            out.push(csv.clone());
        }
        if let Some(d) = &self.dtype_is {
            out.push("--dtype-is".to_string());
            out.push(d.clone());
        }
        if let Some(s) = &self.shape_is {
            out.push("--shape-is".to_string());
            out.push(s.clone());
        }
        for rule in &self.map_rules {
            out.push("--map".to_string());
            out.push(rule.clone());
        }
        if self.only_tensors {
            out.push("--only-tensors".to_string());
        }
        if self.align_fused.is_some() {
            out.push("--align-fused".to_string());
        }
        // `names_list` has no single-flag equivalent — `--names-from` takes a path, and the pasted
        // content is not one. Folded into `--names` so the command still reproduces the selection.
        if let Some(lines) = &self.names_lines {
            let joined: Vec<&str> = lines
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect();
            if !joined.is_empty() {
                out.push("--names".to_string());
                out.push(joined.join(","));
            }
        }
        out
    }

    /// Read a scope from a request's query. An invalid glob or rename rule is an error, so the UI can
    /// say which pattern was rejected instead of showing an empty diff.
    pub(crate) fn from_query(q: &Query) -> Result<Self> {
        let name = list(q, "name");
        let filter = crate::compare::tensor_filter(&crate::compare::ScopeText {
            name: &name,
            names_csv: text(q, "names"),
            names_lines: text(q, "names_list"),
            dtype_is: text(q, "dtype_is"),
            shape_is: text(q, "shape_is"),
        })?;
        let map_rules = list(q, "map");
        let map = crate::compare::name_map(&map_rules, None, text(q, "map_json"))?;
        let align_fused = if flag(q, "align_fused")? {
            Some(NameMap::from_pairs(crate::diff::fused_layout_rules())?)
        } else {
            None
        };
        Ok(Self {
            filter,
            map,
            only_tensors: flag(q, "only_tensors")?,
            align_fused,
            subtree: text(q, "subtree").map(str::to_string),
            subtree_new: text(q, "subtree_new").map(str::to_string),
            name_globs: name,
            names_csv: text(q, "names").map(str::to_string),
            names_lines: text(q, "names_list").map(str::to_string),
            dtype_is: text(q, "dtype_is").map(str::to_string),
            shape_is: text(q, "shape_is").map(str::to_string),
            map_rules,
        })
    }

    /// Whether anything narrows this comparison.
    pub(crate) fn is_active(&self) -> bool {
        self.filter.is_active() || !self.map.is_empty() || self.only_tensors || self.reroots()
    }

    /// Which subtree each side is compared from — `(baseline, newer)`, either or both `None`.
    pub(crate) fn subtrees(&self) -> (Option<&str>, Option<&str>) {
        (self.subtree.as_deref(), self.subtree_new.as_deref())
    }

    /// Whether either side is re-rooted, which is what makes the trees have to be rebuilt.
    pub(crate) fn reroots(&self) -> bool {
        self.subtree.is_some() || self.subtree_new.is_some()
    }

    /// Re-root two summaries, each at its own subtree. `Err` names the side whose prefix matched
    /// nothing — a prefix that selects no tensor is a typo, and an empty comparison would hide it.
    ///
    /// Before the rename rules and the filter, as on the command line: the sub-paths are the names
    /// every later rule is written against.
    pub(crate) fn reroot_sides(
        &self,
        old: &mut CheckpointSummary,
        new: &mut CheckpointSummary,
    ) -> Result<()> {
        for (sum, prefix, side) in [
            (old, self.subtree.as_deref(), "the baseline"),
            (new, self.subtree_new.as_deref(), "the newer side"),
        ] {
            if let Some(prefix) = prefix
                && sum.reroot(prefix) == 0
            {
                anyhow::bail!(
                    "'#{prefix}' matched no tensors in {side} — no names start with '{prefix}.'"
                );
            }
        }
        Ok(())
    }

    /// One side's tensors re-rooted, for rebuilding its tree. Borrowed when that side has no subtree —
    /// a 117k-tensor list is not something to clone for nothing.
    pub(crate) fn reroot_tensors<'a>(
        &self,
        side: Sub,
        tensors: &'a [crate::tree::TensorInfo],
    ) -> std::borrow::Cow<'a, [crate::tree::TensorInfo]> {
        let prefix = match side {
            Sub::Baseline => self.subtree.as_deref(),
            Sub::Newer => self.subtree_new.as_deref(),
        };
        // The same re-keying the CLI does (`main::scope_tensors`), so a subtree means one thing.
        prefix.map_or(std::borrow::Cow::Borrowed(tensors), |p| {
            std::borrow::Cow::Owned(crate::scope_tensors(tensors, p))
        })
    }

    /// Whether an unfused/fused alignment is being applied.
    pub(crate) fn aligns_fused(&self) -> bool {
        self.align_fused.is_some()
    }

    /// Apply the alignment to a summary, in place. A no-op unless `align_fused=1`.
    fn align(&self, sum: &mut CheckpointSummary) {
        if let Some(rules) = &self.align_fused {
            rules.remap_summary_with(sum, crate::diff::OnCollision::Fold);
        }
    }

    /// Whether any rule rewrites the names — `--map`, or the fused alignment. The side-by-side rebuilds
    /// its tree from the renamed names when so (see [`Self::rename_tensors`]).
    pub(crate) fn has_rename_rules(&self) -> bool {
        !self.map.is_empty() || self.align_fused.is_some()
    }

    /// The baseline's tensors and metadata with the rename rules applied, ready to rebuild a tree from.
    ///
    /// Renaming a *tree* is not rewriting its leaves: the groups above a leaf are named from the segments
    /// of the name it used to have, so a rewritten leaf sits under a path that no longer describes it —
    /// and the two sides stop lining up for the very reason the rules were given. So the names are mapped
    /// here and the tree is rebuilt from them (`difftree::tree_from_tensors`).
    ///
    /// Returns the renamed tensors and the names two rules collapsed onto one, which silently lose a
    /// tensor from the comparison if unreported.
    pub(crate) fn rename_tensors(&self, tensors: &[crate::tree::TensorInfo]) -> Renamed {
        // An unfused/fused alignment folds: the 256 per-expert tensors become **one** leaf standing for
        // all of them, labelled `×256`, because that is what the fused side has one of. Dropping the
        // other 255 instead (what a collision does) would show a tree missing the tensors it is
        // supposed to be accounting for.
        if let Some(rules) = &self.align_fused {
            let mut folded: Vec<crate::tree::TensorInfo> = Vec::new();
            let mut at: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            let mut parts: std::collections::BTreeMap<String, usize> =
                std::collections::BTreeMap::new();
            for t in tensors {
                let name = rules.map(&t.name).into_owned();
                *parts.entry(name.clone()).or_insert(0) += 1;
                // The leaf stands for one more tensor: add its bytes and elements, so the row's size
                // is the size of everything behind it. The first of them creates the leaf.
                if let Some(&i) = at.get(&name) {
                    if let Some(into) = folded.get_mut(i) {
                        into.size_bytes += t.size_bytes;
                        into.num_elements += t.num_elements;
                    }
                } else {
                    at.insert(name.clone(), folded.len());
                    folded.push(crate::tree::TensorInfo { name, ..t.clone() });
                }
            }
            parts.retain(|_, n| *n > 1);
            // A fold is not a collision: nothing was lost, so there is nothing to warn about.
            return Renamed {
                tensors: folded,
                collisions: Vec::new(),
                folds: parts,
            };
        }
        if self.map.is_empty() {
            return Renamed {
                tensors: tensors.to_vec(),
                collisions: Vec::new(),
                folds: std::collections::BTreeMap::new(),
            };
        }
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut collisions = Vec::new();
        let mut out = Vec::with_capacity(tensors.len());
        for t in tensors {
            let mapped = self.map.map(&t.name).into_owned();
            if let Some(n) = seen.get_mut(&mapped) {
                *n += 1;
                collisions.push(mapped.clone());
                continue;
            }
            seen.insert(mapped.clone(), 1);
            let mut t = t.clone();
            t.name = mapped;
            out.push(t);
        }
        collisions.sort_unstable();
        collisions.dedup();
        Renamed {
            tensors: out,
            collisions,
            folds: std::collections::BTreeMap::new(),
        }
    }

    /// Whether metadata is compared: `--only-tensors` turns it off, and so does *any* filter — see
    /// [`Self::compare`] for why that is the CLI's rule rather than an oversight.
    pub(crate) fn keeps_metadata(&self) -> bool {
        !self.only_tensors && !self.filter.is_active()
    }

    /// Why the metadata section is empty, when it is — `None` when metadata *was* compared.
    ///
    /// The CLI prints `metadata: not compared (filtered subset)`; the browser printed `Metadata (0)`,
    /// which says the opposite thing: that nothing differed. Stated by the side that knows the rule
    /// rather than re-derived from the parameters, so the two cannot disagree about it.
    ///
    /// A filter is named before `--only-tensors` for the same reason the CLI names it first: under a
    /// filter the metadata is out of scope whether or not `--only-tensors` was also given.
    /// Whether a *filter* narrows the comparison — the CLI's `DiffOpts::filtered`, which decides both
    /// the metadata rule and whether whole-prefix S3 changes count. A rename map alone is not one: it
    /// lines two schemes up without dropping anything.
    pub(crate) fn is_filtered(&self) -> bool {
        self.filter.is_active()
    }

    pub(crate) fn metadata_note(&self) -> Option<&'static str> {
        if self.is_filtered() {
            Some("filtered subset")
        } else if self.only_tensors {
            Some("--only-tensors")
        } else {
            None
        }
    }

    /// Narrow a tree to a set of tensor names.
    ///
    /// The set comes from [`Self::compare`]'s `matched`, computed over **both** summaries — not from
    /// testing this tree alone. A dtype or shape glob selects a tensor when *either* side matches, so a
    /// retyped tensor is in scope; asking one side in isolation would drop it from whichever side does
    /// not match and quietly turn a change into a removal.
    ///
    /// Pruning before alignment, so the counts, the `(N differ)` badges and the difference list all
    /// describe the selected subset — see `difftree::prune`.
    pub(crate) fn prune_tree(
        &self,
        nodes: &[crate::tree::TreeNode],
        keep: &std::collections::HashSet<String>,
    ) -> Vec<crate::tree::TreeNode> {
        let selects = |name: &str| keep.contains(name);
        crate::difftree::prune(nodes, &selects, self.keeps_metadata())
    }

    /// Apply the scope to two summaries and compare them — the CLI's own order.
    ///
    /// Renaming first, then filtering: the rules exist to make two naming schemes line up, so the
    /// filter's globs are written against the *renamed* names, as they are on the command line.
    /// The tensor names that **line up on both sides** once the alignment is applied.
    ///
    /// What the browser's exact-name picker offers. It is the alignment's answer, not the filter's:
    /// the point of the list is to choose names *from* it, so the name globs and the dtype/shape
    /// selectors are deliberately not applied — narrowing the list by the very thing being edited
    /// would make it impossible to widen a selection.
    ///
    /// The intersection rather than the union, for the same reason: a name that exists on one side
    /// only is not a comparison of anything, and 79,732 one-sided names would bury the ones that are.
    /// Re-rooting happens in the handler, before this, exactly as it does for the report.
    pub(crate) fn aligned_names(
        &self,
        mut old: CheckpointSummary,
        mut new: CheckpointSummary,
    ) -> Vec<String> {
        self.align(&mut old);
        self.align(&mut new);
        self.map.remap_summary(&mut old);
        let mut names: Vec<String> = old
            .tensors
            .keys()
            .filter(|n| new.tensors.contains_key(*n))
            .cloned()
            .collect();
        names.sort_unstable();
        names
    }

    pub(crate) fn compare(&self, mut old: CheckpointSummary, mut new: CheckpointSummary) -> Scoped {
        // No `#subtree` re-rooting here: the handlers do it first, because a prefix that matches nothing
        // has to be a 400 rather than an empty comparison — and because doing it twice would empty
        // everything (the sub-paths no longer start with the prefix).
        // The layout alignment first, and on both sides: it says what the two *layouts* are, which is a
        // fact about the pair rather than about which side is the baseline (see `align_fused`).
        self.align(&mut old);
        self.align(&mut new);
        let rename_collisions = self.map.remap_summary(&mut old);
        // Counted before filtering, so "matched 19 of 117,664" can say what the 19 came out of.
        let total = {
            let mut names: Vec<&String> = old.tensors.keys().chain(new.tensors.keys()).collect();
            names.sort_unstable();
            names.dedup();
            names.len()
        };
        let active = self.filter.is_active();
        self.filter.apply(&mut old, &mut new);
        // **A scoped diff does not compare metadata**, and neither does `--only-tensors`.
        //
        // The CLI's rule, from `DiffOpts { metadata: !only_tensors && !filtered }`, and what `--name`'s
        // own help promises: "Scopes the whole diff to the matching subset; metadata is not compared."
        // It is not a tensor, so no glob can select it, and reporting every metadata change alongside
        // nineteen chosen tensors buries them.
        //
        // Cleared from the comparison rather than hidden at render time, which is what the CLI does. The
        // aligned tree has no "section" to suppress — it would have to drop rows — so suppression would
        // have to be remembered in two views, and the one that forgot would disagree with the report.
        // Emptying the input makes both correct by construction.
        if self.only_tensors || active {
            old.metadata.clear();
            new.metadata.clear();
        }
        let matched = active.then(|| {
            let mut names: Vec<String> = old
                .tensors
                .keys()
                .chain(new.tensors.keys())
                .cloned()
                .collect();
            names.sort_unstable();
            names.dedup();
            Matched {
                selected: names.len(),
                total,
                names,
            }
        });
        Scoped {
            report: crate::diff::compare(&old, &new),
            matched,
            rename_collisions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{MetaVal, TensorSig};

    fn q(pairs: &[(&str, &str)]) -> Query {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    fn sig(dtype: &str, shape: &[usize]) -> TensorSig {
        TensorSig {
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
        }
    }

    fn summary(tensors: &[(&str, TensorSig)], meta: &[(&str, &str)]) -> CheckpointSummary {
        CheckpointSummary {
            tensors: tensors
                .iter()
                .map(|(n, s)| ((*n).to_string(), s.clone()))
                .collect(),
            metadata: meta
                .iter()
                .map(|(n, v)| {
                    (
                        (*n).to_string(),
                        MetaVal {
                            value: (*v).to_string(),
                            value_type: "str".to_string(),
                        },
                    )
                })
                .collect(),
            // A footprint per tensor (one byte per element), so a scoped comparison's totals can be
            // asserted to cover the selected tensors only.
            footprints: tensors
                .iter()
                .map(|(n, s)| {
                    let params: usize = s.shape.iter().product();
                    (
                        (*n).to_string(),
                        crate::diff::Footprint {
                            bytes: params,
                            params,
                            parts: 1,
                        },
                    )
                })
                .collect(),
        }
    }

    fn pair() -> (CheckpointSummary, CheckpointSummary) {
        let old = summary(
            &[
                ("model.layers.0.w", sig("F16", &[4])),
                ("model.layers.1.w", sig("F16", &[4])),
                ("model.layers.1.b", sig("F32", &[8])),
                ("lm_head.weight", sig("F16", &[2, 2])),
            ],
            &[("format", "pt")],
        );
        let new = summary(
            &[
                ("model.layers.0.w", sig("F16", &[4])),
                ("model.layers.1.w", sig("BF16", &[4])),
                ("model.layers.1.b", sig("F32", &[8])),
                ("lm_head.weight", sig("F16", &[2, 2])),
            ],
            &[("format", "safetensors")],
        );
        (old, new)
    }

    #[test]
    fn no_parameters_is_an_unscoped_comparison() {
        let scope = DiffScope::from_query(&q(&[])).expect("an empty query is a valid scope");
        assert!(!scope.is_active());
        let (old, new) = pair();
        let out = scope.compare(old, new);
        assert!(
            out.matched.is_none(),
            "with nothing narrowing it there is no match line to show"
        );
        assert_eq!(out.report.tensors_changed.len(), 1, "layers.1.w retyped");
        assert_eq!(out.report.meta_changed.len(), 1, "format changed");
    }

    /// The case this module exists for: a glob scoping the comparison, with the CLI's own counts.
    #[test]
    fn a_name_glob_scopes_the_comparison_and_says_what_it_matched() {
        let scope = DiffScope::from_query(&q(&[("name", "model.layers.1.*")])).expect("valid glob");
        let (old, new) = pair();
        let out = scope.compare(old, new);
        let matched = out.matched.expect("a scoped comparison reports its match");
        assert_eq!(matched.selected, 2, "layers.1.w and layers.1.b");
        assert_eq!(matched.total, 4, "out of four distinct tensors");
        assert_eq!(matched.names, ["model.layers.1.b", "model.layers.1.w"]);
        // And the report covers only those: `layers.0.w` matched but is unchanged, so the proof is
        // that the *unchanged* count fell to the one selected match.
        assert_eq!(out.report.tensors_changed.len(), 1);
        assert_eq!(out.report.tensors_unchanged, 1, "only layers.1.b remains");
    }

    #[test]
    fn a_negated_glob_excludes() {
        let scope = DiffScope::from_query(&q(&[("name", "*\n!*.b")])).expect("valid globs");
        let out = scope.compare(pair().0, pair().1);
        let matched = out.matched.expect("scoped");
        assert!(
            !matched
                .names
                .iter()
                .any(|n| n.rsplit('.').next() == Some("b")),
            "`!*.b` should exclude: {:?}",
            matched.names
        );
        assert_eq!(matched.selected, 3);
    }

    #[test]
    fn dtype_and_shape_globs_scope_too() {
        // Either side matching is enough, so a retyped tensor is still selected.
        let out = DiffScope::from_query(&q(&[("dtype_is", "bf*")]))
            .expect("valid")
            .compare(pair().0, pair().1);
        assert_eq!(out.matched.expect("scoped").names, ["model.layers.1.w"]);

        let out = DiffScope::from_query(&q(&[("shape_is", "2,2")]))
            .expect("valid")
            .compare(pair().0, pair().1);
        assert_eq!(out.matched.expect("scoped").names, ["lm_head.weight"]);
    }

    #[test]
    fn an_exact_name_list_scopes_to_those_names() {
        let out = DiffScope::from_query(&q(&[("names", "lm_head.weight, model.layers.0.w")]))
            .expect("valid")
            .compare(pair().0, pair().1);
        assert_eq!(
            out.matched.expect("scoped").names,
            ["lm_head.weight", "model.layers.0.w"]
        );

        // The pasted-file form, with comments, as `--names-from` accepts.
        let out = DiffScope::from_query(&q(&[("names_list", "# pick one\nlm_head.weight\n\n")]))
            .expect("valid")
            .compare(pair().0, pair().1);
        assert_eq!(out.matched.expect("scoped").names, ["lm_head.weight"]);
    }

    /// A scoped diff does not compare metadata — the CLI's rule, and what `--name`'s help promises.
    #[test]
    fn a_name_filter_also_drops_the_metadata_comparison() {
        let out = DiffScope::from_query(&q(&[("name", "model.layers.1.*")]))
            .expect("valid")
            .compare(pair().0, pair().1);
        assert!(
            out.report.meta_changed.is_empty(),
            "`format` changed, but a scoped diff does not report metadata"
        );
        assert_eq!(out.report.meta_unchanged, 0);
        assert_eq!(
            out.report.tensors_changed.len(),
            1,
            "tensors still compared"
        );
    }

    #[test]
    fn only_tensors_drops_the_metadata_comparison() {
        let out = DiffScope::from_query(&q(&[("only_tensors", "1")]))
            .expect("valid")
            .compare(pair().0, pair().1);
        assert!(
            out.report.meta_changed.is_empty() && out.report.meta_added.is_empty(),
            "metadata should not be compared at all"
        );
        assert_eq!(out.report.tensors_changed.len(), 1, "tensors still are");
    }

    /// Rename rules run *before* the filter, so a glob is written against the names as they will read.
    #[test]
    fn a_rename_rule_lines_the_two_schemes_up_before_filtering() {
        let old = summary(&[("blocks.1.w", sig("F16", &[4]))], &[]);
        let new = summary(&[("model.layers.1.w", sig("F16", &[4]))], &[]);
        // Without a rule this is one added and one removed.
        let bare = DiffScope::from_query(&q(&[])).expect("valid").compare(
            summary(&[("blocks.1.w", sig("F16", &[4]))], &[]),
            summary(&[("model.layers.1.w", sig("F16", &[4]))], &[]),
        );
        assert_eq!(
            (
                bare.report.tensors_added.len(),
                bare.report.tensors_removed.len()
            ),
            (1, 1)
        );

        let scope = DiffScope::from_query(&q(&[
            ("map", r"^blocks\.=>model.layers."),
            ("name", "model.layers.1.*"),
        ]))
        .expect("valid rule and glob");
        let out = scope.compare(old, new);
        assert_eq!(out.report.tensors_unchanged, 1, "the two now line up");
        assert!(out.report.tensors_added.is_empty());
        assert!(out.report.tensors_removed.is_empty());
        assert_eq!(
            out.matched.expect("scoped").selected,
            1,
            "the glob matched the renamed name"
        );
    }

    #[test]
    fn a_bad_glob_is_reported_rather_than_matching_nothing() {
        let Err(err) = DiffScope::from_query(&q(&[("dtype_is", "[")])) else {
            panic!("an unclosed character class is not a valid dtype glob");
        };
        assert!(
            format!("{err:#}").contains("dtype"),
            "the message should name what was wrong: {err:#}"
        );
    }

    /// The copyable command must reproduce what is on screen, scope included.
    #[test]
    fn the_scope_renders_back_as_the_flags_that_produced_it() {
        let scope = DiffScope::from_query(&q(&[
            ("name", "model.layers.1.*\n!*.bias"),
            ("dtype_is", "F*"),
            ("shape_is", "768,**"),
            ("map", r"^blocks\.=>model.layers."),
            ("only_tensors", "1"),
        ]))
        .expect("valid");
        assert_eq!(
            scope.cli_args(),
            [
                "--name",
                "model.layers.1.*",
                "--name",
                "!*.bias",
                "--dtype-is",
                "F*",
                "--shape-is",
                "768,**",
                "--map",
                r"^blocks\.=>model.layers.",
                "--only-tensors",
            ]
        );
        // An unscoped comparison adds nothing, so the command stays the short form.
        assert!(
            DiffScope::from_query(&q(&[]))
                .expect("valid")
                .cli_args()
                .is_empty()
        );
    }

    /// A pasted name list has no `--names-from` equivalent (that flag takes a path), so it folds into
    /// `--names` — otherwise the command silently drops the selection.
    #[test]
    fn a_pasted_name_list_becomes_an_explicit_names_flag() {
        let scope =
            DiffScope::from_query(&q(&[("names_list", "# pick\na.w\n\nb.w\n")])).expect("valid");
        assert_eq!(scope.cli_args(), ["--names", "a.w,b.w"]);
    }

    /// An empty parameter means "unset", not "a pattern that matches nothing" — a UI that always sends
    /// its boxes would otherwise scope every comparison to zero tensors.
    #[test]
    fn empty_parameters_do_not_scope() {
        let scope = DiffScope::from_query(&q(&[
            ("name", ""),
            ("names", ""),
            ("dtype_is", ""),
            ("shape_is", ""),
            ("only_tensors", "0"),
        ]))
        .expect("valid");
        assert!(!scope.is_active(), "blank boxes leave the comparison whole");
    }

    /// **The web offers the same alignment the CLI does, through one parameter.**
    ///
    /// `align_fused=1` is the browser's `--align-fused`: the same rules, the same fold, and the same
    /// copyable command — so a comparison set up in a browser reproduces in a terminal.
    #[test]
    fn align_fused_folds_the_unfused_side_and_offers_the_flag_back() {
        let q: Query = std::iter::once(("align_fused".to_string(), "1".to_string())).collect();
        let scope = DiffScope::from_query(&q).expect("the canonical rules compile");
        assert!(scope.aligns_fused());
        assert!(scope.cli_args().contains(&"--align-fused".to_string()));

        let unfused = summary(
            &[
                (
                    "model.layers.0.block_sparse_moe.experts.0.w2.weight",
                    sig("U8", &[4, 2]),
                ),
                (
                    "model.layers.0.block_sparse_moe.experts.1.w2.weight",
                    sig("U8", &[4, 2]),
                ),
            ],
            &[],
        );
        let fused = summary(
            &[(
                "model.layers.0.block_sparse_moe.experts.down_proj.weight",
                sig("U8", &[2, 4, 2]),
            )],
            &[],
        );
        let scoped = scope.compare(unfused, fused);
        assert!(
            scoped.report.tensors_removed.is_empty(),
            "the two experts fold onto the fused tensor rather than being removed"
        );
        assert_eq!(scoped.report.tensors_changed.len(), 1);
        assert_eq!(
            scoped
                .report
                .folded
                .get("model.layers.0.block_sparse_moe.experts.down_proj.weight"),
            Some(&(2, 1)),
            "the row records what folded: {:?}",
            scoped.report.folded
        );
    }

    /// `#subtree`: two checkpoints under different namespaces line up, and the siblings are out of
    /// scope rather than "removed".
    #[test]
    fn a_subtree_reroot_lines_up_two_namespaces() {
        let scope = DiffScope::from_query(&q(&[("subtree", "language_model")])).expect("compile");
        let mut old = summary(
            &[
                ("language_model.model.norm.weight", sig("F16", &[4])),
                // A sibling of the subtree: out of scope, not a removal.
                ("vision_tower.patch.weight", sig("F16", &[8])),
            ],
            &[],
        );
        let mut new = summary(&[("model.norm.weight", sig("F16", &[4]))], &[]);
        scope
            .reroot_sides(&mut old, &mut new)
            .expect("the prefix matches");
        let scoped = scope.compare(old, new);
        assert!(
            !scoped.report.has_differences(),
            "the subtree matches the other side's root, and the sibling is out of scope, not removed: \
             {} added / {} removed",
            scoped.report.tensors_added.len(),
            scoped.report.tensors_removed.len()
        );
    }

    /// A prefix that selects nothing is a typo, and the message says which side it was on — an empty
    /// report would look like two checkpoints with nothing in common.
    #[test]
    fn a_subtree_that_matches_nothing_is_an_error_naming_its_side() {
        let scope = DiffScope::from_query(&q(&[("subtree_new", "vision_tower")])).expect("compile");
        let mut old = summary(&[("model.w", sig("F16", &[4]))], &[]);
        let mut new = summary(&[("model.w", sig("F16", &[4]))], &[]);
        let e = scope
            .reroot_sides(&mut old, &mut new)
            .expect_err("nothing starts with vision_tower.");
        let msg = format!("{e:#}");
        assert!(msg.contains("vision_tower"), "{msg}");
        assert!(msg.contains("the newer side"), "{msg}");
    }

    /// Re-rooting narrows the totals to the subtree — the report's header describes what it compared.
    #[test]
    fn a_subtree_narrows_the_totals_to_itself() {
        let scope = DiffScope::from_query(&q(&[("subtree", "language_model")])).expect("compile");
        let mut old = summary(
            &[
                ("language_model.w", sig("F16", &[4])),
                ("vision_tower.w", sig("F16", &[100])),
            ],
            &[],
        );
        let mut new = summary(&[("w", sig("F16", &[4]))], &[]);
        scope.reroot_sides(&mut old, &mut new).expect("matches");
        assert_eq!(
            old.total_params(),
            4,
            "the 100-element sibling is out of scope"
        );
    }

    /// The tree side folds too: one leaf per fused name, labelled with what it stands for.
    #[test]
    fn align_fused_folds_the_tree_into_one_labelled_leaf() {
        let q: Query = std::iter::once(("align_fused".to_string(), "1".to_string())).collect();
        let scope = DiffScope::from_query(&q).expect("compile");
        let t = |name: &str| crate::tree::TensorInfo {
            name: name.to_string(),
            dtype: "U8".to_string(),
            shape: vec![4, 2],
            size_bytes: 8,
            num_elements: 8,
            storage: crate::tree::Storage::Unknown,
            source_path: "s".to_string(),
            layout: crate::tree::Layout::None,
        };
        let renamed = scope.rename_tensors(&[
            t("model.layers.0.block_sparse_moe.experts.0.w2.weight"),
            t("model.layers.0.block_sparse_moe.experts.1.w2.weight"),
        ]);
        assert_eq!(renamed.tensors.len(), 1, "one leaf, not two");
        assert_eq!(
            renamed.tensors[0].name,
            "model.layers.0.block_sparse_moe.experts.down_proj.weight"
        );
        assert_eq!(
            renamed.tensors[0].size_bytes, 16,
            "the leaf carries both parts' bytes"
        );
        assert_eq!(renamed.folds.values().copied().collect::<Vec<_>>(), vec![2]);
        assert!(
            renamed.collisions.is_empty(),
            "a fold loses nothing, so it warns about nothing"
        );
    }
}
