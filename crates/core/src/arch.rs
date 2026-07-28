//! **Inferred architecture**: what a model card's summary table says, derived from the
//! tensors instead of read from prose.
//!
//! A card tells you 93 layers, 896 experts, a 160K vocabulary. All of that is *in* the
//! checkpoint — in the names, shapes and dtypes — and deriving it has two advantages over
//! trusting the card: it works for a checkpoint that has no card, and it disagrees when the
//! card is wrong or stale.
//!
//! **What it will not do is guess.** Some facts genuinely are not in the tensors: how many
//! experts a router selects per token, the context length, the activation function's name.
//! Those are listed in [`Architecture::not_in_tensors`] rather than approximated, because a
//! plausible wrong number here is worse than an admitted gap — it would be indistinguishable
//! from a derived one.
//!
//! ## Stored vs logical parameters
//!
//! The one place this deliberately reports *two* numbers. A safetensors header describes
//! **stored elements**: a `U8` tensor of N bytes is N elements. When weights are packed
//! below a byte — MXFP4 at two values per byte, 4-bit at two, 3-bit fused-codebook at
//! more — the number of **logical parameters** is larger. Kimi-K3 is the worked example:
//! the header sums to 1.5T stored, its card says 2.8T, and the Hub's own API agrees with
//! the card. Neither is wrong; they answer different questions, and a summary that printed
//! one without naming which would be quietly misleading.

use std::collections::{BTreeMap, BTreeSet};

use crate::tree::TensorInfo;

/// One inferred fact, and how confident the inference is.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Fact {
    /// What was inferred, e.g. `"93"` or `"896 per layer"`.
    pub value: String,
    /// The tensor evidence it came from, so a reader can check it rather than trust it.
    pub from: String,
    /// A keyboard shortcut this fact points at, when one is relevant — carried as data, not
    /// spelled into `from`, so each frontend renders it in its own idiom (the terminal as a
    /// key chip like every other hint, the browser however it shows keys). Prose containing
    /// `` `k` `` looked like markdown and rendered literally in both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

impl Fact {
    fn new(value: impl Into<String>, from: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            from: from.into(),
            key: None,
        }
    }

    /// The shortcut this fact points at.
    fn with_key(mut self, key: &str) -> Self {
        self.key = Some(key.to_string());
        self
    }
}

/// The architecture summary derived from a tensor list.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Architecture {
    /// Facts in presentation order, each with its evidence.
    pub facts: Vec<(String, Fact)>,
    /// Summary rows a model card has that the tensors cannot supply, with why.
    pub not_in_tensors: Vec<(String, String)>,
}

impl Architecture {
    /// Look up a fact by label — for tests and for a frontend that wants one row.
    #[must_use]
    pub fn get(&self, label: &str) -> Option<&Fact> {
        self.facts.iter().find(|(l, _)| l == label).map(|(_, f)| f)
    }

    fn push(&mut self, label: &str, fact: Fact) {
        self.facts.push((label.to_string(), fact));
    }
}

/// Whether a tensor is quantization **metadata** rather than a parameter — a scale, a
/// zero-point, a codebook.
///
/// These are excluded from both parameter counts, and getting it wrong is measurable: for
/// Kimi-K3, counting the `weight_scale` sidecars put the logical total at 2.9T against the
/// card's 2.8T. Excluding them and doubling only the packed weights reproduces the Hub's own
/// figure (2.7227T of U8) to four decimal places, which is how this rule was confirmed
/// rather than guessed.
fn is_quant_metadata(name: &str) -> bool {
    [
        "weight_scale",
        "qscale",
        "codebook",
        "zero_point",
        "zeros",
        "g_idx",
        "scales",
    ]
    .iter()
    .any(|m| name.contains(m))
}

/// How many logical values each stored element of `dtype` holds, when the name says the
/// tensor is packed.
///
/// Deliberately conservative: only a tensor whose *name* marks it as packed
/// (`weight_packed`, `qweight`, …) alongside a byte-width dtype is counted as holding more
/// than one value. Guessing from the dtype alone would inflate every `U8` scale tensor.
fn values_per_element(name: &str, dtype: &str, bits: Option<u32>) -> u32 {
    let packed = name.contains("weight_packed")
        || name.contains("qweight")
        || name.ends_with(".packed")
        || name.contains("packed_weight");
    if !packed {
        return 1;
    }
    let stored_bits = match dtype {
        "U8" | "I8" => 8,
        "U16" | "I16" | "F16" | "BF16" => 16,
        "U32" | "I32" | "F32" => 32,
        _ => return 1,
    };
    // `bits` is the packing width when the checkpoint says so (a quantization schema); 4 is
    // the common default for a packed byte tensor, and never claim fewer than one value.
    bits.map_or(stored_bits / 4, |b| stored_bits / b.max(1))
        .max(1)
}

/// Infer what can be inferred from `tensors`. Pure; no disk, no config.
///
/// `packed_bits` is the packing width if the checkpoint's metadata stated one; `None` means
/// the common 4-bit assumption is used for name-marked packed tensors, and the report says
/// which was used.
#[must_use]
pub fn infer(tensors: &[TensorInfo], packed_bits: Option<u32>) -> Architecture {
    let mut a = Architecture::default();
    if tensors.is_empty() {
        return a;
    }
    // Prefer a width the shapes prove over the caller's (or the default) assumption.
    let packed_bits = packed_bits.or_else(|| bits_from_shapes(tensors));

    // ---- parameters, stored and logical --------------------------------------------
    let stored: u64 = tensors.iter().map(|t| t.num_elements as u64).sum();
    // Parameters exclude quantization metadata: a scale block is not a weight. See
    // `is_quant_metadata` for how that was verified against a real checkpoint.
    let params: Vec<&TensorInfo> = tensors
        .iter()
        .filter(|t| !is_quant_metadata(&t.name))
        .collect();
    let logical: u64 = params
        .iter()
        .map(|t| {
            t.num_elements as u64 * u64::from(values_per_element(&t.name, &t.dtype, packed_bits))
        })
        .sum();
    let metadata_elements = stored - params.iter().map(|t| t.num_elements as u64).sum::<u64>();
    a.push(
        "Stored elements",
        Fact::new(
            crate::utils::format_parameters(stored as usize),
            "Σ over tensor shapes — what the header describes",
        ),
    );
    if metadata_elements > 0 {
        a.push(
            "Quantization metadata",
            Fact::new(
                crate::utils::format_parameters(metadata_elements as usize),
                "scale / zero-point / codebook elements — counted separately because they \
                 are not parameters",
            ),
        );
    }
    if logical > stored - metadata_elements {
        let assumed = packed_bits.map_or_else(
            || {
                "assuming 4-bit packing (no width stated, and none provable from the shapes)"
                    .to_string()
            },
            |b| format!("{b}-bit packing, from the packed-weight / scale shape ratio"),
        );
        a.push(
            "Logical parameters",
            Fact::new(
                crate::utils::format_parameters(logical as usize),
                format!("packed weights hold more than one value per stored element — {assumed}"),
            ),
        );
    }

    // ---- how much of the checkpoint is repetition ------------------------------------
    // The question "what does 31.3K tensors in 19 families mean?" belongs where there is
    // room to answer it, so the number carries its explanation here rather than only in the
    // compact view's header.
    let families = crate::compact::compact_tree(tensors).counts.len();
    if families > 0 && families < tensors.len() {
        a.push(
            "Tensor families",
            Fact::new(
                format!(
                    "{families} kinds across {} tensors",
                    crate::utils::format_parameters(tensors.len())
                ),
                "tensors whose names differ only by an index — a layer or expert number — \
                 are one kind repeated; the compact view shows each as a single row",
            )
            .with_key("k"),
        );
    }

    // ---- layers, and which of them differ -------------------------------------------
    let layers = layer_indices(tensors);
    if !layers.is_empty() {
        a.push(
            "Layers",
            Fact::new(
                layers.len().to_string(),
                "distinct `layers.N` indices in the tensor names",
            ),
        );
        let (dense, moe) = dense_and_moe(tensors, &layers);
        if moe > 0 {
            a.push(
                "Architecture",
                Fact::new("Mixture-of-Experts (MoE)", "layers containing `experts.N`"),
            );
            a.push(
                "Dense layers",
                Fact::new(dense.to_string(), "layers with no `experts.N` beneath them"),
            );
        }
        // Layers whose tensor sets differ are the interesting irregularity: a card writes
        // this as "69 KDA + 24 Gated MLA".
        let classes = layer_classes(tensors, &layers);
        if classes.len() > 1 {
            let mut counts: Vec<usize> = classes.values().copied().collect();
            counts.sort_unstable_by(|a, b| b.cmp(a));
            let composition = counts
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" + ");
            a.push(
                "Layer composition",
                Fact::new(
                    format!("{composition} layers of {} kinds", classes.len()),
                    "layers grouped by which tensors they have — the kinds' *names* are not \
                     in the tensors, only that they differ",
                ),
            );
        }
    }

    // ---- experts ---------------------------------------------------------------------
    if let Some(n) = max_index_after(tensors, "experts.") {
        a.push(
            "Experts",
            Fact::new(
                format!("{n} per layer"),
                "highest `experts.N` index, plus one",
            ),
        );
    }
    let shared = tensors
        .iter()
        .filter(|t| t.name.contains("shared_expert"))
        .count();
    if shared > 0 {
        a.push(
            "Shared experts",
            Fact::new(
                format!("{shared} tensors"),
                "tensors named `shared_expert*` — the *count* of experts needs their own \
                 index, which these names may not carry",
            ),
        );
    }

    // ---- embedding shapes ------------------------------------------------------------
    if let Some(t) = tensors.iter().find(|t| t.name.contains("embed_tokens")) {
        if let [vocab, hidden] = t.shape.as_slice() {
            a.push(
                "Vocabulary size",
                Fact::new(vocab.to_string(), format!("{}'s first dimension", t.name)),
            );
            a.push(
                "Hidden dimension",
                Fact::new(hidden.to_string(), format!("{}'s second dimension", t.name)),
            );
        }
        let tied = !tensors.iter().any(|t| t.name.contains("lm_head"));
        a.push(
            "Tied embeddings",
            Fact::new(
                if tied { "yes" } else { "no" },
                "whether a separate `lm_head` tensor exists",
            ),
        );
    }

    // ---- a vision tower --------------------------------------------------------------
    let vision: Vec<&TensorInfo> = tensors
        .iter()
        .filter(|t| t.name.contains("vision") || t.name.contains("visual"))
        .collect();
    if !vision.is_empty() {
        let params: u64 = vision.iter().map(|t| t.num_elements as u64).sum();
        a.push(
            "Modality",
            Fact::new("text + image", "a vision tower is present in the tensors"),
        );
        a.push(
            "Vision encoder parameters",
            Fact::new(
                crate::utils::format_parameters(params as usize),
                format!("Σ over the {} `vision*`/`visual*` tensors", vision.len()),
            ),
        );
    }

    // ---- quantization ----------------------------------------------------------------
    let mut markers = BTreeSet::new();
    for t in tensors {
        for m in [
            "weight_packed",
            "weight_scale",
            "qscale",
            "codebook",
            "qweight",
            "zeros",
        ] {
            if t.name.contains(m) {
                markers.insert(m);
            }
        }
    }
    if !markers.is_empty() {
        a.push(
            "Quantization",
            Fact::new(
                format!(
                    "packed weights with {}",
                    markers.iter().copied().collect::<Vec<_>>().join(", ")
                ),
                "sidecar tensor names — the *scheme's* name (MXFP4, AWQ, …) is not in them",
            ),
        );
    }

    // ---- what the tensors cannot say -------------------------------------------------
    a.not_in_tensors = vec![
        (
            "Selected experts per token".to_string(),
            "the router's top-k is a config value; no tensor records it".to_string(),
        ),
        (
            "Context length".to_string(),
            "a position-embedding shape bounds it at most; the trained length is config"
                .to_string(),
        ),
        (
            "Activation function".to_string(),
            "a gate+up pair implies a GLU family, but the exact function is config".to_string(),
        ),
        (
            "Activated parameters per token".to_string(),
            "needs the router's top-k, so it cannot be derived from tensors alone".to_string(),
        ),
    ];
    a
}

/// The packing width proved by a packed weight and its scale sibling.
///
/// A scale covers a fixed block of logical values, so `packed_last_dim / scale_last_dim` is
/// how many *stored bytes* one block occupies. With the near-universal block of 32 values,
/// that ratio gives the values-per-byte and hence the width: Kimi-K3's `[3072, 1792]` packed
/// against `[3072, 112]` scales is 16 bytes per block → 2 values per byte → 4 bits. The
/// derived in-features (1792 × 2 = 3584) match that model card's latent dimension, which is
/// the independent check that the reasoning holds.
///
/// `None` when no such pair exists, or the ratio implies a width outside 2–8 bits — better
/// to fall back to the stated assumption than to report a width from a coincidence.
fn bits_from_shapes(tensors: &[TensorInfo]) -> Option<u32> {
    const BLOCK: u32 = 32;
    for t in tensors {
        let Some(stem) = t.name.strip_suffix(".weight_packed") else {
            continue;
        };
        let scale = tensors
            .iter()
            .find(|s| s.name == format!("{stem}.weight_scale"))?;
        let (Some(&packed_last), Some(&scale_last)) = (t.shape.last(), scale.shape.last()) else {
            continue;
        };
        if scale_last == 0 || packed_last % scale_last != 0 {
            continue;
        }
        let bytes_per_block = u32::try_from(packed_last / scale_last).ok()?;
        if bytes_per_block == 0 {
            continue;
        }
        let values_per_byte = BLOCK / bytes_per_block;
        if values_per_byte == 0 {
            continue;
        }
        let bits = 8 / values_per_byte;
        if (2..=8).contains(&bits) {
            return Some(bits);
        }
    }
    None
}

/// The distinct `layers.N` indices present.
fn layer_indices(tensors: &[TensorInfo]) -> BTreeSet<u64> {
    tensors
        .iter()
        .filter_map(|t| index_after(&t.name, "layers."))
        .collect()
}

/// `(dense, moe)` layer counts — a layer is `MoE` when any of its tensors names an expert.
fn dense_and_moe(tensors: &[TensorInfo], layers: &BTreeSet<u64>) -> (usize, usize) {
    let mut moe = BTreeSet::new();
    for t in tensors {
        if t.name.contains("experts.")
            && let Some(l) = index_after(&t.name, "layers.")
        {
            moe.insert(l);
        }
    }
    (layers.len() - moe.len(), moe.len())
}

/// Layers grouped by the set of tensor suffixes they have → how many layers per group.
fn layer_classes(tensors: &[TensorInfo], layers: &BTreeSet<u64>) -> BTreeMap<String, usize> {
    let mut per_layer: BTreeMap<u64, BTreeSet<String>> = BTreeMap::new();
    for t in tensors {
        let Some(l) = index_after(&t.name, "layers.") else {
            continue;
        };
        // The part after the layer number, with every index blanked, so two layers with the
        // same structure hash the same however many experts they hold.
        let (template, _) = crate::diff::templatize(&t.name);
        if let Some(rest) = template.split("layers.{}").nth(1) {
            per_layer.entry(l).or_default().insert(rest.to_string());
        }
    }
    let mut classes: BTreeMap<String, usize> = BTreeMap::new();
    for l in layers {
        if let Some(sig) = per_layer.get(l) {
            let key = sig.iter().cloned().collect::<Vec<_>>().join("|");
            *classes.entry(key).or_insert(0) += 1;
        }
    }
    classes
}

/// The index immediately after `marker` in `name`, if any.
fn index_after(name: &str, marker: &str) -> Option<u64> {
    let rest = name.split(marker).nth(1)?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// One more than the highest index after `marker`, i.e. how many there are.
fn max_index_after(tensors: &[TensorInfo], marker: &str) -> Option<u64> {
    tensors
        .iter()
        .filter_map(|t| index_after(&t.name, marker))
        .max()
        .map(|m| m + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Layout, Storage};

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

    #[test]
    fn nothing_is_inferred_from_nothing() {
        let a = infer(&[], None);
        assert!(a.facts.is_empty());
        assert!(a.not_in_tensors.is_empty(), "no model, nothing to caveat");
    }

    /// The headline distinction: a packed tensor holds more logical parameters than stored
    /// elements, and the report names both rather than picking one. This is the 1.5T-vs-2.8T
    /// disagreement between a header's sum and a model card.
    #[test]
    fn packed_weights_report_stored_and_logical_separately() {
        let tensors = vec![
            t("model.layers.0.mlp.weight_packed", "U8", &[1000]),
            t("model.layers.0.mlp.weight_scale", "U8", &[10]),
        ];
        let a = infer(&tensors, None);
        let stored = a.get("Stored elements").expect("stored");
        let logical = a.get("Logical parameters").expect("logical");
        assert!(stored.value.contains("1.0K"), "{:?}", stored.value);
        // 1000 packed bytes at 4 bits = 2000 values; the scale tensor is NOT doubled.
        assert!(logical.value.contains("2.0K"), "{:?}", logical.value);
        assert!(
            logical.from.contains("4-bit"),
            "it says which assumption it used: {}",
            logical.from
        );
    }

    /// The packing width is *derived*, not assumed, when a scale sibling proves it — and the
    /// scales are excluded from the parameter counts. Both were established against Kimi-K3:
    /// `[3072, 1792]` packed with `[3072, 112]` scales is 16 bytes per 32-value block → 4
    /// bits, and excluding the scales reproduced the Hub's own parameter total exactly, where
    /// counting them overshot the model card by 0.1T.
    #[test]
    fn the_packing_width_comes_from_the_shapes_and_scales_are_not_parameters() {
        let tensors = vec![
            t("m.experts.0.w1.weight_packed", "U8", &[3072, 1792]),
            t("m.experts.0.w1.weight_scale", "U8", &[3072, 112]),
        ];
        let a = infer(&tensors, None);

        let logical = a.get("Logical parameters").expect("logical");
        assert!(
            logical.from.contains("shape ratio"),
            "the width should be derived, not assumed: {}",
            logical.from
        );
        assert!(logical.from.contains("4-bit"), "{}", logical.from);

        // 3072×1792 packed bytes × 2 values = 11,010,048 logical parameters, and the
        // 344,064 scale elements are NOT among them.
        let expected = crate::utils::format_parameters(3072 * 1792 * 2);
        assert_eq!(
            logical.value, expected,
            "packed weights doubled, scales excluded"
        );
        let meta = a
            .get("Quantization metadata")
            .expect("the scales are reported");
        assert_eq!(meta.value, crate::utils::format_parameters(3072 * 112));
    }

    /// An unpacked checkpoint reports one number, not two — claiming a "logical" count equal
    /// to the stored one would imply packing where there is none.
    #[test]
    fn an_unpacked_checkpoint_reports_one_parameter_count() {
        let a = infer(&[t("model.embed_tokens.weight", "BF16", &[100, 8])], None);
        assert!(a.get("Stored elements").is_some());
        assert!(a.get("Logical parameters").is_none());
    }

    /// The Kimi-K3 shape in miniature: 4 layers, one of them dense, experts, a vision tower,
    /// and two kinds of attention layer.
    #[test]
    fn a_moe_model_with_a_vision_tower_is_described() {
        let mut tensors = vec![
            t(
                "language_model.model.embed_tokens.weight",
                "BF16",
                &[160, 8],
            ),
            t("language_model.lm_head.weight", "BF16", &[160, 8]),
            t(
                "vision_tower.patch_embed.proj.weight",
                "BF16",
                &[8, 3, 2, 2],
            ),
        ];
        // Layer 0 is dense; 1..4 are MoE, and layer 3 has an extra attention tensor.
        tensors.push(t("language_model.model.layers.0.mlp.w", "BF16", &[8, 8]));
        for l in 1..4 {
            tensors.push(t(
                &format!("language_model.model.layers.{l}.mlp.experts.0.w"),
                "BF16",
                &[8, 8],
            ));
            tensors.push(t(
                &format!("language_model.model.layers.{l}.mlp.experts.1.w"),
                "BF16",
                &[8, 8],
            ));
        }
        tensors.push(t(
            "language_model.model.layers.3.self_attn.extra",
            "F32",
            &[4],
        ));

        let a = infer(&tensors, None);
        assert_eq!(a.get("Layers").map(|f| f.value.as_str()), Some("4"));
        assert_eq!(
            a.get("Architecture").map(|f| f.value.as_str()),
            Some("Mixture-of-Experts (MoE)")
        );
        assert_eq!(
            a.get("Dense layers").map(|f| f.value.as_str()),
            Some("1"),
            "layer 0 has no experts"
        );
        assert_eq!(
            a.get("Experts").map(|f| f.value.as_str()),
            Some("2 per layer")
        );
        assert_eq!(
            a.get("Vocabulary size").map(|f| f.value.as_str()),
            Some("160")
        );
        assert_eq!(
            a.get("Hidden dimension").map(|f| f.value.as_str()),
            Some("8")
        );
        assert_eq!(
            a.get("Tied embeddings").map(|f| f.value.as_str()),
            Some("no"),
            "there is an lm_head"
        );
        assert_eq!(
            a.get("Modality").map(|f| f.value.as_str()),
            Some("text + image")
        );
        // Layer 3 differs from 1-2, and layer 0 from both: three kinds.
        let comp = a.get("Layer composition").expect("a composition");
        assert!(comp.value.contains("3 kinds"), "{}", comp.value);
        assert!(
            comp.from.contains("not in the tensors"),
            "it admits it cannot name them: {}",
            comp.from
        );
    }

    /// The families fact answers "what does N tensors in M families mean" in place, and only
    /// appears when folding actually saves something — for a checkpoint where every tensor is
    /// its own kind, "7 kinds across 7 tensors" is noise.
    #[test]
    fn the_family_count_explains_itself_and_is_omitted_when_useless() {
        let mut tensors = Vec::new();
        for l in 0..4 {
            tensors.push(t(&format!("model.layers.{l}.mlp.w"), "BF16", &[4, 4]));
        }
        let a = infer(&tensors, None);
        let fact = a.get("Tensor families").expect("a families fact");
        assert!(fact.value.contains("across"), "{}", fact.value);
        assert!(
            fact.from.contains("differ only by an index"),
            "it should say what a family IS: {}",
            fact.from
        );
        assert_eq!(
            fact.key.as_deref(),
            Some("k"),
            "the shortcut is data, so each frontend can style it as a key"
        );
        assert!(
            !fact.from.contains('`'),
            "no markdown in prose that renders literally: {}",
            fact.from
        );

        // Nothing folds here, so the row would say "1 kinds across 1 tensors".
        let single = infer(&[t("lone.weight", "BF16", &[2])], None);
        assert!(
            single.get("Tensor families").is_none(),
            "no repetition, nothing worth reporting"
        );
    }

    /// Every fact carries its evidence, and the gaps are named rather than filled.
    #[test]
    fn facts_cite_evidence_and_gaps_are_admitted() {
        let a = infer(&[t("model.embed_tokens.weight", "BF16", &[10, 4])], None);
        for (label, fact) in &a.facts {
            assert!(
                !fact.from.is_empty(),
                "{label} should say where it came from"
            );
        }
        let gaps: Vec<&String> = a.not_in_tensors.iter().map(|(l, _)| l).collect();
        for expected in [
            "Selected experts per token",
            "Context length",
            "Activation function",
        ] {
            assert!(
                gaps.iter().any(|g| g.as_str() == expected),
                "{expected} is not in the tensors and must be listed as such: {gaps:?}"
            );
        }
    }
}
