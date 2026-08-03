//! Local (in-process) repack-equivalence verification — the same check
//! [`crate::remote::RemoteRead::verify_repack`] runs on the ssh proxy for `s3://`
//! cstorch checkpoints, but for local checkpoint files (`.safetensors` / `.hdf5`).
//!
//! Two expert-weight tensors encode the same indices in different packings. Each side's packing is a
//! [`crate::sample::PackingSchema`] — a list of bit widths: the **sparse** encoding is `[4]` (one index
//! per 16-bit word, only the low four bits used) and a **merged** one is `[3,3,3,3,3]` (five consecutive
//! experts in one word, each shifted three bits past the last). Expert `e` then lives in word
//! `e / lenP` at bit offset `offset(e % lenP)`, which for the uniform case is the familiar
//! `e / fold`, `(e % fold) * bits`.
//!
//! This module reads the raw 16-bit words locally, decodes both sides *by their own schema*, checks
//! they match, and validates each packing (bits above a side's total width must be zero). It also
//! diffs the sibling codebook / scale tensors' values. Results are the same [`RepackResult`] the
//! remote path produces, so the reporting is shared.

use crate::remote::{RepackAux, RepackFallback, RepackResult, RepackSample};
use crate::sample::PackingSchema;
use crate::tree::TensorInfo;

/// How each side packs its expert indices — the pair a decode needs.
///
/// A pair rather than one schema plus a fold, because the two sides are exactly what differs: the point
/// of the verification is that the same indices are packed *differently* on each side.
#[derive(Clone, Debug)]
pub struct Packing {
    pub old: PackingSchema,
    pub new: PackingSchema,
}

impl Packing {
    /// The uniform pair a fold ratio implies: one `bits`-wide index per old word, `fold` of them per
    /// new word. This is what shape detection alone can say, and what every caller used before a
    /// schema could be given.
    pub fn uniform(fold: usize, bits: usize) -> Result<Self, String> {
        let bits = u32::try_from(bits).map_err(|_| format!("bit width {bits} is not a width"))?;
        Ok(Self {
            old: PackingSchema::new(vec![bits], None)?,
            new: PackingSchema::new(vec![bits; fold.max(1)], None)?,
        })
    }

    /// Where expert `ei` is on one side: the word index, the shift within it, and the field's mask.
    fn place(schema: &PackingSchema, ei: usize) -> (usize, u16, u16) {
        let lp = schema.len_p().max(1);
        let k = ei % lp;
        // `sum(widths) <= 16` is a schema invariant, so both the shift and the mask fit a `u16`.
        let shift = u16::try_from(schema.offset(k)).unwrap_or(0);
        let mask = u16::try_from((1u32 << schema.width(k)) - 1).unwrap_or(u16::MAX);
        (ei / lp, shift, mask)
    }

    /// Words with a set bit above the schema's total width — the packing assumption is wrong if any.
    ///
    /// A schema that fills the word has no unused bits (and shifting a `u16` by 16 would panic), so
    /// that case answers zero without looking.
    fn unused_bits_set(schema: &PackingSchema, words: &[u16]) -> u64 {
        let total = schema.offset(schema.len_p());
        if total >= 16 {
            return 0;
        }
        words.iter().filter(|&&x| (x >> total) != 0).count() as u64
    }
}

/// The index-comparison outcome (everything a [`RepackResult`] needs except the
/// codebook/scale diffs, `fold`, `bytes`, and `error`).
#[derive(Default)]
pub struct IdxCompare {
    pub elements: u64,
    pub differing: u64,
    pub sparse_bad: u64,
    pub dense_bad: u64,
    pub max_delta: u32,
    pub differing_gt1: u64,
    pub sum_abs: u64,
    pub sum_old: u64,
    pub sum_new: u64,
    /// Decoded indices equal to 0, counted across both sides (`2 * elements` total) —
    /// the "amount of zeroes" that marks a tensor as sparse-packed.
    pub zeros: u64,
    pub first_mismatch: Option<(u64, u64, u32, u32)>,
    pub sample: Option<RepackSample>,
}

/// Decode + compare the two sides' 16-bit words for the uniform packing a fold ratio implies — one
/// `bits`-wide index per old word, `fold` of them per new word. [`compare_packed`] is the general form.
#[must_use]
pub fn compare_indices(
    old: &[u16],
    new: &[u16],
    e: usize,
    inner: usize,
    fold: usize,
    bits: usize,
) -> IdxCompare {
    // An unusable fold/bits pair decodes nothing; the format counters then say the words did not fit
    // the assumption, which is what a caller checks before believing an "equivalent" verdict.
    let Ok(packing) = Packing::uniform(fold, bits) else {
        return IdxCompare {
            sparse_bad: old.len() as u64,
            dense_bad: new.len() as u64,
            ..IdxCompare::default()
        };
    };
    compare_packed(old, new, e, inner, &packing)
}

/// Decode + compare the two sides' 16-bit words, each side by **its own** schema.
///
/// `e` is the expert count; `inner` the elements per expert. The old side is `ceil(e / oldP) × inner`
/// words and the new side `ceil(e / newP) × inner`. Pure — unit-tested with synthetic slices.
#[must_use]
pub fn compare_packed(
    old: &[u16],
    new: &[u16],
    e: usize,
    inner: usize,
    packing: &Packing,
) -> IdxCompare {
    let sparse_bad = Packing::unused_bits_set(&packing.old, old);
    let dense_bad = Packing::unused_bits_set(&packing.new, new);

    let (mut differing, mut sum_abs, mut sum_old, mut sum_new) = (0u64, 0u64, 0u64, 0u64);
    let (mut max_delta, mut differing_gt1) = (0u32, 0u64);
    let mut zeros = 0u64;
    let mut first: Option<(u64, u64, u32, u32)> = None;
    // Both sides are walked by expert, each looking up its own row: with the sparse schema `[4]` the
    // old lookup is the identity (one expert per row) and the new one folds, which is the case this
    // started as. `verify_local` has already checked both lengths against the shapes; the `else break`
    // is what makes that structural rather than a comment (a short slice ends the compare instead of
    // panicking).
    //
    // The dense rows are collected ONCE: `chunks_exact(..).nth(word)` inside the loop would
    // rescan from the start for every expert, turning a linear compare into a quadratic one
    // on a path that runs over millions of elements.
    //
    // `inner == 0` means a tensor with a zero-length dimension — nothing to compare, and
    // `chunks_exact(0)` panics, so it is answered here rather than reached.
    let nrows: Vec<&[u16]> = if inner == 0 {
        Vec::new()
    } else {
        new.chunks_exact(inner).collect()
    };
    let orows: Vec<&[u16]> = if inner == 0 {
        Vec::new()
    } else {
        old.chunks_exact(inner).collect()
    };
    for ei in 0..e {
        let (oword, oshift, omask) = Packing::place(&packing.old, ei);
        let (nword, nshift, nmask) = Packing::place(&packing.new, ei);
        let (Some(orow), Some(nrow)) = (orows.get(oword), nrows.get(nword)) else {
            break;
        };
        for (n, (o, nd)) in orow.iter().zip(nrow.iter()).enumerate() {
            let o = i64::from((*o >> oshift) & omask);
            let nd = i64::from((*nd >> nshift) & nmask);
            sum_old += o as u64;
            sum_new += nd as u64;
            zeros += u64::from(o == 0) + u64::from(nd == 0);
            if o != nd {
                let d = (o - nd).unsigned_abs();
                differing += 1;
                sum_abs += d;
                if d as u32 > max_delta {
                    max_delta = d as u32;
                }
                if d > 1 {
                    differing_gt1 += 1;
                }
                if first.is_none() {
                    first = Some((ei as u64, n as u64, o as u32, nd as u32));
                }
            }
        }
    }
    let sample = build_sample(old, new, e, inner, packing, first);
    IdxCompare {
        elements: (e * inner) as u64,
        differing,
        sparse_bad,
        dense_bad,
        max_delta,
        differing_gt1,
        sum_abs,
        sum_old,
        sum_new,
        zeros,
        first_mismatch: first,
        sample,
    }
}

/// A small decoded window (16 experts × 48 inner offsets), centred on the first
/// mismatch (or the top-left corner), so the caller can show where old and new
/// diverge — matching the remote script's sample.
fn build_sample(
    old: &[u16],
    new: &[u16],
    e: usize,
    inner: usize,
    packing: &Packing,
    first: Option<(u64, u64, u32, u32)>,
) -> Option<RepackSample> {
    if e == 0 || inner == 0 {
        return None;
    }
    let (fe, foff) = first.map_or((0, 0), |(e, o, _, _)| (e as usize, o as usize));
    let e0 = fe.saturating_sub(6);
    let off0 = foff.saturating_sub(16);
    let e1 = (e0 + 16).min(e);
    let off1 = (off0 + 48).min(inner);
    let mut oldg = Vec::with_capacity(e1 - e0);
    let mut newg = Vec::with_capacity(e1 - e0);
    // Rows by slice rather than by index, so a window that runs past either side ends the
    // sample instead of panicking mid-frame. The window is clamped to `e`/`inner` above, so
    // in practice it stops only on a slice shorter than its declared shape.
    // `inner > 0` is guaranteed by the early return above, so `chunks_exact` is safe here.
    let orows: Vec<&[u16]> = old.chunks_exact(inner).collect();
    let nrows: Vec<&[u16]> = new.chunks_exact(inner).collect();
    for ei in e0..e1 {
        let (oword, oshift, omask) = Packing::place(&packing.old, ei);
        let (nword, nshift, nmask) = Packing::place(&packing.new, ei);
        let (Some(orow), Some(nrow)) = (orows.get(oword), nrows.get(nword)) else {
            break;
        };
        let decode = |row: &[u16], shift: u16, mask: u16| -> Vec<u32> {
            row.get(off0..off1)
                .unwrap_or_default()
                .iter()
                .map(|w| u32::from((w >> shift) & mask))
                .collect()
        };
        oldg.push(decode(orow, oshift, omask));
        newg.push(decode(nrow, nshift, nmask));
    }
    Some(RepackSample {
        e0: e0 as u64,
        off0: off0 as u64,
        old: oldg,
        new: newg,
    })
}

/// Verify one local expert-weight pair (`old_w` sparse, `new_w` dense) and diff its
/// sibling codebook / scale tensors. Reads the raw words locally via
/// [`crate::sample`]. `fold`/`bits` come from the shape detection; `schemas` overrides what each side's
/// packing *is* when the checkpoint does not declare it (which is the normal case).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn verify_local(
    old_w: &TensorInfo,
    new_w: &TensorInfo,
    fold: usize,
    bits: usize,
    schemas: (Option<&PackingSchema>, Option<&PackingSchema>),
    codebook: (&str, &str, Option<&TensorInfo>, Option<&TensorInfo>),
    qscale: (&str, &str, Option<&TensorInfo>, Option<&TensorInfo>),
) -> RepackResult {
    let err = |e: String| RepackResult {
        error: Some(e),
        fold,
        ..Default::default()
    };
    let (ow, nw) = match (
        crate::sample::read_all_u16(old_w),
        crate::sample::read_all_u16(new_w),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return err(e),
    };
    // Each side's packing: as given, else the uniform pair the fold implies.
    let mut packing = match Packing::uniform(fold, bits) {
        Ok(p) => p,
        Err(e) => return err(e),
    };
    if let Some(o) = schemas.0 {
        packing.old = o.clone();
    }
    if let Some(n) = schemas.1 {
        packing.new = n.clone();
    }
    let rows = *old_w.shape.first().unwrap_or(&0);
    let inner: usize = old_w.shape.iter().skip(1).product();
    if rows == 0 || inner == 0 || ow.len() != rows * inner {
        return err(format!("unexpected old shape {:?}", old_w.shape));
    }
    let w = *new_w.shape.first().unwrap_or(&0);
    if w == 0 || nw.len() != w * inner {
        return err(format!("unexpected new shape {:?}", new_w.shape));
    }
    // Experts, not rows: the old side's rows each hold `lenP` of them, which is one apiece for the
    // sparse encoding and more for a merged baseline.
    let e = rows * packing.old.len_p();
    if w * packing.new.len_p() < e {
        return err(format!(
            "the schemas do not span the pair: {} over {rows} rows is {e} experts, {} over {w} is {}",
            packing.old.label(),
            packing.new.label(),
            w * packing.new.len_p()
        ));
    }
    let c = compare_packed(&ow, &nw, e, inner, &packing);
    let bytes = (ow.len() * 2 + nw.len() * 2) as u64;
    let mean = |s: u64| {
        if c.elements > 0 {
            s as f64 / c.elements as f64
        } else {
            0.0
        }
    };
    let zero_frac = if c.elements > 0 {
        c.zeros as f64 / (2.0 * c.elements as f64)
    } else {
        0.0
    };
    // The top-bits format check failed ⇒ the words don't look like packed indices;
    // compare them as plain stored-dtype values instead (the auto `--values`
    // fallback), so a mis-detected tensor is still meaningfully diffed.
    let fallback = (c.sparse_bad > 0 || c.dense_bad > 0).then(|| fallback_local(old_w, new_w));
    RepackResult {
        elements: c.elements,
        differing: c.differing,
        max_delta: c.max_delta,
        differing_gt1: c.differing_gt1,
        sum_abs: c.sum_abs,
        mean_abs: mean(c.sum_abs),
        mean_old: mean(c.sum_old),
        mean_new: mean(c.sum_new),
        sparse_bad: c.sparse_bad,
        dense_bad: c.dense_bad,
        fold,
        bits,
        zero_frac,
        fallback,
        first_mismatch: c.first_mismatch,
        sample: c.sample,
        codebook: Some(aux_local(codebook.0, codebook.1, codebook.2, codebook.3)),
        qscale: Some(aux_local(qscale.0, qscale.1, qscale.2, qscale.3)),
        bytes,
        error: None,
    }
}

/// Element-wise `|Δ|` summary of two equal-length value slices: how many differ, and the
/// largest and mean absolute difference.
///
/// Exact equality on purpose, in both callers: repack verification is a proof that two
/// packings hold the *same* weights, so an approximate compare would defeat the point.
// (`clippy::float_cmp` is allowed here for exactly that reason.)
#[allow(clippy::float_cmp)]
fn diff_summary(old: &[f64], new: &[f64]) -> (u64, f64, f64) {
    let (mut differing, mut sum_abs, mut max_abs) = (0u64, 0f64, 0f64);
    for (a, b) in old.iter().zip(new) {
        let d = (a - b).abs();
        if a != b {
            differing += 1;
        }
        sum_abs += d;
        if d > max_abs {
            max_abs = d;
        }
    }
    let mean_abs = if old.is_empty() {
        0.0
    } else {
        sum_abs / old.len() as f64
    };
    (differing, max_abs, mean_abs)
}

/// Compare two weight tensors as plain stored-dtype values (decoded to f64) — the
/// fallback when the sparse format check fails. Returns `elements`/`differing` and
/// max/mean `|Δ|`, or a zeroed result (dtype only) if either side can't be read.
fn fallback_local(old_w: &TensorInfo, new_w: &TensorInfo) -> RepackFallback {
    let mut fb = RepackFallback {
        dtype: new_w.dtype.clone(),
        ..Default::default()
    };
    let (Ok(oa), Ok(na)) = (
        crate::sample::read_all_f64(old_w),
        crate::sample::read_all_f64(new_w),
    ) else {
        return fb;
    };
    let (differing, max_abs, mean_abs) = diff_summary(&oa, &na);
    fb.elements = oa.len() as u64;
    fb.differing = differing;
    fb.max_abs = max_abs;
    fb.mean_abs = mean_abs;
    fb
}

/// Value-diff a sibling float tensor (codebook / scale) locally, recording the names
/// tried + whether each side was found (so a wrong-name inference is visible).
fn aux_local(
    old_name: &str,
    new_name: &str,
    old: Option<&TensorInfo>,
    new: Option<&TensorInfo>,
) -> RepackAux {
    let mut ax = RepackAux {
        old_name: old_name.to_string(),
        new_name: new_name.to_string(),
        old_present: old.is_some(),
        new_present: new.is_some(),
        shape: Vec::new(),
        shape_mismatch: None,
        elements: 0,
        differing: 0,
        max_abs: 0.0,
        mean_abs: 0.0,
    };
    let (Some(o), Some(n)) = (old, new) else {
        return ax;
    };
    // Couldn't read one side — leave it marked present but uncompared.
    let (Ok(oa), Ok(na)) = (
        crate::sample::read_all_f64(o),
        crate::sample::read_all_f64(n),
    ) else {
        return ax;
    };
    ax.shape.clone_from(&n.shape);
    if o.shape != n.shape {
        ax.shape_mismatch = Some((o.shape.clone(), n.shape.clone()));
        return ax;
    }
    let (differing, max_abs, mean_abs) = diff_summary(&oa, &na);
    ax.elements = oa.len() as u64;
    ax.differing = differing;
    ax.max_abs = max_abs;
    ax.mean_abs = mean_abs;
    ax
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No schema given — decode with what the fold ratio implies, which is what every path did before
    /// a schema could be said and what most of these cases exercise.
    const INFERRED: (Option<&PackingSchema>, Option<&PackingSchema>) = (None, None);

    /// Build a `TensorInfo` for a tensor written into a real one-tensor-per-file
    /// safetensors fixture, so the read path (`sample::read_all_*`) is the real one.
    ///
    /// The rest of this module tests `compare_indices` on synthetic slices, which is
    /// where the packing arithmetic lives; these fixtures are what let the *file*-facing
    /// half — `verify_local` and the codebook/qscale diff — be tested at all, since it
    /// only ever sees tensors that came off disk.
    fn on_disk(
        dir: &std::path::Path,
        name: &str,
        dtype: &str,
        shape: &[usize],
        data: &[u8],
    ) -> TensorInfo {
        let path = dir.join(format!("{name}.safetensors"));
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":{:?},"data_offsets":[0,{}]}}}}"#,
            shape,
            data.len()
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(&path, bytes).unwrap();
        let ckpt = crate::readers::read_local(&[path]).unwrap();
        ckpt.shards[0]
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap()
            .clone()
    }

    fn u16_tensor(dir: &std::path::Path, name: &str, shape: &[usize], words: &[u16]) -> TensorInfo {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        on_disk(dir, name, "U16", shape, &bytes)
    }

    fn f32_tensor(dir: &std::path::Path, name: &str, shape: &[usize], vals: &[f32]) -> TensorInfo {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        on_disk(dir, name, "F32", shape, &bytes)
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("cs_repack_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    // Pack `e` experts' 3-bit indices (shape e×inner) into the dense `fold`-per-word
    // layout, returning (sparse_words, dense_words, w).
    fn pack(idx: &[Vec<u16>], fold: usize, bits: usize) -> (Vec<u16>, Vec<u16>, usize) {
        let e = idx.len();
        let inner = idx[0].len();
        let w = e.div_ceil(fold);
        let sparse: Vec<u16> = idx.iter().flatten().copied().collect();
        let mut dense = vec![0u16; w * inner];
        for (ei, row) in idx.iter().enumerate() {
            let word = ei / fold;
            let shift = ((ei % fold) * bits) as u16;
            for (n, &v) in row.iter().enumerate() {
                dense[word * inner + n] |= v << shift;
            }
        }
        (sparse, dense, w)
    }

    /// Pack `idx` (expert-major) per an explicit schema: expert `ei` goes into word `ei / lenP` at the
    /// field's own offset. The general form of `pack` above, so a non-uniform layout can be built.
    fn pack_as(idx: &[Vec<u16>], schema: &PackingSchema) -> (Vec<u16>, usize) {
        let (e, inner) = (idx.len(), idx[0].len());
        let lp = schema.len_p();
        let w = e.div_ceil(lp);
        let mut out = vec![0u16; w * inner];
        for (ei, row) in idx.iter().enumerate() {
            let shift = u16::try_from(schema.offset(ei % lp)).unwrap();
            for (n, &v) in row.iter().enumerate() {
                out[(ei / lp) * inner + n] |= v << shift;
            }
        }
        (out, w)
    }

    /// The pairing the user's checkpoints actually are: a **sparse** baseline (`[4]` — one index per
    /// word, four low bits used) against a **merged** candidate (`[3,3,3,3,3]` — five experts per word).
    /// The two sides' widths differ, which no single `bits` can describe, so this is the case the
    /// schemas exist for.
    #[test]
    fn a_sparse_baseline_and_a_merged_candidate_compare_equal() {
        let idx = vec![
            vec![1u16, 5, 3, 7, 0, 2],
            vec![4, 4, 1, 6, 2, 5],
            vec![7, 0, 3, 3, 1, 4],
            vec![2, 6, 5, 1, 0, 7],
            vec![3, 3, 2, 4, 5, 1],
            vec![6, 1, 0, 2, 7, 3], // 6 experts, 5 per merged word -> 2 words (1 pad slot)
        ];
        let packing = Packing {
            old: PackingSchema::parse("[4]").unwrap(),
            new: PackingSchema::parse("[3,3,3,3,3]").unwrap(),
        };
        let (sparse, ow) = pack_as(&idx, &packing.old);
        let (merged, nw) = pack_as(&idx, &packing.new);
        assert_eq!((ow, nw), (6, 2));
        let c = compare_packed(&sparse, &merged, 6, 6, &packing);
        assert_eq!((c.differing, c.elements), (0, 36));
        assert_eq!((c.sparse_bad, c.dense_bad), (0, 0));
        assert_eq!(c.sum_old, c.sum_new);

        // And the point of *saying* the schema: the uniform assumption the fold implies (`16/2 = 8`
        // bits, two per word) decodes the merged side from the wrong bits and finds differences —
        // silently, if nobody could override it.
        let wrong = compare_indices(&sparse, &merged, 6, 6, 2, 8);
        assert!(
            wrong.differing > 0,
            "a wrong packing must not read as equal"
        );
    }

    /// A schema whose fields differ in width — the offsets are a running sum, not `k * bits`, so a
    /// uniform decoder reads every field after the first from the wrong bits.
    #[test]
    fn a_non_uniform_schema_decodes_each_field_at_its_own_width() {
        // Widths [4,4,4,3]: the first three experts hold values up to 15, the last up to 7.
        let schema = PackingSchema::parse("4,4,4,3").unwrap();
        let idx = vec![vec![15u16, 9], vec![12, 0], vec![10, 15], vec![7, 5]];
        let (merged, w) = pack_as(&idx, &schema);
        assert_eq!(w, 1);
        let packing = Packing {
            old: PackingSchema::parse("[4]").unwrap(),
            new: schema,
        };
        let (sparse, _) = pack_as(&idx, &packing.old);
        let c = compare_packed(&sparse, &merged, 4, 2, &packing);
        assert_eq!((c.differing, c.elements), (0, 8));
        // 4+4+4+3 = 15 of the 16 bits: the top bit is unused and clear on both sides.
        assert_eq!((c.sparse_bad, c.dense_bad), (0, 0));
    }

    #[test]
    fn equivalent_packing_compares_equal() {
        let idx = vec![
            vec![1u16, 5, 3, 7, 0, 2],
            vec![4, 4, 1, 6, 2, 5],
            vec![7, 0, 3, 3, 1, 4],
            vec![2, 6, 5, 1, 0, 7],
            vec![3, 3, 2, 4, 5, 1],
            vec![6, 1, 0, 2, 7, 3], // 6 experts, fold 5 -> w=2 (1 pad slot)
        ];
        let (sparse, dense, _w) = pack(&idx, 5, 3);
        let c = compare_indices(&sparse, &dense, 6, 6, 5, 3);
        assert_eq!(c.differing, 0);
        assert_eq!(c.sparse_bad, 0);
        assert_eq!(c.dense_bad, 0);
        assert_eq!(c.elements, 36);
        assert_eq!(c.sum_old, c.sum_new);
    }

    #[test]
    fn four_bit_fold4_has_no_unused_msb() {
        // 4-bit ×4 uses all 16 bits: fold 4, bits 4, dense_shift == 16 ⇒ no MSB slack,
        // and the shift-by-16 guard must not panic or false-flag.
        let idx = vec![
            vec![1u16, 9, 15, 0, 8],
            vec![4, 12, 3, 6, 11],
            vec![7, 0, 13, 3, 1],
            vec![2, 6, 5, 14, 10], // 4 experts, fold 4 -> w=1, all slots used
        ];
        let (sparse, dense, w) = pack(&idx, 4, 4);
        assert_eq!(w, 1);
        let c = compare_indices(&sparse, &dense, 4, 5, 4, 4);
        assert_eq!(c.differing, 0);
        assert_eq!(c.sparse_bad, 0);
        assert_eq!(c.dense_bad, 0); // fold*bits == 16 ⇒ check skipped, not a spurious hit
        assert_eq!(c.sum_old, c.sum_new);
    }

    #[test]
    fn fold1_compares_two_sparse_sides_directly() {
        // Sparse↔sparse (the auto `--values` case): fold 1, one index per word on
        // both sides, same shape. compare_indices decodes each word directly.
        let old = vec![0u16, 3, 0, 7, 1, 0]; // 3 experts × 2 inner
        let mut new = old.clone();
        new[3] = 6; // one index differs by 1 (7 -> 6)
        let c = compare_indices(&old, &new, 3, 2, 1, 3);
        assert_eq!(c.differing, 1);
        assert_eq!(c.max_delta, 1);
        assert_eq!(c.differing_gt1, 0);
        assert_eq!(c.sparse_bad, 0);
        assert_eq!(c.dense_bad, 0); // fold*bits = 3 < 16 ⇒ new side also checked, clean
        // Zeros counted across both sides: old has 3 (idx 0), new has 3 ⇒ 6 of 12.
        assert_eq!(c.zeros, 6);
        // A high bit above `bits` on either side trips the format check (→ fallback).
        let mut bad = old.clone();
        bad[0] |= 1 << 5;
        assert!(compare_indices(&bad, &new, 3, 2, 1, 3).sparse_bad >= 1);
        assert!(compare_indices(&old, &bad, 3, 2, 1, 3).dense_bad >= 1);
    }

    /// A decoded window is returned even with no mismatch (anchored top-left), and it's
    /// centred on the first mismatch when there is one — that window is what the report
    /// shows, so its origin has to line up with the reported `(expert, offset)`.
    #[test]
    fn the_sample_window_is_centred_on_the_first_mismatch() {
        // 30 experts × 80 inner, so the 16×48 window is a real window, not the whole
        // tensor, and both the clamp and the centring have something to do.
        let idx: Vec<Vec<u16>> = (0..30)
            .map(|e| (0..80).map(|n| ((e + n) % 8) as u16).collect())
            .collect();
        let (sparse, mut dense, _w) = pack(&idx, 5, 3);

        let clean = compare_indices(&sparse, &dense, 30, 80, 5, 3)
            .sample
            .unwrap();
        assert_eq!((clean.e0, clean.off0), (0, 0), "no mismatch ⇒ top-left");
        assert_eq!(clean.old.len(), 16);
        assert_eq!(clean.old[0].len(), 48);
        assert_eq!(clean.old, clean.new, "equal packings decode identically");

        // Corrupt expert 20's slot at offset 40 (word 4, shift (20%5)*3 = 0).
        dense[4 * 80 + 40] ^= 0b111;
        let c = compare_indices(&sparse, &dense, 30, 80, 5, 3);
        assert_eq!(c.first_mismatch.map(|(e, o, _, _)| (e, o)), Some((20, 40)));
        let s = c.sample.unwrap();
        // Centred: 6 experts and 16 offsets of context before the mismatch.
        assert_eq!((s.e0, s.off0), (14, 24));
        let (row, col) = ((20 - s.e0) as usize, (40 - s.off0) as usize);
        assert_ne!(s.old[row][col], s.new[row][col], "the mismatch is in frame");
    }

    /// The window clamps to the tensor rather than reading out of bounds — a tensor
    /// smaller than 16×48 must still produce a sample.
    #[test]
    fn a_tensor_smaller_than_the_window_is_returned_whole() {
        let idx = vec![vec![1u16, 2, 3], vec![4, 5, 6]];
        let (sparse, dense, _w) = pack(&idx, 5, 3);
        let s = compare_indices(&sparse, &dense, 2, 3, 5, 3).sample.unwrap();
        assert_eq!(s.old, vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(s.new, s.old);
        // A zero-sized side has no window at all (rather than an empty-grid panic).
        assert!(compare_indices(&[], &[], 0, 3, 5, 3).sample.is_none());
        assert!(compare_indices(&[], &[], 2, 0, 5, 3).sample.is_none());
    }

    #[test]
    fn detects_value_and_format_problems() {
        let idx = vec![vec![1u16, 2, 3, 4], vec![5, 6, 7, 0], vec![2, 2, 2, 2]];
        let (mut sparse, mut dense, _w) = pack(&idx, 5, 3); // 3 experts, fold 5, w=1
        // Corrupt one decoded index in the dense side (expert 0, offset 1: was 2).
        dense[1] ^= 0b111; // flips expert-0 slot-0 value at offset 1
        let c = compare_indices(&sparse, &dense, 3, 4, 5, 3);
        assert!(c.differing >= 1);
        assert_eq!(c.first_mismatch.map(|(e, o, _, _)| (e, o)), Some((0, 1)));
        // A stray high bit in the sparse side trips the format check.
        sparse[0] |= 1 << 5;
        let c2 = compare_indices(&sparse, &dense, 3, 4, 5, 3);
        assert!(c2.sparse_bad >= 1);
        // A bit above fold*bits (15) in the dense side trips the dense check.
        dense[0] |= 1 << 15;
        let c3 = compare_indices(&sparse, &dense, 3, 4, 5, 3);
        assert!(c3.dense_bad >= 1);
    }

    // --- the file-facing half ------------------------------------------------------
    //
    // `verify_local` is the whole point of this module: it's what `diff --verify-repack`
    // runs for local checkpoints, and it produces the `RepackResult` the report prints.
    // These read from real safetensors files, so the shape checks, the codebook/qscale
    // diff and the value fallback are exercised the way the CLI exercises them.

    #[test]
    fn a_matching_local_repack_verifies_clean() {
        let dir = scratch("clean");
        let idx: Vec<Vec<u16>> = (0..6)
            .map(|e| (0..4).map(|n| ((e * 3 + n) % 8) as u16).collect())
            .collect();
        let (sparse, dense, w) = pack(&idx, 5, 3);
        let old = u16_tensor(&dir, "old_w", &[6, 4], &sparse);
        let new = u16_tensor(&dir, "new_w", &[w, 4], &dense);
        let cb_old = f32_tensor(&dir, "cb_old", &[2, 2], &[0.5, -1.5, 2.0, 0.0]);
        let cb_new = f32_tensor(&dir, "cb_new", &[2, 2], &[0.5, -1.5, 2.0, 0.0]);
        let qs_old = f32_tensor(&dir, "qs_old", &[2], &[1.0, 4.0]);
        let qs_new = f32_tensor(&dir, "qs_new", &[2], &[1.0, 4.0]);

        let r = verify_local(
            &old,
            &new,
            5,
            3,
            INFERRED,
            ("cb_old", "cb_new", Some(&cb_old), Some(&cb_new)),
            ("qs_old", "qs_new", Some(&qs_old), Some(&qs_new)),
        );
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!((r.elements, r.differing), (24, 0));
        assert_eq!((r.sparse_bad, r.dense_bad), (0, 0));
        assert_eq!((r.fold, r.bits), (5, 3));
        // Both sides read: 24 sparse words + 8 dense words, 2 bytes each.
        assert_eq!(r.bytes, (24 + w * 4) as u64 * 2);
        assert!(
            r.fallback.is_none(),
            "a clean format needs no value fallback"
        );
        let cb = r.codebook.unwrap();
        assert!(cb.old_present && cb.new_present);
        assert_eq!((cb.elements, cb.differing), (4, 0));
        assert_eq!(cb.shape, vec![2, 2]);
        assert_eq!(r.qscale.unwrap().differing, 0);
    }

    /// A tensor whose words carry bits above `bits` isn't index-packed at all, so the
    /// verdict falls back to comparing the stored values — the `--values` auto path.
    #[test]
    fn a_bad_format_falls_back_to_a_value_compare() {
        let dir = scratch("fallback");
        // Same words on both sides, but with high bits set: format check fails, values
        // still agree — so the fallback must report zero differing, not a failure.
        let words: Vec<u16> = (0..12).map(|i| 0xF000 | (i % 8)).collect();
        let old = u16_tensor(&dir, "old_w", &[3, 4], &words);
        let new = u16_tensor(&dir, "new_w", &[3, 4], &words);
        let r = verify_local(
            &old,
            &new,
            1,
            3,
            INFERRED,
            ("cb_old", "cb_new", None, None),
            ("qs_old", "qs_new", None, None),
        );
        assert!(r.sparse_bad > 0 && r.dense_bad > 0, "{r:?}");
        let fb = r
            .fallback
            .expect("a failed format check triggers the fallback");
        assert_eq!(fb.dtype, "U16");
        assert_eq!((fb.elements, fb.differing), (12, 0));
        assert_eq!(fb.max_abs, 0.0);
        // A sibling that was never found is reported absent rather than silently equal.
        let cb = r.codebook.unwrap();
        assert!(!cb.old_present && !cb.new_present);
        assert_eq!(cb.elements, 0);
    }

    /// The value fallback has to *find* differences too, and report the largest.
    #[test]
    fn the_value_fallback_reports_the_largest_difference() {
        let dir = scratch("fallback_diff");
        let a: Vec<u16> = (0..8).map(|i| 0xF000 | i).collect();
        let mut b = a.clone();
        b[5] = 0xF000; // decoded value drops by 5
        let old = u16_tensor(&dir, "old_w", &[2, 4], &a);
        let new = u16_tensor(&dir, "new_w", &[2, 4], &b);
        let r = verify_local(
            &old,
            &new,
            1,
            3,
            INFERRED,
            ("cb_old", "cb_new", None, None),
            ("qs_old", "qs_new", None, None),
        );
        let fb = r.fallback.unwrap();
        assert_eq!((fb.elements, fb.differing), (8, 1));
        assert_eq!(fb.max_abs, 5.0);
        assert_eq!(fb.mean_abs, 5.0 / 8.0);
    }

    /// Shapes that don't square with the words on disk are errors naming the side — not a
    /// panic on an out-of-range slice, which is what the index decode would otherwise do
    /// (it slices `old[ei * inner..]` per expert with no bounds check of its own).
    #[test]
    fn a_shape_that_does_not_match_the_data_is_an_error() {
        let dir = scratch("shape");
        let old = u16_tensor(&dir, "old_w", &[3, 4], &[0u16; 12]);
        let short = u16_tensor(&dir, "new_w", &[1, 4], &[0u16; 4]);
        let aux = ("a", "b", None, None);
        // A leading dimension of zero: legal on disk, but there is nothing to decode.
        let empty = u16_tensor(&dir, "empty_w", &[0, 4], &[]);
        let r = verify_local(&empty, &short, 5, 3, INFERRED, aux, aux);
        assert!(
            r.error.unwrap().contains("unexpected old shape"),
            "expected an old-shape error"
        );
        // A `new` side whose inner extent disagrees with `old`'s: w=1 × inner 4 is wanted,
        // 3 words are stored.
        let narrow = u16_tensor(&dir, "narrow_w", &[1, 3], &[0u16; 3]);
        let r = verify_local(&old, &narrow, 5, 3, INFERRED, aux, aux);
        assert!(
            r.error.unwrap().contains("unexpected new shape"),
            "expected a new-shape error"
        );
        // A header that declares more than it stores is rejected by the reader, so the
        // verdict carries that error instead of crashing partway through the decode.
        let lying = TensorInfo {
            shape: vec![9, 4],
            ..short.clone()
        };
        let e = verify_local(&old, &lying, 5, 3, INFERRED, aux, aux)
            .error
            .unwrap();
        assert!(e.contains("only 8 bytes are stored"), "{e}");
        // An unreadable side reports the read error rather than proceeding.
        let missing = TensorInfo {
            source_path: dir.join("gone.safetensors").to_string_lossy().into_owned(),
            ..old
        };
        assert!(
            verify_local(&missing, &short, 5, 3, INFERRED, aux, aux)
                .error
                .is_some()
        );
    }

    /// Sibling tensors of different shapes are flagged as a shape mismatch instead of
    /// being compared element-wise (which would compare the wrong pairs and call it a
    /// clean match, because `zip` stops at the shorter side).
    #[test]
    fn mismatched_sibling_shapes_are_reported_not_compared() {
        let dir = scratch("aux_shape");
        let idx = vec![vec![1u16, 2], vec![3, 4]];
        let (sparse, dense, w) = pack(&idx, 5, 3);
        let old = u16_tensor(&dir, "old_w", &[2, 2], &sparse);
        let new = u16_tensor(&dir, "new_w", &[w, 2], &dense);
        let cb_old = f32_tensor(&dir, "cb_old", &[4], &[1.0, 2.0, 3.0, 4.0]);
        let cb_new = f32_tensor(&dir, "cb_new", &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        // Same values, different shape.
        let qs_old = f32_tensor(&dir, "qs_old", &[2], &[1.0, 2.0]);
        let qs_new = f32_tensor(&dir, "qs_new", &[2], &[1.0, 2.5]);
        let r = verify_local(
            &old,
            &new,
            5,
            3,
            INFERRED,
            ("cb_old", "cb_new", Some(&cb_old), Some(&cb_new)),
            ("qs_old", "qs_new", Some(&qs_old), Some(&qs_new)),
        );
        let cb = r.codebook.unwrap();
        assert_eq!(cb.shape_mismatch, Some((vec![4], vec![2, 2])));
        assert_eq!(
            cb.differing, 0,
            "a shape mismatch short-circuits the compare"
        );
        // The qscale sibling has the same shape and one differing value.
        let qs = r.qscale.unwrap();
        assert!(qs.shape_mismatch.is_none());
        assert_eq!((qs.elements, qs.differing), (2, 1));
        assert_eq!(qs.max_abs, 0.5);
    }
}
