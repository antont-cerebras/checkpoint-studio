//! Local (in-process) repack-equivalence verification — the same check
//! [`crate::remote::RemoteRead::verify_repack`] runs on the ssh proxy for `s3://`
//! cstorch checkpoints, but for local checkpoint files (`.safetensors` / `.hdf5`).
//!
//! Two expert-weight tensors encode the same 3-bit indices in different packings:
//! the **sparse** side stores one index per 16-bit word (`w & mask`); the **dense**
//! side folds `fold` experts along dim 0 into one word (`fold` indices per word, the
//! expert `e` at word `e / fold`, bit shift `(e % fold) * bits`). This module reads
//! the raw 16-bit words locally, decodes both sides, checks they match, and
//! validates the packing (the old words' bits above `bits` and the new words' bits
//! above `fold * bits` must be zero). It also diffs the sibling codebook / scale
//! tensors' values. Results are the same [`RepackResult`] the remote path produces,
//! so the reporting is shared.

use crate::remote::{RepackAux, RepackResult, RepackSample};
use crate::tree::TensorInfo;

/// The index-comparison outcome (everything a [`RepackResult`] needs except the
/// codebook/scale diffs, `fold`, `bytes`, and `error`).
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
    pub first_mismatch: Option<(u64, u64, u32, u32)>,
    pub sample: Option<RepackSample>,
}

/// Decode + compare the two sides' 16-bit words. `old` is `e × inner` words (one
/// index each); `new` is `w × inner` words (`fold` indices each, `w = ceil(e/fold)`).
/// Pure — unit-tested with synthetic slices.
pub fn compare_indices(
    old: &[u16],
    new: &[u16],
    e: usize,
    inner: usize,
    fold: usize,
    bits: usize,
) -> IdxCompare {
    let mask = (1u16 << bits) - 1;
    // Format checks: no bits above `bits` (old) / `fold*bits` (new). A shift equal to
    // the word width has no unused bits (and would panic), so guard it.
    let sparse_bad = old.iter().filter(|&&x| (x >> bits) != 0).count() as u64;
    let dense_shift = fold * bits;
    let dense_bad = if dense_shift >= 16 {
        0
    } else {
        new.iter().filter(|&&x| (x >> dense_shift) != 0).count() as u64
    };

    let (mut differing, mut sum_abs, mut sum_old, mut sum_new) = (0u64, 0u64, 0u64, 0u64);
    let (mut max_delta, mut differing_gt1) = (0u32, 0u64);
    let mut first: Option<(u64, u64, u32, u32)> = None;
    for ei in 0..e {
        let word = ei / fold;
        let shift = ((ei % fold) * bits) as u16;
        let orow = &old[ei * inner..(ei + 1) * inner];
        let nrow = &new[word * inner..(word + 1) * inner];
        for n in 0..inner {
            let o = (orow[n] & mask) as i64;
            let nd = ((nrow[n] >> shift) & mask) as i64;
            sum_old += o as u64;
            sum_new += nd as u64;
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
    let sample = build_sample(old, new, e, inner, fold, bits, mask, first);
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
        first_mismatch: first,
        sample,
    }
}

/// A small decoded window (16 experts × 48 inner offsets), centred on the first
/// mismatch (or the top-left corner), so the caller can show where old and new
/// diverge — matching the remote script's sample.
#[allow(clippy::too_many_arguments)]
fn build_sample(
    old: &[u16],
    new: &[u16],
    e: usize,
    inner: usize,
    fold: usize,
    bits: usize,
    mask: u16,
    first: Option<(u64, u64, u32, u32)>,
) -> Option<RepackSample> {
    if e == 0 || inner == 0 {
        return None;
    }
    let (fe, foff) = first
        .map(|(e, o, _, _)| (e as usize, o as usize))
        .unwrap_or((0, 0));
    let e0 = fe.saturating_sub(6);
    let off0 = foff.saturating_sub(16);
    let e1 = (e0 + 16).min(e);
    let off1 = (off0 + 48).min(inner);
    let mut oldg = Vec::with_capacity(e1 - e0);
    let mut newg = Vec::with_capacity(e1 - e0);
    for ei in e0..e1 {
        let word = ei / fold;
        let shift = ((ei % fold) * bits) as u16;
        let orow = &old[ei * inner..(ei + 1) * inner];
        let nrow = &new[word * inner..(word + 1) * inner];
        oldg.push((off0..off1).map(|n| (orow[n] & mask) as u32).collect());
        newg.push(
            (off0..off1)
                .map(|n| ((nrow[n] >> shift) & mask) as u32)
                .collect(),
        );
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
/// [`crate::sample`]. `fold`/`bits` come from the shape detection.
#[allow(clippy::too_many_arguments)]
pub fn verify_local(
    old_w: &TensorInfo,
    new_w: &TensorInfo,
    fold: usize,
    bits: usize,
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
    let e = *old_w.shape.first().unwrap_or(&0);
    let inner: usize = old_w.shape.iter().skip(1).product();
    if e == 0 || inner == 0 || ow.len() != e * inner {
        return err(format!("unexpected old shape {:?}", old_w.shape));
    }
    let w = *new_w.shape.first().unwrap_or(&0);
    if w == 0 || nw.len() != w * inner {
        return err(format!("unexpected new shape {:?}", new_w.shape));
    }
    let c = compare_indices(&ow, &nw, e, inner, fold, bits);
    let bytes = (ow.len() * 2 + nw.len() * 2) as u64;
    let mean = |s: u64| {
        if c.elements > 0 {
            s as f64 / c.elements as f64
        } else {
            0.0
        }
    };
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
        first_mismatch: c.first_mismatch,
        sample: c.sample,
        codebook: Some(aux_local(codebook.0, codebook.1, codebook.2, codebook.3)),
        qscale: Some(aux_local(qscale.0, qscale.1, qscale.2, qscale.3)),
        bytes,
        error: None,
    }
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
    let (oa, na) = match (
        crate::sample::read_all_f64(o),
        crate::sample::read_all_f64(n),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        // Couldn't read one side — leave it marked present but uncompared.
        _ => return ax,
    };
    ax.shape = n.shape.clone();
    if o.shape != n.shape {
        ax.shape_mismatch = Some((o.shape.clone(), n.shape.clone()));
        return ax;
    }
    let (mut differing, mut sum_abs, mut max_abs) = (0u64, 0f64, 0f64);
    for (a, b) in oa.iter().zip(&na) {
        let d = (a - b).abs();
        if a != b {
            differing += 1;
        }
        sum_abs += d;
        if d > max_abs {
            max_abs = d;
        }
    }
    ax.elements = oa.len() as u64;
    ax.differing = differing;
    ax.max_abs = max_abs;
    ax.mean_abs = if oa.is_empty() {
        0.0
    } else {
        sum_abs / oa.len() as f64
    };
    ax
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
