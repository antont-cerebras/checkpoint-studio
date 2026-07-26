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

use crate::remote::{RepackAux, RepackFallback, RepackResult, RepackSample};
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
    /// Decoded indices equal to 0, counted across both sides (`2 * elements` total) —
    /// the "amount of zeroes" that marks a tensor as sparse-packed.
    pub zeros: u64,
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
    let mut zeros = 0u64;
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
        zeros,
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
    let (fe, foff) = first.map_or((0, 0), |(e, o, _, _)| (e as usize, o as usize));
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
        let r = verify_local(&empty, &short, 5, 3, aux, aux);
        assert!(
            r.error.unwrap().contains("unexpected old shape"),
            "expected an old-shape error"
        );
        // A `new` side whose inner extent disagrees with `old`'s: w=1 × inner 4 is wanted,
        // 3 words are stored.
        let narrow = u16_tensor(&dir, "narrow_w", &[1, 3], &[0u16; 3]);
        let r = verify_local(&old, &narrow, 5, 3, aux, aux);
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
        let e = verify_local(&old, &lying, 5, 3, aux, aux).error.unwrap();
        assert!(e.contains("only 8 bytes are stored"), "{e}");
        // An unreadable side reports the read error rather than proceeding.
        let missing = TensorInfo {
            source_path: dir.join("gone.safetensors").to_string_lossy().into_owned(),
            ..old
        };
        assert!(
            verify_local(&missing, &short, 5, 3, aux, aux)
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
