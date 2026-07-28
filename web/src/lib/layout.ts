// Derivations over a served `LayoutMap`. The Rust side computes these in
// `crates/core/src/safelayout.rs`; a browser can't call into it, so the rule lives twice
// and both copies are tested (see shared/parity/README.md).

import type { LayoutMap } from './types';

/** Unaccounted bytes between segments: how many gaps, and how many bytes in total. */
export interface GapSummary {
  count: number;
  bytes: number;
}

/**
 * Total the file's padding — the mirror of `LayoutMap::gap_summary`.
 *
 * Gap segments were always drawn but never added up, so "is this file padded, and by how
 * much?" meant scrolling the whole segment list looking for them. It is the one fact about
 * a safetensors layout that the tensor list cannot show.
 */
export function gapSummary(map: Pick<LayoutMap, 'segments'>): GapSummary {
  let count = 0;
  let bytes = 0;
  for (const s of map.segments) {
    if (s.kind.kind === 'gap') {
      count += 1;
      bytes += s.end - s.start;
    }
  }
  return { count, bytes };
}

/**
 * A dtype's family — mirrors `DtypeClass::of` in `crates/core/src/stats.rs`.
 *
 * Grouped by what a reader asks (how wide, float or integer) rather than one entry per
 * dtype name: a dozen-colour palette stops being readable, and "is this shard
 * half-precision or 8-bit quantised" is a question about the family.
 *
 * Returns the same keys `DtypeClass::key()` does, so the two UIs group identically even
 * though each paints from its own palette.
 */
export type DtypeClass =
  | 'float-wide'
  | 'float-half'
  | 'float-narrow'
  | 'int-wide'
  | 'int-narrow'
  | 'bool'
  | 'other';

export function dtypeClass(dtype: string): DtypeClass {
  const d = dtype.trim().toUpperCase();
  // Before the plain names: `F8_E4M3` must not fall through to the wide-float arm on its
  // leading `F`. It is the narrowest float there is, not the widest.
  if (d.startsWith('F8')) return 'float-narrow';
  switch (d) {
    case 'F64':
    case 'F32':
    case 'TF32':
      return 'float-wide';
    case 'F16':
    case 'BF16':
      return 'float-half';
    case 'I64':
    case 'U64':
    case 'I32':
    case 'U32':
      return 'int-wide';
    case 'I16':
    case 'U16':
    case 'I8':
    case 'U8':
      return 'int-narrow';
    case 'BOOL':
      return 'bool';
    default:
      return 'other';
  }
}

/**
 * The CSS variable that paints a dtype family.
 *
 * Variables rather than literals so the canvas follows the app's theme switch — including
 * Fallout, which has no Shiki-style counterpart and simply resolves these to its own greens.
 */
export function dtypeVar(dtype: string): string {
  switch (dtypeClass(dtype)) {
    case 'float-wide':
      return '--dt-float-wide';
    case 'float-half':
      return '--dt-float-half';
    case 'float-narrow':
      return '--dt-float-narrow';
    case 'int-wide':
      return '--dt-int-wide';
    case 'int-narrow':
      return '--dt-int-narrow';
    case 'bool':
      return '--dt-bool';
    default:
      return '--dt-other';
  }
}

/** One dtype's share of a file — mirrors `DtypeStat` in `crates/core/src/stats.rs`. */
export interface DtypeShare {
  dtype: string;
  count: number;
  bytes: number;
}

/**
 * The file's dtype composition, biggest byte share first — mirrors
 * `LayoutMap::dtype_tally`.
 *
 * Header and gap segments are excluded: those bytes are not any dtype. The ordering is the
 * contract, not a detail — the dominant dtype leads, and equal shares fall back to the name
 * so the list is stable rather than dependent on iteration order.
 */
export function dtypeTally(map: Pick<LayoutMap, 'segments'>): DtypeShare[] {
  const by = new Map<string, DtypeShare>();
  for (const s of map.segments) {
    if (s.kind.kind !== 'tensor') continue;
    const row = by.get(s.kind.dtype) ?? { dtype: s.kind.dtype, count: 0, bytes: 0 };
    row.count += 1;
    row.bytes += s.end - s.start;
    by.set(s.kind.dtype, row);
  }
  return [...by.values()].sort((a, b) => b.bytes - a.bytes || a.dtype.localeCompare(b.dtype));
}
