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
