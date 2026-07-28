import { describe, it, expect } from 'vitest';
import { gapSummary } from './layout';
import type { LayoutMap } from './types';

const seg = (start: number, end: number, kind: string) =>
  ({ name: '', start, end, kind: { kind } }) as unknown as LayoutMap['segments'][number];

describe('gapSummary', () => {
  it('counts and totals the gaps, matching the Rust gap_summary', () => {
    // Same fixture as `gaps_are_counted_and_totalled` in safelayout.rs: 28 + 100 bytes.
    const map = {
      segments: [
        seg(0, 8, 'header'),
        seg(8, 100, 'tensor'),
        seg(100, 128, 'gap'),
        seg(128, 300, 'tensor'),
        seg(300, 400, 'gap'),
      ],
    };
    expect(gapSummary(map)).toEqual({ count: 2, bytes: 128 });
  });

  it('reports nothing for a tightly packed file', () => {
    // The norm — which is why the summary line omits the field rather than showing a zero.
    expect(gapSummary({ segments: [seg(0, 8, 'header'), seg(8, 100, 'tensor')] })).toEqual({
      count: 0,
      bytes: 0,
    });
  });

  it('handles a file with no segments at all', () => {
    expect(gapSummary({ segments: [] })).toEqual({ count: 0, bytes: 0 });
  });
});
