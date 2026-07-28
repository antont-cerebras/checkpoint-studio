import { describe, it, expect } from 'vitest';
import { gapSummary, dtypeClass, dtypeVar, dtypeTally } from './layout';
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

describe('dtypeClass', () => {
  it('agrees with the Rust DtypeClass::of table', () => {
    // Same cases as `every_dtype_the_codebase_knows_lands_in_a_family` in stats.rs.
    const cases: [string, string][] = [
      ['F64', 'float-wide'],
      ['F32', 'float-wide'],
      ['F16', 'float-half'],
      ['BF16', 'float-half'],
      ['F8_E4M3', 'float-narrow'],
      ['F8_E5M2', 'float-narrow'],
      ['I64', 'int-wide'],
      ['U64', 'int-wide'],
      ['I32', 'int-wide'],
      ['U32', 'int-wide'],
      ['I16', 'int-narrow'],
      ['U16', 'int-narrow'],
      ['I8', 'int-narrow'],
      ['U8', 'int-narrow'],
      ['BOOL', 'bool'],
    ];
    for (const [dtype, want] of cases) expect(dtypeClass(dtype), dtype).toBe(want);
  });

  it('does not mistake an 8-bit float for a wide one', () => {
    // The ordering trap both implementations have to get right.
    expect(dtypeClass('F8_E4M3')).toBe('float-narrow');
    expect(dtypeClass('F8_E4M3')).not.toBe(dtypeClass('F32'));
  });

  it('ignores spelling', () => {
    for (const n of ['bf16', 'BF16', ' bf16 ', 'Bf16']) expect(dtypeClass(n), n).toBe('float-half');
  });

  it('leaves an unknown dtype neutral rather than guessing', () => {
    for (const n of ['', 'u4', 'MXFP4', 'complex64', '???']) expect(dtypeClass(n), n).toBe('other');
  });

  it('maps every family to a distinct CSS variable', () => {
    const vars = new Set(
      ['F32', 'BF16', 'F8_E4M3', 'I32', 'I8', 'BOOL', 'weird'].map((d) => dtypeVar(d)),
    );
    expect(vars.size).toBe(7);
  });
});

describe('dtypeTally', () => {
  it('matches the Rust LayoutMap::dtype_tally on the same fixture', () => {
    // Same segments as `the_dtype_tally_describes_the_file_and_excludes_non_tensor_bytes`.
    const t = (start: number, end: number, dtype: string) =>
      ({ name: '', start, end, kind: { kind: 'tensor', dtype, shape: [1] } }) as never;
    const map = {
      segments: [
        { name: 'header', start: 0, end: 128, kind: { kind: 'header' } } as never,
        t(128, 628, 'BF16'),
        t(628, 728, 'U8'),
        t(728, 1028, 'BF16'),
        { name: 'gap', start: 1028, end: 1128, kind: { kind: 'gap' } } as never,
      ],
    };
    expect(dtypeTally(map)).toEqual([
      { dtype: 'BF16', count: 2, bytes: 800 },
      { dtype: 'U8', count: 1, bytes: 100 },
    ]);
  });

  it('orders equal shares by name so the legend is stable', () => {
    const t = (start: number, end: number, dtype: string) =>
      ({ name: '', start, end, kind: { kind: 'tensor', dtype, shape: [1] } }) as never;
    const got = dtypeTally({ segments: [t(0, 8, 'U8'), t(8, 16, 'I8'), t(16, 24, 'F16')] });
    expect(got.map((d) => d.dtype)).toEqual(['F16', 'I8', 'U8']);
  });

  it('has no rows for a file with no tensors', () => {
    expect(dtypeTally({ segments: [] })).toEqual([]);
  });
});
