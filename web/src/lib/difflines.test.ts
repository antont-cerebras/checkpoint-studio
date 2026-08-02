// The row-level reading rules of the diff report. The interesting one is `shapeDiff`: a fused tensor is
// its unfused parts stacked along a new leading axis, so the comparison has to align from the right or
// every dimension "changes" and the row says nothing.

import { describe, expect, it } from 'vitest';
import {
  byChangeKind,
  changeKind,
  changeKindLabel,
  shapeDiff,
  sigCells,
  sigText,
  splitName,
} from './difflines';
import type { TensorChange } from './types';

const sig = (dtype: string, shape: number[]) => ({ dtype, shape });
const change = (name: string, from: [string, number[]], to: [string, number[]]): TensorChange => ({
  name,
  old: sig(...from),
  new: sig(...to),
});

describe('splitting a name into its path and leaf', () => {
  it('keeps the trailing dot on the path, so the two concatenate back', () => {
    const { path, leaf } = splitName('model.layers.0.self_attn.qkv_proj.weight');
    expect(path).toBe('model.layers.0.self_attn.qkv_proj.');
    expect(leaf).toBe('weight');
    expect(path + leaf).toBe('model.layers.0.self_attn.qkv_proj.weight');
  });

  it('treats a top-level name as all leaf', () => {
    expect(splitName('lm_head')).toEqual({ path: '', leaf: 'lm_head' });
  });
});

describe('marking the dimensions that changed', () => {
  it('aligns from the right, so a fold reads as one added dimension', () => {
    // The real case: 256 unfused `(3072, 1536)` tensors against one fused `(256, 3072, 1540)`.
    const d = shapeDiff([3072, 1536], [256, 3072, 1540]);
    expect(d.old).toEqual([false, true]); // 3072 matches; 1536 → 1540 is the change
    expect(d.new).toEqual([true, false, true]); // the leading 256 has no counterpart
  });

  it('marks nothing when the shapes agree', () => {
    expect(shapeDiff([4, 8], [4, 8])).toEqual({ old: [false, false], new: [false, false] });
  });

  it('marks every dimension of a shape whose counterpart is empty', () => {
    expect(shapeDiff([4, 8], [])).toEqual({ old: [true, true], new: [] });
    expect(shapeDiff([], [4])).toEqual({ old: [], new: [true] });
  });

  it('marks a differing dimension in the middle', () => {
    expect(shapeDiff([2, 3, 4], [2, 9, 4]).new).toEqual([false, true, false]);
  });
});

describe('classifying a changed row', () => {
  it('tells the four kinds apart', () => {
    expect(changeKind(change('a', ['F16', [4]], ['BF16', [4]]))).toBe('dtype');
    expect(changeKind(change('a', ['F16', [4]], ['F16', [8]]))).toBe('shape');
    expect(changeKind(change('a', ['F16', [4]], ['U8', [8]]))).toBe('both');
    // Same signature: the report only calls that changed when `--values` found differing numbers.
    expect(changeKind(change('a', ['F16', [4]], ['F16', [4]]))).toBe('values');
  });

  it('has a heading for each', () => {
    for (const k of ['dtype', 'shape', 'both', 'values'] as const) {
      expect(changeKindLabel(k).length).toBeGreaterThan(0);
    }
    expect(changeKindLabel('both')).toContain('dtype');
    expect(changeKindLabel('both')).toContain('shape');
  });

  it('groups in a fixed order, and drops the kinds with no rows', () => {
    const rows = [
      change('one', ['F16', [4]], ['BF16', [4]]), // dtype
      change('two', ['F16', [4]], ['U8', [8]]), // both
      change('three', ['F16', [4]], ['BF16', [4]]), // dtype
    ];
    const grouped = byChangeKind(rows);
    // `both` first — the biggest change reads first — then dtype. No `shape`, no `values`.
    expect(grouped.map((g) => g.kind)).toEqual(['both', 'dtype']);
    expect(grouped.map((g) => g.rows.length)).toEqual([1, 2]);
    // Every row appears exactly once.
    expect(grouped.flatMap((g) => g.rows.map((r) => r.name)).sort()).toEqual([
      'one',
      'three',
      'two',
    ]);
  });

  it('groups nothing into nothing', () => {
    expect(byChangeKind([])).toEqual([]);
  });
});

describe('the one-sided rows', () => {
  it('read as dtype and shape', () => {
    expect(sigText(sig('BF16', [6, 4]))).toBe('BF16 (6, 4)');
    expect(sigText(sig('F32', []))).toBe('F32 ()');
  });
});

describe('marking only the piece of a signature that differs', () => {
  it('marks the dtype and leaves a shape that matches alone', () => {
    const c = sigCells(sig('BF16', [6, 4]), sig('F16', [6, 4]));
    expect(c.dtype).toEqual({ text: 'BF16', differs: true });
    expect(c.dims.map((d) => d.differs)).toEqual([false, false]);
  });

  it('marks the dimension and leaves a dtype that matches alone', () => {
    const c = sigCells(sig('F32', [8]), sig('F32', [4]));
    expect(c.dtype.differs).toBe(false);
    expect(c.dims).toEqual([{ text: '8', differs: true }]);
  });

  it('aligns from the right, so the fused side marks only its new leading axis', () => {
    const unfused = sigCells(sig('F16', [3072, 1536]), sig('U16', [256, 3072, 1540]));
    expect(unfused.dims.map((d) => d.differs)).toEqual([false, true]);
    const fused = sigCells(sig('U16', [256, 3072, 1540]), sig('F16', [3072, 1536]));
    expect(fused.dims.map((d) => d.differs)).toEqual([true, false, true]);
  });

  it('marks nothing at all when the two signatures agree', () => {
    const c = sigCells(sig('F16', [4, 2]), sig('F16', [4, 2]));
    expect(c.dtype.differs).toBe(false);
    expect(c.dims.every((d) => !d.differs)).toBe(true);
  });

  // A one-sided row: the band and the name already say it is missing from the other side, and there is
  // no counterpart for "differs" to mean anything against.
  it('marks nothing when there is no other side', () => {
    const c = sigCells(sig('F16', [4, 2]), null);
    expect(c.dtype.differs).toBe(false);
    expect(c.dims.map((d) => d.differs)).toEqual([false, false]);
    expect(c.dims.map((d) => d.text)).toEqual(['4', '2']);
  });
});
