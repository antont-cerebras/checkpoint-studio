// The packing schemas' URL round trip. The server's side of the same contract is `src/web/params.rs`
// (which generates the wire keys these use) and `src/compare.rs` (which parses the specs).

import { describe, expect, it } from 'vitest';
import {
  emptyPacking,
  isPackingSet,
  packingFromQuery,
  packingSummary,
  packingToQuery,
  samePacking,
} from './packing';

describe('an unsaid packing', () => {
  it('adds nothing to the URL, because absent means "infer it"', () => {
    expect(isPackingSet(emptyPacking())).toBe(false);
    expect(isPackingSet(undefined)).toBe(false);
    expect(packingToQuery(emptyPacking())).toEqual([]);
    expect(packingToQuery(undefined)).toEqual([]);
    // A field the reader opened and left blank is still unsaid.
    expect(isPackingSet({ baseline: '  ', candidate: '' })).toBe(false);
  });

  it('reads back as nothing at all, so an untouched link stays what it was', () => {
    expect(packingFromQuery(new URLSearchParams('lhs=%2Fa'))).toBeUndefined();
    expect(packingFromQuery(new URLSearchParams('repack_schema='))).toBeUndefined();
  });
});

describe('the URL round trip', () => {
  // The pairing this feature exists for: a sparse baseline against a merged candidate.
  it('carries each side under its own key', () => {
    const p = { baseline: '[4]', candidate: '[3,3,3,3,3]' };
    expect(packingToQuery(p)).toEqual([
      ['repack_schema', '[4]'],
      ['repack_schema_new', '[3,3,3,3,3]'],
    ]);
    const back = packingFromQuery(
      new URLSearchParams(new URLSearchParams(packingToQuery(p)).toString()),
    );
    expect(back).toEqual(p);
    expect(samePacking(p, back)).toBe(true);
  });

  it('carries one side alone — the other is still inferred', () => {
    expect(packingToQuery({ baseline: '', candidate: '3,3,3,3,3' })).toEqual([
      ['repack_schema_new', '3,3,3,3,3'],
    ]);
    const back = packingFromQuery(new URLSearchParams('repack_schema=4'));
    expect(back).toEqual({ baseline: '4', candidate: '' });
  });

  it('trims what it carries, so an edit that changes nothing refetches nothing', () => {
    expect(packingToQuery({ baseline: ' [4] ', candidate: '' })).toEqual([['repack_schema', '[4]']]);
    expect(samePacking({ baseline: '[4]', candidate: '' }, { baseline: ' [4]', candidate: ' ' })).toBe(
      true,
    );
    expect(samePacking({ baseline: '[4]', candidate: '' }, undefined)).toBe(false);
    expect(samePacking(undefined, undefined)).toBe(true);
  });
});

describe('saying the packing in one line', () => {
  it('names the side each schema applies to', () => {
    expect(packingSummary({ baseline: '[4]', candidate: '[3,3,3,3,3]' })).toBe(
      'baseline [4] · candidate [3,3,3,3,3]',
    );
    expect(packingSummary({ baseline: '', candidate: '[3,3,3,3,3]' })).toBe('candidate [3,3,3,3,3]');
    expect(packingSummary(emptyPacking())).toBe('');
    expect(packingSummary(undefined)).toBe('');
  });
});
