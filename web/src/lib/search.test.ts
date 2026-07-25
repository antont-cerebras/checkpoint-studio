import { describe, expect, it } from 'vitest';
import { SEARCH_LIMIT, fuzzyScore, searchTree } from './search';
import type { TreeNode } from './types';

function tensor(name: string): TreeNode {
  return {
    kind: 'tensor',
    label: null,
    info: {
      name,
      dtype: 'F16',
      shape: [2, 2],
      size_bytes: 8,
      num_elements: 4,
      storage: null,
      source_path: 's.safetensors',
      layout: null,
    },
  };
}
function group(name: string, children: TreeNode[]): TreeNode {
  return {
    kind: 'group',
    name,
    children,
    expanded: false,
    tensor_count: children.length,
    params: 0,
    total_size: 0,
    stored_size: 0,
  };
}
const names = (t: TreeNode[], q: string) =>
  searchTree(t, q).rows.map((r) => (r.node.kind === 'tensor' ? r.node.info.name : ''));

const TREE = [
  group('model', [
    group('layers', [tensor('model.layers.0.q_proj.weight'), tensor('model.layers.10.q_proj.weight')]),
    tensor('model.norm.weight'),
  ]),
  tensor('lm_head.weight'),
];

describe('fuzzyScore', () => {
  it('matches a subsequence, not just a substring', () => {
    expect(fuzzyScore('qpw', 'q_proj.weight')).toBeGreaterThanOrEqual(0);
  });
  // Smart case, the rule the TUI's matcher uses: a lowercase query ignores case, but a
  // query carrying uppercase is matched literally. Pinned across both UIs by
  // parity.test.ts.
  it('ignores case for a lowercase query', () => {
    expect(fuzzyScore('qproj', 'Q_PROJ')).toBeGreaterThanOrEqual(0);
    expect(fuzzyScore('upper', 'MODEL.UPPER.weight')).toBeGreaterThanOrEqual(0);
  });
  it('matches literally once the query carries uppercase', () => {
    expect(fuzzyScore('QPROJ', 'q_proj')).toBe(-1);
    expect(fuzzyScore('UPPER', 'MODEL.UPPER.weight')).toBeGreaterThanOrEqual(0);
  });
  it('rejects a non-subsequence with -1', () => {
    expect(fuzzyScore('zzz', 'q_proj')).toBe(-1);
    expect(fuzzyScore('jorp', 'q_proj')).toBe(-1); // right letters, wrong order
  });
  it('scores a contiguous run above a scattered one', () => {
    expect(fuzzyScore('proj', 'q_proj')).toBeGreaterThan(fuzzyScore('proj', 'p.r.o.j'));
  });
  it('treats an empty needle as a match', () => {
    expect(fuzzyScore('', 'anything')).toBe(0);
  });
});

describe('searchTree', () => {
  it('finds leaves at any depth and skips groups', () => {
    expect(names(TREE, 'q_proj').sort()).toEqual([
      'model.layers.0.q_proj.weight',
      'model.layers.10.q_proj.weight',
    ]);
  });
  it('reports the untruncated total alongside the capped rows', () => {
    const many = Array.from({ length: SEARCH_LIMIT + 50 }, (_, i) => tensor(`w${i}.weight`));
    const found = searchTree([group('g', many)], 'weight');
    expect(found.total).toBe(SEARCH_LIMIT + 50);
    expect(found.rows).toHaveLength(SEARCH_LIMIT);
  });
  it('returns everything for an empty query', () => {
    expect(searchTree(TREE, '').total).toBe(4);
  });
  it('returns nothing (and total 0) when nothing matches', () => {
    const found = searchTree(TREE, 'zzzz');
    expect(found.rows).toEqual([]);
    expect(found.total).toBe(0);
  });
  it('gives every row a distinct id so keyed rendering is stable', () => {
    const ids = searchTree(TREE, 'weight').rows.map((r) => r.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
  it('re-searching the same tree is consistent (the index cache must not stale)', () => {
    const a = searchTree(TREE, 'norm');
    const b = searchTree(TREE, 'norm');
    expect(b.rows.map((r) => r.id)).toEqual(a.rows.map((r) => r.id));
    // A DIFFERENT tree object must not reuse the first tree's index.
    const other = [tensor('completely.different.weight')];
    expect(searchTree(other, 'norm').total).toBe(0);
    expect(searchTree(other, 'different').total).toBe(1);
  });
});
