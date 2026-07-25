// Row shaping and cursor movement. The movement rules are the ones the TUI defines
// (←/→ walk the hierarchy, Shift+↑/↓ stay among siblings), so they're worth pinning
// exactly: a cursor that quietly jumps into a cousin group is the kind of bug that
// reads as "the arrow keys are broken" without being obviously wrong in the code.

import { describe, expect, it } from 'vitest';
import { flatten, nodeId, type Row } from './flatten';
import {
  clampIndex,
  firstChildIndex,
  matchRows,
  parentIndex,
  rowIndexOf,
  siblingIndex,
  sortRows,
} from './rows';
import type { TreeNode } from './types';

interface LeafOpts {
  dtype?: string;
  shape?: number[];
  size?: number;
  elements?: number;
}
const leaf = (name: string, o: LeafOpts = {}): TreeNode => ({
  kind: 'tensor',
  label: null,
  info: {
    name,
    dtype: o.dtype ?? 'F16',
    shape: o.shape ?? [1],
    size_bytes: o.size ?? 2,
    num_elements: o.elements ?? 1,
    storage: null,
    source_path: 's',
    layout: null,
  },
});
const grp = (name: string, children: TreeNode[]): TreeNode => ({
  kind: 'group',
  name,
  children,
  expanded: false,
  tensor_count: children.length,
  params: 0,
  total_size: 0,
  stored_size: 0,
});
const row = (node: TreeNode, depth = 0): Row => ({
  id: nodeId(node, ''),
  node,
  depth,
  hasChildren: node.kind === 'group' && node.children.length > 0,
});

describe('sortRows', () => {
  const rows = [
    row(leaf('b', { dtype: 'F32', shape: [2, 2], size: 16, elements: 4 })),
    row(leaf('a', { dtype: 'BF16', shape: [8], size: 16, elements: 8 })),
    row(leaf('c', { dtype: 'I8', shape: [4, 4, 4], size: 64, elements: 64 })),
  ];
  const names = (rs: Row[]) => rs.map((r) => (r.node.kind === 'tensor' ? r.node.info.name : '?'));

  it('sorts by each facet ascending', () => {
    expect(names(sortRows(rows, 'name', 'asc'))).toEqual(['a', 'b', 'c']);
    expect(names(sortRows(rows, 'size', 'asc'))).toEqual(['b', 'a', 'c']); // ties keep order
    expect(names(sortRows(rows, 'params', 'asc'))).toEqual(['b', 'a', 'c']);
    expect(names(sortRows(rows, 'rank', 'asc'))).toEqual(['a', 'b', 'c']);
    expect(names(sortRows(rows, 'dtype', 'asc'))).toEqual(['a', 'b', 'c']); // BF16 < F32 < I8
  });

  it('reverses for descending', () => {
    expect(names(sortRows(rows, 'params', 'desc'))).toEqual(['c', 'a', 'b']);
  });

  it('leaves the input untouched', () => {
    const before = names(rows);
    sortRows(rows, 'name', 'desc');
    expect(names(rows)).toEqual(before);
  });

  // Plain string order puts layers.10 before layers.2, which reads as scrambled to
  // anyone scanning a 48-layer checkpoint.
  it('orders layer numbers numerically, not lexically', () => {
    const layers = [10, 2, 1, 21, 3].map((i) => row(leaf(`model.layers.${i}.weight`)));
    expect(names(sortRows(layers, 'name', 'asc'))).toEqual([
      'model.layers.1.weight',
      'model.layers.2.weight',
      'model.layers.3.weight',
      'model.layers.10.weight',
      'model.layers.21.weight',
    ]);
  });

  it('treats rows with no tensor info as equal, so they hold their place', () => {
    const mixed = [row(grp('g', [])), row(leaf('a')), row(leaf('b'))];
    expect(sortRows(mixed, 'size', 'asc')).toHaveLength(3);
  });
});

describe('matchRows', () => {
  const tree = [grp('model', [grp('layers', [leaf('model.layers.0.w'), leaf('model.layers.1.w')]), leaf('model.norm')])];

  it('flattens to just the matching tensors, in tree order, at depth 0', () => {
    const rows = matchRows(tree, new Set(['model.norm', 'model.layers.1.w']));
    expect(rows.map((r) => r.id)).toEqual(['t:model.layers.1.w', 't:model.norm']);
    expect(rows.every((r) => r.depth === 0 && !r.hasChildren)).toBe(true);
  });

  it('ignores group names — only tensor names are matched', () => {
    expect(matchRows(tree, new Set(['model', 'layers']))).toEqual([]);
  });

  it('gives the same ids as the hierarchical flatten, so the cursor survives filtering', () => {
    const all = flatten(tree, new Set(['model', 'model/layers']));
    const matched = matchRows(tree, new Set(['model.layers.0.w']));
    expect(all.map((r) => r.id)).toContain(matched[0]!.id);
  });
});

describe('cursor', () => {
  //  0 model            depth 0
  //  1   layers         depth 1
  //  2     0.w          depth 2
  //  3     1.w          depth 2
  //  4   norm           depth 1
  //  5 lm_head          depth 0
  const rows: Row[] = [
    { id: 'model', node: grp('model', [leaf('x')]), depth: 0, hasChildren: true },
    { id: 'model/layers', node: grp('layers', [leaf('x')]), depth: 1, hasChildren: true },
    { id: 't:0.w', node: leaf('0.w'), depth: 2, hasChildren: false },
    { id: 't:1.w', node: leaf('1.w'), depth: 2, hasChildren: false },
    { id: 't:norm', node: leaf('norm'), depth: 1, hasChildren: false },
    { id: 't:lm_head', node: leaf('lm_head'), depth: 0, hasChildren: false },
  ];

  it('finds a row by id, and falls back to the top when it is gone', () => {
    expect(rowIndexOf(rows, 't:1.w')).toBe(3);
    expect(rowIndexOf(rows, 't:vanished')).toBe(0);
    expect(rowIndexOf(rows, null)).toBe(0);
    expect(rowIndexOf([], 't:x')).toBe(0);
  });

  it('clamps an index to the list', () => {
    expect(clampIndex(rows, -5)).toBe(0);
    expect(clampIndex(rows, 99)).toBe(5);
    expect(clampIndex([], 3)).toBe(0);
  });

  it('walks ← to the nearest shallower row, and stops at the top level', () => {
    expect(parentIndex(rows, 3)).toBe(1); // 1.w → layers
    expect(parentIndex(rows, 1)).toBe(0); // layers → model
    expect(parentIndex(rows, 4)).toBe(0); // norm → model
    expect(parentIndex(rows, 0)).toBeNull();
    expect(parentIndex(rows, 5)).toBeNull();
  });

  it('walks → into the first child only when one is showing', () => {
    expect(firstChildIndex(rows, 0)).toBe(1);
    expect(firstChildIndex(rows, 1)).toBe(2);
    expect(firstChildIndex(rows, 3)).toBeNull(); // 1.w → norm is shallower
    expect(firstChildIndex(rows, 5)).toBeNull(); // last row
  });

  it('stays among siblings and never crosses into another parent', () => {
    expect(siblingIndex(rows, 2, true)).toBe(3); // 0.w → 1.w
    expect(siblingIndex(rows, 3, false)).toBe(2); // 1.w → 0.w
    expect(siblingIndex(rows, 3, true)).toBeNull(); // no further depth-2 row under layers
    expect(siblingIndex(rows, 1, true)).toBe(4); // layers → norm
    expect(siblingIndex(rows, 0, true)).toBe(5); // model → lm_head
    expect(siblingIndex(rows, 2, false)).toBeNull(); // first child of layers
  });

  it('does not treat a cousin at the same depth as a sibling', () => {
    // a/x and b/y are both depth 1, but under different parents.
    const cousins: Row[] = [
      { id: 'a', node: grp('a', [leaf('x')]), depth: 0, hasChildren: true },
      { id: 't:a.x', node: leaf('a.x'), depth: 1, hasChildren: false },
      { id: 'b', node: grp('b', [leaf('y')]), depth: 0, hasChildren: true },
      { id: 't:b.y', node: leaf('b.y'), depth: 1, hasChildren: false },
    ];
    expect(siblingIndex(cousins, 1, true)).toBeNull();
    expect(siblingIndex(cousins, 3, false)).toBeNull();
  });

  it('reports no parent when the nested row is the first one showing', () => {
    // A truncated or re-shaped list can start mid-hierarchy; walking off the top must
    // stay put rather than select row 0 (which is not the parent).
    const nested: Row[] = [
      { id: 't:a', node: leaf('a'), depth: 1, hasChildren: false },
      { id: 't:b', node: leaf('b'), depth: 2, hasChildren: false },
    ];
    expect(parentIndex(nested, 0)).toBeNull();
    expect(parentIndex(nested, 1)).toBe(0);
  });

  it('handles an empty list without throwing', () => {
    expect(parentIndex([], 0)).toBeNull();
    expect(firstChildIndex([], 0)).toBeNull();
    expect(siblingIndex([], 0, true)).toBeNull();
  });
});
