import { describe, expect, it } from 'vitest';
import { flatten, nodeId } from './flatten';
import type { TreeNode } from './types';

const leaf = (name: string): TreeNode => ({
  kind: 'tensor',
  label: null,
  info: {
    name, dtype: 'F16', shape: [1], size_bytes: 2, num_elements: 1,
    storage: null, source_path: 's', layout: null,
  },
});
const grp = (name: string, children: TreeNode[]): TreeNode => ({
  kind: 'group', name, children, expanded: false,
  tensor_count: children.length, params: 0, total_size: 0, stored_size: 0,
});

const TREE = [grp('model', [grp('layers', [leaf('model.layers.0.w')]), leaf('model.norm')])];

describe('nodeId', () => {
  it('paths groups and namespaces leaves by kind', () => {
    expect(nodeId(grp('model', []), '')).toBe('model');
    expect(nodeId(grp('layers', []), 'model')).toBe('model/layers');
    expect(nodeId(leaf('a.b'), 'model')).toBe('t:a.b');
  });
  it('cannot collide a tensor with a metadata entry of the same name', () => {
    const meta: TreeNode = { kind: 'metadata', info: { name: 'x', value: 'v', value_type: 'str' } };
    expect(nodeId(leaf('x'), '')).not.toBe(nodeId(meta, ''));
  });
});

describe('flatten', () => {
  it('shows only the top level when nothing is expanded', () => {
    const rows = flatten(TREE, new Set());
    expect(rows).toHaveLength(1);
    expect(rows[0]!.depth).toBe(0);
    expect(rows[0]!.hasChildren).toBe(true);
  });
  it('reveals children in order, deepening as groups expand', () => {
    const rows = flatten(TREE, new Set(['model', 'model/layers']));
    expect(rows.map((r) => [r.id, r.depth])).toEqual([
      ['model', 0],
      ['model/layers', 1],
      ['t:model.layers.0.w', 2],
      ['t:model.norm', 1],
    ]);
  });
  it('expanding an inner group without its parent reveals nothing extra', () => {
    expect(flatten(TREE, new Set(['model/layers']))).toHaveLength(1);
  });
  it('marks an empty group as having no children (so it cannot be entered)', () => {
    const rows = flatten([grp('empty', [])], new Set(['empty']));
    expect(rows[0]!.hasChildren).toBe(false);
  });
});
