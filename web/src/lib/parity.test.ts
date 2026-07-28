// The TypeScript half of the cross-language parity contract.
//
// `shared/parity/format.json` is generated from the Rust implementations by
// `cargo test --test parity`; Rust is the reference. This asserts the browser
// produces the same strings for the same inputs, so a display rule can't drift out of
// step between the TUI and the web UI without a test failing.
//
// A failure here means one of two things: the Rust side changed deliberately (match
// it), or the TypeScript is wrong (fix it). Never edit the fixture by hand — regenerate
// it with `UPDATE_PARITY=1 cargo test --test parity`.
//
// The rules that are deliberately NOT shared are listed in shared/parity/README.md.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { expandedIds, flatten, nodeId, type Row } from './flatten';
import { sortRows } from './rows';
import { humanCount, humanSize, percent, shardNote } from './format';
import { searchTree } from './search';
import type { TreeNode } from './types';

/** `[depth, kind, name, hasChildren]` — one tree row, as both flatteners see it. */
type RowProjection = [number, string, string, boolean];

/** A sort case: `key.dir` and the order it must produce. */
type SortCase = [string, string[]];

interface SortFixtureTensor {
  name: string;
  dtype: string;
  shape: number[];
  size_bytes: number;
  num_elements: number;
}

interface Fixture {
  size: [number, string][];
  count: [number, string][];
  percent: [number, number, string][];
  shard: [number, number, number, string][];
  tree: { nodes: TreeNode[]; rows: RowProjection[] };
  sort: { tensors: SortFixtureTensor[]; orders: SortCase[] };
  search: { names: string[]; matches: [string, string[]][] };
}

const here = dirname(fileURLToPath(import.meta.url));
// Read rather than import: the fixture lives outside the Vite project root, and this
// keeps the failure message ("regenerate it") in one place.
const fixturePath = join(here, '../../../shared/parity/format.json');
const fixture = JSON.parse(readFileSync(fixturePath, 'utf8')) as Fixture;

const HINT = 'regenerate with `UPDATE_PARITY=1 cargo test --test parity` after an intentional Rust change';

describe('byte sizes match the Rust format_size', () => {
  it.each(fixture.size)('%i → %s', (bytes, expected) => {
    expect(humanSize(bytes), HINT).toBe(expected);
  });
});

describe('parameter counts match the Rust format_parameters', () => {
  it.each(fixture.count)('%i → %s', (n, expected) => {
    expect(humanCount(n), HINT).toBe(expected);
  });
});

describe('zero fractions match the Rust format_percent', () => {
  it.each(fixture.percent)('%i of %i → %s', (zeros, count, expected) => {
    expect(percent(zeros / count, zeros === 0), HINT).toBe(expected);
  });
});

// The share arrives from the server rather than being divided here, so the fixture's
// own `params_share` is the input — exactly as `/api/files` delivers it.
describe('shard rows read the same as the TUI file browser', () => {
  it.each(fixture.shard)(
    '%i tensors, %i params → %s',
    (tensors, params, paramsShare, expected) => {
      expect(shardNote({ tensors, params, params_share: paramsShare }), HINT).toBe(expected);
    },
  );
});

describe('the search matcher matches the same names as the TUI', () => {
  // A flat tree of the fixture's names, which is what the matcher walks.
  const tree: TreeNode[] = fixture.search.names.map((name) => ({
    kind: 'tensor',
    label: null,
    info: {
      name,
      dtype: 'F16',
      shape: [1],
      size_bytes: 2,
      num_elements: 1,
      storage: null,
      source_path: 's',
      layout: null,
    },
  }));

  // Only the SET is contracted — the two matchers score differently, so the order is
  // not comparable (see the README).
  const matched = (q: string) =>
    searchTree(tree, q)
      .rows.map((r) => (r.node.kind === 'tensor' ? r.node.info.name : ''))
      .sort();

  it.each(fixture.search.matches)('%j matches the same names', (query, expected) => {
    expect(matched(query), HINT).toEqual(expected);
  });

  it('reports the match total, not the capped row count', () => {
    // The cap only applies to rows; `total` must stay honest (finding F2).
    const many: TreeNode[] = Array.from({ length: 1500 }, (_, i) => ({
      kind: 'tensor',
      label: null,
      info: {
        name: `model.layers.${i}.weight`,
        dtype: 'F16',
        shape: [1],
        size_bytes: 2,
        num_elements: 1,
        storage: null,
        source_path: 's',
        layout: null,
      },
    }));
    const { rows, total } = searchTree(many, 'weight');
    expect(rows).toHaveLength(1000);
    expect(total).toBe(1500);
  });
});

describe('the tree flattens to the same rows as the Rust flattener', () => {
  // The structural half of "the web UI looks like the TUI": the same rows, in the same
  // order, at the same depths, with the same expand affordance — starting from the fold
  // state the server sent. The rendered *text* of a row is deliberately not contracted
  // (see shared/parity/README.md); the row list is.
  it('produces the fixture rows', () => {
    const { nodes, rows } = fixture.tree;
    const got: RowProjection[] = flatten(nodes, expandedIds(nodes)).map((r) => [
      r.depth,
      r.node.kind,
      r.node.kind === 'group' ? r.node.name : r.node.info.name,
      r.hasChildren,
    ]);
    expect(got, HINT).toEqual(rows);
  });

  it('honors the served fold state rather than expanding only the root', () => {
    // Pinned because the seed used to be `new Set([rootId])`, which opened the browser
    // on a different first screen than the terminal for the very same checkpoint.
    const { nodes } = fixture.tree;
    const ids = expandedIds(nodes);
    expect(ids.size, 'the fixture should have an expanded group to find').toBeGreaterThan(0);
    const deepest = Math.max(...flatten(nodes, ids).map((r) => r.depth));
    expect(deepest, 'a group the server sent expanded must contribute deeper rows').toBeGreaterThan(0);
  });
});

describe('the flat list sorts the same way the Rust side sorts', () => {
  // `sortRows` and `kernel::sort_rows` are two implementations of one rule. The cases
  // are chosen so each key produces a different winner — a sort that ignored its key
  // would pass a fixture where every order happened to coincide.
  const rows = (): Row[] =>
    fixture.sort.tensors.map((info) => {
      const node = { kind: 'tensor', info, label: null } as unknown as TreeNode;
      return { id: nodeId(node, ''), node, depth: 0, hasChildren: false };
    });

  it.each(fixture.sort.orders)('%s', (label, expected) => {
    const [key, dir] = label.split('.') as [Parameters<typeof sortRows>[1], 'asc' | 'desc'];
    const got = sortRows(rows(), key, dir).map((r) =>
      r.node.kind === 'tensor' ? r.node.info.name : '',
    );
    expect(got, HINT).toEqual(expected);
  });

  it('leaves the natural order alone for the sortless case', () => {
    // `none` is not in the fixture because it is a no-op by definition; pin that it
    // really is one, since a "sort by nothing" that reordered would be invisible above.
    const before = rows().map((r) => (r.node.kind === 'tensor' ? r.node.info.name : ''));
    const after = sortRows(rows(), 'name', 'asc').map((r) =>
      r.node.kind === 'tensor' ? r.node.info.name : '',
    );
    expect(after).not.toEqual(before); // the sample is deliberately unsorted
  });
});
