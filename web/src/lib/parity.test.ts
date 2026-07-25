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
import { humanCount, humanSize, percent } from './format';
import { searchTree } from './search';
import type { TreeNode } from './types';

interface Fixture {
  size: [number, string][];
  count: [number, string][];
  percent: [number, number, string][];
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
