// Shaping the flattened row list and moving the cursor through it. Both are pure
// functions of (rows, index) — the store only supplies the rows and applies the
// result — so the movement rules that mirror the TUI's (`kernel::TreeState`) can be
// read side by side and tested directly, instead of being buried in store callbacks.

import { nodeId, type Row } from './flatten';
import type { SortKey } from './hash';
import type { TreeNode } from './types';

/** Sort a flat tensor-row list by a facet, ascending or descending. Rows that carry
 * no tensor info (groups, metadata) compare equal, so they keep their relative order.
 * Never mutates the input. */
export function sortRows(rows: Row[], key: Exclude<SortKey, 'none'>, dir: 'asc' | 'desc'): Row[] {
  const info = (r: Row) => (r.node.kind === 'tensor' ? r.node.info : null);
  const cmp = (a: Row, b: Row): number => {
    const ia = info(a);
    const ib = info(b);
    if (!ia || !ib) return 0;
    switch (key) {
      case 'name':
        // Numeric collation, so `layers.2` sorts before `layers.10`.
        return ia.name.localeCompare(ib.name, undefined, { numeric: true });
      case 'size':
        return ia.size_bytes - ib.size_bytes;
      case 'params':
        return ia.num_elements - ib.num_elements;
      case 'rank':
        return ia.shape.length - ib.shape.length;
      case 'dtype':
        return ia.dtype.localeCompare(ib.dtype);
    }
  };
  // Negate the comparator for descending rather than reversing the sorted array:
  // `Array.sort` is stable, and reversing afterwards flips the order of *equal* rows too,
  // so two same-rank tensors came out in the opposite order from the Rust side (which
  // negates). Caught by the sort case in `shared/parity/format.json`.
  const sign = dir === 'asc' ? 1 : -1;
  return [...rows].sort((a, b) => sign * cmp(a, b));
}

/** Flat list of the tensor rows whose names the server said pass the filter. */
export function matchRows(nodes: TreeNode[], matches: Set<string>): Row[] {
  const out: Row[] = [];
  const walk = (ns: TreeNode[], parentId: string) => {
    for (const n of ns) {
      const id = nodeId(n, parentId);
      if (n.kind === 'group') walk(n.children, id);
      else if (n.kind === 'tensor' && matches.has(n.info.name))
        out.push({ id, node: n, depth: 0, hasChildren: false });
    }
  };
  walk(nodes, '');
  return out;
}

/** Where a row id sits in the list; 0 (the top) when it isn't there any more — a
 * filter or a fold can drop the selected row out from under the cursor. */
export function rowIndexOf(rows: Row[], id: string | null): number {
  const i = rows.findIndex((r) => r.id === id);
  return i < 0 ? 0 : i;
}

/** Clamp an index into the list (an empty list yields 0, which no row occupies). */
export function clampIndex(rows: Row[], i: number): number {
  return Math.max(0, Math.min(rows.length - 1, i));
}

/** ← : the parent group — the nearest preceding shallower row. `null` at depth 0. */
export function parentIndex(rows: Row[], i: number): number | null {
  const depth = rows[i]?.depth ?? 0;
  if (depth === 0) return null;
  for (let k = i - 1; k >= 0; k--) {
    const row = rows[k];
    if (row && row.depth < depth) return k;
  }
  return null;
}

/** → : the group's first child, if the next row is one (i.e. it's expanded). */
export function firstChildIndex(rows: Row[], i: number): number | null {
  const row = rows[i];
  const next = rows[i + 1];
  if (!row || !next) return null;
  return next.depth === row.depth + 1 ? i + 1 : null;
}

/** Shift+↑/↓ : the previous/next row at the same depth, without leaving the parent —
 * the scan stops at the first shallower row rather than jumping into a cousin. */
export function siblingIndex(rows: Row[], i: number, forward: boolean): number | null {
  const depth = rows[i]?.depth ?? 0;
  const step = forward ? 1 : -1;
  for (let k = i + step; k >= 0 && k < rows.length; k += step) {
    const row = rows[k];
    if (!row || row.depth < depth) return null;
    if (row.depth === depth) return k;
  }
  return null;
}
