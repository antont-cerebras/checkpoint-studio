// The client-owned VIEW STATE, mirroring the TUI's full-screen modal flow: one
// active screen at a time, a browser-style history stack (Backspace / \), and the
// persistent tree fold/selection/search state. Navigation + tree-cursor helpers
// live here so the global key handler (App.svelte) and the views share them.

import { derived, get, writable } from 'svelte/store';
import { api } from '../lib/api';
import { flatten, nodeId, type Row } from '../lib/flatten';
import { SEARCH_LIMIT, searchMatchCount, searchRows } from '../lib/search';
import type { TreeNode } from '../lib/types';
import { tree as treeData } from './server';

export type DataTab = 'info' | 'heatmap' | 'values' | 'histogram';

/** Data-view (heatmap / numeric grid) params carried in the detail hash so a
 * specific view — mode, sample size, dtype override, window offsets, base/zebra,
 * ratio lock — is reproducible from a shared/bookmarked URL and across back/forward.
 * Raw strings; `DataView` owns the semantics + defaults. */
export const DV_KEYS = ['mode', 'rows', 'cols', 'dtype', 'roff', 'coff', 'slice', 'base', 'zebra', 'lock'] as const;
export type DvParams = Partial<Record<(typeof DV_KEYS)[number], string>>;

export type Screen =
  | { kind: 'tree' }
  | { kind: 'detail'; tensor: string; tab: DataTab; dv?: DvParams }
  | { kind: 'files' }
  | { kind: 'layout'; file?: string }
  | { kind: 'stats' }
  | { kind: 'health' }
  | { kind: 'preview'; path: string; name: string };

// The current screen is driven by the URL hash, so the browser's back/forward
// buttons work natively and every screen+mode has a shareable link.
export const screen = writable<Screen>(parseHash());

function screenToHash(s: Screen): string {
  const enc = encodeURIComponent;
  switch (s.kind) {
    case 'tree':
      return 'tree';
    case 'detail': {
      let h = `detail?t=${enc(s.tensor)}&tab=${s.tab}`;
      for (const k of DV_KEYS) {
        const v = s.dv?.[k];
        if (v != null) h += `&${k}=${enc(v)}`;
      }
      return h;
    }
    case 'files':
      return 'files';
    case 'layout':
      return s.file ? `layout?file=${enc(s.file)}` : 'layout';
    case 'stats':
      return 'stats';
    case 'health':
      return 'health';
    case 'preview':
      return `preview?path=${enc(s.path)}&name=${enc(s.name)}`;
  }
}

function parseHash(): Screen {
  const raw = location.hash.replace(/^#/, '');
  const [kind, queryStr] = raw.split('?');
  const q = new URLSearchParams(queryStr ?? '');
  switch (kind) {
    case 'detail': {
      const t = q.get('t');
      const raw = q.get('tab') ?? 'info';
      const tab = (['info', 'heatmap', 'values', 'histogram'].includes(raw) ? raw : 'info') as DataTab;
      if (t) {
        const dv: DvParams = {};
        for (const k of DV_KEYS) {
          const v = q.get(k);
          if (v != null) dv[k] = v;
        }
        return { kind: 'detail', tensor: t, tab, dv: Object.keys(dv).length ? dv : undefined };
      }
      break;
    }
    case 'files':
      return { kind: 'files' };
    case 'layout':
      return { kind: 'layout', file: q.get('file') ?? undefined };
    case 'stats':
      return { kind: 'stats' };
    case 'health':
      return { kind: 'health' };
    case 'preview': {
      const path = q.get('path');
      if (path) return { kind: 'preview', path, name: q.get('name') ?? path };
      break;
    }
  }
  return { kind: 'tree' };
}

// Keep the stores in sync with the URL (covers browser back/forward + shared links).
window.addEventListener('hashchange', () => {
  screen.set(parseHash());
  restoreGlobals();
});

// Persistent tree state (survives screen changes, like the TUI).
export const expanded = writable<Set<string>>(new Set());
export const selectedId = writable<string | null>(null);
export const searching = writable<boolean>(false);
export const search = writable<string>('');

/** The tensor filter as a text query (the `tensorfilter` grammar), matched
 * server-side by the one shared matcher — so web and TUI filter identically.
 * `filterMatches` is the set of matching tensor names (null = inactive → show
 * all); `filterError` holds a parse error to show inline; `filterResolvedFor` is the
 * (trimmed) query that `filterMatches`/`filterError` currently reflect. The UI
 * derives "still filtering" as `query !== filterResolvedFor` — a pure reactive check
 * off stores set in the async resolve, so it renders reliably (setting a flag
 * synchronously inside this subscriber did not repaint the count in Svelte 4). */
export const filterQuery = writable<string>('');
export const filterMatches = writable<Set<string> | null>(null);
export const filterError = writable<string | null>(null);
export const filterResolvedFor = writable<string>('');

// Debounced fetch: whenever the query changes, ask the server which tensors pass.
// `filterReq` is bumped on every edit (not just when the timer fires), so an
// in-flight response for a superseded query — even one still in the debounce
// window — is dropped and can't desync the count/rows.
let filterTimer: ReturnType<typeof setTimeout> | undefined;
let filterReq = 0;
filterQuery.subscribe((q) => {
  clearTimeout(filterTimer);
  const query = q.trim();
  const req = ++filterReq;
  if (!query) {
    filterMatches.set(null);
    filterError.set(null);
    filterResolvedFor.set(''); // empty query is "resolved" (shows all) — no pending state
    return;
  }
  filterTimer = setTimeout(async () => {
    try {
      const res = await api.filter(query);
      if (req !== filterReq) return; // superseded by a newer edit
      filterMatches.set(res.active ? new Set(res.names ?? []) : null);
      filterError.set(null);
    } catch (e) {
      if (req !== filterReq) return;
      filterError.set(e instanceof Error ? e.message : String(e));
      filterMatches.set(null); // don't leave the prior result behind the error
    } finally {
      // Mark THIS query resolved (drops "filtering…"); a newer edit already bumped
      // `filterReq`, so a superseded response leaves the pending state in place.
      if (req === filterReq) filterResolvedFor.set(query);
    }
  }, 200);
});

/** Command palette (Space / `:`) open state. */
export const paletteOpen = writable<boolean>(false);

/** Sorting for the flat (filter / search) tensor list. `none` keeps the natural
 * order (fuzzy-score for search, tree order for a filter); the tree view is never
 * reordered. */
export type SortKey = 'none' | 'name' | 'size' | 'params' | 'dtype' | 'rank';
export const sortKey = writable<SortKey>('none');
export const sortDir = writable<'asc' | 'desc'>('asc');

/** Pick a sort facet, defaulting its direction sensibly: size / params read
 * biggest-first (what you want when eyeballing "the heavy tensors"), the rest
 * ascending. Still toggleable afterwards via the direction button. Restoring a
 * shared link sets the stores directly and bypasses this default. */
export function setSort(k: SortKey): void {
  sortKey.set(k);
  sortDir.set(k === 'size' || k === 'params' ? 'desc' : 'asc');
}

/** Compact per-family view toggle (the `≡` button) — a view mode, so it's in the URL. */
export const compact = writable<boolean>(false);

/** The flattened rows — fold-aware, or a flat list while filtering / searching.
 * Shared by TreeView (render) and the key handler (cursor movement). A filter
 * (server-matched) takes precedence over an in-progress fuzzy search; a flat list
 * is optionally sorted. */
export const visibleRows = derived(
  [treeData, expanded, search, searching, filterMatches, sortKey, sortDir],
  ([$t, $exp, $q, $searching, $matches, $sk, $sd]) => {
    if (!$t) return [] as Row[];
    // An in-progress fuzzy search wins; otherwise a set filter; otherwise the tree.
    let rows: Row[];
    if ($searching && $q.trim()) rows = searchRows($t.tree, $q.trim());
    else if ($matches) rows = matchRows($t.tree, $matches);
    else return flatten($t.tree, $exp); // hierarchical view is never reordered
    return $sk === 'none' ? rows : sortRows(rows, $sk, $sd);
  },
);

/** Total tensors matching the current fuzzy search, untruncated — so the search
 * label can be honest when the row list is capped ("showing 1000 of N"). Only
 * the fuzzy path is capped; the server-side filter returns every match. */
export { SEARCH_LIMIT };
export const searchTotal = derived([treeData, search, searching], ([$t, $q, $searching]) =>
  $t && $searching && $q.trim() ? searchMatchCount($t.tree, $q.trim()) : 0,
);

/** Sort a flat tensor-row list by a facet, ascending or descending. */
function sortRows(rows: Row[], key: Exclude<SortKey, 'none'>, dir: 'asc' | 'desc'): Row[] {
  const info = (r: Row) => (r.node.kind === 'tensor' ? r.node.info : null);
  const cmp = (a: Row, b: Row): number => {
    const ia = info(a);
    const ib = info(b);
    if (!ia || !ib) return 0;
    switch (key) {
      case 'name':
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
  const sorted = [...rows].sort(cmp);
  return dir === 'asc' ? sorted : sorted.reverse();
}

/** Flat list of the tensor rows whose names the server said pass the filter. */
function matchRows(nodes: TreeNode[], matches: Set<string>): Row[] {
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

/** Append one or more filter terms (badge click / builder), space-joined + deduped. */
export function addFilterTerms(terms: string[]): void {
  const add = terms.map((t) => t.trim()).filter(Boolean);
  if (!add.length) return;
  filterQuery.update((q) => {
    let cur = q.trim();
    for (const t of add) if (!` ${cur} `.includes(` ${t} `)) cur = cur ? `${cur} ${t}` : t;
    return cur;
  });
  searching.set(false);
  search.set('');
  navigate({ kind: 'tree' });
}
export function addFilterTerm(term: string): void {
  addFilterTerms([term]);
}
export function filterByDtype(value: string): void {
  addFilterTerm(`dtype:${value}`);
}
export function filterByShape(dims: number[]): void {
  addFilterTerm(`shape:(${dims.join(',')})`);
}
export function filterByDim(value: number): void {
  addFilterTerm(`dim:${value}`);
}
export function clearFilter(): void {
  filterQuery.set('');
}

// Expand the synthetic root node once the tree first loads, so its children show.
let seededExpand = false;
treeData.subscribe((t) => {
  if (t && !seededExpand) {
    seededExpand = true;
    const first = t.tree[0];
    if (first) expanded.set(new Set([nodeId(first, '')]));
  }
});

// ---- navigation + URL state (the hash is the single source of truth; a shared
// link and browser back/forward both restore the full view state) ----

/** The screen-independent view state carried in every hash: filter query, sort,
 * the compact toggle, and search. So any state is reproducible from the URL. */
function globalQuery(): string {
  const p = new URLSearchParams();
  const f = get(filterQuery).trim();
  if (f) p.set('filter', f);
  const sk = get(sortKey);
  if (sk !== 'none') p.set('sort', `${sk}.${get(sortDir)}`);
  if (get(compact)) p.set('compact', '1');
  if (get(searching)) p.set('q', get(search)); // presence ⇒ search mode (empty ok)
  return p.toString();
}

/** A screen's own hash plus the global state. */
function hashFor(s: Screen): string {
  const base = screenToHash(s);
  const g = globalQuery();
  if (!g) return base;
  return base.includes('?') ? `${base}&${g}` : `${base}?${g}`;
}

let restoring = false;

/** Restore the global stores from the current hash (initial load + back/forward). */
function restoreGlobals(): void {
  restoring = true;
  const q = new URLSearchParams(location.hash.replace(/^#/, '').split('?')[1] ?? '');
  filterQuery.set(q.get('filter') ?? '');
  const sort = q.get('sort');
  if (sort) {
    const [k, d] = sort.split('.');
    sortKey.set((['name', 'size', 'params', 'dtype', 'rank'].includes(k) ? k : 'none') as SortKey);
    sortDir.set(d === 'desc' ? 'desc' : 'asc');
  } else {
    sortKey.set('none');
  }
  compact.set(q.get('compact') === '1');
  const qs = q.get('q');
  searching.set(qs !== null);
  search.set(qs ?? '');
  restoring = false;
}

/** Mirror the current screen + global state into the hash without a new history
 * entry (replaceState doesn't fire hashchange, so this can't loop). */
function syncHash(): void {
  if (restoring) return;
  const h = `#${hashFor(get(screen))}`;
  if (location.hash !== h) history.replaceState(history.state, '', h);
}

export function navigate(s: Screen, replace = false): void {
  const h = `#${hashFor(s)}`;
  if (replace) {
    // Replace the current entry (no new history) — for view-state changes within a
    // screen (e.g. a detail tab) so Back/Esc leaves the screen in one step.
    history.replaceState(history.state, '', h);
  } else if (location.hash !== h) {
    location.hash = h; // pushes a history entry (hashchange also confirms the store)
  }
  screen.set(s); // optimistic; the hashchange listener confirms on the push path
}

// Restore any global state from the initial URL, THEN mirror later edits into the
// hash. Order matters: restore first so the subscriptions' initial fire doesn't
// stomp the link's params before they're read.
restoreGlobals();
filterQuery.subscribe(syncHash);
sortKey.subscribe(syncHash);
sortDir.subscribe(syncHash);
compact.subscribe(syncHash);
search.subscribe(syncHash);
searching.subscribe(syncHash);
export function back(): void {
  history.back();
}
export function forward(): void {
  history.forward();
}

export function openDetail(tensor: string): void {
  navigate({ kind: 'detail', tensor, tab: 'info' });
}

/** Open a file from the browser: a safetensors shard jumps to its byte-layout map;
 * anything else gets a text preview. */
export function openFile(path: string, name: string, fileKind: string): void {
  if (fileKind === 'Checkpoint') navigate({ kind: 'layout', file: name });
  else navigate({ kind: 'preview', path, name });
}
export function setTab(tab: DataTab): void {
  const s = get(screen);
  // A tab is view state within the detail, not a navigation step: replace the URL
  // so Esc / Back leaves the detail in one press from any tab (still deep-linkable).
  // Drop the data-view params — they're per-tab (a heatmap's rows/mode don't carry
  // to the grid), so each tab starts from its own defaults.
  if (s.kind === 'detail') navigate({ kind: 'detail', tensor: s.tensor, tab, dv: undefined }, true);
}

/** The saved data-view params for the current detail screen (empty otherwise) —
 * `DataView` reads these on mount to restore a deep-linked view. */
export function getDataView(): DvParams {
  const s = get(screen);
  return s.kind === 'detail' ? { ...(s.dv ?? {}) } : {};
}

/** Mirror `DataView`'s current params into the detail hash (replace, so it's view
 * state within the screen, not a history step). No-op off a detail screen. */
export function setDataView(dv: DvParams): void {
  const s = get(screen);
  if (s.kind !== 'detail') return;
  navigate({ ...s, dv }, true);
}

// ---- tree cursor movement (mirrors kernel::TreeState nav) ----

function rowIndex(): number {
  const rows = get(visibleRows);
  const id = get(selectedId);
  const i = rows.findIndex((r) => r.id === id);
  return i < 0 ? 0 : i;
}

function selectAt(i: number): void {
  const rows = get(visibleRows);
  if (!rows.length) return;
  const clamped = Math.max(0, Math.min(rows.length - 1, i));
  selectedId.set(rows[clamped].id);
}

export function moveSelection(delta: number): void {
  selectAt(rowIndex() + delta);
}

/** ← : jump to the parent group (nearest preceding shallower row). */
export function selectParent(): void {
  const rows = get(visibleRows);
  const i = rowIndex();
  const depth = rows[i]?.depth ?? 0;
  if (depth === 0) return;
  for (let k = i - 1; k >= 0; k--) {
    if (rows[k].depth < depth) {
      selectedId.set(rows[k].id);
      return;
    }
  }
}

/** → : enter the selected group (expand if needed) and move to its first child. */
export function enterChild(): void {
  const rows = get(visibleRows);
  const i = rowIndex();
  const row = rows[i];
  if (!row || !row.hasChildren) return;
  const exp = get(expanded);
  if (!exp.has(row.id)) {
    toggle(row.id);
    // first child is the next row once re-flattened
    const next = get(visibleRows)[i + 1];
    if (next && next.depth === row.depth + 1) selectedId.set(next.id);
  } else {
    const next = rows[i + 1];
    if (next && next.depth === row.depth + 1) selectedId.set(next.id);
  }
}

/** Shift+↑/↓ : previous/next sibling (same depth, without leaving the parent). */
export function selectSibling(forwardDir: boolean): void {
  const rows = get(visibleRows);
  const i = rowIndex();
  const depth = rows[i]?.depth ?? 0;
  const range = forwardDir
    ? Array.from({ length: rows.length - i - 1 }, (_, k) => i + 1 + k)
    : Array.from({ length: i }, (_, k) => i - 1 - k);
  for (const k of range) {
    if (rows[k].depth < depth) break;
    if (rows[k].depth === depth) {
      selectedId.set(rows[k].id);
      break;
    }
  }
}

/** Enter : expand/collapse a group, or open a tensor's detail. */
export function activateSelection(): void {
  const rows = get(visibleRows);
  const row = rows[rowIndex()];
  if (!row) return;
  if (row.node.kind === 'tensor') openDetail(row.node.info.name);
  else if (row.hasChildren) toggle(row.id);
}

export function toggle(id: string): void {
  expanded.update((set) => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });
}

export function setAllExpanded(on: boolean): void {
  if (!on) {
    expanded.set(new Set());
    return;
  }
  const t = get(treeData);
  if (!t) return;
  const ids = new Set<string>();
  const walk = (nodes: typeof t.tree, parentId: string) => {
    for (const n of nodes) {
      if (n.kind === 'group') {
        const id = nodeId(n, parentId);
        ids.add(id);
        walk(n.children, id);
      }
    }
  };
  walk(t.tree, '');
  expanded.set(ids);
}

export function startSearch(): void {
  searching.set(true);
}
export function exitSearch(): void {
  searching.set(false);
  search.set('');
}
