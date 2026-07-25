// The client-owned VIEW STATE, mirroring the TUI's full-screen modal flow: one
// active screen at a time, a browser-style history stack (Backspace / \), and the
// persistent tree fold/selection/search state. Navigation + tree-cursor helpers
// live here so the global key handler (App.svelte) and the views share them.
//
// This module is the wiring: the stores, the `location.hash` / `history` plumbing and
// the debounced filter request. The logic it wires up is pure and lives next door —
// `lib/hash.ts` (URL ↔ view state) and `lib/rows.ts` (row shaping + cursor rules) —
// so both can be tested without a DOM.

import { derived, get, writable } from 'svelte/store';
import { api } from '../lib/api';
import { flatten, nodeId, type Row } from '../lib/flatten';
import {
  DV_KEYS,
  hashFor,
  parseGlobals,
  parseScreen,
  type DataTab,
  type DvParams,
  type Globals,
  type Screen,
  type SortKey,
} from '../lib/hash';
import {
  clampIndex,
  firstChildIndex,
  matchRows,
  parentIndex,
  rowIndexOf,
  siblingIndex,
  sortRows,
} from '../lib/rows';
import { SEARCH_LIMIT, searchTree } from '../lib/search';
import { tree as treeData } from './server';

export { DV_KEYS };
export type { DataTab, DvParams, Screen, SortKey };

// The current screen is driven by the URL hash, so the browser's back/forward
// buttons work natively and every screen+mode has a shareable link.
export const screen = writable<Screen>(parseScreen(location.hash));

// Keep the stores in sync with the URL (covers browser back/forward + shared links).
window.addEventListener('hashchange', () => {
  screen.set(parseScreen(location.hash));
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
  // `void`: the request handles its own errors, and a timer callback has nowhere to
  // return a promise to.
  filterTimer = setTimeout(() => void resolveFilter(query, req), 200);
});

/** Ask the server which tensors pass `query` and publish the result — unless a newer
 * edit has already superseded this request. */
async function resolveFilter(query: string, req: number): Promise<void> {
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
}

/** Command palette (Space / `:`) open state. */
export const paletteOpen = writable<boolean>(false);

/** Sort facet and direction for the flat (filter / search) list; `none` keeps the
 * natural order and the hierarchical tree is never reordered. See `lib/hash.ts`. */
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
/** The fuzzy-search result — rows AND the untruncated total — computed ONCE per
 * keystroke. `visibleRows` and `searchTotal` both read from this, because computing
 * them independently meant two full walks of the tree scoring every tensor twice. */
const searchResult = derived([treeData, search, searching], ([$t, $q, $searching]) =>
  $t && $searching && $q.trim() ? searchTree($t.tree, $q.trim()) : null,
);

export const visibleRows = derived(
  [treeData, expanded, searchResult, filterMatches, sortKey, sortDir],
  ([$t, $exp, $found, $matches, $sk, $sd]) => {
    if (!$t) return [] as Row[];
    // An in-progress fuzzy search wins; otherwise a set filter; otherwise the tree.
    let rows: Row[];
    if ($found) rows = $found.rows;
    else if ($matches) rows = matchRows($t.tree, $matches);
    else return flatten($t.tree, $exp); // hierarchical view is never reordered
    return $sk === 'none' ? rows : sortRows(rows, $sk, $sd);
  },
);

/** Total tensors matching the current fuzzy search, untruncated — so the search
 * label can be honest when the row list is capped ("showing 1000 of N"). Only
 * the fuzzy path is capped; the server-side filter returns every match. */
export { SEARCH_LIMIT };
export const searchTotal = derived(searchResult, ($found) => $found?.total ?? 0);

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

/** The screen-independent view state, read off the stores for `globalQuery`. */
function globals(): Globals {
  return {
    filter: get(filterQuery),
    sortKey: get(sortKey),
    sortDir: get(sortDir),
    compact: get(compact),
    searching: get(searching),
    search: get(search),
  };
}

let restoring = false;

/** Restore the global stores from the current hash (initial load + back/forward). */
function restoreGlobals(): void {
  restoring = true;
  const g = parseGlobals(location.hash);
  filterQuery.set(g.filter);
  sortKey.set(g.sortKey);
  sortDir.set(g.sortDir);
  compact.set(g.compact);
  searching.set(g.searching);
  search.set(g.search);
  restoring = false;
}

/** Mirror the current screen + global state into the hash without a new history
 * entry (replaceState doesn't fire hashchange, so this can't loop). */
function syncHash(): void {
  if (restoring) return;
  const h = `#${hashFor(get(screen), globals())}`;
  if (location.hash !== h) history.replaceState(history.state, '', h);
}

export function navigate(s: Screen, replace = false): void {
  const h = `#${hashFor(s, globals())}`;
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
  return rowIndexOf(get(visibleRows), get(selectedId));
}

function selectAt(i: number | null): void {
  if (i === null) return;
  const rows = get(visibleRows);
  const row = rows[clampIndex(rows, i)];
  if (row) selectedId.set(row.id);
}

export function moveSelection(delta: number): void {
  selectAt(rowIndex() + delta);
}

/** ← : jump to the parent group (nearest preceding shallower row). */
export function selectParent(): void {
  selectAt(parentIndex(get(visibleRows), rowIndex()));
}

/** → : enter the selected group (expand if needed) and move to its first child. */
export function enterChild(): void {
  const i = rowIndex();
  const row = get(visibleRows)[i];
  if (!row || !row.hasChildren) return;
  // Expanding re-flattens the list, so ask for the child index afterwards.
  if (!get(expanded).has(row.id)) toggle(row.id);
  selectAt(firstChildIndex(get(visibleRows), i));
}

/** Shift+↑/↓ : previous/next sibling (same depth, without leaving the parent). */
export function selectSibling(forwardDir: boolean): void {
  selectAt(siblingIndex(get(visibleRows), rowIndex(), forwardDir));
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
