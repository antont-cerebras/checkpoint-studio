// The URL hash IS the view state: which screen, which tensor, which data-view
// params, plus the screen-independent bits (filter, sort, compact toggle, search).
//
// These are pure string ↔ state functions, deliberately free of `location` and of
// any store, because the round-trip is a contract two features depend on: a shared
// link must reopen the exact view, and the TUI's `y` command prints a CLI
// invocation for the same state. `stores/view.ts` owns the stores and the
// `location.hash` / `history` plumbing and calls in here.

/** Which pane of the tensor detail is showing. */
export type DataTab = 'info' | 'heatmap' | 'values' | 'histogram';
const DATA_TABS: readonly string[] = ['info', 'heatmap', 'values', 'histogram'];

/** Data-view (heatmap / numeric grid) params carried in the detail hash so a
 * specific view — mode, sample size, dtype override, window offsets, base/zebra,
 * ratio lock — is reproducible from a shared/bookmarked URL and across back/forward.
 * Raw strings; `DataView` owns the semantics + defaults. */
export const DV_KEYS = [
  'mode',
  'rows',
  'cols',
  'dtype',
  'roff',
  'coff',
  'slice',
  'base',
  'zebra',
  'lock',
] as const;
export type DvParams = Partial<Record<(typeof DV_KEYS)[number], string>>;

// `?: T | undefined` (rather than plain `?: T`) because `exactOptionalPropertyTypes`
// is on and these are set explicitly to `undefined` in places — "absent" and
// "present but cleared" mean the same thing for a URL parameter.
export type Screen =
  | { kind: 'tree' }
  | { kind: 'detail'; tensor: string; tab: DataTab; dv?: DvParams | undefined }
  | { kind: 'files' }
  | { kind: 'layout'; file?: string | undefined }
  | { kind: 'stats' }
  | { kind: 'health' }
  | { kind: 'preview'; path: string; name: string };

/** Sorting for the flat (filter / search) tensor list. `none` keeps the natural
 * order (fuzzy-score for search, tree order for a filter); the tree view is never
 * reordered. */
export type SortKey = 'none' | 'name' | 'size' | 'params' | 'dtype' | 'rank';
const SORT_KEYS: readonly string[] = ['name', 'size', 'params', 'dtype', 'rank'];

/** The screen-independent view state every hash carries, so any state is
 * reproducible from the URL regardless of which screen is open. */
export interface Globals {
  filter: string;
  sortKey: SortKey;
  sortDir: 'asc' | 'desc';
  compact: boolean;
  /** Search MODE — an empty query with the box open is a distinct state from no
   * search at all, so it round-trips as a present-but-empty `q`. */
  searching: boolean;
  search: string;
}

/** Strip the leading `#` and split a hash into its kind and query string. */
function splitHash(hash: string): [string, string] {
  const raw = hash.replace(/^#/, '');
  const i = raw.indexOf('?');
  return i < 0 ? [raw, ''] : [raw.slice(0, i), raw.slice(i + 1)];
}

/** A screen's own part of the hash (no global state). */
export function screenToHash(s: Screen): string {
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

/** The screen a hash names — falling back to the tree for anything unrecognised or
 * missing its required parameter, so a hand-edited link can't strand the app. */
export function parseScreen(hash: string): Screen {
  const [kind, queryStr] = splitHash(hash);
  const q = new URLSearchParams(queryStr);
  switch (kind) {
    case 'detail': {
      const t = q.get('t');
      const raw = q.get('tab') ?? 'info';
      const tab = (DATA_TABS.includes(raw) ? raw : 'info') as DataTab;
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

/** The global state as query parameters — only what differs from the defaults, so a
 * plain view stays a plain URL. */
export function globalQuery(g: Globals): string {
  const p = new URLSearchParams();
  const f = g.filter.trim();
  if (f) p.set('filter', f);
  if (g.sortKey !== 'none') p.set('sort', `${g.sortKey}.${g.sortDir}`);
  if (g.compact) p.set('compact', '1');
  if (g.searching) p.set('q', g.search); // presence ⇒ search mode (empty ok)
  return p.toString();
}

/** The global state a hash carries, with every field defaulted. */
export function parseGlobals(hash: string): Globals {
  const q = new URLSearchParams(splitHash(hash)[1]);
  const sort = q.get('sort');
  const [k, d] = sort ? sort.split('.') : [];
  const qs = q.get('q');
  return {
    filter: q.get('filter') ?? '',
    sortKey: (SORT_KEYS.includes(k ?? '') ? k : 'none') as SortKey,
    sortDir: d === 'desc' ? 'desc' : 'asc',
    compact: q.get('compact') === '1',
    searching: qs !== null,
    search: qs ?? '',
  };
}

/** A screen's hash plus the global state — the complete, shareable view. */
export function hashFor(s: Screen, g: Globals): string {
  const base = screenToHash(s);
  const q = globalQuery(g);
  if (!q) return base;
  return base.includes('?') ? `${base}&${q}` : `${base}?${q}`;
}
