// The URL hash IS the view state: which screen, which tensor, which data-view
// params, plus the screen-independent bits (filter, sort, compact toggle, search).
//
// These are pure string ↔ state functions, deliberately free of `location` and of
// any store, because the round-trip is a contract two features depend on: a shared
// link must reopen the exact view, and the TUI's `y` command prints a CLI
// invocation for the same state. `stores/view.ts` owns the stores and the
// `location.hash` / `history` plumbing and calls in here.

import {
  isScopeActive,
  scopeFromQuery,
  scopeToQuery,
  type DiffScopeParams,
} from './diffscope';
import { packingFromQuery, packingToQuery, type Packing } from './packing';

/**
 * The parsed scope, but only when the URL actually carries one.
 *
 * Omitted rather than always present: an unscoped screen is then the same value it has always been, so
 * `hash → parse → hash` is an identity for every existing link and nothing has to special-case an
 * "empty" scope object. Matches the field's own meaning — absent *is* unscoped.
 */
function scopeIfAny(q: URLSearchParams): { scope?: DiffScopeParams } {
  const scope = scopeFromQuery(q);
  return isScopeActive(scope) ? { scope } : {};
}

/** The packing, but only when the URL says one — absent *is* "infer it", as it has always been. */
function packingIfAny(q: URLSearchParams): { packing?: Packing } {
  const packing = packingFromQuery(q);
  return packing ? { packing } : {};
}

/**
 * The scope as a hash-query tail — `&name=…&dtype_is=…`, or nothing at all.
 *
 * Emitted only for what is set, so an unscoped comparison keeps the short URL it always had.
 */
function scopeQuery(s: DiffScopeParams | undefined): string {
  return s === undefined
    ? ''
    : scopeToQuery(s)
        .map(([k, v]) => `&${k}=${encodeURIComponent(v)}`)
        .join('');
}

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
  // `scope` is the CLI's selection flags (`--name`, `--dtype-is`, …). Optional because absent *means*
  // unscoped: the palette entry and the report's own link have no selection to carry, and requiring one
  // would put `scope: emptyScope()` at a dozen call sites that have nothing to do with scoping.
  // `swapped` turns the report round: the open checkpoint becomes the baseline. In the URL because it
  // changes what the report *says* — added and removed trade places — so a link has to carry it.
  // **One comparison, three ways of reading it.** There were two screens — a report and an aligned
  // tree — reached from two places, so the reader chose a representation before seeing the result and
  // switching meant starting again. The pair, the direction and the scope are the comparison; the
  // view is how it is being read, and it lives here with the rest of the view state.
  | {
      kind: 'compare';
      /** The baseline — the left-hand side of `diff OLD NEW`. */
      lhs: string;
      /** The candidate; empty means the checkpoint the server has open. */
      rhs: string;
      /** Which of the three views is showing; `summary` by default. */
      view?: CompareView | undefined;
      scope?: DiffScopeParams | undefined;
      /** `--full`: every layer as its own row, rather than uniform families folded onto one each.
       * In the URL for the reason every view control is: a link shows what the sender was reading. */
      full?: boolean | undefined;
      /** Which way round the pair is read. The *operands* stay canonical — the scope is directional,
       * so swapping them would describe a comparison the server would answer differently. */
      swapped?: boolean | undefined;
      /** Summary sections the reader has folded away, by key. In the URL so a reload — or a link —
       * lands on the report as it was being read, which is the point of folding away 31,247 rows. */
      closed?: string[] | undefined;
      /** How each side packs its expert indices, for a repack verification (`lib/packing`). In the URL
       * because it changes what the verification *decodes* — a link to a verified pair that dropped it
       * would answer the same question differently. */
      packing?: Packing | undefined;
    }
  | { kind: 'preview'; path: string; name: string }
  // The open prompt carries no state of its own: what it does is change the *server*, and a
  // URL cannot capture that. It round-trips as a bare `open` so a reload lands on the prompt
  // rather than on a blank screen — deliberately without the typed path, because a bookmark
  // that silently re-pointed the server on load would be a URL with a side effect.
  | { kind: 'open' };

/** Sorting for the flat (filter / search) tensor list. `none` keeps the natural
 * order (fuzzy-score for search, tree order for a filter); the tree view is never
 * reordered. */
/** The comparison screen's state, for the code that changes one thing about it. */
export type CompareScreen = Extract<Screen, { kind: 'compare' }>;

/** How a comparison is being read: the categorised summary, the aligned tree, or the data checks. */
export type CompareView = 'summary' | 'browse' | 'data';
const COMPARE_VIEWS: readonly string[] = ['summary', 'browse', 'data'];

export type SortKey = 'none' | 'name' | 'size' | 'params' | 'dtype' | 'rank';
const SORT_KEYS: readonly string[] = ['name', 'size', 'params', 'dtype', 'rank'];

/** The screen-independent view state every hash carries, so any state is
 * reproducible from the URL regardless of which screen is open. */
export interface Globals {
  /**
   * Which checkpoint the view is of.
   *
   * A URL that named a screen, a filter and a selection but not the *checkpoint* described a
   * view of whatever happened to be loaded — so a link, a bookmark or a restored tab could
   * land you on the same screen of a different checkpoint and look right. Carrying it makes
   * the URL the whole answer, the way the terminal's `y` command emits a complete invocation.
   *
   * Empty before the first tree lands (there is nothing to name yet), and omitted from the
   * hash then.
   */
  ckpt: string;
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
    case 'compare': {
      const rhs = s.rhs ? `&rhs=${enc(s.rhs)}` : '';
      // `summary` is the default, so an ordinary comparison keeps a short URL.
      const view = s.view && s.view !== 'summary' ? `&view=${s.view}` : '';
      const swap = s.swapped ? '&swap=1' : '';
      const full = s.full ? '&full=1' : '';
      const closed = s.closed?.length ? `&closed=${s.closed.map(enc).join(',')}` : '';
      const pack = packingToQuery(s.packing)
        .map(([k, v]) => `&${k}=${enc(v)}`)
        .join('');
      return `compare?lhs=${enc(s.lhs)}${rhs}${view}${swap}${full}${closed}${pack}${scopeQuery(s.scope)}`;
    }
    case 'preview':
      return `preview?path=${enc(s.path)}&name=${enc(s.name)}`;
    case 'open':
      return 'open';
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
    case 'compare': {
      const view = q.get('view') ?? '';
      // No baseline means no comparison to draw, but the screen is still where you pick one — so it
      // opens empty rather than falling through to the tree, which is what made the palette's entry
      // appear to do nothing at all.
      return {
        kind: 'compare',
        lhs: q.get('lhs') ?? '',
        rhs: q.get('rhs') ?? '',
        ...(COMPARE_VIEWS.includes(view) ? { view: view as CompareView } : {}),
        ...(q.get('swap') === '1' ? { swapped: true } : {}),
        ...(q.get('full') === '1' ? { full: true } : {}),
        ...(q.get('closed')
          ? { closed: (q.get('closed') ?? '').split(',').filter((k) => k !== '') }
          : {}),
        ...packingIfAny(q),
        ...scopeIfAny(q),
      };
    }
    case 'preview': {
      const path = q.get('path');
      if (path) return { kind: 'preview', path, name: q.get('name') ?? path };
      break;
    }
    case 'open':
      return { kind: 'open' };
  }
  return { kind: 'tree' };
}

/** The global state as query parameters — only what differs from the defaults, so a
 * plain view stays a plain URL. */
export function globalQuery(g: Globals): string {
  const p = new URLSearchParams();
  // First, so a shared link reads as "this checkpoint, this view" rather than burying which
  // checkpoint behind the view parameters.
  if (g.ckpt) p.set('ckpt', g.ckpt);
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
    ckpt: q.get('ckpt') ?? '',
    filter: q.get('filter') ?? '',
    sortKey: (SORT_KEYS.includes(k ?? '') ? k : 'none') as SortKey,
    sortDir: d === 'desc' ? 'desc' : 'asc',
    compact: q.get('compact') === '1',
    searching: qs !== null,
    search: qs ?? '',
  };
}

/**
 * Which screens the *list* state (filter, sort, family fold, search) belongs to.
 *
 * It describes the tensor list, so it is noise anywhere else: a comparison link carried `compact=1`
 * and a `filter=` that changed nothing on screen, in a URL people paste to each other. The detail
 * view keeps it because it is a view *of a row in that list* — Back returns to the list, and to the
 * list as it was.
 */
const LIST_SCREENS: readonly Screen['kind'][] = ['tree', 'detail'];

/** The globals a screen can actually use — the rest would describe a screen you are not on. */
function forScreen(g: Globals, s: Screen): Globals {
  const kept = LIST_SCREENS.includes(s.kind)
    ? g
    : {
        ...g,
        filter: '',
        sortKey: 'none' as const,
        compact: false,
        searching: false,
        search: '',
      };
  // **A comparison that names both of its checkpoints does not need to name a third.**
  //
  // `ckpt` says which checkpoint a view is *of*, and opening a link that names one the server is not
  // serving switches to it (`App`'s startup). On a comparison of two named checkpoints that is neither
  // true nor harmless: the parameter carried whatever the sender's server happened to hold —
  // `#compare?lhs=…12-boxes&rhs=s3://…&ckpt=/tmp/mapfix/new.safetensors`, a third checkpoint with no
  // part in the comparison, which the recipient's server would then be told to open.
  //
  // It stays when the candidate is *implicit*: an empty `rhs` means "the checkpoint that is open", so
  // then the link genuinely does depend on which one that is.
  const namesBoth = s.kind === 'compare' && s.lhs !== '' && s.rhs !== '';
  return namesBoth ? { ...kept, ckpt: '' } : kept;
}

/** A screen's hash plus the global state — the complete, shareable view. */
export function hashFor(s: Screen, g: Globals): string {
  const base = screenToHash(s);
  const q = globalQuery(forScreen(g, s));
  if (!q) return base;
  return base.includes('?') ? `${base}&${q}` : `${base}?${q}`;
}
