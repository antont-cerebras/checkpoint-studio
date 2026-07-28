// Fetched server DATA: loaded once and cached. The tree backs the main screen;
// per-tensor data-view results are memoized so re-selecting doesn't refetch.

import { derived, writable } from 'svelte/store';
import { api } from '../lib/api';
import type {
  CompactTree,
  HistogramDto,
  SampleDto,
  StatsDto,
  TreeNode,
  TreeResponse,
} from '../lib/types';

export const tree = writable<TreeResponse | null>(null);

/// The compact (family-folded) tree for the current filter. A *store* rather than
/// component state so the shared fold actions (`e` / `c`, the palette's expand/collapse
/// all) can walk whichever tree is on screen — they used to walk only the full tree, so
/// they silently did nothing while the compact view was showing.
export const compactTree = writable<CompactTree | null>(null);
export const compactError = writable<string>('');

let compactSeq = 0;
/** Fetch the compact tree for `q`, ignoring a response a later call superseded. Returns
 * the tree it stored — so a caller can seed fold state from exactly the trees that
 * actually landed, without keeping its own "have I seeded this one?" flag. */
export async function loadCompact(q: string): Promise<CompactTree | null> {
  const s = ++compactSeq;
  compactError.set('');
  try {
    const r = await api.compact(q);
    if (s !== compactSeq) return null; // superseded
    compactTree.set(r);
    return r;
  } catch (e) {
    if (s !== compactSeq) return null;
    compactTree.set(null);
    compactError.set(e instanceof Error ? e.message : String(e));
    return null;
  }
}
export const treeError = writable<string | null>(null);

/** What the source supports, as the server derived it — the one question a feature asks
 * before offering itself. Null until the tree lands. */
export const caps = derived(tree, ($t) => $t?.capabilities ?? null);

/** Why the data views are unavailable, or null when they are. The server's wording. */
export const dataViewNote = derived(tree, ($t) => $t?.data_view_note ?? null);

/** Tensors on disk but not listed in the index, by `source_path` — a Set, because the
 * tree tests it once per visible row. Marked with the same glyph and vivid red the
 * terminal uses (see `palette::UNINDEXED`); a loader following only the index will not
 * read these. */
export const unindexed = derived(tree, ($t) => new Set($t?.unindexed ?? []));

/** Every tensor name, and every shard basename, in the checkpoint. These back the
 * universal `<Ref>` link resolver: a name found here is turned into a link to the
 * tensor-detail or byte-layout screen; anything else stays plain text. */
export const tensorNames = derived(tree, ($t) => collectNames($t, 'tensor'));
export const shardNames = derived(tree, ($t) => collectNames($t, 'shard'));

/** Every dtype the checkpoint actually contains, sorted — the choices the filter builder
 * offers and the per-dtype commands the palette lists. Derived here rather than walked in
 * each component: both had their own copy of the walk, and this way a 31k-tensor tree is
 * traversed once per load instead of once per component that asks. */
export const dtypesPresent = derived(tree, ($t) => {
  const set = new Set<string>();
  if (!$t) return [] as string[];
  const walk = (ns: TreeNode[]) => {
    for (const n of ns) {
      if (n.kind === 'tensor') set.add(n.info.dtype);
      else if (n.kind === 'group') walk(n.children);
    }
  };
  walk($t.tree);
  return [...set].sort();
});

function collectNames(t: TreeResponse | null, want: 'tensor' | 'shard'): Set<string> {
  const set = new Set<string>();
  if (!t) return set;
  const walk = (ns: TreeNode[]) => {
    for (const n of ns) {
      if (n.kind === 'tensor') {
        if (want === 'tensor') set.add(n.info.name);
        else {
          const p = n.info.source_path;
          set.add(p.split('/').pop() || p);
        }
      } else if (n.kind === 'group') {
        walk(n.children);
      }
    }
  };
  walk(t.tree);
  return set;
}

let treeStarted = false;
export async function ensureTree(): Promise<void> {
  if (treeStarted) return;
  treeStarted = true;
  try {
    tree.set(await api.tree());
  } catch (e) {
    treeError.set(e instanceof Error ? e.message : String(e));
  }
  // Warm the whole-checkpoint stats in the background (~8 KB, precomputed
  // server-side) so opening the Stats screen is instant rather than showing a
  // spinner on first visit. Fire-and-forget: failures surface when StatsView awaits.
  void cachedCheckpointStats().catch(() => {});
}

// Whole-checkpoint stats (`/api/stats`): fetched once and reused, so the Stats
// screen renders immediately on every visit after the first.
let checkpointStats: Promise<Record<string, unknown>> | null = null;
export function cachedCheckpointStats(): Promise<Record<string, unknown>> {
  if (!checkpointStats) checkpointStats = api.stats();
  return checkpointStats;
}

// --- per-tensor data-view memo caches (keyed by request) ---

const statsCache = new Map<string, Promise<StatsDto>>();
const sampleCache = new Map<string, Promise<SampleDto>>();
const histCache = new Map<string, Promise<HistogramDto>>();

/** Memoize an in-flight/settled request, but **evict a rejection** — caching a failed
 * promise makes the error sticky: every later view of that tensor (and every "retry")
 * replays the original failure instead of asking the server again. Only successes are
 * worth keeping. */
function memo<T>(cache: Map<string, Promise<T>>, key: string, start: () => Promise<T>): Promise<T> {
  let p = cache.get(key);
  if (!p) {
    p = start().catch((e: unknown) => {
      cache.delete(key); // a transient 500 must not poison this key forever
      throw e;
    });
    cache.set(key, p);
  }
  return p;
}

export function cachedStats(name: string, dtype?: string): Promise<StatsDto> {
  return memo(statsCache, `${name}|${dtype ?? ''}`, () => api.tensorStats(name, dtype));
}

export function cachedSample(name: string, params: Parameters<typeof api.sample>[1]): Promise<SampleDto> {
  return memo(sampleCache, `${name}|${JSON.stringify(params)}`, () => api.sample(name, params));
}

export function cachedHistogram(name: string, bins?: number, dtype?: string): Promise<HistogramDto> {
  return memo(histCache, `${name}|${bins ?? ''}|${dtype ?? ''}`, () =>
    api.histogram(name, bins, dtype),
  );
}
