// Fetched server DATA: loaded once and cached. The tree backs the main screen;
// per-tensor data-view results are memoized so re-selecting doesn't refetch.

import { derived, writable } from 'svelte/store';
import { api } from '../lib/api';
import type { HistogramDto, SampleDto, StatsDto, TreeNode, TreeResponse } from '../lib/types';

export const tree = writable<TreeResponse | null>(null);
export const treeError = writable<string | null>(null);

/** Every tensor name, and every shard basename, in the checkpoint. These back the
 * universal `<Ref>` link resolver: a name found here is turned into a link to the
 * tensor-detail or byte-layout screen; anything else stays plain text. */
export const tensorNames = derived(tree, ($t) => collectNames($t, 'tensor'));
export const shardNames = derived(tree, ($t) => collectNames($t, 'shard'));

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
