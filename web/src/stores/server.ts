// Fetched server DATA: loaded once and cached. The tree backs the main screen;
// per-tensor data-view results are memoized so re-selecting doesn't refetch.

import { derived, writable } from 'svelte/store';
import { api } from '../lib/api';
import { startedNow, type Progress } from '../lib/progress';
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

/** How long the current fold has been running; null when none is. Timer only — the response is
 * ~7 KB, so a byte bar would be theatre (see the fold's own measurements). */
export const compactProgress = writable<Progress | null>(null);

let compactSeq = 0;
/** Fetch the compact tree for `q`, ignoring a response a later call superseded. Returns
 * the tree it stored — so a caller can seed fold state from exactly the trees that
 * actually landed, without keeping its own "have I seeded this one?" flag. */
export async function loadCompact(q: string): Promise<CompactTree | null> {
  const s = ++compactSeq;
  compactError.set('');
  compactProgress.set(startedNow());
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
  } finally {
    // Only the newest fetch owns the indicator; a superseded one clearing it would hide the
    // wait that is still running.
    if (s === compactSeq) compactProgress.set(null);
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

/** How far the tensor tree's download has got, while it is in flight; null before it
 * starts and after it lands. Drives the loading bar — the terminal shows a gauge and an
 * elapsed timer for the same wait, and this is the browser's half of that. */
export const treeProgress = writable<Progress | null>(null);

let treeStarted = false;
export async function ensureTree(): Promise<void> {
  if (treeStarted) return;
  treeStarted = true;
  const startedAt = performance.now();
  treeProgress.set({ received: 0, total: null, startedAt });
  try {
    tree.set(
      await api.tree((received, total) => treeProgress.set({ received, total, startedAt })),
    );
  } catch (e) {
    treeError.set(e instanceof Error ? e.message : String(e));
  } finally {
    treeProgress.set(null);
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

// --- changing which checkpoint is being served ---

/** The checkpoints opened this run, most recent first — the open prompt's list. */
export const recents = writable<string[]>([]);
/** Whether the server reads over an ssh proxy, which decides what paths it can open. */
export const proxied = writable<boolean>(false);
/** How long the current open has been running; null when none is. Timer only — the server
 * reads shard headers and *then* answers, so there is no fraction to show (the same rule the
 * scan and histogram waits follow). */
export const openProgress = writable<Progress | null>(null);
/** What the in-flight open is reading — the loading screen names it, since during an open the
 * tree still on screen belongs to the checkpoint being replaced. */
export const openingSpec = writable<string>('');

export async function loadRecents(): Promise<void> {
  try {
    const r = await api.recents();
    recents.set(r.recents);
    proxied.set(r.proxied);
  } catch {
    // A recents list is a convenience; failing to fetch it must not stop the prompt from
    // accepting a typed path.
  }
}

/**
 * Forget every cached answer about the checkpoint we were on.
 *
 * The mirror of the server's swap: over there, every per-checkpoint cache lives inside the
 * state object, so replacing it discards them. Here they are module-level, so they have to be
 * cleared by hand — and *all* of them, because tensor names, shard paths and even dtypes can
 * repeat across two checkpoints. A surviving entry would not look stale; it would look like
 * an answer.
 */
function forgetCheckpoint(): void {
  treeStarted = false;
  tree.set(null);
  treeError.set(null);
  treeProgress.set(null);
  compactTree.set(null);
  compactError.set('');
  // Supersede any compact fetch still in flight, so its response can't land on the new
  // checkpoint's screen.
  compactSeq++;
  checkpointStats = null;
  statsCache.clear();
  sampleCache.clear();
  histCache.clear();
}

/**
 * Ask the server to serve `spec`, and return its new root.
 *
 * Throws with the server's own message when the path doesn't resolve — and in that case
 * nothing here has been discarded, because the server only swaps after a successful read.
 * A failed open leaves you exactly where you were.
 *
 * Deliberately does **not** refetch: the caller has view state to reset first, and the order
 * matters (see [`reloadCheckpoint`]).
 */
export async function openCheckpoint(spec: string): Promise<string> {
  openingSpec.set(spec);
  openProgress.set(startedNow());
  try {
    const r = await api.open(spec);
    recents.set(r.recents);
    return r.root;
  } finally {
    openProgress.set(null);
  }
}

/**
 * Drop every cached answer and fetch the new checkpoint.
 *
 * Call this **after** resetting view state, not before. The fold state is seeded by a
 * subscription that fires when a tree lands, and it only seeds once per checkpoint — so a tree
 * that arrives before the seed flag is cleared gets no initial expansion, and the new
 * checkpoint opens fully collapsed while the first one opened expanded. That is the bug this
 * split exists to prevent.
 */
export async function reloadCheckpoint(): Promise<void> {
  forgetCheckpoint();
  await ensureTree();
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
