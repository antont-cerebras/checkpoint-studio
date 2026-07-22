// Fetched server DATA: loaded once and cached. The tree backs the main screen;
// per-tensor data-view results are memoized so re-selecting doesn't refetch.

import { writable } from 'svelte/store';
import { api } from '../lib/api';
import type { HistogramDto, SampleDto, StatsDto, TreeResponse } from '../lib/types';

export const tree = writable<TreeResponse | null>(null);
export const treeError = writable<string | null>(null);

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

export function cachedStats(name: string, dtype?: string): Promise<StatsDto> {
  const key = `${name}|${dtype ?? ''}`;
  let p = statsCache.get(key);
  if (!p) {
    p = api.tensorStats(name, dtype);
    statsCache.set(key, p);
  }
  return p;
}

export function cachedSample(name: string, params: Parameters<typeof api.sample>[1]): Promise<SampleDto> {
  const key = `${name}|${JSON.stringify(params)}`;
  let p = sampleCache.get(key);
  if (!p) {
    p = api.sample(name, params);
    sampleCache.set(key, p);
  }
  return p;
}

export function cachedHistogram(name: string, bins?: number, dtype?: string): Promise<HistogramDto> {
  const key = `${name}|${bins ?? ''}|${dtype ?? ''}`;
  let p = histCache.get(key);
  if (!p) {
    p = api.histogram(name, bins, dtype);
    histCache.set(key, p);
  }
  return p;
}
