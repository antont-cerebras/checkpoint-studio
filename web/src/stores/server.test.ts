// The fetched-data stores. No DOM needed: this module only talks to `fetch`, which is
// stubbed per test, so each one gets a fresh module (module-level caches are the thing
// under test) via `vi.resetModules()` + a dynamic import.
//
// The memo semantics are the reason this file exists. Caching an in-flight promise is
// what makes re-selecting a tensor instant; caching a *rejected* one makes the error
// sticky, so every later view — and every "retry" button — replays the original failure
// instead of asking the server again. That distinction is invisible until a user hits a
// transient 500 and the pane stays broken until reload.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';
import type { TreeNode, TreeResponse } from '../lib/types';

/** A tensor leaf, with the shard path the `<Ref>` resolver reads. */
const leaf = (name: string, shard: string, dtype = 'F16'): TreeNode => ({
  kind: 'tensor',
  label: null,
  info: {
    name,
    dtype,
    shape: [2],
    size_bytes: 4,
    num_elements: 2,
    storage: null,
    source_path: shard,
    layout: null,
  },
});
const group = (name: string, children: TreeNode[]): TreeNode => ({
  kind: 'group',
  name,
  children,
  expanded: false,
  tensor_count: children.length,
  params: 0,
  total_size: 0,
  stored_size: 0,
});

const TREE: TreeResponse = {
  tree: [
    group('model', [
      leaf('model.embed_tokens.weight', '/ckpt/model-00001-of-00002.safetensors'),
      group('layers', [leaf('model.layers.0.w', '/ckpt/model-00002-of-00002.safetensors')]),
    ]),
    leaf('lm_head.weight', '/ckpt/model-00002-of-00002.safetensors'),
  ],
} as TreeResponse;

/** Replies queued per URL prefix; each entry is used once, then the last repeats. */
function stubFetch(replies: { status?: number; body?: unknown }[]) {
  const calls: string[] = [];
  let i = 0;
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string) => {
      calls.push(url);
      const r = replies[Math.min(i++, replies.length - 1)] ?? { body: {} };
      const status = r.status ?? 200;
      const text = JSON.stringify(r.body ?? null);
      const bytes = new TextEncoder().encode(text);
      return Promise.resolve({
        ok: status >= 200 && status < 300,
        status,
        // `headers` and `body` because `api.tree` streams and measures — see the
        // `response` helper in lib/api.test.ts for the same shape.
        headers: new Headers({ 'Content-Length': String(bytes.length) }),
        json: () => Promise.resolve(r.body),
        text: () => Promise.resolve(text),
        body: {
          getReader() {
            let sent = false;
            return {
              read: () =>
                Promise.resolve(
                  sent
                    ? { done: true, value: undefined }
                    : ((sent = true), { done: false, value: bytes }),
                ),
            };
          },
        },
      });
    }),
  );
  return calls;
}

async function load() {
  vi.resetModules();
  return import('./server');
}

beforeEach(() => {
  vi.resetModules();
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ensureTree', () => {
  it('fetches the tree once, however many callers ask', async () => {
    const calls = stubFetch([{ body: TREE }]);
    const s = await load();
    await s.ensureTree();
    await s.ensureTree();
    await s.ensureTree();
    expect(calls.filter((u) => u === '/api/tree')).toHaveLength(1);
    expect(get(s.tree)).toEqual(TREE);
    expect(get(s.treeError)).toBeNull();
  });

  it('surfaces a failure in treeError instead of throwing at the caller', async () => {
    stubFetch([{ status: 500, body: { error: 'the reader exploded' } }]);
    const s = await load();
    await expect(s.ensureTree()).resolves.toBeUndefined();
    expect(get(s.treeError)).toBe('the reader exploded');
    expect(get(s.tree)).toBeNull();
  });

  it('warms the checkpoint stats in the background', async () => {
    // Opening the Stats screen should not show a spinner on first visit.
    const calls = stubFetch([{ body: TREE }, { body: { n_tensors: 3 } }]);
    const s = await load();
    await s.ensureTree();
    expect(calls).toContain('/api/stats');
  });
});

describe('name indexes', () => {
  it('collects every tensor name, at any depth', async () => {
    const s = await load();
    s.tree.set(TREE);
    expect([...get(s.tensorNames)].sort()).toEqual([
      'lm_head.weight',
      'model.embed_tokens.weight',
      'model.layers.0.w',
    ]);
  });

  it('collects shard basenames, not their full paths', async () => {
    // `<Ref>` matches the names the UI prints, which are basenames.
    const s = await load();
    s.tree.set(TREE);
    expect([...get(s.shardNames)].sort()).toEqual([
      'model-00001-of-00002.safetensors',
      'model-00002-of-00002.safetensors',
    ]);
  });

  it('is empty before the tree arrives', async () => {
    const s = await load();
    expect(get(s.tensorNames).size).toBe(0);
    expect(get(s.shardNames).size).toBe(0);
  });

  // The filter builder's dtype chips and the palette's per-dtype commands both read this.
  // It lives here rather than in each component because they had a copy of the walk each,
  // and a 31k-tensor tree should be traversed once per load, not once per asker.
  it('lists each dtype present once, sorted, at any depth', async () => {
    const s = await load();
    s.tree.set({
      tree: [
        group('model', [
          leaf('a', '/s.safetensors', 'F32'),
          group('layers', [
            leaf('b', '/s.safetensors', 'BF16'),
            // A repeat of an outer dtype: listed once, not twice.
            leaf('c', '/s.safetensors'),
          ]),
        ]),
        leaf('d', '/s.safetensors'),
      ],
    } as TreeResponse);
    expect(get(s.dtypesPresent)).toEqual(['BF16', 'F16', 'F32']);
  });

  it('has no dtypes before the tree arrives', async () => {
    const s = await load();
    expect(get(s.dtypesPresent)).toEqual([]);
  });
});

describe('per-tensor memo caches', () => {
  it('serves a repeat request from cache without a second fetch', async () => {
    const calls = stubFetch([{ body: { min: 0, max: 1 } }]);
    const s = await load();
    await s.cachedStats('a.weight');
    await s.cachedStats('a.weight');
    expect(calls).toHaveLength(1);
  });

  it('keys on the dtype override, so a reinterpreted view is its own entry', async () => {
    const calls = stubFetch([{ body: {} }]);
    const s = await load();
    await s.cachedStats('a.weight');
    await s.cachedStats('a.weight', 'F32');
    expect(calls).toHaveLength(2);
    expect(calls[1]).toContain('dtype=F32');
  });

  it('keys a sample on its whole parameter set', async () => {
    const calls = stubFetch([{ body: {} }]);
    const s = await load();
    await s.cachedSample('a.weight', { mode: 'grid', rows: 8, cols: 8 });
    await s.cachedSample('a.weight', { mode: 'grid', rows: 8, cols: 8 }); // cached
    await s.cachedSample('a.weight', { mode: 'window', rows: 8, cols: 8 }); // different
    expect(calls).toHaveLength(2);
  });

  it('caches histograms per bin count', async () => {
    const calls = stubFetch([{ body: {} }]);
    const s = await load();
    await s.cachedHistogram('a.weight', 64);
    await s.cachedHistogram('a.weight', 64);
    await s.cachedHistogram('a.weight', 128);
    expect(calls).toHaveLength(2);
  });

  // The point of the whole memo: a failure must not become permanent.
  it('evicts a rejection, so a retry actually retries', async () => {
    const calls = stubFetch([
      { status: 500, body: { error: 'transient' } },
      { body: { min: -1, max: 1 } },
    ]);
    const s = await load();
    await expect(s.cachedStats('a.weight')).rejects.toThrow('transient');
    // The retry must hit the server again and succeed — not replay the failure.
    await expect(s.cachedStats('a.weight')).resolves.toEqual({ min: -1, max: 1 });
    expect(calls).toHaveLength(2);
    // …and the success is then cached.
    await s.cachedStats('a.weight');
    expect(calls).toHaveLength(2);
  });

  it('evicts rejections for samples and histograms too', async () => {
    const calls = stubFetch([{ status: 500, body: { error: 'nope' } }, { body: {} }]);
    const s = await load();
    await expect(s.cachedSample('a.weight', { mode: 'grid' })).rejects.toThrow('nope');
    await expect(s.cachedSample('a.weight', { mode: 'grid' })).resolves.toEqual({});
    expect(calls).toHaveLength(2);

    const more = stubFetch([{ status: 500, body: { error: 'nope' } }, { body: {} }]);
    const s2 = await load();
    await expect(s2.cachedHistogram('a.weight')).rejects.toThrow('nope');
    await expect(s2.cachedHistogram('a.weight')).resolves.toEqual({});
    expect(more).toHaveLength(2);
  });

  it('shares one in-flight request between concurrent callers', async () => {
    const calls = stubFetch([{ body: { min: 0, max: 1 } }]);
    const s = await load();
    const [a, b] = await Promise.all([s.cachedStats('a.weight'), s.cachedStats('a.weight')]);
    expect(a).toBe(b); // the same promise, so the same object
    expect(calls).toHaveLength(1);
  });
});

describe('cachedCheckpointStats', () => {
  it('fetches once and reuses the result', async () => {
    const calls = stubFetch([{ body: { n_tensors: 7 } }]);
    const s = await load();
    await expect(s.cachedCheckpointStats()).resolves.toEqual({ n_tensors: 7 });
    await s.cachedCheckpointStats();
    expect(calls.filter((u) => u === '/api/stats')).toHaveLength(1);
  });
});
