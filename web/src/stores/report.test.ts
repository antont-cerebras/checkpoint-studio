// The report store's cache and supersede logic.
//
// The same class of stateful shortcut as the comparison store's, and worth pinning for the same
// reason: a key that is too coarse suppresses a needed refetch (the reader changes the scope and the
// old report stays on screen), and one that is too fine re-fetches on every render. Both are quiet.

import { afterEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

import { emptyScope } from '../lib/diffscope';

/** What `GET /api/diff` answers, reduced to what the store touches. */
const answer = (id: number) => ({
  against: '/base',
  candidate: '/candidate',
  swapped: false,
  verdict: `report ${id}`,
  command: 'checkpoint-studio diff /base /candidate',
  matched: null,
  rename_collisions: [],
  report: {},
});

/** Stub `fetch`, recording every URL. */
function stubFetch(body: (url: string) => unknown = (u) => answer(Number(u.slice(-1)))) {
  const calls: string[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string) => {
      calls.push(url);
      const payload = body(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        headers: new Headers(),
        json: () => Promise.resolve(payload),
        text: () => Promise.resolve(JSON.stringify(payload)),
      });
    }),
  );
  return calls;
}

/** A fresh module, so one test's cache is not another's. */
async function load() {
  vi.resetModules();
  return import('./report');
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('loading a report', () => {
  it('asks by comparison id, and keeps the answer', async () => {
    const calls = stubFetch();
    const r = await load();
    await r.loadReport(7, undefined, false, false);
    expect(calls).toEqual(['/api/diff?id=7']);
    expect(get(r.diffReport)?.verdict).toBe('report 7');
    expect(get(r.reportWait)).toBeNull();
  });

  it('does not ask twice for the report already showing', async () => {
    const calls = stubFetch();
    const r = await load();
    await r.loadReport(7, undefined, false, false);
    await r.loadReport(7, undefined, false, false);
    expect(calls).toHaveLength(1);
  });

  // Each of these changes the answer, so each is part of the key: the selection, the direction, and
  // — because the offered command has to match the screen — the family fold.
  it('asks again when the selection, the direction or the fold changes', async () => {
    const calls = stubFetch();
    const r = await load();
    await r.loadReport(7, undefined, false, false);
    await r.loadReport(7, { ...emptyScope(), name: 'model.*' }, false, false);
    await r.loadReport(7, { ...emptyScope(), name: 'model.*' }, true, false);
    await r.loadReport(7, { ...emptyScope(), name: 'model.*' }, true, true);
    expect(calls).toHaveLength(4);
  });

  it('has nothing to show for no comparison', async () => {
    const calls = stubFetch();
    const r = await load();
    await r.loadReport(null, undefined, false, false);
    expect(calls).toHaveLength(0);
    expect(get(r.diffReport)).toBeNull();
  });

  it('surfaces the server’s reason, and holds no stale report behind it', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve({
          ok: false,
          status: 409,
          headers: new Headers(),
          json: () => Promise.resolve({ error: 'no comparison set up' }),
          text: () => Promise.resolve('{"error":"no comparison set up"}'),
        }),
      ),
    );
    const r = await load();
    await r.loadReport(7, undefined, false, false);
    expect(get(r.diffReport)).toBeNull();
    expect(get(r.reportError)).toBe('no comparison set up');
    expect(get(r.reportWait)).toBeNull();
  });

  // A failed request must not poison the cache: the next attempt has to actually try again.
  it('retries after a failure', async () => {
    let fail = true;
    const calls: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        calls.push(url);
        const ok = !fail;
        fail = false;
        return Promise.resolve({
          ok,
          status: ok ? 200 : 500,
          headers: new Headers(),
          json: () => Promise.resolve(ok ? answer(7) : { error: 'boom' }),
          text: () => Promise.resolve('{}'),
        });
      }),
    );
    const r = await load();
    await r.loadReport(7, undefined, false, false);
    await r.loadReport(7, undefined, false, false);
    expect(calls).toHaveLength(2);
    expect(get(r.diffReport)?.verdict).toBe('report 7');
  });

  it('forgets everything when the comparison is cleared', async () => {
    stubFetch();
    const r = await load();
    await r.loadReport(7, undefined, false, false);
    r.clearReport();
    expect(get(r.diffReport)).toBeNull();
    expect(get(r.reportError)).toBeNull();
    // And the cache with it: the same id must be readable again after a clear.
    const calls = stubFetch();
    await r.loadReport(7, undefined, false, false);
    expect(calls).toHaveLength(1);
  });
});
