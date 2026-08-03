// The read-progress poll. `fetch` is stubbed and the timer is faked, so this is about the poll's
// *lifetime* rather than its payload: one timer however many screens are watching, stopped and cleared
// when the last of them goes away.
//
// A leaked interval here is not a visible bug — it is a request every 400ms for the rest of the
// session, against a server that is doing nothing.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

const PROGRESS = {
  seconds: 3.2,
  // Two rows: a comparison reads both of its checkpoints at once, and each reports for itself.
  sides: [
    {
      spec: 's3://bucket/ckpt',
      done: 44,
      total: 66,
      unit: 'S3 objects',
      stage: 'reading S3 storage metadata',
      finished: false,
    },
    {
      spec: 'lab@host:/opt/models/ckpt',
      done: 12,
      total: 30,
      unit: 'shards',
      stage: 'reading shard headers',
      finished: false,
    },
  ],
};

describe('the read-progress poll', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  /** A `fetch` that answers `/api/reading` with `body`, counting the calls. */
  function stubFetch(body: unknown): { calls: () => number } {
    let calls = 0;
    vi.stubGlobal('fetch', () => {
      calls += 1;
      return Promise.resolve({
        ok: true,
        // A `Response` has headers, and the API layer reads one off every reply (the build the
        // server serves) — a stub without them is not a reply this app can be handed.
        headers: new Headers(),
        json: () => Promise.resolve(body),
      } as unknown as Response);
    });
    return { calls: () => calls };
  }

  it('polls immediately, then on an interval, and stops when the watcher goes', async () => {
    const f = stubFetch({ reading: PROGRESS });
    const { reading, watchReading } = await import('./reading');

    const stop = watchReading();
    // Straight away, so the first numbers do not wait out an interval.
    expect(f.calls()).toBe(1);
    await vi.advanceTimersByTimeAsync(400);
    expect(f.calls()).toBe(2);
    expect(get(reading)).toEqual(PROGRESS);

    stop();
    await vi.advanceTimersByTimeAsync(2000);
    expect(f.calls(), 'no polling once nothing is watching').toBe(2);
    expect(get(reading), 'and no stale progress left on screen').toBeNull();
  });

  it('runs one timer however many screens are watching', async () => {
    const f = stubFetch({ reading: PROGRESS });
    const { watchReading } = await import('./reading');

    const stopA = watchReading();
    const stopB = watchReading();
    expect(f.calls(), 'the second watcher joins the running poll').toBe(1);
    await vi.advanceTimersByTimeAsync(400);
    expect(f.calls(), 'one request per interval, not one per watcher').toBe(2);

    // The first to leave must not take the other's poll with it.
    stopA();
    await vi.advanceTimersByTimeAsync(400);
    expect(f.calls()).toBe(3);
    stopB();
    await vi.advanceTimersByTimeAsync(400);
    expect(f.calls()).toBe(3);
  });

  it('reports an idle server as no progress rather than as an error', async () => {
    stubFetch({ reading: null });
    const { reading, watchReading } = await import('./reading');
    const stop = watchReading();
    await vi.advanceTimersByTimeAsync(0);
    expect(get(reading)).toBeNull();
    stop();
  });

  it('swallows a failed poll — the request it accompanies does the reporting', async () => {
    vi.stubGlobal('fetch', () => Promise.reject(new Error('network gone')));
    const { reading, watchReading } = await import('./reading');
    const stop = watchReading();
    await vi.advanceTimersByTimeAsync(0);
    expect(get(reading)).toBeNull();
    stop();
  });
});
