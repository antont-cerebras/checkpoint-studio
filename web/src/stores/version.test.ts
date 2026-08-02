// The stale-build check: from a response header, at start, when the tab comes back, and — while it is
// in front — on a slow timer. Never un-said.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'svelte/store';

/**
 * A `fetch` answering `/api/version` with `assets`, counting the calls.
 *
 * With the header every real response carries: the API layer reads it off each one, so a stub without
 * it is not a response this app can be handed.
 */
function stubVersion(assets: string | null): { calls: () => number } {
  let calls = 0;
  vi.stubGlobal('fetch', () => {
    calls += 1;
    return Promise.resolve({
      ok: true,
      headers: new Headers(assets === null ? {} : { 'X-App-Build': assets }),
      json: () => Promise.resolve({ app: '0.2.1', assets, spec: '/ckpt' }),
    } as unknown as Response);
  });
  return { calls: () => calls };
}

/** A document that can be brought to the front. */
function stubDocument(): { show: () => void; hide: () => void; listeners: () => number } {
  const handlers: (() => void)[] = [];
  let state = 'visible';
  vi.stubGlobal('document', {
    get visibilityState() {
      return state;
    },
    addEventListener: (_: string, h: () => void) => handlers.push(h),
    removeEventListener: (_: string, h: () => void) => {
      const i = handlers.indexOf(h);
      if (i >= 0) handlers.splice(i, 1);
    },
  });
  return {
    show: () => {
      state = 'visible';
      handlers.forEach((h) => h());
    },
    hide: () => {
      state = 'hidden';
      handlers.forEach((h) => h());
    },
    listeners: () => handlers.length,
  };
}

describe('noticing that the server was rebuilt under this tab', () => {
  beforeEach(() => vi.resetModules());
  afterEach(() => vi.unstubAllGlobals());

  it('does nothing when this tab has no build id — a dev server', async () => {
    const f = stubVersion('index-bbb.js');
    stubDocument();
    vi.doMock('../lib/build', async (orig) => ({
      ...(await orig<typeof import('../lib/build')>()),
      currentBuild: () => '',
    }));
    const { checkBuild, staleBuild } = await import('./version');
    await checkBuild();
    expect(f.calls(), 'nothing to compare, so nothing is asked').toBe(0);
    expect(get(staleBuild)).toBe(false);
  });

  it('raises the flag when the served build differs, and asks again on return', async () => {
    const f = stubVersion('index-bbb.js');
    const doc = stubDocument();
    vi.doMock('../lib/build', async (orig) => ({
      ...(await orig<typeof import('../lib/build')>()),
      currentBuild: () => 'index-aaa.js',
    }));
    const { watchBuild, staleBuild } = await import('./version');
    const stop = watchBuild();
    await vi.waitFor(() => expect(get(staleBuild)).toBe(true));
    expect(f.calls()).toBe(1);

    // Backgrounded and brought back: asked again, which is the case this exists for.
    doc.hide();
    doc.show();
    // The hidden event fires the listener too, and is ignored: only coming back asks again.
    await vi.waitFor(() => expect(f.calls()).toBe(2));
    stop();
    expect(doc.listeners()).toBe(0);
  });

  // The case this was reported over: the page sat open and *watched* while the server was reinstalled
  // under it, so nothing was in flight and nothing brought the tab to the front.
  it('keeps asking, slowly, while the tab is in front', async () => {
    vi.useFakeTimers();
    const f = stubVersion('index-aaa.js');
    const doc = stubDocument();
    vi.doMock('../lib/build', async (orig) => ({
      ...(await orig<typeof import('../lib/build')>()),
      currentBuild: () => 'index-aaa.js',
    }));
    const { watchBuild } = await import('./version');
    const stop = watchBuild();
    expect(f.calls()).toBe(1);
    await vi.advanceTimersByTimeAsync(35_000);
    expect(f.calls(), 'twice more over 35s').toBe(3);
    // Out of sight, out of mind: a backgrounded tab asks nothing at all.
    doc.hide();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(f.calls()).toBe(3);
    stop();
    vi.useRealTimers();
  });

  it('stays quiet while the builds match, and a failed check says nothing', async () => {
    stubVersion('index-aaa.js');
    stubDocument();
    vi.doMock('../lib/build', async (orig) => ({
      ...(await orig<typeof import('../lib/build')>()),
      currentBuild: () => 'index-aaa.js',
    }));
    const { checkBuild, staleBuild } = await import('./version');
    await checkBuild();
    expect(get(staleBuild)).toBe(false);

    // The server going away mid-restart is not evidence of anything.
    vi.stubGlobal('fetch', () => Promise.reject(new Error('connection refused')));
    await checkBuild();
    expect(get(staleBuild)).toBe(false);
  });

  it('does not un-say it once said', async () => {
    stubVersion('index-bbb.js');
    stubDocument();
    vi.doMock('../lib/build', async (orig) => ({
      ...(await orig<typeof import('../lib/build')>()),
      currentBuild: () => 'index-aaa.js',
    }));
    const { checkBuild, staleBuild } = await import('./version');
    await checkBuild();
    expect(get(staleBuild)).toBe(true);
    // A later check that cannot answer must not clear a warning that is still true.
    vi.stubGlobal('fetch', () => Promise.reject(new Error('restarting')));
    await checkBuild();
    expect(get(staleBuild)).toBe(true);
  });
});
