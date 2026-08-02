// The browser half of the query-parameter contract.
//
// `shared/parity/queryparams.json` is generated from the server's own allowlist
// (`src/web/mod.rs::accepted_params`, which is what `unknown_params` refuses against). This drives
// the real `api.*` functions through a stubbed `fetch` and asserts that every key they put on a URL
// is one the server accepts.
//
// **Why it is done this way.** The check this replaces was a hand-copied list of the client's keys,
// held in Rust. By the time it was read it was missing `align_fused`, `subtree`, `subtree_new`,
// `full`, `names_list` and `map_json` — and it passed, because a stale copy of the client agrees
// with itself. A client parameter the server does not accept is a `400` on a screen that used to
// work, so the only check worth having compares against what the client *actually sends*.
//
// A failure means one of two things: the client gained a parameter the server does not take (add it
// to `accepted_params` and regenerate), or the server's list changed under the client (regenerate
// with `UPDATE_PARITY=1 cargo test the_accepted_parameters` and make the client match).

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';
import { emptyScope, scopeToQuery, type DiffScopeParams } from './diffscope';

interface Fixture {
  scope: string[];
  routes: Record<string, string[]>;
}

const fixture = JSON.parse(
  readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), '../../../shared/parity/queryparams.json'),
    'utf8',
  ),
) as Fixture;

/** Every field set, so `scopeToQuery` emits everything it knows how to emit. */
const FULL_SCOPE: DiffScopeParams = {
  ...emptyScope(),
  name: 'model.layers.1.*',
  names: 'lm_head.weight',
  dtypeIs: 'F*',
  shapeIs: '768,**',
  map: '^blocks\\.=>model.layers.',
  onlyTensors: true,
  alignFused: true,
  subtree: 'language_model',
  subtreeNew: 'model',
};

/** Record every URL requested; answer with something each caller can parse. */
function recordUrls(): string[] {
  const urls: string[] = [];
  const body = { rows: [], differences: [], base: {}, current: {} };
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string) => {
      urls.push(url);
      return Promise.resolve({
        ok: true,
        status: 200,
        headers: new Headers(),
        json: () => Promise.resolve(body),
        text: () => Promise.resolve(JSON.stringify(body)),
        body: null,
      });
    }),
  );
  return urls;
}

/** The route an `/api/...` URL names, and the keys it carries. */
function parts(url: string): { route: string; keys: string[] } {
  const [path, query = ''] = url.replace(/^\/api\//, '').split('?');
  return { route: path ?? '', keys: [...new URLSearchParams(query).keys()] };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('every parameter the browser sends', () => {
  it('is one the server accepts, for every call the API layer makes', async () => {
    const urls = recordUrls();
    // One call per API function that takes parameters — the scoped ones with *everything* set, so
    // the check covers the keys a plain call would leave out.
    await api.open('/ckpt', true);
    await api.forgetRecent('/ckpt');
    await api.setComparison('/a', '/b', true);
    await api.difftree(1, FULL_SCOPE, undefined, undefined, undefined, true);
    await api.diff(1, FULL_SCOPE, true, true);
    await api.tensor('model.w');
    await api.layout('shard.safetensors');
    await api.file('config.json');
    await api.filter('dtype:f16');
    await api.compact('');
    await api.schema('');
    await api.tensorStats('model.w', 'f16');
    await api.sample('model.w', { mode: 'window', rows: 8, cols: 8, row_off: 1, col_off: 2 });
    await api.histogram('model.w', 64, 'f16');
    await api.startJob('values', [
      ['left', '/a'],
      ['right', '/b'],
      ['values', '1'],
      ...scopeToQuery(FULL_SCOPE),
    ]);
    await api.startJob('verify-repack', [
      ['left', '/a'],
      ['right', '/b'],
      ['repack_bits', '3'],
      ...scopeToQuery(FULL_SCOPE),
    ]);

    expect(urls.length).toBeGreaterThan(10);
    for (const url of urls) {
      const { route, keys } = parts(url);
      const accepted = fixture.routes[route];
      expect(accepted, `${route} is not a route the server publishes`).toBeDefined();
      for (const key of keys) {
        expect(accepted, `/api/${route} sends ?${key}=, which the server refuses`).toContain(key);
      }
    }
  });

  // The other direction, for the scope specifically: a parameter the server takes and no client
  // sends is dead weight at best — and at worst a feature that shipped without a way to use it,
  // which is how `subtree` spent a day being server-only.
  it('covers every scope parameter the server takes, or says why not', () => {
    // `names_list` is the pasted content of what `--names-from` reads and `map_json` its rename
    // equivalent: both are accepted so a script or a future screen can post one, and neither has a
    // control in the UI today. Anything else missing here is an oversight.
    const deliberatelyUnsent = ['names_list', 'map_json'];
    const sent = scopeToQuery(FULL_SCOPE).map(([k]) => k);
    for (const key of fixture.scope) {
      if (deliberatelyUnsent.includes(key)) {
        expect(sent, `${key} is listed as unsent but the client now sends it`).not.toContain(key);
        continue;
      }
      expect(sent, `the server takes ?${key}= and no control produces it`).toContain(key);
    }
  });
});
