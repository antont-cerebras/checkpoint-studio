// The fetch layer. Two things here have bitten us before and are worth pinning:
// the error path (a failed request has to surface the server's `{error}` message, not
// a blank pane — finding N1 in the UI review), and query-string building (a tensor
// name with a `/` or `&` in it must reach the server intact, and `undefined` params
// must not turn into the string "undefined").

import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from './api';

/**
 * A stand-in `Response` that answers what the real one does: status, `headers` (so
 * `totalBytes` has something to read), `json`/`text`, and a readable `body` — which
 * `api.tree(onProgress)` streams. A mock without those looks like a `Response` right up
 * until the code asks it something.
 */
function response(
  status: number,
  body: unknown,
  opts: { malformed?: boolean; headers?: Record<string, string> } = {},
) {
  const text = opts.malformed ? 'not json' : JSON.stringify(body ?? null);
  const bytes = new TextEncoder().encode(text);
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers(opts.headers ?? { 'Content-Length': String(bytes.length) }),
    json: () => (opts.malformed ? Promise.reject(new Error('not json')) : Promise.resolve(body)),
    text: () => Promise.resolve(text),
    body: {
      getReader() {
        let sent = false;
        return {
          read: () =>
            Promise.resolve(
              sent ? { done: true, value: undefined } : ((sent = true), { done: false, value: bytes }),
            ),
        };
      },
    },
  };
}

/** Record the URLs requested and reply with a canned response. */
function stubFetch(reply: { status?: number; body?: unknown; malformed?: boolean }) {
  const urls: string[] = [];
  const fetchStub = vi.fn((url: string) => {
    urls.push(url);
    const status = reply.status ?? 200;
    return Promise.resolve(
      response(status, reply.body, { malformed: reply.malformed ?? false }),
    );
  });
  vi.stubGlobal('fetch', fetchStub);
  return urls;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('responses', () => {
  it('returns the parsed body on success', async () => {
    stubFetch({ body: { tree: [], count: 3 } });
    await expect(api.tree()).resolves.toEqual({ tree: [], count: 3 });
  });

  it("surfaces the server's error message, not the status code", async () => {
    stubFetch({ status: 400, body: { error: 'unknown filter field: dtpye' } });
    await expect(api.filter('dtpye:F32')).rejects.toThrow('unknown filter field: dtpye');
  });

  it('falls back to the status when the error body is not the {error} envelope', async () => {
    stubFetch({ status: 500, body: { message: 'oops' } });
    await expect(api.stats()).rejects.toThrow('HTTP 500');
  });

  it('falls back to the status when the body is not JSON at all', async () => {
    stubFetch({ status: 502, malformed: true });
    await expect(api.stats()).rejects.toThrow('HTTP 502');
  });

  it('rejects an {error} value that is not a string rather than showing "[object Object]"', async () => {
    stubFetch({ status: 400, body: { error: { code: 7 } } });
    await expect(api.stats()).rejects.toThrow('HTTP 400');
  });

  it('accepts a null body on a 200 (the check endpoint reports "not run" that way)', async () => {
    stubFetch({ body: null });
    await expect(api.check()).resolves.toBeNull();
  });
});

describe('urls', () => {
  it('encodes a tensor name once, so slashes and dots survive', async () => {
    const urls = stubFetch({ body: {} });
    await api.tensor('model.layers.0/gate.weight');
    expect(urls[0]).toBe('/api/tensor?name=model.layers.0%2Fgate.weight');
  });

  it('encodes the characters that would otherwise split the query string', async () => {
    const urls = stubFetch({ body: {} });
    await api.filter('name:a&b q=?#');
    expect(urls[0]).toBe('/api/filter?q=name%3Aa%26b%20q%3D%3F%23');
  });

  it('drops undefined and empty params instead of sending them', async () => {
    const urls = stubFetch({ body: {} });
    await api.tensorStats('w');
    expect(urls[0]).toBe('/api/tensor/stats?name=w');
    await api.tensorStats('w', 'F32');
    expect(urls[1]).toBe('/api/tensor/stats?name=w&dtype=F32');
    await api.histogram('w', undefined, undefined);
    expect(urls[2]).toBe('/api/tensor/histogram?name=w');
  });

  it('keeps a 0 offset — it is meaningful, unlike undefined', async () => {
    const urls = stubFetch({ body: {} });
    await api.sample('w', { mode: 'window', rows: 8, cols: 8, row_off: 0, col_off: 16 });
    expect(urls[0]).toBe('/api/tensor/sample?name=w&mode=window&rows=8&cols=8&row_off=0&col_off=16');
  });

  it('hits the endpoint each accessor names', async () => {
    const urls = stubFetch({ body: {} });
    await Promise.all([api.tree(), api.files(), api.health(), api.check(), api.stats()]);
    expect(urls).toEqual(['/api/tree', '/api/files', '/api/health', '/api/check', '/api/stats']);
  });

  it('asks the schema endpoint for the family breakdown of a query', async () => {
    const urls = stubFetch({ body: { families: [] } });
    await expect(api.schema('*.mlp.*')).resolves.toEqual({ families: [] });
    expect(urls[0]).toBe('/api/schema?q=*.mlp.*');
  });

  /* The durable version of the two tests below. `api.compact` and `api.diff` went
     uncovered because every other case names its endpoint by hand, so a method added with
     a feature is only tested if someone remembers to come back here. This calls all of
     them generically: a new accessor that never reaches the network fails immediately,
     whatever it is called. */
  it('every accessor issues exactly one request to an /api/ path', async () => {
    const urls = stubFetch({ body: {} });
    const names = Object.keys(api) as (keyof typeof api)[];
    expect(names.length, 'the api object is not empty').toBeGreaterThan(10);
    for (const name of names) {
      // One string argument satisfies every accessor's first parameter (name / path /
      // query / against); the ones taking none ignore it, and `tree` — whose parameter is
      // a progress callback — checks `typeof` rather than truthiness so it does too.
      const call = api[name] as (...args: unknown[]) => Promise<unknown>;
      await call('x');
    }
    expect(urls).toHaveLength(names.length);
    for (const [i, url] of urls.entries()) {
      expect(url, `${names[i]} must hit the JSON API`).toMatch(/^\/api\//);
    }
  });

  it('asks the compact endpoint for the family-folded tree, scoped by the filter', async () => {
    const urls = stubFetch({ body: { tree: [], counts: {}, varying: [], tensor_count: 0 } });
    await expect(api.compact('')).resolves.toEqual({
      tree: [],
      counts: {},
      varying: [],
      tensor_count: 0,
    });
    // The scope rides along even when empty, so the server decides what "unscoped" means
    // rather than the two callers disagreeing about it.
    expect(urls[0]).toBe('/api/compact?q=');
    await api.compact('dtype:F32 *.mlp.*');
    expect(urls[1]).toBe('/api/compact?q=dtype%3AF32%20*.mlp.*');
  });

  it('encodes a compare path for the diff endpoint', async () => {
    const urls = stubFetch({ body: { added: [], removed: [], changed: [] } });
    await api.diff('/cb/home/u/ws/Qwen3 A3B/');
    expect(urls[0]).toBe('/api/diff?against=%2Fcb%2Fhome%2Fu%2Fws%2FQwen3%20A3B%2F');
  });

  it("surfaces the server's reason when a compare path is not a checkpoint", async () => {
    // The 400 path is the one a user actually hits — a typo'd or non-checkpoint path —
    // and the message has to reach the compare box instead of an empty report.
    stubFetch({ status: 400, body: { error: 'no checkpoint files found at /tmp/x' } });
    await expect(api.diff('/tmp/x')).rejects.toThrow('no checkpoint files found at /tmp/x');
  });

  it('encodes a shard path for the layout and file endpoints', async () => {
    const urls = stubFetch({ body: {} });
    await api.layout('model-00001-of-00002.safetensors');
    await api.file('/ckpt/a b.json');
    expect(urls).toEqual([
      '/api/layout?file=model-00001-of-00002.safetensors',
      '/api/file?path=%2Fckpt%2Fa%20b.json',
    ]);
  });
});
