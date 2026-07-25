// The hash is the whole view state, so what matters is that it ROUND-TRIPS: a link
// someone shares (or the browser's back button) must reopen the exact view. These
// tests drive that in both directions, and pin the fallbacks that keep a
// hand-edited or stale link from stranding the app on a blank screen.

import { describe, expect, it } from 'vitest';
import {
  DV_KEYS,
  globalQuery,
  hashFor,
  parseGlobals,
  parseScreen,
  screenToHash,
  type Globals,
  type Screen,
} from './hash';

const DEFAULTS: Globals = {
  filter: '',
  sortKey: 'none',
  sortDir: 'asc',
  compact: false,
  searching: false,
  search: '',
};

const SCREENS: Screen[] = [
  { kind: 'tree' },
  { kind: 'files' },
  { kind: 'stats' },
  { kind: 'health' },
  { kind: 'layout' },
  { kind: 'layout', file: 'model-00001-of-00002.safetensors' },
  { kind: 'detail', tensor: 'model.layers.0.mlp.gate_proj.weight', tab: 'info' },
  { kind: 'detail', tensor: 'lm_head.weight', tab: 'heatmap' },
  { kind: 'preview', path: '/ckpt/config.json', name: 'config.json' },
];

describe('screen round-trip', () => {
  it.each(SCREENS)('survives hash → parse → hash for %j', (s) => {
    const parsed = parseScreen(`#${screenToHash(s)}`);
    expect(parsed).toEqual({ ...s, ...(s.kind === 'detail' ? { dv: undefined } : {}) });
    expect(screenToHash(parsed)).toBe(screenToHash(s));
  });

  it('carries every data-view param through unchanged', () => {
    const dv = Object.fromEntries(DV_KEYS.map((k, i) => [k, String(i)]));
    const s: Screen = { kind: 'detail', tensor: 'w', tab: 'values', dv };
    expect(parseScreen(`#${screenToHash(s)}`)).toEqual(s);
  });

  it('omits the dv object entirely when no param is set', () => {
    const h = screenToHash({ kind: 'detail', tensor: 'w', tab: 'info' });
    expect(h).toBe('detail?t=w&tab=info');
    expect(parseScreen(h)).toEqual({ kind: 'detail', tensor: 'w', tab: 'info', dv: undefined });
  });

  // Tensor names carry `/` and `.`, and metadata keys have carried `&`/`?`/`#`/spaces.
  // A single round of encoding has to survive each of them, or the link opens the
  // wrong tensor (or no tensor).
  it.each([
    'model/layers.0/weight',
    'a&b=c?d#e',
    'name with spaces',
    'unicode·näme',
    'plus+sign',
    '100%',
  ])('encodes and recovers the awkward name %s', (name) => {
    expect(parseScreen(`#${screenToHash({ kind: 'detail', tensor: name, tab: 'info' })}`)).toEqual({
      kind: 'detail',
      tensor: name,
      tab: 'info',
      dv: undefined,
    });
  });

  it('a file path with a hash character still opens the preview it names', () => {
    const s: Screen = { kind: 'preview', path: '/ckpt/a#b.json', name: 'a#b.json' };
    expect(parseScreen(`#${screenToHash(s)}`)).toEqual(s);
  });
});

describe('screen fallbacks', () => {
  it('falls back to the tree for an unknown screen, an empty hash, or bare "#"', () => {
    for (const h of ['', '#', '#nope', '#detail'] /* detail without ?t= */) {
      expect(parseScreen(h)).toEqual({ kind: 'tree' });
    }
  });

  it('falls back to the tree when a preview has no path', () => {
    expect(parseScreen('#preview?name=x')).toEqual({ kind: 'tree' });
  });

  it('names a preview after its path when the name is missing', () => {
    expect(parseScreen('#preview?path=/a/b.json')).toEqual({
      kind: 'preview',
      path: '/a/b.json',
      name: '/a/b.json',
    });
  });

  it('falls back to the info tab for an unknown tab', () => {
    expect(parseScreen('#detail?t=w&tab=bogus')).toMatchObject({ tab: 'info' });
    expect(parseScreen('#detail?t=w')).toMatchObject({ tab: 'info' });
  });

  it('treats a layout with no file as the layout screen (the app picks a shard)', () => {
    expect(parseScreen('#layout')).toEqual({ kind: 'layout', file: undefined });
  });
});

describe('global state round-trip', () => {
  it('is empty for the default view, so a plain URL stays plain', () => {
    expect(globalQuery(DEFAULTS)).toBe('');
    expect(hashFor({ kind: 'tree' }, DEFAULTS)).toBe('tree');
  });

  it('round-trips filter, sort, compact and search together', () => {
    const g: Globals = {
      filter: 'dtype:BF16 shape:(2048,2048)',
      sortKey: 'size',
      sortDir: 'desc',
      compact: true,
      searching: true,
      search: 'gate_proj',
    };
    expect(parseGlobals(`#tree?${globalQuery(g)}`)).toEqual(g);
  });

  it('keeps search MODE with an empty query distinct from no search', () => {
    const open = { ...DEFAULTS, searching: true, search: '' };
    expect(globalQuery(open)).toBe('q=');
    expect(parseGlobals('#tree?q=')).toEqual(open);
    expect(parseGlobals('#tree')).toEqual(DEFAULTS);
  });

  it('trims the filter, so trailing whitespace never lands in the URL', () => {
    expect(globalQuery({ ...DEFAULTS, filter: '  dtype:F32  ' })).toBe('filter=dtype%3AF32');
  });

  it('drops an unknown sort key and defaults the direction to ascending', () => {
    expect(parseGlobals('#tree?sort=bogus.desc')).toMatchObject({ sortKey: 'none' });
    expect(parseGlobals('#tree?sort=name')).toMatchObject({ sortKey: 'name', sortDir: 'asc' });
    expect(parseGlobals('#tree?sort=name.sideways')).toMatchObject({ sortDir: 'asc' });
  });

  it('reads compact only from an exact "1"', () => {
    expect(parseGlobals('#tree?compact=1').compact).toBe(true);
    expect(parseGlobals('#tree?compact=0').compact).toBe(false);
    expect(parseGlobals('#tree?compact=true').compact).toBe(false);
  });
});

describe('hashFor', () => {
  const g: Globals = { ...DEFAULTS, filter: 'w', compact: true };

  it('joins the global state with & when the screen already has a query', () => {
    expect(hashFor({ kind: 'detail', tensor: 'w', tab: 'info' }, g)).toBe(
      'detail?t=w&tab=info&filter=w&compact=1',
    );
  });

  it('joins it with ? when the screen has none', () => {
    expect(hashFor({ kind: 'stats' }, g)).toBe('stats?filter=w&compact=1');
  });

  it('produces a hash both halves can be read back out of', () => {
    const s: Screen = { kind: 'detail', tensor: 'x/y.w', tab: 'values', dv: { mode: 'window', rows: '16' } };
    const full = `#${hashFor(s, g)}`;
    expect(parseScreen(full)).toEqual(s);
    expect(parseGlobals(full)).toEqual(g);
  });
});
