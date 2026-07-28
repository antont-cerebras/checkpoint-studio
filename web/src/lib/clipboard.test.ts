// Copying. The VM serves this over plain http on a hostname, which is NOT a secure
// context, so the async Clipboard API is simply absent there — the legacy textarea
// path is the one that actually runs in the deployment we use, and it has to leave no
// stray element behind. Both branches are stubbed here because a broken copy button
// looks identical to a working one until you paste.

import { afterEach, describe, expect, it, vi } from 'vitest';
import { copyText } from './clipboard';

interface Fake {
  appended: unknown[];
  removed: unknown[];
  selected: string[];
}

/** Stub just enough DOM for the legacy path, plus an optional Clipboard API. */
function stubDom(opts: {
  secure: boolean;
  clipboard?: { writeText: (t: string) => Promise<void> };
  execCommand?: () => boolean;
  throwOnCreate?: boolean;
}): Fake {
  const fake: Fake = { appended: [], removed: [], selected: [] };
  const doc = {
    createElement: () => {
      if (opts.throwOnCreate) throw new Error('no DOM');
      return {
        style: {},
        value: '',
        setAttribute: () => undefined,
        focus: () => undefined,
        select() {
          fake.selected.push((this as { value: string }).value);
        },
      };
    },
    body: {
      appendChild: (el: unknown) => fake.appended.push(el),
      removeChild: (el: unknown) => fake.removed.push(el),
    },
    execCommand: opts.execCommand ?? (() => true),
  };
  vi.stubGlobal('document', doc);
  vi.stubGlobal('window', { isSecureContext: opts.secure });
  vi.stubGlobal('navigator', opts.clipboard ? { clipboard: opts.clipboard } : {});
  return fake;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('copyText', () => {
  it('uses the Clipboard API in a secure context', () => {
    const writeText = vi.fn(() => Promise.resolve());
    const fake = stubDom({ secure: true, clipboard: { writeText } });
    expect(copyText('hello')).toBe(true);
    expect(writeText).toHaveBeenCalledWith('hello');
    expect(fake.appended).toHaveLength(0); // no textarea needed
  });

  it('falls back to the textarea when the Clipboard API rejects', async () => {
    const writeText = vi.fn(() => Promise.reject(new Error('denied')));
    const fake = stubDom({ secure: true, clipboard: { writeText } });
    expect(copyText('hi')).toBe(true);
    await Promise.resolve(); // let the rejection handler run
    expect(fake.selected).toEqual(['hi']);
  });

  it('uses the textarea over plain http, where navigator.clipboard is absent', () => {
    const fake = stubDom({ secure: false });
    expect(copyText('model.layers.0.weight')).toBe(true);
    expect(fake.selected).toEqual(['model.layers.0.weight']);
  });

  it('removes the textarea it added, even when the copy fails', () => {
    const fake = stubDom({ secure: false, execCommand: () => false });
    expect(copyText('x')).toBe(false);
    expect(fake.appended).toHaveLength(1);
    expect(fake.removed).toEqual(fake.appended);
  });

  it('reports failure instead of throwing when the DOM refuses', () => {
    stubDom({ secure: false, throwOnCreate: true });
    expect(copyText('x')).toBe(false);
  });

  it('copies an empty string without claiming success it did not have', () => {
    const fake = stubDom({ secure: false });
    expect(copyText('')).toBe(true);
    expect(fake.selected).toEqual(['']);
  });
});
