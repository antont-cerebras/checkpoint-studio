// Who owns a keystroke. The rule is one line and was written out four times, differently — see
// `keys.ts`. These are the cases those four disagreed on.

import { describe, expect, it } from 'vitest';
import { isEditable } from './keys';

/** An event target shaped like an element of `tag`. Plain objects because this suite runs without a
 * DOM on purpose (see vitest.config.ts) — which is also why the rule itself is duck-typed. */
function el(tag: string, contentEditable = false): EventTarget {
  return { tagName: tag.toUpperCase(), isContentEditable: contentEditable } as unknown as EventTarget;
}

describe('is this key being typed into something', () => {
  it('says yes for every kind of field', () => {
    expect(isEditable(el('input'))).toBe(true);
    // The one the compare screen's handler missed, so `s` typed into a scope box swapped the sides.
    expect(isEditable(el('textarea'))).toBe(true);
    // A `select` navigates by letter, so a shortcut would fight its own type-ahead.
    expect(isEditable(el('select'))).toBe(true);
    expect(isEditable(el('div', true))).toBe(true);
  });

  it('says no for the page itself', () => {
    expect(isEditable(el('div'))).toBe(false);
    expect(isEditable(el('button'))).toBe(false);
    expect(isEditable(null)).toBe(false);
  });

  it('says no for a target that is not an element', () => {
    // `window` and `document` are event targets too; a shortcut handler must not throw on them.
    expect(isEditable({} as EventTarget)).toBe(false);
    expect(isEditable({ tagName: 42 } as unknown as EventTarget)).toBe(false);
  });

  it('is case-insensitive about the tag, as the DOM is about markup', () => {
    expect(isEditable({ tagName: 'input' } as unknown as EventTarget)).toBe(true);
  });
});
