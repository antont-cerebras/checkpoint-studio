// Who owns a keystroke.
//
// Every screen here has single-letter shortcuts — `s` swaps a comparison, `n` steps to the next
// difference, `:` opens the command palette — and every screen also has boxes you type paths and globs
// into. The rule between them is one line long and was written out four times, slightly differently:
// one handler checked `INPUT` and `SELECT` but not `TEXTAREA`, so `s` typed into a scope box swapped the
// two checkpoints instead of appearing on screen; the global one checked the right tags but ran *after*
// the palette shortcut, so `:` could not be typed at all — in the very boxes whose placeholder offers
// `:/path` as an accepted spelling.
//
// So the rule lives here, once, and every handler asks it. Same reasoning as `crate::capability` on the
// Rust side: a fact worth getting right is a fact worth having one answer to.

/**
 * Whether a key event is being typed into something editable.
 *
 * `true` for an `<input>` of any type, a `<textarea>`, a `<select>` (which navigates by letter) and
 * anything `contenteditable`. A handler that sees `true` must return without touching the event: the
 * field owns the key, including `s`, `n`, `:`, Space and Backspace.
 */
export function isEditable(target: EventTarget | null): boolean {
  // Duck-typed rather than `instanceof HTMLElement`: an event target can come from another realm (an
  // iframe has its own `HTMLElement`), and this way the rule is testable without a DOM — which this
  // suite deliberately runs without. `window` and `document` have no `tagName`, so they are not fields.
  const el = target as { tagName?: unknown; isContentEditable?: unknown } | null;
  if (!el || typeof el.tagName !== 'string') return false;
  const tag = el.tagName.toUpperCase();
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable === true;
}
