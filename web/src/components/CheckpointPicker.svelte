<script lang="ts">
  /**
   * A checkpoint address box, with its recents dropdown. **The** one, used everywhere a checkpoint is
   * named: the header's address bar, both sides of the comparison screen, and the open prompt.
   *
   * They were three implementations of the same control and differed in ways nobody chose. The
   * comparison boxes used a native `<datalist>`, which suggests entries and offers *no per-entry
   * action*, so a stale path could be picked there for ever and only deleted from the header. The
   * header had a hand-rolled dropdown with the forget flow, its own keyboard, and its own
   * Escape-anywhere rule. The open prompt had neither.
   *
   * What is shared: the field (`TextField`, so it owns its keys), the list (`RecentsMenu`, so it can
   * forget an entry), ↓/↑/Escape/Enter, and dismissal. What the caller keeps is what it *does* — Enter
   * on the address bar opens a checkpoint, Enter on a comparison box compares — which is why those are
   * callbacks rather than behaviour baked in here.
   */
  import TextField from './TextField.svelte';
  import RecentsMenu from './RecentsMenu.svelte';
  import { recents, tree } from '../stores/server';

  export let id: string;
  export let value = '';
  export let placeholder = '';
  export let title = '';
  export let ariaLabel = '';
  /** `bare` for the header, where the box is the page's title bar rather than a control on it. */
  export let variant: 'boxed' | 'bare' = 'boxed';
  /** Read-only and inert while something is loading. */
  export let busy = false;
  /**
   * Show the dropdown at all.
   *
   * Off for the open screen, which already lists every recent underneath — the same entries in a popup
   * over a list of them is one list too many. The *component* is still this one, so the field, the keys
   * and the accepted spellings cannot drift there.
   */
  export let menu = true;
  /** Enter, or the Open/Compare button's equivalent: what this box is *for*. */
  export let onEnter: ((spec: string) => void) | null = null;
  /**
   * A recent was chosen. The default fills the box, which is what a comparison side wants; the address
   * bar passes its own, because choosing a recent there means "open it".
   */
  export let onPickRecent: ((spec: string) => void) | null = null;
  /** Escape with the menu closed. The address bar reverts its draft; the others do nothing. */
  export let onEscape: (() => void) | null = null;

  let open = false;
  /** Which row the keyboard is on; `-1` = none, so Enter means "what is typed". */
  let cursor = -1;
  let el: HTMLInputElement | HTMLTextAreaElement | null = null;

  /** Everything recent except what is already in the box — offering it back is a no-op. */
  const trim = (s: string) => s.replace(/\/+$/, '');
  $: options = $recents.filter((s) => trim(s) !== trim(value));

  export function focus() {
    el?.focus();
  }

  /** Give up the caret — what the address bar does when Enter names the checkpoint already served.
   * Declared, not improvised: the header called `picker?.blur?.()` on a method that did not exist, so
   * the field kept focus and nothing said why. */
  export function blur() {
    el?.blur();
  }

  function close() {
    open = false;
    cursor = -1;
  }

  function pick(spec: string) {
    close();
    if (onPickRecent) onPickRecent(spec);
    else value = spec;
  }

  function onKeydown(e: KeyboardEvent) {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!menu) return;
        if (!open) {
          open = true;
          cursor = -1;
        } else if (options.length) {
          cursor = Math.min(cursor + 1, options.length - 1);
        }
        return;
      case 'ArrowUp':
        e.preventDefault();
        // Up from the first entry returns to what was typed, rather than wrapping to the end.
        if (open) cursor = Math.max(cursor - 1, -1);
        return;
      case 'Escape':
        e.preventDefault();
        if (open) close();
        else onEscape?.();
        return;
      case 'Enter':
        e.preventDefault();
        if (open && cursor >= 0) pick(options[cursor] ?? value);
        else onEnter?.(value);
        return;
      default:
        // Any edit invalidates a highlighted recent.
        cursor = -1;
    }
  }

  /**
   * Escape while the menu is open, wherever focus is.
   *
   * The field's own handler only fires when the caret is in it — and after clicking a cross, focus is on
   * that button, so a pending "forget this?" could not be dismissed with the keyboard. `|capture` so
   * this runs before the field stops propagation.
   */
  function onWindowKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && open) {
      e.preventDefault();
      e.stopPropagation();
      close();
    }
  }
</script>

<svelte:window on:keydown|capture={onWindowKeydown} />

<span class="picker">
  <TextField
    {id}
    {variant}
    bind:el
    bind:value
    on:keydown={onKeydown}
    on:focus
    readonly={busy}
    spellcheck="false"
    autocomplete="off"
    aria-label={ariaLabel || undefined}
    {placeholder}
    {title}
  />
  {#if menu && options.length}
    <button
      type="button"
      class="caret"
      aria-expanded={open}
      aria-label="Recently opened checkpoints"
      title="Recently opened (↓)"
      disabled={busy}
      on:click={() => (open ? close() : ((open = true), (cursor = -1)))}>▾</button
    >
  {/if}
  <RecentsMenu
    {options}
    open={menu && open}
    {cursor}
    {busy}
    current={$tree?.spec ?? ''}
    label={ariaLabel ? `Recent checkpoints — ${ariaLabel}` : 'Recently opened checkpoints'}
    onPick={pick}
    onClose={close}
  />
</span>

<style>
  /* The field and its dropdown: positioned, so the popup anchors to this box rather than to the page. */
  .picker {
    position: relative;
    display: flex;
    align-items: center;
    gap: 3px;
    flex: 1 1 auto;
    min-width: 0;
  }
  .caret {
    flex: none;
    font: inherit;
    font-size: 11px;
    color: var(--fg-dim);
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 5px;
    cursor: pointer;
  }
  .caret:hover:not(:disabled) {
    color: var(--fg);
  }
</style>
