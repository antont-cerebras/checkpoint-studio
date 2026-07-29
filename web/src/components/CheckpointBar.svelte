<script lang="ts">
  // The checkpoint address bar: the path in the header, editable in place.
  //
  // A real `<input>`, not text that becomes one on click. Copy, paste, select-all, drag-select
  // and the caret all work because the browser already implements them — a `<span>` that swaps
  // itself for an input on click reimplements a worse version of each, and gets the first click
  // wrong (the click that focuses lands before the input exists, so it doesn't place a caret).
  //
  // Styled as the dim text it replaces until you touch it, so the header still reads as a
  // header: no border, transparent background, and a background fill on hover/focus — the way
  // the rest of this UI outlines things (see the dropdown below).
  //
  // Terminal parity: this is the browser's spelling of the palette's `Open another checkpoint…`,
  // which is what the terminal offers. An always-live editable address bar is a browser idiom
  // (it *is* the URL bar's affordance); the terminal's equivalent is its prompt with `↑` history.
  import { onMount } from 'svelte';
  import { forgetRecent, loadRecents, proxied, recents, tree } from '../stores/server';
  import { switchCheckpoint } from '../stores/open';
  import Spinner from './Spinner.svelte';

  let box: HTMLElement;
  let input: HTMLInputElement;
  let draft = '';
  /** True once the text diverges from the served root, so a switch elsewhere can't overwrite
   * what is being typed. */
  let dirty = false;
  let listOpen = false;
  /** Which recent the keyboard is on; -1 = none, so Enter opens what is typed. */
  let cursor = -1;
  let busy = false;
  let error = '';
  /** Which entry is asking "forget this?"; one at a time, so the answer is never ambiguous. */
  let confirming: string | null = null;

  onMount(loadRecents);

  // The *address*, not the display root: for a single-file checkpoint the root is its containing
  // directory, and opening that would read a different checkpoint (a directory of three HDF5
  // files has one root and three addresses).
  $: root = $tree?.spec ?? $tree?.root ?? '';
  // Follow the served checkpoint unless the box is being edited — otherwise a switch made
  // somewhere else (the open screen, another tab's doing) would leave the header lying.
  $: if (!dirty && !busy) draft = root;
  $: options = $recents.filter((s) => s !== draft);


  async function submit(spec: string) {
    const path = spec.trim();
    if (!path || busy) return;
    listOpen = false;
    cursor = -1;
    // Re-opening what is already served would re-read a 31k-tensor checkpoint to arrive back
    // where we started; treat it as "done" instead.
    if (path === root) {
      dirty = false;
      input?.blur();
      return;
    }
    busy = true;
    error = '';
    try {
      await switchCheckpoint(path);
      dirty = false;
      input?.blur();
    } catch (e) {
      // Leave the text as typed, with the caret still in it: the point of an editable bar is
      // that a wrong path is corrected where it was entered. `readonly` rather than `disabled`
      // during the wait is what makes that possible — a disabled input cannot hold focus, so
      // the browser had already moved it to <body> and Escape no longer reached this box.
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  function onKeydown(e: KeyboardEvent) {
    // The header sits inside the app's global key handling; every key typed here is for the
    // box, so none of them reach the tree.
    e.stopPropagation();
    switch (e.key) {
      case 'Enter':
        e.preventDefault();
        void submit(cursor >= 0 ? (options[cursor] ?? draft) : draft);
        return;
      case 'Escape':
        e.preventDefault();
        // Innermost state first: a pending "forget this?" is what Escape should answer, before it
        // starts closing the list or discarding what was typed.
        if (confirming !== null) {
          confirming = null;
        } else if (listOpen) {
          listOpen = false;
          cursor = -1;
        } else {
          draft = root;
          dirty = false;
          error = '';
          input.blur();
        }
        return;
      case 'ArrowDown':
        e.preventDefault();
        if (!listOpen) {
          listOpen = true;
          cursor = -1;
        } else if (options.length) {
          cursor = Math.min(cursor + 1, options.length - 1);
        }
        return;
      case 'ArrowUp':
        e.preventDefault();
        // Up from the first entry returns to what was typed, rather than wrapping to the end.
        if (listOpen) cursor = Math.max(cursor - 1, -1);
        return;
      default:
        // Any edit invalidates a highlighted recent and the stale error.
        cursor = -1;
        error = '';
    }
  }

  /** Remove one entry from the list. The checkpoint itself, and the one being served, are
   * untouched — forgetting the open one is allowed and leaves you looking at it. */
  async function forget(spec: string) {
    confirming = null;
    try {
      await forgetRecent(spec);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  function onInput() {
    dirty = draft !== root;
  }

  /**
   * Escape while the menu is open, wherever focus is.
   *
   * The input's own handler only fires when the caret is in it — and after clicking a cross, focus
   * is on that button, so a pending "forget this?" could not be dismissed with the keyboard. The
   * input stops propagation, so this never double-handles.
   */
  function onWindowKeydown(e: KeyboardEvent) {
    if (e.key !== 'Escape') return;
    if (confirming !== null) {
      e.preventDefault();
      e.stopPropagation();
      confirming = null;
    } else if (listOpen) {
      e.preventDefault();
      e.stopPropagation();
      listOpen = false;
      cursor = -1;
    }
  }

  /** Click anywhere else closes the dropdown — the same dismiss rule the palette follows. */
  function onWindowPointerDown(e: Event) {
    if (listOpen && box && !box.contains(e.target as Node)) {
      listOpen = false;
      cursor = -1;
      // An unanswered question does not survive the menu it was asked in.
      confirming = null;
    }
  }

  const short = (s: string) => s.replace(/\/+$/, '').split('/').pop() || s;
</script>

<svelte:window on:pointerdown={onWindowPointerDown} on:keydown|capture={onWindowKeydown} />

<div class="bar" bind:this={box}>
  <input
    bind:this={input}
    bind:value={draft}
    on:input={onInput}
    on:keydown={onKeydown}
    on:focus={() => (error = '')}
    readonly={busy}
    spellcheck="false"
    autocomplete="off"
    aria-label="Checkpoint path — edit to open another"
    title={busy ? 'opening…' : (draft || 'no checkpoint')}
    placeholder={$proxied ? 'path on the ssh proxy' : 'path, glob, or hf://owner/repo'}
  />

  {#if busy}
    <!-- A spinner only: the elapsed time belongs to the loading screen in the main area, which
         is already saying what is being read and for how long. Two clocks on one wait invites
         the reader to check whether they agree. -->
    <span class="working"><Spinner label="" /></span>
  {:else if dirty}
    <!-- Only while the text differs from what is served: a permanent Open button in the header
         would suggest the header does something on its own. -->
    <button class="go" type="button" title="Open this checkpoint (Enter)" on:click={() => void submit(draft)}
      >Open</button
    >
  {/if}

  {#if $recents.length > 1 || (options.length && !dirty)}
    <button
      class="caret"
      type="button"
      aria-expanded={listOpen}
      aria-label="Recently opened checkpoints"
      title="Recently opened (↓)"
      on:click={() => {
        listOpen = !listOpen;
        cursor = -1;
        confirming = null;
      }}>▾</button
    >
  {/if}

  {#if listOpen && options.length}
    <!-- Background fill, no border: the same treatment as the command palette and the filter
         builder, rather than a second idea of what a popup looks like. -->
    <ul class="menu" role="listbox" aria-label="Recently opened checkpoints">
      {#each options as spec, i (spec)}
        <li class="row" class:on={i === cursor}>
          {#if confirming === spec}
            <!-- Confirmation in the row itself, not a browser dialog: it keeps the path you are
                 about to forget in front of you, and the dropdown open behind it. -->
            <span class="ask">Forget <b>{short(spec)}</b>?</span>
            <button class="danger" type="button" on:click={() => void forget(spec)}>Forget</button>
            <button class="quiet" type="button" on:click={() => (confirming = null)}>Cancel</button>
          {:else}
            <button
              class="pick"
              type="button"
              role="option"
              aria-selected={i === cursor}
              title={spec}
              on:click={() => void submit(spec)}
              on:mouseenter={() => (cursor = i)}
            >
              <span class="name">{short(spec)}</span>
              <span class="path dim">{spec}</span>
            </button>
            <!-- A sibling button, not nested in the row's: nested buttons are invalid, and a click
                 on the cross must not also open the checkpoint. -->
            <button
              class="drop"
              type="button"
              title="Forget this checkpoint (removes it from the list only)"
              aria-label="Forget {short(spec)}"
              on:click={() => (confirming = spec)}>×</button
            >
          {/if}
        </li>
      {/each}
    </ul>
  {/if}

  {#if error}
    <p class="err" role="alert">{error}</p>
  {/if}
</div>

<style>
  .bar {
    position: relative;
    flex: 1 1 auto;
    min-width: 8ch;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  /* Reads as the dim path text it replaced until touched. */
  input {
    flex: 1 1 auto;
    min-width: 0;
    font: inherit;
    font-size: 12px;
    color: var(--fg-dim);
    background: none;
    border: none;
    border-radius: 4px;
    padding: 3px 6px;
    text-overflow: ellipsis;
  }
  input:hover:not(:read-only) {
    background: var(--bg-hover);
  }
  input:focus {
    outline: none;
    color: var(--fg);
    background: var(--bg-elev);
    text-overflow: clip;
  }
  /* Still selectable and copyable while a checkpoint loads; just not editable. */
  input:read-only {
    color: var(--fg-dim);
    cursor: default;
  }
  .working {
    flex: 0 0 auto;
    display: flex;
    align-items: baseline;
    gap: 5px;
    font-size: 12px;
  }
  .go {
    flex: 0 0 auto;
    font: inherit;
    font-size: 12px;
    color: var(--bg);
    background: var(--accent);
    border: none;
    border-radius: 4px;
    padding: 2px 8px;
    cursor: pointer;
  }
  .caret {
    flex: 0 0 auto;
    font: inherit;
    font-size: 11px;
    line-height: 1;
    color: var(--fg-dim);
    background: none;
    border: none;
    border-radius: 4px;
    padding: 3px 5px;
    cursor: pointer;
  }
  .caret:hover {
    background: var(--bg-hover);
    color: var(--fg);
  }
  .menu {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    z-index: 50;
    margin: 0;
    padding: 4px;
    list-style: none;
    min-width: 100%;
    max-width: 90vw;
    max-height: 50vh;
    overflow: auto;
    background: var(--bg-elev);
    border-radius: 6px;
    box-shadow: 0 6px 20px rgb(0 0 0 / 35%);
  }
  .row {
    display: flex;
    align-items: center;
    gap: 4px;
    border-radius: 4px;
  }
  .row.on,
  .row:hover {
    background: var(--bg-hover);
  }
  .menu button {
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: none;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    white-space: nowrap;
  }
  .menu .pick {
    display: flex;
    align-items: baseline;
    gap: 10px;
    flex: 1 1 auto;
    min-width: 0;
    text-align: left;
    padding: 4px 8px;
  }
  /* Dim until the row is under the cursor: a row of crosses reads as a list of delete buttons
     rather than a list of checkpoints. */
  .menu .drop {
    flex: 0 0 auto;
    padding: 2px 7px;
    color: transparent;
  }
  .row:hover .drop,
  .row.on .drop {
    color: var(--fg-dim);
  }
  .menu .drop:hover {
    color: var(--danger);
    background: var(--bg-elev);
  }
  .ask {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: 4px 8px;
    font-size: 12px;
  }
  .menu .danger {
    flex: 0 0 auto;
    color: var(--bg);
    background: var(--danger);
    padding: 2px 8px;
  }
  .menu .quiet {
    flex: 0 0 auto;
    color: var(--fg-dim);
    padding: 2px 8px;
  }
  .menu .quiet:hover {
    color: var(--fg);
    background: var(--bg-elev);
  }
  .menu .name {
    color: var(--accent);
    flex: 0 0 auto;
  }
  .menu .path {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* Under the box, like the dropdown — the header stays one row whatever goes wrong. */
  .err {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    z-index: 50;
    margin: 0;
    max-width: 90vw;
    font-size: 12px;
    color: var(--danger);
    background: var(--bg-elev);
    border-radius: 6px;
    padding: 5px 9px;
    box-shadow: 0 6px 20px rgb(0 0 0 / 35%);
  }
</style>
