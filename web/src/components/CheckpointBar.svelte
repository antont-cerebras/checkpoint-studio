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
  import { loadRecents, proxied, recents, tree } from '../stores/server';
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

  onMount(loadRecents);

  $: root = $tree?.root ?? '';
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
        // Close the list first, revert second — two states, two presses, so Escape never
        // discards typing that the user only meant to un-list.
        if (listOpen) {
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

  function onInput() {
    dirty = draft !== root;
  }

  /** Click anywhere else closes the dropdown — the same dismiss rule the palette follows. */
  function onWindowPointerDown(e: Event) {
    if (listOpen && box && !box.contains(e.target as Node)) {
      listOpen = false;
      cursor = -1;
    }
  }

  const short = (s: string) => s.replace(/\/+$/, '').split('/').pop() || s;
</script>

<svelte:window on:pointerdown={onWindowPointerDown} />

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
      }}>▾</button
    >
  {/if}

  {#if listOpen && options.length}
    <!-- Background fill, no border: the same treatment as the command palette and the filter
         builder, rather than a second idea of what a popup looks like. -->
    <ul class="menu" role="listbox" aria-label="Recently opened checkpoints">
      {#each options as spec, i (spec)}
        <li>
          <button
            type="button"
            role="option"
            aria-selected={i === cursor}
            class:on={i === cursor}
            title={spec}
            on:click={() => void submit(spec)}
            on:mouseenter={() => (cursor = i)}
          >
            <span class="name">{short(spec)}</span>
            <span class="path dim">{spec}</span>
          </button>
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
  .menu button {
    display: flex;
    align-items: baseline;
    gap: 10px;
    width: 100%;
    text-align: left;
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: none;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    cursor: pointer;
    white-space: nowrap;
  }
  .menu button.on,
  .menu button:hover {
    background: var(--bg-hover);
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
