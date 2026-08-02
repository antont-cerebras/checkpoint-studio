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
  import { loadRecents, proxied, proxyHost, tree } from '../stores/server';
  import { specHelp } from '../lib/format';
  import { switchCheckpoint } from '../stores/open';
  import CheckpointPicker from './CheckpointPicker.svelte';
  import Spinner from './Spinner.svelte';

  /**
   * The picker, as *what this bar asks of it*.
   *
   * A structural type rather than the component class: a `.svelte` import carries no type outside
   * svelte-check, so `picker.blur()` would be a call on `any` — and an `any` here is how
   * `picker?.blur?.()` came to call a method that did not exist at all.
   */
  let picker: { blur: () => void } | undefined;
  let draft = '';
  let busy = false;
  let error = '';

  onMount(loadRecents);

  /** Escape with the menu closed: put back what is being served, and stop editing. The header should
   * never be left showing a path that is not the checkpoint behind it. */
  function revert() {
    draft = root;
    error = '';
  }

  // The *address*, not the display root: for a single-file checkpoint the root is its containing
  // directory, and opening that would read a different checkpoint (a directory of three HDF5
  // files has one root and three addresses).
  $: root = $tree?.spec ?? $tree?.root ?? '';
  // Follow the served checkpoint, unless the box has been edited — otherwise a switch made somewhere
  // else (the open screen, another tab's doing) would leave the header lying.
  //
  // Tracked against what was last *shown* rather than derived from `draft !== root`: the derived form
  // is a cycle (`draft → dirty → draft`), and it is the same mistake the comparison boxes made — a
  // statement that both reads and writes the value it is about.
  //
  // A field of an object rather than a plain `let`: what is stored is the value of the *previous* run,
  // which no static analysis can see being read (`no-useless-assignment` says so), and a bare
  // suppression would leave the next reader wondering whether the assignment matters.
  const applied = { root: '' };
  $: if (root !== applied.root && !busy) {
    applied.root = root;
    draft = root;
  }
  /** Edited away from what is served — which is when the Open button appears. Declared by the reactive
   * statement, since any initial value here would be overwritten before it could be read. */
  $: dirty = draft.trim() !== '' && draft.trim() !== root;


  /** Enter, or a recent chosen from the dropdown: both mean "open this". Named and typed rather than
   * inline, because a callback prop's parameter has no type inside the template. */
  function openSpec(spec: string) {
    void submit(spec);
  }

  async function submit(spec: string) {
    const path = spec.trim();
    if (!path || busy) return;
    // Re-opening what is already served would re-read a 31k-tensor checkpoint to arrive back
    // where we started; treat it as "done" instead.
    if (path === root) {
      picker?.blur();
      return;
    }
    busy = true;
    error = '';
    try {
      await switchCheckpoint(path);
      draft = path;
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






</script>

<div class="bar">
  <!-- The one checkpoint box (`CheckpointPicker`): the same field, dropdown, keys and forget flow as
       the two on the comparison screen. `bare` because here the box *is* the header rather than a
       control sitting on it. Choosing a recent here means "open it", which is what `onPickRecent` says;
       elsewhere it fills the box. -->
  <CheckpointPicker
    id="ckpt-path"
    variant="bare"
    bind:this={picker}
    bind:value={draft}
    on:focus={() => (error = '')}
    {busy}
    ariaLabel="Checkpoint path — edit to open another"
    title={busy ? 'opening…' : draft || 'no checkpoint'}
    placeholder={specHelp($proxied, $proxyHost ?? '')}
    onEnter={openSpec}
    onPickRecent={openSpec}
    onEscape={revert}
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
  /* The field's own look lives in `TextField` (variant `bare`, which reads as the dim path text it
     replaced until touched) — it was eight declarations repeated in six components, drifting by a
     pixel of padding and a shade of background. */
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
