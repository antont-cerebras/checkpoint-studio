<script lang="ts">
  // One entry in a recents list: pick it, or forget it after confirming.
  //
  // Shared by the address bar's dropdown and the open screen's list. Extracted rather than copied
  // because both need the same two-step confirmation, and two copies of a confirmation flow is two
  // places for "does this delete immediately?" to diverge.
  import { forgetRecent } from '../stores/server';
  import { checkpointLabel } from '../lib/format';

  export let spec: string;
  /** Marked as the one being served. */
  export let current = false;
  /** Highlighted by the keyboard (the dropdown's cursor). */
  export let active = false;
  /** Disabled while something else is loading. */
  export let busy = false;
  export let onPick: (spec: string) => void;

  /** Asking "forget this?" — one row at a time, so the answer is never ambiguous. */
  let confirming = false;
  let error = '';

  // The shared rule, not a local `split('/').pop()`: that called every `s3://…/checkpoint`
  // "checkpoint" — see `checkpointLabel`, which Rust's `model::checkpoint_label` is contracted with.
  const short = checkpointLabel;

  async function forget() {
    confirming = false;
    try {
      await forgetRecent(spec);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }

  /** Escape answers the question, wherever focus is — the cross holds it after a click. */
  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && confirming) {
      e.preventDefault();
      e.stopPropagation();
      confirming = false;
    }
  }
</script>

<svelte:window on:keydown|capture={onKeydown} />

<div class="row" class:active>
  {#if confirming}
    <!-- Confirmation in the row itself, not a browser dialog: it keeps the path you are about to
         forget in front of you, and the list open behind it. -->
    <span class="ask">Forget <b>{short(spec)}</b>?</span>
    <button class="danger" type="button" on:click={forget}>Forget</button>
    <button class="quiet" type="button" on:click={() => (confirming = false)}>Cancel</button>
  {:else}
    <button
      class="pick"
      type="button"
      title={spec}
      disabled={busy}
      on:click={() => onPick(spec)}
    >
      <span class="name">{short(spec)}</span>
      <span class="path dim">{spec}</span>
      {#if current}<span class="badge">open</span>{/if}
    </button>
    <button
      class="drop"
      type="button"
      title="Forget this checkpoint (removes it from the list only)"
      aria-label="Forget {short(spec)}"
      disabled={busy}
      on:click={() => (confirming = true)}>×</button
    >
  {/if}
</div>
{#if error}<p class="err" role="alert">{error}</p>{/if}

<style>
  .row {
    display: flex;
    align-items: center;
    gap: 4px;
    border-radius: 4px;
  }
  .row.active,
  .row:hover {
    background: var(--bg-hover);
  }
  button {
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: none;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    white-space: nowrap;
  }
  .pick {
    display: flex;
    align-items: baseline;
    gap: 10px;
    flex: 1 1 auto;
    min-width: 0;
    text-align: left;
    padding: 4px 8px;
  }
  .name {
    color: var(--accent);
    flex: 0 0 auto;
  }
  .path {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .badge {
    flex: 0 0 auto;
    font-size: 11px;
    color: var(--bg);
    background: var(--accent);
    border-radius: 3px;
    padding: 0 5px;
  }
  /* Dim until the row is under the cursor: a column of crosses reads as a list of delete buttons
     rather than a list of checkpoints. */
  .drop {
    flex: 0 0 auto;
    padding: 2px 7px;
    color: transparent;
  }
  .row:hover .drop,
  .row.active .drop {
    color: var(--fg-dim);
  }
  .drop:hover {
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
  .danger {
    flex: 0 0 auto;
    color: var(--bg);
    background: var(--danger);
    padding: 2px 8px;
  }
  .quiet {
    flex: 0 0 auto;
    color: var(--fg-dim);
    padding: 2px 8px;
  }
  .quiet:hover {
    color: var(--fg);
    background: var(--bg-elev);
  }
  .err {
    margin: 2px 8px;
    font-size: 12px;
    color: var(--danger);
  }
</style>
