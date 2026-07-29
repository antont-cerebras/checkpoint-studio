<script lang="ts">
  // The open screen: read a different checkpoint into this server, without restarting it.
  //
  // Mirrors the terminal's palette command (`Open another checkpoint…`) — same accepted
  // spellings, same recents list, same wording, because both go through `crate::opening`.
  // The path box is the compare screen's, deliberately: they take the same kind of input and
  // looking different would suggest they don't.
  //
  // One checkpoint is served at a time, so this changes what every *other* browser tab sees
  // too. Said plainly below rather than left to be discovered.
  import { onMount } from 'svelte';
  import {
    loadRecents,
    openCheckpoint,
    openProgress,
    proxied,
    recents,
    reloadCheckpoint,
    tree,
  } from '../stores/server';
  import { navigate, resetViewForNewCheckpoint } from '../stores/view';
  import LoadingBar from './LoadingBar.svelte';

  let draft = '';
  let error: string | null = null;
  let busy = false;

  // The list is also returned by every open, but this screen can be the first thing loaded
  // (`#open`, or the recovery button on a failed load), and then nothing has fetched it yet.
  onMount(loadRecents);

  // Trailing slashes stripped on both sides: a recents entry is the spec as typed
  // (`…/Qwen3-Coder-30B-A3B-lut-3bit/`) while the root is the resolved directory (no slash),
  // so comparing them raw never matched and the badge never appeared.
  $: current = trim($tree?.root ?? '');
  const trim = (s: string) => s.replace(/\/+$/, '');

  async function open(spec: string) {
    const path = spec.trim();
    if (!path || busy) return;
    busy = true;
    error = null;
    try {
      await openCheckpoint(path);
      // Order matters, and only gets here on success:
      //   1. reset the view — so the incoming tree seeds its own fold state,
      //   2. navigate — so the tree is what's on screen while it loads,
      //   3. refetch — the new checkpoint's data.
      // Refetching first would land a tree before step 1 cleared the seed flag, and the new
      // checkpoint would open collapsed where the first opened expanded.
      resetViewForNewCheckpoint();
      navigate({ kind: 'tree' });
      await reloadCheckpoint();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      // A failure between the open and the refetch would leave the screen behind; come back
      // so the message is visible next to the box that produced it.
      navigate({ kind: 'open' });
    } finally {
      busy = false;
    }
  }

  /** The last path component, for a readable recents row; the full path is the title. */
  function short(spec: string): string {
    return trim(spec).split('/').pop() || spec;
  }
</script>

<div class="open">
  <form class="pick" on:submit|preventDefault={() => open(draft)}>
    <label for="open-path">Open checkpoint</label>
    <input
      id="open-path"
      bind:value={draft}
      placeholder={$proxied
        ? 'path on the ssh proxy, or an s3:// prefix'
        : 'file, directory, glob, or hf://owner/repo'}
      spellcheck="false"
      autocomplete="off"
    />
    <button type="submit" disabled={!draft.trim() || busy}>Open</button>
  </form>

  {#if busy}
    <!-- Timer only: the server reads shard headers and *then* answers, so a byte bar would
         sit at zero and jump. Same rule as the tensor scan. -->
    <LoadingBar label="reading the checkpoint" progress={$openProgress} />
  {:else if error}
    <p class="err" role="alert">{error}</p>
  {/if}

  {#if $recents.length}
    <h3>Recent</h3>
    <ul class="recents">
      {#each $recents as spec (spec)}
        <li>
          <button
            type="button"
            class="recent"
            title={spec}
            disabled={busy}
            on:click={() => open(spec)}
          >
            <span class="name">{short(spec)}</span>
            <span class="path dim">{spec}</span>
            {#if trim(spec) === current}<span class="badge">open</span>{/if}
          </button>
        </li>
      {/each}
    </ul>
  {/if}

  <p class="note dim">
    The server serves one checkpoint at a time, so this changes what every browser tab
    connected to it shows.
    {#if $proxied}
      This server reads over an ssh proxy, so paths are resolved there rather than on this
      machine.
    {/if}
  </p>
</div>

<style>
  .open {
    padding: 14px 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    /* The content is a path box and a short list; letting it run the full width of a wide
       window would leave the eye travelling for no reason. */
    max-width: 90ch;
  }
  .pick {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .pick label {
    color: var(--fg-dim);
    flex: 0 0 auto;
  }
  .pick input {
    flex: 1 1 auto;
    min-width: 0;
    font: inherit;
    padding: 5px 8px;
    color: var(--fg);
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  h3 {
    margin: 0;
    font-size: 12px;
    font-weight: 400;
    color: var(--fg-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .recents {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  /* Background fill rather than a box border, like the other lists in this UI. */
  .recent {
    display: flex;
    align-items: baseline;
    gap: 10px;
    width: 100%;
    text-align: left;
    font: inherit;
    color: var(--fg);
    background: none;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    cursor: pointer;
  }
  .recent:hover:not(:disabled) {
    background: var(--bg-hover);
  }
  .recent:disabled {
    cursor: default;
    opacity: 0.6;
  }
  .recent .name {
    color: var(--accent);
    flex: 0 0 auto;
  }
  .recent .path {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 12px;
  }
  .badge {
    flex: 0 0 auto;
    font-size: 11px;
    color: var(--bg);
    background: var(--accent);
    border-radius: 3px;
    padding: 0 5px;
  }
  .err {
    color: var(--danger);
    margin: 0;
    white-space: pre-wrap;
  }
  .note {
    margin: 0;
    font-size: 12px;
    max-width: 70ch;
  }
</style>
