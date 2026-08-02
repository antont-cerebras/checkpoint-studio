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
    proxied,
    proxyHost,
    recents,
    reloadCheckpoint,
    tree,
  } from '../stores/server';
  import { specHelp } from '../lib/format';
  import { navigate, resetViewForNewCheckpoint } from '../stores/view';
  import RecentRow from './RecentRow.svelte';
  import CheckpointPicker from './CheckpointPicker.svelte';

  let draft = '';
  let error: string | null = null;
  let busy = false;

  // The list is also returned by every open, but this screen can be the first thing loaded
  // (`#open`, or the recovery button on a failed load), and then nothing has fetched it yet.
  onMount(loadRecents);

  /**
   * What this box accepts — the same sentence the two comparison boxes show.
   *
   * It used to promise less than they do ("path on the ssh proxy, or an s3:// prefix"), which reads as
   * though a glob or an `hf://` repo were not for this box. All three resolve through
   * `crate::opening::resolve` and accept exactly the same set, so a narrower hint was a narrower
   * *label*, not a narrower feature.
   */
  $: help = specHelp($proxied, $proxyHost ?? '');

  // Trailing slashes stripped on both sides: a recents entry is the spec as typed
  // (`…/Qwen3-Coder-30B-A3B-lut-3bit/`) while the root is the resolved directory (no slash),
  // so comparing them raw never matched and the badge never appeared.
  // Compared against the *address*, not the display root — a single-file checkpoint's root is
  // its directory, which no recents entry equals.
  $: current = trim($tree?.spec ?? $tree?.root ?? '');
  const trim = (s: string) => s.replace(/\/+$/, '');

  /** Enter in the box: open what it names. Named and typed rather than inline, because a callback
   * prop's parameter has no type inside the template. */
  function openSpec(spec: string) {
    void open(spec);
  }

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

</script>

<div class="open">
  <form class="pick" on:submit|preventDefault={() => open(draft)}>
    <label for="open-path">Open checkpoint</label>
    <!-- The same box as the header's and the comparison screen's (`CheckpointPicker`), with its
         dropdown off: every recent is listed underneath already, and the same entries in a popup over a
         list of them is one list too many. -->
    <CheckpointPicker
      id="open-path"
      bind:value={draft}
      {busy}
      menu={false}
      ariaLabel="Checkpoint to open"
      placeholder={help}
      title={help}
      onEnter={openSpec}
    />
    <button type="submit" disabled={!draft.trim() || busy}>Open</button>
  </form>

  <!-- No wait shown here: an open in flight is owned by the shared loading screen (App.svelte),
       which takes over the whole pane and names the step. This screen only reports failures. -->
  {#if error}
    <p class="err" role="alert">{error}</p>
  {/if}

  {#if $recents.length}
    <h3>Recent</h3>
    <ul class="recents">
      {#each $recents as spec (spec)}
        <li>
          <!-- The same row the address bar's dropdown uses, so both lists pick and forget alike. -->
          <RecentRow {spec} current={trim(spec) === current} busy={busy} onPick={open} />
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
  /* No cap on the screen itself. Measured at 668px, against 1572px for the *same* address box on the
     two comparison screens — so arriving here narrowed the app and leaving it widened it again, for a
     box holding the same kind of value: a checkpoint address, which is long, with a hint naming six
     accepted forms, which does not fit in 480px.
     What is capped instead is the prose and the list, which is what the old cap was really for — a
     line of text 1500px wide is a line you lose your place in. */
  .open {
    padding: 14px 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  /* The boxes are `TextField`, which owns their look — see that component. */
  .pick {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .pick label {
    color: var(--fg-dim);
    flex: 0 0 auto;
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
    /* A row is a path plus two small controls; stretching it across a 1600px window would put the
       controls a screen away from the name they act on. */
    max-width: 120ch;
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
