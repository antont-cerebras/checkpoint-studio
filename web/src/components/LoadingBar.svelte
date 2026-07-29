<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { humanSize } from '../lib/format';
  import { elapsedSeconds, fraction, type Progress } from '../lib/progress';
  import Spinner from './Spinner.svelte';

  export let label = '';
  export let progress: Progress | null = null;

  // The terminal's load screen ticks its elapsed time every frame; here a timer drives it,
  // since nothing else re-renders while a download is in flight.
  let now = 0;
  let timer: ReturnType<typeof setInterval> | undefined;
  onMount(() => {
    now = performance.now();
    timer = setInterval(() => (now = performance.now()), 100);
  });
  onDestroy(() => clearInterval(timer));

  $: frac = progress ? fraction(progress) : null;
  $: elapsed = progress ? elapsedSeconds(progress, now) : '';
</script>

<div class="load">
  <div class="head">
    <!-- With a known denominator the bar carries the progress and the spinner would be
         redundant motion; without one it is the only sign of life. Same rule as the
         terminal's load screen. -->
    <span class="lbl">{#if frac === null}<Spinner {label} />{:else}{label}{/if}</span>
    {#if elapsed}<span class="dim time">({elapsed})</span>{/if}
  </div>

  {#if progress && frac !== null && progress.total !== null}
    <div class="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(frac * 100)}>
      <i style="width:{(frac * 100).toFixed(1)}%"></i>
    </div>
    <div class="count dim">
      {humanSize(progress.received)} / {humanSize(progress.total)} · {Math.round(frac * 100)}%
    </div>
  {/if}
</div>

<style>
  .load {
    display: flex;
    flex-direction: column;
    gap: 6px;
    /* Capped, not stretched. A column flex fills its container, which on the full-screen
       load put a hairline bar across the whole window — the width said "this will take a
       while" while the bar itself was 4px tall. The TUI caps the same gauge at 30 cells
       for the same reason (`max_line` in ui::detail::render_line_gauge), and clamps it to
       the pane; `max-width` is that clamp. 40ch is measured, not guessed: the longest
       label and its timer — `⠋ reading checkpoint structure (12.4s)` — is 38 characters,
       and at 34ch it wrapped mid-phrase. */
    width: 40ch;
    max-width: 100%;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 8px;
  }
  /* One element in both states, so the truncation rule below covers the spinner branch too
     (the spinner renders the label inside itself). */
  .lbl {
    color: var(--accent);
    /* Stay on one line. Every label is a fixed phrase that fits the 40ch box, but the box is
       capped, so a label that outgrows it should lose its tail rather than push the timer
       onto a line of its own. `min-width: 0` is what lets a flex item shrink below its
       content at all. */
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* Never the item that gives way: the elapsed time is the whole point of the row. */
  .time {
    flex: 0 0 auto;
  }
  /* The same rail-and-fill the file browser's size bar uses, so progress looks like the
     one bar this app draws rather than a second idea of one. */
  .bar {
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 120ms linear;
  }
  .count {
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
</style>
