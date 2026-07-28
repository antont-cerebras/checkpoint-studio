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
    {#if frac === null}
      <Spinner {label} />
    {:else}
      <span class="lbl">{label}</span>
    {/if}
    {#if elapsed}<span class="dim">({elapsed})</span>{/if}
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
    min-width: 34ch;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 8px;
  }
  .lbl {
    color: var(--accent);
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
