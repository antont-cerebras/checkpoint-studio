<script lang="ts">
  // The one screen shown while there is no checkpoint content yet — whichever step is running.
  //
  // Replaces three near-identical panes that each said something different (see
  // `lib/loadstep.ts`). The bar itself is still `LoadingBar`, so a checkpoint wait looks like
  // every other wait in the app; what this adds is *which* step and *whose* work it is.
  import { onMount } from 'svelte';
  import {
    resolvedSpec,
    shortSpec,
    stepDetail,
    stepLabel,
    stepSubject,
    type LoadStep,
  } from '../lib/loadstep';
  import { proxyHost } from '../stores/server';
  import { reading, watchReading } from '../stores/reading';
  import LoadingBar from './LoadingBar.svelte';

  export let step: LoadStep;

  // The resolved address, not the `:` shorthand: only the server knows which host `:` names, and
  // it has told us (see `resolvedSpec`).
  $: subject = stepSubject(step, $proxyHost);
  $: detail = stepDetail(step, $proxyHost);

  // Poll the server's own read for exactly as long as this screen is up — `onMount` returning the
  // stopper is Svelte's own idiom for a subscription that lives with the component.
  onMount(() => watchReading());
  /** The server-side read's counts, for the two steps that *are* a server-side read. The other steps
   * are this tab's own work, which the browser can already measure in bytes. */
  $: server = step.kind === 'opening' || step.kind === 'comparing' ? $reading : null;
  /**
   * A comparison reads two checkpoints, one after the other — so it gets a row each.
   *
   * One bar named after the baseline said nothing about which of the two you were waiting for, and they
   * are not comparable: a local directory lands in a second, an `s3://` prefix in twenty. Which row is
   * live comes from the server's own answer (`reading.spec`), so it is the read that says so rather
   * than this screen guessing from the order.
   */
  $: sides =
    step.kind === 'comparing'
      ? [
          { label: 'baseline', spec: resolvedSpec(step.spec, $proxyHost) },
          { label: 'candidate', spec: resolvedSpec(step.right, $proxyHost) },
        ]
      : [];
  /**
   * Which of the two the server says it is reading; `-1` before it says.
   *
   * Both sides are put through `shortSpec` first, because the two ends spell the same checkpoint
   * differently: the server reports the spec it was *given* (`:/opt/…`, as typed) while this screen
   * resolves it for display (`host:/opt/…`). Comparing the two raw meant neither row ever lit up.
   */
  $: liveSide = sides.findIndex(
    (s) =>
      s.spec !== '' &&
      shortSpec(s.spec, $proxyHost) === shortSpec(server?.spec ?? '', $proxyHost),
  );
  /**
   * The furthest side seen being read, so a finished one says `read` rather than reverting to `waiting`.
   *
   * Between the server finishing a read and the aligned tree arriving there is nothing in flight, so
   * `liveSide` is `-1` — and without a high-water mark the row that had just been read went back to
   * looking like one that had not started.
   */
  let reached = -1;
  $: if (liveSide > reached) reached = liveSide;
  /** What to say about side `i`. */
  function sideState(i: number, live: number, high: number): 'live' | 'read' | 'waiting' {
    if (i === live) return 'live';
    return i <= high ? 'read' : 'waiting';
  }
  /** `44 / 66 shards`, or `44 shards` before the reader knows a total. Empty until it has counted
   * something: `0` with no unit says less than the timer does. */
  $: counted =
    server && server.done > 0
      ? server.total > 0
        ? `${server.done.toLocaleString()} / ${server.total.toLocaleString()} ${server.unit}`.trim()
        : `${server.done.toLocaleString()} ${server.unit}`.trim()
      : '';
  $: fraction = server && server.total > 0 ? Math.min(1, server.done / server.total) : null;
  /**
   * Whether to keep a place for the server's counts — for the whole of a server-side read, not only
   * while there is something in it.
   *
   * **This is what stopped the screen flickering.** A read moves through phases that count different
   * things (`listing` → `40/40 tensors` → `2/60 S3 objects` → the next checkpoint's `1/30 shards`),
   * and at every boundary the count is briefly empty and the total briefly unknown. Rendering the bar
   * and the count only when they had values took them out of the layout and put them back several
   * times a read, and everything below — the two checkpoint rows, one of which you are reading —
   * jumped 40 pixels each time. The panel is the same height throughout now, and the count fades in
   * where it will be.
   */
  $: serverStep = step.kind === 'opening' || step.kind === 'comparing';
</script>

<div class="screen">
  <LoadingBar label={stepLabel(step)} progress={step.progress} />
  <!-- Who is working and why it takes what it takes. The three steps drag for unrelated
       reasons — a slow disk on the server, a slow link to the browser, a large tally — and a
       wait you can attribute is one you can act on. -->
  <p class="detail dim">{detail}</p>
  <!-- What the server has got through, in the units the reader itself counts in. A synchronous open
       tells the browser nothing until it lands, so without this the only honest thing on screen was an
       elapsed timer — for a read that a terminal reports shard by shard. -->
  {#if serverStep}
    <div class="server">
      <!-- The rail is always drawn; the fill appears once there is a denominator. An empty rail is
           honest — the read has started and nobody has said how much there is yet — and it holds the
           four pixels so nothing below moves when the answer arrives. -->
      <div
        class="bar"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}
      >
        {#if fraction !== null}<i style="width:{(fraction * 100).toFixed(1)}%"></i>{/if}
      </div>
      <!-- The separator belongs to the pair, not to the stage: before the first count lands there is
           nothing to separate, and the line read `· loading the checkpoint index`. -->
      <p class="count dim">
        {counted}{#if server?.stage}<span class="stage">{counted ? ' · ' : ''}{server.stage}</span
          >{/if}
      </p>
    </div>
  {/if}
  {#if sides.length}
    <!-- One row per checkpoint. The one being read carries the count; the other says where it is in the
         queue, which is the question a single bar left unanswered. -->
    <ul class="sides">
      {#each sides as side, i (side.label)}
        {@const state = sideState(i, liveSide, reached)}
        <li class:live={state === 'live'} class:done={state === 'read'}>
          <span class="what">{side.label}</span>
          <span class="spec mono">{shortSpec(side.spec, $proxyHost) || '(the open checkpoint)'}</span>
          <!-- Where this side is, and nothing else. It used to repeat the count from the line above —
               the same `2 / 60 S3 objects` twice, in two places that changed width at different
               moments. The count belongs to the read; the row belongs to the side. -->
          <span class="state dim">
            {#if state === 'live'}
              reading…
            {:else if state === 'read'}
              read
            {:else}
              waiting
            {/if}
          </span>
        </li>
      {/each}
    </ul>
  {:else if subject}
    <!-- No `title`: the text is all there, so a tooltip repeating it is noise. -->
    <p class="subject mono">{subject}</p>
  {/if}
</div>

<style>
  .screen {
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: flex-start;
  }
  .detail {
    margin: 0;
    font-size: 12px;
  }
  /* The bar keeps the same 40ch cap as the one above it — two bars of different widths for one wait
     would read as two unrelated things — while the count beneath may run as long as it needs on its
     one line. */
  .server {
    display: flex;
    flex-direction: column;
    gap: 5px;
    width: fit-content;
    min-width: 40ch;
    max-width: 100%;
  }
  /* The same rail-and-fill as every other bar here. */
  .bar {
    width: 40ch;
    max-width: 100%;
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 200ms linear;
  }
  /* One line, always: a count that wrapped onto a second line as the phase changed
     (`2 / 60 S3 objects · reading S3 storage metadata`) shifted everything under it. */
  .count {
    margin: 0;
    min-height: 1.2em;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* A row per checkpoint: what it is, which one, and where it has got to. */
  .sides {
    list-style: none;
    margin: 4px 0 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 12px;
    max-width: 100%;
  }
  .sides li {
    display: flex;
    align-items: baseline;
    gap: 8px;
    color: var(--fg-dim);
  }
  .sides li.live {
    color: var(--fg);
  }
  .sides .what {
    flex: 0 0 8ch;
    color: var(--fg-dim);
  }
  .sides .spec {
    min-width: 0;
    word-break: break-all;
  }
  .sides li.live .spec {
    color: var(--accent);
  }
  .sides .state {
    flex: none;
    font-variant-numeric: tabular-nums;
  }
  .stage {
    /* Already dim; the step is context for the count rather than the point of the line. */
    opacity: 0.85;
  }
  /* Shown in full, wrapping if it must.
     This used to be one ellipsised line capped at 80ch, on the theory that a long path would
     otherwise set the width of the pane. The pane here is the whole screen, so there was nothing to
     protect and plenty of room — and what the ellipsis cut off was the *end* of the path, which is the
     part that says which checkpoint is being read (`…/Kimi-K2.6-3bit…`). A tooltip is no answer either:
     there is nothing to hover on a touch screen, and this is the one fact the screen exists to state. */
  .subject {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--fg-dim);
    max-width: 100%;
    word-break: break-all;
  }
</style>
