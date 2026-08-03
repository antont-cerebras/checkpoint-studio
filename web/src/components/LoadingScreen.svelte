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
  import { reading, watchReading, type SideProgress } from '../stores/reading';
  import LoadingBar from './LoadingBar.svelte';

  export let step: LoadStep;

  // The resolved address, not the `:` shorthand: only the server knows which host `:` names, and
  // it has told us (see `resolvedSpec`).
  $: subject = stepSubject(step, $proxyHost);
  $: detail = stepDetail(step, $proxyHost);

  // Poll the server's own read for exactly as long as this screen is up — `onMount` returning the
  // stopper is Svelte's own idiom for a subscription that lives with the component.
  onMount(() => watchReading());
  /** The server-side read's counts, for the steps that *are* a server-side read. The other steps are
   * this tab's own work, which the browser can already measure in bytes. */
  $: server = step.kind === 'opening' || step.kind === 'comparing' ? $reading : null;

  /**
   * A comparison reads its two checkpoints **at the same time**, so it gets a live row each.
   *
   * The rows come from the server's own answer, which says what it is reading and how far each side
   * has got. Until the first poll lands they come from the step, so the pair appears immediately
   * rather than a beat later — with no counters, which is honestly all that is known then.
   *
   * They used to be read one after the other, and this screen guessed which one was live by matching
   * the single spec the server reported against the two it drew. Two readers, two rows, two sets of
   * counters: nothing left to match up.
   */
  $: sides =
    step.kind === 'comparing'
      ? (server?.sides.length
          ? server.sides
          : [step.spec, step.right].filter(Boolean).map((spec) => ({
              spec,
              done: 0,
              total: 0,
              unit: '',
              stage: null,
              finished: false,
            })))
      : [];
  /** `baseline` / `candidate` — the pair as the screen above names it, by position. */
  const ROLE = ['baseline', 'candidate'];
  /** `44 / 66 shards`, or `44 shards` before the reader knows a total. Empty until it has counted
   * something: `0` with no unit says less than the timer does. */
  function counted(side: { done: number; total: number; unit: string }): string {
    if (side.done <= 0) return '';
    const of = side.total > 0 ? ` / ${side.total.toLocaleString()}` : '';
    return `${side.done.toLocaleString()}${of} ${side.unit}`.trim();
  }
  /** What to say after the count, or in place of it. */
  function say(side: { done: number; total: number; unit: string; finished: boolean }): string {
    if (side.finished) return 'read';
    return counted(side) || 'reading…';
  }
  /** The whole state line for a row: how far, and which step it is on. */
  function sideLine(side: SideProgress): string {
    const stage = side.finished ? '' : (side.stage ?? '');
    return stage ? `${say(side)} · ${stage}` : say(side);
  }
  /** `0..1` for a row's own bar, or `null` while there is no denominator to divide by. */
  function share(side: { done: number; total: number; finished: boolean }): number | null {
    if (side.finished) return 1;
    return side.total > 0 ? Math.min(1, side.done / side.total) : null;
  }

  /**
   * The single read's counts — an *open*, which is one checkpoint and needs no rows.
   *
   * Kept rendered for the whole read rather than only while it has values (**this is what stopped
   * the screen flickering**): a read moves through phases that count different things, and at every
   * boundary the count is briefly empty and the total briefly unknown. Rendering the bar only when
   * it had one took it out of the layout and put it back several times a read, jolting everything
   * below it.
   */
  $: one = step.kind === 'opening' ? (server?.sides[0] ?? null) : null;
  $: oneCounted = one ? counted(one) : '';
  $: oneShare = one ? share(one) : null;
</script>

<div class="screen">
  <LoadingBar label={stepLabel(step)} progress={step.progress} />
  <!-- Who is working and why it takes what it takes. The three steps drag for unrelated
       reasons — a slow disk on the server, a slow link to the browser, a large tally — and a
       wait you can attribute is one you can act on. -->
  <p class="detail dim">{detail}</p>
  {#if step.kind === 'opening'}
    <!-- What the server has got through, in the units the reader itself counts in. A synchronous open
         tells the browser nothing until it lands, so without this the only honest thing on screen was
         an elapsed timer — for a read that a terminal reports shard by shard. -->
    <div class="server">
      <!-- The rail is always drawn; the fill appears once there is a denominator. An empty rail is
           honest — the read has started and nobody has said how much there is yet — and it holds the
           four pixels so nothing below moves when the answer arrives. -->
      <div
        class="bar"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={oneShare === null ? undefined : Math.round(oneShare * 100)}
      >
        {#if oneShare !== null}<i style="width:{(oneShare * 100).toFixed(1)}%"></i>{/if}
      </div>
      <!-- The separator belongs to the pair, not to the stage: before the first count lands there is
           nothing to separate, and the line read `· loading the checkpoint index`. -->
      <p class="count dim">
        {oneCounted}{#if one?.stage}<span class="stage">{oneCounted ? ' · ' : ''}{one.stage}</span
          >{/if}
      </p>
    </div>
  {/if}
  {#if sides.length}
    <!-- One row per checkpoint, each with its own bar and its own count: they are read at the same
         time, on machines that answer at wildly different speeds — a local directory in a second, an
         `s3://` prefix in twenty — and one bar for the pair could describe neither. -->
    <ul class="sides">
      {#each sides as side, i (side.spec || i)}
        {@const frac = share(side)}
        <li class:done={side.finished}>
          <span class="what">{ROLE[i] ?? 'checkpoint'}</span>
          <span class="spec mono">{shortSpec(resolvedSpec(side.spec, $proxyHost), $proxyHost) || '(the open checkpoint)'}</span>
          <span class="rail" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={frac === null ? undefined : Math.round(frac * 100)}>
            {#if frac !== null}<i style="width:{(frac * 100).toFixed(1)}%"></i>{/if}
          </span>
          <!-- The separator is assembled with the sentence rather than written as markup in front of
               a block: Svelte trims the space before a mid-sentence `{#if}`, which read
               `9 / 30 shards· reading shard headers`. -->
          <span class="state dim">{sideLine(side)}</span>
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
  /* Each row's own bar — narrow, since the row already carries the words. */
  .rail {
    flex: 0 0 12ch;
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    overflow: hidden;
  }
  .rail i {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 200ms linear;
  }
  .sides li.done .rail i {
    background: var(--ok, var(--accent));
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
    align-items: center;
    gap: 8px;
    color: var(--fg);
  }
  .sides li.done {
    color: var(--fg-dim);
  }
  .sides .what {
    flex: 0 0 8ch;
    color: var(--fg-dim);
  }
  .sides .spec {
    min-width: 0;
    word-break: break-all;
  }
  .sides .spec {
    color: var(--accent);
  }
  .sides li.done .spec {
    color: var(--fg-dim);
  }
  /* Fixed width and one line: the count changes length as a phase changes, and a row that reflowed
     under the eye is the flicker this screen was reported for. */
  .sides .state {
    flex: 1 1 auto;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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
