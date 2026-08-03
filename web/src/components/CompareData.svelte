<script lang="ts">
  /**
   * Comparing the **data**: do the numbers differ, not just the structure?
   *
   * `--values`, `--histogram` and `--verify-repack` read every selected tensor on both sides, so the
   * server runs them as jobs and this starts, polls and stops them (`stores/jobs`).
   *
   * **Why it is a view and not a panel.** These lived in a collapsed strip called *Compare the data*
   * on one of the two diff screens, while the other screen — the one most people land on — said the
   * numeric comparison had to be run in a terminal. Both statements were about the same pair, and one
   * of them was false. It is a reading of the comparison, like the summary and the tree, so it is a
   * view beside them, and the terminal command is offered as an alternative rather than as the only
   * way.
   *
   * The scope applies: reading 117k tensors when nineteen were asked for is the mistake that showing
   * the count and the bytes *before* starting is meant to prevent.
   */
  import { humanCount, humanSize } from '../lib/format';
  import { cancelJob, clearJob, job, jobError, startJob, type JobKind } from '../stores/jobs';
  import { diffTree } from '../stores/compare';
  import { diffReport } from '../stores/report';
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import type { DiffScopeParams } from '../lib/diffscope';
  import { copyText } from '../lib/clipboard';

  /** The two checkpoints, and the selection to apply. */
  export let left: string;
  export let right: string;
  export let scope: DiffScopeParams | undefined = undefined;

  $: running = $job?.state === 'running';
  // A denominator only once the work knows one; until then a spinner, as every other wait here does.
  $: fraction = $job && $job.total > 0 ? Math.min(1, $job.done / $job.total) : null;
  $: verdict = $job?.findings.find((f) => f.kind === 'verdict');
  $: tensors = $job?.findings.filter((f) => f.kind === 'tensor') ?? [];

  /**
   * What this run would read, from the comparison already on screen.
   *
   * The totals follow the scope (the server labels them so), so this is the honest size of the work:
   * both sides, every selected tensor. Shown before the button rather than after the wait, because a
   * minute of reading is a decision and `27 GB` is what makes it one.
   */
  //
  // From whichever result is loaded: the summary is the cheap one and the view most people arrive
  // from, so sizing the run must not depend on having fetched the 91 MB aligned tree first.
  /** Why the values cannot be compared, from the server — `''` when they can. */
  $: valuesNote = $diffReport?.values_note ?? '';
  $: matched = $diffTree?.matched ?? $diffReport?.matched ?? null;
  $: selected = matched?.selected ?? $diffTree?.base.tensor_count ?? reportTensors($diffReport);
  $: bytes =
    $diffTree !== null
      ? $diffTree.base.bytes + $diffTree.current.bytes
      : ($diffReport?.report.old_bytes ?? 0) + ($diffReport?.report.new_bytes ?? 0);

  /** How many tensors a report covers: the ones present on both sides plus the one-sided ones — the
   * same set the value comparison would walk. */
  function reportTensors(r: typeof $diffReport): number {
    if (!r) return 0;
    return (
      r.report.tensors_unchanged +
      r.report.tensors_changed.length +
      r.report.tensors_added.length +
      r.report.tensors_removed.length
    );
  }

  /** The two runs that are not the common one. */
  const MORE: { kind: JobKind; label: string; flag: string; hint: string }[] = [
    {
      kind: 'histogram',
      label: 'Compare distributions',
      flag: '--histogram',
      hint: 'Bins both sides over a shared layout and reports the total variation distance.',
    },
    {
      kind: 'verify-repack',
      label: 'Verify repack',
      flag: '--verify-repack',
      hint: 'Are these the same weights in different packings? Decodes the packed indices on both sides.',
    },
  ];
  let moreOpen = false;

  /** A number the way the CLI prints it. */
  const n = (v: number | undefined) => (v === undefined ? '—' : v.toLocaleString());

  /**
   * The elapsed time, on its own clock.
   *
   * The server measures it and every poll carries a fresh figure — but a poll is twice a second, so the
   * timer moved in half-second steps *with* the counters and read as frozen whenever they were: exactly
   * when a reader is asking whether anything is still happening. So the server's figure is a baseline
   * and this adds the wall-clock time since that answer arrived. It stays the server's measure — the
   * clock only fills the gaps between its updates — and it stops when the job does.
   */
  let ticking = 0;
  // Fields of an object, so the reactive block below can *read the previous run's* baseline without
  // Svelte treating it as a dependency — and without an assignment that would re-trigger the block.
  const seen = { at: 0, elapsed: 0 };
  $: if ($job && $job.elapsed_s !== seen.elapsed) {
    // A new answer: re-baseline. This is also what lands the timer exactly on the server's final figure
    // when the run ends, since the last poll carries it and the clock below has stopped by then.
    seen.elapsed = $job.elapsed_s;
    seen.at = performance.now();
    ticking = $job.elapsed_s;
  }
  // One interval for the component's life rather than one started and stopped by a reactive statement —
  // which would assign the handle it reads, and is the shape `svelte/infinite-reactive-loop` warns about.
  onMount(() => {
    const id = setInterval(() => {
      if (running) ticking = seen.elapsed + (performance.now() - seen.at) / 1000;
    }, 100);
    return () => clearInterval(id);
  });

  /**
   * The equivalent command, for a terminal — an alternative, not the only way.
   *
   * **Asked for, not assembled.** This line used to be built here out of the two addresses, which meant
   * it silently dropped the entire selection: a comparison scoped to one tensor with a fused alignment
   * offered `diff --values OLD NEW`, a command that compares every tensor of both checkpoints,
   * unaligned. A string built beside the state it describes is always one control behind it, so the
   * server renders it from the same parameter table that decides which parameters exist
   * (`GET /api/command`, `src/web/params.rs`).
   */
  let command = '';
  $: void loadCommand(left, right, scope);
  async function loadCommand(l: string, r: string, s: DiffScopeParams | undefined) {
    if (!l || !r) {
      command = '';
      return;
    }
    try {
      command = (await api.command(l, r, s, 'values')).command ?? '';
    } catch {
      // The command is an alternative, not the answer: if it cannot be rendered, the panel simply does
      // not offer one, and the run itself reports its own failures.
      command = '';
    }
  }
  let copied = false;
  function copy() {
    if (copyText(command)) {
      copied = true;
      setTimeout(() => (copied = false), 1500);
    }
  }
</script>

<div class="data">
  {#if !left || !right}
    <p class="dim">Name both checkpoints above to compare their data.</p>
  {:else}
    <!-- **Say it before the button, not after the wait.** A remote checkpoint serves its structure and
         not its bytes, so a value comparison over one cannot happen — and used to find that out having
         already read both sides. The server answers from the two addresses (`values_note`), so the
         reason is on screen and the action that cannot work is not offered. -->
    {#if valuesNote}
      <p class="method">{valuesNote}</p>
    {/if}
    <div class="acts">
      <!-- One primary action: the question almost everyone has is "are the weights the same". -->
      <button
        type="button"
        class="go"
        disabled={running || !!valuesNote}
        title={valuesNote ||
          'Reads every selected tensor on both sides and reports how many elements differ (--values).'}
        on:click={() => void startJob('values', left, right, scope)}
      >
        Compare tensor values
      </button>
      <div class="more">
        <button
          type="button"
          class="quiet"
          aria-expanded={moreOpen}
          on:click={() => (moreOpen = !moreOpen)}>More checks ▾</button
        >
        {#if moreOpen}
          <ul class="menu">
            {#each MORE as k (k.kind)}
              <!-- Distributions read the bytes as well, so they are out for the same reason; a repack
                   verification decodes on the proxy, which is why it stays available and is what the
                   note above points a remote pair at. -->
              {@const blocked = k.kind === 'histogram' ? valuesNote : ''}
              <li>
                <button
                  type="button"
                  title={blocked || k.hint}
                  disabled={running || !!blocked}
                  on:click={() => {
                    moreOpen = false;
                    void startJob(k.kind, left, right, scope);
                  }}
                >
                  {k.label} <code>{k.flag}</code>
                </button>
              </li>
            {/each}
          </ul>
        {/if}
      </div>
      {#if running}
        <button type="button" class="quiet" on:click={() => void cancelJob()}>Stop</button>
      {:else if $job}
        <button type="button" class="quiet" on:click={clearJob}>Clear results</button>
      {/if}
    </div>

    <!-- What it will cost, before it is started. -->
    <p class="cost dim">
      {humanCount(selected)} tensor{selected === 1 ? '' : 's'} selected · about {humanSize(bytes)} to
      read across both sides{matched ? ` (of ${humanCount(matched.total)} in the checkpoints)` : ''}
    </p>
  {/if}

  {#if $jobError}
    <p class="err" role="alert">{$jobError}</p>
  {/if}

  {#if $job}
    <div class="state">
      <span class:bad={$job.state === 'failed'}>{$job.state}</span>
      <span class="dim tick">
        {n($job.done)}{$job.total > 0 ? ` / ${n($job.total)}` : ''}
        {#if $job.bytes > 0}· {humanSize($job.bytes)} read{/if}
        · {ticking.toFixed(1)}s
      </span>
    </div>
    {#if running}
      <div
        class="bar"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={fraction === null ? 0 : Math.round(fraction * 100)}
      >
        <!-- Indeterminate until a total is known: a bar pinned at zero for a minute reads as stuck. -->
        <i
          class:indeterminate={fraction === null}
          style={fraction === null ? '' : `width:${(fraction * 100).toFixed(1)}%`}
        ></i>
      </div>
      {#if $job.current}<p class="now dim mono" title={$job.current}>{$job.current}</p>{/if}
    {/if}

    {#if $job.error}
      <p class="err" role="alert">{$job.error}</p>
    {/if}

    {#if verdict}
      <p class="verdict">
        {#if verdict.equivalent !== undefined}
          <!-- verify-repack: the one answer the whole run exists to give. -->
          <strong class:good={verdict.equivalent}>
            {verdict.equivalent
              ? 'Equivalent — the same weights in different packings'
              : 'Not equivalent — the decoded indices differ'}
          </strong>
          <span class="dim">
            {n(verdict.pairs)} pair{verdict.pairs === 1 ? '' : 's'} at {n(verdict.bits)}-bit{verdict.other_differs
              ? ' · something outside the verified pairs also differs'
              : ''}
          </span>
        {:else}
          <strong>{verdict.verdict ?? ''}</strong>
          <span class="dim"
            >{n(verdict.differ)} of {n(verdict.compared)} compared tensor(s) differ</span
          >
        {/if}
      </p>
    {/if}

    {#if tensors.length}
      <ul class="findings">
        {#each tensors as f (f['name'])}
          <li>
            <span class="mono nm" title={f.name}>{f.name}</span>
            {#if f.values}
              <span class="dim">{n(f.values.differing)} of {n(f.values.elements)} differ</span>
              <span class="dim">max |Δ| {f.values.max_abs}</span>
            {/if}
            {#if f.histogram}
              <span class="dim"
                >tvd {f.histogram.tvd.toFixed(4)} over {n(f.histogram.bins)} bins</span
              >
            {/if}
            {#if f.max_delta !== undefined}
              <!-- verify-repack counts decoded *indices*, not element values. -->
              <span class="dim">{n(f.differing)} of {n(f.elements)} indices differ</span>
              <span class="dim">max Δ {n(f.max_delta)}</span>
            {/if}
          </li>
        {/each}
      </ul>
    {/if}
  {/if}

  {#if left && right && command}
    <!-- The same run, for a terminal — for a pipeline, or for somewhere this tab is not. -->
    <p class="alt dim">Or run it in a terminal:</p>
    <div class="cmd">
      <code>{command}</code>
      <button type="button" class="quiet" on:click={copy}>{copied ? '✓ copied' : 'copy'}</button>
    </div>
  {/if}
</div>

<style>
  /* A note about what can be done, not a result — the shape the report gives its own method notes. */
  .method {
    margin: 0 0 8px;
    padding: 5px 9px;
    border-radius: 4px;
    background: var(--bg-elev);
    color: var(--fg-dim);
    font-size: 12px;
  }
  .data {
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
    padding-top: 8px;
    font-size: 12px;
  }
  .acts {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .go {
    font: inherit;
    font-size: 12.5px;
    color: var(--fg);
    background: var(--bg-elev);
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 5px 12px;
    cursor: pointer;
  }
  .go:hover:not(:disabled) {
    background: var(--bg-hover);
  }
  .quiet {
    color: var(--fg-dim);
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    font: inherit;
    font-size: 12px;
    padding: 4px 9px;
    cursor: pointer;
  }
  /* The secondary runs, behind one press — background fill, no border, like every popup here. */
  .more {
    position: relative;
  }
  .menu {
    position: absolute;
    z-index: 20;
    top: calc(100% + 4px);
    left: 0;
    margin: 0;
    padding: 4px;
    list-style: none;
    min-width: 26ch;
    background: var(--bg-elev);
    border-radius: 6px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
  }
  .menu button {
    display: block;
    width: 100%;
    text-align: left;
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: none;
    border: none;
    padding: 5px 8px;
    border-radius: 4px;
    cursor: pointer;
  }
  .menu button:hover:not(:disabled) {
    background: var(--bg-hover);
  }
  .cost {
    margin: 6px 0 0;
  }
  .state {
    display: flex;
    align-items: baseline;
    gap: 9px;
    margin-top: 10px;
  }
  .bad {
    color: var(--danger);
  }
  .tick {
    font-variant-numeric: tabular-nums;
  }
  .bar {
    height: 4px;
    margin: 6px 0;
    background: var(--bg-panel);
    border-radius: 2px;
    overflow: hidden;
  }
  .bar i {
    display: block;
    height: 100%;
    background: var(--accent);
  }
  .bar i.indeterminate {
    width: 30%;
    animation: slide 1.2s ease-in-out infinite;
  }
  @keyframes slide {
    from {
      margin-left: -30%;
    }
    to {
      margin-left: 100%;
    }
  }
  .now {
    margin: 2px 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .verdict {
    margin: 8px 0 2px;
    display: flex;
    gap: 8px;
    align-items: baseline;
    flex-wrap: wrap;
  }
  .verdict strong {
    color: var(--warn);
  }
  .verdict strong.good {
    color: var(--ok, #3fb950);
  }
  .err {
    margin: 6px 0 0;
    color: var(--danger);
    white-space: pre-wrap;
  }
  .findings {
    margin: 6px 0 0;
    padding: 0;
    list-style: none;
  }
  .findings li {
    display: flex;
    gap: 10px;
    align-items: baseline;
    padding: 1px 0;
  }
  .nm {
    flex: 0 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .alt {
    margin: 14px 0 4px;
  }
  .cmd {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .cmd code {
    flex: 1 1 auto;
    overflow-x: auto;
    white-space: nowrap;
    padding: 5px 8px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
</style>
