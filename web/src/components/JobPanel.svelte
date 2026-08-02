<script lang="ts">
  // The value-reading comparisons: do the *numbers* differ, not just the structure?
  //
  // `--values`, `--histogram` and `--verify-repack` read every selected tensor on both sides, so the
  // server runs them as jobs and this starts, polls and stops them (`stores/jobs`). They were CLI-only
  // until now — the browser could tell you two checkpoints had the same shapes and nothing more.
  //
  // Deliberately next to the scope bar: these read *what the scope selected*, and reading 117k tensors
  // when nineteen were asked for is the mistake that pairing them prevents.
  import { humanSize } from '../lib/format';
  import { cancelJob, clearJob, job, jobError, startJob, type JobKind } from '../stores/jobs';
  import type { DiffScopeParams } from '../lib/diffscope';

  /** The two checkpoints, and the selection to apply. */
  export let left: string;
  export let right: string;
  export let scope: DiffScopeParams | undefined = undefined;

  let open = false;
  $: running = $job?.state === 'running';
  // A denominator only once the work knows one; until then a spinner, as every other wait here does.
  $: fraction = $job && $job.total > 0 ? Math.min(1, $job.done / $job.total) : null;
  $: verdict = $job?.findings.find((f) => f.kind === 'verdict');
  $: tensors = $job?.findings.filter((f) => f.kind === 'tensor') ?? [];

  const KINDS: { kind: JobKind; label: string; flag: string; hint: string }[] = [
    {
      kind: 'values',
      label: 'Compare values',
      flag: '--values',
      hint: 'Reads every selected tensor on both sides and reports how many elements differ.',
    },
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

  /** A number the way the CLI prints it. */
  const n = (v: number | undefined) => (v === undefined ? '—' : v.toLocaleString());
</script>

<div class="jobs">
  <div class="head">
    <button type="button" class="toggle" aria-expanded={open} on:click={() => (open = !open)}>
      <span class="caret">{open ? '▾' : '▸'}</span> Compare the data
    </button>
    {#if $job}
      <span class="state" class:bad={$job.state === 'failed'}>{$job.state}</span>
      <span class="dim tick">
        {n($job.done)}{$job.total > 0 ? ` / ${n($job.total)}` : ''}
        {#if $job.bytes > 0}· {humanSize($job.bytes)} read{/if}
        · {$job.elapsed_s.toFixed(1)}s
      </span>
      {#if running}
        <button type="button" class="quiet" on:click={() => void cancelJob()}>Stop</button>
      {:else}
        <button type="button" class="quiet" on:click={clearJob}>Clear results</button>
      {/if}
    {/if}
  </div>

  {#if open}
    <div class="acts">
      {#each KINDS as k (k.kind)}
        <button
          type="button"
          title={k.hint}
          disabled={running || !left || !right}
          on:click={() => void startJob(k.kind, left, right, scope)}
        >
          {k.label} <code>{k.flag}</code>
        </button>
      {/each}
      {#if !right}
        <span class="dim">Name both checkpoints above to compare their data.</span>
      {/if}
    </div>
  {/if}

  {#if $jobError}
    <p class="err" role="alert">{$jobError}</p>
  {/if}

  {#if $job}
    {#if running}
      <div class="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={fraction === null ? 0 : Math.round(fraction * 100)}>
        <!-- Indeterminate until a total is known: a bar pinned at zero for a minute reads as stuck. -->
        <i class:indeterminate={fraction === null} style={fraction === null ? '' : `width:${(fraction * 100).toFixed(1)}%`}></i>
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
          <span class="dim">{n(verdict.differ)} of {n(verdict.compared)} compared tensor(s) differ</span>
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
              <span class="dim">tvd {f.histogram.tvd.toFixed(4)} over {n(f.histogram.bins)} bins</span>
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
</div>

<style>
  /* Background fill, no border — as the scope bar and the other panels here. */
  .jobs {
    flex: 0 0 auto;
    padding: 6px 9px;
    border-radius: 4px;
    background: var(--bg-elev);
    font-size: 12px;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 9px;
    flex-wrap: wrap;
  }
  .toggle {
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
  }
  .caret {
    color: var(--fg-dim);
  }
  .state {
    color: var(--accent);
  }
  .state.bad {
    color: var(--danger);
  }
  .tick {
    font-variant-numeric: tabular-nums;
  }
  .quiet {
    margin-left: auto;
    color: var(--fg-dim);
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    font: inherit;
    font-size: 12px;
    padding: 2px 8px;
    cursor: pointer;
  }
  .acts {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
  }
  /* The same rail-and-fill every other progress bar in this app uses. */
  .bar {
    height: 4px;
    margin-top: 7px;
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
  /* No denominator yet: a moving sliver rather than an empty bar. */
  .bar i.indeterminate {
    width: 35%;
    animation: slide 1.4s ease-in-out infinite;
  }
  @keyframes slide {
    from {
      margin-left: -35%;
    }
    to {
      margin-left: 100%;
    }
  }
  .now,
  .verdict {
    margin: 5px 0 0;
  }
  .now {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .verdict strong {
    color: var(--warn);
  }
  .verdict strong.good {
    color: var(--ok);
  }
  .findings {
    margin: 6px 0 0;
    padding: 0;
    list-style: none;
    max-height: 40vh;
    overflow: auto;
  }
  .findings li {
    display: flex;
    gap: 10px;
    align-items: baseline;
    padding: 1px 0;
  }
  .nm {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .err {
    margin: 5px 0 0;
    color: var(--danger);
    white-space: pre-wrap;
  }
</style>
