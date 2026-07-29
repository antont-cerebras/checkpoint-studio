<script lang="ts">
  // The compare screen: a structural diff of the served checkpoint against another one
  // on the server's filesystem. Mirrors the terminal UI's `ui/diff.rs` — same sections in
  // the same order, same +/-/~ markers, same green/red/yellow, same one-line verdict (the
  // server computes that string, so the two UIs cannot word it differently).
  //
  // Structure only: names, dtypes and shapes. The value comparison reads every byte of
  // both checkpoints, so it stays on the CLI where it has a progress bar — the footer
  // hands over the exact command, extended with --values.
  import { api } from '../lib/api';
  import { humanCount, humanSize } from '../lib/format';
  import type { DiffResponse, TensorSig } from '../lib/types';
  import { navigate } from '../stores/view';
  import { copyText } from '../lib/clipboard';
  import LoadingBar from './LoadingBar.svelte';
  import { startedNow, type Progress } from '../lib/progress';
  // The server reads the baseline checkpoint's headers before it can answer.
  let waitStarted: Progress | null = null;

  export let against: string;

  let result: DiffResponse | null = null;
  let error: string | null = null;
  let loading = false;
  /** The path in the input, which may differ from the one being shown. */
  let draft = against;
  let copied = false;

  // Re-run whenever the URL's baseline changes (including back/forward), not just on
  // mount — the screen is addressable, so arriving at it twice with different paths has
  // to show different reports.
  $: void load(against);

  async function load(path: string) {
    if (!path) return;
    draft = path;
    loading = true;
    waitStarted = startedNow();
    error = null;
    try {
      result = await api.diff(path);
    } catch (e) {
      result = null;
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  function submit() {
    const p = draft.trim();
    // Navigating rather than loading directly keeps the URL the source of truth, so the
    // comparison is shareable and survives a reload.
    if (p) navigate({ kind: 'diff', against: p });
  }

  function sig(s: TensorSig): string {
    return `[${s.dtype}, (${s.shape.join(', ')})]`;
  }

  /** How a changed tensor changed — dtype, shape, or both. */
  function change(o: TensorSig, n: TensorSig): string {
    const dtype = o.dtype !== n.dtype;
    const shape = o.shape.join() !== n.shape.join();
    if (dtype && shape) return `${sig(o)} → ${sig(n)}`;
    if (dtype) return `dtype ${o.dtype} → ${n.dtype}`;
    if (shape) return `shape (${o.shape.join(', ')}) → (${n.shape.join(', ')})`;
    return 'values differ';
  }

  function copyCommand() {
    if (result && copyText(result.command)) {
      copied = true;
      setTimeout(() => (copied = false), 1500);
    }
  }

  $: report = result?.report ?? null;
  $: metaTotal = report
    ? report.meta_added.length + report.meta_removed.length + report.meta_changed.length
    : 0;
</script>

<div class="diff">
  <form
    class="pick"
    on:submit|preventDefault={submit}
  >
    <label for="diff-against">Compare with</label>
    <input
      id="diff-against"
      bind:value={draft}
      placeholder="path to a checkpoint file, directory, or glob on the server"
      spellcheck="false"
    />
    <button type="submit" disabled={!draft.trim() || draft.trim() === against}>Compare</button>
  </form>

  {#if loading}
    <LoadingBar label="reading {against}" progress={waitStarted} />
  {:else if error}
    <p class="error" role="alert">{error}</p>
  {:else if result && report}
    <header>
      <div class="sides">
        <span class="side old">old</span><span class="path">{result.against}</span>
      </div>
      <div class="sides">
        <span class="side new">new</span><span class="path">{$$props.root ?? 'this checkpoint'}</span>
      </div>
      <p class="verdict">{result.verdict}</p>
      <p class="delta dim">
        {#if report.old_bytes === report.new_bytes}
          {humanSize(report.new_bytes)} (unchanged)
        {:else}
          {humanSize(report.old_bytes)} → {humanSize(report.new_bytes)}
        {/if}
        ·
        {#if report.old_params === report.new_params}
          {humanCount(report.new_params)} params
        {:else}
          {humanCount(report.old_params)} → {humanCount(report.new_params)} params
        {/if}
      </p>
    </header>

    <section>
      <h3 class="added">Tensors added ({report.tensors_added.length})</h3>
      {#if !report.tensors_added.length}<p class="none dim">none</p>{/if}
      {#each report.tensors_added as [name, s] (name)}
        <div class="row"><span class="mark added">+</span><span class="name">{name}</span
          ><span class="detail dim">{sig(s)}</span></div>
      {/each}
    </section>

    <section>
      <h3 class="removed">Tensors removed ({report.tensors_removed.length})</h3>
      {#if !report.tensors_removed.length}<p class="none dim">none</p>{/if}
      {#each report.tensors_removed as [name, s] (name)}
        <div class="row"><span class="mark removed">-</span><span class="name">{name}</span
          ><span class="detail dim">{sig(s)}</span></div>
      {/each}
    </section>

    <section>
      <h3 class="changed">Tensors changed ({report.tensors_changed.length})</h3>
      {#if !report.tensors_changed.length}<p class="none dim">none</p>{/if}
      {#each report.tensors_changed as c (c.name)}
        <div class="row"><span class="mark changed">~</span><span class="name">{c.name}</span
          ><span class="detail dim">{change(c.old, c.new)}</span></div>
      {/each}
    </section>

    <p class="none dim">{report.tensors_unchanged} tensors unchanged</p>

    <section>
      <h3 class="meta">Metadata ({metaTotal})</h3>
      {#if !metaTotal}<p class="none dim">none</p>{/if}
      {#each report.meta_added as [name, v] (name)}
        <div class="row"><span class="mark added">+</span><span class="name">{name}</span
          ><span class="detail dim">{v.value}</span></div>
      {/each}
      {#each report.meta_removed as [name, v] (name)}
        <div class="row"><span class="mark removed">-</span><span class="name">{name}</span
          ><span class="detail dim">{v.value}</span></div>
      {/each}
      {#each report.meta_changed as c (c.name)}
        <div class="row"><span class="mark changed">~</span><span class="name">{c.name}</span
          ><span class="detail dim">{c.old.value} → {c.new.value}</span></div>
      {/each}
      {#if report.meta_unchanged}
        <p class="none dim">{report.meta_unchanged} metadata entries unchanged</p>
      {/if}
    </section>

    <footer>
      <p class="dim">
        Structure only. To compare the numbers, run this in a terminal with
        <code>--values</code>:
      </p>
      <div class="cmd">
        <code>{result.command}</code>
        <button on:click={copyCommand}>{copied ? '✓ copied' : 'copy'}</button>
      </div>
    </footer>
  {/if}
</div>

<style>
  .diff {
    padding: 10px 14px;
    overflow: auto;
  }
  .pick {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }
  .pick label {
    font-size: 12px;
    color: var(--fg-dim);
  }
  .pick input {
    flex: 1;
    min-width: 0;
    padding: 4px 8px;
    font: inherit;
    font-size: 12.5px;
    color: var(--fg);
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 3px;
  }
  header {
    margin-bottom: 14px;
  }
  .sides {
    display: flex;
    align-items: baseline;
    gap: 8px;
    font-size: 12.5px;
  }
  .side {
    flex: none;
    width: 2.2em;
    font-weight: 600;
  }
  .path {
    word-break: break-all;
  }
  .verdict {
    margin: 6px 0 0;
    font-weight: 600;
  }
  .delta {
    margin: 2px 0 0;
    font-size: 12px;
  }
  section {
    margin-bottom: 14px;
  }
  h3 {
    margin: 0 0 4px;
    font-size: 12.5px;
    font-weight: 600;
  }
  .row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 1px 0 1px 8px;
    font-size: 12.5px;
  }
  .mark {
    flex: none;
    width: 1em;
    font-weight: 600;
  }
  .name {
    word-break: break-all;
  }
  .detail {
    white-space: nowrap;
  }
  .none {
    margin: 2px 0 2px 8px;
    font-size: 12px;
  }
  /* Green / red / yellow, matching both the terminal UI's palette and `diff`'s own
     ANSI output, so the same change is the same colour everywhere. */
  .added {
    color: var(--ok, #4ec94e);
  }
  .removed {
    color: var(--err, #e05c5c);
  }
  .changed {
    color: var(--warn, #d8b530);
  }
  .meta {
    color: var(--accent);
  }
  .error {
    color: var(--err, #e05c5c);
  }
  .dim {
    color: var(--fg-dim);
  }
  footer {
    margin-top: 18px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    font-size: 12px;
  }
  .cmd {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .cmd code {
    flex: 1;
    min-width: 0;
    padding: 4px 8px;
    overflow-x: auto;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 3px;
    white-space: pre;
  }
</style>
