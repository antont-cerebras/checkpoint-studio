<script lang="ts">
  import { tree, cachedStats, unindexed, caps, dataViewNote } from '../stores/server';
  import { setTab, navigate, type DataTab } from '../stores/view';
  import type { StatsDto, TensorInfo, TreeNode } from '../lib/types';
  import { humanCount, humanSize, num, percent } from '../lib/format';
  import DataView from './DataView.svelte';
  import HistogramView from './HistogramView.svelte';
  import Dtype from './Dtype.svelte';
  import Shape from './Shape.svelte';
  import LoadingBar from './LoadingBar.svelte';
  import { startedNow, type Progress } from '../lib/progress';

  export let tensor: string;
  export let tab: DataTab;

  $: info = findTensor($tree?.tree ?? [], tensor);

  function findTensor(nodes: TreeNode[], name: string): TensorInfo | null {
    for (const n of nodes) {
      if (n.kind === 'tensor' && n.info.name === name) return n.info;
      if (n.kind === 'group') {
        const found = findTensor(n.children, name);
        if (found) return found;
      }
    }
    return null;
  }

  function baseName(p: string): string {
    return p.split('/').pop() || p;
  }

  function offsets(layout: unknown): string | null {
    const l = layout as Record<string, { start?: number; end?: number }>;
    if (l && l.ByteRange) return `${l.ByteRange.start} – ${l.ByteRange.end} (within file data)`;
    return null;
  }

  // The source line adapts to where the tensor lives. An `s3://` cstorch checkpoint
  // isn't a local file, so label it "Source", and only offer the byte-layout-map
  // link when the tensor actually HAS a byte-range layout (safetensors) — for
  // cstorch (no byte ranges) / HDF5 (chunked) that map doesn't exist, so the link
  // would go nowhere.
  // The source's own answer, not a guess from the path's shape — the same reason the Rust
  // side asks `capabilities` instead of testing for a prefix.
  $: isS3 = $tree?.location === 's3';
  // Whether the data views can work at all here. A remote source carries only the
  // structure, so offering a heatmap that 400s teaches the user nothing: the tabs are
  // disabled and the server's own sentence says why.
  $: canReadBytes = $caps?.read_bytes ?? false;
  // A new wait each time a scan is started; tied to the promise so a re-render doesn't
  // restart the clock. No fraction: the server scans and then answers, so bytes received
  // would be 0 until the moment it finishes.
  let scanStarted: Progress | null;
  $: scanStarted = statsPromise ? startedNow() : null;
  $: isExtra = info !== null && info !== undefined && $unindexed.has(info.source_path);
  $: hasByteLayout = info ? offsets(info.layout) != null : false;

  // Whole-tensor statistics are shown on the Info tab, scanned on demand.
  let statsPromise: Promise<StatsDto> | null = null;
  $: if (tensor) statsPromise = null; // reset when the selected tensor changes
  function scan() {
    statsPromise = cachedStats(tensor);
  }

  // The three that read tensor bytes are gated on that capability; Info never is.
  const tabs: { id: DataTab; label: string; needsBytes: boolean }[] = [
    { id: 'info', label: 'Info', needsBytes: false },
    { id: 'heatmap', label: 'Heatmap', needsBytes: true },
    { id: 'values', label: 'Values', needsBytes: true },
    { id: 'histogram', label: 'Histogram', needsBytes: true },
  ];
</script>

<div class="detail">
  <div class="tabbar">
    {#each tabs as t (t.id)}
      <button
        class:active={tab === t.id}
        disabled={t.needsBytes && !canReadBytes}
        title={t.needsBytes && !canReadBytes ? ($dataViewNote ?? '') : undefined}
        on:click={() => setTab(t.id)}>{t.label}</button
      >
    {/each}
  </div>

  <div class="body">
    {#if !info}
      <p class="dim">Tensor not found: {tensor}</p>
    {:else if tab === 'info'}
      <h2>{info.name}</h2>
      <table>
        <tbody>
          <tr><th>Data Type</th><td><Dtype dtype={info.dtype} /></td></tr>
          <tr><th>Shape</th><td><Shape shape={info.shape} /></td></tr>
          <tr><th>Parameters</th><td class="mono">{humanCount(info.num_elements)} ({info.num_elements.toLocaleString()})</td></tr>
          <tr><th>Size</th><td class="mono">{humanSize(info.size_bytes)}</td></tr>
          {#if offsets(info.layout)}<tr><th>Data offsets</th><td class="mono">{offsets(info.layout)}</td></tr>{/if}
          <tr>
            <th>{isS3 ? 'Source' : 'File'}</th>
            <td>
              {#if hasByteLayout}
                <button class="link src" title="Show this shard's byte-layout map" on:click={() => navigate({ kind: 'layout', file: baseName(info.source_path) })}>{info.source_path}</button>
              {:else}
                <span class="src mono">{info.source_path}</span>
              {/if}
            </td>
          </tr>
          <!-- The terminal's detail screen carries this flag too, in the same red:
           the file is on disk but the index never names it, so a loader following
           only the index will not read this tensor. -->
          {#if isExtra}
            <tr>
              <th></th>
              <td class="extra">✚ on disk but not listed in model.safetensors.index.json</td>
            </tr>
          {/if}
        </tbody>
      </table>

      <div class="statsblock">
        <h3>Statistics</h3>
        {#if !canReadBytes}
          <!-- The server's sentence, not a second wording of it. -->
          <p class="dim note">{$dataViewNote ?? ''}</p>
        {:else if !statsPromise}
          <button on:click={scan}>Scan tensor</button>
          <span class="dim">reads the whole tensor's values</span>
        {:else}
          {#await statsPromise}
            <LoadingBar label="scanning the tensor" progress={scanStarted} />
          {:then st}
            <table>
              <tbody>
                <tr><th>min</th><td class="mono">{num(st.min)}</td><th>max</th><td class="mono">{num(st.max)}</td></tr>
                <tr><th>mean</th><td class="mono">{num(st.mean)}</td><th>std</th><td class="mono">{num(st.std)}</td></tr>
                <tr><th>zeros</th><td class="mono">{percent(st.zero_fraction, st.zeros === 0)}</td><th>non-finite</th><td class="mono">{st.nonfinite.toLocaleString()}</td></tr>
              </tbody>
            </table>
            <span class="dim">scanned {st.count.toLocaleString()} elements in {st.elapsed_ms.toFixed(0)} ms</span>
          {:catch e}
            <p class="err">{e.message}</p>
          {/await}
        {/if}
      </div>
    {:else if tab === 'heatmap'}
      <DataView {tensor} kind="heatmap" />
    {:else if tab === 'values'}
      <DataView {tensor} kind="values" />
    {:else if tab === 'histogram'}
      <HistogramView name={tensor} dtype={info.dtype} />
    {/if}
  </div>
</div>

<style>
  .detail {
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  .tabbar {
    flex: 0 0 auto;
    display: flex;
    gap: 6px;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
  }
  .body {
    flex: 1 1 auto;
    overflow: auto;
    padding: 14px 18px;
  }
  h2 {
    margin: 0 0 12px;
    font-size: 15px;
    color: var(--accent);
    word-break: break-all;
  }
  h3 {
    margin: 0 0 8px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--fg-dim);
  }
  table {
    border-collapse: collapse;
    margin-bottom: 14px;
  }
  th {
    text-align: right;
    color: var(--fg-dim);
    font-weight: 400;
    padding: 2px 12px 2px 0;
    vertical-align: top;
    white-space: nowrap;
  }
  td {
    padding: 2px 18px 2px 0;
  }
  .statsblock {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-start;
  }
  .statsblock table {
    margin: 0;
  }
  .tabbar button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .note {
    max-width: 70ch;
    margin: 4px 0 0;
    line-height: 1.5;
  }
  /* The same vivid red the terminal marks an unindexed tensor with. */
  .extra {
    color: var(--unindexed);
  }
  .src {
    word-break: break-all;
  }
  .link {
    background: none;
    border: none;
    padding: 0;
    text-align: left;
    color: var(--accent);
    text-decoration: underline;
    text-decoration-style: dotted;
    cursor: pointer;
    font: inherit;
  }
  .err {
    color: var(--danger);
  }
</style>
