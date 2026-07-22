<script lang="ts">
  import { cachedSample } from '../stores/server';
  import type { SampleDto } from '../lib/types';
  import { viridis } from '../lib/color';
  import { num } from '../lib/format';
  import Spinner from './Spinner.svelte';

  export let tensor: string;
  export let kind: 'heatmap' | 'values';

  type Mode = 'overview' | 'window' | 'edges';
  let mode: Mode = 'overview';
  let dtype = ''; // '' = stored
  let slice = 0;
  let rowOff = 0;
  let colOff = 0;
  let rows = kind === 'heatmap' ? 128 : 24;
  let cols = kind === 'heatmap' ? 128 : 16;
  let base: 'dec' | 'hex' | 'oct' | 'bin' = 'dec';
  let zebra: 'off' | 'rows' | 'cols' = 'rows';

  let data: SampleDto | null = null;
  let err = '';
  let loading = false;
  let canvas: HTMLCanvasElement;
  let cell = 4;
  let hover = '';
  // Live size of the canvas's container, so the heatmap fills the pane (fit-to-box,
  // square cells, centered) and re-renders on window resize / tab switch.
  let wrapW = 0;
  let wrapH = 0;

  const MODES: Mode[] = ['overview', 'window', 'edges'];
  const DTYPES = ['', 'f16', 'bf16', 'f32', 'f64', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'u4', 'i4'];
  const serverMode = (m: Mode): 'grid' | 'window' | 'edges' => (m === 'overview' ? 'grid' : m);

  $: params = {
    mode: serverMode(mode),
    rows,
    cols,
    slice,
    dtype: dtype || undefined,
    row_off: mode === 'window' ? rowOff : undefined,
    col_off: mode === 'window' ? colOff : undefined,
    raw: kind === 'values' && base !== 'dec' ? 1 : undefined,
  };
  $: load(tensor, params);
  // Only the latest request may write `data` — rapid panning fires overlapping
  // requests, and without this guard an earlier one resolving late would desync
  // the view from the current offset.
  let reqSeq = 0;
  async function load(t: string, p: typeof params) {
    const seq = ++reqSeq;
    loading = true;
    err = '';
    try {
      const d = await cachedSample(t, p);
      if (seq !== reqSeq) return; // superseded
      data = d;
    } catch (e) {
      if (seq !== reqSeq) return;
      err = e instanceof Error ? e.message : String(e);
      data = null;
    }
    if (seq === reqSeq) loading = false;
  }

  $: nSlices = data?.slices ?? 1;
  // Furthest valid top-left of the window, so offsets/seek/pan can't run past the
  // end (the server already clamps; this keeps the client in sync with it).
  $: maxRow = data ? Math.max(0, data.total_rows - rows) : 0;
  $: maxCol = data ? Math.max(0, data.total_cols - cols) : 0;
  $: if (data && mode === 'window') {
    const r = Math.min(Math.max(0, rowOff), maxRow);
    const c = Math.min(Math.max(0, colOff), maxCol);
    if (r !== rowOff) rowOff = r;
    if (c !== colOff) colOff = c;
  }

  // ---- heatmap ----
  $: if (kind === 'heatmap' && data && canvas && wrapW && wrapH) draw(data);
  function draw(d: SampleDto) {
    const r = d.values.length;
    const c = r ? d.values[0].length : 0;
    if (!r || !c) return;
    // Largest square cell that fits the sampled grid into the container (minus the
    // 1px border each side), capped so a tiny grid doesn't balloon. The canvas is
    // centered by its flex wrapper, so it fills the pane's limiting dimension.
    const availW = Math.max(32, wrapW - 2);
    const availH = Math.max(32, wrapH - 2);
    cell = Math.max(1, Math.min(48, Math.floor(Math.min(availW / c, availH / r))));
    canvas.width = c * cell;
    canvas.height = r * cell;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const range = d.max - d.min || 1;
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) {
        const v = d.values[i][j];
        ctx.fillStyle = Number.isFinite(v) ? viridis((v - d.min) / range) : '#000';
        ctx.fillRect(j * cell, i * cell, cell, cell);
      }
    }
  }
  function onMove(e: MouseEvent) {
    if (!data) return;
    const j = Math.floor(e.offsetX / cell);
    const i = Math.floor(e.offsetY / cell);
    const row = data.values[i];
    if (!row || j < 0 || j >= row.length) {
      hover = '';
      return;
    }
    hover = `[${data.rows[i]}, ${data.cols[j]}] = ${num(row[j])}`;
  }

  // ---- values ----
  function cellText(i: number, j: number): string {
    if (!data) return '';
    if (base === 'dec') return num(data.values[i][j]);
    const hex = data.raw?.[i]?.[j];
    if (hex == null) return '';
    if (base === 'hex') return hex;
    const w = data.raw_width ?? hex.length * 4;
    const big = BigInt('0x' + hex);
    return base === 'oct' ? big.toString(8).padStart(Math.ceil(w / 3), '0') : big.toString(2).padStart(w, '0');
  }

  function pan(dr: number, dc: number) {
    rowOff = Math.min(maxRow, Math.max(0, rowOff + dr * rows));
    colOff = Math.min(maxCol, Math.max(0, colOff + dc * cols));
  }

  // Keyboard panning in window mode — mirrors the TUI's data view (arrows pan by a
  // window, Home/End = col start/end, PageUp/PageDown = row start/end). Ignored when
  // typing into a control so sliders/inputs keep their own arrow behavior.
  function onKey(e: KeyboardEvent) {
    if (mode !== 'window' || e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    switch (e.key) {
      case 'ArrowUp': pan(-1, 0); break;
      case 'ArrowDown': pan(1, 0); break;
      case 'ArrowLeft': pan(0, -1); break;
      case 'ArrowRight': pan(0, 1); break;
      case 'PageUp': rowOff = 0; break;
      case 'PageDown': rowOff = maxRow; break;
      case 'Home': colOff = 0; break;
      case 'End': colOff = maxCol; break;
      default: return;
    }
    e.preventDefault();
    e.stopPropagation();
  }
</script>

<svelte:window on:keydown={onKey} />

<div class="dv">
  <div class="controls">
    <div class="grp">
      {#each MODES as m}
        <button class:active={mode === m} on:click={() => (mode = m)}>{m}</button>
      {/each}
    </div>

    <label class="res">rows
      <input type="range" min="8" max="256" bind:value={rows} />
      <input type="number" min="1" bind:value={rows} />
    </label>
    <label class="res">cols
      <input type="range" min="8" max="256" bind:value={cols} />
      <input type="number" min="1" bind:value={cols} />
    </label>

    {#if mode === 'window'}
      <div class="grp pan">
        <button on:click={() => pan(-1, 0)} disabled={rowOff <= 0} title="up (↑ · PageUp = top)">↑</button>
        <button on:click={() => pan(1, 0)} disabled={rowOff >= maxRow} title="down (↓ · PageDown = bottom)">↓</button>
        <button on:click={() => pan(0, -1)} disabled={colOff <= 0} title="left (← · Home = start)">←</button>
        <button on:click={() => pan(0, 1)} disabled={colOff >= maxCol} title="right (→ · End = end)">→</button>
      </div>
      <label class="res">row
        <input type="number" min="0" max={maxRow} step={rows} bind:value={rowOff} />
      </label>
      <label class="res">col
        <input type="number" min="0" max={maxCol} step={cols} bind:value={colOff} />
      </label>
    {/if}

    {#if nSlices > 1}
      <label>slice <input type="number" min="0" max={nSlices - 1} bind:value={slice} /> / {nSlices - 1}</label>
    {/if}

    <label>view
      <select bind:value={dtype}>
        {#each DTYPES as d}<option value={d}>{d === '' ? 'stored' : d}</option>{/each}
      </select>
    </label>

    {#if kind === 'values'}
      <label>base
        <select bind:value={base}>
          <option value="dec">dec</option>
          <option value="hex">hex</option>
          <option value="oct">oct</option>
          <option value="bin">bin</option>
        </select>
      </label>
      <label>zebra
        <select bind:value={zebra}>
          <option value="off">off</option>
          <option value="rows">rows</option>
          <option value="cols">cols</option>
        </select>
      </label>
    {/if}
  </div>

  {#if loading}
    <Spinner label={kind === 'heatmap' ? 'sampling…' : 'reading…'} />
  {:else if err}
    <p class="err">{err}</p>
  {:else if data}
    <div class="meta dim">
      {data.values.length}×{data.values[0]?.length ?? 0} of {data.total_rows}×{data.total_cols}
      · view {data.view}{data.mode !== 'grid' ? ` · ${data.mode}` : ''}
      {#if mode === 'window' && data.rows.length && data.cols.length}
        · <span class="mono">rows {data.rows[0]}–{data.rows[data.rows.length - 1]} · cols {data.cols[0]}–{data.cols[data.cols.length - 1]}</span>
      {:else if mode === 'overview'}
        · <span class="pill" title="Overview is a strided subsample — a lone outlier between sampled indices may not appear. Use window mode for exact 1:1 inspection.">subsample</span>
      {/if}
      <span class="hover mono">{hover}</span>
    </div>

    {#if kind === 'heatmap'}
      <div class="canvaswrap" bind:clientWidth={wrapW} bind:clientHeight={wrapH}>
        <canvas bind:this={canvas} on:mousemove={onMove} on:mouseleave={() => (hover = '')}></canvas>
      </div>
      <div class="scale">
        <span class="mono">{num(data.min)}</span>
        <span class="ramp"></span>
        <span class="mono">{num(data.max)}</span>
      </div>
    {:else}
      <div class="tablewrap">
        <table class="zebra-{zebra}">
          <thead>
            <tr><th></th>{#each data.cols as c}<th class="dim">{c}</th>{/each}</tr>
          </thead>
          <tbody>
            {#each data.values as row, i}
              <tr>
                <th class="dim">{data.rows[i]}</th>
                {#each row as _, j}<td class="mono">{cellText(i, j)}</td>{/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  {/if}
</div>

<style>
  .dv {
    height: 100%;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 12px 16px;
    flex-wrap: wrap;
    margin-bottom: 10px;
    flex: 0 0 auto;
  }
  .grp {
    display: flex;
    gap: 3px;
  }
  .pan button {
    width: 26px;
    padding: 2px 0;
  }
  .pan button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .pill {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 16%, transparent);
    border-radius: 4px;
    padding: 0 5px;
    cursor: help;
  }
  label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--fg-dim);
    font-size: 12px;
  }
  .res input[type='range'] {
    width: 90px;
  }
  .res input[type='number'] {
    width: 62px;
  }
  .meta {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 8px;
    font-size: 12px;
  }
  .hover {
    color: var(--accent);
    margin-left: auto;
  }
  .canvaswrap {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  canvas {
    image-rendering: pixelated;
    border: 1px solid var(--border);
    max-width: 100%;
    max-height: 100%;
  }
  .scale {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
    font-size: 11px;
    flex: 0 0 auto;
  }
  .ramp {
    width: 160px;
    height: 10px;
    border-radius: 2px;
    background: linear-gradient(to right, rgb(68, 1, 84), rgb(59, 82, 139), rgb(33, 145, 140), rgb(94, 201, 98), rgb(253, 231, 37));
  }
  .tablewrap {
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  table {
    border-collapse: collapse;
    font-size: 12px;
  }
  th,
  td {
    padding: 2px 8px;
    text-align: right;
    white-space: nowrap;
  }
  thead th {
    position: sticky;
    top: 0;
    background: var(--bg-panel);
  }
  .zebra-rows tbody tr:nth-child(odd) td {
    background: var(--bg-hover);
  }
  .zebra-cols tbody td:nth-child(even) {
    background: var(--bg-hover);
  }
  .err {
    color: var(--danger);
  }
</style>
