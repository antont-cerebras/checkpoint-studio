<script lang="ts">
  import { isEditable } from '../lib/keys';
  import { tick } from 'svelte';
  import { cachedSample } from '../stores/server';
  import { getDataView, setDataView, type DvParams } from '../stores/view';
  import type { SampleDto } from '../lib/types';
  import { viridis } from '../lib/color';
  import { num } from '../lib/format';
  import LoadingBar from './LoadingBar.svelte';
  import { startedNow, type Progress } from '../lib/progress';
  // The server reads tensor bytes and then answers; a multi-GB tensor takes seconds, which
  // is what the timer is for.
  let waitStarted: Progress | null = null;

  export let tensor: string;
  export let kind: 'heatmap' | 'values';

  type Mode = 'overview' | 'absmax' | 'window' | 'edges';
  // Any params saved in the URL (a shared / bookmarked deep link) override the
  // computed defaults below, so the exact view is reproducible.
  const dv0 = getDataView();
  // The numeric grid defaults to a navigable window (real, contiguous values you can
  // scroll through the whole tensor); the heatmap defaults to the strided overview.
  let mode: Mode = (dv0.mode as Mode) ?? (kind === 'values' ? 'window' : 'overview');
  let dtype = dv0.dtype ?? ''; // '' = stored
  let slice = dv0.slice ? +dv0.slice : 0;
  let rowOff = dv0.roff ? +dv0.roff : 0;
  let colOff = dv0.coff ? +dv0.coff : 0;
  let rows = dv0.rows ? +dv0.rows : kind === 'heatmap' ? 128 : 24;
  let cols = dv0.cols ? +dv0.cols : kind === 'heatmap' ? 128 : 16;
  let base: 'dec' | 'hex' | 'oct' | 'bin' = (dv0.base as 'dec' | 'hex' | 'oct' | 'bin') ?? 'dec';
  let zebra: 'off' | 'rows' | 'cols' = (dv0.zebra as 'off' | 'rows' | 'cols') ?? 'rows';
  // Heatmap: keep the sampled grid's aspect ~ the tensor's true shape, so a
  // 151936×2048 tensor samples as a tall strip, not a misleading 256×128. On by
  // default; the toggle frees the two dimensions.
  let lockRatio = dv0.lock != null ? dv0.lock === '1' : kind === 'heatmap';

  let data: SampleDto | null = null;
  let err = '';
  let loading = false;
  let canvas: HTMLCanvasElement;
  let cell = 4;
  let hover = '';
  // The numeric grid auto-sizes rows×cols to fill its pane (no scrollbars, no empty
  // space) until the user sets a size (via a control or a deep link) — then it's
  // theirs to keep. Measured from the rendered table + the pane box.
  let autoFit = kind === 'values' && !dv0.rows && !dv0.cols;
  let tableEl: HTMLTableElement;
  let tableWrap: HTMLDivElement;
  // Live size of the canvas's container, so the heatmap fills the pane (fit-to-box,
  // square cells, centered) and re-renders on window resize / tab switch.
  let wrapW = 0;
  let wrapH = 0;

  // abs-max (a full-scan magnitude map, nothing sampled away) only makes sense for
  // the heatmap; the numeric grid shows real values, not magnitudes.
  $: modes = (kind === 'heatmap'
    ? ['overview', 'absmax', 'window', 'edges']
    : ['overview', 'window', 'edges']) as Mode[];
  const DTYPES = ['', 'f16', 'bf16', 'f32', 'f64', 'i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'i64', 'u64', 'u4', 'i4'];
  const serverMode = (m: Mode): 'grid' | 'max' | 'window' | 'edges' =>
    m === 'overview' ? 'grid' : m === 'absmax' ? 'max' : m;
  const modeLabel = (m: Mode): string => (m === 'absmax' ? 'abs-max' : m);

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
  $: void load(tensor, params);

  // Mirror the current params into the URL (replace, so no history spam) — a shared
  // link reproduces the exact view. mode/rows/cols always (their defaults are
  // dynamic — snapped / per-tab); the rest only when non-default, to keep links lean.
  // References every param directly so Svelte re-runs it on any change.
  $: {
    const dv: DvParams = { mode, rows: String(rows), cols: String(cols) };
    if (dtype) dv.dtype = dtype;
    if (mode === 'window' && rowOff) dv.roff = String(rowOff);
    if (mode === 'window' && colOff) dv.coff = String(colOff);
    if (slice) dv.slice = String(slice);
    if (kind === 'values' && base !== 'dec') dv.base = base;
    if (kind === 'values' && zebra !== 'rows') dv.zebra = zebra;
    if (kind === 'heatmap' && !lockRatio) dv.lock = '0';
    setDataView(dv);
  }

  // Only the latest request may write `data` — rapid panning fires overlapping
  // requests, and without this guard an earlier one resolving late would desync
  // the view from the current offset.
  let reqSeq = 0;
  async function load(t: string, p: typeof params) {
    const seq = ++reqSeq;
    loading = true;
    waitStarted = startedNow();
    try {
      const d = await cachedSample(t, p);
      if (seq !== reqSeq) return; // superseded
      data = d;
      err = '';
    } catch (e) {
      if (seq !== reqSeq) return;
      err = e instanceof Error ? e.message : String(e);
      // Keep the last good `data` on a refetch error, so panning never blanks the
      // view — the error shows as a chip while the previous window stays visible.
    }
    if (seq === reqSeq) loading = false;
  }

  /** Re-request the current view after a failure. `cachedSample` drops rejected
   * entries, so this really re-hits the server rather than replaying the error. */
  function retry() {
    err = '';
    void load(tensor, params);
  }

  // Size the numeric grid to fill its pane: measure the rendered cell size and floor
  // to whole cells (a slight under-fill so a rounding overshoot can't trip a
  // scrollbar). Runs after each render while auto-fitting — converges in one step,
  // then a no-op — and on window resize.
  async function fitToPane() {
    if (!autoFit || kind !== 'values') return;
    await tick();
    const head = tableEl?.querySelector<HTMLElement>('thead tr');
    const body = tableEl?.querySelector<HTMLElement>('tbody tr');
    if (!head || !body || !tableWrap) return;
    const rowH = body.offsetHeight;
    const headH = head.offsetHeight;
    const rowHeaderW = (body.children[0] as HTMLElement | undefined)?.offsetWidth ?? 0;
    // Widest data cell in the sample row — conservative, so columns never overflow.
    const dataCols = Array.from(body.children).slice(1) as HTMLElement[];
    const colW = Math.max(1, ...dataCols.map((c) => c.offsetWidth));
    if (rowH < 2 || colW < 2) return;
    const fitR = Math.max(1, Math.floor((tableWrap.clientHeight - headH - 2) / rowH));
    const fitC = Math.max(1, Math.floor((tableWrap.clientWidth - rowHeaderW - 2) / colW));
    if (fitR !== rows || fitC !== cols) {
      rows = fitR;
      cols = fitC;
    }
  }
  // Re-fit after each render (data change) while auto-fitting, and on resize.
  $: if (data && autoFit) void fitToPane();

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

  // ---- aspect-ratio lock (heatmap) ----
  // Source aspect = sampled rows-per-col we want to mirror; from the loaded 2-D dims
  // (`null` until the first sample lands, and only meaningful for the heatmap).
  $: aspect = kind === 'heatmap' && data && data.total_cols ? data.total_rows / data.total_cols : null;
  const clampDim = (n: number) => Math.max(1, Math.min(4096, Math.round(n)));
  // Snap both dims to the source ratio once per tensor: budget the long side to ~256
  // cells, the short side proportional (>= 1). So an extreme shape reads as a strip.
  // Dims restored from the URL win over the auto-snap (mark this tensor done).
  let snappedFor = dv0.rows || dv0.cols ? tensor : '';
  $: if (lockRatio && aspect && tensor !== snappedFor) {
    const b = 256;
    if (aspect >= 1) { rows = clampDim(b); cols = clampDim(b / aspect); }
    else { cols = clampDim(b); rows = clampDim(b * aspect); }
    // Read on the NEXT run of this reactive block (the once-per-tensor guard), which
    // ESLint's per-block flow analysis can't see.
    // eslint-disable-next-line no-useless-assignment
    snappedFor = tensor;
  }
  // Editing one dimension recomputes the other from the source aspect (either drives
  // the other). No-op when unlocked or for the numeric grid.
  // A manual rows/cols edit takes over from auto-fit (and, on the heatmap, keeps the
  // aspect ratio when locked).
  function onRows() { autoFit = false; if (lockRatio && aspect) cols = clampDim(rows / aspect); }
  function onCols() { autoFit = false; if (lockRatio && aspect) rows = clampDim(cols * aspect); }
  function toggleLock() {
    lockRatio = !lockRatio;
    if (lockRatio) onRows(); // re-apply the ratio, keeping the current row count
  }

  // ---- heatmap ----
  $: if (kind === 'heatmap' && data && canvas && wrapW && wrapH) draw(data);
  function draw(d: SampleDto) {
    const r = d.values.length;
    const c = d.values[0]?.length ?? 0;
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
        const v = d.values[i]?.[j];
        ctx.fillStyle =
          v != null && Number.isFinite(v) ? viridis((v - d.min) / range) : '#000';
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
    hover = `[${data.rows[i]}, ${data.cols[j]}] = ${num(row[j] ?? Number.NaN)}`;
  }

  // Click an overview / abs-max cell to drop a window there — a discoverable seek
  // into the exact region (the window is centered on the clicked block).
  function onClick(e: MouseEvent) {
    if (!data || (mode !== 'overview' && mode !== 'absmax')) return;
    const j = Math.floor(e.offsetX / cell);
    const i = Math.floor(e.offsetY / cell);
    const r = data.rows[i];
    const c = data.cols[j];
    if (r == null || c == null) return;
    rowOff = Math.max(0, r - Math.floor(rows / 2));
    colOff = Math.max(0, c - Math.floor(cols / 2));
    mode = 'window';
  }

  // ---- values ----
  // Takes `d` and `b` as arguments (rather than closing over `data`/`base`) so the
  // `{cellText(data, base, i, j)}` call in the markup names them as dependencies —
  // otherwise Svelte, seeing only `i`/`j` in the expression, would render each cell
  // once and never update its text on pan/seek/base-change (the header, bound to
  // `data.rows[i]`, would relabel while the values stayed frozen at the origin).
  /** Exact decimal for an integer cell, decoded from its raw bits. `values` arrives as
   * JSON numbers (f64), which round past 2^53 and can't represent a u64 above 2^63 at
   * all — so a wide I64/U64 tensor would display the wrong number. */
  function exactInt(hex: string, width: number | undefined, signed: boolean): string {
    const w = BigInt(width ?? hex.length * 4);
    let v = BigInt('0x' + hex);
    if (signed && w > 0n && v >= 1n << (w - 1n)) v -= 1n << w; // two's complement
    return v.toString();
  }

  function cellText(d: SampleDto, b: 'dec' | 'hex' | 'oct' | 'bin', i: number, j: number): string {
    if (b === 'dec') {
      const bits = d.integer ? d.raw?.[i]?.[j] : undefined;
      return bits != null ? exactInt(bits, d.raw_width, d.signed) : num(d.values[i]?.[j] ?? Number.NaN);
    }
    const hex = d.raw?.[i]?.[j];
    if (hex == null) return '';
    if (b === 'hex') return hex;
    const w = d.raw_width ?? hex.length * 4;
    const big = BigInt('0x' + hex);
    return b === 'oct' ? big.toString(8).padStart(Math.ceil(w / 3), '0') : big.toString(2).padStart(w, '0');
  }

  // Hovering a value cell shows its exact coordinates + full-precision value on the
  // meta line — the same readout the heatmap has, so both views answer "what is this
  // cell?". Delegated on the table (one listener, not one per cell).
  function onCellHover(e: Event) {
    const td = (e.target as HTMLElement)?.closest<HTMLElement>('td[data-i]');
    if (!td || !data) {
      hover = '';
      return;
    }
    const i = Number(td.dataset.i);
    const j = Number(td.dataset.j);
    const v = data.values[i]?.[j];
    if (v == null) {
      hover = '';
      return;
    }
    // Exact for integers (see `exactInt`); full f64 precision otherwise.
    const bits = data.integer ? data.raw?.[i]?.[j] : undefined;
    const shown = bits != null ? exactInt(bits, data.raw_width, data.signed) : v;
    hover = `[${data.rows[i]}, ${data.cols[j]}] = ${shown}`;
  }

  function pan(dr: number, dc: number) {
    rowOff = Math.min(maxRow, Math.max(0, rowOff + dr * rows));
    colOff = Math.min(maxCol, Math.max(0, colOff + dc * cols));
  }

  // Mouse-wheel panning in window mode (both views): a vertical wheel moves the
  // window down/up, a horizontal wheel (trackpad / shift-wheel) moves it right/left —
  // a fine scroll complementing the ←↑↓→ (page) keys and the "go to row" jump. Each
  // move fetches the new window, so you can reach any part of the tensor.
  function onWheel(e: WheelEvent) {
    if (mode !== 'window' || !data) return;
    e.preventDefault(); // capture the gesture as a pan, not a page/inner scroll
    const stepR = Math.max(1, Math.round(rows / 4));
    const stepC = Math.max(1, Math.round(cols / 4));
    if (e.deltaY) rowOff = Math.min(maxRow, Math.max(0, rowOff + Math.sign(e.deltaY) * stepR));
    if (e.deltaX) colOff = Math.min(maxCol, Math.max(0, colOff + Math.sign(e.deltaX) * stepC));
  }

  // Edges mode skips a contiguous middle block; find where the row/col index jumps so
  // the table can show a "⋯ skipped ⋯" divider there (−1 = no gap / other modes).
  const gapAt = (idx: number[]): number => {
    for (let i = 0; i + 1 < idx.length; i++) if ((idx[i + 1] ?? 0) - (idx[i] ?? 0) > 1) return i;
    return -1;
  };
  $: rowGap = mode === 'edges' && data ? gapAt(data.rows) : -1;
  $: colGap = mode === 'edges' && data ? gapAt(data.cols) : -1;
  $: rowsSkipped =
    rowGap >= 0 && data ? (data.rows[rowGap + 1] ?? 0) - (data.rows[rowGap] ?? 0) - 1 : 0;
  $: colsSkipped =
    colGap >= 0 && data ? (data.cols[colGap + 1] ?? 0) - (data.cols[colGap] ?? 0) - 1 : 0;
  // Total table columns for the skipped-rows divider's colspan: index header + data
  // cols + the skipped-cols divider column (when present).
  $: colspan = data ? 1 + data.cols.length + (colGap >= 0 ? 1 : 0) : 1;

  // Give the data pane keyboard focus on mount so the arrow / Page / Home / End pan
  // keys work immediately. Without a focusable pane the focus sits on a control (or
  // body): the window key handler then either adjusts that control (its INPUT/SELECT
  // guard) or, since a plain table can't take focus, never gets a pannable target —
  // which read as "arrows do nothing" while a stray keystroke hit whatever WAS
  // focused. `preventScroll` so grabbing focus can't jump the page.
  function grabFocus(node: HTMLElement) {
    node.focus({ preventScroll: true });
  }
  // Also take focus on any click into the pane (belt-and-suspenders with the mount
  // grab): clicking a cell must leave the grid focused so the very next arrow key
  // pans. `currentTarget` is the pane itself, so a click on any descendant counts.
  function focusPane(e: MouseEvent) {
    (e.currentTarget as HTMLElement).focus({ preventScroll: true });
  }

  // Keyboard panning in window mode — mirrors the TUI's data view (arrows pan by a
  // window, Home/End = col start/end, PageUp/PageDown = row start/end). Bound in the
  // capture phase on window (see the markup): the app-global nav handler binds window
  // too and was consuming the arrows before the grid ever saw them, so the data view
  // claims them first and stops propagation for the keys it handles. Keys typed into a
  // control (the rows/cols/seek inputs) are left alone, and unhandled keys propagate
  // normally to the global shortcuts.
  function onKey(e: KeyboardEvent) {
    if (mode !== 'window' || e.ctrlKey || e.metaKey || e.altKey) return;
    if (isEditable(e.target)) return;
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

<!-- Pan keys are claimed in the CAPTURE phase on window, so they reach the grid before
     ANY other keydown listener (the app-global nav handler also binds window and was
     swallowing the arrows — capture runs before every bubble-phase listener, whatever
     the attach order). `onKey` stops propagation only for keys it actually handles, so
     everything else still reaches the global shortcuts. This component is mounted only
     while a data view is on screen, so it can't shadow the tree's j/k nav. -->
<svelte:window on:keydown|capture={onKey} on:resize={fitToPane} />

<div class="dv">
  <div class="controls">
    <div class="grp">
      {#each modes as m (m)}
        <button class:active={mode === m} on:click={() => (mode = m)}>{modeLabel(m)}</button>
      {/each}
    </div>

    <!-- A wrapping <label> names only its FIRST input, so the number boxes need their
         own aria-label (else they're unnamed to a screen reader). -->
    <label class="res">rows
      <input type="range" min="8" max="256" bind:value={rows} on:input={onRows} aria-label="rows (slider)" />
      <input type="number" min="1" bind:value={rows} on:input={onRows} aria-label="rows" />
    </label>
    <label class="res">cols
      <input type="range" min="8" max="256" bind:value={cols} on:input={onCols} aria-label="cols (slider)" />
      <input type="number" min="1" bind:value={cols} on:input={onCols} aria-label="cols" />
    </label>
    {#if kind === 'heatmap'}
      <button
        class="lock"
        class:on={lockRatio}
        on:click={toggleLock}
        title={lockRatio
          ? 'Aspect ratio locked to the tensor shape — editing rows or cols adjusts the other. Click to unlock.'
          : 'Aspect ratio unlocked — set rows and cols freely. Click to lock to the tensor shape.'}
      >{lockRatio ? '🔒' : '🔓'} ratio</button>
    {/if}

    {#if mode === 'window'}
      <div class="grp pan">
        <button on:click={() => pan(-1, 0)} disabled={rowOff <= 0} title="up (↑ · PageUp = top)">↑</button>
        <button on:click={() => pan(1, 0)} disabled={rowOff >= maxRow} title="down (↓ · PageDown = bottom)">↓</button>
        <button on:click={() => pan(0, -1)} disabled={colOff <= 0} title="left (← · Home = start)">←</button>
        <button on:click={() => pan(0, 1)} disabled={colOff >= maxCol} title="right (→ · End = end)">→</button>
      </div>
      <label class="res">go&nbsp;to&nbsp;row
        <input type="number" min="0" max={maxRow} step={rows} bind:value={rowOff} />
      </label>
      <label class="res">col
        <input type="number" min="0" max={maxCol} step={cols} bind:value={colOff} />
      </label>
    {/if}

    {#if nSlices > 1}
      <label>slice <input type="number" min="0" max={nSlices - 1} bind:value={slice} /> / {nSlices - 1}</label>
    {/if}

    <label title="Reinterpret the raw bytes as another dtype before display (e.g. read a packed weight as u4). 'stored' uses the tensor's real dtype.">override&nbsp;dtype
      <select bind:value={dtype}>
        {#each DTYPES as d (d)}<option value={d}>{d === '' ? 'stored' : d}</option>{/each}
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

  <!-- Keep the last-loaded view mounted while a new window is fetched, so scrolling
       updates in place instead of blanking to a spinner (no blink). The spinner
       shows only on the very first load; a refetch error is a chip, not a blank. -->
  {#if data}
    <div class="meta dim">
      {data.values.length}×{data.values[0]?.length ?? 0} of {data.total_rows}×{data.total_cols}
      · view {data.view}{data.mode !== 'grid' ? ` · ${data.mode}` : ''}
      {#if mode === 'window' && data.rows.length && data.cols.length}
        · <span class="mono">rows {data.rows[0]}–{data.rows[data.rows.length - 1]} · cols {data.cols[0]}–{data.cols[data.cols.length - 1]}</span>
      {:else if mode === 'overview'}
        · <span class="pill" title="Overview is a strided subsample — a lone outlier between sampled indices may not appear. Use window mode for exact 1:1 inspection.">subsample</span>
      {/if}
      {#if loading}<span class="busy" title="loading…">⟳</span>{/if}
      {#if err}<span class="ferr" title={err}>⚠</span>{/if}
      <span class="hover mono">{hover}</span>
    </div>

    {#if kind === 'heatmap'}
      <!-- svelte-ignore a11y-no-noninteractive-tabindex a11y-no-static-element-interactions -->
      <div
        class="canvaswrap"
        tabindex="0"
        use:grabFocus
        bind:clientWidth={wrapW}
        bind:clientHeight={wrapH}
        on:wheel|nonpassive={onWheel}
        on:mousedown={focusPane}
      >
        <canvas
          bind:this={canvas}
          class:clickable={mode === 'overview' || mode === 'absmax'}
          title={mode === 'overview' || mode === 'absmax' ? 'Click a cell to open a window there' : ''}
          on:mousemove={onMove}
          on:mouseleave={() => (hover = '')}
          on:click={onClick}
        ></canvas>
      </div>
      <div class="scale">
        <span class="mono">{num(data.min)}</span>
        <span class="ramp"></span>
        <span class="mono">{num(data.max)}</span>
      </div>
    {:else}
      <!-- svelte-ignore a11y-no-static-element-interactions a11y-no-noninteractive-tabindex -->
      <div class="tablewrap" tabindex="0" use:grabFocus bind:this={tableWrap} on:wheel|nonpassive={onWheel} on:mousedown={focusPane}>
        <!-- svelte-ignore a11y-mouse-events-have-key-events -->
        <!-- (the keyboard equivalent is on:focusin — `focus` doesn't bubble to the
             table, so the rule's suggested on:focus would never fire here) -->
        <table
          class="zebra-{zebra}"
          bind:this={tableEl}
          on:mouseover={onCellHover}
          on:focusin={onCellHover}
          on:mouseleave={() => (hover = '')}
        >
          <thead>
            <tr>
              <th></th>
              {#each data.cols as c, j (j)}
                <th class="dim" scope="col">{c}</th>
                {#if j === colGap}<th class="sep" scope="col" title="{colsSkipped.toLocaleString()} columns skipped">⋯</th>{/if}
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each data.values as row, i (i)}
              <tr>
                <th class="dim" scope="row">{data.rows[i]}</th>
                {#each row as _, j (j)}
                  <td class="mono" data-i={i} data-j={j}>{cellText(data, base, i, j)}</td>
                  {#if j === colGap}<td class="sep">⋯</td>{/if}
                {/each}
              </tr>
              {#if i === rowGap}
                <tr class="seprow"><td colspan={colspan}>⋯ {rowsSkipped.toLocaleString()} rows skipped ⋯</td></tr>
              {/if}
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  {:else if err}
    <!-- Error before `loading`, and a catch-all `{:else}` below: a data view must
         always render a terminal state. A silent blank pane is indistinguishable from
         "empty tensor" or "still loading", and leaves no way to recover. -->
    <div class="failed">
      <p class="err">⚠ {err}</p>
      <button on:click={retry}>Retry</button>
    </div>
  {:else if loading}
    <LoadingBar
      label={kind === 'heatmap' ? 'sampling the tensor' : 'reading the values'}
      progress={waitStarted}
    />
  {:else}
    <div class="failed">
      <p class="dim">No data returned for this view.</p>
      <button on:click={retry}>Retry</button>
    </div>
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
  .lock {
    white-space: nowrap;
    font-size: 12px;
  }
  .lock.on {
    background: var(--bg-sel);
    border-color: var(--accent);
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
  /* Non-blocking refetch indicators: the view stays put while a new window loads. */
  .busy {
    color: var(--accent);
    display: inline-block;
    animation: spin 0.9s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .ferr {
    color: var(--warn);
    cursor: help;
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
  canvas.clickable {
    cursor: crosshair;
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
  /* The panes hold keyboard focus (arrow/Page/Home/End pan). Suppress the default
     ring on the programmatic mount-focus; show a subtle accent only on keyboard nav. */
  .tablewrap:focus,
  .canvaswrap:focus {
    outline: none;
  }
  .tablewrap:focus-visible,
  .canvaswrap:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
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
  /* Edges mode: a clear divider where the skipped middle block was elided, so the
     first- and last-index rows/cols aren't mistaken for contiguous data. */
  td.sep,
  th.sep {
    text-align: center;
    color: var(--fg-dim);
    background: var(--bg-hover);
    border-left: 1px dashed var(--border);
    border-right: 1px dashed var(--border);
  }
  .seprow td {
    text-align: center;
    color: var(--fg-dim);
    background: var(--bg-hover);
    border-top: 1px dashed var(--border);
    border-bottom: 1px dashed var(--border);
    font-size: 11px;
    letter-spacing: 0.03em;
  }
  .err {
    color: var(--danger);
  }
  /* Terminal failure state: never leave the pane blank — say what went wrong and
     offer a way out. */
  .failed {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
    padding: 14px 0;
  }
  .failed .err {
    margin: 0;
  }
</style>
