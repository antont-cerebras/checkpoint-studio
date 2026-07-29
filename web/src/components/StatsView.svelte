<script lang="ts">
  import { onMount } from 'svelte';
  import { cachedCheckpointStats } from '../stores/server';
  import { humanCount, humanSize } from '../lib/format';
  import LoadingBar from './LoadingBar.svelte';
  import { startedNow, type Progress } from '../lib/progress';
  let waitStarted: Progress | null = null;
  import Dtype from './Dtype.svelte';
  import Ref from './Ref.svelte';

  interface DtypeStat {
    dtype: string;
    count: number;
    bytes: number;
  }
  interface LayerRow {
    tensors: number;
    params: number;
    bytes: number;
    attn_bytes: number;
    ffn_bytes: number;
    other_bytes: number;
  }
  interface Stats {
    n_tensors: number;
    params: number;
    logical_bytes: number;
    disk_bytes: number;
    model_type?: string;
    files: { count: number };
    layers?: { count: number };
    largest?: { name: string; bytes: number };
    smallest?: { name: string; bytes: number };
    dtypes: DtypeStat[];
    per_layer?: { rows: LayerRow[] };
    experts?: {
      layout: { storage: string; per_layer: number | null };
      gate_up_fused: boolean;
      by_category: { name: string; bytes: number }[];
    };
    footprint?: {
      Disk?: { shards: { name: string; apparent: number; allocated: number }[] };
      /** An `s3://` source: one object per tensor plus the checkpoint index. */
      S3?: { objects: { key: string; size: number }[] };
    };
    /** The S3 section's ready-made phrases, worded by the server exactly as the TUI
     * words them (absent for a local checkpoint). */
    s3_summary?: {
      count: number;
      total_bytes: number;
      checksums: string;
      etags: string;
      tags: string;
      modified?: string | null;
      user_meta_objects: number;
      object_detail: Record<string, string>;
      warnings: string[];
    };
    /** The architecture inferred from the tensors alone, each fact with the evidence it
     * came from, plus the summary rows tensors cannot supply. Same data the terminal's
     * stats screen shows — it rides on `/api/stats` so there is no second fetch and no
     * chance of the two surfaces disagreeing. */
    arch?: {
      facts: [string, { value: string; from: string; key?: string }][];
      not_in_tensors: [string, string][];
    };
  }

  let s: Stats | null = null;
  let err = '';

  onMount(async () => {
    waitStarted = startedNow();
    try {
      s = (await cachedCheckpointStats()) as unknown as Stats;
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    }
  });

  const pct = (v: number, total: number) => (total ? (v / total) * 100 : 0);
  $: dtypeTotal = s ? s.dtypes.reduce((a, d) => a + d.bytes, 0) : 0;
  $: layerMax = s?.per_layer ? Math.max(1, ...s.per_layer.rows.map((r) => r.bytes)) : 1;
  $: shards = s?.footprint?.Disk?.shards ?? [];
  $: s3 = s?.s3_summary ?? null;
  $: s3Objects = s?.footprint?.S3?.objects ?? [];
  // Folded away by default, like the TUI's per-object breakdown: a 1155-object
  // checkpoint would otherwise bury everything below it.
  let s3Open = false;
</script>

<div class="stats">
  <div class="inner">
  {#if err}
    <p class="err">{err}</p>
  {:else if !s}
    <LoadingBar label="reading the statistics" progress={waitStarted} />
  {:else}
    <div class="cards">
      <div class="card"><span class="k">Parameters</span><span class="v">{humanCount(s.params)}</span></div>
      <div class="card"><span class="k">Tensors</span><span class="v">{s.n_tensors.toLocaleString()}</span></div>
      <div class="card"><span class="k">Size</span><span class="v">{humanSize(s.logical_bytes)}</span></div>
      <div class="card"><span class="k">On disk</span><span class="v">{humanSize(s.disk_bytes)}</span></div>
      <div class="card"><span class="k">Files</span><span class="v">{s.files.count}</span></div>
      {#if s.layers}<div class="card"><span class="k">Layers</span><span class="v">{s.layers.count}</span></div>{/if}
      {#if s.model_type}<div class="card"><span class="k">Model</span><span class="v small">{s.model_type}</span></div>{/if}
    </div>

    {#if s.arch?.facts.length}
      <section>
        <h3>Inferred from tensors</h3>
        <p class="note">
          Derived from the tensor names, shapes and dtypes — not from a model card, so it
          works for a checkpoint that has none and disagrees when one is stale.
        </p>
        <dl class="arch">
          {#each s.arch.facts as [label, fact] (label)}
            <dt>{label}</dt>
            <dd>
              <span class="v">{fact.value}</span>
              <span class="from">← {fact.from}</span>
              {#if fact.key}<kbd>{fact.key}</kbd>{/if}
            </dd>
          {/each}
        </dl>
        {#if s.arch.not_in_tensors.length}
          <p class="note">Not in the tensors (a model card reads these from the config):</p>
          <ul class="gaps">
            {#each s.arch.not_in_tensors as [label, why] (label)}
              <li><strong>{label}</strong> — {why}</li>
            {/each}
          </ul>
        {/if}
      </section>
    {/if}

    <section>
      <h3>Data types</h3>
      <div class="dtypes">
        {#each [...s.dtypes].sort((a, b) => b.bytes - a.bytes) as d (d.dtype)}
          <div class="drow">
            <span class="pillcell"><Dtype dtype={d.dtype} /></span>
            <div class="track"><div class="fill" style="width:{pct(d.bytes, dtypeTotal)}%"></div></div>
            <span class="dim mono cnt">{d.count.toLocaleString()} tensors</span>
            <span class="mono sz">{humanSize(d.bytes)}</span>
          </div>
        {/each}
      </div>
    </section>

    {#if s.largest || s.smallest}
      <section class="extremes">
        {#if s.largest}<div><span class="dim">Largest tensor</span> <Ref name={s.largest.name} kind="tensor" class="mono" /> <span class="dim">· {humanSize(s.largest.bytes)}</span></div>{/if}
        {#if s.smallest}<div><span class="dim">Smallest tensor</span> <Ref name={s.smallest.name} kind="tensor" class="mono" /> <span class="dim">· {humanSize(s.smallest.bytes)}</span></div>{/if}
      </section>
    {/if}

    {#if s.experts}
      <section>
        <h3>MoE experts</h3>
        <p class="dim">{s.experts.layout.storage}{s.experts.layout.per_layer ? ` · ${s.experts.layout.per_layer} per layer` : ''}{s.experts.gate_up_fused ? ' · gate/up fused' : ''}</p>
        <div class="dtypes">
          {#each s.experts.by_category as c, ci (ci)}
            {@const total = s.experts.by_category.reduce((a, x) => a + x.bytes, 0)}
            <div class="drow">
              <span class="pill cat">{c.name}</span>
              <div class="track"><div class="fill" style="width:{pct(c.bytes, total)}%"></div></div>
              <span class="mono sz">{humanSize(c.bytes)}</span>
            </div>
          {/each}
        </div>
      </section>
    {/if}

    {#if s.per_layer && s.per_layer.rows.length}
      <section>
        <h3>Per-layer size <span class="dim">(attn / ffn / other)</span></h3>
        <div class="layers">
          {#each s.per_layer.rows as r, i (i)}
            <div class="lrow">
              <span class="li dim mono">{i}</span>
              <div class="stack" style="width:{pct(r.bytes, layerMax)}%">
                <div class="seg attn" style="flex:{r.attn_bytes}" title="attn {humanSize(r.attn_bytes)}"></div>
                <div class="seg ffn" style="flex:{r.ffn_bytes}" title="ffn {humanSize(r.ffn_bytes)}"></div>
                <div class="seg other" style="flex:{r.other_bytes}" title="other {humanSize(r.other_bytes)}"></div>
              </div>
              <span class="lsz dim mono">{humanSize(r.bytes)}</span>
            </div>
          {/each}
        </div>
        <div class="legend">
          <span><i class="attn"></i> attention</span><span><i class="ffn"></i> ffn</span><span><i class="other"></i> other</span>
        </div>
      </section>
    {/if}

    {#if s3}
      <section>
        <h3>☁ S3 objects <span class="count">×{s3.count}</span></h3>
        <table class="rows">
          <tbody>
            <tr><td class="k">Total</td><td class="mono">{humanSize(s3.total_bytes)}</td></tr>
            <tr><td class="k">Checksums</td><td>{s3.checksums}</td></tr>
            <tr><td class="k">ETags</td><td>{s3.etags}</td></tr>
            <tr><td class="k">Tags</td><td>{s3.tags}</td></tr>
            {#if s3.modified}
              <tr><td class="k">Modified</td><td>{s3.modified}</td></tr>
            {/if}
            {#if s3.user_meta_objects}
              <tr>
                <td class="k">User meta</td>
                <td>{s3.user_meta_objects} object{s3.user_meta_objects === 1 ? '' : 's'}</td>
              </tr>
            {/if}
          </tbody>
        </table>
        {#each s3.warnings as w, wi (wi)}
          <p class="warn">⚠ {w}</p>
        {/each}
        {#if s3Objects.length}
          <button class="fold" on:click={() => (s3Open = !s3Open)}>
            {s3Open ? '▾' : '▸'} per-object breakdown ({s3Objects.length})
          </button>
          {#if s3Open}
            <table class="shards">
              <thead><tr><th>object</th><th>detail</th></tr></thead>
              <tbody>
                {#each s3Objects as o, oi (oi)}
                  <tr>
                    <td class="mono">{o.key}</td>
                    <td class="mono dim">{s3.object_detail[o.key] ?? humanSize(o.size)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}
        {/if}
      </section>
    {/if}

    {#if shards.length}
      <section>
        <h3>On-disk footprint</h3>
        <table class="shards">
          <thead><tr><th>shard</th><th>apparent</th><th>allocated</th></tr></thead>
          <tbody>
            {#each shards as sh, shi (shi)}
              <tr><td class="mono"><Ref name={sh.name} kind="file" /></td><td class="mono">{humanSize(sh.apparent)}</td><td class="mono">{humanSize(sh.allocated)}</td></tr>
            {/each}
          </tbody>
        </table>
      </section>
    {/if}
  {/if}
  </div>
</div>

<style>
  /* One row per inferred fact, with its evidence under it — a derived number the reader
     cannot check is only an assertion. */
  .arch {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 2px 12px;
    margin: 6px 0 0;
    font-size: 12.5px;
  }
  .arch dt {
    color: var(--fg-dim);
  }
  .arch dd {
    margin: 0;
  }
  /* Smaller than the label: several of these values are long sentences ("packed weights
     with codebook, qscale, weight_packed"), and at label size they read as headings rather
     than as data. The weight, not the size, is what marks them as the answer. */
  .arch .v {
    font-size: 11.5px;
    font-weight: 600;
  }
  .arch .from {
    margin-left: 8px;
    color: var(--fg-dim);
  }
  /* A shortcut the fact points at, styled as a key rather than written into the prose —
     the evidence text used to say `` `k` ``, which rendered the backticks literally. */
  .arch kbd {
    margin-left: 6px;
    padding: 0 4px;
    font: inherit;
    font-size: 11px;
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: 3px;
  }
  .note {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--fg-dim);
  }
  .gaps {
    margin: 4px 0 0;
    padding-left: 18px;
    font-size: 12px;
    color: var(--fg-dim);
  }
  .stats {
    height: 100%;
    overflow: auto;
  }
  .inner {
    max-width: 960px;
    margin: 0 auto;
    padding: 18px 22px;
  }
  .cards {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 22px;
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 12px 16px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    min-width: 120px;
  }
  .k {
    color: var(--fg-dim);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .v {
    font-size: 20px;
    color: var(--accent);
  }
  .v.small {
    font-size: 14px;
  }
  h3 {
    margin: 0 0 10px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--fg-dim);
  }
  section {
    margin-bottom: 26px;
  }
  .drow {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 3px 0;
  }
  .pillcell {
    flex: 0 0 auto;
    min-width: 60px;
  }
  .pill {
    flex: 0 0 auto;
    min-width: 54px;
    text-align: center;
    font-size: 11px;
    color: var(--dtype);
    background: color-mix(in srgb, var(--dtype) 16%, transparent);
    border-radius: 4px;
    padding: 2px 8px;
  }
  .pill.cat {
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 16%, transparent);
    min-width: 90px;
  }
  .track {
    flex: 1 1 auto;
    height: 10px;
    background: var(--bg-hover);
    border-radius: 5px;
    overflow: hidden;
  }
  .fill {
    height: 100%;
    background: var(--accent);
  }
  .cnt {
    flex: 0 0 auto;
    font-size: 12px;
    width: 120px;
    text-align: right;
  }
  .sz {
    flex: 0 0 auto;
    width: 90px;
    text-align: right;
  }
  .extremes {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
  }
  .layers {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .lrow {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .li {
    flex: 0 0 28px;
    text-align: right;
    font-size: 11px;
  }
  .stack {
    height: 12px;
    display: flex;
    border-radius: 3px;
    overflow: hidden;
    min-width: 2px;
  }
  .seg.attn {
    background: var(--accent);
  }
  .seg.ffn {
    background: var(--dtype);
  }
  .seg.other {
    background: var(--meta);
  }
  .lsz {
    flex: 0 0 auto;
    font-size: 11px;
    width: 80px;
  }
  .legend {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 12px;
    color: var(--fg-dim);
  }
  .legend i {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 2px;
    margin-right: 4px;
    vertical-align: middle;
  }
  .legend i.attn {
    background: var(--accent);
  }
  .legend i.ffn {
    background: var(--dtype);
  }
  .legend i.other {
    background: var(--meta);
  }
  .shards {
    border-collapse: collapse;
    font-size: 12px;
  }
  .shards th {
    text-align: left;
    color: var(--fg-dim);
    font-weight: 400;
    border-bottom: 1px solid var(--border);
    padding: 3px 18px 3px 0;
  }
  .shards td {
    padding: 2px 18px 2px 0;
  }
  /* The S3 section's label/value rows — the same two-column shape the TUI prints. */
  .rows {
    border-collapse: collapse;
    font-size: 12px;
    margin-bottom: 8px;
  }
  .rows td {
    padding: 2px 18px 2px 0;
    vertical-align: top;
  }
  .rows td.k {
    color: var(--fg-dim);
    font-size: 12px;
    text-transform: none;
    letter-spacing: normal;
    white-space: nowrap;
  }
  h3 .count {
    color: var(--fg-dim);
    font-weight: 400;
  }
  /* Fold toggle for the per-object breakdown, matching the tree's disclosure look. */
  .fold {
    background: none;
    border: none;
    color: var(--fg-dim);
    cursor: pointer;
    font: inherit;
    font-size: 12px;
    padding: 2px 0;
  }
  .fold:hover {
    color: var(--accent);
  }
  .warn {
    color: var(--warn);
    font-size: 12px;
    margin: 4px 0;
  }
  .err {
    color: var(--danger);
  }
</style>
