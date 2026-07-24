<script lang="ts">
  // A compact per-family listing: tensors that differ only by an index (layer /
  // expert number) collapse into one row — `model.layers.{0-47}.…experts.{0-3}.
  // down_proj.weight` × N — with the uniform dtype/shape and rolled-up params/size.
  // Reflects the current filter query (server-collapsed, same as `diff`).
  import { api } from '../lib/api';
  import { filterQuery } from '../stores/view';
  import { humanCount, humanSize, pyShape } from '../lib/format';
  import Dtype from './Dtype.svelte';
  import Ref from './Ref.svelte';

  interface Family {
    name: string;
    count: number;
    dtype: string | null;
    shape: number[] | null;
    params: number;
    size_bytes: number;
  }
  let families: Family[] = [];
  let err = '';
  let loading = false;
  let seq = 0;

  async function load(q: string) {
    const s = ++seq;
    loading = true;
    err = '';
    try {
      const r = await api.schema(q);
      if (s !== seq) return;
      families = r.families ?? [];
    } catch (e) {
      if (s !== seq) return;
      err = e instanceof Error ? e.message : String(e);
    }
    if (s === seq) loading = false;
  }
  $: void load($filterQuery);
  $: total = families.reduce((a, f) => a + f.count, 0);
</script>

<div class="compact">
  {#if err}
    <p class="err">{err}</p>
  {:else if loading && !families.length}
    <p class="dim">grouping…</p>
  {:else}
    <div class="rows">
      <div class="frow hdr">
        <span class="c">#</span><span class="n">family</span><span class="d">dtype</span>
        <span class="s">shape</span><span class="p">params</span><span class="z">size</span>
      </div>
      {#each families as f (f.name)}
        <div class="frow">
          <span class="c mono">{f.count}</span>
          <span class="n" title={f.name}><Ref name={f.name} /></span>
          <span class="d">
            {#if f.dtype}<Dtype dtype={f.dtype} bubble={false} />{:else}<span class="dim">varies</span>{/if}
          </span>
          <span class="s mono">{f.shape ? pyShape(f.shape) : '—'}</span>
          <span class="p mono">{humanCount(f.params)}</span>
          <span class="z mono">{humanSize(f.size_bytes)}</span>
        </div>
      {/each}
    </div>
    <div class="foot dim">
      {families.length} famil{families.length === 1 ? 'y' : 'ies'} · {total} tensor{total === 1 ? '' : 's'}
    </div>
  {/if}
</div>

<style>
  .compact {
    height: 100%;
    display: flex;
    flex-direction: column;
    min-height: 0;
    font-size: 13px;
  }
  .rows {
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
  }
  .frow {
    display: grid;
    grid-template-columns: 48px minmax(0, 1fr) 84px 150px 90px 90px;
    gap: 10px;
    align-items: center;
    padding: 2px 14px;
    white-space: nowrap;
  }
  .frow:not(.hdr):hover {
    background: var(--bg-hover);
  }
  .hdr {
    position: sticky;
    top: 0;
    background: var(--bg-panel);
    color: var(--fg-dim);
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
    padding-top: 5px;
    padding-bottom: 5px;
  }
  .c {
    text-align: right;
    color: var(--accent);
  }
  .n {
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: ui-monospace, monospace;
    color: var(--tensor);
  }
  .p,
  .z {
    text-align: right;
  }
  .foot {
    flex: 0 0 auto;
    padding: 5px 14px;
    border-top: 1px solid var(--border);
    font-size: 12px;
  }
  .err {
    padding: 16px;
    color: var(--danger);
  }
</style>
