<script lang="ts">
  // A mouse-driven alternative to typing the filter query: each facet is a control
  // (dtype chips, name + mode, shape, dim, rank, size/params ranges, shard), each
  // negatable. "Add to filter" assembles the non-empty facets into query terms and
  // appends them to the shared filter query (which the server then matches) — so it
  // composes with anything already typed, and OR/AND/negation stay identical.
  import { tree } from '../stores/server';
  import { addFilterTerms } from '../stores/view';
  import type { TreeNode } from '../lib/types';
  import Dtype from './Dtype.svelte';

  function distinctDtypes(nodes: TreeNode[]): string[] {
    const set = new Set<string>();
    const walk = (ns: TreeNode[]) => {
      for (const n of ns) {
        if (n.kind === 'tensor') set.add(n.info.dtype);
        else if (n.kind === 'group') walk(n.children);
      }
    };
    walk(nodes);
    return [...set].sort();
  }
  $: present = $tree ? distinctDtypes($tree.tree) : [];

  let dtypes = new Set<string>();
  let name = '';
  let nameMode: 'contains' | 're' | 'glob' = 'contains';
  let shape = '';
  let dim = '';
  let rank = '';
  let sizeMin = '';
  let sizeMax = '';
  let paramsMin = '';
  let paramsMax = '';
  let shard = '';
  let neg = {
    dtype: false,
    name: false,
    shape: false,
    dim: false,
    rank: false,
    size: false,
    params: false,
    shard: false,
  };

  function toggleDtype(d: string) {
    if (dtypes.has(d)) dtypes.delete(d);
    else dtypes.add(d);
    dtypes = dtypes; // nudge reactivity
  }

  const not = (on: boolean) => (on ? '!' : '');
  function range(facet: string, min: string, max: string, negate: boolean): string {
    const m = min.trim();
    const x = max.trim();
    let v = '';
    if (m && x) v = `${m}..${x}`;
    else if (m) v = `${m}..`;
    else if (x) v = `..${x}`;
    else return '';
    return `${not(negate)}${facet}:${v}`;
  }

  function assemble(): string[] {
    const t: string[] = [];
    if (dtypes.size) t.push(`${not(neg.dtype)}dtype:${[...dtypes].join(',')}`);
    if (name.trim()) {
      const pfx = nameMode === 'contains' ? '' : `${nameMode}:`;
      t.push(`${not(neg.name)}name:${pfx}${name.trim()}`);
    }
    if (shape.trim()) {
      let s = shape.trim();
      if (!s.startsWith('(')) s = `(${s})`;
      t.push(`${not(neg.shape)}shape:${s}`);
    }
    if (dim.trim()) t.push(`${not(neg.dim)}dim:${dim.trim()}`);
    if (rank.trim()) t.push(`${not(neg.rank)}rank:${rank.trim()}`);
    t.push(range('size', sizeMin, sizeMax, neg.size));
    t.push(range('params', paramsMin, paramsMax, neg.params));
    if (shard.trim()) t.push(`${not(neg.shard)}shard:${shard.trim()}`);
    return t.filter(Boolean);
  }

  $: preview = assemble().join('  ');
  function apply() {
    addFilterTerms(assemble());
  }
</script>

<div class="builder">
  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={neg.dtype} title="negate" /></label>
    <span class="k">dtype</span>
    <div class="chips">
      {#each present as d}
        <button class="chip" class:on={dtypes.has(d)} on:click={() => toggleDtype(d)}>
          <Dtype dtype={d} bubble={false} />
        </button>
      {/each}
    </div>
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={neg.name} title="negate" /></label>
    <span class="k">name</span>
    <select bind:value={nameMode} aria-label="name match mode">
      <option value="contains">contains</option>
      <option value="re">regex</option>
      <option value="glob">glob</option>
    </select>
    <input class="v" spellcheck="false" placeholder="q_proj  /  ^model\.layers  /  *.weight" bind:value={name} />
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={neg.shape} title="negate" /></label>
    <span class="k">shape</span>
    <input class="v" spellcheck="false" placeholder="6,_,42   (_ = any dim, .. = any run)" bind:value={shape} />
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={neg.dim} title="negate" /></label>
    <span class="k">dim</span>
    <input class="v short" spellcheck="false" placeholder="4096  /  >1000" bind:value={dim} />
    <span class="k">rank</span>
    <label class="not"><input type="checkbox" bind:checked={neg.rank} title="negate" /></label>
    <input class="v short" spellcheck="false" placeholder="2  /  >=3" bind:value={rank} />
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={neg.size} title="negate" /></label>
    <span class="k">size</span>
    <input class="v short" spellcheck="false" placeholder="1MiB" bind:value={sizeMin} />
    <span class="to">…</span>
    <input class="v short" spellcheck="false" placeholder="1GiB" bind:value={sizeMax} />
    <span class="k">params</span>
    <label class="not"><input type="checkbox" bind:checked={neg.params} title="negate" /></label>
    <input class="v short" spellcheck="false" placeholder="1M" bind:value={paramsMin} />
    <span class="to">…</span>
    <input class="v short" spellcheck="false" placeholder="1B" bind:value={paramsMax} />
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={neg.shard} title="negate" /></label>
    <span class="k">shard</span>
    <input class="v" spellcheck="false" placeholder="00001  /  model-00001" bind:value={shard} />
  </div>

  <div class="foot">
    <code class="preview">{preview || '—'}</code>
    <button class="add" disabled={!preview} on:click={apply}>Add to filter</button>
  </div>
</div>

<style>
  .builder {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 14px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .k {
    color: var(--fg-dim);
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.04em;
    min-width: 40px;
  }
  .not {
    display: inline-flex;
    align-items: center;
  }
  .not input {
    margin: 0;
    cursor: pointer;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .chip {
    background: none;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 0;
    cursor: pointer;
    opacity: 0.55;
  }
  .chip.on {
    opacity: 1;
    border-color: var(--accent);
  }
  .v {
    flex: 1 1 200px;
    min-width: 0;
    font-family: ui-monospace, monospace;
    font-size: 12px;
  }
  .v.short {
    flex: 0 0 84px;
  }
  .to {
    color: var(--fg-dim);
  }
  select {
    font-size: 12px;
    padding: 1px 2px;
  }
  .foot {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 2px;
  }
  .preview {
    flex: 1 1 auto;
    min-width: 0;
    color: var(--accent);
    font-family: ui-monospace, monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .add {
    flex: 0 0 auto;
    background: color-mix(in srgb, var(--accent) 18%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
    border-radius: 4px;
    color: var(--fg);
    cursor: pointer;
    font: inherit;
    padding: 2px 10px;
  }
  .add:disabled {
    opacity: 0.4;
    cursor: default;
  }
</style>
