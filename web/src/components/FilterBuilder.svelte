<script lang="ts">
  // A mouse-driven view of the filter query that stays in sync with the raw text
  // input: it parses the current query into facet controls, and every control edits
  // the query live (rebuilding it), so raw ⇄ builder are two views of one query.
  // Terms the builder doesn't model (bare words, unusual facets) are preserved.
  import { tree } from '../stores/server';
  import { filterQuery } from '../stores/view';
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

  interface Neg {
    dtype: boolean;
    name: boolean;
    shape: boolean;
    dim: boolean;
    rank: boolean;
    size: boolean;
    params: boolean;
    shard: boolean;
  }
  interface Fields {
    dtypes: Set<string>;
    name: string;
    nameMode: 'contains' | 're' | 'glob';
    shape: string;
    dim: string;
    rank: string;
    sizeMin: string;
    sizeMax: string;
    paramsMin: string;
    paramsMax: string;
    shard: string;
    neg: Neg;
    rest: string[]; // terms the builder doesn't model — preserved verbatim
  }
  const blank = (): Fields => ({
    dtypes: new Set(),
    name: '',
    nameMode: 'contains',
    shape: '',
    dim: '',
    rank: '',
    sizeMin: '',
    sizeMax: '',
    paramsMin: '',
    paramsMax: '',
    shard: '',
    neg: { dtype: false, name: false, shape: false, dim: false, rank: false, size: false, params: false, shard: false },
    rest: [],
  });

  // Split a query into terms, keeping `(…)` groups and `"…"` intact.
  function tokenize(q: string): string[] {
    const out: string[] = [];
    let cur = '';
    let depth = 0;
    let quote = false;
    for (const c of q) {
      if (c === '"') quote = !quote;
      else if (c === '(' && !quote) (depth++, (cur += c));
      else if (c === ')' && !quote) ((depth = Math.max(0, depth - 1)), (cur += c));
      else if (/\s/.test(c) && depth === 0 && !quote) {
        if (cur) out.push(cur), (cur = '');
      } else cur += c;
    }
    if (cur) out.push(cur);
    return out;
  }

  function splitRange(v: string): [string, string] {
    if (v.includes('..')) {
      const i = v.indexOf('..');
      return [v.slice(0, i).trim(), v.slice(i + 2).trim()];
    }
    if (v.startsWith('>=')) return [v.slice(2).trim(), ''];
    if (v.startsWith('>')) return [v.slice(1).trim(), ''];
    if (v.startsWith('<=')) return ['', v.slice(2).trim()];
    if (v.startsWith('<')) return ['', v.slice(1).trim()];
    return [v.trim(), v.trim()];
  }

  function parseQuery(q: string): Fields {
    const f = blank();
    for (let tok of tokenize(q)) {
      let negate = false;
      if (tok.startsWith('!')) {
        negate = true;
        tok = tok.slice(1);
      }
      const ci = tok.indexOf(':');
      if (ci < 0) {
        f.rest.push((negate ? '!' : '') + tok);
        continue;
      }
      const facet = tok.slice(0, ci);
      const val = tok.slice(ci + 1);
      switch (facet) {
        case 'dtype':
          val.split(',').forEach((d) => d.trim() && f.dtypes.add(d.trim()));
          f.neg.dtype = negate;
          break;
        case 'name':
          if (val.startsWith('re:')) (f.nameMode = 're'), (f.name = val.slice(3));
          else if (val.startsWith('glob:')) (f.nameMode = 'glob'), (f.name = val.slice(5));
          else (f.nameMode = 'contains'), (f.name = val);
          f.neg.name = negate;
          break;
        case 'shape':
          f.shape = val;
          f.neg.shape = negate;
          break;
        case 'dim':
          f.dim = val;
          f.neg.dim = negate;
          break;
        case 'rank':
          f.rank = val;
          f.neg.rank = negate;
          break;
        case 'size':
          [f.sizeMin, f.sizeMax] = splitRange(val);
          f.neg.size = negate;
          break;
        case 'params':
          [f.paramsMin, f.paramsMax] = splitRange(val);
          f.neg.params = negate;
          break;
        case 'shard':
          f.shard = val;
          f.neg.shard = negate;
          break;
        default:
          f.rest.push((negate ? '!' : '') + tok);
      }
    }
    return f;
  }

  const not = (on: boolean) => (on ? '!' : '');
  function rangeTerm(facet: string, min: string, max: string, negate: boolean): string {
    const m = min.trim();
    const x = max.trim();
    let v = '';
    if (m && x) v = m === x ? m : `${m}..${x}`;
    else if (m) v = `${m}..`;
    else if (x) v = `..${x}`;
    else return '';
    return `${not(negate)}${facet}:${v}`;
  }

  function buildQuery(f: Fields): string {
    const t: string[] = [];
    if (f.dtypes.size) t.push(`${not(f.neg.dtype)}dtype:${[...f.dtypes].join(',')}`);
    if (f.name.trim()) {
      const pfx = f.nameMode === 'contains' ? '' : `${f.nameMode}:`;
      t.push(`${not(f.neg.name)}name:${pfx}${f.name.trim()}`);
    }
    if (f.shape.trim()) {
      let s = f.shape.trim();
      if (!s.startsWith('(')) s = `(${s})`;
      t.push(`${not(f.neg.shape)}shape:${s}`);
    }
    if (f.dim.trim()) t.push(`${not(f.neg.dim)}dim:${f.dim.trim()}`);
    if (f.rank.trim()) t.push(`${not(f.neg.rank)}rank:${f.rank.trim()}`);
    const sz = rangeTerm('size', f.sizeMin, f.sizeMax, f.neg.size);
    if (sz) t.push(sz);
    const pa = rangeTerm('params', f.paramsMin, f.paramsMax, f.neg.params);
    if (pa) t.push(pa);
    if (f.shard.trim()) t.push(`${not(f.neg.shard)}shard:${f.shard.trim()}`);
    t.push(...f.rest);
    return t.join(' ');
  }

  let fields = parseQuery($filterQuery);
  let lastBuilt = $filterQuery;
  // Reflect EXTERNAL query changes (raw input, badges) into the controls; our own
  // writes set `lastBuilt` first, so they don't round-trip and disturb typing.
  $: if ($filterQuery !== lastBuilt) {
    fields = parseQuery($filterQuery);
    lastBuilt = $filterQuery;
  }
  function commit() {
    const q = buildQuery(fields);
    lastBuilt = q;
    filterQuery.set(q);
  }
  function toggleDtype(d: string) {
    if (fields.dtypes.has(d)) fields.dtypes.delete(d);
    else fields.dtypes.add(d);
    fields = fields;
    commit();
  }

  // Move keyboard focus into the builder on open, and keep its keystrokes from
  // triggering the tree's global shortcuts.
  function focusOnMount(node: HTMLElement) {
    node.focus();
  }
</script>

<!-- svelte-ignore a11y-no-static-element-interactions -->
<div class="builder" on:keydown={(e) => e.stopPropagation()}>
  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={fields.neg.dtype} on:change={commit} title="negate" /></label>
    <span class="k">dtype</span>
    <div class="chips">
      {#each present as d}
        <button class="chip" class:on={fields.dtypes.has(d)} on:click={() => toggleDtype(d)}>
          <Dtype dtype={d} bubble={false} />
        </button>
      {/each}
    </div>
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={fields.neg.name} on:change={commit} title="negate" /></label>
    <span class="k">name</span>
    <select bind:value={fields.nameMode} on:change={commit} aria-label="name match mode">
      <option value="contains">contains</option>
      <option value="re">regex</option>
      <option value="glob">glob</option>
    </select>
    <input class="v" use:focusOnMount spellcheck="false" placeholder="q_proj  /  ^model\.layers  /  *.weight" bind:value={fields.name} on:input={commit} />
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={fields.neg.shape} on:change={commit} title="negate" /></label>
    <span class="k">shape</span>
    <input class="v" spellcheck="false" placeholder="6,_,42   (_ = any dim, .. = any run)" bind:value={fields.shape} on:input={commit} />
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={fields.neg.dim} on:change={commit} title="negate" /></label>
    <span class="k">dim</span>
    <input class="v short" spellcheck="false" placeholder="4096  /  >1000" bind:value={fields.dim} on:input={commit} />
    <label class="not"><input type="checkbox" bind:checked={fields.neg.rank} on:change={commit} title="negate" /></label>
    <span class="k">rank</span>
    <input class="v short" spellcheck="false" placeholder="2  /  >=3" bind:value={fields.rank} on:input={commit} />
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={fields.neg.size} on:change={commit} title="negate" /></label>
    <span class="k">size</span>
    <input class="v short" spellcheck="false" placeholder="1MiB" bind:value={fields.sizeMin} on:input={commit} />
    <span class="to">…</span>
    <input class="v short" spellcheck="false" placeholder="1GiB" bind:value={fields.sizeMax} on:input={commit} />
    <label class="not"><input type="checkbox" bind:checked={fields.neg.params} on:change={commit} title="negate" /></label>
    <span class="k">params</span>
    <input class="v short" spellcheck="false" placeholder="1M" bind:value={fields.paramsMin} on:input={commit} />
    <span class="to">…</span>
    <input class="v short" spellcheck="false" placeholder="1B" bind:value={fields.paramsMax} on:input={commit} />
  </div>

  <div class="row">
    <label class="not"><input type="checkbox" bind:checked={fields.neg.shard} on:change={commit} title="negate" /></label>
    <span class="k">shard</span>
    <input class="v" spellcheck="false" placeholder="00001  /  model-00001" bind:value={fields.shard} on:input={commit} />
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
</style>
