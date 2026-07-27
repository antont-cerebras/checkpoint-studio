<script lang="ts">
  // The compact view: the tensor tree with uniform layer / expert stacks folded into one
  // templated subtree each, so what stays visible is the irregularities. A 48-layer MoE
  // with 31k tensors reads as ~25 rows, and the one layer that has an extra tensor — or a
  // dtype its siblings don't share — is immediately obvious next to its folded neighbours.
  //
  // It is a TREE, deliberately. This was a flat per-family list, which destroys the
  // nesting — and the nesting is exactly what makes an outlier stand out from its
  // siblings. The rows come from the same `flatten` the tree view uses, over the same
  // `TreeNode` shape, so folding and indentation behave identically in both.
  //
  // The folding is `crate::compact::compact_tree`, which is `diff`'s own family
  // collapsing — so a "family" means the same thing here and in a diff.
  import { expandedIds, flatten, type Row } from '../lib/flatten';
  import { humanCount, humanSize } from '../lib/format';
  import { expanded, filterQuery, toggle } from '../stores/view';
  import { compactError, compactTree, loadCompact } from '../stores/server';
  import Dtype from './Dtype.svelte';
  import Shape from './Shape.svelte';

  // Fold state is the *shared* `expanded` store, not this component's own — so the tree
  // view's existing controls all work here unchanged: `e` / `c`, the palette's expand /
  // collapse all, and clicking a group. (They used to do nothing in this view, because
  // `setAllExpanded` walked the full tree, whose ids don't occur in the folded one.)
  $: data = $compactTree;
  $: err = $compactError;
  $: void refresh($filterQuery);

  /** Load, then seed the fold state from what actually landed — so the view opens at the
   * depth the server sent, the way the tree view does. After that the user owns folding,
   * so this runs once per loaded tree rather than on every store change. */
  async function refresh(q: string) {
    const t = await loadCompact(q);
    if (t) expanded.set(expandedIds(t.tree));
  }

  $: rows = data ? flatten(data.tree, $expanded) : ([] as Row[]);
  $: familyCount = data ? Object.keys(data.counts).length : 0;

  /** How many real tensors a family row stands for. */
  function count(row: Row): number {
    if (row.node.kind !== 'tensor' || !data) return 0;
    return data.counts[row.node.info.name] ?? 0;
  }
  /** Which attributes disagree across a family's members, when any do. */
  function varies(row: Row): { dtype: boolean; shape: boolean } | undefined {
    if (row.node.kind !== 'tensor' || !data) return undefined;
    return data.varying[row.node.info.name];
  }
</script>

<div class="compact">
  {#if err}
    <p class="err">{err}</p>
  {:else if !data}
    <p class="dim">folding…</p>
  {:else if data}
    <div class="hdr">
      <span>
        {humanCount(data.tensor_count)} tensors in <strong>{familyCount}</strong>
        {familyCount === 1 ? 'family' : 'families'}
      </span>
    </div>
    <div class="rows">
      {#each rows as row (row.id)}
        {@const n = row.node}
        <div class="row" style="padding-left:{row.depth * 14 + 4}px">
          {#if n.kind === 'group'}
            <button class="grp" on:click={() => toggle(row.id)}>
              <span class="arrow">{$expanded.has(row.id) ? '▾' : '▸'}</span>
              <span class="gname">{n.name}</span>
              <span class="dim">▦ {n.tensor_count} · {humanSize(n.total_size)}</span>
            </button>
          {:else if n.kind === 'tensor'}
            {@const v = varies(row)}
            <span class="mark dim">·</span>
            <span class="name">{n.label ?? n.info.name}</span>
            <span class="times" title="how many real tensors this family stands for"
              >×{count(row)}</span
            >
            {#if v?.dtype}
              <span class="varies" title="the members of this family do not share one dtype"
                >dtype varies</span
              >
            {:else}
              <Dtype dtype={n.info.dtype} />
            {/if}
            {#if v?.shape}
              <span class="varies" title="the members of this family do not share one shape"
                >shape varies</span
              >
            {:else}
              <Shape shape={n.info.shape} />
            {/if}
            <span class="dim">{humanSize(n.info.size_bytes)}</span>
          {/if}
        </div>
      {/each}
      {#if !rows.length}<p class="dim">nothing matches the filter</p>{/if}
    </div>
  {/if}
</div>

<style>
  .compact {
    height: 100%;
    overflow: auto;
    font-size: 12.5px;
  }
  .hdr {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
    position: sticky;
    top: 0;
  }
  .rows {
    padding: 4px 0;
  }
  .row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 1px 8px 1px 0;
    white-space: nowrap;
  }
  .row:hover {
    background: var(--bg-hover, rgba(127, 127, 127, 0.12));
  }
  /* A group header is the fold's click target, so it is a real button — but it has to
     read as a tree row, not as a control. */
  .grp {
    display: flex;
    align-items: baseline;
    gap: 6px;
    padding: 0;
    font: inherit;
    color: inherit;
    background: none;
    border: 0;
    cursor: pointer;
  }
  .arrow,
  .gname {
    color: var(--accent);
  }
  .mark {
    width: 1em;
  }
  /* The multiplier is the whole point of a folded row — give it weight. */
  .times {
    font-weight: 600;
  }
  .varies {
    padding: 0 4px;
    color: var(--warn, #d8b530);
    border: 1px solid currentColor;
    border-radius: 3px;
    font-size: 11px;
  }
  .dim {
    color: var(--fg-dim);
  }
  .err {
    padding: 10px;
    color: var(--err, #e05c5c);
  }
</style>
