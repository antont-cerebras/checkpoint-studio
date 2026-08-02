<script lang="ts">
  /**
   * One row of the diff report: a mark, a name, and what changed about it.
   *
   * Every part is a separate element so it can be *pointed at*: the path dimmed and the leaf bright
   * (a report is read leaf-first — `qscale` among sixty identical prefixes), the two dtypes coloured
   * old-red/new-green like the terminal, and **only the dimensions that actually differ** marked. A row
   * used to be one string, which left finding the difference to the reader on every one of thousands of
   * rows.
   */
  import { shapeDiff, splitName } from '../lib/difflines';
  import type { TensorSig } from '../lib/types';

  /** `+` added, `-` removed, `~` changed — the terminal's marks, and the same colours. */
  export let mark: '+' | '-' | '~';
  export let name: string;
  /** The two signatures. `old`/`new` alone for a one-sided row; both for a change. */
  export let old: TensorSig | null = null;
  export let neu: TensorSig | null = null;
  /** `×256 → ×1` when an alignment folded this row — see `--align-fused`. */
  export let fold = '';
  /** How many tensors this row stands for, when it is a *family* row (`model.layers.{0-61}.…`). `1`
   * means one tensor and shows nothing — a `(×1)` on every ordinary row would be noise. */
  export let count = 1;
  /** Clickable when the tensor is in the checkpoint being served, so a finding can be opened. */
  export let onOpen: ((name: string) => void) | null = null;
  /** Why it is not clickable, when it is not — a removed tensor exists only in the baseline. */
  export let why = '';
  /** Identifies this row for `n`/`N`, which scroll it into view by this attribute. Unique across the
   * report: a name can appear in two sections (removed here, added there, under a rename). */
  export let rowId = '';
  /** The row `n`/`N` have stepped to. Marked, not focused: focus would scroll on its own terms and
   * steal the caret from the *Find in results* box. */
  export let cursor = false;

  $: parts = splitName(name);
  $: dims = old && neu ? shapeDiff(old.shape, neu.shape) : null;
  // Which half moved decides how the row is written — see the branches below.
  $: dtypeChanged = !!old && !!neu && old.dtype !== neu.dtype;
  $: shapeChanged = !!old && !!neu && old.shape.join() !== neu.shape.join();
</script>

<div
  class="row {mark === '+' ? 'added' : mark === '-' ? 'removed' : 'changed'}"
  class:cursor
  data-row={rowId || null}
>
  <span class="mark" aria-hidden="true">{mark}</span>
  {#if onOpen}
    <button type="button" class="name" title="Open {name}" on:click={() => onOpen?.(name)}>
      <span class="path">{parts.path}</span><span class="leaf">{parts.leaf}</span>
    </button>
  {:else}
    <span class="name" title={why}>
      <span class="path">{parts.path}</span><span class="leaf">{parts.leaf}</span>
    </span>
  {/if}

  <span class="sig">
    {#if old && neu}
      {#if !dtypeChanged && shapeChanged}
        <!-- Only the shape moved: say the dtype once. Repeating it either side of the arrow made the
             reader compare two strings to find out that half of them were the same. -->
        <span class="dtype">{old.dtype}</span>
        <span class="shape"
          >({#each old.shape as d, i (i)}<span class:diff={dims?.old[i]}>{d}</span>{#if i < old.shape.length - 1}<span
                class="sep">, </span>{/if}{/each})</span>
        <span class="arrow" aria-hidden="true">→</span>
        <span class="shape"
          >({#each neu.shape as d, i (i)}<span class:diff={dims?.new[i]}>{d}</span>{#if i < neu.shape.length - 1}<span
                class="sep">, </span>{/if}{/each})</span>
      {:else if dtypeChanged && !shapeChanged}
        <!-- Only the dtype changed — the case that was hardest to see, because the two identical shapes
             either side of the arrow *look* like the row's content and the one thing that differed was
             two coloured words among them. Now the shape is stated once, and the change is the row. -->
        <span class="dtype was">{old.dtype}</span>
        <span class="arrow" aria-hidden="true">→</span>
        <span class="dtype now">{neu.dtype}</span>
        <span class="shape">({old.shape.join(', ')})</span>
      {:else if dtypeChanged && shapeChanged}
        <span class="side">
          <span class="dtype was">{old.dtype}</span>
          <span class="shape"
            >({#each old.shape as d, i (i)}<span class:diff={dims?.old[i]}>{d}</span>{#if i < old.shape.length - 1}<span
                  class="sep">, </span>{/if}{/each})</span>
        </span>
        <span class="arrow" aria-hidden="true">→</span>
        <span class="side">
          <span class="dtype now">{neu.dtype}</span>
          <span class="shape"
            >({#each neu.shape as d, i (i)}<span class:diff={dims?.new[i]}>{d}</span>{#if i < neu.shape.length - 1}<span
                  class="sep">, </span>{/if}{/each})</span>
        </span>
      {:else}
        <!-- Same dtype, same shape: the report only calls that changed when the *values* differ. -->
        <span class="side">
          <span class="dtype">{old.dtype}</span>
          <span class="shape">({old.shape.join(', ')})</span>
        </span>
        <span class="kind">values differ</span>
      {/if}
      {#if fold}<span class="fold" title="What this row stands for on each side">{fold}</span>{/if}
      <!-- Parenthesised, and the fold is not: the two counts mean different things — how many tensors
           this *name template* covers, against how many tensors each *side of one name* holds — and
           `×256 → ×1 ×62` side by side read as one number too many. -->
      {#if count > 1}<span class="count" title="{count.toLocaleString()} tensors share this name template"
          >(×{count.toLocaleString()})</span
        >{/if}
    {:else if old ?? neu}
      {@const only = old ?? neu}
      <span class="side">
        <span class="dtype">{only?.dtype}</span>
        <span class="shape">({(only?.shape ?? []).join(', ')})</span>
      </span>
      {#if count > 1}<span class="count" title="{count.toLocaleString()} tensors share this name template"
          >(×{count.toLocaleString()})</span
        >{/if}
    {/if}
  </span>
</div>

<style>
  /* Where `n`/`N` have got to. A background rather than an outline, the treatment the tree's cursor
     and the palette's selection use, so "the row I am on" looks the same everywhere. */
  .row.cursor {
    background: var(--bg-hover);
    box-shadow: inset 2px 0 0 var(--accent);
  }
  .row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 1px 0 1px 8px;
    font-size: 12.5px;
    /* Capped, so the signature column stays near the names it belongs to. Right-aligned signatures are
       what make a list of shapes scannable; a screen's width of gap between name and shape is not. */
    max-width: 150ch;
  }
  .row:hover {
    background: var(--bg-hover);
  }
  .mark {
    flex: none;
    width: 1em;
    font-weight: 600;
  }
  .added .mark {
    color: var(--ok, #4ec94e);
  }
  .removed .mark {
    color: var(--err, #e05c5c);
  }
  .changed .mark {
    color: var(--warn, #d8b530);
  }
  .name {
    flex: 1 1 auto;
    min-width: 0;
    font: inherit;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--fg);
    background: none;
    border: none;
    padding: 0;
    text-align: left;
    word-break: break-all;
  }
  button.name {
    cursor: pointer;
  }
  button.name:hover .leaf {
    color: var(--accent);
    text-decoration: underline;
  }
  /* Where it lives, dimmed; what it *is*, bright. */
  .path {
    color: var(--fg-dim);
  }
  .leaf {
    color: var(--fg);
  }
  .sig {
    flex: none;
    display: flex;
    align-items: baseline;
    gap: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    color: var(--fg-dim);
    white-space: nowrap;
  }
  /* Old red, new green — the terminal's convention, and `diff`'s own. Applied to the dtype itself
     rather than to a side, since a dtype-only row has no two sides to colour. */
  .dtype.was {
    color: var(--err, #e05c5c);
    font-weight: 600;
  }
  .dtype.now {
    color: var(--ok, #4ec94e);
    font-weight: 600;
  }
  .kind {
    color: var(--warn);
  }
  /* A space between the dtype and its shape, which `F32(16)` was missing. */
  .side {
    display: inline-flex;
    align-items: baseline;
    gap: 4px;
  }
  /* Only the dimensions that moved. The rest stay dim, so the eye lands on the change. */
  .shape .diff {
    color: var(--warn, #d8b530);
    font-weight: 600;
  }
  .sep {
    color: var(--fg-dim);
  }
  .arrow {
    color: var(--fg-dim);
  }
  .fold {
    color: var(--warn);
    font-variant-numeric: tabular-nums;
  }
  /* What a family row stands for. Dim: it is the row's size, not its subject. */
  .count {
    color: var(--fg-dim);
    font-variant-numeric: tabular-nums;
  }
</style>
