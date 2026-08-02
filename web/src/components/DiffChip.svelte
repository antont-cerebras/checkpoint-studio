<script lang="ts">
  /**
   * One count from a comparison: `1 added`, `620 removed`, `2 metadata`.
   *
   * Both diff screens show this strip, and each had its own copy — with its own greens. `added` was
   * `#3fb950` on the side-by-side and `#4ec94e` on the report, `removed` `--danger` against
   * `--err`: the same fact in two colours, on two screens a click apart. The tone is a named kind
   * here, so there is one answer to "what colour is added".
   *
   * A chip is a button when it *does* something (the report's scroll to a section) and a plain span
   * when it does not (the side-by-side's counts, which describe the tree already on screen). Same
   * shape either way — the difference is whether there is somewhere to go.
   */
  export let tone: 'same' | 'added' | 'removed' | 'changed' | 'meta';
  /** The count, already localised by the caller — it knows whether the number is a total or a
   * filtered subset. */
  export let count: string;
  /** What the count is of: `added`, `unchanged`, `metadata`. */
  export let label: string;
  /** Given, the chip is a button that goes somewhere. */
  export let onPick: (() => void) | null = null;
  /** A chip standing for nothing stays quiet rather than inviting a click that reveals "none". */
  export let empty = false;
  export let title = '';
  /** Which way round to read it: the report leads with the word, the side-by-side with the number. */
  export let order: 'count-first' | 'label-first' = 'count-first';
</script>

{#if onPick}
  <button
    type="button"
    class="chip {tone}"
    class:empty
    class:pick={true}
    {title}
    aria-pressed={!empty}
    on:click={onPick}
  >
    {#if order === 'label-first'}{label} <b>{count}</b>{:else}{count} <b>{label}</b>{/if}
  </button>
{:else}
  <span class="chip {tone}" class:empty {title}>
    {#if order === 'label-first'}{label} <b>{count}</b>{:else}{count} <b>{label}</b>{/if}
  </span>
{/if}

<style>
  .chip {
    display: inline-block;
    font: inherit;
    font-size: 12px;
    color: var(--fg-dim);
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 2px 10px;
    font-variant-numeric: tabular-nums;
  }
  .pick {
    cursor: pointer;
  }
  .pick:hover {
    color: var(--fg);
  }
  .empty {
    opacity: 0.55;
  }
  b {
    font-weight: 600;
  }
  /* One colour per kind, for both screens. */
  .added b {
    color: var(--ok, #3fb950);
  }
  .removed b {
    color: var(--danger);
  }
  .changed b {
    color: var(--warn);
  }
  .meta b {
    color: var(--accent);
  }
</style>
