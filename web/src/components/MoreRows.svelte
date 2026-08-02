<script lang="ts">
  /**
   * The tail of a section the cap is holding rows back from.
   *
   * One component rather than the same twelve lines under each of four sections — which is how the
   * count came to be computed from the wrong list in two of them: a folded section draws *family*
   * rows and the tail counted *tensors*, so five rows on screen were followed by "show the remaining
   * 79,532" and pressing it produced nothing more. The number a reader is offered has to be the number
   * of rows they will get, so the caller passes exactly that.
   */

  /** How many rows are being held back — always of the same list the section is drawing. */
  export let n: number;
  export let onShowAll: () => void;
  /** Offered past `BROWSE_AT`, where a flat list is the wrong instrument. */
  export let onBrowse: (() => void) | null = null;
</script>

{#if n > 0}
  <p class="more">
    <button type="button" on:click={onShowAll}>
      Show {n.toLocaleString()} more {n === 1 ? 'row' : 'rows'}
    </button>
    {#if onBrowse}
      <!-- Past a few thousand, a flat list is the wrong instrument: the aligned tree is virtualized
           and folds, and this one puts every row in the DOM. -->
      <button type="button" on:click={onBrowse}>or open in Browse</button>
    {/if}
  </p>
{/if}

<style>
  .more {
    margin: 4px 0 0 8px;
    font-size: 12px;
  }
  .more button {
    font: inherit;
    font-size: 12px;
    color: var(--accent);
    background: none;
    border: none;
    padding: 0 6px 0 0;
    cursor: pointer;
    text-decoration: underline;
  }
</style>
