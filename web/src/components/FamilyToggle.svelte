<script lang="ts">
  /**
   * `Fold layer families` — the switch both diff screens carry, worded and behaving alike.
   *
   * It read `Collapse families` — two words that name no object: families *of what*, and collapse in
   * what sense. What it folds is the run of tensors whose names differ only by an index, layer by
   * layer or expert by expert, onto one row that says how many.
   *
   * The state it edits is `full` (every layer as its own row), which is what the URL and the server
   * call it; the checkbox is its inverse, because "collapse families" is the thing being asked for.
   * That inversion was written out twice, which is one place too many for a control whose two
   * spellings are opposites.
   */
  export let full: boolean;
  export let onChange: (full: boolean) => void;

  function toggled(e: Event & { currentTarget: HTMLInputElement }) {
    onChange(!e.currentTarget.checked);
  }
</script>

<label
  class="only"
  title="Fold runs of tensors whose names differ only by an index — 62 layers onto one row reading model.layers.&#123;0-61&#125;.weight (×62). The terminal's `k`."
>
  <input type="checkbox" checked={!full} on:change={toggled} />
  Fold layer families
</label>

<style>
  .only {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: var(--fg-dim);
    cursor: pointer;
    white-space: nowrap;
  }
</style>
