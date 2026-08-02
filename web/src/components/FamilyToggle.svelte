<script lang="ts">
  /**
   * `Collapse families` — the switch both diff screens carry, worded and behaving alike.
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

<label class="only">
  <input type="checkbox" checked={!full} on:change={toggled} />
  Collapse families
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
