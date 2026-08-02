<script lang="ts">
  /**
   * Turn a comparison round — the one control, on both diff screens.
   *
   * They had a button each, built and styled separately: the same glyph, the same word and a
   * different padding, and the accent on the `⇄` had to be added twice before the two matched. What
   * differs between the screens is *where* it sits (a grid column in the report's header, between the
   * two boxes on the side-by-side), which is the caller's business — so the caller positions the
   * element it wraps this in, and what the control *is* lives here.
   *
   * A control, not a glyph: a bare dim `⇄` beside a path was invisible, and reported as missing.
   */
  export let onSwap: () => void;
  /** Inert while a comparison is being read, where the caller has such a state. */
  export let disabled = false;
  /** The full sentence, including the key that does the same thing on that screen. */
  export let title = 'Swap the two sides';
</script>

<button type="button" class="swap" {title} aria-label="Swap the two sides" {disabled} on:click={onSwap}
  ><span class="glyph" aria-hidden="true">⇄</span> Swap</button
>

<style>
  .swap {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font: inherit;
    font-size: 12px;
    line-height: 1;
    color: var(--fg);
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 5px 11px;
    cursor: pointer;
  }
  /* The glyph is the thing the eye finds; the word is what confirms it. */
  .glyph {
    font-size: 15px;
    color: var(--accent);
  }
  .swap:hover:not(:disabled) {
    color: var(--accent);
    border-color: var(--accent);
    background: var(--bg-hover);
  }
  .swap:disabled {
    opacity: 0.6;
    cursor: default;
  }
</style>
