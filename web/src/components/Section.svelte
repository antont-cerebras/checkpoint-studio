<script lang="ts">
  /**
   * A foldable section of the diff report: a heading that says what and how many, and a body.
   *
   * The report was five flat lists one after another, so a re-quantization's `Tensors changed (747)`
   * pushed everything below it off the screen and there was no way to put it away. Folding is the
   * cheapest possible answer to "I want to see the other four".
   *
   * The heading is a `<button>` because it does something. `aria-expanded` and `aria-controls` tie it to
   * the body, so a screen reader announces the fold state rather than reading a heading that silently
   * does nothing.
   */
  export let title: string;
  /** Shown after the title. A string, not a number, because the caller may be reporting `12 of 31,247`
   * (filtered) and the difference matters. */
  export let count = '';
  /** Whether the body is showing. **Owned by the caller**, which keeps it in the URL — folding a
   * 31,247-row section away and having it spring back on reload is the same complaint as a filter that
   * does not survive one. */
  export let open = true;
  /** Told when the heading is clicked. Without one the component owns nothing and the caller's state
   * cannot follow — the reason this is not a two-way binding. */
  export let onToggle: ((open: boolean) => void) | null = null;
  /** Colours the heading the way the marks below it are coloured — added green, removed red, changed
   * yellow — so the eye can find a section without reading. */
  export let tone: 'added' | 'removed' | 'changed' | 'meta' | 'plain' = 'plain';
  /** A one-line note beside the count: what a section says about itself when it is empty *for a
   * reason* (`not compared (filtered subset)`), rather than being empty. */
  export let note = '';

  /** Distinct per instance, so `aria-controls` points at this body and no other. */
  const id = `section-${Math.random().toString(36).slice(2, 9)}`;
</script>

<section class:folded={!open}>
  <h3 class={tone}>
    <button
      type="button"
      aria-expanded={open}
      aria-controls={id}
      on:click={() => onToggle?.(!open)}
    >
      <span class="caret" aria-hidden="true">{open ? '▾' : '▸'}</span>{title}{#if count}
        <span class="count">({count})</span>{/if}{#if note}
        <span class="note">— {note}</span>{/if}
    </button>
  </h3>
  <div {id} hidden={!open}>
    <slot />
  </div>
</section>

<style>
  section {
    margin-bottom: 12px;
  }
  /* Folded, the heading is the whole section: no gap below it pretending something is there. */
  section.folded {
    margin-bottom: 4px;
  }
  h3 {
    margin: 0 0 4px;
    font-size: 12.5px;
    font-weight: 600;
  }
  button {
    display: inline-flex;
    align-items: baseline;
    gap: 5px;
    font: inherit;
    font-weight: 600;
    color: inherit;
    background: none;
    border: none;
    padding: 1px 0;
    cursor: pointer;
  }
  button:hover .count,
  button:hover .caret {
    color: var(--fg);
  }
  .caret {
    color: var(--fg-dim);
    font-weight: 400;
    /* Fixed width so the titles line up whichever way the carets point. */
    width: 0.8em;
    display: inline-block;
  }
  .count,
  .note {
    color: var(--fg-dim);
    font-weight: 400;
  }
  /* The section's own colour, matching the +/-/~ marks in its rows. */
  .added {
    color: var(--ok, #4ec94e);
  }
  .removed {
    color: var(--err, #e05c5c);
  }
  .changed {
    color: var(--warn, #d8b530);
  }
  .meta {
    color: var(--accent);
  }
</style>
