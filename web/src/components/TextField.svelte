<script lang="ts">
  /**
   * The text field this app types into — one component, so every box behaves and looks the same.
   *
   * **Behaviour first.** A field owns every key: a keystroke that lands here never reaches a screen's
   * shortcuts, so `s` does not swap a comparison, `n` does not step to the next difference and `:` types
   * a colon instead of opening the palette. That last one was the bug that prompted this: the proxy
   * shorthand `:/path`, which these very placeholders offer, could not be typed into any box.
   *
   * Stopping propagation here rather than only fixing the handlers is deliberate. There are five window
   * handlers today and there will be more; each one that forgets the rule steals keys from every box on
   * its screen. Here the field defends itself, and `lib/keys::isEditable` is the same rule for the
   * handlers that legitimately watch the window (a `|capture` handler still sees the key first — that is
   * how the address bar's dropdown keeps its Escape).
   *
   * **Look second**, but for the same reason: the same eight CSS declarations were repeated in six
   * components, drifting by a pixel of padding and a shade of background.
   */

  /** `bare` is the address bar's borderless look; `dense` is the filter and scope builders' tighter
   * grid. Named rather than three booleans, since a field is exactly one of them. */
  export let variant: 'boxed' | 'bare' | 'dense' = 'boxed';
  /** Rows for a multi-line field; `0` renders an `<input>`. The scope bar's name and rename-rule boxes
   * take one entry per line, and they are the same field as the rest. */
  export let rows = 0;
  /** Fill the space the parent gives it. Off for the short numeric-ish boxes in the filter builder. */
  export let grow = true;
  /** An explicit width (`'84px'`, `'28ch'`) for a box that should not fill. */
  export let width = '';
  export let value = '';
  /**
   * Declared props, not passed through with the rest — and that is not a style choice.
   *
   * A spread puts `readonly={false}` on the element as the *string* `"false"`, and an input with a
   * `readonly` attribute of any value is read-only. Every box in the app went uneditable the moment
   * they were spread. Declared here, Svelte treats them as the boolean attributes they are.
   */
  export let readonly = false;
  export let disabled = false;
  /** The element itself, for a caller that needs to focus or select it. */
  export let el: HTMLInputElement | HTMLTextAreaElement | null = null;

  /**
   * Keys typed here go no further.
   *
   * Escape blurs, which is what the global handler used to do for fields — kept here so the behaviour
   * travels with the field rather than depending on a handler that may or may not be listening.
   */
  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && el) el.blur();
    // Ancestors only: a `stopPropagation` does not stop the *other* handler on this same element, so a
    // caller's own `on:keydown` (the address bar's ↓ and Enter) still runs — it is forwarded below.
    e.stopPropagation();
  }

  $: style = `${grow ? 'flex:1 1 auto;min-width:0;' : 'flex:0 0 auto;'}${width ? `width:${width};` : ''}`;
</script>

{#if rows > 0}
  <textarea
    class="field {variant}"
    {style}
    {rows}
    {readonly}
    {disabled}
    bind:this={el}
    bind:value
    on:keydown={onKeydown}
    on:keydown
    on:input
    on:focus
    on:blur
    on:change
    {...$$restProps}
  ></textarea>
{:else}
  <input
    class="field {variant}"
    {style}
    {readonly}
    {disabled}
    bind:this={el}
    bind:value
    on:keydown={onKeydown}
    on:keydown
    on:input
    on:focus
    on:blur
    on:change
    {...$$restProps}
  />
{/if}

<style>
  .field {
    font: inherit;
    font-size: 12.5px;
    color: var(--fg);
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 4px 8px;
  }
  textarea.field {
    resize: vertical;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  /* The address bar: part of the header rather than a control sitting on it, so no border and no
     fill until you point at it. */
  .field.bare {
    font-size: 12px;
    color: var(--fg-dim);
    background: none;
    border: none;
    border-radius: 4px;
    padding: 3px 6px;
    text-overflow: ellipsis;
  }
  .field.bare:hover:not(:read-only) {
    background: var(--bg-hover);
  }
  .field.bare:focus {
    color: var(--fg);
    background: var(--bg-elev);
  }
  /* The filter and scope builders: many small boxes in a grid, so tighter and monospaced — these hold
     globs and shapes, where character alignment is the point. */
  .field.dense {
    font-size: 12px;
    padding: 3px 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .field:read-only {
    color: var(--fg-dim);
  }
</style>
