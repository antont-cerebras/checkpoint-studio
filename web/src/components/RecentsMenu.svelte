<script lang="ts">
  /**
   * The recents dropdown: pick a checkpoint, or forget one.
   *
   * Extracted so every checkpoint box has the same list. The address bar had this one; the two boxes on
   * the comparison screen had a native `<datalist>`, which gives suggestions and *no per-entry actions*
   * — so a stale path could be picked there for ever and only removed from somewhere else. A list you
   * can pick from but not prune is a list that grows wrong.
   *
   * The rows are `RecentRow`, which owns the two-step forget, so "does the cross delete immediately?"
   * has one answer everywhere.
   *
   * The **keyboard stays with the owner**: each box drives its own `open`/`cursor` (the address bar's ↓
   * also means "open", the comparison boxes' Enter means "compare"). What this owns is the popup — the
   * markup, the outside-click dismissal, and the listbox semantics.
   */
  import { tick } from 'svelte';
  import RecentRow from './RecentRow.svelte';

  export let options: string[];
  export let open = false;
  /** Which row the keyboard is on; `-1` for none. */
  export let cursor = -1;
  export let busy = false;
  /** The spec being served, for the `open` badge. */
  export let current = '';
  export let onPick: (spec: string) => void;
  /** Told when a click outside should dismiss the menu. */
  export let onClose: () => void;
  /** A label, since a page can hold three of these. */
  export let label = 'Recently opened checkpoints';

  let box: HTMLElement | undefined;
  /** The element this menu hangs under — its parent, which is the picker. */
  let anchor: HTMLElement | undefined;
  /** Where the popup actually sits, in viewport coordinates. */
  let placement = '';

  /**
   * Place the popup under its box and **keep it on screen**.
   *
   * Reported: the right-hand comparison box's menu ran off the edge of the window, which does not merely
   * look wrong — the rows out there cannot be clicked, so the crosses that forget a stale entry were
   * unreachable. A left-anchored popup 680px wide under a box that starts two-thirds of the way across
   * is off the edge by construction.
   *
   * So it is positioned rather than anchored: measured against the viewport and clamped into it, which
   * also frees it from any `overflow: hidden` ancestor. Recomputed when it opens and whenever the page
   * moves under it.
   */
  function place() {
    const host = anchor?.parentElement;
    if (!open || !host) return;
    const r = host.getBoundingClientRect();
    const margin = 8;
    const width = Math.min(680, window.innerWidth - margin * 2);
    // Prefer the box's own left edge; slide left only as far as it must to fit.
    const left = Math.max(margin, Math.min(r.left, window.innerWidth - width - margin));
    // Below the box, and never taller than the room beneath it.
    const top = r.bottom + 4;
    const maxHeight = Math.max(120, window.innerHeight - top - margin);
    placement = `left:${Math.round(left)}px;top:${Math.round(top)}px;width:${Math.round(width)}px;max-height:${Math.round(maxHeight)}px;`;
  }

  // On open, and after the rows exist — the height depends on them.
  $: if (open) void tick().then(place);

  /** Click anywhere else closes it — the same dismiss rule the palette follows. */
  function onWindowPointerDown(e: Event) {
    if (!open) return;
    const host = anchor?.parentElement;
    const inside = (box && box.contains(e.target as Node)) || host?.contains(e.target as Node);
    if (!inside) onClose();
  }

  const trim = (s: string) => s.replace(/\/+$/, '');
</script>

<!-- The popup is positioned in viewport coordinates, so anything that moves the box under it — a
     resize, a scroll — has to move it too. -->
<svelte:window on:pointerdown={onWindowPointerDown} on:resize={place} on:scroll|capture={place} />

<!-- A zero-size marker, purely to find the box this menu belongs to: the popup itself is `fixed`, so it
     is no longer a child of anything that could tell it where it is. -->
<span class="anchor" bind:this={anchor}></span>

{#if open && options.length}
  <!-- Background fill, no border: the same treatment as the command palette and the filter builder,
       rather than a second idea of what a popup looks like. -->
  <ul class="menu" role="listbox" aria-label={label} style={placement} bind:this={box}>
    {#each options as spec, i (spec)}
      <li>
        <RecentRow
          {spec}
          active={i === cursor}
          current={trim(spec) === trim(current)}
          {busy}
          {onPick}
        />
      </li>
    {/each}
  </ul>
{/if}

<style>
  .anchor {
    display: none;
  }
  .menu {
    /* Fixed and measured (see `place`): a left-anchored popup under the right-hand box ran off the
       edge of the window, and rows off the edge cannot be clicked. */
    position: fixed;
    z-index: 30;
    overflow: auto;
    margin: 0;
    padding: 4px;
    list-style: none;
    background: var(--bg-elev);
    border-radius: 6px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
  }
  li {
    margin: 0;
  }
</style>
