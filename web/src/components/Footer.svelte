<script lang="ts">
  import {
    screen,
    back,
    navigate,
    activateSelection,
    toggleAllExpanded,
    startSearch,
    setTab,
    paletteOpen,
    compact,
    type Screen,
  } from '../stores/view';

  interface Hint {
    keys: string;
    label: string;
    act: () => void;
  }

  // The tree footer mirrors the TUI's `tree_hint_lines` (src/ui.rs), minus the
  // TUI-only actions (quit / repack / rename / OSC-52 copy / command palette).
  const treeHints: Hint[] = [
    { keys: '↑↓', label: 'navigate', act: () => {} },
    { keys: '←→', label: 'parent/child', act: () => {} },
    { keys: '⇧↑↓', label: 'sibling', act: () => {} },
    { keys: 'Enter', label: 'open', act: activateSelection },
    { keys: 'Tab', label: 'files', act: () => navigate({ kind: 'files' }) },
    { keys: 'Space/:', label: 'commands', act: () => paletteOpen.set(true) },
    { keys: 'e', label: 'expand/collapse all', act: toggleAllExpanded },
    { keys: '/', label: 'search', act: startSearch },
    { keys: 'h', label: 'health', act: () => navigate({ kind: 'health' }) },
    { keys: 's', label: 'stats', act: () => navigate({ kind: 'stats' }) },
    { keys: 'k', label: 'compact', act: () => compact.update((v) => !v) },
    { keys: '⇧L', label: 'layout', act: () => navigate({ kind: 'layout' }) },
  ];

  const detailHints: Hint[] = [
    { keys: 'i', label: 'info', act: () => setTab('info') },
    { keys: 'm', label: 'heatmap', act: () => setTab('heatmap') },
    { keys: 'v', label: 'values', act: () => setTab('values') },
    { keys: 'h', label: 'histogram', act: () => setTab('histogram') },
    { keys: 'Esc/⌫', label: 'back', act: back },
  ];

  // Mirrors the TUI's `hints::compare_hint_lines`, so the same screen advertises the same keys in
  // both surfaces. The footer used to fall through to a bare "back" here, which meant `n`/`N` and `s`
  // were mentioned only in one dim line inside the view — and `s` means *stats* on every other
  // screen, so a collision that is documented nowhere global is a collision nobody expects.
  //
  // The acts are no-ops because the handlers belong to the view that owns the state; same as `↑↓` in
  // the tree footer, which is a legend entry rather than a button.
  const compareHints: Hint[] = [
    { keys: 'n/N', label: 'next/prev difference', act: () => {} },
    { keys: 's', label: 'swap sides', act: () => {} },
    { keys: 'k', label: 'families', act: () => {} },
    { keys: 'Esc/⌫', label: 'back', act: back },
  ];

  const otherHints: Hint[] = [{ keys: 'Esc/⌫', label: 'back', act: back }];

  function hintsFor(s: Screen): Hint[] {
    if (s.kind === 'tree') return treeHints;
    if (s.kind === 'detail') return detailHints;
    if (s.kind === 'compare') return compareHints;
    return otherHints;
  }

  $: hints = hintsFor($screen);
</script>

<div class="footer">
  {#each hints as h, hi (hi)}
    <button class="hint" on:click={h.act} title={h.label}>
      <span class="k">{h.keys}</span><span class="l">{h.label}</span>
    </button>
  {/each}
</div>

<style>
  .footer {
    flex: 0 0 auto;
    display: flex;
    flex-wrap: wrap;
    gap: 4px 10px;
    padding: 5px 12px;
    border-top: 1px solid var(--border);
    background: var(--bg);
    font-size: 12px;
  }
  .hint {
    display: inline-flex;
    align-items: baseline;
    gap: 5px;
    background: none;
    border: none;
    padding: 1px 2px;
    cursor: pointer;
    color: var(--fg-dim);
  }
  .hint:hover {
    background: var(--bg-hover);
    border-radius: 3px;
  }
  .k {
    color: var(--accent);
    font-weight: 600;
  }
  .l {
    color: var(--fg-dim);
  }
</style>
