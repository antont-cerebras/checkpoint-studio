<script lang="ts">
  import { get } from 'svelte/store';
  import {
    paletteOpen,
    screen,
    navigate,
    setAllExpanded,
    startSearch,
    filterByDtype,
    clearFilter,
  } from '../stores/view';
  import { dtypesPresent } from '../stores/server';
  import { theme } from '../stores/theme';
  import { warningDismissed } from '../stores/warning';
  import { fuzzyScore } from '../lib/search';

  interface Cmd {
    group: string;
    label: string;
    run: () => void;
  }

  const base: Cmd[] = [
    { group: 'Go', label: 'Tensor tree', run: () => navigate({ kind: 'tree' }) },
    { group: 'Go', label: 'File browser', run: () => navigate({ kind: 'files' }) },
    { group: 'Go', label: 'Byte layout map', run: () => navigate({ kind: 'layout' }) },
    { group: 'Go', label: 'Stats', run: () => navigate({ kind: 'stats' }) },
    { group: 'Go', label: 'Health check', run: () => navigate({ kind: 'health' }) },
    {
      group: 'Go',
      // One entry, because there is one comparison. There were two — *Diff report* and *Compare side
      // by side* — which made the reader choose a representation before seeing the result; the
      // summary, the tree and the data checks are views on the page now. "diff" stays in the label
      // because that is what this is called everywhere else (the subcommand, the report's headings),
      // and so it is what someone searches the palette for.
      label: 'Compare (diff) with another checkpoint…',
      run: () => {
        // The page carries its own path boxes, so open it with whatever was compared last (or empty)
        // and let the reader type there — one place to enter a path rather than a prompt here and an
        // input there.
        const s = get(screen);
        navigate({
          kind: 'compare',
          lhs: s.kind === 'compare' ? s.lhs : '',
          rhs: s.kind === 'compare' ? s.rhs : '',
        });
      },
    },
    {
      group: 'Go',
      // Same wording as the terminal's palette entry, so the two are recognisably one
      // feature (see `TREE_COMMANDS` in src/explorer/mod.rs).
      label: 'Open another checkpoint…',
      run: () => navigate({ kind: 'open' }),
    },
    { group: 'Tree', label: 'Expand all groups', run: () => setAllExpanded(true) },
    { group: 'Tree', label: 'Collapse all groups', run: () => setAllExpanded(false) },
    { group: 'Tree', label: 'Search tensors', run: () => { navigate({ kind: 'tree' }); startSearch(); } },
    {
      group: 'View',
      label: 'Show the access-control warning',
      run: () => warningDismissed.set(false),
    },
    { group: 'Theme', label: 'Theme: System', run: () => theme.set('system') },
    { group: 'Theme', label: 'Theme: Dark', run: () => theme.set('dark') },
    { group: 'Theme', label: 'Theme: Light', run: () => theme.set('light') },
    { group: 'Theme', label: 'Theme: Fallout', run: () => theme.set('fallout') },
  ];


  // Filter commands are data-driven: one per dtype present, plus a clear.
  $: dtypes = $dtypesPresent;
  $: commands = [
    ...base,
    ...dtypes.map((d) => ({ group: 'Filter', label: `Filter dtype: ${d}`, run: () => filterByDtype(d) })),
    { group: 'Filter', label: 'Clear filter', run: clearFilter },
  ];

  let q = '';
  let sel = 0;

  $: filtered = q.trim()
    ? commands
        .map((c) => ({ c, s: fuzzyScore(q.trim(), `${c.group} ${c.label}`) }))
        .filter((x) => x.s >= 0)
        .sort((a, b) => b.s - a.s)
        .map((x) => x.c)
    : commands;
  $: if (sel >= filtered.length) sel = Math.max(0, filtered.length - 1);

  function run(c: Cmd) {
    paletteOpen.set(false);
    c.run();
  }
  function onKey(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault();
      paletteOpen.set(false);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      sel = Math.min(filtered.length - 1, sel + 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      sel = Math.max(0, sel - 1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const cmd = filtered[sel];
      if (cmd) run(cmd);
    }
  }
</script>

<div
  class="backdrop"
  role="presentation"
  on:click={(e) => {
    if (e.target === e.currentTarget) paletteOpen.set(false);
  }}
>
  <div class="palette" role="dialog" aria-label="Command palette">
    <!-- svelte-ignore a11y-autofocus -->
    <input autofocus placeholder="Run a command…" bind:value={q} on:keydown={onKey} />
    <ul>
      {#each filtered as c, i (`${c.group}/${c.label}`)}
        <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-noninteractive-element-interactions -->
        <li
          class:sel={i === sel}
          role="option"
          aria-selected={i === sel}
          on:click={() => run(c)}
          on:mousemove={() => (sel = i)}
        >
          <span class="cgroup">{c.group}</span><span class="clabel">{c.label}</span>
        </li>
      {/each}
      {#if !filtered.length}<li class="empty dim">no matching commands</li>{/if}
    </ul>
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.45);
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 12vh;
    z-index: 40;
  }
  .palette {
    width: 460px;
    max-width: 92vw;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    overflow: hidden;
  }
  input {
    width: 100%;
    border: none;
    border-bottom: 1px solid var(--border);
    border-radius: 0;
    padding: 10px 14px;
    font-size: 14px;
    background: var(--bg);
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 4px;
    max-height: 50vh;
    overflow: auto;
  }
  li {
    display: flex;
    align-items: baseline;
    gap: 10px;
    padding: 6px 10px;
    border-radius: 5px;
    cursor: pointer;
  }
  li.sel {
    background: var(--bg-sel);
  }
  .cgroup {
    flex: 0 0 54px;
    color: var(--fg-dim);
    font-size: 11px;
    text-transform: uppercase;
  }
  .empty {
    cursor: default;
  }
</style>
