<script lang="ts">
  import { isEditable } from './lib/keys';
  import { staleBuild, watchBuild } from './stores/version';
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import {
    tree,
    treeError,
    ensureTree,
    treeProgress,
    openProgress,
    openingSpec,
    compactTree,
    compactError,
    compactProgress,
  } from './stores/server';
  import { currentStep } from './lib/loadstep';
  import { switchCheckpoint } from './stores/open';
  import { warningDismissed } from './stores/warning';
  import {
    screen,
    searching,
    search,
    selectedId,
    visibleRows,
    back,
    forward,
    navigate,
    moveSelection,
    selectParent,
    enterChild,
    selectSibling,
    activateSelection,
    toggleAllExpanded,
    startSearch,
    exitSearch,
    setTab,
    filterQuery,
    filterError,
    filterMatches,
    filterResolvedFor,
    searchTotal,
    clearFilter,
    sortKey,
    sortDir,
    setSort,
    compact,
    paletteOpen,
    hashCheckpoint,
    currentGlobals,
    applyGlobals,
  } from './stores/view';
  import type { SortKey } from './stores/view';
  import TreeView from './components/TreeView.svelte';
  import Detail from './components/Detail.svelte';
  import FileBrowser from './components/FileBrowser.svelte';
  import LayoutView from './components/LayoutView.svelte';
  import StatsView from './components/StatsView.svelte';
  import HealthView from './components/HealthView.svelte';
  import FilePreview from './components/FilePreview.svelte';
  import StatusBar from './components/StatusBar.svelte';
  import Footer from './components/Footer.svelte';
  import LoadingScreen from './components/LoadingScreen.svelte';
  import CheckpointBar from './components/CheckpointBar.svelte';
  import OpenView from './components/OpenView.svelte';
  import ComparePage from './components/ComparePage.svelte';
  import Palette from './components/Palette.svelte';
  import FilterBuilder from './components/FilterBuilder.svelte';
  import CompactView from './components/CompactView.svelte';
  import { theme } from './stores/theme';
  import { copyText } from './lib/clipboard';
  import type { Screen } from './stores/view';
  import type { TreeNode } from './lib/types';

  let builderOpen = false;

  // Ask whether this page is still the page the server serves — now, and whenever the tab comes back
  // to the front. `onMount`'s return is the unsubscribe.
  onMount(watchBuild);

  onMount(async () => {
    // A link can name a checkpoint (`#tree?ckpt=…`). If it names one the server is not serving,
    // open it — otherwise the link would restore the screen, the filter and the selection of a
    // view *of a different checkpoint*, which is the one thing a shared view must not do.
    //
    // Read FIRST, before the tree is fetched: the first tree to land triggers a hash sync that
    // replaces `ckpt` with the served root, so reading afterwards reads our own overwrite and
    // never sees what the link asked for.
    const wanted = hashCheckpoint();
    // The view state the link carries is captured too, because switching deliberately drops
    // position state (it names rows the new checkpoint may not have) and has to be put back.
    const restore = { screen: get(screen), globals: currentGlobals() };
    await ensureTree();
    if (!wanted || wanted === get(tree)?.root) return;
    try {
      await switchCheckpoint(wanted);
      applyGlobals(restore.globals);
      navigate(restore.screen, true);
    } catch {
      // A link to a checkpoint that no longer resolves: stay on the one that is loaded rather
      // than emptying the screen. The address bar shows what is actually served.
    }
  });

  function onSortChange(e: Event) {
    setSort((e.currentTarget as HTMLSelectElement).value as SortKey);
  }

  // "Still filtering" = a non-empty query whose result hasn't landed yet (its trimmed
  // text differs from the query `filterMatches`/`filterError` reflect). Derived so it
  // repaints reliably from the async-set stores — see `filterResolvedFor`.
  $: filtering = $filterQuery.trim().length > 0 && $filterQuery.trim() !== $filterResolvedFor;

  // One wait, one screen. Three components used to each render their own version of "waiting
  // for this checkpoint"; now they contribute inputs and this decides which step is showing.
  $: step = currentStep({
    opening: $openProgress,
    openingSpec: $openingSpec,
    tree: $treeProgress,
    haveTree: !!$tree,
    treeError: !!$treeError,
    compact: $compact,
    haveCompact: !!$compactTree,
    folding: $compactProgress,
    compactError: !!$compactError,
  });

  function crumb(s: Screen): string {
    switch (s.kind) {
      case 'tree':
        return '';
      case 'detail':
        return `› ${s.tensor}`;
      case 'files':
        return '› Files';
      case 'layout':
        return `› Layout${s.file ? `: ${s.file}` : ''}`;
      case 'stats':
        return '› Stats';
      case 'health':
        return '› Health';
      case 'preview':
        return `› ${s.name}`;
      case 'open':
        return '› Open';
      case 'compare':
        // Just the word. The two addresses are in the two boxes directly below, where they can be
        // read *and* edited; spelling them out here as well made a header three lines deep on the
        // remote pairs this screen exists for.
        return '› Compare';
    }
  }

  const PAGE = 20;

  function selectedRow() {
    const id = get(selectedId);
    return get(visibleRows).find((r) => r.id === id) ?? null;
  }

  function copy(text: string) {
    copyText(text);
  }

  // Cmd/Ctrl-A on the tensor list copies the (possibly filtered) tensor names — the
  // list is virtualized, so a text selection would only cover the rendered slice;
  // copying the full visible set is the useful equivalent. A brief toast confirms.
  let listFlash = '';
  let listFlashTimer: ReturnType<typeof setTimeout>;
  function copyTensorList() {
    const names: string[] = [];
    if (get(searching) || get(filterMatches) !== null) {
      // Flat view: exactly the matched/searched tensors (the "filtered list").
      for (const r of get(visibleRows)) if (r.node.kind === 'tensor') names.push(r.node.info.name);
    } else {
      // Plain tree: every tensor, regardless of which groups are expanded.
      const t = get(tree);
      const walk = (nodes: TreeNode[]) => {
        for (const n of nodes) {
          if (n.kind === 'tensor') names.push(n.info.name);
          else if (n.kind === 'group') walk(n.children);
        }
      };
      if (t) walk(t.tree);
    }
    if (!names.length) return;
    copyText(names.join('\n'));
    listFlash = `Copied ${names.length} tensor name${names.length === 1 ? '' : 's'}`;
    clearTimeout(listFlashTimer);
    listFlashTimer = setTimeout(() => (listFlash = ''), 1600);
  }

  // Focus the search input every time it mounts — not just the first time (the box
  // is unmounted on non-tree screens and remounted on return, where `autofocus`
  // wouldn't re-fire), so returning from a result lands the cursor back in the query.
  function focusOnMount(node: HTMLInputElement) {
    node.focus();
    node.select();
  }

  // While typing a filter query, keep keystrokes out of the global tree shortcuts
  // (so `h`/`s`/`/`/`e` type into the box instead of navigating). Escape blurs.
  function filterKeydown(e: KeyboardEvent) {
    e.stopPropagation();
    if (e.key === 'Escape') (e.currentTarget as HTMLInputElement).blur();
  }

  function onKeydown(e: KeyboardEvent) {
    // Cmd/Ctrl-A on the tensor list → copy the (filtered) tensor names, not the
    // whole page's text. Elsewhere (or while typing) leave the browser default.
    if ((e.metaKey || e.ctrlKey) && !e.altKey && (e.key === 'a' || e.key === 'A')) {
      if (get(screen).kind === 'tree' && !isEditable(e.target)) {
        e.preventDefault();
        copyTensorList();
      }
      return;
    }
    // Let real browser/system chords through.
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // The palette owns the keyboard while open (its input handles keys).
    if (get(paletteOpen)) return;
    const s = get(screen);

    // --- search mode: the input is focused; only steal a few keys ---
    // Only while the tree is showing: once a result opens a detail (or any other
    // screen), that screen's shortcuts (i/m/v/h …) must win, not type into the
    // still-live query. The query is preserved, so Backspace lands back on the
    // filtered tree.
    if (get(searching) && s.kind === 'tree') {
      if (e.key === 'Escape') {
        e.preventDefault();
        exitSearch();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        moveSelection(1);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        moveSelection(-1);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        activateSelection();
      } else if (e.key === '/') {
        // Already searching — swallow `/` instead of inserting a literal slash
        // into the query (matches the TUI, which ignores it).
        e.preventDefault();
      }
      return; // everything else types into the search box
    }

    // **A focused text field owns every key.** Nothing below may take one.
    //
    // This used to sit *below* the palette shortcut, which meant `:` opened the command palette
    // instead of typing a colon — so `:/opt/…`, the proxy shorthand this app advertises in its own
    // placeholder, could not be typed into any box. (Space had the same problem on the tree screen.)
    // Backspace was an earlier instance of the same mistake: it navigated back instead of deleting.
    //
    // The rule is the order, and the order is the point: search first (that input deliberately gives
    // ↑/↓/Enter/Esc back to the tree, and returns above), then this, then everything else.
    if (isEditable(e.target)) {
      // Escape still gets you out of the field (and, with nothing focused, the next
      // Escape leaves the screen) — everything else is text.
      if (e.key === 'Escape') (e.target as HTMLElement).blur();
      return;
    }

    // --- command palette: `:` anywhere, or Space on the tree ---
    if (e.key === ':' || (e.key === ' ' && s.kind === 'tree' && !get(searching))) {
      e.preventDefault();
      paletteOpen.set(true);
      return;
    }

    // --- global (any screen) ---
    if (e.key === 'Backspace') {
      e.preventDefault(); // don't let the browser navigate back
      back();
      return;
    }
    if (e.key === '\\') {
      e.preventDefault();
      forward();
      return;
    }

    if (s.kind === 'tree') {
      treeKey(e);
    } else if (s.kind === 'detail') {
      detailKey(e);
    } else {
      // files / layout / stats / health
      if (e.key === 'Escape' || (e.key === 'Tab' && s.kind === 'files')) {
        e.preventDefault();
        back();
      }
    }
  }

  function treeKey(e: KeyboardEvent) {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (e.shiftKey) selectSibling(true);
        else moveSelection(1);
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (e.shiftKey) selectSibling(false);
        else moveSelection(-1);
        break;
      case 'ArrowLeft':
        e.preventDefault();
        selectParent();
        break;
      case 'ArrowRight':
        e.preventDefault();
        enterChild();
        break;
      case 'PageDown':
        e.preventDefault();
        moveSelection(PAGE);
        break;
      case 'PageUp':
        e.preventDefault();
        moveSelection(-PAGE);
        break;
      case 'Enter':
        e.preventDefault();
        activateSelection();
        break;
      case 'Tab':
        e.preventDefault();
        navigate({ kind: 'files' });
        break;
      // Accept lower- and upper-case: the footer shows plain `e`/`l`, so a Shift
      // requirement would just read as "the feature is broken".
      case 'e':
      case 'E':
        toggleAllExpanded();
        break;
      case 'k':
      case 'K':
        compact.update((v) => !v);
        break;
      case '/':
        e.preventDefault();
        startSearch();
        break;
      case 's':
        navigate({ kind: 'stats' });
        break;
      case 'h':
        navigate({ kind: 'health' });
        break;
      case 'L': // Shift+L: `l` alone is "legend" in the TUI, so don't shadow it
      case 'y': // no CLI-command copy in the browser; reuse for layout
        navigate({ kind: 'layout' });
        break;
      case 'f': {
        const r = selectedRow();
        if (r?.node.kind === 'tensor') copy(r.node.info.source_path);
        break;
      }
      case 'n': {
        const r = selectedRow();
        if (r?.node.kind === 'tensor') copy(r.node.info.name);
        break;
      }
    }
  }

  function detailKey(e: KeyboardEvent) {
    switch (e.key) {
      case 'Escape':
        e.preventDefault();
        back();
        break;
      case 's':
      case 'i':
        setTab('info'); // statistics live on the info tab
        break;
      case 'h':
        setTab('histogram');
        break;
      case 'm':
        setTab('heatmap');
        break;
      case 'v':
        setTab('values');
        break;
    }
  }
</script>

<svelte:window on:keydown={onKeydown} />

<div class="app">
  <!-- The page had no <h1>. The visible title is a nav button, so give assistive tech
       a real document heading naming the app + the open checkpoint. -->
  <h1 class="visually-hidden">Checkpoint Studio — {$tree?.root ?? 'loading checkpoint'}</h1>
  <!-- No authentication exists in front of this server. When it is bound anywhere but
       loopback, say so on the page as well as in the terminal it was started from — the
       person looking at the browser is not necessarily the person who read the banner.
       One narrow strip, not dismissible: the condition lasts as long as the server does. -->
  <!-- The server has been rebuilt under this tab, so its JavaScript is not the JavaScript being
       served. Not dismissible and above everything else, because the failure it prevents is silent:
       an old client reading a newer response shape once declared two unrelated checkpoints
       "structurally identical" rather than erroring (see `lib/build`). -->
  {#if $staleBuild}
    <div class="stale-build" role="alert">
      <span
        >This tab is running an older build of the interface than the server is serving. Reload to
        pick up the current one.</span
      >
      <button class="reload" type="button" on:click={() => location.reload()}>Reload</button>
    </div>
  {/if}
  {#if $tree?.access_warning && !$warningDismissed}
    <div class="access-warning" role="alert">
      <span>{$tree.access_warning}</span>
      <button
        class="dismiss"
        title="Hide this (reachable again from the command palette)"
        aria-label="Hide the access-control warning"
        on:click={() => warningDismissed.set(true)}>×</button
      >
    </div>
  {/if}
  <header>
    <button class="nav" on:click={back} title="Back (Backspace)" aria-label="Back">‹</button>
    <button class="nav" on:click={forward} title="Forward (\\)" aria-label="Forward">›</button>
    <button class="home" on:click={() => navigate({ kind: 'tree' })} title="Tensor tree">
      Checkpoint&nbsp;Studio
    </button>
    {#if $screen.kind !== 'tree'}
      <!-- Truncates at narrow widths, so carry the full text in a tooltip. -->
      <span class="crumb dim" title={crumb($screen)}>{crumb($screen)}</span>
    {/if}
    <!-- The checkpoint address: a real input, so copy/paste/select work, with the recents
         list on `↓`. Editing it opens another checkpoint (see CheckpointBar).
         Not on the comparison screen, which is *about* two addresses and carries a box for each: a
         third box up here named one of them again, could only hold one of the two, and changed
         which checkpoint the server serves rather than which one is being compared. Three controls
         for two operands. -->
    {#if $screen.kind !== 'compare'}
      <CheckpointBar />
    {/if}
    <!-- Compare: the one action that needs a *second* checkpoint, so it sits beside the address
         bar rather than being buried in the palette.
         Not on the comparison screen itself, which has its own Compare button over its own two
         boxes: two buttons of the same name, one of which only re-opens the screen you are on. -->
    {#if $screen.kind !== 'compare'}
      <button
        class="cmp"
        type="button"
        title="Compare with another checkpoint, side by side"
        on:click={() => navigate({ kind: 'compare', lhs: '', rhs: '' })}>Compare</button
      >
    {/if}
    {#if $searching && $screen.kind === 'tree'}
      <span class="search">
        /
        <input
          use:focusOnMount
          spellcheck="false"
          placeholder="fuzzy filter tensors…"
          bind:value={$search}
        />
        <!-- The row count is shared with the filter: while a filter query is still
             resolving, `visibleRows` is the UNFILTERED tree (3 rows when collapsed),
             so printing it here reads as "3 matches" for a query that matches
             thousands. Gate it on the same `filtering` flag as the filter bar. -->
        <span class="dim">
          {#if filtering}filtering…
          {:else if $searchTotal > $visibleRows.length}showing {$visibleRows.length} of {$searchTotal.toLocaleString()}
          {:else}{$visibleRows.length.toLocaleString()} match{$visibleRows.length === 1 ? '' : 'es'}{/if}
          · Esc to exit
        </span>
      </span>
    {/if}
    <select class="theme" bind:value={$theme} title="Color theme" aria-label="Color theme">
      <option value="system">System</option>
      <option value="dark">Dark</option>
      <option value="light">Light</option>
      <option value="fallout">Fallout</option>
    </select>
  </header>

  {#if $screen.kind === 'tree'}
  <!-- The filter bar acts on the tensor tree, so it only shows on the tree screen
       (nothing to filter on detail / stats / health / layout / files). -->
  <div class="filterbar" class:err={$filterError && !filtering}>
    <button
      class="bld"
      class:on={builderOpen}
      title="Filter builder — pick facets with the mouse"
      aria-label="Toggle filter builder"
      on:click={() => (builderOpen = !builderOpen)}>▤</button>
    <button
      class="bld"
      class:on={$compact}
      title="Compact view — collapse per-layer / per-expert families"
      aria-label="Toggle compact family view"
      on:click={() => compact.update((v) => !v)}>≡</button>
    <span
      class="flabel"
      title="dtype:F16,BF16  shape:(6,_,42)  dim:4096  rank:>=3  size:1MiB..1GiB  params:>1M  name:re:^model\.  shard:00001  ·  space = AND, ! = not, comma = OR"
    >⌕ filter</span>
    <input
      class="fq"
      spellcheck="false"
      autocomplete="off"
      placeholder="dtype:F16  shape:(_,4096)  size:>1MiB  name:re:…   (space = AND, ! = not)"
      bind:value={$filterQuery}
      on:keydown={filterKeydown}
    />
    {#if filtering}
      <span class="dim">filtering…</span>
    {:else if $filterError}
      <span class="ferr" title={$filterError}>⚠ {$filterError}</span>
    {:else if $filterMatches}
      <span class="dim">{$visibleRows.length.toLocaleString()} match{$visibleRows.length === 1 ? '' : 'es'}</span>
    {/if}
    {#if $filterMatches !== null || $searching}
      <span class="sort" title="Sort the tensor list">
        sort
        <select value={$sortKey} on:change={onSortChange} aria-label="Sort by">
          <option value="none">—</option>
          <option value="name">name</option>
          <option value="size">size</option>
          <option value="params">params</option>
          <option value="rank">rank</option>
          <option value="dtype">dtype</option>
        </select>
        <button
          class="dir"
          disabled={$sortKey === 'none'}
          title={$sortDir === 'asc' ? 'Ascending (click for descending)' : 'Descending (click for ascending)'}
          on:click={() => sortDir.update((d) => (d === 'asc' ? 'desc' : 'asc'))}
        >{$sortDir === 'asc' ? '↑' : '↓'}</button>
      </span>
    {/if}
    {#if $filterQuery}
      <button class="clear" on:click={clearFilter}>clear</button>
    {/if}
  </div>
  {#if builderOpen}<FilterBuilder />{/if}
  {/if}

  <main>
    <!-- The open screen comes FIRST, before the no-tree and error guards: it is the one screen
         that works without a loaded checkpoint, and it is how you recover from a checkpoint
         that failed to load. Behind those guards it would have been unreachable in exactly the
         situation you most need it. -->
    {#if step}
      <!-- Before the screen dispatch, and before the `open` screen: while a checkpoint is being
           read there is nothing truthful any screen can show, and the pane you were on would be
           showing the checkpoint that is being replaced. -->
      <LoadingScreen {step} />
    {:else if $screen.kind === 'open'}
      <OpenView />
    {:else if $treeError}
      <div class="error">
        Failed to load checkpoint: {$treeError}
        <button type="button" class="recover" on:click={() => navigate({ kind: 'open' })}>
          Open another checkpoint…
        </button>
      </div>
    {:else if !$tree}
      <!-- No tree, no error, no step in flight: not a state that should occur, and the honest
           thing to show for it is the wait we are still in rather than a blank pane. -->
      <LoadingScreen step={{ kind: 'tree', progress: $treeProgress }} />
    {:else if $screen.kind === 'tree'}
      {#if $compact}<CompactView />{:else}<TreeView />{/if}
    {:else if $screen.kind === 'detail'}
      <Detail tensor={$screen.tensor} tab={$screen.tab} />
    {:else if $screen.kind === 'files'}
      <FileBrowser />
    {:else if $screen.kind === 'layout'}
      <LayoutView />
    {:else if $screen.kind === 'stats'}
      <StatsView />
    {:else if $screen.kind === 'health'}
      <HealthView />
    {:else if $screen.kind === 'preview'}
      <FilePreview path={$screen.path} name={$screen.name} />
    {:else if $screen.kind === 'compare'}
      <ComparePage
        lhs={$screen.lhs}
        rhs={$screen.rhs}
        view={$screen.view ?? 'summary'}
        scope={$screen.scope}
        full={$screen.full ?? false}
        swapped={$screen.swapped ?? false}
        closed={$screen.closed ?? []}
      />
    {/if}
  </main>

  <StatusBar />
  <Footer />
  {#if $paletteOpen}<Palette />{/if}
  {#if listFlash}<div class="listflash">✓ {listFlash}</div>{/if}
</div>

<style>
  /* Light red on a tinted strip: a standing caution, not an error — it must not read as
     something that broke. Narrow enough not to cost content height. */
  /* The same narrow strip as the access warning, in the warning colour: one condition, one line,
     no dismissal — it lasts until the page is reloaded, which is the button beside it. */
  .stale-build {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 10px;
    font-size: 12px;
    color: var(--warn);
    background: color-mix(in srgb, var(--warn) 16%, var(--bg-panel));
    border-bottom: 1px solid var(--border);
  }
  .stale-build .reload {
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 9px;
    cursor: pointer;
  }
  .access-warning {
    display: flex;
    align-items: baseline;
    gap: 8px;
    flex: none;
    padding: 3px 12px;
    font-size: 11.5px;
    line-height: 1.4;
    color: #ffb4b4;
    background: #4a1f1f;
    border-bottom: 1px solid #6b2b2b;
  }
  .access-warning span {
    flex: 1;
  }
  .dismiss {
    flex: none;
    padding: 0 4px;
    font: inherit;
    line-height: 1;
    color: inherit;
    background: none;
    border: 0;
    cursor: pointer;
    opacity: 0.8;
  }
  .dismiss:hover {
    opacity: 1;
  }
  .app {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
    flex: 0 0 auto;
  }
  .nav {
    flex: 0 0 auto;
    padding: 0 7px;
    font-size: 16px;
    line-height: 1;
    color: var(--fg-dim);
  }
  .home {
    font-weight: 600;
    color: var(--accent);
    flex: 0 0 auto;
    background: none;
    border: none;
    padding: 2px 4px;
    cursor: pointer;
  }
  .crumb {
    flex: 0 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .search {
    flex: 1 1 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--accent);
  }
  .search input {
    flex: 0 1 360px;
  }
  .filterbar {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 14px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }
  .filterbar.err {
    background: color-mix(in srgb, var(--danger) 10%, var(--bg-panel));
  }
  .bld {
    flex: 0 0 auto;
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg-dim);
    cursor: pointer;
    font: inherit;
    padding: 0 6px;
    line-height: 18px;
  }
  .bld.on,
  .bld:hover {
    color: var(--accent);
    border-color: var(--accent);
  }
  .flabel {
    flex: 0 0 auto;
    color: var(--fg-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 11px;
    cursor: help;
  }
  .fq {
    flex: 1 1 auto;
    min-width: 0;
    font-family: ui-monospace, monospace;
    font-size: 12px;
  }
  .ferr {
    flex: 0 1 auto;
    color: var(--danger);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .clear {
    flex: 0 0 auto;
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg-dim);
    cursor: pointer;
    font: inherit;
    padding: 1px 8px;
  }
  .clear:hover {
    color: var(--fg);
    border-color: var(--accent);
  }
  .sort {
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--fg-dim);
  }
  .sort select {
    font-size: 12px;
    padding: 1px 2px;
  }
  .dir {
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--fg-dim);
    cursor: pointer;
    font: inherit;
    padding: 0 5px;
    line-height: 16px;
  }
  .dir:hover:not(:disabled) {
    color: var(--fg);
    border-color: var(--accent);
  }
  .dir:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .theme {
    margin-left: auto;
    flex: 0 0 auto;
    font-size: 12px;
    padding: 2px 4px;
  }
  main {
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
  }
  .error {
    padding: 16px;
    color: var(--danger);
  }
  /* The way out of a checkpoint that didn't load: without it the error screen is a dead end
     that only a restart clears. */
  .cmp {
    flex: 0 0 auto;
    font: inherit;
    font-size: 12px;
    color: var(--fg-dim);
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 9px;
    cursor: pointer;
  }
  .cmp:hover {
    color: var(--fg);
    background: var(--bg-hover);
  }
  .recover {
    display: block;
    margin-top: 10px;
    font: inherit;
    color: var(--accent);
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 10px;
    cursor: pointer;
  }
  .listflash {
    position: fixed;
    bottom: 16px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 40;
    background: var(--bg-elev);
    color: var(--fg);
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 13px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  }
</style>
